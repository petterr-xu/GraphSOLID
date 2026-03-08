import os
import json
from tqdm import tqdm
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .attacker import Attacker
from .defender import Defender
from src.dataset.abstract_dataset import AbstractDataModule
from src.models.classifier_engine import HeteroClassifierEngine


class DefaultPipeline:
    """
    Evaluate a classifier under three settings on a dataset of hetero subgraphs:
      1) clean graphs
      2) attacked graphs  (attacker.attack)
      3) defended graphs  (defender.defend applied on attacked graphs)

    IMPORTANT for this repo:
    - YelpChihDataModule provides hetero subgraph datasets via `dataset_module.hetero_datasets`,
      where each item is a HeteroData with {train,val,test}_mask on the target node type.
    - HeteroNN.HeteroGNN_classifier.forward(x_dict, edge_index_dict) returns logits for ALL nodes
      of `classifier.target_node` (same ordering as data[target].y/mask).

    This pipeline will:
    - iterate over dataset_module.hetero_datasets[split]
    - compute micro-accuracy across all evaluation nodes (mask==True) over all subgraphs
    - save detailed per-subgraph predictions for later analysis
    """

    def __init__(
        self,
        dataset_module: AbstractDataModule,
        defender: Defender,
        attacker: Attacker,
        classifier: nn.Module,
        classifier_optimizer: torch.optim.Optimizer,
        classifier_criterion,
        cl_scheduler,
        target,
        device = 'cuda:0',
        cl_lr: float = 0.001,
    ):
        self.dataset_module = dataset_module
        self.defender = defender
        self.attacker = attacker
        self.classifier = classifier
        self.device = device
        self.target = target
        self.classifier_optimizer = classifier_optimizer
        self.classifier_criterion = classifier_criterion
        self.cl_scheduler = cl_scheduler
        self.classifier_engine = HeteroClassifierEngine(
            model=self.classifier,
            optimizer=self.classifier_optimizer,
            criterion=self.classifier_criterion,
            scheduler=self.cl_scheduler,
            target_node=self.target,
            device=self.device,
        )
        self._to_device(device)

    def _to_device(self, device):
        if device is None:
            raise ValueError("Device must be specified.")
        self.classifier_engine.to(device)
        self.defender.to(device)


    def attack_onely(
        self,
        split: str = "test",
        need_training: bool = True,
        log_every: int = 1,
        positive_label: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate classifier on clean and attacked versions of the dataset.

        Flow:
          clean:   eval(dataset[i])
          attacked: eval(attacker.attack(dataset[i]))

        Outputs:
          - {split}_attack_only_results.pt  (full tensors)
          - {split}_attack_only_metrics.json (metrics only)
        """
        if need_training:
            print(f"[attack_only] Training classifier on {split} split before evaluation...")
            self.train_classifier_vanilla()

        ds = self._get_split_dataset(split)

        # infer target type from first sample
        sample0 = ds[0] if len(ds) > 0 else None
        if sample0 is None:
            raise ValueError(f"Empty dataset for split='{split}'.")
        target_type = self._infer_target_type(sample0)

        total = len(ds)
        clean_total_eval = 0
        clean_total_correct = 0
        attacked_total_eval = 0
        attacked_total_correct = 0

        clean_confusion = None
        attacked_confusion = None

        clean_per_sample: List[Dict[str, Any]] = []
        attacked_per_sample: List[Dict[str, Any]] = []
        attacked_target_totals = self._init_target_totals()

        for i in tqdm(range(total), desc=f"Evaluating {split} (clean/attacked)"):
            g = ds[i]

            # clean
            clean_res = self._eval_classifier_on_graph(g, split=split, target_type=target_type)
            clean_total_eval += clean_res["num_eval"]
            clean_total_correct += clean_res["num_correct"]
            clean_confusion = (
                clean_res["confusion"].clone()
                if clean_confusion is None
                else clean_confusion + clean_res["confusion"]
            )
            clean_per_sample.append(
                {
                    "sample_id": i,
                    "num_eval": clean_res["num_eval"],
                    "num_correct": clean_res["num_correct"],
                    "acc": clean_res["acc"],
                    "metrics": clean_res["metrics"],
                    "y_true": clean_res["y_true"],
                    "y_pred": clean_res["y_pred"],
                    "logits": clean_res["logits"],
                    "node_idx": clean_res["node_idx"],
                }
            )

            # attacked
            g_attacked = self.attacker.attack(g.clone() if hasattr(g, "clone") else g)
            attacked_res = self._eval_classifier_on_graph(g_attacked, split=split, target_type=target_type)
            attacked_total_eval += attacked_res["num_eval"]
            attacked_total_correct += attacked_res["num_correct"]
            attacked_confusion = (
                attacked_res["confusion"].clone()
                if attacked_confusion is None
                else attacked_confusion + attacked_res["confusion"]
            )
            attack_nodes, attack_target_type = self._extract_attack_target_info(
                g_attacked, default_target_type=target_type
            )
            attacked_target_metrics = None
            if attack_nodes is not None and attack_target_type == target_type:
                attacked_target_metrics = self._compute_transition_metrics_on_targets(
                    baseline_res=clean_res,
                    variant_res=attacked_res,
                    attack_nodes=attack_nodes,
                    positive_label=positive_label,
                )
                self._accumulate_target_totals(attacked_target_totals, attacked_target_metrics)
            attacked_per_sample.append(
                {
                    "sample_id": i,
                    "num_eval": attacked_res["num_eval"],
                    "num_correct": attacked_res["num_correct"],
                    "acc": attacked_res["acc"],
                    "metrics": attacked_res["metrics"],
                    "y_true": attacked_res["y_true"],
                    "y_pred": attacked_res["y_pred"],
                    "logits": attacked_res["logits"],
                    "node_idx": attacked_res["node_idx"],
                    "target_metrics": attacked_target_metrics,
                }
            )

            if log_every and (i + 1) % log_every == 0:
                print(
                    f"[{split} #{i + 1}/{total}] "
                    f"clean acc={clean_res['acc']:.4f} "
                    f"macro_f1={clean_res['metrics']['macro_f1']:.4f} "
                    f"macro_recall={clean_res['metrics']['macro_recall']:.4f} | "
                    f"attacked acc={attacked_res['acc']:.4f} "
                    f"macro_f1={attacked_res['metrics']['macro_f1']:.4f} "
                    f"macro_recall={attacked_res['metrics']['macro_recall']:.4f}"
                )

        clean_micro_acc = float("nan") if clean_total_eval == 0 else (clean_total_correct / clean_total_eval)
        attacked_micro_acc = float("nan") if attacked_total_eval == 0 else (attacked_total_correct / attacked_total_eval)

        clean_metrics = self._metrics_from_confusion(clean_confusion, total_eval=clean_total_eval)
        attacked_metrics = self._metrics_from_confusion(attacked_confusion, total_eval=attacked_total_eval)

        def _mean_var(vals: List[float]) -> Tuple[float, float]:
            if len(vals) == 0:
                return float("nan"), float("nan")
            t = torch.tensor(vals, dtype=torch.float)
            return float(t.mean().item()), float(t.var(unbiased=False).item())

        clean_acc_mean, clean_acc_var = _mean_var([r["acc"] for r in clean_per_sample])
        attacked_acc_mean, attacked_acc_var = _mean_var([r["acc"] for r in attacked_per_sample])

        clean_macro_f1_mean, clean_macro_f1_var = _mean_var([r["metrics"]["macro_f1"] for r in clean_per_sample])
        attacked_macro_f1_mean, attacked_macro_f1_var = _mean_var([r["metrics"]["macro_f1"] for r in attacked_per_sample])

        clean_macro_recall_mean, clean_macro_recall_var = _mean_var([r["metrics"]["macro_recall"] for r in clean_per_sample])
        attacked_macro_recall_mean, attacked_macro_recall_var = _mean_var([r["metrics"]["macro_recall"] for r in attacked_per_sample])
        attacked_target_summary = self._finalize_target_totals(attacked_target_totals)

        summary: Dict[str, Any] = {
            "split": split,
            "target_type": target_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metrics": {
                "clean_micro_acc": clean_micro_acc,
                "attacked_micro_acc": attacked_micro_acc,
                "clean_total_eval": int(clean_total_eval),
                "attacked_total_eval": int(attacked_total_eval),
                "clean_micro_precision": clean_metrics["micro_precision"],
                "clean_micro_recall": clean_metrics["micro_recall"],
                "clean_micro_f1": clean_metrics["micro_f1"],
                "clean_macro_precision": clean_metrics["macro_precision"],
                "clean_macro_recall": clean_metrics["macro_recall"],
                "clean_macro_f1": clean_metrics["macro_f1"],
                "attacked_micro_precision": attacked_metrics["micro_precision"],
                "attacked_micro_recall": attacked_metrics["micro_recall"],
                "attacked_micro_f1": attacked_metrics["micro_f1"],
                "attacked_macro_precision": attacked_metrics["macro_precision"],
                "attacked_macro_recall": attacked_metrics["macro_recall"],
                "attacked_macro_f1": attacked_metrics["macro_f1"],
                "clean_acc_mean": clean_acc_mean,
                "clean_acc_var": clean_acc_var,
                "clean_macro_f1_mean": clean_macro_f1_mean,
                "clean_macro_f1_var": clean_macro_f1_var,
                "clean_macro_recall_mean": clean_macro_recall_mean,
                "clean_macro_recall_var": clean_macro_recall_var,
                "attacked_acc_mean": attacked_acc_mean,
                "attacked_acc_var": attacked_acc_var,
                "attacked_macro_f1_mean": attacked_macro_f1_mean,
                "attacked_macro_f1_var": attacked_macro_f1_var,
                "attacked_macro_recall_mean": attacked_macro_recall_mean,
                "attacked_macro_recall_var": attacked_macro_recall_var,
                "attacked_target_eval_count": attacked_target_summary["target_eval_count"],
                "attacked_target_samples": attacked_target_summary["samples_with_targets"],
                "attacked_target_num_true_pos": attacked_target_summary["num_true_pos"],
                "attacked_target_num_true_pos_undetected_after": attacked_target_summary["num_true_pos_undetected_after"],
                "attacked_target_num_clean_pos": attacked_target_summary["num_clean_pos"],
                "attacked_target_num_clean_neg": attacked_target_summary["num_clean_neg"],
                "attacked_target_num_pos_to_neg": attacked_target_summary["num_pos_to_neg"],
                "attacked_target_num_neg_to_pos": attacked_target_summary["num_neg_to_pos"],
                "attacked_target_asr_good": attacked_target_summary["asr_good"],
                "attacked_target_asr_bad": attacked_target_summary["asr_bad"],
                "attacked_target_asr_post": attacked_target_summary["asr_post"],
                "attacked_target_asr_on_attacked": attacked_target_summary["asr_on_attacked"],
                "attacked_target_nfr": attacked_target_summary["nfr"],
            },
            "details": {
                "clean": {
                    "micro_acc": clean_micro_acc,
                    "total_eval": int(clean_total_eval),
                    "total_correct": int(clean_total_correct),
                    "per_sample": clean_per_sample,
                },
                "attacked": {
                    "micro_acc": attacked_micro_acc,
                    "total_eval": int(attacked_total_eval),
                    "total_correct": int(attacked_total_correct),
                    "target_summary": attacked_target_summary,
                    "per_sample": attacked_per_sample,
                },
            },
        }

        out_dir = self._resolve_output_dir()
        tag = f"{split}_attack_only"
        pt_path = os.path.join(out_dir, f"{tag}_results.pt")
        json_path = os.path.join(out_dir, f"{tag}_metrics.json")

        torch.save(summary, pt_path)

        metrics_only = {
            "split": summary["split"],
            "target_type": summary["target_type"],
            "timestamp": summary["timestamp"],
            "metrics": summary["metrics"],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_only, f, ensure_ascii=False, indent=2)

        print(
            f"[{tag}] clean={summary['metrics']['clean_micro_acc']:.4f} "
            f"(macro_f1={summary['metrics']['clean_macro_f1']:.4f}) | "
            f"attacked={summary['metrics']['attacked_micro_acc']:.4f} "
            f"(macro_f1={summary['metrics']['attacked_macro_f1']:.4f}) "
            f"(saved to {out_dir})"
        )

        return summary

    def poison_then_defend(
        self,
        split: str = "test",
        train_epochs: int = 1,
        shuffle: bool = False,
        log_every: int = 1,
    ) -> Dict[str, Any]:
        """
        Poison and defend on the same subgraph split, then retrain/evaluate classifier per dataset variant.

        Flow (for subgraph split `split`, default "test"):
          1) build poisoned_split = attacker.attack(split[i]) for all subgraphs
          2) build defended_split = defender.defend(poisoned_split[i]) for all subgraphs
          3) reset classifier -> train on clean split -> eval on node-level test_mask
          4) reset classifier -> train on poisoned split -> eval on node-level test_mask
          5) reset classifier -> train on defended split -> eval on node-level test_mask

        Outputs:
          - {split}_poison_then_defend_results.pt  (full tensors)
          - {split}_poison_then_defend_metrics.json (metrics only)
        """
        ds = self._get_split_dataset(split)
        sample0 = ds[0] if len(ds) > 0 else None
        if sample0 is None:
            raise ValueError(f"Empty dataset for split='{split}'.")
        target_type = self._infer_target_type(sample0)

        # Build poisoned/defended datasets from the same subgraph split to avoid cross-split leakage.
        poisoned_ds: List[Any] = []
        defended_ds: List[Any] = []
        for i in tqdm(range(len(ds)), desc=f"Preparing {split} (poison/defend)"):
            g = ds[i]
            g_poison = self.attacker.attack(g.clone() if hasattr(g, "clone") else g)
            g_defend = self.defender.defend(g_poison.clone() if hasattr(g_poison, "clone") else g_poison)
            poisoned_ds.append(g_poison)
            defended_ds.append(g_defend)
            if log_every and (i + 1) % log_every == 0:
                print(f"[prepare #{i + 1}/{len(ds)}] poisoned + defended ready")

        # Snapshot training state; each stage will restart from the same point.
        init_state = self.classifier_engine.snapshot_state()

        def _reset_training_state():
            self.classifier_engine.reset_from_snapshot(init_state, prefer_reset_parameters=True)

        def _mean_var(vals: List[float]) -> Tuple[float, float]:
            if len(vals) == 0:
                return float("nan"), float("nan")
            t = torch.tensor(vals, dtype=torch.float)
            finite = torch.isfinite(t)
            if int(finite.sum().item()) == 0:
                return float("nan"), float("nan")
            t = t[finite]
            return float(t.mean().item()), float(t.var(unbiased=False).item())

        def _train_on_dataset(dataset, name: str) -> List[Dict[str, float]]:
            history: List[Dict[str, float]] = []
            for epoch in tqdm(range(train_epochs), desc=f"Train {name}"):
                order = torch.randperm(len(dataset)).tolist() if shuffle else list(range(len(dataset)))
                train_losses = []
                val_losses = []
                for idx in tqdm(order, desc=f"{name} epoch {epoch + 1}/{train_epochs}", leave=False):
                    g = dataset[idx]
                    g_in = g.clone() if hasattr(g, "clone") else g
                    if hasattr(g_in, "to"):
                        g_in = g_in.to(self.device)
                    train_loss, val_loss = self.train_classifier_vanilla_oneloop(g_in)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                avg_train = float(sum(train_losses) / max(len(train_losses), 1))
                avg_val = float(sum(val_losses) / max(len(val_losses), 1))
                history.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})
                tqdm.write(
                    f"[poison_then_defend:{name}] epoch={epoch + 1}/{train_epochs} | "
                    f"train_loss={avg_train:.6f} | val_loss={avg_val:.6f}"
                )
            return history

        def _eval_with_metrics(dataset, name: str) -> Dict[str, Any]:
            base_eval = self._eval_over_dataset(
                dataset=dataset,
                split="test",  # node-level test mask
                target_type=target_type,
                transform_fn=None,
                desc=f"{name} eval",
            )
            per_sample = base_eval["per_sample"]
            total_eval = int(base_eval["total_eval"])

            max_cls = -1
            for rec in per_sample:
                if rec["y_true"].numel() > 0:
                    max_cls = max(max_cls, int(rec["y_true"].max().item()))
                if rec["y_pred"].numel() > 0:
                    max_cls = max(max_cls, int(rec["y_pred"].max().item()))
            num_classes = max(1, max_cls + 1)
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
            for rec in per_sample:
                if rec["y_true"].numel() == 0:
                    continue
                y_true = rec["y_true"].to(torch.long)
                y_pred = rec["y_pred"].to(torch.long)
                flat = y_true * num_classes + y_pred
                confusion += torch.bincount(flat, minlength=num_classes * num_classes).view(num_classes, num_classes)

            metrics = self._metrics_from_confusion(confusion, total_eval=total_eval)
            acc_mean, acc_var = _mean_var([r["acc"] for r in per_sample])
            macro_f1_mean, macro_f1_var = _mean_var([r["metrics"]["macro_f1"] for r in per_sample if "metrics" in r])
            macro_recall_mean, macro_recall_var = _mean_var([r["metrics"]["macro_recall"] for r in per_sample if "metrics" in r])

            return {
                "micro_acc": base_eval["micro_acc"],
                "total_eval": total_eval,
                "total_correct": int(base_eval["total_correct"]),
                "metrics": metrics,
                "acc_mean": acc_mean,
                "acc_var": acc_var,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_var": macro_f1_var,
                "macro_recall_mean": macro_recall_mean,
                "macro_recall_var": macro_recall_var,
                "per_sample": per_sample,
            }

        stages = {
            "clean": ds,
            "poisoned": poisoned_ds,
            "defended": defended_ds,
        }
        results: Dict[str, Any] = {}

        for name, stage_ds in stages.items():
            _reset_training_state()
            history = _train_on_dataset(stage_ds, name=name)
            eval_result = _eval_with_metrics(stage_ds, name=name)
            results[name] = {
                "train_history": history,
                "eval": eval_result,
            }
            print(
                f"[poison_then_defend:{name}] "
                f"test_micro_acc={eval_result['micro_acc']:.4f} | "
                f"test_macro_f1={eval_result['metrics']['macro_f1']:.4f}"
            )

        summary: Dict[str, Any] = {
            "split": split,
            "target_type": target_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metrics": {
                "clean_micro_acc": results["clean"]["eval"]["micro_acc"],
                "poisoned_micro_acc": results["poisoned"]["eval"]["micro_acc"],
                "defended_micro_acc": results["defended"]["eval"]["micro_acc"],
                "clean_macro_f1": results["clean"]["eval"]["metrics"]["macro_f1"],
                "poisoned_macro_f1": results["poisoned"]["eval"]["metrics"]["macro_f1"],
                "defended_macro_f1": results["defended"]["eval"]["metrics"]["macro_f1"],
                "clean_total_eval": results["clean"]["eval"]["total_eval"],
                "poisoned_total_eval": results["poisoned"]["eval"]["total_eval"],
                "defended_total_eval": results["defended"]["eval"]["total_eval"],
            },
            "details": results,
        }

        out_dir = self._resolve_output_dir()
        tag = f"{split}_poison_then_defend"
        pt_path = os.path.join(out_dir, f"{tag}_results.pt")
        json_path = os.path.join(out_dir, f"{tag}_metrics.json")

        torch.save(summary, pt_path)
        metrics_only = {
            "split": summary["split"],
            "target_type": summary["target_type"],
            "timestamp": summary["timestamp"],
            "metrics": summary["metrics"],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_only, f, ensure_ascii=False, indent=2)

        print(
            f"[{tag}] clean={summary['metrics']['clean_micro_acc']:.4f} | "
            f"poisoned={summary['metrics']['poisoned_micro_acc']:.4f} | "
            f"defended={summary['metrics']['defended_micro_acc']:.4f} "
            f"(saved to {out_dir})"
        )

        return summary

    # -----------------------------
    # Dataset / config helpers
    # -----------------------------
    def _get_split_dataset(self, split: str):
        dm = self.dataset_module
        if hasattr(dm, "hetero_datasets") and isinstance(dm.hetero_datasets, dict):
            if split not in dm.hetero_datasets:
                raise KeyError(f"dataset_module.hetero_datasets has no split '{split}'.")
            return dm.hetero_datasets[split]

        raise AttributeError(
            "Cannot locate split dataset. Expected dataset_module.hetero_datasets[split]"
        )

    def _infer_target_type(self, sample_graph) -> str:
        dm = self.dataset_module
        # YelpChi style: dm.target
        if hasattr(dm, "target") and isinstance(dm.target, str) and len(dm.target) > 0:
            return dm.target
        # common names
        for attr in ["target_node_type", "target_type", "target_node"]:
            if hasattr(dm, attr):
                v = getattr(dm, attr)
                if isinstance(v, str) and len(v) > 0:
                    return v
        # classifier hint
        if hasattr(self.classifier, "target_node") and isinstance(self.classifier.target_node, str):
            return self.classifier.target_node
        # last resort
        if hasattr(sample_graph, "node_types") and len(sample_graph.node_types) == 1:
            return sample_graph.node_types[0]
        raise AttributeError("Cannot infer target node type. Please set datamodule.target or classifier.target_node.")

    def _resolve_output_dir(self) -> str:
        dm = self.dataset_module

        # If cfg exists, try common places
        if hasattr(dm, "cfg"):
            cfg = dm.cfg
            for path_attr in ["out_dir", "output_dir", "log_dir", "save_dir", "result_dir"]:
                if hasattr(cfg, path_attr):
                    d = getattr(cfg, path_attr)
                    if isinstance(d, str) and len(d) > 0:
                        os.makedirs(d, exist_ok=True)
                        return d

        # Try datamodule itself
        for attr in ["out_dir", "output_dir", "log_dir", "save_dir", "result_dir"]:
            if hasattr(dm, attr):
                d = getattr(dm, attr)
                if isinstance(d, str) and len(d) > 0:
                    os.makedirs(d, exist_ok=True)
                    return d

        # Fall back to dataset root if present
        if hasattr(dm, "root") and isinstance(dm.root, str) and len(dm.root) > 0:
            d = os.path.join(dm.root, "pipeline_outputs")
            os.makedirs(d, exist_ok=True)
            return d

        d = os.path.join(os.getcwd(), "pipeline_outputs")
        os.makedirs(d, exist_ok=True)
        return d

    # -----------------------------
    # Evaluation helpers
    # -----------------------------
    @torch.no_grad()
    def _eval_classifier_on_graph(self, graph, split: str, target_type: str) -> Dict[str, Any]:
        """
        Evaluate classifier on ONE graph sample and return per-node predictions (masked) + metrics.
        """
        res = self.classifier_engine.eval_on_graph(graph=graph, split=split, target_type=target_type)
        res["metrics"] = self._metrics_from_confusion(res["confusion"], total_eval=res["num_eval"])
        return res

    @staticmethod
    def _metrics_from_confusion(confusion: torch.Tensor, total_eval: int) -> Dict[str, float]:
        """
        Compute micro/macro precision, recall, f1 from a confusion matrix.
        """
        eps = 1e-12
        tp = torch.diag(confusion).to(torch.float)
        pred_sum = confusion.sum(dim=0).to(torch.float)
        true_sum = confusion.sum(dim=1).to(torch.float)

        precision = tp / (pred_sum + eps)
        recall = tp / (true_sum + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        macro_precision = float(precision.mean().item()) if precision.numel() > 0 else float("nan")
        macro_recall = float(recall.mean().item()) if recall.numel() > 0 else float("nan")
        macro_f1 = float(f1.mean().item()) if f1.numel() > 0 else float("nan")

        micro_tp = float(tp.sum().item())
        micro_precision = micro_tp / (float(pred_sum.sum().item()) + eps)
        micro_recall = micro_tp / (float(true_sum.sum().item()) + eps)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)

        micro_acc = float("nan") if total_eval == 0 else (micro_tp / float(total_eval))

        return {
            "micro_acc": micro_acc,
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    @staticmethod
    def _extract_attack_target_info(graph, default_target_type: str) -> Tuple[Optional[torch.Tensor], str]:
        """
        Extract attacker-provided target node ids for targeted evasion metrics.
        """
        target_type = default_target_type
        for attr in ["attack_target_type", "minta_target_type"]:
            if hasattr(graph, attr):
                v = getattr(graph, attr)
                if isinstance(v, str) and len(v) > 0:
                    target_type = v
                    break

        nodes = None
        for attr in ["attack_target_nodes", "minta_adv_nodes", "adv_nodes_test"]:
            if hasattr(graph, attr):
                nodes = getattr(graph, attr)
                break

        if nodes is None:
            return None, target_type

        if isinstance(nodes, torch.Tensor):
            node_tensor = nodes.detach().to(torch.long).view(-1).cpu()
        else:
            node_tensor = torch.as_tensor(nodes, dtype=torch.long).view(-1).cpu()

        if node_tensor.numel() == 0:
            return None, target_type

        node_tensor = torch.unique(node_tensor, sorted=False)
        return node_tensor, target_type

    @staticmethod
    def _build_pred_map(eval_res: Dict[str, Any]) -> Dict[int, int]:
        idx = eval_res["node_idx"].detach().to(torch.long).view(-1).cpu()
        pred = eval_res["y_pred"].detach().to(torch.long).view(-1).cpu()
        return {int(i.item()): int(p.item()) for i, p in zip(idx, pred)}

    @staticmethod
    def _build_true_map(eval_res: Dict[str, Any]) -> Dict[int, int]:
        idx = eval_res["node_idx"].detach().to(torch.long).view(-1).cpu()
        y_true = eval_res["y_true"].detach().to(torch.long).view(-1).cpu()
        return {int(i.item()): int(t.item()) for i, t in zip(idx, y_true)}

    @classmethod
    def _compute_transition_metrics_on_targets(
        cls,
        baseline_res: Dict[str, Any],
        variant_res: Dict[str, Any],
        attack_nodes: Optional[torch.Tensor],
        positive_label: int = 1,
    ) -> Dict[str, Any]:
        """
        MintA-style targeted transition metrics using clean prediction as baseline.
        """
        empty = {
            "target_eval_count": 0,
            "num_true_pos": 0,
            "num_true_pos_undetected_after": 0,
            "num_clean_pos": 0,
            "num_clean_neg": 0,
            "num_pos_to_neg": 0,
            "num_neg_to_pos": 0,
            "asr_good": float("nan"),
            "asr_bad": float("nan"),
            "asr_post": float("nan"),
            "asr_on_attacked": float("nan"),
            "nfr": float("nan"),
        }
        if attack_nodes is None or attack_nodes.numel() == 0:
            return empty

        base_map = cls._build_pred_map(baseline_res)
        var_map = cls._build_pred_map(variant_res)
        true_map = cls._build_true_map(baseline_res)
        common_nodes = [
            int(n)
            for n in attack_nodes.tolist()
            if (int(n) in base_map and int(n) in var_map and int(n) in true_map)
        ]
        if len(common_nodes) == 0:
            return empty

        base_pred = torch.tensor([base_map[n] for n in common_nodes], dtype=torch.long)
        var_pred = torch.tensor([var_map[n] for n in common_nodes], dtype=torch.long)
        y_true = torch.tensor([true_map[n] for n in common_nodes], dtype=torch.long)

        clean_pos = base_pred == int(positive_label)
        clean_neg = ~clean_pos

        num_clean_pos = int(clean_pos.sum().item())
        num_clean_neg = int(clean_neg.sum().item())
        num_pos_to_neg = int((clean_pos & (var_pred != int(positive_label))).sum().item())
        num_neg_to_pos = int((clean_neg & (var_pred == int(positive_label))).sum().item())

        asr_good = float("nan") if num_clean_pos == 0 else (num_pos_to_neg / num_clean_pos)
        asr_bad = float("nan") if num_clean_neg == 0 else (num_neg_to_pos / num_clean_neg)
        asr_on_attacked = float("nan") if len(common_nodes) == 0 else (num_pos_to_neg / len(common_nodes))

        true_pos = y_true == int(positive_label)
        num_true_pos = int(true_pos.sum().item())
        num_true_pos_undetected_after = int((true_pos & (var_pred != int(positive_label))).sum().item())
        asr_post = float("nan") if num_true_pos == 0 else (num_true_pos_undetected_after / num_true_pos)

        return {
            "target_eval_count": len(common_nodes),
            "num_true_pos": num_true_pos,
            "num_true_pos_undetected_after": num_true_pos_undetected_after,
            "num_clean_pos": num_clean_pos,
            "num_clean_neg": num_clean_neg,
            "num_pos_to_neg": num_pos_to_neg,
            "num_neg_to_pos": num_neg_to_pos,
            "asr_good": asr_good,
            "asr_bad": asr_bad,
            "asr_post": asr_post,
            "asr_on_attacked": asr_on_attacked,
            "nfr": asr_bad,
        }

    @classmethod
    def _compute_defense_recovery_on_targets(
        cls,
        clean_res: Dict[str, Any],
        attacked_res: Dict[str, Any],
        defended_res: Dict[str, Any],
        attack_nodes: Optional[torch.Tensor],
        positive_label: int = 1,
    ) -> Dict[str, Any]:
        """
        Recovery on originally-positive targeted nodes: clean=1, attacked=0, defended=1.
        """
        empty = {
            "target_eval_count": 0,
            "attack_success_pos": 0,
            "attack_success_pos_recovered": 0,
            "recovery_rate_pos": float("nan"),
        }
        if attack_nodes is None or attack_nodes.numel() == 0:
            return empty

        clean_map = cls._build_pred_map(clean_res)
        attacked_map = cls._build_pred_map(attacked_res)
        defended_map = cls._build_pred_map(defended_res)
        common_nodes = [
            int(n)
            for n in attack_nodes.tolist()
            if (int(n) in clean_map and int(n) in attacked_map and int(n) in defended_map)
        ]
        if len(common_nodes) == 0:
            return empty

        clean_pred = torch.tensor([clean_map[n] for n in common_nodes], dtype=torch.long)
        attacked_pred = torch.tensor([attacked_map[n] for n in common_nodes], dtype=torch.long)
        defended_pred = torch.tensor([defended_map[n] for n in common_nodes], dtype=torch.long)

        attack_success = (clean_pred == int(positive_label)) & (attacked_pred != int(positive_label))
        recovered = attack_success & (defended_pred == int(positive_label))

        attack_success_pos = int(attack_success.sum().item())
        attack_success_pos_recovered = int(recovered.sum().item())
        recovery_rate_pos = (
            float("nan")
            if attack_success_pos == 0
            else (attack_success_pos_recovered / attack_success_pos)
        )
        return {
            "target_eval_count": len(common_nodes),
            "attack_success_pos": attack_success_pos,
            "attack_success_pos_recovered": attack_success_pos_recovered,
            "recovery_rate_pos": recovery_rate_pos,
        }

    @staticmethod
    def _init_target_totals() -> Dict[str, float]:
        return {
            "samples_with_targets": 0,
            "target_eval_count": 0,
            "num_true_pos": 0,
            "num_true_pos_undetected_after": 0,
            "num_clean_pos": 0,
            "num_clean_neg": 0,
            "num_pos_to_neg": 0,
            "num_neg_to_pos": 0,
        }

    @staticmethod
    def _accumulate_target_totals(
        totals: Dict[str, float],
        sample_metrics: Optional[Dict[str, Any]],
    ) -> None:
        if sample_metrics is None:
            return
        cnt = int(sample_metrics.get("target_eval_count", 0))
        if cnt <= 0:
            return
        totals["samples_with_targets"] += 1
        totals["target_eval_count"] += cnt
        totals["num_true_pos"] += int(sample_metrics.get("num_true_pos", 0))
        totals["num_true_pos_undetected_after"] += int(sample_metrics.get("num_true_pos_undetected_after", 0))
        totals["num_clean_pos"] += int(sample_metrics.get("num_clean_pos", 0))
        totals["num_clean_neg"] += int(sample_metrics.get("num_clean_neg", 0))
        totals["num_pos_to_neg"] += int(sample_metrics.get("num_pos_to_neg", 0))
        totals["num_neg_to_pos"] += int(sample_metrics.get("num_neg_to_pos", 0))

    @staticmethod
    def _finalize_target_totals(totals: Dict[str, float]) -> Dict[str, Any]:
        pos = int(totals["num_clean_pos"])
        neg = int(totals["num_clean_neg"])
        pos_to_neg = int(totals["num_pos_to_neg"])
        neg_to_pos = int(totals["num_neg_to_pos"])
        asr_good = float("nan") if pos == 0 else (pos_to_neg / pos)
        asr_bad = float("nan") if neg == 0 else (neg_to_pos / neg)
        attacked_cnt = int(totals["target_eval_count"])
        asr_on_attacked = float("nan") if attacked_cnt == 0 else (pos_to_neg / attacked_cnt)
        true_pos = int(totals["num_true_pos"])
        true_pos_undetected_after = int(totals["num_true_pos_undetected_after"])
        asr_post = float("nan") if true_pos == 0 else (true_pos_undetected_after / true_pos)
        return {
            "samples_with_targets": int(totals["samples_with_targets"]),
            "target_eval_count": attacked_cnt,
            "num_true_pos": true_pos,
            "num_true_pos_undetected_after": true_pos_undetected_after,
            "num_clean_pos": pos,
            "num_clean_neg": neg,
            "num_pos_to_neg": pos_to_neg,
            "num_neg_to_pos": neg_to_pos,
            "asr_good": asr_good,
            "asr_bad": asr_bad,
            "asr_post": asr_post,
            "asr_on_attacked": asr_on_attacked,
            "nfr": asr_bad,
        }

    @staticmethod
    def _init_recovery_totals() -> Dict[str, float]:
        return {
            "samples_with_targets": 0,
            "target_eval_count": 0,
            "attack_success_pos": 0,
            "attack_success_pos_recovered": 0,
        }

    @staticmethod
    def _accumulate_recovery_totals(
        totals: Dict[str, float],
        sample_metrics: Optional[Dict[str, Any]],
    ) -> None:
        if sample_metrics is None:
            return
        cnt = int(sample_metrics.get("target_eval_count", 0))
        if cnt <= 0:
            return
        totals["samples_with_targets"] += 1
        totals["target_eval_count"] += cnt
        totals["attack_success_pos"] += int(sample_metrics.get("attack_success_pos", 0))
        totals["attack_success_pos_recovered"] += int(sample_metrics.get("attack_success_pos_recovered", 0))

    @staticmethod
    def _finalize_recovery_totals(totals: Dict[str, float]) -> Dict[str, Any]:
        atk_success = int(totals["attack_success_pos"])
        atk_recovered = int(totals["attack_success_pos_recovered"])
        recovery_rate = float("nan") if atk_success == 0 else (atk_recovered / atk_success)
        return {
            "samples_with_targets": int(totals["samples_with_targets"]),
            "target_eval_count": int(totals["target_eval_count"]),
            "attack_success_pos": atk_success,
            "attack_success_pos_recovered": atk_recovered,
            "recovery_rate_pos": recovery_rate,
        }

    def _eval_over_dataset(
        self,
        dataset,
        split: str,
        target_type: str,
        transform_fn=None,
        desc: str = "",
    ) -> Dict[str, Any]:
        """
        Iterate dataset (assumed indexable Dataset of HeteroData samples).
        Optionally apply transform_fn(graph)->graph per sample before evaluation.

        Returns micro metrics + per-sample prediction records.
        """
        total_eval = 0
        total_correct = 0
        per_sample: List[Dict[str, Any]] = []

        # Deterministic order; dataset is already a processed on-disk dataset.
        for i in tqdm(range(len(dataset)), desc=f"Evaluating {desc}"):
            g = dataset[i]
            if transform_fn is not None:
                # clone to avoid writing back into cached dataset items
                g_in = g.clone() if hasattr(g, "clone") else g
                g = transform_fn(g_in)

            res = self._eval_classifier_on_graph(g, split=split, target_type=target_type)

            total_eval += res["num_eval"]
            total_correct += res["num_correct"]

            per_sample.append(
                {
                    "sample_id": i,
                    "num_eval": res["num_eval"],
                    "num_correct": res["num_correct"],
                    "acc": res["acc"],
                    "y_true": res["y_true"],
                    "y_pred": res["y_pred"],
                    "logits": res["logits"],
                    "node_idx": res["node_idx"],  # indices within this subgraph's target node store
                }
            )

        micro_acc = float("nan") if total_eval == 0 else (total_correct / total_eval)

        return {
            "micro_acc": micro_acc,
            "total_eval": int(total_eval),
            "total_correct": int(total_correct),
            "per_sample": per_sample,
        }
    
    def train_classifier_vanilla_oneloop(self, data, weights=None):
        return self.classifier_engine.train_oneloop(data)
    
    def train_classifier_vanilla(
        self,
        split: str = "train",
        epochs: int = 100,
        shuffle: bool = False,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 0.001,
    ) -> List[Dict[str, float]]:
        """
        Train classifier over the full split dataset for N epochs.
        Returns per-epoch average train/val losses.
        """
        ds = self._get_split_dataset(split)
        if not (hasattr(ds, "__len__") and hasattr(ds, "__getitem__")):
            ds = [ds]
        elif hasattr(ds, "x_dict") or hasattr(ds, "edge_index_dict") or hasattr(ds, "x"):
            ds = [ds]

        num_samples = len(ds)
        if num_samples == 0:
            raise ValueError(f"Empty dataset for split='{split}'.")

        history: List[Dict[str, float]] = []
        best_val = float("inf")
        bad_epochs = 0
        for epoch in tqdm(range(epochs), desc=f"Epochs ({split})", dynamic_ncols=True, leave=True):
            if shuffle:
                order = torch.randperm(num_samples).tolist()
            else:
                order = list(range(num_samples))

            train_losses = []
            val_losses = []
            for i in tqdm(
                order,
                desc=f"Training {split} epoch {epoch + 1}/{epochs}",
                leave=False,
                dynamic_ncols=True,
                position=1,
            ):
                data = ds[i]
                if hasattr(data, "to"):
                    data = data.to(self.device)
                train_loss, val_loss = self.train_classifier_vanilla_oneloop(data)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

            avg_train = float(sum(train_losses) / max(len(train_losses), 1))
            avg_val = float(sum(val_losses) / max(len(val_losses), 1))
            history.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})

            tqdm.write(
                f"[train_classifier_vanilla] epoch={epoch + 1}/{epochs} | "
                f"train_loss={avg_train:.6f} | val_loss={avg_val:.6f}"
            )

            if early_stop_patience and early_stop_patience > 0:
                if (best_val - avg_val) > early_stop_min_delta:
                    best_val = avg_val
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stop_patience:
                        print(
                            f"[train_classifier_vanilla] early stop at epoch {epoch + 1} "
                            f"(best_val={best_val:.6f}, patience={early_stop_patience})"
                        )
                        break

        return history

    # -----------------------------
    # Main API
    # -----------------------------
    def defend_after_attack(
        self,
        split: str = "test",
        need_training: bool = True,
        log_every: int = 1,
        positive_label: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate classifier on clean / attacked / defended versions of the hetero subgraph dataset.

        Flow:
          clean:   eval(dataset[i])
          attacked: eval(attacker.attack(dataset[i]))
          defended: eval(defender.defend(attacker.attack(dataset[i])))

        Outputs:
          - {split}_defend_after_attack_results.pt  (full tensors)
          - {split}_defend_after_attack_metrics.json (metrics only)
        """
        if need_training:
            print(f"[defend_after_attack] Training classifier on {split} split before evaluation...")
            self.train_classifier_vanilla()
        ds = self._get_split_dataset(split)

        # infer target type from first sample
        sample0 = ds[0] if len(ds) > 0 else None
        if sample0 is None:
            raise ValueError(f"Empty dataset for split='{split}'.")
        target_type = self._infer_target_type(sample0)

        # sequential per-sample evaluation for immediate feedback
        total = len(ds)
        clean_total_eval = 0
        clean_total_correct = 0
        attacked_total_eval = 0
        attacked_total_correct = 0
        defended_total_eval = 0
        defended_total_correct = 0

        clean_confusion = None
        attacked_confusion = None
        defended_confusion = None

        clean_per_sample: List[Dict[str, Any]] = []
        attacked_per_sample: List[Dict[str, Any]] = []
        defended_per_sample: List[Dict[str, Any]] = []
        attacked_target_totals = self._init_target_totals()
        defended_target_totals = self._init_target_totals()
        recovery_totals = self._init_recovery_totals()

        for i in tqdm(range(total), desc=f"Evaluating {split} (clean/attacked/defended)"):
            g = ds[i]

            # clean
            clean_res = self._eval_classifier_on_graph(g, split=split, target_type=target_type)
            clean_total_eval += clean_res["num_eval"]
            clean_total_correct += clean_res["num_correct"]
            clean_confusion = (
                clean_res["confusion"].clone()
                if clean_confusion is None
                else clean_confusion + clean_res["confusion"]
            )
            clean_per_sample.append(
                {
                    "sample_id": i,
                    "num_eval": clean_res["num_eval"],
                    "num_correct": clean_res["num_correct"],
                    "acc": clean_res["acc"],
                    "metrics": clean_res["metrics"],
                    "y_true": clean_res["y_true"],
                    "y_pred": clean_res["y_pred"],
                    "logits": clean_res["logits"],
                    "node_idx": clean_res["node_idx"],
                }
            )

            # attacked
            g_attacked = self.attacker.attack(g.clone() if hasattr(g, "clone") else g)
            attacked_res = self._eval_classifier_on_graph(g_attacked, split=split, target_type=target_type)
            attacked_total_eval += attacked_res["num_eval"]
            attacked_total_correct += attacked_res["num_correct"]
            attacked_confusion = (
                attacked_res["confusion"].clone()
                if attacked_confusion is None
                else attacked_confusion + attacked_res["confusion"]
            )
            attack_nodes, attack_target_type = self._extract_attack_target_info(
                g_attacked, default_target_type=target_type
            )
            attacked_target_metrics = None
            if attack_nodes is not None and attack_target_type == target_type:
                attacked_target_metrics = self._compute_transition_metrics_on_targets(
                    baseline_res=clean_res,
                    variant_res=attacked_res,
                    attack_nodes=attack_nodes,
                    positive_label=positive_label,
                )
                self._accumulate_target_totals(attacked_target_totals, attacked_target_metrics)
            attacked_per_sample.append(
                {
                    "sample_id": i,
                    "num_eval": attacked_res["num_eval"],
                    "num_correct": attacked_res["num_correct"],
                    "acc": attacked_res["acc"],
                    "metrics": attacked_res["metrics"],
                    "y_true": attacked_res["y_true"],
                    "y_pred": attacked_res["y_pred"],
                    "logits": attacked_res["logits"],
                    "node_idx": attacked_res["node_idx"],
                    "target_metrics": attacked_target_metrics,
                }
            )

            # defended (attack then defend)
            g_defended = self.defender.defend(g_attacked.clone() if hasattr(g_attacked, "clone") else g_attacked)
            defended_res = self._eval_classifier_on_graph(g_defended, split=split, target_type=target_type)
            defended_total_eval += defended_res["num_eval"]
            defended_total_correct += defended_res["num_correct"]
            defended_confusion = (
                defended_res["confusion"].clone()
                if defended_confusion is None
                else defended_confusion + defended_res["confusion"]
            )
            defended_target_metrics = None
            recovery_metrics = None
            if attack_nodes is not None and attack_target_type == target_type:
                defended_target_metrics = self._compute_transition_metrics_on_targets(
                    baseline_res=clean_res,
                    variant_res=defended_res,
                    attack_nodes=attack_nodes,
                    positive_label=positive_label,
                )
                recovery_metrics = self._compute_defense_recovery_on_targets(
                    clean_res=clean_res,
                    attacked_res=attacked_res,
                    defended_res=defended_res,
                    attack_nodes=attack_nodes,
                    positive_label=positive_label,
                )
                self._accumulate_target_totals(defended_target_totals, defended_target_metrics)
                self._accumulate_recovery_totals(recovery_totals, recovery_metrics)
            defended_per_sample.append(
                {
                    "sample_id": i,
                    "num_eval": defended_res["num_eval"],
                    "num_correct": defended_res["num_correct"],
                    "acc": defended_res["acc"],
                    "metrics": defended_res["metrics"],
                    "y_true": defended_res["y_true"],
                    "y_pred": defended_res["y_pred"],
                    "logits": defended_res["logits"],
                    "node_idx": defended_res["node_idx"],
                    "target_metrics": defended_target_metrics,
                    "recovery_metrics": recovery_metrics,
                }
            )

            if log_every and (i + 1) % log_every == 0:
                print(
                    f"[{split} #{i + 1}/{total}] "
                    f"clean acc={clean_res['acc']:.4f} "
                    f"macro_f1={clean_res['metrics']['macro_f1']:.4f} "
                    f"macro_recall={clean_res['metrics']['macro_recall']:.4f} | "
                    f"attacked acc={attacked_res['acc']:.4f} "
                    f"macro_f1={attacked_res['metrics']['macro_f1']:.4f} "
                    f"macro_recall={attacked_res['metrics']['macro_recall']:.4f} | "
                    f"defended acc={defended_res['acc']:.4f} "
                    f"macro_f1={defended_res['metrics']['macro_f1']:.4f} "
                    f"macro_recall={defended_res['metrics']['macro_recall']:.4f}"
                )

        clean_micro_acc = float("nan") if clean_total_eval == 0 else (clean_total_correct / clean_total_eval)
        attacked_micro_acc = float("nan") if attacked_total_eval == 0 else (attacked_total_correct / attacked_total_eval)
        defended_micro_acc = float("nan") if defended_total_eval == 0 else (defended_total_correct / defended_total_eval)

        clean_metrics = self._metrics_from_confusion(clean_confusion, total_eval=clean_total_eval)
        attacked_metrics = self._metrics_from_confusion(attacked_confusion, total_eval=attacked_total_eval)
        defended_metrics = self._metrics_from_confusion(defended_confusion, total_eval=defended_total_eval)

        def _mean_var(vals: List[float]) -> Tuple[float, float]:
            if len(vals) == 0:
                return float("nan"), float("nan")
            t = torch.tensor(vals, dtype=torch.float)
            return float(t.mean().item()), float(t.var(unbiased=False).item())

        clean_acc_mean, clean_acc_var = _mean_var([r["acc"] for r in clean_per_sample])
        attacked_acc_mean, attacked_acc_var = _mean_var([r["acc"] for r in attacked_per_sample])
        defended_acc_mean, defended_acc_var = _mean_var([r["acc"] for r in defended_per_sample])

        clean_macro_f1_mean, clean_macro_f1_var = _mean_var([r["metrics"]["macro_f1"] for r in clean_per_sample])
        attacked_macro_f1_mean, attacked_macro_f1_var = _mean_var([r["metrics"]["macro_f1"] for r in attacked_per_sample])
        defended_macro_f1_mean, defended_macro_f1_var = _mean_var([r["metrics"]["macro_f1"] for r in defended_per_sample])

        clean_macro_recall_mean, clean_macro_recall_var = _mean_var([r["metrics"]["macro_recall"] for r in clean_per_sample])
        attacked_macro_recall_mean, attacked_macro_recall_var = _mean_var([r["metrics"]["macro_recall"] for r in attacked_per_sample])
        defended_macro_recall_mean, defended_macro_recall_var = _mean_var([r["metrics"]["macro_recall"] for r in defended_per_sample])
        attacked_target_summary = self._finalize_target_totals(attacked_target_totals)
        defended_target_summary = self._finalize_target_totals(defended_target_totals)
        recovery_summary = self._finalize_recovery_totals(recovery_totals)

        summary: Dict[str, Any] = {
            "split": split,
            "target_type": target_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metrics": {
                "clean_micro_acc": clean_micro_acc,
                "attacked_micro_acc": attacked_micro_acc,
                "defended_micro_acc": defended_micro_acc,
                "clean_total_eval": int(clean_total_eval),
                "attacked_total_eval": int(attacked_total_eval),
                "defended_total_eval": int(defended_total_eval),
                "clean_micro_precision": clean_metrics["micro_precision"],
                "clean_micro_recall": clean_metrics["micro_recall"],
                "clean_micro_f1": clean_metrics["micro_f1"],
                "clean_macro_precision": clean_metrics["macro_precision"],
                "clean_macro_recall": clean_metrics["macro_recall"],
                "clean_macro_f1": clean_metrics["macro_f1"],
                "attacked_micro_precision": attacked_metrics["micro_precision"],
                "attacked_micro_recall": attacked_metrics["micro_recall"],
                "attacked_micro_f1": attacked_metrics["micro_f1"],
                "attacked_macro_precision": attacked_metrics["macro_precision"],
                "attacked_macro_recall": attacked_metrics["macro_recall"],
                "attacked_macro_f1": attacked_metrics["macro_f1"],
                "defended_micro_precision": defended_metrics["micro_precision"],
                "defended_micro_recall": defended_metrics["micro_recall"],
                "defended_micro_f1": defended_metrics["micro_f1"],
                "defended_macro_precision": defended_metrics["macro_precision"],
                "defended_macro_recall": defended_metrics["macro_recall"],
                "defended_macro_f1": defended_metrics["macro_f1"],
                "clean_acc_mean": clean_acc_mean,
                "clean_acc_var": clean_acc_var,
                "clean_macro_f1_mean": clean_macro_f1_mean,
                "clean_macro_f1_var": clean_macro_f1_var,
                "clean_macro_recall_mean": clean_macro_recall_mean,
                "clean_macro_recall_var": clean_macro_recall_var,
                "attacked_acc_mean": attacked_acc_mean,
                "attacked_acc_var": attacked_acc_var,
                "attacked_macro_f1_mean": attacked_macro_f1_mean,
                "attacked_macro_f1_var": attacked_macro_f1_var,
                "attacked_macro_recall_mean": attacked_macro_recall_mean,
                "attacked_macro_recall_var": attacked_macro_recall_var,
                "defended_acc_mean": defended_acc_mean,
                "defended_acc_var": defended_acc_var,
                "defended_macro_f1_mean": defended_macro_f1_mean,
                "defended_macro_f1_var": defended_macro_f1_var,
                "defended_macro_recall_mean": defended_macro_recall_mean,
                "defended_macro_recall_var": defended_macro_recall_var,
                "attacked_target_eval_count": attacked_target_summary["target_eval_count"],
                "attacked_target_samples": attacked_target_summary["samples_with_targets"],
                "attacked_target_num_true_pos": attacked_target_summary["num_true_pos"],
                "attacked_target_num_true_pos_undetected_after": attacked_target_summary["num_true_pos_undetected_after"],
                "attacked_target_num_clean_pos": attacked_target_summary["num_clean_pos"],
                "attacked_target_num_clean_neg": attacked_target_summary["num_clean_neg"],
                "attacked_target_num_pos_to_neg": attacked_target_summary["num_pos_to_neg"],
                "attacked_target_num_neg_to_pos": attacked_target_summary["num_neg_to_pos"],
                "attacked_target_asr_good": attacked_target_summary["asr_good"],
                "attacked_target_asr_bad": attacked_target_summary["asr_bad"],
                "attacked_target_asr_post": attacked_target_summary["asr_post"],
                "attacked_target_asr_on_attacked": attacked_target_summary["asr_on_attacked"],
                "attacked_target_nfr": attacked_target_summary["nfr"],
                "defended_target_eval_count": defended_target_summary["target_eval_count"],
                "defended_target_samples": defended_target_summary["samples_with_targets"],
                "defended_target_num_true_pos": defended_target_summary["num_true_pos"],
                "defended_target_num_true_pos_undetected_after": defended_target_summary["num_true_pos_undetected_after"],
                "defended_target_num_clean_pos": defended_target_summary["num_clean_pos"],
                "defended_target_num_clean_neg": defended_target_summary["num_clean_neg"],
                "defended_target_num_pos_to_neg": defended_target_summary["num_pos_to_neg"],
                "defended_target_num_neg_to_pos": defended_target_summary["num_neg_to_pos"],
                "defended_target_asr_good": defended_target_summary["asr_good"],
                "defended_target_asr_bad": defended_target_summary["asr_bad"],
                "defended_target_asr_post": defended_target_summary["asr_post"],
                "defended_target_asr_on_attacked": defended_target_summary["asr_on_attacked"],
                "defended_target_nfr": defended_target_summary["nfr"],
                "defense_recovery_target_eval_count": recovery_summary["target_eval_count"],
                "defense_recovery_target_samples": recovery_summary["samples_with_targets"],
                "defense_recovery_attack_success_pos": recovery_summary["attack_success_pos"],
                "defense_recovery_attack_success_pos_recovered": recovery_summary["attack_success_pos_recovered"],
                "defense_recovery_rate_pos": recovery_summary["recovery_rate_pos"],
            },
            "details": {
                "clean": {
                    "micro_acc": clean_micro_acc,
                    "total_eval": int(clean_total_eval),
                    "total_correct": int(clean_total_correct),
                    "per_sample": clean_per_sample,
                },
                "attacked": {
                    "micro_acc": attacked_micro_acc,
                    "total_eval": int(attacked_total_eval),
                    "total_correct": int(attacked_total_correct),
                    "target_summary": attacked_target_summary,
                    "per_sample": attacked_per_sample,
                },
                "defended": {
                    "micro_acc": defended_micro_acc,
                    "total_eval": int(defended_total_eval),
                    "total_correct": int(defended_total_correct),
                    "target_summary": defended_target_summary,
                    "recovery_summary": recovery_summary,
                    "per_sample": defended_per_sample,
                },
            },
        }

        out_dir = self._resolve_output_dir()
        tag = f"{split}_defend_after_attack"
        pt_path = os.path.join(out_dir, f"{tag}_results.pt")
        json_path = os.path.join(out_dir, f"{tag}_metrics.json")

        # Save full tensors
        torch.save(summary, pt_path)

        # Save metrics-only JSON
        metrics_only = {
            "split": summary["split"],
            "target_type": summary["target_type"],
            "timestamp": summary["timestamp"],
            "metrics": summary["metrics"],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_only, f, ensure_ascii=False, indent=2)

        print(
            f"[{tag}] clean={summary['metrics']['clean_micro_acc']:.4f} "
            f"(macro_f1={summary['metrics']['clean_macro_f1']:.4f}) | "
            f"attacked={summary['metrics']['attacked_micro_acc']:.4f} "
            f"(macro_f1={summary['metrics']['attacked_macro_f1']:.4f}) | "
            f"defended={summary['metrics']['defended_micro_acc']:.4f} "
            f"(macro_f1={summary['metrics']['defended_macro_f1']:.4f}) "
            f"(saved to {out_dir})"
        )

        return summary

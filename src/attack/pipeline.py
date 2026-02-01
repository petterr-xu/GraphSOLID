import os
import json
from tqdm import tqdm
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .attack import Attacker
from .defend import Defender
from src.dataset.abstract_dataset import AbstractDataModule
from src import loss_fn


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
        self._to_device(device)

    def _to_device(self, device):
        if device is None:
            raise ValueError("Device must be specified.")
        self.classifier.to(device)
        self.defender.to(device)


    def attack_onely(self):
        raise NotImplementedError

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
        self.classifier.eval()
        device = next(self.classifier.parameters()).device

        g = graph.to(device)

        # HeteroNN classifier expects (x_dict, edge_index_dict) for hetero graphs.
        if hasattr(g, "x_dict") and hasattr(g, "edge_index_dict"):
            logits = self.classifier(g.x_dict, g.edge_index_dict)
            y = g[target_type].y
            mask = getattr(g[target_type], f"{split}_mask")
        else:
            # homogeneous fallback
            logits = self.classifier(g)
            y = g.y
            mask = getattr(g, f"{split}_mask")

        # labels might be one-hot
        if y.dim() > 1 and y.size(-1) > 1:
            y_true = y.argmax(dim=-1)
        else:
            y_true = y.view(-1)

        pred = logits.argmax(dim=-1)
        num_classes = int(logits.size(-1))

        idx = mask.nonzero(as_tuple=False).view(-1)
        num = int(idx.numel())
        if num == 0:
            correct = 0
            acc = float("nan")
        else:
            correct = int((pred[idx] == y_true[idx]).sum().item())
            acc = correct / num

        # build confusion matrix for masked nodes
        if num == 0:
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        else:
            flat = (y_true[idx] * num_classes + pred[idx]).to(torch.long)
            confusion = torch.bincount(flat, minlength=num_classes * num_classes).view(num_classes, num_classes)

        metrics = self._metrics_from_confusion(confusion, total_eval=num)

        return {
            "acc": acc,
            "num_eval": num,
            "num_correct": correct,
            "num_classes": num_classes,
            "confusion": confusion,
            "metrics": metrics,
            "y_true": y_true[idx].detach().cpu(),
            "y_pred": pred[idx].detach().cpu(),
            "logits": logits[idx].detach().cpu(),
            "node_idx": idx.detach().cpu(),
        }

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
        device = self.device
        target = self.target
        
        self.classifier.train()
        self.classifier_optimizer.zero_grad()
        
        # 调用 HeteroGNN_classifier，传入字典格式的数据
        logits = self.classifier(data.x_dict, data.edge_index_dict)
        
        # 提取目标节点的标签和掩码
        labels = data[target].y
        train_mask = data[target].train_mask
        val_mask = data[target].val_mask
        
        # 计算训练损失
        loss = self.classifier_criterion.compute(logits[train_mask], labels[train_mask])
        loss.backward()
        self.classifier_optimizer.step()

        # 验证步骤
        with torch.no_grad():
            self.classifier.eval()
            output = self.classifier(data.x_dict, data.edge_index_dict)
            val_loss = self.classifier_criterion.compute(output[val_mask], labels[val_mask])
        self.classifier_optimizer.step()
        self.cl_scheduler.step(val_loss)
        return loss.item(), val_loss.item()
    
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
        for epoch in tqdm(range(epochs), desc=f"Epochs ({split})"):
            if shuffle:
                order = torch.randperm(num_samples).tolist()
            else:
                order = list(range(num_samples))

            train_losses = []
            val_losses = []
            for i in tqdm(order, desc=f"Training {split} epoch {epoch + 1}/{epochs}", leave=False):
                data = ds[i]
                if hasattr(data, "to"):
                    data = data.to(self.device)
                train_loss, val_loss = self.train_classifier_vanilla_oneloop(data)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

            avg_train = float(sum(train_losses) / max(len(train_losses), 1))
            avg_val = float(sum(val_losses) / max(len(val_losses), 1))
            history.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})

            print(
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
                    "per_sample": attacked_per_sample,
                },
                "defended": {
                    "micro_acc": defended_micro_acc,
                    "total_eval": int(defended_total_eval),
                    "total_correct": int(defended_total_correct),
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

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from src.dataset.abstract_dataset import AbstractDataModule
from src.models.classifier_engine import ClassifierEngine
from .attacker import Attacker
from .defender import Defender


class SurrogateAttackPipeline:
    """
    Minimal pipeline dedicated to surrogate attacks.
    - Supports attack-only and poison-then-defend flows
    - Enforces classifier-engine/surrogate-engine architecture consistency
    """

    def __init__(
        self,
        dataset_module: AbstractDataModule,
        attacker: Attacker,
        classifier_engine: ClassifierEngine,
        defender: Optional[Defender] = None,
        device: str = "cuda:0",
        target_node: Optional[str] = None,
    ):
        self.dataset_module = dataset_module
        self.attacker = attacker
        self.defender = defender
        self.classifier_engine = classifier_engine
        self.device = device
        self.target_node = target_node
        self.classifier_engine.to(device)
        if self.defender is not None:
            self.defender.to(device)
        self._validate_engine_compatibility()

    def _validate_engine_compatibility(self):
        attacker_engine = getattr(self.attacker, "surrogate_engine", None)
        if attacker_engine is None:
            return
        if type(attacker_engine) is not type(self.classifier_engine):
            raise ValueError(
                "Classifier engine and attacker surrogate engine must share the same class for surrogate pipeline. "
                f"Got classifier={type(self.classifier_engine).__name__}, "
                f"surrogate={type(attacker_engine).__name__}."
            )

    def _get_split_dataset(self, split: str):
        dm = self.dataset_module
        if hasattr(dm, "hetero_datasets") and isinstance(dm.hetero_datasets, dict):
            if split not in dm.hetero_datasets:
                raise KeyError(f"dataset_module.hetero_datasets has no split '{split}'.")
            return dm.hetero_datasets[split]
        raise AttributeError("Cannot locate split dataset. Expected dataset_module.hetero_datasets[split].")

    def _infer_target_type(self, sample_graph) -> str:
        if self.target_node is not None:
            return self.target_node
        dm = self.dataset_module
        if hasattr(dm, "target") and isinstance(dm.target, str) and dm.target:
            return dm.target
        for attr in ["target_node_type", "target_type", "target_node"]:
            if hasattr(dm, attr):
                v = getattr(dm, attr)
                if isinstance(v, str) and v:
                    return v
        raise AttributeError("Cannot infer target node type. Set target_node in pipeline ctor.")

    def _resolve_output_dir(self) -> str:
        dm = self.dataset_module
        if hasattr(dm, "cfg"):
            cfg = dm.cfg
            for path_attr in ["out_dir", "output_dir", "log_dir", "save_dir", "result_dir"]:
                if hasattr(cfg, path_attr):
                    d = getattr(cfg, path_attr)
                    if isinstance(d, str) and d:
                        os.makedirs(d, exist_ok=True)
                        return d
        if hasattr(dm, "root") and isinstance(dm.root, str) and dm.root:
            d = os.path.join(dm.root, "pipeline_outputs")
            os.makedirs(d, exist_ok=True)
            return d
        d = os.path.join(os.getcwd(), "pipeline_outputs")
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _metrics_from_confusion(confusion: torch.Tensor, total_eval: int) -> Dict[str, float]:
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
    def _extract_attack_target_info(graph, default_target_type: str):
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
        true_pos = int(totals["num_true_pos"])
        true_pos_undetected_after = int(totals["num_true_pos_undetected_after"])
        asr_post = float("nan") if true_pos == 0 else (true_pos_undetected_after / true_pos)
        return {
            "samples_with_targets": int(totals["samples_with_targets"]),
            "target_eval_count": int(totals["target_eval_count"]),
            "num_true_pos": true_pos,
            "num_true_pos_undetected_after": true_pos_undetected_after,
            "num_clean_pos": pos,
            "num_clean_neg": neg,
            "num_pos_to_neg": pos_to_neg,
            "num_neg_to_pos": neg_to_pos,
            "asr_good": asr_good,
            "asr_bad": asr_bad,
            "asr_post": asr_post,
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

    def _train_engine(self, dataset, epochs: int, shuffle: bool, stage_name: str):
        history: List[Dict[str, float]] = []
        for epoch in tqdm(range(epochs), desc=f"Train {stage_name}"):
            order = torch.randperm(len(dataset)).tolist() if shuffle else list(range(len(dataset)))
            train_losses = []
            val_losses = []
            for i in tqdm(order, desc=f"{stage_name} epoch {epoch + 1}/{epochs}", leave=False):
                g = dataset[i]
                g_in = g.clone() if hasattr(g, "clone") else g
                if hasattr(g_in, "to"):
                    g_in = g_in.to(self.device)
                train_loss, val_loss = self.classifier_engine.train_oneloop(g_in)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            avg_train = float(sum(train_losses) / max(len(train_losses), 1))
            avg_val = float(sum(val_losses) / max(len(val_losses), 1))
            history.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})
            tqdm.write(
                f"[surrogate:{stage_name}] epoch={epoch + 1}/{epochs} | "
                f"train_loss={avg_train:.6f} | val_loss={avg_val:.6f}"
            )
        return history

    def _eval_engine(self, dataset, target_type: str, eval_split: str = "test"):
        total_eval = 0
        total_correct = 0
        confusion = None
        per_sample: List[Dict[str, Any]] = []
        for i in tqdm(range(len(dataset)), desc=f"Eval {eval_split}"):
            g = dataset[i]
            res = self.classifier_engine.eval_on_graph(g, split=eval_split, target_type=target_type)
            total_eval += res["num_eval"]
            total_correct += res["num_correct"]
            confusion = res["confusion"].clone() if confusion is None else confusion + res["confusion"]
            per_sample.append(
                {
                    "sample_id": i,
                    "acc": res["acc"],
                    "num_eval": res["num_eval"],
                    "num_correct": res["num_correct"],
                    "y_true": res["y_true"],
                    "y_pred": res["y_pred"],
                    "logits": res["logits"],
                    "node_idx": res["node_idx"],
                }
            )
        if confusion is None:
            confusion = torch.zeros((2, 2), dtype=torch.long)
        metrics = self._metrics_from_confusion(confusion, total_eval=total_eval)
        return {
            "micro_acc": float("nan") if total_eval == 0 else total_correct / total_eval,
            "total_eval": int(total_eval),
            "total_correct": int(total_correct),
            "metrics": metrics,
            "per_sample": per_sample,
        }

    def run(
        self,
        split: str = "test",
        train_epochs: int = 1,
        shuffle: bool = False,
        log_every: int = 1,
        eval_split: str = "test",
        include_defended: Optional[bool] = None,
        positive_label: int = 1,
    ) -> Dict[str, Any]:
        ds = self._get_split_dataset(split)
        sample0 = ds[0] if len(ds) > 0 else None
        if sample0 is None:
            raise ValueError(f"Empty dataset for split='{split}'.")
        target_type = self._infer_target_type(sample0)

        if include_defended is None:
            include_defended = self.defender is not None
        if include_defended and self.defender is None:
            raise ValueError("include_defended=True requires a defender, but defender is None.")

        poisoned_ds = []
        defended_ds = [] if include_defended else None
        prep_desc = f"Poison/Defend {split}" if include_defended else f"Poison {split}"
        for i in tqdm(range(len(ds)), desc=prep_desc):
            g = ds[i]
            g_poison = self.attacker.attack(g.clone() if hasattr(g, "clone") else g)
            poisoned_ds.append(g_poison)
            if include_defended:
                g_def = self.defender.defend(g_poison.clone() if hasattr(g_poison, "clone") else g_poison)
                defended_ds.append(g_def)
            if log_every and (i + 1) % log_every == 0:
                if include_defended:
                    print(f"[poison+defend #{i + 1}/{len(ds)}] done")
                else:
                    print(f"[poison #{i + 1}/{len(ds)}] done")

        init_state = self.classifier_engine.snapshot_state()

        def _run_stage(stage_name: str, stage_ds):
            self.classifier_engine.reset_from_snapshot(init_state, prefer_reset_parameters=True)
            history = self._train_engine(stage_ds, epochs=train_epochs, shuffle=shuffle, stage_name=stage_name)
            result = self._eval_engine(stage_ds, target_type=target_type, eval_split=eval_split)
            return {"train_history": history, "eval": result}

        clean_result = _run_stage("clean", ds)
        poison_result = _run_stage("poisoned", poisoned_ds)
        defend_result = _run_stage("defended", defended_ds) if include_defended else None

        poisoned_target_totals = self._init_target_totals()
        defended_target_totals = self._init_target_totals()
        recovery_totals = self._init_recovery_totals()

        for i in range(len(poisoned_ds)):
            clean_eval = clean_result["eval"]["per_sample"][i]
            poison_eval = poison_result["eval"]["per_sample"][i]
            attack_nodes, attack_target_type = self._extract_attack_target_info(
                poisoned_ds[i], default_target_type=target_type
            )

            poisoned_target_metrics = None
            if attack_nodes is not None and attack_target_type == target_type:
                poisoned_target_metrics = self._compute_transition_metrics_on_targets(
                    baseline_res=clean_eval,
                    variant_res=poison_eval,
                    attack_nodes=attack_nodes,
                    positive_label=positive_label,
                )
                self._accumulate_target_totals(poisoned_target_totals, poisoned_target_metrics)
            poison_eval["target_metrics"] = poisoned_target_metrics

            if include_defended and defend_result is not None:
                defend_eval = defend_result["eval"]["per_sample"][i]
                defended_target_metrics = None
                recovery_metrics = None
                if attack_nodes is not None and attack_target_type == target_type:
                    defended_target_metrics = self._compute_transition_metrics_on_targets(
                        baseline_res=clean_eval,
                        variant_res=defend_eval,
                        attack_nodes=attack_nodes,
                        positive_label=positive_label,
                    )
                    recovery_metrics = self._compute_defense_recovery_on_targets(
                        clean_res=clean_eval,
                        attacked_res=poison_eval,
                        defended_res=defend_eval,
                        attack_nodes=attack_nodes,
                        positive_label=positive_label,
                    )
                    self._accumulate_target_totals(defended_target_totals, defended_target_metrics)
                    self._accumulate_recovery_totals(recovery_totals, recovery_metrics)

                defend_eval["target_metrics"] = defended_target_metrics
                defend_eval["recovery_metrics"] = recovery_metrics

        poisoned_target_summary = self._finalize_target_totals(poisoned_target_totals)
        defended_target_summary = self._finalize_target_totals(defended_target_totals)
        recovery_summary = self._finalize_recovery_totals(recovery_totals)

        summary: Dict[str, Any] = {
            "split": split,
            "eval_split": eval_split,
            "target_type": target_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metrics": {
                "clean_micro_acc": clean_result["eval"]["micro_acc"],
                "poisoned_micro_acc": poison_result["eval"]["micro_acc"],
                "clean_macro_f1": clean_result["eval"]["metrics"]["macro_f1"],
                "poisoned_macro_f1": poison_result["eval"]["metrics"]["macro_f1"],
                "clean_total_eval": clean_result["eval"]["total_eval"],
                "poisoned_total_eval": poison_result["eval"]["total_eval"],
                "poisoned_target_eval_count": poisoned_target_summary["target_eval_count"],
                "poisoned_target_samples": poisoned_target_summary["samples_with_targets"],
                "poisoned_target_asr_good": poisoned_target_summary["asr_good"],
                "poisoned_target_asr_bad": poisoned_target_summary["asr_bad"],
                "poisoned_target_asr_post": poisoned_target_summary["asr_post"],
                "poisoned_target_nfr": poisoned_target_summary["nfr"],
            },
            "details": {
                "clean": clean_result,
                "poisoned": {
                    **poison_result,
                    "target_summary": poisoned_target_summary,
                },
            },
        }
        if include_defended and defend_result is not None:
            summary["metrics"]["defended_micro_acc"] = defend_result["eval"]["micro_acc"]
            summary["metrics"]["defended_macro_f1"] = defend_result["eval"]["metrics"]["macro_f1"]
            summary["metrics"]["defended_total_eval"] = defend_result["eval"]["total_eval"]
            summary["metrics"]["defended_target_eval_count"] = defended_target_summary["target_eval_count"]
            summary["metrics"]["defended_target_samples"] = defended_target_summary["samples_with_targets"]
            summary["metrics"]["defended_target_asr_good"] = defended_target_summary["asr_good"]
            summary["metrics"]["defended_target_asr_bad"] = defended_target_summary["asr_bad"]
            summary["metrics"]["defended_target_asr_post"] = defended_target_summary["asr_post"]
            summary["metrics"]["defended_target_nfr"] = defended_target_summary["nfr"]
            summary["metrics"]["defense_recovery_target_eval_count"] = recovery_summary["target_eval_count"]
            summary["metrics"]["defense_recovery_target_samples"] = recovery_summary["samples_with_targets"]
            summary["metrics"]["defense_recovery_attack_success_pos"] = recovery_summary["attack_success_pos"]
            summary["metrics"]["defense_recovery_attack_success_pos_recovered"] = recovery_summary[
                "attack_success_pos_recovered"
            ]
            summary["metrics"]["defense_recovery_rate_pos"] = recovery_summary["recovery_rate_pos"]
            summary["details"]["defended"] = {
                **defend_result,
                "target_summary": defended_target_summary,
                "recovery_summary": recovery_summary,
            }

        out_dir = self._resolve_output_dir()
        tag = f"{split}_surrogate_poison_then_defend" if include_defended else f"{split}_surrogate_attack"
        pt_path = os.path.join(out_dir, f"{tag}_results.pt")
        json_path = os.path.join(out_dir, f"{tag}_metrics.json")
        torch.save(summary, pt_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": summary["split"],
                    "eval_split": summary["eval_split"],
                    "target_type": summary["target_type"],
                    "timestamp": summary["timestamp"],
                    "metrics": summary["metrics"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"[{tag}] clean={summary['metrics']['clean_micro_acc']:.4f} | "
            f"poisoned={summary['metrics']['poisoned_micro_acc']:.4f}"
            + (
                f" | defended={summary['metrics']['defended_micro_acc']:.4f}"
                if include_defended and "defended_micro_acc" in summary["metrics"]
                else ""
            )
            + f" (saved to {out_dir})"
        )
        return summary

    def poison_then_defend(
        self,
        split: str = "test",
        train_epochs: int = 1,
        shuffle: bool = False,
        log_every: int = 1,
        eval_split: str = "test",
        positive_label: int = 1,
    ) -> Dict[str, Any]:
        """
        Wrapper of run() for the standard clean/poisoned/defended flow.
        """
        return self.run(
            split=split,
            train_epochs=train_epochs,
            shuffle=shuffle,
            log_every=log_every,
            eval_split=eval_split,
            include_defended=True,
            positive_label=positive_label,
        )

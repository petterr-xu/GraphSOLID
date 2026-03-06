import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from src.dataset.abstract_dataset import AbstractDataModule
from src.models.classifier_engine import ClassifierEngine
from .attacker import Attacker, MintaAttacker
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

    def _train_engine(
        self,
        dataset,
        epochs: int,
        shuffle: bool,
        stage_name: str,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 1e-3,
        log_interval: int = 10,
    ):
        history: List[Dict[str, float]] = []
        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(epochs):
            order = torch.randperm(len(dataset)).tolist() if shuffle else list(range(len(dataset)))
            train_losses = []
            val_losses = []
            for i in order:
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
            if ((epoch + 1) % log_interval == 0) or epoch == 0 or (epoch + 1) == epochs:
                print(
                    f"[surrogate:{stage_name}] epoch={epoch + 1}/{epochs} "
                    f"train={avg_train:.4f} val={avg_val:.4f}"
                )

            if (best_val - avg_val) > early_stop_min_delta:
                best_val = avg_val
                bad_epochs = 0
            else:
                bad_epochs += 1
                if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
                    print(
                        f"[surrogate:{stage_name}] early stop at epoch {epoch + 1} "
                        f"(best_val={best_val:.4f})"
                    )
                    break
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

    @torch.no_grad()
    def _eval_classifier_on_graph(self, graph, split: str, target_type: str) -> Dict[str, Any]:
        """
        Evaluate classifier on one graph sample (no retraining), with per-node records.
        """
        res = self.classifier_engine.eval_on_graph(graph=graph, split=split, target_type=target_type)
        res["metrics"] = self._metrics_from_confusion(res["confusion"], total_eval=res["num_eval"])
        return res

    def _run_minta_evasion(
        self,
        split: str = "test",
        train_split: str = "train",
        train_epochs: int = 1,
        shuffle: bool = False,
        log_every: int = 1,
        eval_split: str = "test",
        include_defended: Optional[bool] = None,
        positive_label: int = 1,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 1e-3,
        train_log_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        MintA-aligned evaluation protocol:
        - Train ONE victim model on clean train split (optional if train_epochs=0),
        - Keep victim fixed,
        - Compare clean/attacked(/defended) predictions on the same model.
        """
        ds = self._get_split_dataset(split)
        sample0 = ds[0] if len(ds) > 0 else None
        if sample0 is None:
            raise ValueError(f"Empty dataset for split='{split}'.")
        target_type = self._infer_target_type(sample0)

        if include_defended is None:
            include_defended = self.defender is not None
        if include_defended and self.defender is None:
            raise ValueError("include_defended=True requires a defender, but defender is None.")

        train_history: List[Dict[str, float]] = []
        if train_epochs > 0:
            train_ds = self._get_split_dataset(train_split)
            train_history = self._train_engine(
                train_ds,
                epochs=train_epochs,
                shuffle=shuffle,
                stage_name=f"victim_{train_split}",
                early_stop_patience=early_stop_patience,
                early_stop_min_delta=early_stop_min_delta,
                log_interval=train_log_interval,
            )

        if isinstance(self.attacker, MintaAttacker):
            self.attacker.victim_model = self.classifier_engine.model
            self.attacker.victim_engine = self.classifier_engine
            self.attacker.device = self.device

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

        for i in tqdm(range(total), desc=f"Eval {split} (minta-evasion)"):
            g = ds[i]

            clean_res = self._eval_classifier_on_graph(g, split=eval_split, target_type=target_type)
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

            g_attacked = self.attacker.attack(g.clone() if hasattr(g, "clone") else g)
            attacked_res = self._eval_classifier_on_graph(g_attacked, split=eval_split, target_type=target_type)
            attack_info = getattr(self.attacker, "last_attack_info", None)
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
                    "attack_info": attack_info,
                }
            )

            if include_defended:
                g_defended = self.defender.defend(
                    g_attacked.clone() if hasattr(g_attacked, "clone") else g_attacked
                )
                defended_res = self._eval_classifier_on_graph(g_defended, split=eval_split, target_type=target_type)
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
                msg = (
                    f"[minta_evasion {split} #{i + 1}/{total}] "
                    f"clean={clean_res['acc']:.4f} attacked={attacked_res['acc']:.4f}"
                )
                if include_defended and len(defended_per_sample) > 0:
                    msg += f" defended={defended_per_sample[-1]['acc']:.4f}"
                print(msg)

        clean_micro_acc = float("nan") if clean_total_eval == 0 else (clean_total_correct / clean_total_eval)
        attacked_micro_acc = float("nan") if attacked_total_eval == 0 else (attacked_total_correct / attacked_total_eval)
        defended_micro_acc = (
            float("nan") if defended_total_eval == 0 else (defended_total_correct / defended_total_eval)
        )

        clean_metrics = self._metrics_from_confusion(clean_confusion, total_eval=clean_total_eval)
        attacked_metrics = self._metrics_from_confusion(attacked_confusion, total_eval=attacked_total_eval)
        defended_metrics = (
            self._metrics_from_confusion(defended_confusion, total_eval=defended_total_eval)
            if include_defended and defended_confusion is not None
            else None
        )

        attacked_target_summary = self._finalize_target_totals(attacked_target_totals)
        defended_target_summary = self._finalize_target_totals(defended_target_totals)
        recovery_summary = self._finalize_recovery_totals(recovery_totals)

        summary: Dict[str, Any] = {
            "split": split,
            "eval_split": eval_split,
            "train_split": train_split,
            "protocol": "minta_evasion",
            "target_type": target_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "train_history": train_history,
            "metrics": {
                "clean_micro_acc": clean_micro_acc,
                "attacked_micro_acc": attacked_micro_acc,
                "clean_macro_f1": clean_metrics["macro_f1"],
                "attacked_macro_f1": attacked_metrics["macro_f1"],
                "clean_total_eval": int(clean_total_eval),
                "attacked_total_eval": int(attacked_total_eval),
                "attacked_target_eval_count": attacked_target_summary["target_eval_count"],
                "attacked_target_samples": attacked_target_summary["samples_with_targets"],
                "attacked_target_asr_good": attacked_target_summary["asr_good"],
                "attacked_target_asr_bad": attacked_target_summary["asr_bad"],
                "attacked_target_asr_post": attacked_target_summary["asr_post"],
                "attacked_target_nfr": attacked_target_summary["nfr"],
                # compatibility aliases
                "poisoned_micro_acc": attacked_micro_acc,
                "poisoned_macro_f1": attacked_metrics["macro_f1"],
                "poisoned_total_eval": int(attacked_total_eval),
                "poisoned_target_eval_count": attacked_target_summary["target_eval_count"],
                "poisoned_target_samples": attacked_target_summary["samples_with_targets"],
                "poisoned_target_asr_good": attacked_target_summary["asr_good"],
                "poisoned_target_asr_bad": attacked_target_summary["asr_bad"],
                "poisoned_target_asr_post": attacked_target_summary["asr_post"],
                "poisoned_target_nfr": attacked_target_summary["nfr"],
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

        if include_defended:
            summary["metrics"]["defended_micro_acc"] = defended_micro_acc
            summary["metrics"]["defended_macro_f1"] = defended_metrics["macro_f1"] if defended_metrics else float("nan")
            summary["metrics"]["defended_total_eval"] = int(defended_total_eval)
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
                "micro_acc": defended_micro_acc,
                "total_eval": int(defended_total_eval),
                "total_correct": int(defended_total_correct),
                "target_summary": defended_target_summary,
                "recovery_summary": recovery_summary,
                "per_sample": defended_per_sample,
            }

        out_dir = self._resolve_output_dir()
        tag = f"{split}_surrogate_minta_evasion_defended" if include_defended else f"{split}_surrogate_minta_evasion"
        pt_path = os.path.join(out_dir, f"{tag}_results.pt")
        json_path = os.path.join(out_dir, f"{tag}_metrics.json")
        torch.save(summary, pt_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": summary["split"],
                    "eval_split": summary["eval_split"],
                    "train_split": summary["train_split"],
                    "protocol": summary["protocol"],
                    "target_type": summary["target_type"],
                    "timestamp": summary["timestamp"],
                    "metrics": summary["metrics"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"[{tag}] clean={summary['metrics']['clean_micro_acc']:.4f} "
            f"attacked={summary['metrics']['attacked_micro_acc']:.4f}"
            + (
                f" defended={summary['metrics']['defended_micro_acc']:.4f}"
                if include_defended and "defended_micro_acc" in summary["metrics"]
                else ""
            )
            + f" (saved to {out_dir})"
        )
        return summary

    def run(
        self,
        split: str = "test",
        train_epochs: int = 1,
        shuffle: bool = False,
        log_every: int = 1,
        eval_split: str = "test",
        include_defended: Optional[bool] = None,
        positive_label: int = 1,
        protocol: str = "auto",
        train_split: str = "train",
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 1e-3,
        train_log_interval: int = 10,
    ) -> Dict[str, Any]:
        if protocol not in {"auto", "poison_train", "minta_evasion"}:
            raise ValueError(f"Unknown protocol '{protocol}'.")
        if protocol == "auto":
            protocol = "minta_evasion" if isinstance(self.attacker, MintaAttacker) else "poison_train"
        if protocol == "minta_evasion":
            return self._run_minta_evasion(
                split=split,
                train_split=train_split,
                train_epochs=train_epochs,
                shuffle=shuffle,
                log_every=log_every,
                eval_split=eval_split,
                include_defended=include_defended,
                positive_label=positive_label,
                early_stop_patience=early_stop_patience,
                early_stop_min_delta=early_stop_min_delta,
                train_log_interval=train_log_interval,
            )

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
            history = self._train_engine(
                stage_ds,
                epochs=train_epochs,
                shuffle=shuffle,
                stage_name=stage_name,
                early_stop_patience=early_stop_patience,
                early_stop_min_delta=early_stop_min_delta,
                log_interval=train_log_interval,
            )
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
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 1e-3,
        train_log_interval: int = 10,
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
            protocol="poison_train",
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            train_log_interval=train_log_interval,
        )

    def minta_evasion(
        self,
        split: str = "test",
        train_split: str = "train",
        train_epochs: int = 100,
        shuffle: bool = False,
        log_every: int = 1,
        eval_split: str = "test",
        include_defended: Optional[bool] = True,
        positive_label: int = 1,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 1e-3,
        train_log_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        Wrapper for MintA-aligned evasion protocol.
        """
        return self.run(
            split=split,
            train_epochs=train_epochs,
            shuffle=shuffle,
            log_every=log_every,
            eval_split=eval_split,
            include_defended=include_defended,
            positive_label=positive_label,
            protocol="minta_evasion",
            train_split=train_split,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            train_log_interval=train_log_interval,
        )

    def evasion_then_defend(
        self,
        split: str = "test",
        train_split: str = "train",
        train_epochs: int = 100,
        shuffle: bool = False,
        log_every: int = 1,
        eval_split: str = "test",
        positive_label: int = 1,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 1e-3,
        train_log_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        Backward-compatible alias for MintA evasion + defense evaluation.
        """
        return self.minta_evasion(
            split=split,
            train_split=train_split,
            train_epochs=train_epochs,
            shuffle=shuffle,
            log_every=log_every,
            eval_split=eval_split,
            include_defended=True,
            positive_label=positive_label,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            train_log_interval=train_log_interval,
        )

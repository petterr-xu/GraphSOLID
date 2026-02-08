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
            },
            "details": {
                "clean": clean_result,
                "poisoned": poison_result,
            },
        }
        if include_defended and defend_result is not None:
            summary["metrics"]["defended_micro_acc"] = defend_result["eval"]["micro_acc"]
            summary["metrics"]["defended_macro_f1"] = defend_result["eval"]["metrics"]["macro_f1"]
            summary["metrics"]["defended_total_eval"] = defend_result["eval"]["total_eval"]
            summary["details"]["defended"] = defend_result

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
        )

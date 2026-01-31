import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .attack import Attacker
from .defend import Defender
from src.dataset.abstract_dataset import AbstractDataModule


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
    ):
        self.dataset_module = dataset_module
        self.defender = defender
        self.attacker = attacker
        self.classifier = classifier

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

        # fallback (older style) - try dm.<split>_dataset
        for attr in [f"{split}_dataset", f"{split}_data", f"{split}_graph"]:
            if hasattr(dm, attr):
                ds = getattr(dm, attr)
                if ds is not None:
                    return ds

        raise AttributeError(
            "Cannot locate split dataset. Expected dataset_module.hetero_datasets[split] or dataset_module.<split>_dataset."
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

        idx = mask.nonzero(as_tuple=False).view(-1)
        num = int(idx.numel())
        if num == 0:
            correct = 0
            acc = float("nan")
        else:
            correct = int((pred[idx] == y_true[idx]).sum().item())
            acc = correct / num

        return {
            "acc": acc,
            "num_eval": num,
            "num_correct": correct,
            "y_true": y_true[idx].detach().cpu(),
            "y_pred": pred[idx].detach().cpu(),
            "logits": logits[idx].detach().cpu(),
            "node_idx": idx.detach().cpu(),
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
        for i in range(len(dataset)):
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

    # -----------------------------
    # Main API
    # -----------------------------
    def defend_after_attack(self, split: str = "test") -> Dict[str, Any]:
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
        ds = self._get_split_dataset(split)

        # infer target type from first sample
        sample0 = ds[0] if len(ds) > 0 else None
        if sample0 is None:
            raise ValueError(f"Empty dataset for split='{split}'.")
        target_type = self._infer_target_type(sample0)

        # 1) clean
        clean = self._eval_over_dataset(
            dataset=ds, split=split, target_type=target_type, transform_fn=None, desc="clean"
        )

        # 2) attacked
        attacked = self._eval_over_dataset(
            dataset=ds,
            split=split,
            target_type=target_type,
            transform_fn=lambda g: self.attacker.attack(g),
            desc="attacked",
        )

        # 3) defended (attack then defend)
        defended = self._eval_over_dataset(
            dataset=ds,
            split=split,
            target_type=target_type,
            transform_fn=lambda g: self.defender.defend(self.attacker.attack(g)),
            desc="defended",
        )

        summary: Dict[str, Any] = {
            "split": split,
            "target_type": target_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metrics": {
                "clean_micro_acc": clean["micro_acc"],
                "attacked_micro_acc": attacked["micro_acc"],
                "defended_micro_acc": defended["micro_acc"],
                "clean_total_eval": clean["total_eval"],
                "attacked_total_eval": attacked["total_eval"],
                "defended_total_eval": defended["total_eval"],
            },
            "details": {
                "clean": clean,
                "attacked": attacked,
                "defended": defended,
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
            f"[{tag}] clean={summary['metrics']['clean_micro_acc']:.4f} | "
            f"attacked={summary['metrics']['attacked_micro_acc']:.4f} | "
            f"defended={summary['metrics']['defended_micro_acc']:.4f} "
            f"(saved to {out_dir})"
        )

        return summary

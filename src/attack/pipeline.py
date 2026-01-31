import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


from .attack import Attacker
from .defend import Defender
from src.dataset.abstract_dataset import AbstractDataModule

class DefaultPipeline:
    def __init__(self, dataset_module:AbstractDataModule,defender:Defender,attacker:Attacker,classifier:nn.Module):
        self.dataset_module = dataset_module
        self.defender = defender
        self.attacker = attacker
        self.classifier = classifier

    def attack_onely(self):
        # (Optional) kept for backward compatibility
        raise NotImplementedError

    # -----------------------------
    # Utilities
    # -----------------------------
    def _infer_graph(self):
        """Best-effort fetch of the (test) graph from dataset_module."""
        dm = self.dataset_module
        for attr in [
            "test_data",
            "test_graph",
            "data",
            "graph",
        ]:
            if hasattr(dm, attr):
                g = getattr(dm, attr)
                if g is not None:
                    return g
        # common getter names
        for fn in ["get_test_data", "get_test_graph", "get_data", "get_graph"]:
            if hasattr(dm, fn) and callable(getattr(dm, fn)):
                g = getattr(dm, fn)()
                if g is not None:
                    return g
        raise AttributeError(
            "Cannot infer graph from dataset_module. Expected one of: test_data/test_graph/data/graph or get_*()."
        )

    def _infer_target_type(self, data):
        """Infer target node type for hetero graphs."""
        # 1) dataset module hint
        for attr in ["target_node_type", "target_type", "target_node"]:
            if hasattr(self.dataset_module, attr):
                v = getattr(self.dataset_module, attr)
                if isinstance(v, str) and len(v) > 0:
                    return v
        # 2) classifier hint
        if hasattr(self.classifier, "target_node") and isinstance(self.classifier.target_node, str):
            return self.classifier.target_node
        # 3) fallback: single node type
        if hasattr(data, "node_types") and len(data.node_types) == 1:
            return data.node_types[0]
        raise AttributeError(
            "Cannot infer target node type. Please set dataset_module.target_node_type or classifier.target_node."
        )

    @torch.no_grad()
    def _eval_classifier_on_graph(self, data, split: str = "test") -> Dict[str, Any]:
        """Evaluate classifier and collect predictions/metrics on a given (hetero/homo) PyG graph."""
        self.classifier.eval()

        device = next(self.classifier.parameters()).device
        data = data.to(device)

        # HeteroData vs Data dispatch
        if hasattr(data, "x_dict") and hasattr(data, "edge_index_dict"):
            logits = self.classifier(data.x_dict, data.edge_index_dict)
            target_type = self._infer_target_type(data)
            y = data[target_type].y
            mask_name = f"{split}_mask"
            if not hasattr(data[target_type], mask_name):
                raise AttributeError(f"Target node store lacks mask: {target_type}.{mask_name}")
            mask = getattr(data[target_type], mask_name)
        else:
            logits = self.classifier(data.x, data.edge_index) if callable(self.classifier) else self.classifier(data)
            y = data.y
            mask_name = f"{split}_mask"
            if not hasattr(data, mask_name):
                raise AttributeError(f"Graph lacks mask: {mask_name}")
            mask = getattr(data, mask_name)

        # Ensure shapes
        if y.dim() > 1 and y.size(-1) > 1:
            y_true = y.argmax(dim=-1)
        else:
            y_true = y.view(-1)

        pred = logits.argmax(dim=-1)

        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            acc = float("nan")
        else:
            acc = (pred[idx] == y_true[idx]).float().mean().item()

        # Also keep raw logits for later analysis
        out = {
            "acc": acc,
            "num_eval": int(idx.numel()),
            "y_true": y_true.detach().cpu(),
            "y_pred": pred.detach().cpu(),
            "logits": logits.detach().cpu(),
            "mask": mask.detach().cpu(),
        }
        return out

    def _resolve_output_dir(self) -> str:
        for attr in ["output_dir", "out_dir", "log_dir", "save_dir", "result_dir"]:
            if hasattr(self.dataset_module, attr):
                d = getattr(self.dataset_module, attr)
                if isinstance(d, str) and len(d) > 0:
                    os.makedirs(d, exist_ok=True)
                    return d
        # default: current working directory
        d = os.getcwd()
        return d

    # -----------------------------
    # Main pipeline
    # -----------------------------
    def defend_after_attack(self, split: str = "test") -> Dict[str, Any]:
        """
        Evaluate classifier on:
          1) clean split graph
          2) attacked graph (attacker.attack)
          3) purified graph (defender.defend applied on attacked graph)
        Collect predictions and save outputs.

        Returns a dict with all metrics and predictions.
        """
        base_graph = self._infer_graph()

        # 1) Clean evaluation
        clean_graph = base_graph.clone() if hasattr(base_graph, "clone") else base_graph
        clean_res = self._eval_classifier_on_graph(clean_graph, split=split)

        # 2) Attack evaluation (avoid in-place modification surprises)
        attacked_in = base_graph.clone() if hasattr(base_graph, "clone") else base_graph
        attacked_graph = self.attacker.attack(attacked_in)
        attacked_res = self._eval_classifier_on_graph(attacked_graph, split=split)

        # 3) Defend evaluation
        defended_in = attacked_graph.clone() if hasattr(attacked_graph, "clone") else attacked_graph
        defended_graph = self.defender.defend(defended_in)
        defended_res = self._eval_classifier_on_graph(defended_graph, split=split)

        # Package results
        summary = {
            "split": split,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metrics": {
                "clean_acc": clean_res["acc"],
                "attacked_acc": attacked_res["acc"],
                "defended_acc": defended_res["acc"],
                "clean_num": clean_res["num_eval"],
                "attacked_num": attacked_res["num_eval"],
                "defended_num": defended_res["num_eval"],
            },
            "predictions": {
                "clean": {
                    "y_true": clean_res["y_true"],
                    "y_pred": clean_res["y_pred"],
                    "logits": clean_res["logits"],
                    "mask": clean_res["mask"],
                },
                "attacked": {
                    "y_true": attacked_res["y_true"],
                    "y_pred": attacked_res["y_pred"],
                    "logits": attacked_res["logits"],
                    "mask": attacked_res["mask"],
                },
                "defended": {
                    "y_true": defended_res["y_true"],
                    "y_pred": defended_res["y_pred"],
                    "logits": defended_res["logits"],
                    "mask": defended_res["mask"],
                },
            },
        }

        # Save outputs
        out_dir = self._resolve_output_dir()
        tag = f"{split}_defend_after_attack"
        pt_path = os.path.join(out_dir, f"{tag}_results.pt")
        json_path = os.path.join(out_dir, f"{tag}_metrics.json")

        # torch.save can store tensors directly
        torch.save(summary, pt_path)

        # JSON for quick look (tensors -> python scalars/lists)
        json_payload = {
            "split": summary["split"],
            "timestamp": summary["timestamp"],
            "metrics": summary["metrics"],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)

        # Optional: print a concise line
        print(
            f"[{tag}] clean={summary['metrics']['clean_acc']:.4f} | "
            f"attacked={summary['metrics']['attacked_acc']:.4f} | "
            f"defended={summary['metrics']['defended_acc']:.4f} "
            f"(saved: {pt_path})"
        )

        return summary

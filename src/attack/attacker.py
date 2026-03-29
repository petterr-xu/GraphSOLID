import torch
import warnings
import numpy as np
import random
import copy
import torch.nn as nn
from tqdm import tqdm
import tensorflow as tf
import scipy.sparse as sp
from typing import Any, Dict, Optional, Tuple
from torch_geometric.data import Data, HeteroData

from src.utils import graphbuilder
from src.models.classifier_engine import MintaSurrogateEngine, MetaSurrogateEngine
from ..dataset.abstract_dataset import AbstractDataModule
from src.gnn_meta_attack.metattack import meta_gradient_attack as mtk, utils

# class random_attacker():
#     pass

def quiet_tqdm(*args, **kwargs):
    kwargs['leave'] = False  # 让它跑完就消失，不占行
    # kwargs['disable'] = True # 如果想完全看不见，取消注释这一行
    return tqdm(*args, **kwargs)
mtk.tqdm = quiet_tqdm

class Attacker():
    attack_mode = "poison_train"
    def __init__(self, dataset_module:AbstractDataModule):
        self.dataset_module = dataset_module
    def to(self,device):
        return self
    def attack(self,data):
        return data

class RandomAttacker(Attacker):
    """
    Random targeted evasion attacker.

    It samples target nodes from the target-node test split and randomly perturbs
    edges incident to those nodes on one target-target relation.
    """
    attack_mode = "evasion"

    def __init__(
        self,
        dataset_module,
        perturb_ratio: float = 0.05,
        ctrl_nodes_size: int = 10,
        target_label: int = 1,
        only_attack_correctly_detected: bool = True,
        target_node_type: Optional[str] = None,
        edge_type_to_perturb: Optional[Tuple[str, str, str]] = None,
        victim_model: Optional[nn.Module] = None,
        victim_engine: Optional[Any] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        allow_self_loops: bool = False,
        max_sampling_rounds: int = 50,
        oversample_factor: int = 5,
    ):
        super().__init__(dataset_module)
        assert 0.0 <= perturb_ratio <= 1.0
        self.perturb_ratio = float(perturb_ratio)
        self.ctrl_nodes_size = int(ctrl_nodes_size)
        self.positive_label = int(target_label)
        self.only_attack_correctly_detected = bool(only_attack_correctly_detected)
        self.target_node_type = target_node_type
        self.edge_type_to_perturb = edge_type_to_perturb
        self.victim_model = victim_model
        self.victim_engine = victim_engine
        self.device = device
        self.seed = seed
        self.allow_self_loops = allow_self_loops
        self.max_sampling_rounds = int(max_sampling_rounds)
        self.oversample_factor = int(oversample_factor)
        self.last_attack_info: Dict[str, Any] = {}

    @staticmethod
    def _hash_edges(src: torch.Tensor, dst: torch.Tensor, num_dst: int) -> torch.Tensor:
        # unique id for each directed edge (src, dst)
        return src.to(torch.long) * int(num_dst) + dst.to(torch.long)

    def _resolve_target_type(self, data: HeteroData) -> str:
        if self.target_node_type:
            return self.target_node_type
        dm = self.dataset_module
        for attr in ["target", "target_node_type", "target_node"]:
            if hasattr(dm, attr):
                value = getattr(dm, attr)
                if isinstance(value, str) and value:
                    return value
        if hasattr(data, "node_types") and len(data.node_types) == 1:
            return data.node_types[0]
        raise ValueError("[RandomAttacker] Cannot infer target node type.")

    def _predict_nodes_with_victim(self, data: HeteroData, target: str, node_indices: np.ndarray) -> Optional[np.ndarray]:
        if node_indices is None or len(node_indices) == 0:
            return np.array([], dtype=np.int64)
        node_indices = np.asarray(node_indices, dtype=np.int64)

        if self.victim_model is not None:
            try:
                self.victim_model.eval()
                with torch.no_grad():
                    pred_raw = self.victim_model(data.x_dict, data.edge_index_dict)
                    if isinstance(pred_raw, dict):
                        pred_raw = pred_raw[target]
                    pred = pred_raw.argmax(dim=-1)
                return pred[node_indices].detach().cpu().numpy()
            except Exception:
                pass

        if self.victim_engine is not None and hasattr(self.victim_engine, "eval_on_graph"):
            pred_map: Dict[int, int] = {}
            for split_name in ("train", "val", "test"):
                try:
                    res = self.victim_engine.eval_on_graph(data, split=split_name, target_type=target)
                except Exception:
                    continue
                idx = res.get("node_idx")
                y_pred = res.get("y_pred")
                if idx is None or y_pred is None:
                    continue
                idx = idx.detach().to(torch.long).view(-1).cpu()
                y_pred = y_pred.detach().to(torch.long).view(-1).cpu()
                for n, p in zip(idx.tolist(), y_pred.tolist()):
                    pred_map[int(n)] = int(p)
            if pred_map:
                out = np.full((len(node_indices),), -1, dtype=np.int64)
                for i, n in enumerate(node_indices.tolist()):
                    if int(n) in pred_map:
                        out[i] = pred_map[int(n)]
                return out

        return None

    def _select_attack_nodes(self, data: HeteroData, target: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        y = data[target].y
        test_mask = getattr(data[target], "test_mask", None)
        if test_mask is None:
            raise ValueError("[RandomAttacker] target node test_mask is required.")

        test_idx = test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
        y_test = y[test_mask].cpu().numpy()
        pos_idx = test_idx[y_test == self.positive_label]
        if len(pos_idx) == 0:
            raise ValueError("[RandomAttacker] No positive test nodes found for attack.")

        candidates = pos_idx
        used_detected = False
        if self.only_attack_correctly_detected:
            pred_test = self._predict_nodes_with_victim(data, target, test_idx)
            if pred_test is not None:
                detected_idx = test_idx[pred_test == self.positive_label]
                detected_pos = np.intersect1d(pos_idx, detected_idx, assume_unique=False)
                if len(detected_pos) > 0:
                    candidates = detected_pos
                    used_detected = True

        k = min(self.ctrl_nodes_size, len(candidates))
        selected = np.array(random.sample(list(candidates), k), dtype=np.int64) if k > 0 else np.array([], dtype=np.int64)
        return selected, {
            "num_test_nodes": int(len(test_idx)),
            "num_positive_test_nodes": int(len(pos_idx)),
            "selected_adv_nodes": int(len(selected)),
            "used_detected_positive_subset": bool(used_detected),
            "positive_label": int(self.positive_label),
        }

    def _resolve_edge_type(self, data: HeteroData, target: str) -> Tuple[str, str, str]:
        if self.edge_type_to_perturb is not None:
            if self.edge_type_to_perturb not in data.edge_types:
                raise ValueError(f"[RandomAttacker] edge_type_to_perturb {self.edge_type_to_perturb} not in graph.")
            return self.edge_type_to_perturb
        candidates = [et for et in data.edge_types if et[0] == target and et[2] == target]
        if not candidates:
            raise ValueError("[RandomAttacker] No target-target edge type available for random evasion attack.")
        return candidates[0]

    def _compute_edge_budget(self, edge_index: torch.Tensor, attack_nodes: np.ndarray) -> Tuple[int, int]:
        if attack_nodes is None or len(attack_nodes) == 0:
            return 0, 0
        src = edge_index[0].detach().cpu().numpy()
        dst = edge_index[1].detach().cpu().numpy()
        deg_sum = int(np.isin(src, attack_nodes).sum() + np.isin(dst, attack_nodes).sum())
        edge_budget = max(0, int(np.floor(self.perturb_ratio * deg_sum)))
        return edge_budget, deg_sum

    def _sample_targeted_new_edges(
        self,
        num_nodes: int,
        attack_nodes: np.ndarray,
        k: int,
        existing_hash: torch.Tensor,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if k <= 0 or len(attack_nodes) == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        attack_nodes_t = torch.as_tensor(attack_nodes, dtype=torch.long, device=device)
        new_src_list = []
        new_dst_list = []
        existing = existing_hash
        remaining = k

        for _ in range(self.max_sampling_rounds):
            if remaining <= 0:
                break
            m = max(remaining * self.oversample_factor, remaining + 10)
            pick = torch.randint(0, attack_nodes_t.numel(), (m,), device=device, generator=generator)
            cand_src = attack_nodes_t[pick]
            cand_dst = torch.randint(0, num_nodes, (m,), device=device, generator=generator)
            if not self.allow_self_loops:
                mask = cand_src != cand_dst
                cand_src = cand_src[mask]
                cand_dst = cand_dst[mask]
                if cand_src.numel() == 0:
                    continue

            cand_hash = self._hash_edges(cand_src, cand_dst, num_nodes)
            keep = ~torch.isin(cand_hash, existing)
            cand_src = cand_src[keep]
            cand_dst = cand_dst[keep]
            cand_hash = cand_hash[keep]
            if cand_src.numel() == 0:
                continue

            order = torch.argsort(cand_hash)
            cand_hash = cand_hash[order]
            cand_src = cand_src[order]
            cand_dst = cand_dst[order]
            keep2 = torch.ones(cand_hash.size(0), dtype=torch.bool, device=device)
            keep2[1:] = cand_hash[1:] != cand_hash[:-1]
            cand_hash = cand_hash[keep2]
            cand_src = cand_src[keep2]
            cand_dst = cand_dst[keep2]

            take = min(remaining, cand_src.numel())
            if take <= 0:
                continue

            new_src_list.append(cand_src[:take])
            new_dst_list.append(cand_dst[:take])
            existing = torch.cat([existing, cand_hash[:take]], dim=0)
            remaining -= take

        if not new_src_list:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
        return torch.cat(new_src_list, dim=0), torch.cat(new_dst_list, dim=0)


    def attack(self, data):
        # Setup RNG
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        if isinstance(data, Data):
            raise RuntimeError("RandomAttacker expects HeteroData for evasion attacks.")

        if not isinstance(data, HeteroData):
            raise RuntimeError("Expected `data` to be `Data` or `HeteroData`.")

        out = data.clone()
        target = self._resolve_target_type(out)
        edge_type = self._resolve_edge_type(out, target)
        attack_nodes, select_info = self._select_attack_nodes(out, target)

        attack_nodes_tensor = torch.as_tensor(attack_nodes, dtype=torch.long).view(-1).cpu()
        out.attack_target_nodes = attack_nodes_tensor
        out.attack_target_type = target

        if attack_nodes_tensor.numel() == 0:
            self.last_attack_info = {
                "attack_name": "random",
                "target_type": target,
                "num_adv_nodes": 0,
                **select_info,
                "edge_type_to_perturb": edge_type,
                "edge_budget": 0,
                "num_added_edges": 0,
                "num_removed_edges": 0,
            }
            return out

        store = out[edge_type]
        edge_index = store.edge_index
        num_nodes = int(out[target].num_nodes)
        edge_budget, degree_sum = self._compute_edge_budget(edge_index, attack_nodes)
        if edge_budget <= 0 or edge_index.numel() == 0:
            self.last_attack_info = {
                "attack_name": "random",
                "target_type": target,
                "num_adv_nodes": int(attack_nodes_tensor.numel()),
                **select_info,
                "edge_type_to_perturb": edge_type,
                "control_nodes_degree_sum": int(degree_sum),
                "edge_budget": int(edge_budget),
                "num_added_edges": 0,
                "num_removed_edges": 0,
            }
            return out

        src = edge_index[0]
        dst = edge_index[1]
        incident_mask = torch.isin(src, attack_nodes_tensor.to(src.device)) | torch.isin(dst, attack_nodes_tensor.to(dst.device))
        incident_idx = incident_mask.nonzero(as_tuple=False).view(-1)
        del_k = min(int(edge_budget), int(incident_idx.numel()))
        if del_k == 0:
            self.last_attack_info = {
                "attack_name": "random",
                "target_type": target,
                "num_adv_nodes": int(attack_nodes_tensor.numel()),
                **select_info,
                "edge_type_to_perturb": edge_type,
                "control_nodes_degree_sum": int(degree_sum),
                "edge_budget": int(edge_budget),
                "num_added_edges": 0,
                "num_removed_edges": 0,
            }
            return out

        perm = torch.randperm(incident_idx.numel(), device=edge_index.device, generator=generator)
        del_idx = incident_idx[perm[:del_k]]
        keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        keep_mask[del_idx] = False
        kept_edge_index = edge_index[:, keep_mask]
        kept_edge_attr = getattr(store, "edge_attr", None)
        if kept_edge_attr is not None:
            kept_edge_attr = kept_edge_attr[keep_mask]

        existing_hash = self._hash_edges(kept_edge_index[0], kept_edge_index[1], num_nodes)
        new_src, new_dst = self._sample_targeted_new_edges(
            num_nodes=num_nodes,
            attack_nodes=attack_nodes,
            k=del_k,
            existing_hash=existing_hash,
            device=edge_index.device,
            generator=generator,
        )
        if new_src.numel() > 0:
            added_edge_index = torch.stack([new_src, new_dst], dim=0)
            store.edge_index = torch.cat([kept_edge_index, added_edge_index], dim=1)
            if kept_edge_attr is not None:
                if kept_edge_attr.dim() >= 2:
                    feat_dim = kept_edge_attr.size(-1)
                    added_attr = torch.zeros((added_edge_index.size(1), feat_dim), device=edge_index.device, dtype=kept_edge_attr.dtype)
                else:
                    added_attr = torch.zeros((added_edge_index.size(1),), device=edge_index.device, dtype=kept_edge_attr.dtype)
                store.edge_attr = torch.cat([kept_edge_attr, added_attr], dim=0)
        else:
            store.edge_index = kept_edge_index
            if kept_edge_attr is not None:
                store.edge_attr = kept_edge_attr

        num_added_edges = int(new_src.numel())
        num_removed_edges = int(del_k)
        self.last_attack_info = {
            "attack_name": "random",
            "target_type": target,
            "num_adv_nodes": int(attack_nodes_tensor.numel()),
            **select_info,
            "edge_type_to_perturb": edge_type,
            "control_nodes_degree_sum": int(degree_sum),
            "edge_budget": int(edge_budget),
            "num_added_edges": num_added_edges,
            "num_removed_edges": num_removed_edges,
        }

        return out


class RoHeAttacker(Attacker):
    """
    RoHe-style evasion attacker.

    Core idea: connect attacked target nodes to high-degree hub nodes on one
    selected relation, which is the vulnerability highlighted by the RoHe paper.
    """

    attack_mode = "evasion"

    def __init__(
        self,
        dataset_module,
        perturb_ratio: float = 0.1,
        ctrl_nodes_size: int = 10,
        target_label: int = 1,
        only_attack_correctly_detected: bool = True,
        target_node_type: Optional[str] = None,
        edge_type_to_perturb: Optional[Tuple[str, str, str]] = None,
        victim_model: Optional[nn.Module] = None,
        victim_engine: Optional[Any] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(dataset_module)
        assert 0.0 <= perturb_ratio <= 1.0
        self.perturb_ratio = float(perturb_ratio)
        self.ctrl_nodes_size = int(ctrl_nodes_size)
        self.positive_label = int(target_label)
        self.only_attack_correctly_detected = bool(only_attack_correctly_detected)
        self.target_node_type = target_node_type
        self.edge_type_to_perturb = edge_type_to_perturb
        self.victim_model = victim_model
        self.victim_engine = victim_engine
        self.device = device
        self.seed = seed
        self.last_attack_info: Dict[str, Any] = {}

    def _resolve_target_type(self, data: HeteroData) -> str:
        if self.target_node_type:
            return self.target_node_type
        dm = self.dataset_module
        for attr in ["target", "target_node_type", "target_node"]:
            if hasattr(dm, attr):
                value = getattr(dm, attr)
                if isinstance(value, str) and value:
                    return value
        if hasattr(data, "node_types") and len(data.node_types) == 1:
            return data.node_types[0]
        raise ValueError("[RoHeAttacker] Cannot infer target node type.")

    def _predict_nodes_with_victim(
        self,
        data: HeteroData,
        target: str,
        node_indices: np.ndarray,
    ) -> Optional[np.ndarray]:
        if node_indices is None or len(node_indices) == 0:
            return np.array([], dtype=np.int64)
        node_indices = np.asarray(node_indices, dtype=np.int64)

        if self.victim_model is not None:
            try:
                self.victim_model.eval()
                with torch.no_grad():
                    try:
                        pred_raw = self.victim_model(data.x_dict, data.edge_index_dict)
                    except Exception:
                        pred_raw = self.victim_model(data)
                    if isinstance(pred_raw, dict):
                        pred_raw = pred_raw[target]
                    pred = pred_raw.argmax(dim=-1)
                return pred[node_indices].detach().cpu().numpy()
            except Exception:
                pass

        if self.victim_engine is not None and hasattr(self.victim_engine, "eval_on_graph"):
            pred_map: Dict[int, int] = {}
            for split_name in ("train", "val", "test"):
                try:
                    res = self.victim_engine.eval_on_graph(data, split=split_name, target_type=target)
                except Exception:
                    continue
                idx = res.get("node_idx")
                y_pred = res.get("y_pred")
                if idx is None or y_pred is None:
                    continue
                idx = idx.detach().to(torch.long).view(-1).cpu()
                y_pred = y_pred.detach().to(torch.long).view(-1).cpu()
                for n, p in zip(idx.tolist(), y_pred.tolist()):
                    pred_map[int(n)] = int(p)
            if pred_map:
                out = np.full((len(node_indices),), -1, dtype=np.int64)
                for i, n in enumerate(node_indices.tolist()):
                    if int(n) in pred_map:
                        out[i] = pred_map[int(n)]
                return out

        return None

    def _select_attack_nodes(self, data: HeteroData, target: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        y = data[target].y
        test_mask = getattr(data[target], "test_mask", None)
        if test_mask is None:
            raise ValueError("[RoHeAttacker] target node test_mask is required.")

        test_idx = test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
        y_test = y[test_mask].cpu().numpy()
        pos_idx = test_idx[y_test == self.positive_label]
        if len(pos_idx) == 0:
            raise ValueError("[RoHeAttacker] No positive test nodes found for attack.")

        candidates = pos_idx
        used_detected = False
        if self.only_attack_correctly_detected:
            pred_pos = self._predict_nodes_with_victim(data, target, pos_idx)
            if pred_pos is not None:
                detected_pos = pos_idx[pred_pos == self.positive_label]
                if len(detected_pos) > 0:
                    candidates = detected_pos
                    used_detected = True

        k = min(self.ctrl_nodes_size, len(candidates))
        selected = np.array(random.sample(list(candidates), k), dtype=np.int64) if k > 0 else np.array([], dtype=np.int64)
        return selected, {
            "num_test_nodes": int(len(test_idx)),
            "num_positive_test_nodes": int(len(pos_idx)),
            "selected_adv_nodes": int(len(selected)),
            "used_detected_positive_subset": bool(used_detected),
            "positive_label": int(self.positive_label),
        }

    def _resolve_edge_type(self, data: HeteroData, target: str) -> Tuple[str, str, str]:
        if self.edge_type_to_perturb is not None:
            if self.edge_type_to_perturb not in data.edge_types:
                raise ValueError(f"[RoHeAttacker] edge_type_to_perturb {self.edge_type_to_perturb} not in graph.")
            return self.edge_type_to_perturb

        preferred = [et for et in data.edge_types if et[0] == target and et[2] != target]
        if preferred:
            return preferred[0]
        fallback = [et for et in data.edge_types if et[0] == target or et[2] == target]
        if fallback:
            return fallback[0]
        raise ValueError("[RoHeAttacker] No edge type incident to target node type is available.")

    def _compute_edge_budget(self, edge_index: torch.Tensor, attack_nodes: np.ndarray, attack_on_src: bool) -> Tuple[int, int]:
        if attack_nodes is None or len(attack_nodes) == 0:
            return 0, 0
        target_side = edge_index[0] if attack_on_src else edge_index[1]
        deg_sum = int(np.isin(target_side.detach().cpu().numpy(), attack_nodes).sum())
        base = max(deg_sum, len(attack_nodes))
        edge_budget = int(np.floor(self.perturb_ratio * base))
        if edge_budget <= 0 and self.perturb_ratio > 0:
            edge_budget = 1
        return edge_budget, deg_sum

    def _rank_hubs(self, data: HeteroData, edge_type: Tuple[str, str, str], attack_on_src: bool) -> np.ndarray:
        edge_index = data[edge_type].edge_index
        hub_side = edge_index[1] if attack_on_src else edge_index[0]
        num_hubs = int(data[edge_type[2] if attack_on_src else edge_type[0]].num_nodes)
        if hub_side.numel() == 0:
            return np.arange(num_hubs, dtype=np.int64)
        deg = torch.bincount(hub_side.detach().to(torch.long).cpu(), minlength=num_hubs)
        return torch.argsort(deg, descending=True).cpu().numpy().astype(np.int64)

    def _build_new_edges(
        self,
        edge_index: torch.Tensor,
        attack_nodes: np.ndarray,
        ranked_hubs: np.ndarray,
        attack_on_src: bool,
        edge_budget: int,
        same_type: bool,
    ) -> torch.Tensor:
        if edge_budget <= 0 or len(attack_nodes) == 0 or len(ranked_hubs) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        existing = set(zip(edge_index[0].detach().cpu().tolist(), edge_index[1].detach().cpu().tolist()))
        new_edges = []
        budget_left = int(edge_budget)
        hub_list = ranked_hubs.tolist()

        for attack_node in attack_nodes.tolist():
            if budget_left <= 0:
                break
            for hub in hub_list:
                if same_type and int(hub) == int(attack_node):
                    continue
                if attack_on_src:
                    pair = (int(attack_node), int(hub))
                else:
                    pair = (int(hub), int(attack_node))
                if pair in existing:
                    continue
                existing.add(pair)
                new_edges.append(pair)
                budget_left -= 1
                if budget_left <= 0:
                    break

        if not new_edges:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return torch.tensor(new_edges, dtype=torch.long, device=edge_index.device).t().contiguous()

    def attack(self, data):
        if not isinstance(data, HeteroData):
            raise RuntimeError("RoHeAttacker expects HeteroData input.")

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        out = data.clone()
        target = self._resolve_target_type(out)
        edge_type = self._resolve_edge_type(out, target)
        attack_nodes, select_info = self._select_attack_nodes(out, target)

        attack_nodes_tensor = torch.as_tensor(attack_nodes, dtype=torch.long).view(-1).cpu()
        out.attack_target_nodes = attack_nodes_tensor
        out.attack_target_type = target

        if attack_nodes_tensor.numel() == 0:
            self.last_attack_info = {
                "attack_name": "rohe",
                "target_type": target,
                "num_adv_nodes": 0,
                **select_info,
                "edge_type_to_perturb": edge_type,
                "edge_budget": 0,
                "num_added_edges": 0,
                "hub_node_type": edge_type[2] if edge_type[0] == target else edge_type[0],
            }
            return out

        attack_on_src = edge_type[0] == target
        if not attack_on_src and edge_type[2] != target:
            raise ValueError("[RoHeAttacker] Selected edge type is not incident to target node type.")

        store = out[edge_type]
        edge_index = store.edge_index
        hub_type = edge_type[2] if attack_on_src else edge_type[0]
        num_hubs = int(out[hub_type].num_nodes)
        ranked_hubs = self._rank_hubs(out, edge_type, attack_on_src=attack_on_src)
        edge_budget, degree_sum = self._compute_edge_budget(edge_index, attack_nodes, attack_on_src=attack_on_src)
        same_type = edge_type[0] == edge_type[2]
        added_edge_index = self._build_new_edges(
            edge_index=edge_index,
            attack_nodes=attack_nodes,
            ranked_hubs=ranked_hubs,
            attack_on_src=attack_on_src,
            edge_budget=edge_budget,
            same_type=same_type,
        )

        if added_edge_index.numel() > 0:
            store.edge_index = torch.cat([edge_index, added_edge_index], dim=1)
            edge_attr = getattr(store, "edge_attr", None)
            if edge_attr is not None:
                if edge_attr.dim() >= 2:
                    added_attr = torch.zeros(
                        (added_edge_index.size(1), edge_attr.size(-1)),
                        dtype=edge_attr.dtype,
                        device=edge_attr.device,
                    )
                else:
                    added_attr = torch.zeros(
                        (added_edge_index.size(1),),
                        dtype=edge_attr.dtype,
                        device=edge_attr.device,
                    )
                store.edge_attr = torch.cat([edge_attr, added_attr], dim=0)

        self.last_attack_info = {
            "attack_name": "rohe",
            "target_type": target,
            "num_adv_nodes": int(attack_nodes_tensor.numel()),
            **select_info,
            "edge_type_to_perturb": edge_type,
            "hub_node_type": hub_type,
            "control_nodes_degree_sum": int(degree_sum),
            "edge_budget": int(edge_budget),
            "num_edges_before": int(edge_index.size(1)),
            "num_edges_after": int(store.edge_index.size(1)),
            "num_added_edges": int(added_edge_index.size(1)),
        }
        return out

class Metattacker(Attacker):
    def __init__(
        self,
        dataset_module: AbstractDataModule,
        perturb_ratio,
        attack_varient='Meta-Self',
        re_trainings=5,
        device=0,
        train_iters=200,
        surrogate_engine: Optional[MetaSurrogateEngine] = None,
    ):
        super().__init__(dataset_module)
        self.GPU_ID = device
        self.share_perturbations = perturb_ratio
        self.train_iters = train_iters
        self.re_trainings = re_trainings
        self.dtype = tf.float32
        self.ENFORCE_LL_CONSTRAINT = False
        self.attack_variant = attack_varient
        self.surrogate_engine = surrogate_engine or MetaSurrogateEngine(
            gpu_id=self.GPU_ID, hidden_sizes=[16], with_relu=False, train_iters=self.train_iters
        )

    def attack(self, data):
        """
        Run MetaAttack on a single (sub)graph and return the attacked graph in PyG format.

        Notes:
        - MetaAttack implementation here only supports homogeneous graphs.
        - If `data` is a `HeteroData`, we first convert it to homogeneous via `to_homogeneous()`,
          run the attack, and then restore it back to `HeteroData` via `to_heterogeneous()`.
        - The returned object will be `Data` for homogeneous inputs, and `HeteroData` for heterogeneous inputs.
        """
        is_hetero = isinstance(data, HeteroData)

        # 1) Convert to homogeneous if needed (MetaAttack only supports homogeneous graphs)
        if is_hetero:
            homo = data.to_homogeneous()
        else:
            warnings.warn("Metattacker works best with HeteroData. Proceeding with Data (homogeneous graph).")
            homo = data

        # 2) Build matrices for MetaAttack
        _A_obs, _X_obs, _z_obs, _Z_obs, _N, _K, data_mask = graphbuilder.pyg2matrix(homo)
        split_train, split_val, split_unlabeled = graphbuilder.split_dataset_by_pyg_mask(data_mask)

        # 3) Run meta attack (only structure perturbation here)
        modified_adjacency = self._run_meta_attack_on_single_graph(
            _A_obs, _X_obs, _Z_obs, _N, _K, split_train, split_unlabeled, attack_variant=self.attack_variant
        )

        # 4) Convert attacked adjacency back to edge_index
        if sp.issparse(modified_adjacency):
            modified_adjacency = modified_adjacency.tocoo()
            rows = modified_adjacency.row
            cols = modified_adjacency.col
        else:
            rows, cols = np.nonzero(modified_adjacency)

        # Remove self-loops (optional but usually desired for GCN-style datasets)
        keep = rows != cols
        rows = rows[keep]
        cols = cols[keep]

        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=homo.edge_index.device)

        # 5) Create attacked homogeneous Data (preserve features/labels/masks)
        attacked_homo = Data()
        # Preserve node-level tensors
        if hasattr(homo, "x") and homo.x is not None:
            attacked_homo.x = homo.x
        if hasattr(homo, "y") and homo.y is not None:
            attacked_homo.y = homo.y
        for mask_name in ["train_mask", "val_mask", "test_mask"]:
            if hasattr(homo, mask_name):
                setattr(attacked_homo, mask_name, getattr(homo, mask_name))

        attacked_homo.edge_index = edge_index
        attacked_homo.num_nodes = homo.num_nodes

        # Preserve any additional attributes that are safe and useful (e.g., node_type/edge_type for hetero restore)
        for attr in ["node_type", "edge_type"]:
            if hasattr(homo, attr):
                setattr(attacked_homo, attr, getattr(homo, attr))

        # Preserve to_homogeneous metadata for restoring hetero graphs
        for attr in ["_node_type_names", "_edge_type_names"]:
            if hasattr(homo, attr):
                setattr(attacked_homo, attr, getattr(homo, attr))

        # If we changed edges, edge_type needs to match number of edges for hetero restoration.
        # For the common case where the original hetero graph is effectively single-relation,
        # we assign all edges to relation 0.
        if is_hetero and hasattr(attacked_homo, "edge_type"):
            if attacked_homo.edge_type is None or attacked_homo.edge_type.numel() != attacked_homo.edge_index.size(1):
                attacked_homo.edge_type = torch.zeros(attacked_homo.edge_index.size(1), dtype=torch.long)

        # 6) Restore to hetero if needed
        if is_hetero:
            # Requirement: for multi-relation hetero graphs, any add/remove of an edge in the
            # homogeneous graph should be mirrored across *all* relations that share the same
            # (src_node_type, dst_node_type) pair.
            # Build global->local node index mapping for each node type
            node_type = attacked_homo.node_type  # [num_nodes] long
            node_types, edge_types = data.metadata()

            # local_index[t] gives local node index within its type for every global node
            local_index = torch.empty(attacked_homo.num_nodes, dtype=torch.long, device=attacked_homo.edge_index.device)
            for tid, ntype in enumerate(node_types):
                idx = (node_type == tid).nonzero(as_tuple=False).view(-1)
                local_index[idx] = torch.arange(idx.numel(), device=local_index.device, dtype=torch.long)

            # Group attacked edges by (src_tid, dst_tid)
            src_g, dst_g = attacked_homo.edge_index[0], attacked_homo.edge_index[1]
            src_tid = node_type[src_g]
            dst_tid = node_type[dst_g]

            # We'll construct a fresh hetero graph by cloning original data (keeps node attrs)
            attacked_hetero = data.clone()

            # For each type-pair, compute the shared edge_index (local indices)
            # then assign it to every relation with that type-pair.
            for (src_ntype, rel, dst_ntype) in edge_types:
                s_tid = node_types.index(src_ntype)
                d_tid = node_types.index(dst_ntype)

                m = (src_tid == s_tid) & (dst_tid == d_tid)
                e_src_local = local_index[src_g[m]]
                e_dst_local = local_index[dst_g[m]]
                shared_edge_index = torch.stack([e_src_local, e_dst_local], dim=0)

                attacked_hetero[(src_ntype, rel, dst_ntype)].edge_index = shared_edge_index

            return attacked_hetero
        else:
            warnings.warn("Returning attacked graph as Data (homogeneous graph).")

        return attacked_homo

    def poison(self):
        """
        向测试集投毒，并测试投毒后的测试结果
        """
        loader = self.dataset_module.hetero_datasets['test']
        all_accuracies_clean = []
        all_accuracies_atk = []
        pbar = tqdm(loader, desc="[Overall Progress]", unit="subgraph")
    
        for data in pbar:
            if isinstance(data, HeteroData):
                data = data.to_homogeneous()
            _A_obs, _X_obs, _z_obs, _Z_obs, _N, _K, data_mask = graphbuilder.pyg2matrix(data)

            split_train, split_val, split_unlabeled = graphbuilder.split_dataset_by_pyg_mask(data_mask)

            print(f"--- Debugging Subgraph ---")
            print(f"Nodes: {_N}, Edges: {_A_obs.sum()/2}")
            print(f"Perturbations: {self.share_perturbations * (_A_obs.sum()//2)}")
            print(f"Train nodes: {len(split_train)}, Test nodes: {len(split_unlabeled)}")

            # 检查是否存在越界索引
            if len(split_unlabeled) > 0 and split_unlabeled.max() >= _N:
                print(f"CRITICAL: Test index {split_unlabeled.max()} out of bounds for graph size {_N}!")

            pbar.set_description(f"Attacking subgraph (Nodes: {_N})")
            modified_adjacency = self._run_meta_attack_on_single_graph(
                _A_obs, _X_obs, _Z_obs, _N, _K, split_train, split_unlabeled, attack_variant=self.attack_variant
            )
            pbar.set_description(f"Evaluating accuracy")
            accuracies_clean, accuracies_atk = self._evaluate_accuracy_on_single_graph(
                _A_obs, modified_adjacency, _X_obs, _Z_obs, _z_obs, split_train, split_unlabeled
            )

            all_accuracies_clean.append(np.mean(accuracies_clean))
            all_accuracies_atk.append(np.mean(accuracies_atk))

            curr_clean_avg = np.mean(accuracies_clean)
            curr_atk_avg = np.mean(accuracies_atk)
            drop = curr_clean_avg - curr_atk_avg
            pbar.set_postfix({
                'Clean_Acc': f'{curr_clean_avg:.4f}',
                'Atk_Acc': f'{curr_atk_avg:.4f}',
                'Drop': f'{drop:.4f}'
            })

        return all_accuracies_clean, all_accuracies_atk
    
    def _evaluate_accuracy_on_single_graph(self, _A_obs, modified_adjacency, _X_obs, _Z_obs, _z_obs, split_train, split_unlabeled):
        hidden_sizes = [16]
        
        gcn_before_attack = mtk.GCNSparse(sp.csr_matrix(_A_obs), _X_obs, _Z_obs, hidden_sizes, gpu_id=self.GPU_ID)
        gcn_before_attack.build(with_relu=True)
        accuracies_clean = []
        
        for _it in tqdm(range(self.re_trainings), desc="  └─ Clean Eval", leave=False):
            gcn_before_attack.train(split_train, initialize=True, display=False)
            logits = gcn_before_attack.logits.eval(session=gcn_before_attack.session)
            accuracy_clean = (logits.argmax(1) == _z_obs)[split_unlabeled].mean()
            accuracies_clean.append(accuracy_clean)

        gcn_after_attack = mtk.GCNSparse(sp.csr_matrix(modified_adjacency), _X_obs, _Z_obs, hidden_sizes, gpu_id=self.GPU_ID)
        gcn_after_attack.build(with_relu=True)
        accuracies_atk = []
        
        for _it in tqdm(range(self.re_trainings), desc="  └─ Attack Eval", leave=False):
            gcn_after_attack.train(split_train, initialize=True, display=False)
            logits = gcn_after_attack.logits.eval(session=gcn_after_attack.session)
            accuracy_atk = (logits.argmax(1) == _z_obs)[split_unlabeled].mean()
            accuracies_atk.append(accuracy_atk)

        return accuracies_clean, accuracies_atk
    
    def _run_meta_attack_on_single_graph(self, _A_obs, _X_obs, _Z_obs, _N, _K, split_train, split_unlabeled, attack_variant):
        """
        执行MetaAttack攻击逻辑
        :param _A_obs: 原始邻接矩阵
        :param _X_obs: 节点特征
        :param _Z_obs: one-hot标签
        :param _N: 节点数
        :param _K: 类别数
        :param split_train: 训练集索引
        :param split_unlabeled: 无标签集（val+test）索引
        :param attack_variant: 攻击变体
        :param share_perturbations: 扰动比例
        :return: modified_adjacency（扰动后的邻接矩阵）
        """
        # 1. 初始化并训练代理GCN
        hidden_sizes = [16]
        self.surrogate_engine.fit_matrices(_A_obs, _X_obs, _Z_obs, split_train)

        # 2. 自训练标签预测
        surrogate_logits = self.surrogate_engine.logits()
        labels_self_training = np.eye(_K)[surrogate_logits.argmax(1)]
        labels_self_training[split_train] = _Z_obs[split_train]

        # 3. 攻击参数配置
        approximate_meta_gradient = attack_variant.startswith("A-")
        lambda_ = 1.0 if "Train" in attack_variant else (0.5 if "Both" in attack_variant else 0.0)
        idx_attack = split_train if "Train" in attack_variant else (
            np.union1d(split_train, split_unlabeled) if "Both" in attack_variant else split_unlabeled
        )
        perturbations = int(self.share_perturbations * (_A_obs.sum() // 2))

        # 4. 初始化攻击器
        if approximate_meta_gradient:
            gcn_attack = mtk.GNNMetaApprox(
                _A_obs, _X_obs, labels_self_training, hidden_sizes,
                gpu_id=self.GPU_ID, _lambda=lambda_, train_iters=self.train_iters, dtype=self.dtype
            )
        else:
            gcn_attack = mtk.GNNMeta(
                _A_obs, _X_obs.astype("float32"), labels_self_training, hidden_sizes,
                gpu_id=self.GPU_ID, attack_features=False, train_iters=self.train_iters, dtype=self.dtype
            )

        # 5. 执行攻击
        gcn_attack.build()
        gcn_attack.make_loss(ll_constraint=self.ENFORCE_LL_CONSTRAINT)
        if approximate_meta_gradient:
            gcn_attack.attack(perturbations, split_train, split_unlabeled, idx_attack)
        else:
            gcn_attack.attack(perturbations, split_train, idx_attack)

        # 6. 获取扰动后的邻接矩阵
        modified_adjacency = gcn_attack.modified_adjacency.eval(session=gcn_attack.session)
        return modified_adjacency


class MintaAttacker(Attacker):
    """
    attack_mode = "evasion"
    MintA: Multi-Instance adversarial attack (from MintA demo notebooks).

    Tailored to DNS-style hetero graphs, but will auto-infer the target node type
    and same-type relations when possible (to better fit current project graphs).
    relations like:
      - ('domain_node', 'apex', 'domain_node')
      - ('domain_node', 'similar', 'domain_node')

    It:
      1) samples adversarial nodes from malicious test nodes,
      2) trains a surrogate GCN on a subset of training nodes,
      3) perturbs features + apex edges among adversarial nodes,
      4) returns an attacked HeteroData.
    """

    def __init__(
        self,
        dataset_module: AbstractDataModule,
        perturb_ratio: float = 0.1,
        ctrl_nodes_size: int = 10,
        target_label: int = 1,
        only_attack_correctly_detected: bool = True,
        surrogate_train_size: int = 4000,
        surrogate_hidden: int = 64,
        surrogate_epochs: int = 50,
        surrogate_lr: float = 0.01,
        enable_feature_perturb: bool = False,
        target_node_type: Optional[str] = None,
        edge_types_for_A: Optional[list] = None,
        edge_type_to_perturb: Optional[Tuple[str, str, str]] = None,
        victim_model: Optional[nn.Module] = None,
        victim_engine: Optional[Any] = None,
        surrogate_engine: Optional[MintaSurrogateEngine] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        surrogate_early_stop_patience: Optional[int] = None,
        surrogate_early_stop_min_delta: float = 0.0,
    ):
        super().__init__(dataset_module)
        self.perturb_ratio = float(perturb_ratio)
        self.ctrl_nodes_size = int(ctrl_nodes_size)
        self.positive_label = int(target_label)
        self.only_attack_correctly_detected = bool(only_attack_correctly_detected)
        self.surrogate_train_size = int(surrogate_train_size)
        self.surrogate_hidden = int(surrogate_hidden)
        self.surrogate_epochs = int(surrogate_epochs)
        self.surrogate_lr = float(surrogate_lr)
        self.enable_feature_perturb = bool(enable_feature_perturb)
        self.target_node_type = target_node_type
        self.edge_types_for_A = edge_types_for_A
        self.edge_type_to_perturb = edge_type_to_perturb
        self.victim_model = victim_model
        self.victim_engine = victim_engine
        self.surrogate_engine = surrogate_engine
        self.seed = seed
        self.device = device
        self.surrogate_early_stop_patience = surrogate_early_stop_patience
        self.surrogate_early_stop_min_delta = float(surrogate_early_stop_min_delta)
        self.last_attack_info: Dict[str, Any] = {}

    @staticmethod
    def _largest_indices(ary: np.ndarray, n: int):
        flat = ary.flatten()
        if n <= 0:
            return (np.array([], dtype=int), np.array([], dtype=int))
        n = min(n, flat.size)
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    @staticmethod
    def _find_my_soln(W: np.ndarray) -> np.ndarray:
        ATA = np.dot(W, W.T)
        w, v = np.linalg.eig(ATA)
        return v[:, 0]

    @staticmethod
    def _do_perturb_adj(a: np.ndarray, m: np.ndarray) -> np.ndarray:
        a_bol = np.array(a, dtype=bool)
        m_bol = np.array(m, dtype=bool)
        a2 = np.logical_xor(a_bol, m_bol)
        return a2.astype(float)

    @staticmethod
    def _a_to_edge_index(A: np.ndarray) -> np.ndarray:
        rows, cols = np.nonzero(A)
        return np.vstack([rows, cols])

    def _resolve_target_type(self, data: HeteroData) -> str:
        if self.target_node_type:
            return self.target_node_type
        dm = self.dataset_module
        for attr in ["target", "target_node_type", "target_node"]:
            if hasattr(dm, attr):
                v = getattr(dm, attr)
                if isinstance(v, str) and v:
                    return v
        if self.victim_model is not None and hasattr(self.victim_model, "target_node"):
            v = getattr(self.victim_model, "target_node")
            if isinstance(v, str) and v:
                return v
        if self.victim_engine is not None and hasattr(self.victim_engine, "target_node"):
            v = getattr(self.victim_engine, "target_node")
            if isinstance(v, str) and v:
                return v
        if hasattr(data, "node_types") and len(data.node_types) == 1:
            return data.node_types[0]
        raise ValueError("[MintaAttacker] Cannot infer target node type. Please set target_node_type.")

    def _predict_nodes_with_victim(self, data: HeteroData, target: str, node_indices: np.ndarray) -> Optional[np.ndarray]:
        """
        Return predicted classes for target-node indices using victim_model or victim_engine.
        """
        if node_indices is None or len(node_indices) == 0:
            return np.array([], dtype=np.int64)
        node_indices = np.asarray(node_indices, dtype=np.int64)

        if self.victim_model is not None:
            try:
                self.victim_model.eval()
                with torch.no_grad():
                    try:
                        pred_raw = self.victim_model(data.x_dict, data.edge_index_dict)
                    except Exception:
                        pred_raw = self.victim_model(data)
                    if isinstance(pred_raw, dict):
                        if target in pred_raw:
                            pred_raw = pred_raw[target]
                        elif len(pred_raw) == 1:
                            pred_raw = next(iter(pred_raw.values()))
                        else:
                            raise TypeError(
                                "[MintaAttacker] victim_model returned dict logits without target key."
                            )
                    pred = pred_raw.argmax(dim=-1)
                return pred[node_indices].detach().cpu().numpy()
            except Exception:
                pass

        if self.victim_engine is not None and hasattr(self.victim_engine, "eval_on_graph"):
            pred_map: Dict[int, int] = {}
            for split_name in ("train", "val", "test"):
                try:
                    res = self.victim_engine.eval_on_graph(data, split=split_name, target_type=target)
                except Exception:
                    continue
                idx = res.get("node_idx")
                y_pred = res.get("y_pred")
                if idx is None or y_pred is None:
                    continue
                idx = idx.detach().to(torch.long).view(-1).cpu()
                y_pred = y_pred.detach().to(torch.long).view(-1).cpu()
                for n, p in zip(idx.tolist(), y_pred.tolist()):
                    pred_map[int(n)] = int(p)
            if len(pred_map) > 0:
                out = np.full((len(node_indices),), -1, dtype=np.int64)
                ok = 0
                for i, n in enumerate(node_indices.tolist()):
                    if int(n) in pred_map:
                        out[i] = pred_map[int(n)]
                        ok += 1
                if ok > 0:
                    return out

        return None

    def _resolve_edge_types(self, data: HeteroData, target: str) -> Tuple[list, Tuple[str, str, str]]:
        edge_types_for_A = self.edge_types_for_A
        edge_type_to_perturb = self.edge_type_to_perturb

        if edge_types_for_A is None:
            edge_types_for_A = [et for et in data.edge_types if et[0] == target and et[2] == target]
        if not edge_types_for_A:
            edge_types_for_A = [data.edge_types[0]]

        if edge_type_to_perturb is None:
            edge_type_to_perturb = edge_types_for_A[0]

        return edge_types_for_A, edge_type_to_perturb

    def _select_control_and_attack_nodes(
        self,
        data: HeteroData,
        target: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if target not in data.node_types:
            raise ValueError(f"[MintaAttacker] target_node_type '{target}' not in data.node_types.")

        y = data[target].y
        test_mask = data[target].test_mask if hasattr(data[target], "test_mask") else None
        if test_mask is None:
            raise ValueError("[MintaAttacker] target node test_mask is required to sample adversarial nodes.")

        test_idx = test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
        if y is None:
            raise ValueError("[MintaAttacker] target node labels are required for MintA adversarial-node sampling.")

        y_test = y[test_mask].cpu().numpy()
        mal_idx = test_idx[y_test == self.positive_label]
        if len(mal_idx) == 0:
            raise ValueError("[MintaAttacker] No malicious test nodes found for adversarial sampling.")

        k_control = min(self.ctrl_nodes_size, len(mal_idx))
        control_nodes = np.array(random.sample(list(mal_idx), k_control), dtype=int)

        pred_control = self._predict_nodes_with_victim(data, target, control_nodes)
        detected_attack_nodes = np.array([], dtype=int)
        if pred_control is not None:
            detected_attack_nodes = control_nodes[pred_control == self.positive_label]

        if self.only_attack_correctly_detected:
            attack_nodes = detected_attack_nodes
        else:
            attack_nodes = detected_attack_nodes if pred_control is not None else control_nodes

        info = {
            "num_test_nodes": int(len(test_idx)),
            "num_malicious_test_nodes": int(len(mal_idx)),
            "num_control_nodes": int(len(control_nodes)),
            "num_detected_attack_nodes": int(len(detected_attack_nodes)),
            "selected_adv_nodes": int(len(attack_nodes)),
            "skip_attack_due_to_no_detected_nodes": bool(len(attack_nodes) == 0),
            "positive_label": int(self.positive_label),
        }
        return control_nodes, attack_nodes, info

    def _compute_edge_budget(
        self,
        data: HeteroData,
        control_nodes: np.ndarray,
        edge_types_for_degree: list,
    ) -> Tuple[int, int]:
        """
        Edge budget = floor(perturb_ratio * sum of incident degrees of controllable nodes)
        on selected target-target edge types.
        """
        if control_nodes is None or len(control_nodes) == 0:
            return 0, 0
        control_nodes = np.asarray(control_nodes, dtype=np.int64)
        deg_sum = 0
        for et in edge_types_for_degree:
            if et not in data.edge_types:
                continue
            edge_index = data[et].edge_index
            src = edge_index[0].detach().cpu().numpy()
            dst = edge_index[1].detach().cpu().numpy()
            deg_sum += int(np.isin(src, control_nodes).sum() + np.isin(dst, control_nodes).sum())
        edge_budget = int(np.floor(self.perturb_ratio * deg_sum))
        edge_budget = max(0, edge_budget)
        return edge_budget, deg_sum

    def _extract_A_adv(self, data: HeteroData, adv_nodes: np.ndarray, edge_types_for_A: list, target: str) -> np.ndarray:
        n = len(adv_nodes)
        A = np.zeros((n, n), dtype=int)

        node_map = -np.ones((data[target].num_nodes,), dtype=int)
        node_map[adv_nodes] = np.arange(n)

        for et in edge_types_for_A:
            if et not in data.edge_types:
                continue
            edge_index = data[et].edge_index
            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
            src_m = node_map[src]
            dst_m = node_map[dst]
            mask = (src_m >= 0) & (dst_m >= 0)
            A[src_m[mask], dst_m[mask]] = 1

        return A

    def _feat_perturb(
        self,
        x: torch.Tensor,
        A_adv: np.ndarray,
        surrogate: nn.Module,
        val: int,
        adv_nodes_test: np.ndarray,
        preds: np.ndarray,
    ) -> torch.Tensor:
        x2 = x.clone()
        X = x[adv_nodes_test].cpu().numpy()
        X2 = copy.deepcopy(X)

        W1 = surrogate.gc1.data.cpu().numpy()
        F1 = self._find_my_soln(W1)

        for i in range(min(val, len(adv_nodes_test))):
            if preds[i] > 0:
                temp = A_adv[i, :]
                js = np.flatnonzero(temp)
                messages = np.dot(A_adv, X2)
                F2 = 0 * F1
                message_j = np.zeros_like(F1)
                for kk in np.arange(len(js)):
                    message_j = messages[js[kk], :]
                W2 = copy.deepcopy(W1)
                d_j = len(js)
                for kkk in np.arange(W2.shape[1]):
                    if d_j > 0:
                        W2[:, kkk] = W1[:, kkk] - 1 / d_j * message_j
                F2 = F2 + self._find_my_soln(W2)
                feat_up = (1 * F1 + 1 * F2) / 2
                X2[i, :] = X2[i, :] + feat_up[:]

        x2[adv_nodes_test, :] = torch.from_numpy(X2).to(x2.device)
        return x2

    def _adj_perturb_sim_apex(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        A_adv: np.ndarray,
        surrogate: nn.Module,
        edge_budget: int,
        adv_nodes_test: np.ndarray,
    ) -> torch.Tensor:
        if len(adv_nodes_test) == 0:
            return edge_index.clone()

        temp0 = edge_index.cpu().numpy()
        X = x[adv_nodes_test].cpu().numpy()
        W1 = surrogate.gc1.data.cpu().numpy()
        F1 = self._find_my_soln(W1)

        n = len(adv_nodes_test)
        simi_arr = np.zeros((n, n))
        messages = np.dot(A_adv, X)

        for i in range(n):
            tempx = A_adv[i, :]
            js = np.flatnonzero(tempx)
            for j in range(n):
                W2 = copy.deepcopy(W1)
                d_j = len(js)
                message_j = np.zeros_like(F1)
                for kk in np.arange(len(js)):
                    message_j = messages[js[kk], :]
                for kkk in np.arange(W2.shape[1]):
                    if d_j > 0:
                        W2[:, kkk] = W1[:, kkk] - 1 / d_j * message_j
                F2 = self._find_my_soln(W2)
                simi_arr[i, j] = np.linalg.norm(1 * F1 + 1 * F2)

        top_k = int(edge_budget)
        if top_k <= 0:
            return edge_index.clone()
        largest_idx = self._largest_indices(simi_arr, top_k)

        m = np.zeros((n, n))
        for idx in range(len(largest_idx[0])):
            m[largest_idx[0][idx], largest_idx[1][idx]] = 1

        A2 = self._do_perturb_adj(A_adv, m)
        aa = self._a_to_edge_index(A2)

        conv = np.zeros_like(aa)
        for k in range(aa.shape[1]):
            conv[0, k] = adv_nodes_test[aa[0, k]]
            conv[1, k] = adv_nodes_test[aa[1, k]]

        all_edges = temp0
        adv_edge_lox = np.nonzero(np.in1d(all_edges[0, :], adv_nodes_test))[0]
        non_adv_edges = np.delete(all_edges, adv_edge_lox, axis=1)
        temp2 = np.hstack((non_adv_edges, conv))

        return torch.tensor(temp2, dtype=torch.long, device=edge_index.device)

    def attack(self, data):
        if not isinstance(data, HeteroData):
            raise RuntimeError("MintaAttacker expects HeteroData input.")

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        target = self._resolve_target_type(data)
        edge_types_for_A, edge_type_to_perturb = self._resolve_edge_types(data, target)
        if edge_type_to_perturb not in data.edge_types:
            raise ValueError(f"[MintaAttacker] edge_type_to_perturb {edge_type_to_perturb} not in data.edge_types.")

        device = torch.device(self.device) if self.device is not None else data[target].x.device
        out = data.clone()

        # 1) Sample controllable nodes and attack-effective nodes
        control_nodes, adv_nodes_test, select_info = self._select_control_and_attack_nodes(out, target)

        if len(adv_nodes_test) == 0:
            # If no controllable node can be detected as positive by victim, skip attack on this graph.
            empty_nodes = torch.empty((0,), dtype=torch.long)
            out.attack_target_nodes = empty_nodes
            out.attack_target_type = target
            self.last_attack_info = {
                "attack_name": "minta",
                "target_type": target,
                "adv_nodes": empty_nodes,
                "num_adv_nodes": 0,
                **select_info,
                "enable_feature_perturb": bool(self.enable_feature_perturb),
                "edge_type_to_perturb": edge_type_to_perturb,
                "edge_perturb_ratio": float(self.perturb_ratio),
                "control_nodes_degree_sum": 0,
                "edge_budget": 0,
                "feature_budget": 0,
                "perturb_budget_val": 0,
                "num_edges_before": int(out[edge_type_to_perturb].edge_index.size(1)),
                "num_edges_after": int(out[edge_type_to_perturb].edge_index.size(1)),
                "num_added_edges": 0,
                "num_removed_edges": 0,
                "feature_delta_l1": 0.0,
                "skipped": True,
            }
            return out

        # 2) Build A_adv for adversarial nodes
        A_adv = self._extract_A_adv(out, adv_nodes_test, edge_types_for_A, target)

        # 3) Train surrogate on a subset of training nodes
        train_mask = out[target].train_mask
        labels = out[target].y
        labeled_mask = labels < 2 if labels is not None else train_mask
        train_idx = (train_mask & labeled_mask).nonzero(as_tuple=False).view(-1).cpu().numpy()
        if len(train_idx) == 0:
            raise ValueError("[MintaAttacker] No labeled training nodes available for surrogate training.")

        k = min(self.surrogate_train_size, len(train_idx))
        adv_nodes_train = train_idx[:k]

        A_sur = self._extract_A_adv(out, adv_nodes_train, edge_types_for_A, target)
        features_sur = out[target].x[adv_nodes_train].float()

        pred_train = self._predict_nodes_with_victim(out, target, adv_nodes_train)
        if pred_train is not None:
            mask_known = pred_train >= 0
            if mask_known.all():
                labels_sur = torch.from_numpy(pred_train).to(out[target].y.device, dtype=torch.long)
            else:
                labels_sur = out[target].y[adv_nodes_train].clone()
                labels_sur[torch.from_numpy(mask_known).to(labels_sur.device)] = torch.from_numpy(
                    pred_train[mask_known]
                ).to(labels_sur.device, dtype=torch.long)
        else:
            labels_sur = out[target].y[adv_nodes_train]

        num_classes = int(labels_sur.max().item()) + 1 if labels_sur.numel() > 0 else 2
        if self.surrogate_engine is None:
            self.surrogate_engine = MintaSurrogateEngine(
                input_dim=int(features_sur.size(1)),
                hidden_dim=self.surrogate_hidden,
                num_classes=max(2, num_classes),
                target_node=target,
                edge_types_for_adj=edge_types_for_A,
                lr=self.surrogate_lr,
                epochs=self.surrogate_epochs,
                device=str(device),
            )
        else:
            # Keep engine aligned with current target/relation selection.
            self.surrogate_engine.target_node = target
            self.surrogate_engine.edge_types_for_adj = edge_types_for_A
            self.surrogate_engine.to(str(device))

        self.surrogate_engine.fit_dense(
            features_sur,
            A_sur,
            labels_sur,
            epochs=self.surrogate_epochs,
            early_stop_patience=self.surrogate_early_stop_patience,
            early_stop_min_delta=self.surrogate_early_stop_min_delta,
        )
        surrogate = self.surrogate_engine.model

        # 4) Predictions for adversarial nodes (victim if available, else ground-truth)
        pred_adv = self._predict_nodes_with_victim(out, target, adv_nodes_test)
        if pred_adv is not None:
            mask_known = pred_adv >= 0
            if mask_known.all():
                preds_adv = pred_adv
            else:
                preds_adv = out[target].y[adv_nodes_test].cpu().numpy()
                preds_adv[mask_known] = pred_adv[mask_known]
        else:
            preds_adv = out[target].y[adv_nodes_test].cpu().numpy()

        edge_index = out[edge_type_to_perturb].edge_index
        edge_budget, control_deg_sum = self._compute_edge_budget(out, control_nodes, edge_types_for_A)

        # 5) Feature perturbation budget is still tied to attack-effective nodes.
        val = int(np.floor(self.perturb_ratio * len(adv_nodes_test)))

        # 6) Optional feature perturbation (can be disabled for structure-only MintA)
        x = out[target].x
        if self.enable_feature_perturb and val > 0:
            x2 = self._feat_perturb(x, A_adv, surrogate, val, adv_nodes_test, preds_adv)
            out[target].x = x2

        # 7) Adjacency perturbation (apex relation by default)
        new_edge_index = self._adj_perturb_sim_apex(edge_index, x, A_adv, surrogate, edge_budget, adv_nodes_test)
        out[edge_type_to_perturb].edge_index = new_edge_index

        # Attack diagnostics (helps validate that perturbations are actually applied).
        num_target_nodes = int(out[target].num_nodes)
        before_hash = edge_index[0].to(torch.long) * num_target_nodes + edge_index[1].to(torch.long)
        after_hash = new_edge_index[0].to(torch.long) * num_target_nodes + new_edge_index[1].to(torch.long)
        before_set = set(before_hash.detach().cpu().tolist())
        after_set = set(after_hash.detach().cpu().tolist())
        num_added_edges = len(after_set - before_set)
        num_removed_edges = len(before_set - after_set)

        feature_delta_l1 = float("nan")
        if self.enable_feature_perturb:
            feature_delta_l1 = float((out[target].x - x).abs().sum().item())

        # Expose targeted nodes for evasion metrics (ASR/NFR-style evaluation in pipeline).
        adv_nodes_tensor = torch.as_tensor(adv_nodes_test, dtype=torch.long).view(-1).cpu()
        out.attack_target_nodes = adv_nodes_tensor
        out.attack_target_type = target
        self.last_attack_info = {
            "attack_name": "minta",
            "target_type": target,
            "adv_nodes": adv_nodes_tensor.clone(),
            "num_adv_nodes": int(adv_nodes_tensor.numel()),
            **select_info,
            "enable_feature_perturb": bool(self.enable_feature_perturb),
            "edge_perturb_ratio": float(self.perturb_ratio),
            "control_nodes_degree_sum": int(control_deg_sum),
            "edge_budget": int(edge_budget),
            "feature_budget": int(max(0, val)),
            "perturb_budget_val": int(max(0, val)),
            "edge_type_to_perturb": edge_type_to_perturb,
            "num_edges_before": int(edge_index.size(1)),
            "num_edges_after": int(new_edge_index.size(1)),
            "num_added_edges": int(num_added_edges),
            "num_removed_edges": int(num_removed_edges),
            "feature_delta_l1": feature_delta_l1,
            "skipped": False,
        }

        return out

# class nettack():
#     pass

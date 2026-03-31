import copy
import hashlib
import os
import sys
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from src.gnn_meta_attack.metattack import meta_gradient_attack as mtk
from src.utils import graphbuilder

_ROHE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RoHe")
if _ROHE_ROOT not in sys.path:
    sys.path.insert(0, _ROHE_ROOT)

try:
    import dgl
except ImportError:
    dgl = None

try:
    from HAN_RoHe.model import HAN as RoHeHAN
except ImportError:
    RoHeHAN = None


class ClassifierEngine(ABC):
    """
    Abstract engine interface for classifier/surrogate train-eval-reset lifecycle.
    """

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        pass

    @abstractmethod
    def to(self, device: str):
        pass

    @abstractmethod
    def snapshot_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def reset_from_snapshot(self, snapshot: Dict[str, Any], prefer_reset_parameters: bool = True):
        pass

    @abstractmethod
    def eval_on_graph(self, graph, split: str, target_type: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def train_oneloop(self, data):
        pass


class HeteroClassifierEngine(ClassifierEngine):
    """
    Engine proxy for hetero-graph node classifiers with (x_dict, edge_index_dict) forward.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Any,
        scheduler: Optional[Any],
        target_node: str,
        device: str,
    ):
        self._model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.target_node = target_node
        self.device = device

    @property
    def model(self) -> nn.Module:
        return self._model

    def to(self, device: str):
        if device is None:
            raise ValueError("Device must be specified.")
        self.device = device
        self._model.to(device)
        return self

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "model": copy.deepcopy(self._model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": copy.deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None,
        }

    def reset_from_snapshot(self, snapshot: Dict[str, Any], prefer_reset_parameters: bool = True):
        if prefer_reset_parameters and hasattr(self._model, "reset_parameters"):
            self._model.reset_parameters()
        else:
            self._model.load_state_dict(copy.deepcopy(snapshot["model"]))

        self.optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        if self.scheduler is not None and snapshot.get("scheduler") is not None:
            self.scheduler.load_state_dict(copy.deepcopy(snapshot["scheduler"]))

    @torch.no_grad()
    def eval_on_graph(self, graph, split: str, target_type: str) -> Dict[str, Any]:
        self._model.eval()
        device = next(self._model.parameters()).device

        g = graph.to(device)

        if hasattr(g, "x_dict") and hasattr(g, "edge_index_dict"):
            logits = self._model(g.x_dict, g.edge_index_dict)
            y = g[target_type].y
            mask = getattr(g[target_type], f"{split}_mask")
        else:
            warnings.warn("Graph has no x_dict/edge_index_dict; using homogeneous classifier path.")
            logits = self._model(g)
            y = g.y
            mask = getattr(g, f"{split}_mask")

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
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        else:
            correct = int((pred[idx] == y_true[idx]).sum().item())
            acc = correct / num
            flat = (y_true[idx] * num_classes + pred[idx]).to(torch.long)
            confusion = torch.bincount(flat, minlength=num_classes * num_classes).view(num_classes, num_classes)

        return {
            "acc": acc,
            "num_eval": num,
            "num_correct": correct,
            "num_classes": num_classes,
            "confusion": confusion,
            "y_true": y_true[idx].detach().cpu(),
            "y_pred": pred[idx].detach().cpu(),
            "logits": logits[idx].detach().cpu(),
            "node_idx": idx.detach().cpu(),
        }

    def train_oneloop(self, data):
        target = self.target_node

        self._model.train()
        self.optimizer.zero_grad()

        logits = self._model(data.x_dict, data.edge_index_dict)
        labels = data[target].y
        train_mask = data[target].train_mask
        val_mask = data[target].val_mask

        loss = self.criterion.compute(logits[train_mask], labels[train_mask])
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self._model.eval()
            output = self._model(data.x_dict, data.edge_index_dict)
            val_loss = self.criterion.compute(output[val_mask], labels[val_mask])

        # Keep previous behavior (second optimizer step) for backward compatibility.
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(val_loss)

        return loss.item(), val_loss.item()


class _MintaSurrogateNet(nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int):
        super().__init__()
        # Keep same parameter names/shapes as MintA-style implementation.
        self.gc1 = nn.Parameter(torch.empty(nfeat, nhid))
        self.gc2 = nn.Parameter(torch.empty(nhid, nclass))
        nn.init.xavier_uniform_(self.gc1)
        nn.init.xavier_uniform_(self.gc2)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = adj_norm @ x
        h = h @ self.gc1
        h = F.relu(h)
        h = adj_norm @ h
        out = h @ self.gc2
        return out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gc1)
        nn.init.xavier_uniform_(self.gc2)


class MintaSurrogateEngine(ClassifierEngine):
    """
    Engine proxy for MintA-style surrogate GCN.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        target_node: str,
        edge_types_for_adj: Optional[list] = None,
        lr: float = 0.01,
        epochs: int = 50,
        device: str = "cpu",
    ):
        self._model = _MintaSurrogateNet(input_dim, hidden_dim, num_classes)
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = None
        self.target_node = target_node
        self.edge_types_for_adj = edge_types_for_adj
        self.epochs = int(epochs)
        self.device = device
        self.to(device)

    @property
    def model(self) -> nn.Module:
        return self._model

    def to(self, device: str):
        if device is None:
            raise ValueError("Device must be specified.")
        self.device = device
        self._model.to(device)
        return self

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "model": copy.deepcopy(self._model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
        }

    def reset_from_snapshot(self, snapshot: Dict[str, Any], prefer_reset_parameters: bool = True):
        if prefer_reset_parameters and hasattr(self._model, "reset_parameters"):
            self._model.reset_parameters()
        else:
            self._model.load_state_dict(copy.deepcopy(snapshot["model"]))
        self.optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))

    def _resolve_edge_types(self, graph, target_type: str) -> list:
        if self.edge_types_for_adj is not None:
            return [et for et in self.edge_types_for_adj if et in graph.edge_types]
        return [et for et in graph.edge_types if et[0] == target_type and et[2] == target_type]

    def _build_dense_adj(self, graph, target_type: str) -> torch.Tensor:
        n = int(graph[target_type].num_nodes)
        adj = torch.zeros((n, n), dtype=torch.float, device=self.device)
        edge_types = self._resolve_edge_types(graph, target_type)
        for et in edge_types:
            edge_index = graph[et].edge_index.to(self.device)
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
            src = src[valid]
            dst = dst[valid]
            adj[src, dst] = 1.0
        return adj

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        A = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg = A.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        return deg_inv_sqrt.view(-1, 1) * A * deg_inv_sqrt.view(1, -1)

    @staticmethod
    def _to_label_index(y: torch.Tensor) -> torch.Tensor:
        if y.dim() > 1 and y.size(-1) > 1:
            return y.argmax(dim=-1).to(torch.long)
        return y.view(-1).to(torch.long)

    def fit_dense(
        self,
        features: torch.Tensor,
        adj_dense,
        labels: torch.Tensor,
        epochs: Optional[int] = None,
        early_stop_patience: Optional[int] = None,
        early_stop_min_delta: float = 0.0,
    ):
        if isinstance(adj_dense, np.ndarray):
            adj_dense = torch.tensor(adj_dense, dtype=torch.float, device=self.device)
        elif not torch.is_tensor(adj_dense):
            adj_dense = torch.tensor(np.asarray(adj_dense), dtype=torch.float, device=self.device)
        else:
            adj_dense = adj_dense.to(self.device, dtype=torch.float)

        x = features.to(self.device, dtype=torch.float)
        y = self._to_label_index(labels.to(self.device))
        adj_norm = self._normalize_adj(adj_dense)

        steps = self.epochs if epochs is None else int(epochs)
        best_loss = float("inf")
        bad_epochs = 0
        for _ in range(steps):
            self._model.train()
            self.optimizer.zero_grad()
            out = self._model(x, adj_norm)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            if early_stop_patience is not None and int(early_stop_patience) > 0:
                loss_v = float(loss.detach().item())
                if (best_loss - loss_v) > float(early_stop_min_delta):
                    best_loss = loss_v
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(early_stop_patience):
                        break

    @torch.no_grad()
    def predict_dense(self, features: torch.Tensor, adj_dense) -> torch.Tensor:
        if isinstance(adj_dense, np.ndarray):
            adj_dense = torch.tensor(adj_dense, dtype=torch.float, device=self.device)
        elif not torch.is_tensor(adj_dense):
            adj_dense = torch.tensor(np.asarray(adj_dense), dtype=torch.float, device=self.device)
        else:
            adj_dense = adj_dense.to(self.device, dtype=torch.float)

        x = features.to(self.device, dtype=torch.float)
        adj_norm = self._normalize_adj(adj_dense)
        self._model.eval()
        return self._model(x, adj_norm)

    @torch.no_grad()
    def eval_on_graph(self, graph, split: str, target_type: str) -> Dict[str, Any]:
        g = graph.to(self.device)
        x = g[target_type].x.to(self.device, dtype=torch.float)
        y_true = self._to_label_index(g[target_type].y.to(self.device))
        mask = getattr(g[target_type], f"{split}_mask")

        adj = self._build_dense_adj(g, target_type)
        adj_norm = self._normalize_adj(adj)

        self._model.eval()
        logits = self._model(x, adj_norm)
        pred = logits.argmax(dim=-1)
        num_classes = int(logits.size(-1))

        idx = mask.nonzero(as_tuple=False).view(-1)
        num = int(idx.numel())
        if num == 0:
            correct = 0
            acc = float("nan")
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        else:
            correct = int((pred[idx] == y_true[idx]).sum().item())
            acc = correct / num
            flat = (y_true[idx] * num_classes + pred[idx]).to(torch.long)
            confusion = torch.bincount(flat, minlength=num_classes * num_classes).view(num_classes, num_classes)

        return {
            "acc": acc,
            "num_eval": num,
            "num_correct": correct,
            "num_classes": num_classes,
            "confusion": confusion.detach().cpu(),
            "y_true": y_true[idx].detach().cpu(),
            "y_pred": pred[idx].detach().cpu(),
            "logits": logits[idx].detach().cpu(),
            "node_idx": idx.detach().cpu(),
        }

    def train_oneloop(self, data):
        g = data.to(self.device)
        target = self.target_node

        x = g[target].x.to(self.device, dtype=torch.float)
        y = self._to_label_index(g[target].y.to(self.device))
        train_mask = g[target].train_mask
        val_mask = g[target].val_mask

        adj = self._build_dense_adj(g, target)
        adj_norm = self._normalize_adj(adj)

        self._model.train()
        self.optimizer.zero_grad()
        logits = self._model(x, adj_norm)
        loss = self.criterion(logits[train_mask], y[train_mask])
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self._model.eval()
            output = self._model(x, adj_norm)
            val_loss = self.criterion(output[val_mask], y[val_mask])

        return loss.item(), val_loss.item()


class RoHeClassifierEngine(ClassifierEngine):
    """
    Thin wrapper around the released HAN-RoHe model.

    The engine accepts PyG `HeteroData`, converts it to a DGL heterograph,
    constructs RoHe transition priors for the configured meta-paths, and then
    reuses the current surrogate pipeline train/eval lifecycle.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        target_node: str,
        meta_paths,
        hidden_size: int = 8,
        num_heads: Optional[list] = None,
        dropout: float = 0.6,
        lr: float = 0.005,
        weight_decay: float = 0.001,
        top_t: Any = 5,
        class_weight: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        self.target_node = target_node
        self.meta_paths = self._normalize_meta_paths(meta_paths)
        self.hidden_size = int(hidden_size)
        self.num_heads = list(num_heads) if num_heads is not None else [8]
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.top_t = top_t
        self.class_weight = None if class_weight is None else class_weight.detach().clone().to(torch.float)
        self.device = device
        self._model = self._build_model(input_dim=input_dim, num_classes=num_classes)
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss(
            weight=None if self.class_weight is None else self.class_weight.to(device)
        )
        self.to(device)

    @property
    def model(self) -> nn.Module:
        return self._model

    @staticmethod
    def _check_rohe_backend():
        if dgl is None:
            raise ImportError(
                "RoHeClassifierEngine requires `dgl`, but it is not installed in the current environment."
            )
        if RoHeHAN is None:
            raise ImportError(
                "Failed to import released HAN-RoHe model from `src/RoHe/HAN_RoHe/model.py`."
            )

    @staticmethod
    def _normalize_meta_paths(meta_paths) -> list:
        normalized = []
        for mp in meta_paths:
            if len(mp) == 3 and isinstance(mp[0], str):
                normalized.append([tuple(mp)])
            else:
                normalized.append([tuple(step) for step in mp])
        if len(normalized) == 0:
            raise ValueError("RoHeClassifierEngine requires non-empty meta_paths.")
        return normalized

    @staticmethod
    def _to_label_index(y: torch.Tensor) -> torch.Tensor:
        if y.dim() > 1 and y.size(-1) > 1:
            return y.argmax(dim=-1).to(torch.long)
        return y.view(-1).to(torch.long)

    def _resolve_top_t(self, n_meta_paths: int, num_target_nodes: int) -> list:
        if isinstance(self.top_t, (list, tuple)):
            values = list(self.top_t)
        else:
            values = [self.top_t] * n_meta_paths
        if len(values) != n_meta_paths:
            raise ValueError(f"Expected {n_meta_paths} RoHe top_t values, got {len(values)}.")
        out = []
        max_t = max(1, int(num_target_nodes))
        for v in values:
            t = int(v)
            out.append(max(1, min(t, max_t)))
        return out

    def _build_model(self, input_dim: int, num_classes: int) -> nn.Module:
        self._check_rohe_backend()
        dummy_settings = [
            {"T": 1, "device": self.device, "TransM": sp.eye(1, format="csc")}
            for _ in self.meta_paths
        ]
        return RoHeHAN(
            meta_paths=self.meta_paths,
            in_size=int(input_dim),
            hidden_size=int(self.hidden_size),
            out_size=int(num_classes),
            num_heads=self.num_heads,
            dropout=float(self.dropout),
            settings=dummy_settings,
        )

    def to(self, device: str):
        if device is None:
            raise ValueError("Device must be specified.")
        self.device = device
        self._model.to(device)
        if self.class_weight is not None:
            self.class_weight = self.class_weight.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weight)
        return self

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "model": copy.deepcopy(self._model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
        }

    def reset_from_snapshot(self, snapshot: Dict[str, Any], prefer_reset_parameters: bool = True):
        self._model.load_state_dict(copy.deepcopy(snapshot["model"]))
        self.optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        self.to(self.device)

    def _build_base_adjs(self, graph: HeteroData) -> Dict[tuple, sp.csr_matrix]:
        base_adjs = {}
        for edge_type in graph.edge_types:
            src_type, _, dst_type = edge_type
            edge_index = graph[edge_type].edge_index.detach().cpu()
            num_src = int(graph[src_type].num_nodes)
            num_dst = int(graph[dst_type].num_nodes)
            if edge_index.numel() == 0:
                adj = sp.csr_matrix((num_src, num_dst), dtype=np.float32)
            else:
                src = edge_index[0].numpy()
                dst = edge_index[1].numpy()
                values = np.ones(src.shape[0], dtype=np.float32)
                adj = sp.csr_matrix((values, (src, dst)), shape=(num_src, num_dst), dtype=np.float32)
            base_adjs[tuple(edge_type)] = adj
        return base_adjs

    @staticmethod
    def _row_normalize_sparse(adj: sp.spmatrix) -> sp.csr_matrix:
        adj = adj.tocsr().astype(np.float32)
        deg = np.asarray(adj.sum(1)).reshape(-1)
        deg_inv = np.zeros_like(deg, dtype=np.float32)
        nonzero = deg > 0
        deg_inv[nonzero] = 1.0 / deg[nonzero]
        return sp.diags(deg_inv).dot(adj).tocsr()

    def _build_transition_matrices(self, graph: HeteroData) -> list:
        base_adjs = self._build_base_adjs(graph)
        normalized = {k: self._row_normalize_sparse(v) for k, v in base_adjs.items()}
        transitions = []
        for meta_path in self.meta_paths:
            trans = normalized[meta_path[0]]
            for step in meta_path[1:]:
                trans = trans.dot(normalized[step])
            transitions.append(sp.csc_matrix(trans))
        return transitions

    def _build_settings(self, graph: HeteroData) -> list:
        trans_list = self._build_transition_matrices(graph)
        top_t_values = self._resolve_top_t(len(trans_list), int(graph[self.target_node].num_nodes))
        return [
            {
                "T": int(top_t_values[i]),
                "device": self.device,
                "TransM": trans_list[i],
            }
            for i in range(len(trans_list))
        ]

    def _apply_settings(self, settings: list) -> None:
        for layer in self._model.layers:
            for i, gat_layer in enumerate(layer.gat_layers):
                gat_layer.settings = settings[i]

    def _build_dgl_graph(self, graph: HeteroData):
        self._check_rohe_backend()
        num_nodes_dict = {nt: int(graph[nt].num_nodes) for nt in graph.node_types}
        graph_data = {}
        for edge_type in graph.edge_types:
            edge_index = graph[edge_type].edge_index.detach().cpu()
            graph_data[tuple(edge_type)] = (
                edge_index[0].to(torch.long),
                edge_index[1].to(torch.long),
            )
        hg = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        return hg.to(self.device)

    def _forward_graph(self, graph: HeteroData) -> torch.Tensor:
        if not isinstance(graph, HeteroData):
            raise TypeError(f"RoHeClassifierEngine expects HeteroData, got {type(graph)}")
        if self.target_node not in graph.node_types:
            raise ValueError(f"Target node type '{self.target_node}' not found in graph.")

        settings = self._build_settings(graph)
        self._apply_settings(settings)
        hg = self._build_dgl_graph(graph)
        features = graph[self.target_node].x.to(self.device, dtype=torch.float)
        return self._model(hg, features)

    @torch.no_grad()
    def eval_on_graph(self, graph, split: str, target_type: str) -> Dict[str, Any]:
        if target_type != self.target_node:
            raise ValueError(
                f"RoHeClassifierEngine target mismatch: engine={self.target_node}, requested={target_type}."
            )

        g = graph
        y_true = self._to_label_index(g[target_type].y.to(self.device))
        mask = getattr(g[target_type], f"{split}_mask").to(self.device)

        self._model.eval()
        logits = self._forward_graph(g)
        pred = logits.argmax(dim=-1)
        num_classes = int(logits.size(-1))

        idx = mask.nonzero(as_tuple=False).view(-1)
        num = int(idx.numel())
        if num == 0:
            correct = 0
            acc = float("nan")
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        else:
            correct = int((pred[idx] == y_true[idx]).sum().item())
            acc = correct / num
            flat = (y_true[idx] * num_classes + pred[idx]).to(torch.long)
            confusion = torch.bincount(flat, minlength=num_classes * num_classes).view(num_classes, num_classes)

        return {
            "acc": acc,
            "num_eval": num,
            "num_correct": correct,
            "num_classes": num_classes,
            "confusion": confusion.detach().cpu(),
            "y_true": y_true[idx].detach().cpu(),
            "y_pred": pred[idx].detach().cpu(),
            "logits": logits[idx].detach().cpu(),
            "node_idx": idx.detach().cpu(),
        }

    def train_oneloop(self, data):
        if not isinstance(data, HeteroData):
            raise TypeError(f"RoHeClassifierEngine expects HeteroData, got {type(data)}")

        y = self._to_label_index(data[self.target_node].y.to(self.device))
        train_mask = data[self.target_node].train_mask.to(self.device)
        val_mask = data[self.target_node].val_mask.to(self.device)

        self._model.train()
        self.optimizer.zero_grad()
        logits = self._forward_graph(data)
        loss = self.criterion(logits[train_mask], y[train_mask])
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self._model.eval()
            out = self._forward_graph(data)
            val_loss = self.criterion(out[val_mask], y[val_mask])

        return float(loss.item()), float(val_loss.item())


class MetaSurrogateEngine(ClassifierEngine):
    """
    Engine proxy for MetaAttack surrogate GCNSparse (TensorFlow backend).
    """

    def __init__(
        self,
        gpu_id: Optional[int] = 0,
        hidden_sizes: Optional[list] = None,
        with_relu: bool = False,
        train_iters: int = 200,
        cache_eval: bool = True,
    ):
        self._model_proxy = nn.Identity()
        self.gpu_id = gpu_id
        self.hidden_sizes = hidden_sizes or [16]
        self.with_relu = with_relu
        self.train_iters = int(train_iters)
        self.cache_eval = bool(cache_eval)
        self._gcn = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    @property
    def model(self) -> nn.Module:
        return self._model_proxy

    def to(self, device: str):
        if isinstance(device, str):
            d = device.lower()
            if d.startswith("cuda"):
                if ":" in d:
                    try:
                        self.gpu_id = int(d.split(":")[1])
                    except ValueError:
                        self.gpu_id = 0
                else:
                    self.gpu_id = 0
            elif d == "cpu":
                self.gpu_id = None
        return self

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "hidden_sizes": list(self.hidden_sizes),
            "with_relu": bool(self.with_relu),
            "train_iters": int(self.train_iters),
        }

    def reset_from_snapshot(self, snapshot: Dict[str, Any], prefer_reset_parameters: bool = True):
        if snapshot:
            self.gpu_id = snapshot.get("gpu_id", self.gpu_id)
            self.hidden_sizes = snapshot.get("hidden_sizes", self.hidden_sizes)
            self.with_relu = snapshot.get("with_relu", self.with_relu)
            self.train_iters = snapshot.get("train_iters", self.train_iters)
        self._gcn = None
        self._cache = {}

    def fit_matrices(self, A_obs, X_obs, Z_obs, idx_train):
        if not sp.issparse(A_obs):
            A_obs = sp.csr_matrix(A_obs)
        X_obs = X_obs.astype("float32", copy=False)
        Z_obs = Z_obs.astype("float32", copy=False)
        idx_train = np.asarray(idx_train, dtype=np.int32)
        self._gcn = mtk.GCNSparse(A_obs, X_obs, Z_obs, self.hidden_sizes, gpu_id=self.gpu_id)
        self._gcn.build(with_relu=self.with_relu)
        self._gcn.train(idx_train, n_iters=self.train_iters, initialize=True, display=False)
        return self

    def logits(self) -> np.ndarray:
        if self._gcn is None:
            raise RuntimeError("MetaSurrogateEngine is not fitted. Call fit_matrices first.")
        return self._gcn.logits.eval(session=self._gcn.session)

    @staticmethod
    def _cross_entropy_from_logits(logits: np.ndarray, labels_idx: np.ndarray, idx: np.ndarray) -> float:
        if idx.size == 0:
            return float("nan")
        z = logits[idx]
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        probs = exp_z / (exp_z.sum(axis=1, keepdims=True) + 1e-12)
        p = probs[np.arange(idx.size), labels_idx[idx]]
        return float(-np.log(p + 1e-12).mean())

    @staticmethod
    def _select_idx(data_mask: Dict[str, np.ndarray], split: str) -> np.ndarray:
        if split not in data_mask:
            raise KeyError(f"Unknown split '{split}'. Available: {list(data_mask.keys())}.")
        return np.asarray(data_mask[split], dtype=np.int64)

    @staticmethod
    def _graph_signature(
        A_obs: sp.csr_matrix,
        X_obs: np.ndarray,
        z_obs: np.ndarray,
        data_mask: Dict[str, np.ndarray],
    ) -> str:
        A = A_obs.tocsr()
        X = np.asarray(X_obs, dtype=np.float32, order="C")
        z = np.asarray(z_obs, dtype=np.int64)

        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray(A.shape, dtype=np.int64).tobytes())
        h.update(np.asarray([A.nnz], dtype=np.int64).tobytes())
        h.update(np.asarray(A.indptr, dtype=np.int64).tobytes())
        h.update(np.asarray(A.indices, dtype=np.int64).tobytes())
        h.update(np.asarray(A.data, dtype=np.float32).tobytes())
        h.update(np.asarray(X.shape, dtype=np.int64).tobytes())
        h.update(X.tobytes())
        h.update(z.tobytes())
        for key in ("train", "val", "test"):
            if key in data_mask:
                h.update(np.asarray(data_mask[key], dtype=np.int64).tobytes())
        return h.hexdigest()

    @staticmethod
    def _to_homo_data(graph, target_type: Optional[str] = None) -> Data:
        if isinstance(graph, HeteroData):
            if target_type is not None and target_type not in graph.node_types:
                raise ValueError(f"target_type '{target_type}' is not in graph.node_types.")
            return graph.to_homogeneous()
        if isinstance(graph, Data):
            return graph
        raise RuntimeError("MetaSurrogateEngine expects Data or HeteroData.")

    def _fit_from_graph(self, graph, target_type: Optional[str]):
        homo = self._to_homo_data(graph, target_type=target_type).cpu()
        A_obs, X_obs, z_obs, Z_obs, _N, _K, data_mask = graphbuilder.pyg2matrix(homo)
        sig = self._graph_signature(A_obs, X_obs, z_obs, data_mask)

        self.fit_matrices(A_obs, X_obs, Z_obs, data_mask["train"])
        logits = self.logits()

        entry = {
            "logits": logits,
            "z_obs": np.asarray(z_obs, dtype=np.int64),
            "data_mask": data_mask,
            "num_classes": int(_K),
        }
        if self.cache_eval:
            self._cache[sig] = entry

        train_idx = self._select_idx(data_mask, "train")
        val_idx = self._select_idx(data_mask, "val")
        train_loss = self._cross_entropy_from_logits(logits, entry["z_obs"], train_idx)
        val_loss = self._cross_entropy_from_logits(logits, entry["z_obs"], val_idx)

        return entry, train_loss, val_loss

    def eval_on_graph(self, graph, split: str, target_type: str) -> Dict[str, Any]:
        homo = self._to_homo_data(graph, target_type=target_type).cpu()
        A_obs, X_obs, z_obs, Z_obs, _N, _K, data_mask = graphbuilder.pyg2matrix(homo)
        sig = self._graph_signature(A_obs, X_obs, z_obs, data_mask)

        entry = self._cache.get(sig, None)
        if entry is None:
            self.fit_matrices(A_obs, X_obs, Z_obs, data_mask["train"])
            logits = self.logits()
            entry = {
                "logits": logits,
                "z_obs": np.asarray(z_obs, dtype=np.int64),
                "data_mask": data_mask,
                "num_classes": int(_K),
            }
            if self.cache_eval:
                self._cache[sig] = entry

        logits = np.asarray(entry["logits"])
        y_true_all = np.asarray(entry["z_obs"], dtype=np.int64)
        idx = self._select_idx(entry["data_mask"], split)

        if logits.ndim != 2:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}.")
        y_pred_all = logits.argmax(axis=1).astype(np.int64)
        num_classes = max(int(entry["num_classes"]), int(logits.shape[1]))

        if idx.size == 0:
            correct = 0
            acc = float("nan")
            confusion_np = np.zeros((num_classes, num_classes), dtype=np.int64)
            y_true = np.empty((0,), dtype=np.int64)
            y_pred = np.empty((0,), dtype=np.int64)
            logits_sel = np.empty((0, num_classes), dtype=np.float32)
        else:
            y_true = y_true_all[idx]
            y_pred = y_pred_all[idx]
            logits_sel = logits[idx].astype(np.float32, copy=False)
            correct = int((y_true == y_pred).sum())
            acc = correct / float(idx.size)
            confusion_np = np.zeros((num_classes, num_classes), dtype=np.int64)
            np.add.at(confusion_np, (y_true, y_pred), 1)

        return {
            "acc": acc,
            "num_eval": int(idx.size),
            "num_correct": int(correct),
            "num_classes": int(num_classes),
            "confusion": torch.from_numpy(confusion_np).to(torch.long),
            "y_true": torch.from_numpy(y_true).to(torch.long),
            "y_pred": torch.from_numpy(y_pred).to(torch.long),
            "logits": torch.from_numpy(logits_sel).to(torch.float),
            "node_idx": torch.from_numpy(idx).to(torch.long),
        }

    def train_oneloop(self, data):
        _, train_loss, val_loss = self._fit_from_graph(data, target_type=None)
        return train_loss, val_loss

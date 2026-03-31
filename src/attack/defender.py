import copy
import os
import sys
from typing import Any, List, Sequence, Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import negative_sampling, to_undirected

_GPR_GAE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "GPR-GAE")
if _GPR_GAE_ROOT not in sys.path:
    sys.path.insert(0, _GPR_GAE_ROOT)

from robust_diffusion.models.gprgae import GPRGAE

try:
    # If your project provides it, prefer the exact same extraction logic as the dataset.
    from src.utils.graphbuilder import extract_view_by_transform  # type: ignore
except Exception:  # pragma: no cover
    extract_view_by_transform = None  # will be validated at runtime


class Defender():
    def __init__(self):
        pass

    def defend(self, data):
        raise NotImplementedError
    
    def to(self, device):
        raise NotImplementedError


class DiffusionPurifyDefender(Defender):
    """Run discrete diffusion purification on *homogeneous* multi-view projections of a hetero graph,
    then merge purified edge structures back into a hetero graph without losing original information.
    """

    def __init__(
        self,
        diffusion_steps: int,
        diffusion_model: Any,
        metapaths: Optional[List[Any]] = None,
        target_node_type: Optional[str] = None,
        weighted: bool = False,
    ):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.diffusion_model = diffusion_model
        self.metapaths = metapaths
        self.target_node_type = target_node_type
        self.weighted = weighted

    # ----------------------------
    # Public API
    # ----------------------------
    def to(self, device):
        self.diffusion_model.to(device)
        return self
    def defend(
        self,
        data: HeteroData,
        metapaths: Optional[List[Any]] = None,
        target_node_type: Optional[str] = None,
        weighted: Optional[bool] = None,
    ) -> HeteroData:
        """Defend (purify) a hetero graph by:
        1) splitting into multiple homogeneous views by metapaths,
        2) preprocessing each view for the discrete diffusion model,
        3) calling diffusion_model.purify(view, t_steps),
        4) applying ONLY edge-structure changes back to each view's corresponding relation,
        5) merging all views back to a HeteroData, preserving all original information.

        Notes
        - The diffusion model is assumed to be the DiscreteDenoisingDiffusion from diffusion_model_discrete.py,
          and its purify() returns (PlaceHolder discrete_sampled_s, node_mask).
        - We only use the purified E (edge classes) to update structure.
        """
        if not isinstance(data, HeteroData):
            raise TypeError(f"DiffusionPurifyDefender expects HeteroData, got: {type(data)}")

        metapaths = metapaths if metapaths is not None else self.metapaths
        target_node_type = target_node_type if target_node_type is not None else self.target_node_type
        weighted = weighted if weighted is not None else self.weighted

        if metapaths is None or len(metapaths) == 0:
            raise ValueError("metapaths must be provided (non-empty)")
        if target_node_type is None:
            raise ValueError("target_node_type must be provided")

        if extract_view_by_transform is None:
            raise ImportError(
                "extract_view_by_transform not found. Please ensure it's importable from src.utils.graphbuilder, "
                "or modify defend() to pass your function in."
            )

        # Deep-copy to guarantee we do not lose any original hetero information.
        out: HeteroData = copy.deepcopy(data)

        num_views = len(metapaths)

        # For YelpChi style extraction, each view's nodes are exactly the target node set,
        # so local node indices match hetero[target_node_type] indices.
        for v_idx, mp in enumerate(metapaths):
            # 1) Extract a homogeneous view (Data) on target nodes.
            view = extract_view_by_transform(data, mp, target_node_type, weighted=weighted)

            # 2) Preprocess view for diffusion model (same as YelpChiMultiviewDataset.process).
            view = self._preprocess_view_for_diffusion(view, v_idx=v_idx, num_views=num_views)

            # 3) Run diffusion purification on the homogeneous view.
            purified_discrete, node_mask = self.diffusion_model.purify(view, t_purify_steps=self.diffusion_steps)

            # 4) Convert purified dense E -> sparse edge_index (structure only).
            edge_index = self._dense_E_to_edge_index(purified_discrete.E, node_mask)

            # 5) Write back to hetero (only this view's corresponding relation).
            edge_type = self._edge_type_for_view(data, mp, target_node_type, v_idx)

            out = self._apply_view_structure_to_hetero(
                out,
                edge_type=edge_type,
                edge_index=edge_index,
                device=out[target_node_type].x.device if 'x' in out[target_node_type] else edge_index.device,
            )

        return out

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _preprocess_view_for_diffusion(view: Data, v_idx: int, num_views: int) -> Data:
        """Mirror YelpChiMultiviewDataset.process preprocessing so the diffusion model can compute correctly."""
        # Ensure undirected edge_index for diffusion (expects symmetric adjacency)
        if view.edge_index is not None and view.edge_index.numel() > 0:
            src, dst = view.edge_index[0], view.edge_index[1]
            num_nodes = view.num_nodes
            h = src.to(torch.long) * int(num_nodes) + dst.to(torch.long)
            hr = dst.to(torch.long) * int(num_nodes) + src.to(torch.long)
            if not bool(torch.isin(hr, h).all().item()):
                view.edge_index = to_undirected(view.edge_index, num_nodes=num_nodes)

        # Node features: set as one-hot of node labels if available (YelpChi uses binary node labels).
        if hasattr(view, 'y') and view.y is not None:
            try:
                # If y is node-level labels (shape [num_nodes] or [num_nodes, 1]), convert to one-hot.
                # If y is graph-level already, this will be handled below.
                if view.y.dim() == 1 or (view.y.dim() == 2 and view.y.size(0) == view.num_nodes):
                    view.x = F.one_hot(view.y.view(-1).long(), num_classes=2).float()
            except Exception:
                # If conversion fails, keep original x.
                pass

        # Edge attributes: two-class one-hot; all existing edges are class 1 ("edge present").
        num_edges = view.edge_index.size(1)
        edge_attr = torch.zeros((num_edges, 2), device=view.edge_index.device)
        if num_edges > 0:
            edge_attr[:, 1] = 1.0
        view.edge_attr = edge_attr

        # Graph-level y: one-hot of view id.
        view.y = F.one_hot(torch.tensor([v_idx], device=view.edge_index.device), num_classes=num_views).float()

        # Ensure batch exists for utils.to_dense() in purify()
        if not hasattr(view, 'batch') or view.batch is None:
            view.batch = torch.zeros(view.num_nodes, dtype=torch.long, device=view.edge_index.device)

        return view

    @staticmethod
    def _dense_E_to_edge_index(E_dense: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Convert purified dense edge class tensor E -> sparse edge_index.
        Assumptions:
          - E is discrete class indices with shape [bs, n, n] or [n, n]
          - class 1 means edge exists (aligned with preprocessing edge_attr[:,1]=1)
          - node_mask indicates valid nodes (shape [bs, n] or [n])
        """
        if E_dense.dim() == 3:
            # bs, n, n
            E0 = E_dense[0]
        elif E_dense.dim() == 2:
            E0 = E_dense
        else:
            raise ValueError(f"Unexpected E_dense shape: {tuple(E_dense.shape)}")

        if node_mask.dim() == 2:
            mask0 = node_mask[0]
        elif node_mask.dim() == 1:
            mask0 = node_mask
        else:
            raise ValueError(f"Unexpected node_mask shape: {tuple(node_mask.shape)}")

        # Only keep edges among valid nodes
        valid_idx = mask0.nonzero(as_tuple=False).view(-1)
        if valid_idx.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=E0.device)

        # Build a boolean matrix of edges
        # class 1 => edge exists
        edge_mat = (E0 == 1)

        # Remove self-loops by default (diffusion may generate them)
        n = E0.size(0)
        diag = torch.arange(n, device=E0.device)
        edge_mat[diag, diag] = False

        # Enforce node mask
        full_valid = torch.zeros(n, dtype=torch.bool, device=E0.device)
        full_valid[valid_idx] = True
        edge_mat = edge_mat & full_valid.view(-1, 1) & full_valid.view(1, -1)

        src, dst = edge_mat.nonzero(as_tuple=True)
        if src.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=E0.device)
        return torch.stack([src, dst], dim=0).long()

    @staticmethod
    def _edge_type_for_view(data: HeteroData, metapath_steps: Any, target_node_type: str, v_idx: int) -> Tuple[str, str, str]:
        """Decide which hetero edge_type this view should write to.

        - For single-hop, use the original edge_type if it exists.
        - For multi-hop, create a deterministic metapath relation name.
        """
        # Normalize like your extract_view_by_transform
        if isinstance(metapath_steps[0], str):
            actual_path = [tuple(metapath_steps)]
        else:
            actual_path = [tuple(step) for step in metapath_steps]

        if len(actual_path) == 1:
            et = actual_path[0]
            # Prefer exact triple if present; else fall back to (target, rel, target)
            if et in data.edge_types:
                return et  # type: ignore
            return (target_node_type, et[1], target_node_type)
        else:
            # Build a stable relation name; also compatible with many AddMetaPaths conventions.
            rels = []
            for step in actual_path:
                if len(step) == 3:
                    rels.append(str(step[1]))
                else:
                    rels.append(str(step))
            rel_name = "mp_" + "__".join(rels) + f"__v{v_idx}"
            return (target_node_type, rel_name, target_node_type)

    @staticmethod
    def _apply_view_structure_to_hetero(
        hetero: HeteroData,
        edge_type: Tuple[str, str, str],
        edge_index: torch.Tensor,
        device: torch.device,
    ) -> HeteroData:
        """Update (or create) a single edge_type's structure in-place, preserving everything else."""
        src_type, rel, dst_type = edge_type

        # Ensure the edge store exists; if not, create by assignment.
        hetero[edge_type].edge_index = edge_index.to(device)

        # Edge attributes: keep existing if shape matches and user wants; otherwise set as (num_edges, 2) one-hot.
        num_edges = edge_index.size(1)
        edge_attr = torch.zeros((num_edges, 2), device=device)
        if num_edges > 0:
            edge_attr[:, 1] = 1.0
        # hetero[edge_type].edge_attr = edge_attr

        return hetero


class JaccardDefender(Defender):
    """
    Jaccard edge-pruning defender for hetero graphs.

    The defender keeps the external graph format as `HeteroData`, but internally
    treats all node types as one homogeneous node space with padded features,
    then removes low-similarity edges relation by relation.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        binarize: bool = True,
        remove_self_loops: bool = True,
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.binarize = bool(binarize)
        self.remove_self_loops = bool(remove_self_loops)

    def to(self, device):
        return self

    def _build_global_binary_features(self, data: HeteroData) -> Tuple[torch.Tensor, dict]:
        node_types = list(data.node_types)
        if len(node_types) == 0:
            raise ValueError("JaccardDefender received an empty hetero graph.")

        feat_dim = 0
        device = None
        for node_type in node_types:
            x = getattr(data[node_type], "x", None)
            if x is None:
                continue
            if x.dim() == 1:
                x = x.view(-1, 1)
            feat_dim = max(feat_dim, int(x.size(-1)))
            if device is None:
                device = x.device

        if feat_dim <= 0:
            raise ValueError("JaccardDefender requires node features on at least one node type.")
        if device is None:
            device = torch.device("cpu")

        offsets = {}
        total_nodes = 0
        for node_type in node_types:
            offsets[node_type] = total_nodes
            total_nodes += int(data[node_type].num_nodes)

        x_global = torch.zeros((total_nodes, feat_dim), dtype=torch.float32, device=device)
        for node_type in node_types:
            start = offsets[node_type]
            count = int(data[node_type].num_nodes)
            if count == 0:
                continue
            x = getattr(data[node_type], "x", None)
            if x is None:
                continue
            if x.dim() == 1:
                x = x.view(-1, 1)
            x = x.to(device=device, dtype=torch.float32)
            x_global[start : start + count, : x.size(-1)] = x

        if self.binarize:
            x_global = x_global > 0
        else:
            x_global = x_global != 0
        return x_global, offsets

    def _edge_keep_mask(
        self,
        x_global: torch.Tensor,
        edge_index: torch.Tensor,
        src_offset: int,
        dst_offset: int,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

        global_src = edge_index[0].to(device=x_global.device, dtype=torch.long) + int(src_offset)
        global_dst = edge_index[1].to(device=x_global.device, dtype=torch.long) + int(dst_offset)

        src_feat = x_global[global_src]
        dst_feat = x_global[global_dst]
        inter = torch.logical_and(src_feat, dst_feat).sum(dim=1)
        union = torch.logical_or(src_feat, dst_feat).sum(dim=1)
        sim = inter.to(torch.float32) / union.clamp_min(1).to(torch.float32)

        keep = sim >= self.threshold
        if self.remove_self_loops:
            keep = keep & (global_src != global_dst)
        return keep.to(edge_index.device)

    def defend(self, data: HeteroData) -> HeteroData:
        if not isinstance(data, HeteroData):
            raise TypeError(f"JaccardDefender expects HeteroData, got: {type(data)}")

        out: HeteroData = copy.deepcopy(data)
        x_global, offsets = self._build_global_binary_features(data)

        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            edge_store = data[edge_type]
            edge_index = getattr(edge_store, "edge_index", None)
            if edge_index is None:
                continue

            keep = self._edge_keep_mask(
                x_global=x_global,
                edge_index=edge_index,
                src_offset=offsets[src_type],
                dst_offset=offsets[dst_type],
            )
            out[edge_type].edge_index = edge_index[:, keep]

            edge_attr = getattr(edge_store, "edge_attr", None)
            if edge_attr is not None and edge_attr.size(0) == edge_index.size(1):
                out[edge_type].edge_attr = edge_attr[keep]

        return out


class GPRGAEDefender(Defender):
    """
    Minimal GPR-GAE purifier wrapper for target-target relations in a hetero graph.
    """

    def __init__(
        self,
        target_node_type: str,
        hidden: int = 128,
        K: int = 7,
        dropout_link: float = 0.0,
        dropout_mlp: float = 0.7,
        self_loop: bool = False,
        activation_str: str = "elu",
        concat_activation_str: str = "elu",
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        train_epochs: int = 100,
        negative_ratio: float = 1.0,
        purify_steps: int = 5,
        purify_tol: float = 1e-4,
        edge_keep_threshold: float = 0.5,
        batch_decode: bool = True,
    ):
        super().__init__()
        self.target_node_type = target_node_type
        self.hidden = int(hidden)
        self.K = int(K)
        self.dropout_link = float(dropout_link)
        self.dropout_mlp = float(dropout_mlp)
        self.self_loop = bool(self_loop)
        self.activation_str = activation_str
        self.concat_activation_str = concat_activation_str
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.train_epochs = int(train_epochs)
        self.negative_ratio = float(negative_ratio)
        self.purify_steps = int(purify_steps)
        self.purify_tol = float(purify_tol)
        self.edge_keep_threshold = float(edge_keep_threshold)
        self.batch_decode = bool(batch_decode)
        self.device = "cpu"
        self.model: Optional[GPRGAE] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

    def _ensure_model(self, input_dim: int) -> None:
        if self.model is not None:
            return
        self.model = GPRGAE(
            n_features=int(input_dim),
            hidden=self.hidden,
            K=self.K,
            dropout_link=self.dropout_link,
            dropout_MLP=self.dropout_mlp,
            self_loop=self.self_loop,
            activation_str=self.activation_str,
            concat_activation_str=self.concat_activation_str,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _target_edge_types(self, graph: HeteroData) -> List[Tuple[str, str, str]]:
        return [et for et in graph.edge_types if et[0] == self.target_node_type and et[2] == self.target_node_type]

    def _prepare_relation_graph(self, graph: HeteroData, edge_type: Tuple[str, str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = graph[self.target_node_type].x.to(self.device, dtype=torch.float)
        edge_index = graph[edge_type].edge_index.to(self.device)
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            edge_index = to_undirected(edge_index, num_nodes=int(graph[self.target_node_type].num_nodes))
        edge_weight = torch.ones(edge_index.size(1), dtype=x.dtype, device=self.device)
        return x, edge_index

    def _sample_negative_edges(self, num_nodes: int, pos_edge_index: torch.Tensor) -> torch.Tensor:
        if pos_edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=pos_edge_index.device)
        num_pos_undir = max(1, pos_edge_index.size(1) // 2)
        num_neg = max(1, int(round(self.negative_ratio * num_pos_undir)))
        neg_edge_index = negative_sampling(
            pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg,
            force_undirected=True,
        ).to(pos_edge_index.device)
        return torch.cat([neg_edge_index, neg_edge_index[[1, 0]]], dim=1)

    def _self_supervised_loss(self, x: torch.Tensor, edge_index: torch.Tensor) -> Optional[torch.Tensor]:
        if edge_index.numel() == 0:
            return None
        neg_edge_index = self._sample_negative_edges(x.size(0), edge_index)
        if neg_edge_index.numel() == 0:
            return None

        pos_pred = self.model.link_prediction(x, (edge_index, torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)), edge_index)
        neg_pred = self.model.link_prediction(x, (edge_index, torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)), neg_edge_index)
        combined_edges = torch.cat([edge_index, neg_edge_index], dim=1)
        combined_pred = self.model.link_prediction(x, (edge_index, torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)), combined_edges)
        reverse_pred = self.model.link_prediction(x, (edge_index, torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)), combined_edges[[1, 0]])

        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        reg_loss = F.mse_loss(combined_pred, reverse_pred)
        return pos_loss + neg_loss + 0.2 * reg_loss

    def fit(self, dataset) -> None:
        first_graph = dataset[0]
        input_dim = int(first_graph[self.target_node_type].x.size(-1))
        self._ensure_model(input_dim)
        self.model.train()

        for epoch in range(self.train_epochs):
            losses = []
            for i in range(len(dataset)):
                graph = dataset[i]
                edge_types = self._target_edge_types(graph)
                if not edge_types:
                    continue
                for edge_type in edge_types:
                    x, edge_index = self._prepare_relation_graph(graph, edge_type)
                    loss = self._self_supervised_loss(x, edge_index)
                    if loss is None:
                        continue
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(float(loss.item()))
            if ((epoch + 1) % 10 == 0) or epoch == 0 or (epoch + 1) == self.train_epochs:
                avg_loss = float(sum(losses) / max(len(losses), 1))
                print(f"[gprgae:defender] epoch={epoch + 1}/{self.train_epochs} loss={avg_loss:.4f}")

    @torch.no_grad()
    def defend(self, data: HeteroData) -> HeteroData:
        if self.model is None:
            raise RuntimeError("GPRGAEDefender must be fitted before calling defend().")
        if not isinstance(data, HeteroData):
            raise TypeError(f"GPRGAEDefender expects HeteroData, got: {type(data)}")

        out: HeteroData = copy.deepcopy(data)
        self.model.eval()

        for edge_type in self._target_edge_types(data):
            x, edge_index = self._prepare_relation_graph(data, edge_type)
            if edge_index.numel() == 0:
                out[edge_type].edge_index = edge_index
                continue
            purified_edge_index, purified_edge_weight = self.model.purify_adj(
                x,
                (edge_index, torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)),
                batch=self.batch_decode,
                steps=self.purify_steps,
                tol=self.purify_tol,
            )
            keep = purified_edge_weight >= self.edge_keep_threshold
            new_edge_index = purified_edge_index[:, keep]
            out[edge_type].edge_index = new_edge_index.to(data[edge_type].edge_index.device)

            edge_attr = getattr(data[edge_type], "edge_attr", None)
            if edge_attr is not None:
                out[edge_type].edge_attr = None

        return out

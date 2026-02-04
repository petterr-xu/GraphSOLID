import copy
from typing import Any, List, Sequence, Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected

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

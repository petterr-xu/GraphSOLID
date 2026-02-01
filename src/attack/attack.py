import torch
import warnings
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import scipy.sparse as sp
from typing import Optional, Tuple
from torch_geometric.data import Data, HeteroData

from src.utils import graphbuilder
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
    def __init__(self, dataset_module:AbstractDataModule):
        self.dataset_module = dataset_module
    def to(self,device):
        return self
    def attack(self,data):
        return data

class RandomAttacker(Attacker):
    """
    Randomly perturb a fixed ratio of edges on the *original HeteroData* (no to_homogeneous()).

    Behavior:
    - For each edge_type in a HeteroData, delete k edges and add k new edges, where
      k = floor(perturb_ratio * num_edges_of_that_edge_type).
      (So every relation is perturbed with the same ratio.)
    - Only edge structure (edge_index) is changed.
    - Node/edge attributes are preserved as much as possible:
        * existing edge_attr (if any) is kept for retained edges
        * new edges get zero edge_attr with the same feature dimension (if edge_attr exists)
    - For Data (homogeneous) input, it perturbs the single edge_index similarly.

    Notes:
    - For (src_type == dst_type), self-loops are avoided by default (allow_self_loops=False).
    - Duplicated edges are avoided.
    """

    def __init__(
        self,
        dataset_module,
        perturb_ratio: float = 0.05,
        seed: Optional[int] = None,
        allow_self_loops: bool = False,
        max_sampling_rounds: int = 50,
        oversample_factor: int = 5,
    ):
        super().__init__(dataset_module)
        assert 0.0 <= perturb_ratio <= 1.0
        self.perturb_ratio = float(perturb_ratio)
        self.seed = seed
        self.allow_self_loops = allow_self_loops
        self.max_sampling_rounds = int(max_sampling_rounds)
        self.oversample_factor = int(oversample_factor)

    @staticmethod
    def _hash_edges(src: torch.Tensor, dst: torch.Tensor, num_dst: int) -> torch.Tensor:
        # unique id for each directed edge (src, dst)
        return src.to(torch.long) * int(num_dst) + dst.to(torch.long)

    def _sample_new_edges(
        self,
        num_src: int,
        num_dst: int,
        k: int,
        existing_hash: torch.Tensor,
        device: torch.device,
        forbid_self_loops: bool,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample k new edges not in existing_hash.
        Returns (new_src, new_dst) each shape [k].
        """
        if k <= 0:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device))

        existing = existing_hash
        new_src_list = []
        new_dst_list = []

        remaining = k
        for _ in range(self.max_sampling_rounds):
            if remaining <= 0:
                break

            m = max(remaining * self.oversample_factor, remaining + 10)
            cand_src = torch.randint(0, num_src, (m,), device=device, generator=generator)
            cand_dst = torch.randint(0, num_dst, (m,), device=device, generator=generator)

            if forbid_self_loops and num_src == num_dst:
                mask = cand_src != cand_dst
                cand_src, cand_dst = cand_src[mask], cand_dst[mask]
                if cand_src.numel() == 0:
                    continue

            cand_hash = self._hash_edges(cand_src, cand_dst, num_dst)

            # Filter out existing edges
            keep = ~torch.isin(cand_hash, existing)
            cand_src, cand_dst, cand_hash = cand_src[keep], cand_dst[keep], cand_hash[keep]
            if cand_src.numel() == 0:
                continue

            # Deduplicate candidates themselves
            cand_hash, uniq_idx = torch.unique(cand_hash, return_inverse=False, return_counts=False, sorted=False, return_index=True)
            cand_src = cand_src[uniq_idx]
            cand_dst = cand_dst[uniq_idx]

            take = min(remaining, cand_src.numel())
            if take <= 0:
                continue

            new_src_list.append(cand_src[:take])
            new_dst_list.append(cand_dst[:take])

            # Update existing set with newly taken edges to prevent duplicates across rounds
            taken_hash = self._hash_edges(cand_src[:take], cand_dst[:take], num_dst)
            existing = torch.cat([existing, taken_hash], dim=0)

            remaining -= take

        if len(new_src_list) == 0:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device))

        new_src = torch.cat(new_src_list, dim=0)
        new_dst = torch.cat(new_dst_list, dim=0)

        if new_src.numel() > k:
            new_src = new_src[:k]
            new_dst = new_dst[:k]

        return new_src, new_dst

    def _perturb_edge_index(
        self,
        edge_index: torch.Tensor,
        num_src: int,
        num_dst: int,
        edge_attr: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        forbid_self_loops: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Delete k edges and add k new edges. Return (new_edge_index, new_edge_attr).
        """
        device = edge_index.device
        E = edge_index.size(1)
        k = int(self.perturb_ratio * E)

        if E == 0 or k == 0:
            return edge_index, edge_attr

        # 1) random delete k edges
        perm = torch.randperm(E, device=device, generator=generator)
        del_idx = perm[:k]
        keep_mask = torch.ones(E, dtype=torch.bool, device=device)
        keep_mask[del_idx] = False

        kept_edge_index = edge_index[:, keep_mask]

        kept_edge_attr = None
        if edge_attr is not None:
            kept_edge_attr = edge_attr[keep_mask]

        # 2) random add k new edges
        src, dst = kept_edge_index[0], kept_edge_index[1]
        existing_hash = self._hash_edges(src, dst, num_dst)
        new_src, new_dst = self._sample_new_edges(
            num_src=num_src,
            num_dst=num_dst,
            k=k,
            existing_hash=existing_hash,
            device=device,
            forbid_self_loops=forbid_self_loops,
            generator=generator,
        )

        if new_src.numel() == 0:
            return kept_edge_index, kept_edge_attr

        added_edge_index = torch.stack([new_src, new_dst], dim=0)
        new_edge_index = torch.cat([kept_edge_index, added_edge_index], dim=1)

        # edge_attr: preserve kept; added edges -> zeros
        new_edge_attr = kept_edge_attr
        if edge_attr is not None:
            if edge_attr.dim() >= 2:
                feat_dim = edge_attr.size(-1)
                added_attr = torch.zeros((added_edge_index.size(1), feat_dim), device=device, dtype=edge_attr.dtype)
            else:
                added_attr = torch.zeros((added_edge_index.size(1),), device=device, dtype=edge_attr.dtype)
            new_edge_attr = torch.cat([kept_edge_attr, added_attr], dim=0)

        return new_edge_index, new_edge_attr

    def attack(self, data):
        # Setup RNG
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        # Homogeneous graph case
        if isinstance(data, Data):
            warnings.warn("RandomAttacker works best with HeteroData. Proceeding with Data (homogeneous graph).")
            out = data.clone()
            num_nodes = out.num_nodes
            forbid_self_loops = (not self.allow_self_loops)
            new_ei, new_ea = self._perturb_edge_index(
                edge_index=out.edge_index,
                num_src=num_nodes,
                num_dst=num_nodes,
                edge_attr=getattr(out, "edge_attr", None),
                generator=generator,
                forbid_self_loops=forbid_self_loops,
            )
            out.edge_index = new_ei
            if getattr(out, "edge_attr", None) is not None:
                out.edge_attr = new_ea
            return out

        # Heterogeneous graph case (recommended)
        if not isinstance(data, HeteroData):
            raise RuntimeError("Expected `data` to be `Data` or `HeteroData`.")

        out = data.clone()

        for edge_type in out.edge_types:
            store = out[edge_type]
            if not hasattr(store, "edge_index") or store.edge_index is None:
                continue

            src_type, _, dst_type = edge_type
            num_src = out[src_type].num_nodes
            num_dst = out[dst_type].num_nodes

            forbid_self_loops = (src_type == dst_type) and (not self.allow_self_loops)

            edge_attr = getattr(store, "edge_attr", None)
            new_ei, new_ea = self._perturb_edge_index(
                edge_index=store.edge_index,
                num_src=num_src,
                num_dst=num_dst,
                edge_attr=edge_attr,
                generator=generator,
                forbid_self_loops=forbid_self_loops,
            )

            store.edge_index = new_ei
            if edge_attr is not None:
                store.edge_attr = new_ea

        return out

class Metattacker(Attacker):
    def __init__(self, dataset_module:AbstractDataModule, perturb_ratio, attack_varient='Meta-Self', re_trainings=5, device=0, train_iters = 200):
        super().__init__(dataset_module)
        self.GPU_ID = device
        self.share_perturbations = perturb_ratio
        self.train_iters = train_iters
        self.re_trainings = re_trainings
        self.dtype = tf.float32
        self.ENFORCE_LL_CONSTRAINT = False
        self.attack_variant = attack_varient

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
        surrogate = mtk.GCNSparse(_A_obs, _X_obs, _Z_obs, hidden_sizes, gpu_id=self.GPU_ID)
        surrogate.build(with_relu=False)
        surrogate.train(split_train)

        # 2. 自训练标签预测
        labels_self_training = np.eye(_K)[surrogate.logits.eval(session=surrogate.session).argmax(1)]
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

# class nettack():
#     pass

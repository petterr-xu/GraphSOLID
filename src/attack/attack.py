import torch
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import scipy.sparse as sp
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
    def attack(self,data):
        return data

class Metattacker(Attacker):
    def __init__(self, dataset_module:AbstractDataModule, share_perturbations, attack_varient='Meta-Self', re_trainings=5, device=0, train_iters = 200):
        super().__init__(dataset_module)
        self.GPU_ID = device
        self.share_perturbations = share_perturbations
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

        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

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
            try:
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
            except Exception:
                # Fallback: if anything goes wrong, return homogeneous attacked graph
                return attacked_homo

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

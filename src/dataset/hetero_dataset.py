import os
import torch
import random
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn.functional as F
from torch_geometric.utils import subgraph, k_hop_subgraph, to_undirected
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import InMemoryDataset, Dataset, HeteroData, Data
from torch_geometric.loader import ClusterData, ClusterLoader, HGTLoader
from .abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.DiGress.src import utils
from torch_geometric.transforms import RandomNodeSplit

from src.utils.graphbuilder import (
    extract_view_by_transform, 
    hetero_cluster_split,
    extract_view_by_matrix
)

class MultiviewDataset(Dataset):
    def __init__(self, stage, root, hetero_dataset, metapaths, target, transform=None, pre_transform=None):
        self.stage = stage
        self.metapaths = metapaths
        self.target = target
        self.hetero_base = hetero_dataset # 此时 hetero_base 也是一个 Dataset 对象
        
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        # 视图总数 = 基础异构图数 * 元路径数
        total_views = len(self.hetero_base) * len(self.metapaths)
        return [f'view_{self.stage}_{i}.pt' for i in range(total_views)]

    def len(self):
        return len(self.hetero_base) * len(self.metapaths)

    def get(self, idx):
        path = osp.join(self.processed_dir, f'view_{self.stage}_{idx}.pt')
        return torch.load(path)

    def process(self):
        print(f"Generating multiview subgraphs for {self.stage}...")
        global_idx = 0
        
        # 遍历基础异构数据集（此时 hetero_base[sub_idx] 会触发磁盘读取）
        for sub_idx in range(len(self.hetero_base)):
            h_sub = self.hetero_base[sub_idx]
            
            for v_idx, mp in enumerate(self.metapaths):
                view_data = extract_view_by_transform(h_sub, mp, self.target)
                view_data.node_y = view_data.y.clone()

                # DiGress expects symmetric adjacency; ensure every edge has its reverse.
                if view_data.edge_index is not None and view_data.edge_index.numel() > 0:
                    view_data.edge_index = to_undirected(
                        view_data.edge_index,
                        num_nodes=view_data.num_nodes
                    )
                
                # 特征处理（保持你原有的逻辑）
                if hasattr(view_data, 'y') and view_data.y is not None:
                    view_data.x = torch.nn.functional.one_hot(view_data.y.long(), num_classes=2).float()
                
                num_edges = view_data.edge_index.size(1)
                edge_attr = torch.zeros((num_edges, 2))
                edge_attr[:, 1] = 1.0
                view_data.edge_attr = edge_attr
                
                view_data.y = torch.nn.functional.one_hot(torch.tensor([v_idx]), num_classes=len(self.metapaths)).float()
                view_data.parent_hetero_idx = sub_idx
                
                # 【核心逻辑】：处理一个存一个，避免内存溢出
                save_path = osp.join(self.processed_dir, f'view_{self.stage}_{global_idx}.pt')
                torch.save(view_data, save_path)
                global_idx += 1

class HeteroDataset(Dataset):
    def __init__(
        self,
        stage,
        root,
        original_data: HeteroData,
        target_node_type,
        num_parts=100,
        transform=None,
        pre_transform=None,
        strategy='HGT',
        node_val_ratio: float = 0.2,
        node_test_ratio: float = 0.4,
        rebalance_train_majority: bool = False,
        train_majority_ratio_cap: float = 0.85,
    ):
        self.stage = stage
        self.original_data = original_data
        self.num_parts = num_parts
        self.target_node_type = target_node_type
        self.node_val_ratio = float(node_val_ratio)
        self.node_test_ratio = float(node_test_ratio)
        self.rebalance_train_majority = bool(rebalance_train_majority)
        self.train_majority_ratio_cap = float(train_majority_ratio_cap)
        # 计算各阶段应有的数量（用于 len()）
        self.train_n = int(num_parts * 0.8)
        self.val_n = int(num_parts * 0.1)
        self.test_n = num_parts - self.train_n - self.val_n
        self.subgraph_strategy = strategy
        
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        # 只要存在这些“哨兵文件”，PyG 就不会重新执行 process
        return [f'hetero_{self.stage}_{i}.pt' for i in range(self.len())]

    def len(self):
        return {'train': self.train_n, 'val': self.val_n, 'test': self.test_n}[self.stage]

    def get(self, idx):
        # 懒加载：只有在访问 dataset[idx] 时才读磁盘
        path = osp.join(self.processed_dir, f'hetero_{self.stage}_{idx}.pt')
        return torch.load(path)
    
    def _build_type_ptr(self):
        ptr = {}
        offset = 0
        for ntype in self.original_data.node_types:
            ptr[ntype] = offset
            offset += self.original_data[ntype].num_nodes
        return ptr
    
    def _cap_subset(self, subset, max_nodes):
        if subset.numel() <= max_nodes:
            return subset
        perm = torch.randperm(subset.numel())
        return subset[perm[:max_nodes]]
    
    def _validate_label_coverage(self, sub_hetero, target_type):
        if not hasattr(sub_hetero[target_type], "y"):
            return True

        sub_labels = sub_hetero[target_type].y.cpu().numpy()
        orig_labels = self.original_data[target_type].y.cpu().numpy()

        return set(np.unique(sub_labels)) == set(np.unique(orig_labels))

    @staticmethod
    def _to_label_index(y: torch.Tensor) -> torch.Tensor:
        if y.dim() > 1 and y.size(-1) > 1:
            return y.argmax(dim=-1).to(torch.long)
        return y.view(-1).to(torch.long)

    def _split_and_rebalance(self, sub_hetero: HeteroData) -> HeteroData:
        sub_hetero = RandomNodeSplit(num_val=self.node_val_ratio, num_test=self.node_test_ratio)(sub_hetero)
        if not self.rebalance_train_majority or self.stage != "train":
            return sub_hetero

        target = self.target_node_type
        if not hasattr(sub_hetero[target], "y") or not hasattr(sub_hetero[target], "train_mask"):
            return sub_hetero
        if not (0.5 < self.train_majority_ratio_cap < 1.0):
            return sub_hetero

        y = self._to_label_index(sub_hetero[target].y).cpu()
        train_mask = sub_hetero[target].train_mask.clone().to(torch.bool).cpu()
        idx_train = train_mask.nonzero(as_tuple=False).view(-1)
        if idx_train.numel() == 0:
            return sub_hetero

        y_train = y[idx_train]
        class_counts = torch.bincount(y_train)
        if class_counts.numel() < 2:
            return sub_hetero

        majority_cls = int(torch.argmax(class_counts).item())
        majority_count = int(class_counts[majority_cls].item())
        total_count = int(class_counts.sum().item())
        other_count = total_count - majority_count
        if other_count <= 0:
            return sub_hetero

        max_majority = int((self.train_majority_ratio_cap / (1.0 - self.train_majority_ratio_cap)) * other_count)
        if majority_count <= max_majority:
            return sub_hetero
        max_majority = max(1, max_majority)

        majority_idx_global = idx_train[y_train == majority_cls]
        keep_perm = torch.randperm(majority_idx_global.numel())[:max_majority]
        keep_majority = majority_idx_global[keep_perm]

        new_train_mask = train_mask.clone()
        new_train_mask[majority_idx_global] = False
        new_train_mask[keep_majority] = True
        sub_hetero[target].train_mask = new_train_mask.to(sub_hetero[target].train_mask.device)
        return sub_hetero
    
    def HGT_partitioning(self):
        self.num_hops = 3
        self.fanout = 5
        hetero = self.original_data
        target_type = self.target_node_type
        if not hasattr(hetero[target_type], "y"):
            raise ValueError(f"Target node type '{target_type}' has no labels, cannot compute class statistics.")

        target_nodes = torch.arange(hetero[target_type].num_nodes)
        perm = torch.randperm(len(target_nodes))
        target_nodes = target_nodes[perm]

        # seed_nodes = target_nodes[:self.num_parts]
        # growth = 2
        num_samples = {
                ntype: [self.fanout ** (i+1) for i in range(self.num_hops)]
                for ntype in hetero.node_types
            }

        loader = HGTLoader(
            data=hetero,
            input_nodes=(target_type, target_nodes),
            num_samples=num_samples,
            batch_size=1,
            shuffle=False
        )

        all_hetero_subs = []

        max_size = 0
        min_size = float("inf")
        size_count = 0
        sub_nums = 0
        reject_nums = 0

        target_y_full = self._to_label_index(hetero[target_type].y)
        num_classes = int(target_y_full.max().item()) + 1 if target_y_full.numel() > 0 else 0
        split_class_sum = {
            "train": np.zeros(num_classes, dtype=float),
            "val": np.zeros(num_classes, dtype=float),
            "test": np.zeros(num_classes, dtype=float),
        }

        pbar = tqdm(total=self.num_parts, desc="Sampling Subgraphs (HGT)", ncols=120)
        for batch in loader:
            sub_hetero = batch

            if not self._validate_label_coverage(sub_hetero, target_type):
                reject_nums += 1
                continue
            else:
                sub_nums += 1

            sub_hetero = self._split_and_rebalance(sub_hetero)

            target_y = self._to_label_index(sub_hetero[target_type].y)
            for split_name in ["train", "val", "test"]:
                mask_name = f"{split_name}_mask"
                if hasattr(sub_hetero[target_type], mask_name):
                    mask = getattr(sub_hetero[target_type], mask_name).to(torch.bool)
                    if int(mask.sum().item()) > 0:
                        cnt = torch.bincount(target_y[mask], minlength=num_classes).to(torch.float).cpu().numpy()
                    else:
                        cnt = np.zeros(num_classes, dtype=float)
                else:
                    cnt = np.zeros(num_classes, dtype=float)
                split_class_sum[split_name] += cnt

            n = sub_hetero.num_nodes
            max_size = max(max_size, n)
            min_size = min(min_size, n)
            size_count += n
            all_hetero_subs.append(sub_hetero)

            pbar.update(1)
            pbar.set_postfix({"size": n, "avg_nodes": f"{size_count / sub_nums:.1f}", "rejected": reject_nums})
            
            if sub_nums >= self.num_parts:
                break

        pbar.close()

        if sub_nums == 0:
            print("[HGT_partitioning] No valid subgraphs sampled. Please check label coverage constraints.")
            return all_hetero_subs

        print(
            "[HGT_partitioning] "
            f"accepted={sub_nums} rejected={reject_nums} "
            f"nodes(avg/min/max)={size_count / sub_nums:.1f}/{int(min_size)}/{int(max_size)}"
        )
        for split_name in ["train", "val", "test"]:
            mean_counts = split_class_sum[split_name] / float(sub_nums)
            denom = float(mean_counts.sum())
            mean_ratio = (mean_counts / denom) if denom > 0 else np.zeros(num_classes, dtype=float)
            mean_counts_str = ", ".join(f"{v:.1f}" for v in mean_counts.tolist())
            mean_ratio_str = ", ".join(f"{v:.3f}" for v in mean_ratio.tolist())
            print(
                f"[HGT_partitioning:{split_name}] mean class counts=[{mean_counts_str}] "
                f"mean ratio=[{mean_ratio_str}]"
            )

        return all_hetero_subs
    
    def cluster_partitioning(self):

        homo_data = self.original_data.to_homogeneous()
        cluster_data = ClusterData(homo_data, num_parts=self.num_parts, recursive=False)
        loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)

        node_types, edge_types = self.original_data.node_types, self.original_data.edge_types
        all_hetero_subs = []
        for sub_homo in enumerate(tqdm(loader, desc='Sampling Subgraphs (cluster strategy)')):
            # 还原为异构图
            sub_hetero = sub_homo.to_heterogeneous(node_type_names=node_types, edge_type_names=edge_types)
            sub_hetero = self._split_and_rebalance(sub_hetero)
            all_hetero_subs.append(sub_hetero)
        return all_hetero_subs
    
    def random_partitioning(self):
        hetero = self.original_data
        homo = hetero.to_homogeneous()
        num_nodes = homo.num_nodes
        type_ptr = self._build_type_ptr()
        type_id_map = {ntype: i for i, ntype in enumerate(hetero.node_types)}

        all_nodes = list(range(num_nodes))
        random.shuffle(all_nodes)
        center_nodes = all_nodes[:self.num_parts]

        all_hetero_subs = []

        max_size = -1
        min_size = self.original_data.num_nodes
        size_count = 0
        for center in tqdm(center_nodes, desc="Sampling Subgraphs (random strategy)"):
            # 同构空间采样
            subset, _, _, _ = k_hop_subgraph(
                center,
                num_hops=2,
                edge_index=homo.edge_index,
                relabel_nodes=False,
                num_nodes=num_nodes
            )

            subset = subset.cpu()
            subset = self._cap_subset(subset, 200)
            node_types = homo.node_type[subset]

            subset_dict = {}

            for ntype in hetero.node_types:
                tid = type_id_map[ntype]
                mask = node_types == tid

                if mask.sum() == 0:
                    continue

                global_ids = subset[mask]
                local_ids = global_ids - type_ptr[ntype]

                subset_dict[ntype] = local_ids

            sub_hetero = hetero.subgraph(subset_dict)
            sub_hetero = self._split_and_rebalance(sub_hetero)

            max_size = max(max_size, sub_hetero.num_nodes)
            min_size = min(min_size, sub_hetero.num_nodes)
            size_count += sub_hetero.num_nodes

            all_hetero_subs.append(sub_hetero)

        print(f"Max subgraph size: {max_size}, Min subgraph size: {min_size}, Avg size: {size_count / self.num_parts}")

        return all_hetero_subs

    def process(self):
        if self.original_data is None:
            # 如果没有传原图且没找到处理好的文件，报错
            raise ValueError("Processed files not found. Please provide 'original_data' to partition.")

        print(f"Partitioning original hetero-graph into {self.num_parts} parts...")

        partition_func_map = {'cluster':self.cluster_partitioning, 'random':self.random_partitioning, 'HGT':self.HGT_partitioning}
        if self.subgraph_strategy not in partition_func_map:
            raise ValueError(f"Unknown subgraph strategy: {self.subgraph_strategy}")
        all_hetero_subs = partition_func_map[self.subgraph_strategy]()

        random.seed(42)
        random.shuffle(all_hetero_subs)
        
        # 划分列表
        train_list = all_hetero_subs[:self.train_n]
        val_list = all_hetero_subs[self.train_n : self.train_n + self.val_n]
        test_list = all_hetero_subs[self.train_n + self.val_n :]

        # 【核心逻辑】：循环单独保存每一个子图文件
        for stage_name, data_list in zip(['train', 'val', 'test'], [train_list, val_list, test_list]):
            for i, data in enumerate(data_list):
                torch.save(data, osp.join(self.processed_dir, f'hetero_{stage_name}_{i}.pt'))

class HeteroDataModule(AbstractDataModule):
    def __init__(self, cfg, hetero_graph: HeteroData):
        self.cfg = cfg
        self.hetero_graph = hetero_graph
        self.root = cfg.dataset.datadir
        self.metapaths = cfg.dataset.metapaths
        self.target = cfg.dataset.target
        self.num_parts = cfg.dataset.num_parts
        train_node_val_ratio = float(getattr(cfg.dataset, "train_node_val_ratio", 0.2))
        train_node_test_ratio = float(getattr(cfg.dataset, "train_node_test_ratio", 0.4))
        rebalance_train_majority = bool(getattr(cfg.dataset, "rebalance_train_majority", False))
        train_majority_ratio_cap = float(getattr(cfg.dataset, "train_majority_ratio_cap", 0.85))
        # 构造异构图数据集
        train_hetero = HeteroDataset(
            'train',
            self.root,
            self.hetero_graph,
            self.target,
            self.num_parts,
            node_val_ratio=train_node_val_ratio,
            node_test_ratio=train_node_test_ratio,
            rebalance_train_majority=rebalance_train_majority,
            train_majority_ratio_cap=train_majority_ratio_cap,
        )
        val_hetero = HeteroDataset(
            'val',
            self.root,
            self.hetero_graph,
            self.target,
            self.num_parts,
            node_val_ratio=0.2,
            node_test_ratio=0.4,
            rebalance_train_majority=False,
        )
        test_hetero = HeteroDataset(
            'test',
            self.root,
            self.hetero_graph,
            self.target,
            self.num_parts,
            node_val_ratio=0.2,
            node_test_ratio=0.4,
            rebalance_train_majority=False,
        )
        self.hetero_datasets = {
            'train': train_hetero,
            'val': val_hetero,
            'test': test_hetero
        }
        # 根据异构图构造同构图数据集
        datasets = {
            'train': MultiviewDataset('train', self.root, train_hetero, self.metapaths, self.target),
            'val': MultiviewDataset('val', self.root, val_hetero, self.metapaths, self.target),
            'test': MultiviewDataset('test', self.root, test_hetero, self.metapaths, self.target)
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset
    
    def __getitem__(self, item):
        return self.inner[item]
    
    def node_types(self):
            return super().node_types()
    
    def edge_counts(self):
        # 没有 edge_attr，默认边只有两类：0 (无边), 1 (有边)
        num_classes = 2
        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            # 1. 计算当前 batch 中所有图中可能的总边数 (n * (n-1))
            # 注意：这里假设是全连接的有向图视角，如果不允许自环，总对数为 n*(n-1)
            unique, counts = torch.unique(data.batch, return_counts=True)
            
            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            # 2. 计算当前 batch 中实际存在的边数
            # data.edge_index.shape[1] 即为存在的边（类别 1）
            num_edges = data.edge_index.shape[1]
            
            # 3. 计算不存在的边数（类别 0）
            num_non_edges = all_pairs - num_edges
            assert num_non_edges >= 0, "实际边数超过了最大可能边数，请检查数据是否有重复边或自环"

            # 4. 累加计数
            d[0] += num_non_edges
            d[1] += num_edges

        # 5. 归一化得到概率分布
        d = d / d.sum()
        return d


class HeteroDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.datamodule = datamodule
        self.name = 'HeteroCustom'
        # 获取基础统计信息
        # 注意：由于是异构图，这里的 max_n_nodes 是所有类型节点总和的最大值
        self.n_nodes = datamodule.node_counts() 
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
    
        # 调用父类完成 DistributionNodes 等初始化
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': example_batch['y'].size(1)}

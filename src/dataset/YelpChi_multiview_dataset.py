import os
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Dataset, HeteroData
from torch_geometric.loader import ClusterData, ClusterLoader
from src.DiGress.src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.DiGress.src import utils

from src.utils.graphbuilder import (
    extract_view_by_transform, 
    hetero_cluster_split,
    extract_view_by_matrix
)

class YelpChiMultiviewDataset(InMemoryDataset):
    def __init__(self, stage, root, hetero_dataset, metapaths, target, transform=None, pre_transform=None):
        self.stage = stage
        self.metapaths = metapaths
        self.target = target
        
        # 【核心】：显式持有异构数据集对象的引用
        self.hetero_base = hetero_dataset 
        
        super().__init__(root, transform, pre_transform)
        idx = {'train': 0, 'val': 1, 'test': 2}[stage]
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def processed_file_names(self):
        return ['view_subgraphs_train.pt', 'view_subgraphs_val.pt', 'view_subgraphs_test.pt']

    def process(self):
        print(f"Extracting {len(self.metapaths)} views from hetero_dataset for {self.stage}...")
        multiview_list = []

        # 直接遍历传入的异构数据集实例
        for sub_idx, h_sub in enumerate(self.hetero_base):
            for v_idx, mp in enumerate(self.metapaths):
                # 调用你的 API 提取视图
                view_data = extract_view_by_transform(h_sub, mp, self.target)
                
                # --- 特征处理 ---
                # 节点特征 x: 根据原标签 y 制作 one-hot (维度适配 DiGress)
                if hasattr(view_data, 'y') and view_data.y is not None:
                    view_data.x = F.one_hot(view_data.y.long(), num_classes=2).float()
                
                # 边特征 edge_attr: [E, 2], [0, 1] 表示有边
                num_edges = view_data.edge_index.size(1)
                edge_attr = torch.zeros((num_edges, 2))
                edge_attr[:, 1] = 1.0
                view_data.edge_attr = edge_attr
                
                # 图级标签 y: 标记这是第几个视图 (作为 Diffusion 的 Condition)
                view_data.y = F.one_hot(torch.tensor([v_idx]), num_classes=len(self.metapaths)).float()
                
                # 保存溯源索引，方便后期联查
                view_data.parent_hetero_idx = sub_idx
                
                multiview_list.append(view_data)

        data, slices = self.collate(multiview_list)
        idx = {'train': 0, 'val': 1, 'test': 2}[self.stage]
        torch.save((data, slices), self.processed_paths[idx])

class YelpChiHeteroDataset(InMemoryDataset):
    def __init__(self, stage, root, original_data=None, num_parts=100, transform=None, pre_transform=None):
        self.stage = stage
        self.original_data = original_data
        self.num_parts = num_parts
        super().__init__(root, transform, pre_transform)
        
        # 加载对应的异构切片文件
        idx = {'train': 0, 'val': 1, 'test': 2}[stage]
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def processed_file_names(self):
        return ['hetero_subgraphs_train.pt', 'hetero_subgraphs_val.pt', 'hetero_subgraphs_test.pt']

    def process(self):
        if self.original_data is None:
            raise ValueError("First run requires 'original_data' to partition the graph.")

        print(f"Partitioning original hetero-graph into {self.num_parts} parts...")
        # 1. 转同构切分
        homo_data = self.original_data.to_homogeneous()
        cluster_data = ClusterData(homo_data, num_parts=self.num_parts, recursive=False)
        loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)

        # 2. 还原异构并存入列表
        node_types, edge_types = self.original_data.node_types, self.original_data.edge_types
        all_hetero_subs = []
        for sub_homo in loader:
            sub_hetero = sub_homo.to_heterogeneous(node_types, edge_types)
            all_hetero_subs.append(sub_hetero)

        # 3. 划分
        random.seed(42)
        random.shuffle(all_hetero_subs)
        n = len(all_hetero_subs)
        train_n, val_n = int(n * 0.8), int(n * 0.1)
        
        splits = [
            all_hetero_subs[:train_n],
            all_hetero_subs[train_n : train_n + val_n],
            all_hetero_subs[train_n + val_n :]
        ]

        # 4. 分别保存三个文件
        for i, data_list in enumerate(splits):
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])


class YelpChihDataModule(AbstractDataModule):
    def __init__(self, cfg, hetero_graph=None):
        self.cfg = cfg
        self.hetero_graph = hetero_graph
        self.root = cfg.dataset.root
        self.metapaths = cfg.dataset.metapaths
        self.target = cfg.dataset.target
        self.num_parts = cfg.dataset.num_parts
        # 构造异构图数据集
        train_hetero = YelpChiHeteroDataset('train', self.root, self.hetero_graph, self.num_parts)
        val_hetero = YelpChiHeteroDataset('val', self.root, self.hetero_graph, self.num_parts)
        test_hetero = YelpChiHeteroDataset('test', self.root, self.hetero_graph, self.num_parts)
        self.hetero_datasets = {
            'train': train_hetero,
            'val': val_hetero,
            'test': test_hetero
        }
        # 根据异构图构造同构图数据集
        datasets = {
            'train': YelpChiMultiviewDataset('train', self.root, train_hetero, self.metapaths, self.target),
            'val': YelpChiMultiviewDataset('val', self.root, val_hetero, self.metapaths, self.target),
            'test': YelpChiMultiviewDataset('test', self.root, test_hetero, self.metapaths, self.target)
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset
    
    def __getitem__(self, item):
        return self.inner[item]

    def node_types(self):
        example_batch = next(iter(self.train_dataloader()))
        # 【异构情况】
        # 此时“节点类型”通常指不同的实体（如 User, Review）
        node_types_list = list(example_batch.x_dict.keys())
        num_classes = len(node_types_list)
        counts = torch.zeros(num_classes, dtype=torch.float)
        
        for data in self.train_dataloader():
            for i, n_type in enumerate(node_types_list):
                # 统计该 Batch 中每种类型的节点数量
                counts[i] += data[n_type].num_nodes
                
        # 2. 归一化，得到概率分布 (例如 [0.9, 0.1] 表示 90% 是正常节点)
        return counts / counts.sum()
    
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

class YelpChiDatasetInfos(AbstractDatasetInfos):
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
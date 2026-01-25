import os
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.loader import ClusterData
from src.DiGress.src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.DiGress.src import utils

from src.utils.graphbuilder import extract_view_by_transform

class YelpChiMultiviewSubgraphDataset(InMemoryDataset):
    def __init__(self, stage, root, original_data, metapaths, target, is_hetero = False, num_parts=100, transform=None, pre_transform=None):
        self.stage = stage
        self.original_data = original_data
        self.num_parts = num_parts
        self.is_hetero = is_hetero
        self.metapaths = metapaths
        self.target = target
        # 映射 stage 到文件索引
        self.file_idx = {'train': 0, 'val': 1, 'test': 2}[stage]
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def processed_file_names(self):
        return ['train_subgraphs.pt', 'val_subgraphs.pt', 'test_subgraphs.pt']

    def process(self):
        if self.original_data is None:
            raise ValueError("First run requires 'original_data' to partition the graph.")

        data = self.original_data
        # 1. 转换为同构图进行切分
        homo_data = data.to_homogeneous()
        
        # 确保 node_type 是 Tensor (修复之前讨论的那个 bug)
        if hasattr(homo_data, 'node_type') and not isinstance(homo_data.node_type, torch.Tensor):
            homo_data.node_type = torch.tensor(homo_data.node_type)

        cluster_data = ClusterData(homo_data, num_parts=self.num_parts, recursive=False)
        
        all_subgraphs = []
        max_size = 0
        min_size = 1e9
        size_count = 0
        for i in range(self.num_parts):
            sub_homo = cluster_data[i]
            max_size = max(max_size, sub_homo.num_nodes)
            min_size = min(min_size, sub_homo.num_nodes)
            size_count += sub_homo.num_nodes
            # 使用节点级标签y代替节点属性x
            if hasattr(sub_homo, 'y') and sub_homo.y is not None:
                y_idx = sub_homo.y.long()
                sub_homo.x = F.one_hot(y_idx, num_classes=2).float()
                if sub_homo.x.dim() == 1:
                    sub_homo.x = sub_homo.x.unsqueeze(-1)
            else:
                raise ValueError(f"Subgraph {i} does not have 'y' labels!")
            # 再次检查子图 Tensor 状态
            if not isinstance(sub_homo.node_type, torch.Tensor):
                sub_homo.node_type = torch.tensor(sub_homo.node_type)
            # 边属性赋值
            num_edges = sub_homo.edge_index.size(1)
            # 创建一个 [num_edges, 2] 的浮点张量
            # 索引 0 位留给“无边”，索引 1 位给“有边”
            edge_attr = torch.zeros((num_edges, 2), dtype=torch.float)
            # 将所有存在的边标记为类别 1
            edge_attr[:, 1] = 1.0
            # 赋值回子图对象
            sub_homo.edge_attr = edge_attr
            # 子图级标签赋值
            y = torch.zeros([1, 0]).float()
            sub_homo.y = y
            sub_g = sub_homo
            all_subgraphs.append(sub_g)
        print(f"Max subgraph size: {max_size}, Min subgraph size: {min_size}, Average subgraph size: {size_count / self.num_parts}")

        # 2. 划分数据集
        random.seed(42) # 保证划分可复现
        random.shuffle(all_subgraphs)
        n = len(all_subgraphs)
        train_n, val_n = int(n * 0.8), int(n * 0.1)

        lists = [
            all_subgraphs[:train_n], 
            all_subgraphs[train_n : train_n + val_n], 
            all_subgraphs[train_n + val_n :]
        ]

        # 3. 保存
        for i, data_list in enumerate(lists):
            torch.save(self.collate(data_list), self.processed_paths[i])


class YelpChiMultiviewSubgraphDataModule(AbstractDataModule):
    def __init__(self, cfg, original_data=None):
        root_path = cfg.dataset.datadir
        num_parts = cfg.dataset.num_parts # 需在 config 中定义
        self.is_hetero = cfg.dataset.keep_hetero
        
        datasets = {
            'train': YelpChiMultiviewSubgraphDataset('train', root_path, original_data, is_hetero=cfg.dataset.keep_hetero, num_parts=num_parts),
            'val': YelpChiMultiviewSubgraphDataset('val', root_path, original_data, is_hetero=cfg.dataset.keep_hetero, num_parts=num_parts),
            'test': YelpChiMultiviewSubgraphDataset('test', root_path, original_data, is_hetero=cfg.dataset.keep_hetero, num_parts=num_parts)
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset
    
    def __getitem__(self, item):
        return self.inner[item]

    def node_types(self):
        
        if not self.is_hetero:
            # 【同构情况】
            return super().node_types()
        else:
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

class YelpChiMultiviewSubgraphDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.is_hetero = cfg.dataset.keep_hetero
        self.datamodule = datamodule

        self.name = 'HeteroCustom'
        # 获取基础统计信息
        # 注意：由于是异构图，这里的 max_n_nodes 是所有类型节点总和的最大值
        self.n_nodes = datamodule.node_counts() 
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        
        # 调用父类完成 DistributionNodes 等初始化
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
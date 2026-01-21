import os
import torch
import random
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import ClusterData
from src.DiGress.src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

class YelpChiSubgraphDataset(InMemoryDataset):
    def __init__(self, stage, root, original_data=None, num_parts=100, transform=None, pre_transform=None):
        self.stage = stage
        self.original_data = original_data
        self.num_parts = num_parts
        
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
        node_types, edge_types = data.node_types, data.edge_types
        
        # 1. 转换为同构图进行切分
        homo_data = data.to_homogeneous()
        
        # 确保 node_type 是 Tensor (修复之前讨论的那个 bug)
        if hasattr(homo_data, 'node_type') and not isinstance(homo_data.node_type, torch.Tensor):
            homo_data.node_type = torch.tensor(homo_data.node_type)

        cluster_data = ClusterData(homo_data, num_parts=self.num_parts, recursive=False)
        
        all_subgraphs = []
        for i in range(self.num_parts):
            sub_homo = cluster_data[i]
            
            # 再次检查子图 Tensor 状态
            if not isinstance(sub_homo.node_type, torch.Tensor):
                sub_homo.node_type = torch.tensor(sub_homo.node_type)
            
            # 还原为异构图
            sub_hetero = sub_homo.to_heterogeneous(
                node_type_names=node_types,
                edge_type_names=edge_types
            )
            
            # # 赋予图标签 (根据你的任务修改，这里假设取第一个节点类型的标签)
            # main_type = node_types[0]
            # if main_type in sub_hetero.node_types and hasattr(sub_hetero[main_type], 'y'):
            #     sub_hetero.y = sub_hetero[main_type].y[0].view(1, -1)
            all_subgraphs.append(sub_hetero)

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


class YelpChiSubgraphDataModule(AbstractDataModule):
    def __init__(self, cfg, original_data=None):
        root_path = cfg.dataset.datadir
        num_parts = cfg.dataset.num_parts # 需在 config 中定义
        
        datasets = {
            'train': YelpChiSubgraphDataset('train', root_path, original_data, num_parts),
            'val': YelpChiSubgraphDataset('val', root_path, original_data, num_parts),
            'test': YelpChiSubgraphDataset('test', root_path, original_data, num_parts)
        }
        super().__init__(cfg, datasets)

    # 重写 node_types 统计逻辑以适应异构字典
    def node_types(self):
        # 找到所有节点类型的特征维度之和
        example = self.train_dataset[0]
        total_counts = None
        
        for data in self.train_dataloader():
            # 将 x_dict 中所有类型的特征按行求和聚合
            # 这里假设你要统计的是“所有节点中各特征出现的频率”
            current_counts = torch.cat([x for x in data.x_dict.values()], dim=0).sum(dim=0)
            if total_counts is None:
                total_counts = torch.zeros_like(current_counts)
            total_counts += current_counts
            
        return total_counts / total_counts.sum()

class YelpChiSubgraphDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg = None):
        self.name = 'HeteroCustom'
        # 获取基础统计信息
        # 注意：由于是异构图，这里的 max_n_nodes 是所有类型节点总和的最大值
        self.n_nodes = datamodule.node_counts() 
        self.node_types = datamodule.node_types()
        
        # 调用父类完成 DistributionNodes 等初始化
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        
        # 设置输入维度 (针对异构模型，你可能需要根据各个 dict 动态设置)
        example_batch = next(iter(datamodule.train_dataloader()))
        self.input_dims = {
            'node_types': {k: v.size(1) for k, v in example_batch.x_dict.items()},
            'edge_types': {k: v.size(1) for k, v in example_batch.edge_index_dict.items()}
        }
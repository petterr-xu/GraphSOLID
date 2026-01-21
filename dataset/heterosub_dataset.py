import os
import torch
import random
from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric.loader import ClusterData

class HeteroSubgraphDataset(InMemoryDataset):
    def __init__(self, root, original_data=None, num_parts=100, split='train', transform=None, pre_transform=None):
        """
        root: 数据保存的根目录
        original_data: 原始的大图 HeteroData (仅在第一次运行时需要)
        num_parts: METIS 切分的子图数量
        split: 'train', 'val', 或 'test'
        """
        self.original_data = original_data
        self.num_parts = num_parts
        self.split = split
        
        super().__init__(root, transform, pre_transform)
        
        # 根据 split 加载对应的数据文件
        path = self.processed_paths[self._get_split_idx()]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        # 如果原始数据是以文件形式存在的，可以写在这里
        return ['original_hetero_graph.pt']

    @property
    def processed_file_names(self):
        # 预期的三个处理后的文件名
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def _get_split_idx(self):
        # 辅助函数：根据 split 返回对应的索引
        mapping = {'train': 0, 'val': 1, 'test': 2}
        return mapping[self.split]

    def process(self):
        if self.original_data is None:
            raise ValueError("第一次运行 process 时必须提供 original_data 参数来生成子图。")

        data = self.original_data
        print(f"正在进行 METIS 切分（共 {self.num_parts} 份）...")

        # 1. 转换为同构图进行切分
        node_types, edge_types = data.node_types, data.edge_types
        homo_data = data.to_homogeneous()
        
        cluster_data = ClusterData(homo_data, num_parts=self.num_parts, recursive=False)
        
        # 2. 还原异构子图并收集
        all_subgraphs = []
        for i in range(self.num_parts):
            sub_homo = cluster_data[i]
            if hasattr(sub_homo, 'node_type') and not isinstance(sub_homo.node_type, torch.Tensor):
                sub_homo.node_type = torch.tensor(sub_homo.node_type)
            if hasattr(sub_homo, 'edge_type') and not isinstance(sub_homo.edge_type, torch.Tensor):
                sub_homo.edge_type = torch.tensor(sub_homo.edge_type)
            # 还原为异构
            sub_hetero = sub_homo.to_heterogeneous(
                node_type_names=node_types,
                edge_type_names=edge_types
            )
            
            # --- 这里添加你的图标签逻辑 ---
            all_subgraphs.append(sub_hetero)

        # 3. 随机划分数据集
        random.shuffle(all_subgraphs)
        n = len(all_subgraphs)
        train_n = int(n * 0.8)
        val_n = int(n * 0.1)

        train_list = all_subgraphs[:train_n]
        val_list = all_subgraphs[train_n : train_n + val_n]
        test_list = all_subgraphs[train_n + val_n :]

        # 4. 分别保存三个文件
        lists = [train_list, val_list, test_list]
        for i, data_list in enumerate(lists):
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])
            print(f"保存完毕: {self.processed_file_names[i]}, 包含 {len(data_list)} 个子图")

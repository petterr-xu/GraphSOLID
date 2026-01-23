import os
import torch
import random
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import ClusterData
from src.DiGress.src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.DiGress.src import utils

class YelpChiSubgraphDataset(InMemoryDataset):
    def __init__(self, stage, root, original_data, is_hetero = False, num_parts=100, transform=None, pre_transform=None):
        self.stage = stage
        self.original_data = original_data
        self.num_parts = num_parts
        self.is_hetero = is_hetero
        
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
            if self.is_hetero:
                sub_g = sub_homo.to_heterogeneous(
                    node_type_names=node_types,
                    edge_type_names=edge_types
                )
            else:
                sub_g = sub_homo
            # # 赋予图标签 (根据你的任务修改，这里假设取第一个节点类型的标签)
            # main_type = node_types[0]
            # if main_type in sub_hetero.node_types and hasattr(sub_hetero[main_type], 'y'):
            #     sub_hetero.y = sub_hetero[main_type].y[0].view(1, -1)
            all_subgraphs.append(sub_g)

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
        self.is_hetero = cfg.dataset.keep_hetero
        
        datasets = {
            'train': YelpChiSubgraphDataset('train', root_path, original_data, is_hetero=cfg.dataset.keep_hetero, num_parts=num_parts),
            'val': YelpChiSubgraphDataset('val', root_path, original_data, is_hetero=cfg.dataset.keep_hetero, num_parts=num_parts),
            'test': YelpChiSubgraphDataset('test', root_path, original_data, is_hetero=cfg.dataset.keep_hetero, num_parts=num_parts)
        }
        super().__init__(cfg, datasets)

    def node_types(self):
        # 1. 确定类别总数
        # 如果是同构图，类别数通常是 y 的最大值 + 1
        # 如果是异构图，类别数就是节点类型的数量
        example_batch = next(iter(self.train_dataloader()))
        
        if not self.is_hetero:
            # 【同构情况】
            # 假设 y 是节点级的标签索引 [num_nodes]
            num_classes = int(example_batch.y.max()) + 1
            counts = torch.zeros(num_classes, dtype=torch.float)
            
            for data in self.train_dataloader():
                # 使用 bincount 统计每个标签出现的次数
                # 注意：确保 y 是 long 类型
                node_y = data.y.long()
                current_counts = torch.bincount(node_y, minlength=num_classes)
                counts += current_counts
                
        else:
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

class YelpChiSubgraphDatasetInfos(AbstractDatasetInfos):
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

    # def compute_input_output_dims(self, datamodule, extra_features, domain_features):
    #     # 设置输入维度 (针对异构模型，你可能需要根据各个 dict 动态设置)
    #     example_batch = next(iter(datamodule.train_dataloader()))
    #     if self.is_hetero:
    #         # 异构模式：返回各个类型的特征维度字典
    #         self.input_dims = {
    #             'X': {k: v.size(1) for k, v in example_batch.x_dict.items()},
    #             # 边特征维度：如果是异构，通常 edge_attr 也会分散在 edge_attr_dict 中
    #             'E': {k: v.size(1) for k, v in example_batch.edge_attr_dict.items()} if hasattr(example_batch, 'edge_attr_dict') else {},
    #             # y 维度保持框架要求的格式：原始维度 + 1 (时间步条件)
    #             'y': example_batch.y.size(1) + 1
    #         }
    #         self.output_dims = {'X': None, 'E': None, 'y': 0}

    #     else:
    #         example_batch = next(iter(datamodule.train_dataloader()))
    #         ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
    #                                             example_batch.batch)
    #         example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

    #         self.input_dims = {'X': example_batch['x'].size(1),
    #                         'E': example_batch['edge_attr'].size(1),
    #                         'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning
    #         ex_extra_feat = extra_features(example_data)
    #         self.input_dims['X'] += ex_dense.X.size(-1)
    #         self.input_dims['E'] += ex_extra_feat.E.size(-1)
    #         self.input_dims['y'] += ex_extra_feat.y.size(-1)

    #         ex_extra_molecular_feat = domain_features(example_data)
    #         self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
    #         self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
    #         self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

    #         self.output_dims = {'X': example_batch['x'].size(1),
    #                             'E': example_batch['edge_attr'].size(1),
    #                             'y': 0}
    #         # 同构模式：适配标准 Data 对象属性
    #         # 使用 getattr 安全获取属性，并检查是否为 None
    #         x_attr = getattr(example_batch, 'x', None)
    #         e_attr = getattr(example_batch, 'edge_attr', None)
    #         # y_attr = getattr(example_batch, 'y', None)
            
    #         self.input_dims = {
    #             'X': x_attr.size(1) if x_attr is not None else 0,
    #             'E': e_attr.size(1) if e_attr is not None else 0,
    #             'y': 0
    #         }
    #         self.output_dims = {
    #             'X': self.input_dims['X'],
    #             'E': self.input_dims['E'],
    #             'y': 0
    #         }
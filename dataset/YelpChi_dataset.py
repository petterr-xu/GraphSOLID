class MinimalDatasetInfos:
    def __init__(self, num_node_types, num_edge_types):
        """
        Args:
            num_node_types: 你的异构图转换后总共有多少种节点类别 (X的维度)
            num_edge_types: 总共有多少种边类别 (E的维度)
        """
        # 1. 定义输入和输出维度 (X: 节点, E: 边, y: 全局特征)
        # 通常输入维度和输出维度是一样的
        self.input_dims = {'X': num_node_types, 'E': num_edge_types, 'y': 0}
        self.output_dims = {'X': num_node_types, 'E': num_edge_types, 'y': 0}
        
        # 2. 节点数量分布 (Training 阶段其实用不到，但 init 会读取)
        # 我们可以给个 None 或者随便给个对象，只要不运行 sample_batch 就不会报错
        self.nodes_dist = None
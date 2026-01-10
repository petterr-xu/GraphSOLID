"""
异构图到同构图的转换工具，用于将 HeteroData 转换为 DiGress 可用的格式
"""
from torch_geometric.data import HeteroData
from src.utils.hetero_to_homo import hetero_to_homo_for_digress, extract_metapath_view


def convert_hetero_to_digress_format(
    hetero_data: HeteroData,
    target_node: str,
    metapath=None,
    use_node_types: bool = True,
    use_edge_types: bool = True,
    device=None
):
    """
    将异构图转换为 DiGress 可以使用的同构图格式
    
    这是一个便捷函数，用于快速转换异构图数据
    
    Args:
        hetero_data: PyG HeteroData 对象
        target_node: 目标节点类型（例如 'review', 'user'）
        metapath: 可选的元路径列表，例如 [('user', 'buys', 'item')]
        use_node_types: 是否使用 one-hot 编码节点类型
        use_edge_types: 是否使用 one-hot 编码边类型
        device: 设备
        
    Returns:
        homo_data: PyG Data 对象（同构图，DiGress 格式）
        metadata: 包含转换信息的字典
        
    Example:
        >>> from utils.hetero_dataset_util import GraphDataLoader
        >>> from hetero_solid import convert_hetero_to_digress_format
        >>> 
        >>> # 加载异构图
        >>> context = GraphDataLoader.load_from_config('config.json', 'data.mat')
        >>> hetero_data = context.g
        >>> target_node = context.target_node
        >>> 
        >>> # 转换为 DiGress 格式
        >>> homo_data, metadata = convert_hetero_to_digress_format(
        ...     hetero_data=hetero_data,
        ...     target_node=target_node,
        ...     use_node_types=True,
        ...     use_edge_types=True
        ... )
        >>> 
        >>> # 现在 homo_data 可以用于 DiGress 训练/生成
        >>> print(f"节点数: {homo_data.x.shape[0]}")
        >>> print(f"边数: {homo_data.edge_index.shape[1]}")
    """
    return hetero_to_homo_for_digress(
        hetero_data=hetero_data,
        target_node=target_node,
        metapath=metapath,
        use_node_types=use_node_types,
        use_edge_types=use_edge_types,
        device=device
    )


def convert_with_metapath(
    hetero_data: HeteroData,
    metapath: list,
    target_node_type: str,
    device=None
):
    """
    使用元路径提取同构图视图
    
    Args:
        hetero_data: PyG HeteroData 对象
        metapath: 元路径列表，例如 [('user', 'buys', 'item'), ('item', 'bought_by', 'user')]
        target_node_type: 目标节点类型（元路径视图中的节点类型）
        device: 设备
        
    Returns:
        homo_data: PyG Data 对象
        metadata: 包含转换信息的字典
    """
    return extract_metapath_view(
        hetero_data=hetero_data,
        metapath=metapath,
        target_node_type=target_node_type,
        device=device
    )


# 兼容性：保留原有接口（如果存在）
__all__ = [
    'convert_hetero_to_digress_format',
    'convert_with_metapath',
]

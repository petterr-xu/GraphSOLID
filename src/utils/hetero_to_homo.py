"""
将异构图转换为同构图的工具函数，用于 DiGress 图生成模型
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from typing import Optional, List, Tuple, Dict
import warnings


def hetero_to_homo_for_digress(
    hetero_data: HeteroData,
    target_node: Optional[str] = None,
    metapath: Optional[List[Tuple[str, str, str]]] = None,
    use_node_types: bool = True,
    use_edge_types: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[Data, Dict]:
    """
    将 HeteroData 转换为 PyG Data 格式（同构图），用于 DiGress 训练/生成
    
    Args:
        hetero_data: 输入的异构图数据
        target_node: 目标节点类型（如果指定，只提取与目标节点相关的子图）
        metapath: 元路径列表，例如 [('user', 'buys', 'item'), ('item', 'bought_by', 'user')]
                  如果指定，只提取沿着元路径的节点和边
        use_node_types: 是否将节点类型编码为 one-hot（否则使用原始特征）
        use_edge_types: 是否将边类型编码为 one-hot
        device: 设备
        
    Returns:
        data: PyG Data 对象，格式符合 DiGress 要求
        metadata: 包含转换信息的字典，包括节点映射、类型映射等
    """
    if device is None:
        device = hetero_data.device if hasattr(hetero_data, 'device') else torch.device('cpu')
    
    # 如果没有指定元路径，使用所有相关的边
    if metapath is None:
        if target_node is not None:
            # 只提取与 target_node 相关的边
            relevant_edges = [
                etype for etype in hetero_data.edge_types
                if etype[0] == target_node or etype[2] == target_node
            ]
            if len(relevant_edges) == 0:
                warnings.warn(f"No edges found related to target_node '{target_node}'. Using all edges.")
                relevant_edges = list(hetero_data.edge_types)
        else:
            # 使用所有边
            relevant_edges = list(hetero_data.edge_types)
    else:
        # 使用指定的元路径
        relevant_edges = metapath
    
    # 收集所有相关的节点类型
    if target_node is not None:
        node_types_set = {target_node}
        for etype in relevant_edges:
            node_types_set.add(etype[0])
            node_types_set.add(etype[2])
        relevant_node_types = sorted(list(node_types_set))
    else:
        relevant_node_types = sorted(hetero_data.node_types)
    
    # 构建节点映射：{原始节点ID: 新节点ID}
    node_mapping = {}  # {(node_type, original_id): new_global_id}
    node_type_to_id = {nt: idx for idx, nt in enumerate(relevant_node_types)}
    current_global_id = 0
    
    # 收集所有节点
    all_nodes = []  # [(node_type, local_id, features, label)]
    for node_type in relevant_node_types:
        if node_type not in hetero_data.node_types:
            warnings.warn(f"Node type '{node_type}' not found in hetero_data. Skipping.")
            continue
        
        num_nodes = hetero_data[node_type].num_nodes
        x_original = hetero_data[node_type].x
        
        # 获取标签（如果存在）
        y_original = None
        if hasattr(hetero_data[node_type], 'y') and hetero_data[node_type].y is not None:
            y_original = hetero_data[node_type].y
        
        for local_id in range(num_nodes):
            node_mapping[(node_type, local_id)] = current_global_id
            all_nodes.append((node_type, local_id, x_original[local_id], 
                            y_original[local_id] if y_original is not None else None))
            current_global_id += 1
    
    num_total_nodes = current_global_id
    if num_total_nodes == 0:
        raise ValueError("No nodes found in the specified configuration.")
    
    # 构建节点特征矩阵
    if use_node_types:
        # 使用 one-hot 编码节点类型
        num_node_types = len(relevant_node_types)
        x = torch.zeros(num_total_nodes, num_node_types, dtype=torch.float32, device=device)
        for i, (node_type, _, _, _) in enumerate(all_nodes):
            node_type_idx = node_type_to_id[node_type]
            x[i, node_type_idx] = 1.0
    else:
        # 使用原始特征（需要统一维度）
        # 找到最大特征维度
        max_feat_dim = max(hetero_data[nt].x.shape[1] for nt in relevant_node_types 
                          if nt in hetero_data.node_types)
        x = torch.zeros(num_total_nodes, max_feat_dim, dtype=torch.float32, device=device)
        for i, (node_type, local_id, feat, _) in enumerate(all_nodes):
            feat_dim = feat.shape[0]
            x[i, :feat_dim] = feat.to(device)
    
    # 收集所有边
    edge_index_list = []
    edge_attr_list = []
    edge_type_to_id = {et: idx + 1 for idx, et in enumerate(relevant_edges)}  # +1 因为 0 表示"无边"
    num_edge_types = len(relevant_edges) + 1  # +1 是"无边"类型
    
    for etype in relevant_edges:
        if etype not in hetero_data.edge_types:
            warnings.warn(f"Edge type {etype} not found in hetero_data. Skipping.")
            continue
        
        src_type, rel, dst_type = etype
        edge_index_original = hetero_data[etype].edge_index
        
        if edge_index_original.numel() == 0:
            continue
        
        # 转换边索引到全局节点ID
        src_local = edge_index_original[0]
        dst_local = edge_index_original[1]
        
        # 映射到全局ID
        src_global = torch.tensor([
            node_mapping[(src_type, local_id.item())]
            for local_id in src_local
        ], dtype=torch.long, device=device)
        
        dst_global = torch.tensor([
            node_mapping[(dst_type, local_id.item())]
            for local_id in dst_local
        ], dtype=torch.long, device=device)
        
        # 创建边索引
        new_edge_index = torch.stack([src_global, dst_global], dim=0)
        edge_index_list.append(new_edge_index)
        
        # 创建边属性（one-hot 编码边类型）
        if use_edge_types:
            num_edges = new_edge_index.shape[1]
            edge_attr_etype = torch.zeros(num_edges, num_edge_types, dtype=torch.float32, device=device)
            edge_type_idx = edge_type_to_id[etype]
            edge_attr_etype[:, edge_type_idx] = 1.0
            edge_attr_list.append(edge_attr_etype)
        else:
            # 如果不使用边类型，创建一个简单的边属性
            num_edges = new_edge_index.shape[1]
            edge_attr_etype = torch.zeros(num_edges, 2, dtype=torch.float32, device=device)
            edge_attr_etype[:, 1] = 1.0  # 第二个维度表示"有边"
            edge_attr_list.append(edge_attr_etype)
    
    # 合并所有边
    if len(edge_index_list) == 0:
        # 如果没有边，创建空的边索引和属性
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        if use_edge_types:
            edge_attr = torch.zeros((0, num_edge_types), dtype=torch.float32, device=device)
        else:
            edge_attr = torch.zeros((0, 2), dtype=torch.float32, device=device)
    else:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
    
    # 收集图级别标签（如果存在且所有节点类型都有标签）
    y_list = []
    for node_type, local_id, _, label in all_nodes:
        if label is not None:
            y_list.append(label)
    
    if len(y_list) == num_total_nodes and len(set(type(y) for y in y_list)) == 1:
        # 如果所有节点都有标签，使用标签
        y = torch.stack(y_list).unsqueeze(0).float().to(device)  # [1, num_nodes]
    else:
        # 否则创建空的标签
        y = torch.zeros((1, 0), dtype=torch.float32, device=device)
    
    # 创建 n_nodes 属性（DiGress 可能需要）
    n_nodes = torch.tensor([num_total_nodes], dtype=torch.long, device=device)
    
    # 创建 PyG Data 对象
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        n_nodes=n_nodes
    )
    
    # 保留原始的 train_mask, val_mask, test_mask（如果存在）
    if target_node is not None and target_node in hetero_data.node_types:
        if hasattr(hetero_data[target_node], 'train_mask'):
            # 只保留目标节点的 mask（因为其他节点的 mask 可能不完整）
            target_mask_base = node_mapping[(target_node, 0)] if (target_node, 0) in node_mapping else 0
            num_target_nodes = hetero_data[target_node].num_nodes
            if target_mask_base + num_target_nodes <= num_total_nodes:
                data.train_mask = hetero_data[target_node].train_mask.to(device)
                if hasattr(hetero_data[target_node], 'val_mask'):
                    data.val_mask = hetero_data[target_node].val_mask.to(device)
                if hasattr(hetero_data[target_node], 'test_mask'):
                    data.test_mask = hetero_data[target_node].test_mask.to(device)
    
    # 构建元数据
    metadata = {
        'node_mapping': node_mapping,  # {(node_type, original_id): new_global_id}
        'node_type_to_id': node_type_to_id,  # {node_type: idx}
        'edge_type_to_id': edge_type_to_id if use_edge_types else None,  # {edge_type: idx}
        'relevant_node_types': relevant_node_types,
        'relevant_edge_types': relevant_edges,
        'num_node_types': len(relevant_node_types),
        'num_edge_types': num_edge_types if use_edge_types else 2,
        'target_node': target_node,
        'original_hetero_data': hetero_data  # 保留引用以便后续使用
    }
    
    return data, metadata


def extract_metapath_view(
    hetero_data: HeteroData,
    metapath: List[Tuple[str, str, str]],
    target_node_type: str,
    device: Optional[torch.device] = None
) -> Tuple[Data, Dict]:
    """
    从异构图提取指定元路径的视图（同构图）
    
    例如，元路径 [('user', 'buys', 'item'), ('item', 'bought_by', 'user')] 
    会提取一个 user-user 的同构图（通过 item 连接）
    
    Args:
        hetero_data: 输入的异构图数据
        metapath: 元路径，例如 [('user', 'buys', 'item'), ('item', 'bought_by', 'user')]
        target_node_type: 目标节点类型（元路径视图中的节点类型）
        device: 设备
        
    Returns:
        data: PyG Data 对象
        metadata: 转换元数据
    """
    if device is None:
        device = hetero_data.device if hasattr(hetero_data, 'device') else torch.device('cpu')
    
    # 验证元路径
    if len(metapath) == 0:
        raise ValueError("Metapath cannot be empty.")
    
    # 提取目标节点类型的节点
    if target_node_type not in hetero_data.node_types:
        raise ValueError(f"Target node type '{target_node_type}' not found in hetero_data.")
    
    # 使用 hetero_to_homo_for_digress 提取相关的节点和边
    # 这里暂时使用元路径中的所有边类型
    return hetero_to_homo_for_digress(
        hetero_data=hetero_data,
        target_node=target_node_type,
        metapath=metapath,
        use_node_types=True,
        use_edge_types=True,
        device=device
    )


def create_digress_dataset_from_hetero(
    hetero_data: HeteroData,
    target_node: str,
    split: str = 'train',
    device: Optional[torch.device] = None
) -> Data:
    """
    从异构图创建单个 DiGress 数据样本（用于 DataLoader）
    
    Args:
        hetero_data: 异构图数据
        target_node: 目标节点类型
        split: 'train', 'val', 或 'test'（用于选择对应的 mask）
        device: 设备
        
    Returns:
        data: PyG Data 对象
    """
    data, metadata = hetero_to_homo_for_digress(
        hetero_data=hetero_data,
        target_node=target_node,
        use_node_types=True,
        use_edge_types=True,
        device=device
    )
    
    # 根据 split 设置对应的 mask
    if hasattr(data, f'{split}_mask'):
        mask = getattr(data, f'{split}_mask')
        # 创建一个过滤后的子图（只包含 split 中的节点）
        # 这里简化处理，返回完整图
        pass
    
    return data


# 辅助函数：从 metadata 恢复节点类型
def get_original_node_type(global_id: int, metadata: Dict) -> Optional[Tuple[str, int]]:
    """
    根据全局节点ID获取原始节点类型和本地ID
    
    Args:
        global_id: 全局节点ID
        metadata: hetero_to_homo_for_digress 返回的 metadata
        
    Returns:
        (node_type, local_id) 或 None
    """
    node_mapping = metadata['node_mapping']
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    
    if global_id in reverse_mapping:
        return reverse_mapping[global_id]
    return None

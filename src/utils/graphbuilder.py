import torch
import numpy as np
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData


def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask, max_n=500):
    # Sort from major to minor
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    MAX = min(sorted_n_data[0].item(), max_n)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(MAX * np.power(mu, i), sorted_n_data[i], max_n)))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1 and MAX >= sorted_n_data[0].item():
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    # Compute the number of nodes which would be removed for each class
    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask),device=train_mask.device)
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            # Compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            # Accumulation does not be problem since
            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.to('cpu').numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, node_mask, edge_mask

def make_hetero_longtailed_data_remove(data, target_node, n_data, n_cls, ratio, train_mask, max_n=500):
    """
    针对异构图目标节点的长尾化处理
    :data: HeteroData 对象
    :target_node: 字符串，如 'review'
    :n_data: 初始各类别样本数列表
    :n_cls: 类别数
    :ratio: 不平衡率 (Imbalance Ratio)
    :train_mask: 目标节点的训练掩码
    """
    device = train_mask.device
    label = data[target_node].y
    
    # 1. 计算各类别目标样本数 (LT 规则)
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    MAX = min(sorted_n_data[0].item(), max_n)
    
    # 计算逆索引
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i

    mu = np.power(1/ratio, 1/(n_cls - 1))
    class_num_list = []
    n_round = []
    for i in range(n_cls):
        class_num_list.append(int(min(MAX * np.power(mu, i), sorted_n_data[i], max_n)))
        n_round.append(1 if (i < 1 and MAX >= sorted_n_data[0].item()) else 10)
    
    class_num_list = np.array(class_num_list)[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    # 2. 准备移除逻辑
    remove_class_num_list = [n_data[i].item() - class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    
    index_list = torch.arange(len(train_mask), device=device)
    cls_idx_list = [(index_list[(label == i) & train_mask]) for i in range(n_cls)]

    # 3. 核心循环：基于度数移除节点
    for i in indices.numpy():
        for r in range(1, n_round[i] + 1):
            # 当前有效的节点掩码
            current_node_mask = torch.ones(label.size(0), dtype=torch.bool, device=device)
            current_node_mask[sum(remove_idx_list, [])] = False

            # 计算目标节点在所有关系中的总度数
            total_degree = torch.zeros(label.size(0), device=device)
            
            for edge_type in data.edge_types:
                src, rel, dst = edge_type
                edge_index = data[edge_type].edge_index
                
                # 情况 A: 目标节点作为源节点
                if src == target_node:
                    # 只有当目标节点和另一端的节点都还存在时，这条边才计入度数
                    # (假设非目标节点的 node_mask 始终为 True，或者你也想过滤它们)
                    valid_edge_mask = current_node_mask[edge_index[0]]
                    total_degree += scatter_add(torch.ones_like(edge_index[0][valid_edge_mask]), 
                                                edge_index[0][valid_edge_mask], 
                                                dim_size=label.size(0))
                
                # 情况 B: 目标节点作为目标节点 (处理入度)
                if dst == target_node:
                    valid_edge_mask = current_node_mask[edge_index[1]]
                    total_degree += scatter_add(torch.ones_like(edge_index[1][valid_edge_mask]), 
                                                edge_index[1][valid_edge_mask], 
                                                dim_size=label.size(0))

            # 选出当前类目中度数最低的节点进行移除
            target_cls_degree = total_degree[cls_idx_list[i]]
            num_to_remove = (r * remove_class_num_list[i]) // n_round[i]
            
            if num_to_remove > 0:
                _, remove_idx_relative = torch.topk(target_cls_degree, num_to_remove, largest=False)
                remove_idx = cls_idx_list[i][remove_idx_relative]
                remove_idx_list[i] = list(remove_idx.cpu().numpy())

    # 4. 生成最终结果
    final_node_mask = torch.ones(label.size(0), dtype=torch.bool, device=device)
    final_node_mask[sum(remove_idx_list, [])] = False

    # 生成各关系的 edge_mask_dict
    edge_mask_dict = {}
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        edge_index = data[edge_type].edge_index
        
        # 只有当边两端的节点都在保留列表中时，边才保留
        m_src = final_node_mask if src == target_node else torch.ones(data[src].num_nodes, dtype=torch.bool, device=device)
        m_dst = final_node_mask if dst == target_node else torch.ones(data[dst].num_nodes, dtype=torch.bool, device=device)
        
        edge_mask_dict[edge_type] = m_src[edge_index[0]] & m_dst[edge_index[1]]

    new_train_mask = final_node_mask & train_mask
    
    return list(class_num_list), new_train_mask, final_node_mask, edge_mask_dict

def extract_view_by_transform(pyg_graph, metapath_steps, target_node, weighted=False):
    """
    底层逻辑：自动适配单跳边和多跳元路径。
    """
    # 1. 格式标准化：确保是 List[Tuple]
    if isinstance(metapath_steps[0], str): 
        # 处理 ["review", "rur", "review"] 这种情况
        actual_path = [tuple(metapath_steps)]
    else:
        # 处理 [("paper", "to", "author"), ("author", "to", "paper")] 这种情况
        actual_path = [tuple(step) for step in metapath_steps]

    # --- 核心分流逻辑 ---
    
    # 情况 A：单跳边 (YelpChi 现状)
    if len(actual_path) == 1:
        edge_type = actual_path[0]
        # 直接提取 edge_index，避开 AddMetaPaths 的 len>=2 限制
        # 如果 HeteroData 中没有完整三元组 key，则尝试用关系名 key
        if edge_type in pyg_graph.edge_types:
            edge_index = pyg_graph[edge_type].edge_index
        else:
            edge_index = pyg_graph[edge_type[1]].edge_index
            
        view_data = Data(
            x=pyg_graph[target_node].x,
            y=pyg_graph[target_node].y,
            edge_index=edge_index
        )
        if weighted: # 单跳边默认权重为 1
            view_data.edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    # 情况 B：多跳路径 (需要矩阵乘法压缩)
    else:
        # 使用官方 Transform
        transform = T.AddMetaPaths(
            metapaths=[actual_path], 
            drop_orig_edge_types=True,
            weighted=weighted  # 你可以开启这个选项来获取路径数量作为权重
        )
        temp_hetero = transform(pyg_graph.clone())
        
        # 提取新生成的元路径边
        new_edge_type = temp_hetero.edge_types[0]
        view_data = Data(
            x=temp_hetero[target_node].x, 
            y=temp_hetero[target_node].y,
            edge_index=temp_hetero[new_edge_type].edge_index
        )
        if weighted and hasattr(temp_hetero[new_edge_type], 'edge_weight'):
            view_data.edge_weight = temp_hetero[new_edge_type].edge_weight

    # 2. 通用后处理：继承掩码
    for mask in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(pyg_graph[target_node], mask):
            setattr(view_data, mask, getattr(pyg_graph[target_node], mask))
            
    return view_data

def extract_view_by_matrix(pyg_graph, metapath_steps, target_node):
    
    """
    pyg_graph: 原始异构图
    metapath_steps: 元路径列表，例如 [('review', 'user'), ('user', 'review')]
    target_node: 投影后的同构图节点类型
    """
    # 1. 格式标准化：确保 metapath_steps 是一个包含完整边定义的列表
    # 如果是 ['review', 'rur', 'review'] -> 变成 [('review', 'rur', 'review')]
    if isinstance(metapath_steps[0], str) and len(metapath_steps) == 3:
        metapath_steps = [tuple(metapath_steps)]
    else:
        # 如果是 [[...], [...]] -> 确保内部是元组
        metapath_steps = [tuple(step) for step in metapath_steps]
    adj = None
    device = pyg_graph[target_node].x.device
    
    # 2. 逐跳进行矩阵乘法
    for etype in metapath_steps:
        # 此时 etype 一定是 ('src', 'rel', 'dst') 元组，PyG 会准确返回 EdgeStorage
        if etype not in pyg_graph.edge_types:
            # 兼容性处理：如果元组匹配失败，尝试使用关系名字符串 'rur'
            rel_name = etype[1]
            edge_index = pyg_graph[rel_name].edge_index
        else:
            edge_index = pyg_graph[etype].edge_index
            
        size = (pyg_graph[etype[0]].num_nodes, pyg_graph[etype[2]].num_nodes)
        
        # 构造稀疏张量
        curr_adj = SparseTensor(
            row=edge_index[0], col=edge_index[1], 
            sparse_sizes=size
        ).to(device)
        
        if adj is None:
            adj = curr_adj
        else:
            adj = adj.matmul(curr_adj)
    
    # 3. 提取结果
    row, col, value = adj.coo()
    
    view_data = Data(
        x=pyg_graph[target_node].x, 
        y=pyg_graph[target_node].y,
        edge_index=torch.stack([row, col], dim=0),
        edge_attr=value # 存储路径计数
    )
    
    # 继承掩码
    for mask in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(pyg_graph[target_node], mask):
            setattr(view_data, mask, getattr(pyg_graph[target_node], mask))
            
    return view_data
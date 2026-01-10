"""
异构图转同构图的使用示例
"""
import torch
from torch_geometric.data import HeteroData
from hetero_to_homo import hetero_to_homo_for_digress, extract_metapath_view


def example_basic_conversion():
    """
    基本转换示例：将异构图转换为同构图
    """
    # 创建一个示例异构图
    hetero_data = HeteroData()
    
    # 添加节点类型 'review'
    num_reviews = 100
    hetero_data['review'].x = torch.randn(num_reviews, 10)  # 10维特征
    hetero_data['review'].y = torch.randint(0, 2, (num_reviews,))  # 二分类标签
    hetero_data['review'].train_mask = torch.ones(num_reviews, dtype=torch.bool)
    hetero_data['review'].val_mask = torch.zeros(num_reviews, dtype=torch.bool)
    hetero_data['review'].test_mask = torch.zeros(num_reviews, dtype=torch.bool)
    
    # 添加边类型 'rur' (review-user-review)
    edge_index_rur = torch.randint(0, num_reviews, (2, 50), dtype=torch.long)
    hetero_data['review', 'rur', 'review'].edge_index = edge_index_rur
    
    # 添加边类型 'rsr' (review-shop-review)
    edge_index_rsr = torch.randint(0, num_reviews, (2, 50), dtype=torch.long)
    hetero_data['review', 'rsr', 'review'].edge_index = edge_index_rsr
    
    print(f"异构图信息:")
    print(f"  节点类型: {hetero_data.node_types}")
    print(f"  边类型: {hetero_data.edge_types}")
    print(f"  Review 节点数: {hetero_data['review'].num_nodes}")
    
    # 转换为同构图（针对 review 节点）
    homo_data, metadata = hetero_to_homo_for_digress(
        hetero_data=hetero_data,
        target_node='review',
        use_node_types=True,
        use_edge_types=True
    )
    
    print(f"\n同构图信息:")
    print(f"  节点数: {homo_data.x.shape[0]}")
    print(f"  节点特征维度: {homo_data.x.shape[1]} (one-hot 节点类型)")
    print(f"  边数: {homo_data.edge_index.shape[1]}")
    print(f"  边特征维度: {homo_data.edge_attr.shape[1]} (one-hot 边类型)")
    print(f"  图标签维度: {homo_data.y.shape}")
    
    print(f"\n元数据信息:")
    print(f"  相关节点类型: {metadata['relevant_node_types']}")
    print(f"  相关边类型: {len(metadata['relevant_edge_types'])} 种")
    print(f"  节点类型数: {metadata['num_node_types']}")
    print(f"  边类型数: {metadata['num_edge_types']}")
    
    # 验证转换正确性
    assert homo_data.x.shape[0] == num_reviews, "节点数应该匹配"
    assert homo_data.x.shape[1] == metadata['num_node_types'], "节点特征维度应该匹配节点类型数"
    assert homo_data.edge_attr.shape[0] == homo_data.edge_index.shape[1], "边数和边属性数应该匹配"
    
    print("\n✅ 转换成功！数据格式符合 DiGress 要求")
    
    return homo_data, metadata


def example_metapath_extraction():
    """
    元路径提取示例
    """
    # 创建包含多种节点类型的异构图
    hetero_data = HeteroData()
    
    # 添加 User 节点
    num_users = 50
    hetero_data['user'].x = torch.randn(num_users, 8)
    
    # 添加 Item 节点
    num_items = 30
    hetero_data['item'].x = torch.randn(num_items, 8)
    
    # 添加边：user 购买 item
    edge_index_buys = torch.stack([
        torch.randint(0, num_users, (20,)),
        torch.randint(0, num_items, (20,))
    ], dim=0)
    hetero_data['user', 'buys', 'item'].edge_index = edge_index_buys
    
    # 添加边：item 被 user 购买（反向关系，可选）
    # 这里假设是同一个关系，但在异构图中有方向性
    
    print(f"异构图信息:")
    print(f"  节点类型: {hetero_data.node_types}")
    print(f"  边类型: {hetero_data.edge_types}")
    
    # 定义元路径：user -> item -> user (通过共同购买的 item 连接)
    # 注意：这需要两步，但我们的实现会提取所有相关节点
    metapath = [('user', 'buys', 'item')]
    
    # 提取元路径视图（针对 user 节点）
    try:
        homo_data, metadata = extract_metapath_view(
            hetero_data=hetero_data,
            metapath=metapath,
            target_node_type='user'
        )
        
        print(f"\n元路径视图同构图信息:")
        print(f"  节点数: {homo_data.x.shape[0]}")
        print(f"  边数: {homo_data.edge_index.shape[1]}")
        print(f"  相关节点类型: {metadata['relevant_node_types']}")
        
    except Exception as e:
        print(f"⚠️  元路径提取遇到问题: {e}")
        print("   使用基本转换方式...")
        homo_data, metadata = hetero_to_homo_for_digress(
            hetero_data=hetero_data,
            target_node='user',
            use_node_types=True,
            use_edge_types=True
        )
    
    return homo_data, metadata


def example_with_digress_format():
    """
    展示如何将转换后的数据用于 DiGress
    """
    # 创建示例数据
    hetero_data = HeteroData()
    num_reviews = 100
    hetero_data['review'].x = torch.randn(num_reviews, 10)
    hetero_data['review', 'rur', 'review'].edge_index = torch.randint(0, num_reviews, (2, 50), dtype=torch.long)
    
    # 转换为 DiGress 格式
    homo_data, metadata = hetero_to_homo_for_digress(
        hetero_data=hetero_data,
        target_node='review',
        use_node_types=True,
        use_edge_types=True
    )
    
    print("DiGress 数据格式检查:")
    print(f"  ✅ x.shape = {homo_data.x.shape} (节点特征，one-hot 编码)")
    print(f"  ✅ edge_index.shape = {homo_data.edge_index.shape} (边索引)")
    print(f"  ✅ edge_attr.shape = {homo_data.edge_attr.shape} (边属性，one-hot 编码)")
    print(f"  ✅ y.shape = {homo_data.y.shape} (图级别标签)")
    print(f"  ✅ n_nodes = {homo_data.n_nodes.item()} (节点数)")
    
    # 检查数据是否符合 DiGress 要求
    assert homo_data.x.dtype == torch.float32, "节点特征应该是 float32"
    assert homo_data.edge_attr.dtype == torch.float32, "边属性应该是 float32"
    assert torch.allclose(homo_data.x.sum(dim=1), torch.ones(homo_data.x.shape[0])), \
        "节点特征应该是 one-hot 编码（每行和为1）"
    
    # 检查边属性（第一列应该是"无边"，其他列应该是边类型）
    if homo_data.edge_attr.shape[1] > 1:
        # 每行的和应该为1（one-hot）
        assert torch.allclose(homo_data.edge_attr.sum(dim=1), 
                            torch.ones(homo_data.edge_attr.shape[0])), \
            "边属性应该是 one-hot 编码"
    
    print("\n✅ 数据格式完全符合 DiGress 要求！")
    print("\n可以用于:")
    print("  1. 创建 PyG DataLoader")
    print("  2. 训练 DiGress 模型")
    print("  3. 使用 DiGress 进行图生成")
    
    return homo_data, metadata


if __name__ == "__main__":
    print("=" * 60)
    print("示例 1: 基本转换")
    print("=" * 60)
    example_basic_conversion()
    
    print("\n" + "=" * 60)
    print("示例 2: 元路径提取")
    print("=" * 60)
    example_metapath_extraction()
    
    print("\n" + "=" * 60)
    print("示例 3: DiGress 格式验证")
    print("=" * 60)
    example_with_digress_format()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

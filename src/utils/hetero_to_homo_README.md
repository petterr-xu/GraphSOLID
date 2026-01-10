# 异构图转同构图转换工具 (MVP)

这个模块提供了将 PyG 异构图 (`HeteroData`) 转换为同构图 (`Data`) 的工具函数，用于与 DiGress 图生成模型集成。

## 快速开始

### 基本用法

```python
from torch_geometric.data import HeteroData
from src.utils.hetero_to_homo import hetero_to_homo_for_digress

# 假设你有一个异构图数据
# hetero_data = ...  # 你的 HeteroData 对象
# target_node = 'review'  # 目标节点类型

# 转换为同构图
homo_data, metadata = hetero_to_homo_for_digress(
    hetero_data=hetero_data,
    target_node=target_node,
    use_node_types=True,  # 使用 one-hot 编码节点类型
    use_edge_types=True,  # 使用 one-hot 编码边类型
)

# 现在 homo_data 可以用于 DiGress
print(f"节点数: {homo_data.x.shape[0]}")
print(f"边数: {homo_data.edge_index.shape[1]}")
print(f"节点特征维度: {homo_data.x.shape[1]}")  # one-hot 节点类型
print(f"边特征维度: {homo_data.edge_attr.shape[1]}")  # one-hot 边类型
```

### 与 HeteroGraphContext 集成

```python
from src.utils.hetero_dataset_util import GraphDataLoader
from src.hetero_solid import convert_hetero_to_digress_format

# 加载异构图
context = GraphDataLoader.load_from_config('config.json', 'data.mat')
hetero_data = context.g
target_node = context.target_node

# 转换为 DiGress 格式
homo_data, metadata = convert_hetero_to_digress_format(
    hetero_data=hetero_data,
    target_node=target_node,
)
```

## API 文档

### `hetero_to_homo_for_digress()`

主要的转换函数。

**参数:**
- `hetero_data` (HeteroData): 输入的异构图数据
- `target_node` (str, optional): 目标节点类型。如果指定，只提取与目标节点相关的子图
- `metapath` (List[Tuple[str, str, str]], optional): 元路径列表。如果指定，只提取沿着元路径的节点和边
- `use_node_types` (bool, default=True): 是否将节点类型编码为 one-hot
- `use_edge_types` (bool, default=True): 是否将边类型编码为 one-hot
- `device` (torch.device, optional): 设备

**返回:**
- `data` (Data): PyG Data 对象，格式符合 DiGress 要求
- `metadata` (Dict): 包含转换信息的字典：
  - `node_mapping`: {(node_type, original_id): new_global_id}
  - `node_type_to_id`: {node_type: idx}
  - `edge_type_to_id`: {edge_type: idx}
  - `relevant_node_types`: 相关的节点类型列表
  - `relevant_edge_types`: 相关的边类型列表
  - `num_node_types`: 节点类型数量
  - `num_edge_types`: 边类型数量（包含"无边"）

### `extract_metapath_view()`

从异构图提取指定元路径的视图。

**参数:**
- `hetero_data` (HeteroData): 输入的异构图数据
- `metapath` (List[Tuple[str, str, str]]): 元路径，例如 [('user', 'buys', 'item'), ('item', 'bought_by', 'user')]
- `target_node_type` (str): 目标节点类型（元路径视图中的节点类型）
- `device` (torch.device, optional): 设备

**返回:**
- `data` (Data): PyG Data 对象
- `metadata` (Dict): 转换元数据

## 数据格式说明

转换后的数据符合 DiGress 的要求：

### 节点特征 (`data.x`)
- 形状: `[num_nodes, num_node_types]`
- 类型: `torch.float32`
- 格式: One-hot 编码，每行表示一个节点的类型

### 边索引 (`data.edge_index`)
- 形状: `[2, num_edges]`
- 类型: `torch.long`
- 格式: 标准的 PyG 边索引格式

### 边属性 (`data.edge_attr`)
- 形状: `[num_edges, num_edge_types + 1]`
- 类型: `torch.float32`
- 格式: One-hot 编码，第一个维度表示"无边"，后续维度表示不同的边类型

### 图标签 (`data.y`)
- 形状: `[1, num_nodes]` 或 `[1, 0]`
- 类型: `torch.float32`
- 格式: 图级别的标签（如果存在）

### 节点数 (`data.n_nodes`)
- 形状: `[1]`
- 类型: `torch.long`
- 格式: 标量，表示图的节点数

## 示例

完整示例请查看 `hetero_to_homo_example.py`。

### 示例 1: 基本转换

```python
from torch_geometric.data import HeteroData
from src.utils.hetero_to_homo import hetero_to_homo_for_digress
import torch

# 创建示例异构图
hetero_data = HeteroData()
num_reviews = 100
hetero_data['review'].x = torch.randn(num_reviews, 10)
hetero_data['review', 'rur', 'review'].edge_index = torch.randint(0, num_reviews, (2, 50), dtype=torch.long)

# 转换
homo_data, metadata = hetero_to_homo_for_digress(
    hetero_data=hetero_data,
    target_node='review',
    use_node_types=True,
    use_edge_types=True
)

print(f"转换成功！节点数: {homo_data.x.shape[0]}, 边数: {homo_data.edge_index.shape[1]}")
```

### 示例 2: 使用元路径

```python
from src.utils.hetero_to_homo import extract_metapath_view

# 定义元路径
metapath = [('user', 'buys', 'item')]

# 提取元路径视图
homo_data, metadata = extract_metapath_view(
    hetero_data=hetero_data,
    metapath=metapath,
    target_node_type='user'
)
```

## 注意事项

1. **节点映射**: 转换后的节点 ID 是全局编号，原始节点类型和本地 ID 信息保存在 `metadata['node_mapping']` 中。

2. **边类型**: 如果 `use_edge_types=True`，边属性会包含所有相关边类型的 one-hot 编码，第一个维度（索引 0）表示"无边"。

3. **目标节点**: 如果指定了 `target_node`，只会提取与目标节点相关的边和节点类型。

4. **设备**: 转换后的数据会在指定的设备上。如果不指定，会尝试从 `hetero_data` 获取，否则使用 CPU。

5. **Mask 保留**: 如果目标节点有 `train_mask`, `val_mask`, `test_mask`，这些会被保留在转换后的数据中。

## 下一步

转换后的数据可以：
1. 用于创建 PyG DataLoader
2. 用于训练 DiGress 模型
3. 用于使用 DiGress 进行图生成

关于如何将生成的图整合回异构图，请参考 `meta_path_digress_analysis.md` 中的方案 B 和方案 C。

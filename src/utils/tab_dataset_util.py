import os.path as osp
import json
from ..data import *
from . import VNG_utils
import torch
import numpy as np
from typing import Tuple, List, Optional

def _get_graph_from_dataset(graph_dataset):
    # 支持 PyG dataset-like 或直接的图对象
    try:
        # dataset[0] 常见于 PyG 风格
        graph = graph_dataset[0]
    except Exception:
        graph = graph_dataset
    return graph

def _get_feature_tensor(graph) -> torch.Tensor:
    # 支持 PyG 的 Data.x 或 DGL 的 ndata['feat']
    if hasattr(graph, "x") and graph.x is not None:
        feat = graph.x
    elif hasattr(graph, "ndata") and "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        raise RuntimeError("无法在图对象中找到特征矩阵(期望 `x` 或 `ndata['feat']`)")
    if not isinstance(feat, torch.Tensor):
        feat = torch.tensor(feat)
    return feat.detach().cpu()

def _reorder_features(feat: torch.Tensor, num_cols: List[int]) -> Tuple[torch.Tensor, List[int]]:
    total_cols = feat.shape[1]
    num_set = set(num_cols)
    remaining = [i for i in range(total_cols) if i not in num_set]
    new_order = num_cols + remaining
    feat_reordered = feat[:, new_order]
    return feat_reordered, new_order

def _compute_categories_counts(feat_np: np.ndarray, cat_cols: List[int]) -> np.ndarray:
    if len(cat_cols) == 0:
        return np.array([], dtype=np.int64)
    cat_counts = []
    for cidx in cat_cols:
        col_vals = feat_np[:, cidx]
        unique_vals = np.unique(col_vals)
        cat_counts.append(int(unique_vals.size))
    return np.array(cat_counts, dtype=np.int64)

def _infer_n_classes(graph) -> Optional[int]:
    y = getattr(graph, "y", None)
    if y is None:
        return None
    if isinstance(y, torch.Tensor):
        if y.dim() == 0:
            return None
        if y.dim() == 1:
            return int(y.max().item() + 1)
        # 多维标签（one-hot）
        return int(y.shape[1])
    y_arr = np.array(y)
    if y_arr.ndim == 1:
        return int(y_arr.max() + 1)
    return int(y_arr.shape[1])

def load_tab_dataset_info(name, path, split_type='public') -> TabDataset:
    """
    加载表格数据集的数值/类别列信息，并重排图的特征矩阵，使数值列位于前面。
    返回包含数值列数量、类别列类别数数组、类别数量及重排后图对象的 TabDataset 实例。
    说明：
    - `name`: 数据集名称，如 "Cora"。
    - `path`: 数据集存储路径。
    - `split_type`: 图数据集的划分类型，默认为 'public'
    """
    graph_dataset = VNG_utils.get_dataset(name, path, split_type, normalize_features=False)
    n_feat = graph_dataset.num_features if hasattr(graph_dataset, 'num_features') else None
    graph = _get_graph_from_dataset(graph_dataset)

    feat = _get_feature_tensor(graph)
    total_cols = feat.shape[1]

    meta_info = load_meta_info(path,name=name)
    num_cols = meta_info.get('num_col_idx', []) or []
    cat_cols = meta_info.get('cat_col_idx', []) or []

    # 校验并过滤越界索引
    num_cols = [int(i) for i in num_cols if 0 <= int(i) < total_cols]
    cat_cols = [int(i) for i in cat_cols if 0 <= int(i) < total_cols]

    # 在重排前根据原始列索引统计分类列类别数（避免重排后索引变化）
    feat_np = feat.numpy()
    categories = _compute_categories_counts(feat_np, cat_cols)

    # 重排特征矩阵：数值列 -> 其余列
    feat_reordered, new_order = _reorder_features(feat, num_cols)

    # 写回图对象
    if hasattr(graph, "x") and graph.x is not None:
        graph.x = feat_reordered
    elif hasattr(graph, "ndata"):
        graph.ndata["feat"] = feat_reordered

    # 填充 TabDataset
    dataset = TabDataset()
    dataset.name = name
    dataset.num_numerical_features = len(num_cols)
    dataset.categories = categories
    dataset.n_labels = _infer_n_classes(graph)
    dataset.graph = graph
    dataset.n_features = n_feat

    return dataset

def load_meta_info(path,name):
    path = osp.join(path, f'{name}_info.json')
    with open(path, "r", encoding="utf-8") as f:
        meta_info = json.load(f)
    return meta_info
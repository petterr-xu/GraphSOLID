import copy
import torch
import random
import warnings
import statistics
import numpy as np
import os.path as osp
from torch_geometric.utils import train_test_split_edges, negative_sampling, from_scipy_sparse_matrix
from torch_geometric.data import Data

from src import solid
from args import parse_args
from src.utils import VNG_utils, graphbuilder
from solid_trainer import SolidTrainer
from src.utils.hetero_dataset_util import GraphDataLoader
from src.TabDiff.tabdiff.modules.main_modules import UniModMLP
from src.TabDiff.tabdiff.modules.main_modules import Model
from src.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from src.models import gnn,sage,edge_learner,teacher,diffusion,mlp,HeteroNN
from src.denoise import unet

from matplotlib import pyplot as plt
from src.nettack.nettack import utils, GCN
from src.nettack.nettack import nettack as ntk
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp

from src.gnn_meta_attack.metattack import utils
from src.gnn_meta_attack.metattack import meta_gradient_attack as mtk
import numpy as np
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Planetoid  # 替换为你的数据集导入
from torch_geometric.utils import to_scipy_sparse_matrix, remove_self_loops, to_undirected

# 禁用TF Eager Execution（原代码要求）
tf.compat.v1.disable_eager_execution()
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# ===================== 2. 全局配置 =====================
GPU_ID = None
SEED = 15
SHARE_PERTURBATIONS = 0.05  # 扰动边的比例
TRAIN_ITERS = 100
DTYPE = tf.float32  # 内存不足可换tf.float16
RE_TRAININGS = 20  # 攻击后重复训练验证次数
ATTACK_VARIANT = "Meta-Self"  # 攻击变体（可选：Meta-Train/Meta-Self/A-Meta-Train等）
ENFORCE_LL_CONSTRAINT = False

# ===================== 3. 核心函数封装 =====================
def convert_pyg_data(data: Data):
    """
    将PyG数据集转换为原代码兼容的格式（scipy稀疏矩阵/NumPy数组）
    :return: _A_obs, _X_obs, _z_obs, _Z_obs, _N, _K, data_mask （mask包含train/val/test索引）
    """
    # 2. 处理边：去自环、转无向 → scipy CSR邻接矩阵
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index)
    _A_obs = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes).tocsr()

    # 3. 对齐原代码的邻接矩阵后处理
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)  # 保留最大连通子图
    _A_obs = _A_obs[lcc][:, lcc]
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()

    # 4. 验证邻接矩阵（原代码校验逻辑）
    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
    assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # 5. 转换节点特征：PyTorch张量 → NumPy数组
    _X_obs = data.x.cpu().numpy().astype("float32")
    _X_obs = _X_obs[lcc]  # 映射到最大连通子图

    # 6. 转换标签：PyTorch张量 → NumPy数组（one-hot）
    _z_obs = data.y.cpu().numpy().squeeze()
    _z_obs = _z_obs[lcc]  # 映射到最大连通子图
    _K = len(np.unique(_z_obs))  # 类别数
    _Z_obs = np.eye(_K)[_z_obs]

    # 7. 处理PyG自带的mask → 映射到最大连通子图后的索引
    _N = _A_obs.shape[0]  # 最大连通子图节点数
    original_node_idx = np.arange(data.num_nodes)[lcc]  # 最大连通子图的原始节点索引

    # 构建mask字典：存储train/val/test的索引（基于最大连通子图）
    data_mask = {
        "train": np.where(data.train_mask.cpu().numpy()[original_node_idx])[0],
        "val": np.where(data.val_mask.cpu().numpy()[original_node_idx])[0],
        "test": np.where(data.test_mask.cpu().numpy()[original_node_idx])[0]
    }

    return _A_obs, _X_obs, _z_obs, _Z_obs, _N, _K, data_mask


def split_dataset_by_pyg_mask(data_mask):
    """
    基于PyG自带的mask划分数据集（对齐原代码的split_train/split_val/split_unlabeled）
    :param data_mask: 包含train/val/test索引的字典
    :return: split_train, split_val, split_unlabeled
    """
    split_train = data_mask["train"]
    split_val = data_mask["val"]
    split_unlabeled = data_mask["test"]  # 原代码中unlabeled包含val+test，此处对齐
    split_unlabeled = np.union1d(split_val, split_unlabeled)
    return split_train, split_val, split_unlabeled


def run_meta_attack(_A_obs, _X_obs, _Z_obs, _N, _K, split_train, split_unlabeled, attack_variant):
    """
    执行MetaAttack攻击逻辑
    :param _A_obs: 原始邻接矩阵
    :param _X_obs: 节点特征
    :param _Z_obs: one-hot标签
    :param _N: 节点数
    :param _K: 类别数
    :param split_train: 训练集索引
    :param split_unlabeled: 无标签集（val+test）索引
    :param attack_variant: 攻击变体
    :return: modified_adjacency（扰动后的邻接矩阵）
    """
    # 1. 初始化并训练代理GCN
    hidden_sizes = [16]
    surrogate = mtk.GCNSparse(_A_obs, _X_obs, _Z_obs, hidden_sizes, gpu_id=GPU_ID)
    surrogate.build(with_relu=False)
    surrogate.train(split_train)

    # 2. 自训练标签预测
    labels_self_training = np.eye(_K)[surrogate.logits.eval(session=surrogate.session).argmax(1)]
    labels_self_training[split_train] = _Z_obs[split_train]

    # 3. 攻击参数配置
    approximate_meta_gradient = attack_variant.startswith("A-")
    lambda_ = 1.0 if "Train" in attack_variant else (0.5 if "Both" in attack_variant else 0.0)
    idx_attack = split_train if "Train" in attack_variant else (
        np.union1d(split_train, split_unlabeled) if "Both" in attack_variant else split_unlabeled
    )
    perturbations = int(SHARE_PERTURBATIONS * (_A_obs.sum() // 2))

    # 4. 初始化攻击器
    if approximate_meta_gradient:
        gcn_attack = mtk.GNNMetaApprox(
            _A_obs, _X_obs, labels_self_training, hidden_sizes,
            gpu_id=GPU_ID, _lambda=lambda_, train_iters=TRAIN_ITERS, dtype=DTYPE
        )
    else:
        gcn_attack = mtk.GNNMeta(
            _A_obs, _X_obs.astype("float32"), labels_self_training, hidden_sizes,
            gpu_id=GPU_ID, attack_features=False, train_iters=TRAIN_ITERS, dtype=DTYPE
        )

    # 5. 执行攻击
    gcn_attack.build()
    gcn_attack.make_loss(ll_constraint=ENFORCE_LL_CONSTRAINT)
    if approximate_meta_gradient:
        gcn_attack.attack(perturbations, split_train, split_unlabeled, idx_attack)
    else:
        gcn_attack.attack(perturbations, split_train, idx_attack)

    # 6. 获取扰动后的邻接矩阵
    modified_adjacency = gcn_attack.modified_adjacency.eval(session=gcn_attack.session)
    return modified_adjacency


def evaluate_accuracy(_A_obs, modified_adjacency, _X_obs, _Z_obs, _z_obs, split_train, split_unlabeled):
    """
    评估攻击前后的模型准确率
    :param _A_obs: 原始邻接矩阵
    :param modified_adjacency: 扰动后的邻接矩阵
    :param _X_obs: 节点特征
    :param _Z_obs: one-hot标签
    :param _z_obs: 类别索引标签
    :param split_train: 训练集索引
    :param split_unlabeled: 无标签集（val+test）索引
    :return: accuracies_clean, accuracies_atk
    """
    hidden_sizes = [16]
    # 1. 攻击前准确率
    gcn_before_attack = mtk.GCNSparse(sp.csr_matrix(_A_obs), _X_obs, _Z_obs, hidden_sizes, gpu_id=GPU_ID)
    gcn_before_attack.build(with_relu=True)
    accuracies_clean = []
    for _it in tqdm(range(RE_TRAININGS), desc="Evaluating clean accuracy"):
        gcn_before_attack.train(split_train, initialize=True, display=False)
        accuracy_clean = (gcn_before_attack.logits.eval(session=gcn_before_attack.session).argmax(1) == _z_obs)[split_unlabeled].mean()
        accuracies_clean.append(accuracy_clean)

    # 2. 攻击后准确率
    gcn_after_attack = mtk.GCNSparse(sp.csr_matrix(modified_adjacency), _X_obs, _Z_obs, hidden_sizes, gpu_id=GPU_ID)
    gcn_after_attack.build(with_relu=True)
    accuracies_atk = []
    for _it in tqdm(range(RE_TRAININGS), desc="Evaluating attacked accuracy"):
        gcn_after_attack.train(split_train, initialize=True, display=False)
        accuracy_atk = (gcn_after_attack.logits.eval(session=gcn_after_attack.session).argmax(1) == _z_obs)[split_unlabeled].mean()
        accuracies_atk.append(accuracy_atk)

    return accuracies_clean, accuracies_atk


def plot_accuracy(accuracies_clean, accuracies_atk):
    """
    绘制攻击前后准确率对比图
    """
    plt.figure(figsize=(6, 6))
    print("clean acc:",accuracies_clean)
    print("atk acc:",accuracies_atk)
    x = ["Acc. Clean"]*len(accuracies_clean) + ["Acc. Perturbed"]*len(accuracies_atk)
    y = accuracies_clean + accuracies_atk
    sns.boxplot(x = x, y = y)
    plt.title(f"Accuracy before and after perturbing {int(SHARE_PERTURBATIONS*100)}% edges using {ATTACK_VARIANT}")
    plt.savefig("example.png", dpi=600)
    plt.savefig("example.svg")
    plt.show()

# ===================== 4. 主函数（执行流程） =====================
def main():
    args = parse_args()
    device = args.device
    root_path = osp.dirname(osp.realpath(__file__))

    max_n=500

    if args.dataset in ['YelpChi', 'Amazon-Products']:
        loader = GraphDataLoader()
        data_path = osp.join(root_path, 'data', args.dataset, 'data', args.dataset + '.mat')
        cnfg_path = osp.join(root_path, 'data', args.dataset, 'meta', args.dataset + '.json')
        hetero_ctx = loader.load_from_config(cnfg_path, data_path)
        target = hetero_ctx.target_node  # 'review' 或 'user'
        data = hetero_ctx.g.to(device)
        n_feat = hetero_ctx.n_features
        n_cls = hetero_ctx.n_classes
        print(data)
        data_train_mask, data_val_mask, data_test_mask = data[target].train_mask.clone(), data[target].val_mask.clone(), data[target].test_mask.clone()
        stats = data[target].y[data_train_mask]
        n_data = []
        for i in range(n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        idx_info = VNG_utils.get_idx_info(data[target].y, n_cls, data_train_mask)
        class_num_list = n_data
        print("num of class in original training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
        class_num_list, data_train_mask, _, edge_mask_dict = graphbuilder.make_hetero_longtailed_data_remove(data, target, n_data, n_cls, args.imb_ratio, data_train_mask.clone(), max_n)
        # 更新 HeteroData
        hetero_ctx.g[hetero_ctx.target_node].train_mask = data_train_mask
        # 更新边索引 (可选，取决于是否想物理删除边)
        if not args.keep_edge:
            for etype, mask in edge_mask_dict.items():
                hetero_ctx.g[etype].edge_index = hetero_ctx.g[etype].edge_index[:, mask]
        print("num of class in LT-training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
        minority_mask = class_num_list < (sum(class_num_list)/n_cls)
        minority_class = [i for i in range(n_cls) if minority_mask[i]]
        print("minority classes {}".format(minority_class))
        data = data.to_homogeneous() # 转换为同质图，便于Nettack处理
    elif args.dataset in ['Cora','CiteSeer','PubMed']:
        path = osp.join(root_path, 'data', args.dataset, 'data')
        dataset = VNG_utils.get_dataset(args.dataset, path)
        data = dataset[0]
        data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
        edge_index = data.edge_index.clone()
        stats = data.y[data_train_mask]
        n_data = []
        n_cls = data.y.max().item()+1
        for i in range(n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        idx_info = VNG_utils.get_idx_info(data.y, n_cls, data_train_mask)
        class_num_list = n_data
        print("num of class in original training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
        class_num_list, data_train_mask, train_node_mask, train_edge_mask = graphbuilder.make_longtailed_data_remove(edge_index, data.y, n_data, n_cls, args.imb_ratio, data_train_mask.clone(), max_n)
        if args.keep_edge:
            train_edge_mask = torch.ones_like(train_edge_mask,dtype=torch.bool,device=train_edge_mask.device)
        print("num of class in LT-training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
        minority_mask = class_num_list < (sum(class_num_list)/n_cls)
        minority_class = [i for i in range(n_cls) if minority_mask[i]]
        print("minority classes {}".format(minority_class))
        print("number of edges {}".format(sum(train_edge_mask)))
    else:
        raise NotImplementedError("Not implemented for dataset {}".format(args.dataset))
    # 步骤1：加载并转换PyG数据（替换dataset_name为你的数据集）
    _A_obs, _X_obs, _z_obs, _Z_obs, _N, _K, data_mask = graphbuilder.pyg2matrix(data)

    # 步骤2：基于PyG mask划分数据集
    split_train, split_val, split_unlabeled = split_dataset_by_pyg_mask(data_mask)

    # 步骤3：执行MetaAttack攻击
    modified_adjacency = run_meta_attack(_A_obs, _X_obs, _Z_obs, _N, _K, split_train, split_unlabeled, ATTACK_VARIANT)

    # 步骤4：评估攻击前后准确率
    accuracies_clean, accuracies_atk = evaluate_accuracy(_A_obs, modified_adjacency, _X_obs, _Z_obs, _z_obs, split_train, split_unlabeled)

    # 打印关键结果
    print(f"Clean Accuracy (mean±std): {np.mean(accuracies_clean):.4f} ± {np.std(accuracies_clean):.4f}")
    print(f"Attacked Accuracy (mean±std): {np.mean(accuracies_atk):.4f} ± {np.std(accuracies_atk):.4f}")
    # 步骤5：绘制并保存结果图
    plot_accuracy(accuracies_clean, accuracies_atk)


if __name__ == "__main__":
    # 固定随机种子（可选）
    np.random.seed(SEED)
    tf.compat.v1.set_random_seed(SEED)
    # 执行主流程
    main()
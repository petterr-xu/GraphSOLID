from matplotlib import pyplot as plt
from src.nettack.nettack import utils, GCN
from src.nettack import nettack as ntk
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp

# ======================== 封装1：PyG数据 → 原代码所需格式（核心数据转换） ========================
def pyg_to_nettack_format(data):
    """
    将PyG的Data对象转换为Nettack原代码所需的scipy/numpy格式
    Args:
        data (torch_geometric.data.Data): PyG数据对象（需包含x, edge_index, y）
    Returns:
        _A_obs (scipy.sparse.csr_matrix): 对称化去重后的邻接矩阵
        _X_obs (scipy.sparse.csr_matrix): 特征矩阵（float32）
        _z_obs (np.ndarray): 一维标签数组
    """
    # 1. 转换邻接矩阵：edge_index → scipy csr 矩阵（对称化+去重，对齐原代码）
    _A_obs = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    _A_obs = _A_obs + _A_obs.T  # 对称化
    _A_obs[_A_obs > 1] = 1      # 去重边（避免重边影响攻击逻辑）
    _A_obs.eliminate_zeros()    # 移除零元素，优化稀疏矩阵存储

    # 2. 转换特征矩阵：PyG张量 → scipy csr 矩阵（float32格式）
    _X_obs = sp.csr_matrix(data.x.numpy().astype('float32'))

    # 3. 转换标签：PyG张量 → 一维numpy数组
    _z_obs = data.y.numpy().squeeze()  # 确保维度为(N,)，适配原代码分层划分

    return _A_obs, _X_obs, _z_obs

# ======================== 封装2：数据预处理与攻击准备（划分、代理模型训练） ========================
def prepare_nettack(_A_obs, _X_obs, _z_obs, gpu_id=None, seed=15, attack_node=0):
    """
    完成数据LCC筛选、数据划分、代理模型训练，为Nettack攻击做准备
    Args:
        _A_obs/_X_obs/_z_obs: 由pyg_to_nettack_format返回的格式数据
        gpu_id (int/None): TensorFlow GPU配置，None为CPU
        seed (int): 随机种子，保证数据划分可复现
        attack_node (int): 待攻击的节点ID（需在无标签集内）
    Returns:
        nettack_ready (dict): 包含攻击所需的所有关键参数
    """
    # 1. 筛选最大连通分量（LCC），对齐原代码逻辑
    lcc = utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc][:, lcc]
    _X_obs = _X_obs[lcc].astype('float32')
    _z_obs = _z_obs[lcc]

    # 2. 数据校验（避免后续攻击报错，可选但推荐）
    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "邻接矩阵必须是对称的"
    assert _A_obs.max() == 1, "邻接矩阵必须是无权图"
    assert _A_obs.sum(0).A1.min() > 0, "图中不能包含孤立节点"

    # 3. 关键参数计算
    _N = _A_obs.shape[0]
    _K = _z_obs.max() + 1
    _Z_obs = np.eye(_K)[_z_obs]  # 标签one-hot编码
    _An = utils.preprocess_graph(_A_obs)  # 邻接矩阵预处理（归一化）
    sizes = [16, _K]  # GCN模型层尺寸（和原代码一致）
    degrees = _A_obs.sum(0).A1  # 各节点的度

    # 4. 数据划分（训练/验证/无标签集，比例和原代码一致）
    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share
    np.random.seed(seed)
    split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(
        np.arange(_N),
        train_size=train_share,
        val_size=val_share,
        test_size=unlabeled_share,
        stratify=_z_obs
    )

    # 5. 验证攻击节点有效性
    assert attack_node in split_unlabeled, "攻击节点必须在无标签集中"

    # 6. 训练原代理模型（TensorFlow GCN，不改动核心逻辑）
    surrogate_model = GCN.GCN(sizes, _An, _X_obs, with_relu=False, name="surrogate", gpu_id=gpu_id)
    surrogate_model.train(split_train, split_val, _Z_obs)
    W1 = surrogate_model.W1.eval(session=surrogate_model.session)
    W2 = surrogate_model.W2.eval(session=surrogate_model.session)

    # 7. 封装返回所有关键参数（便于后续调用，避免全局变量泛滥）
    nettack_ready = {
        "A_obs": _A_obs,
        "X_obs": _X_obs,
        "z_obs": _z_obs,
        "Z_obs": _Z_obs,
        "An": _An,
        "sizes": sizes,
        "degrees": degrees,
        "N": _N,
        "K": _K,
        "split_train": split_train,
        "split_val": split_val,
        "split_unlabeled": split_unlabeled,
        "attack_node": attack_node,
        "W1": W1,
        "W2": W2,
        "gpu_id": gpu_id
    }

    return nettack_ready

# ======================== 封装3：执行Nettack攻击（核心扰动生成） ========================
def run_nettack(nettack_ready, perturb_structure=True, perturb_features=True):
    """
    执行Nettack攻击，生成图结构/特征扰动
    Args:
        nettack_ready (dict): 由prepare_nettack返回的准备参数
        perturb_structure (bool): 是否扰动图结构
        perturb_features (bool): 是否扰动节点特征
    Returns:
        nettack (ntk.Nettack): 执行完攻击的Nettack对象（包含扰动结果）
        attack_config (dict): 攻击配置参数
    """
    # 1. 提取准备参数
    A_obs = nettack_ready["A_obs"]
    X_obs = nettack_ready["X_obs"]
    z_obs = nettack_ready["z_obs"]
    W1 = nettack_ready["W1"]
    W2 = nettack_ready["W2"]
    u = nettack_ready["attack_node"]
    degrees = nettack_ready["degrees"]

    # 2. 攻击配置（可灵活调整，封装后便于修改）
    direct_attack = True
    n_influencers = 1 if direct_attack else 5
    n_perturbations = int(degrees[u])  # 扰动数量=攻击节点的度（原代码默认）
    attack_config = {
        "direct_attack": direct_attack,
        "n_influencers": n_influencers,
        "n_perturbations": n_perturbations,
        "perturb_structure": perturb_structure,
        "perturb_features": perturb_features
    }

    # 3. 初始化并执行攻击
    nettack = ntk.Nettack(A_obs, X_obs, z_obs, W1, W2, u, verbose=True)
    nettack.reset()
    nettack.attack_surrogate(
        n_perturbations,
        perturb_structure=perturb_structure,
        perturb_features=perturb_features,
        direct=direct_attack,
        n_influencers=n_influencers
    )

    # 4. 打印扰动结果（可选，便于调试）
    print("="*50)
    print(f"结构扰动列表：{nettack.structure_perturbations}")
    print(f"特征扰动列表：{nettack.feature_perturbations}")
    print("="*50)

    return nettack, attack_config

# ======================== 封装4：攻击结果评估与可视化 ========================
def evaluate_and_visualize(nettack_ready, nettack, attack_config, retrain_iters=5):
    """
    评估攻击前后模型性能，可视化分类概率对比
    Args:
        nettack_ready (dict): 攻击准备参数
        nettack (ntk.Nettack): 执行完攻击的Nettack对象
        attack_config (dict): 攻击配置参数
        retrain_iters (int): 重复训练次数，用于评估稳定性
    """
    # 1. 提取关键参数
    sizes = nettack_ready["sizes"]
    An = nettack_ready["An"]
    X_obs = nettack_ready["X_obs"]
    Z_obs = nettack_ready["Z_obs"]
    split_train = nettack_ready["split_train"]
    split_val = nettack_ready["split_val"]
    gpu_id = nettack_ready["gpu_id"]
    u = nettack_ready["attack_node"]
    n_perturbations = attack_config["n_perturbations"]

    # 2. 干净数据上的模型训练与评估
    print(f"\n开始在干净数据上训练GCN（{retrain_iters}次重复）...")
    class_distrs_clean = []
    gcn_before = GCN.GCN(sizes, An, X_obs, "gcn_orig", gpu_id=gpu_id)
    for _ in range(retrain_iters):
        gcn_before.train(split_train, split_val, Z_obs)
        probs_before = gcn_before.predictions.eval(
            session=gcn_before.session,
            feed_dict={gcn_before.node_ids: [u]}
        )[0]
        class_distrs_clean.append(probs_before)
    class_distrs_clean = np.array(class_distrs_clean)

    # 3. 扰动后数据上的模型训练与评估
    print(f"\n开始在扰动后数据上训练GCN（{retrain_iters}次重复）...")
    class_distrs_retrain = []
    gcn_retrain = GCN.GCN(sizes, nettack.adj_preprocessed, nettack.X_obs.tocsr(), "gcn_retrain", gpu_id=gpu_id)
    for _ in range(retrain_iters):
        gcn_retrain.train(split_train, split_val, Z_obs)
        probs_after = gcn_retrain.predictions.eval(
            session=gcn_retrain.session,
            feed_dict={gcn_retrain.node_ids: [u]}
        )[0]
        class_distrs_retrain.append(probs_after)
    class_distrs_retrain = np.array(class_distrs_retrain)

    # 4. 可视化辅助函数（内部封装，不对外暴露）
    def _make_xlabel(ix, correct):
        if ix == correct:
            return f"Class {ix}\n(correct)"
        return f"Class {ix}"

    # 5. 绘制对比图
    plt.figure(figsize=(12, 4))

    # 干净数据结果
    plt.subplot(1, 2, 1)
    center_ixs_clean = []
    for ix, block in enumerate(class_distrs_clean.T):
        x_ixs = np.arange(len(block)) + ix * (len(block) + 2)
        center_ixs_clean.append(np.mean(x_ixs))
        color = 'darkgreen' if ix == nettack.label_u else '#555555'
        plt.bar(x_ixs, block, color=color)
    ax = plt.gca()
    plt.ylim((-.05, 1.05))
    plt.ylabel("Predicted probability")
    ax.set_xticks(center_ixs_clean)
    ax.set_xticklabels([_make_xlabel(k, nettack.label_u) for k in range(nettack_ready["K"])])
    ax.set_title(f"Clean Data - Node {u}\n({retrain_iters} re-trainings)")

    # 扰动后数据结果
    plt.subplot(1, 2, 2)
    center_ixs_retrain = []
    for ix, block in enumerate(class_distrs_retrain.T):
        x_ixs = np.arange(len(block)) + ix * (len(block) + 2)
        center_ixs_retrain.append(np.mean(x_ixs))
        color = 'darkgreen' if ix == nettack.label_u else '#555555'
        plt.bar(x_ixs, block, color=color)
    ax = plt.gca()
    plt.ylim((-.05, 1.05))
    ax.set_xticks(center_ixs_retrain)
    ax.set_xticklabels([_make_xlabel(k, nettack.label_u) for k in range(nettack_ready["K"])])
    ax.set_title(f"After {n_perturbations} Perturbations - Node {u}\n({retrain_iters} re-trainings)")

    plt.tight_layout()
    plt.show()

# ======================== 主函数：串联所有模块（一键运行，便于修改参数） ========================
def main():
    """主函数：串联所有封装模块，一键执行PyG数据适配→Nettack攻击→结果评估"""
    # 1. 配置参数（后续修改只需调整这里，无需改动核心函数）
    PYG_DATASET_NAME = "Citeseer"  # 替换为你的PyG数据集
    GPU_ID = None  # 如需GPU，设置为0/1等
    SEED = 15  # 随机种子
    ATTACK_NODE = 0  # 待攻击节点
    PERTURB_STRUCTURE = True  # 是否扰动结构
    PERTURB_FEATURES = True  # 是否扰动特征
    RETRAIN_ITERS = 5  # 重复训练次数

    # 2. 加载PyG数据（替换为你的自定义数据加载逻辑即可）
    print(f"正在加载PyG数据集：{PYG_DATASET_NAME}...")
    dataset = Planetoid(root='./data', name=PYG_DATASET_NAME)
    data = dataset[0]

    # 3. 步骤1：PyG数据格式转换
    _A_obs, _X_obs, _z_obs = pyg_to_nettack_format(data)

    # 4. 步骤2：攻击前准备（数据划分、代理模型训练）
    nettack_ready = prepare_nettack(
        _A_obs, _X_obs, _z_obs,
        gpu_id=GPU_ID,
        seed=SEED,
        attack_node=ATTACK_NODE
    )

    # 5. 步骤3：执行Nettack攻击
    nettack, attack_config = run_nettack(
        nettack_ready,
        perturb_structure=PERTURB_STRUCTURE,
        perturb_features=PERTURB_FEATURES
    )

    # 6. 步骤4：评估与可视化
    evaluate_and_visualize(nettack_ready, nettack, attack_config, retrain_iters=RETRAIN_ITERS)

# ======================== 一键运行入口 ========================
if __name__ == "__main__":
    main()
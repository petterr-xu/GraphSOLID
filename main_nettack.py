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

warnings.filterwarnings("ignore")
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
        masks (dict): train/val/test mask的numpy格式
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
    masks = {
            "train_mask": data.train_mask.numpy(),
            "val_mask": data.val_mask.numpy(),
            "test_mask": data.test_mask.numpy()
        }

    return _A_obs, _X_obs, _z_obs, masks

# ======================== 封装2：数据预处理与攻击准备（划分、代理模型训练） ========================
def prepare_nettack(_A_obs, _X_obs, _z_obs, masks, gpu_id=None, attack_node=0):
    """
    完成数据LCC筛选、从PyG mask提取节点划分、代理模型训练，为Nettack攻击做准备
    核心修改：使用PyG自带mask划分，而非随机比例划分
    Args:
        _A_obs/_X_obs/_z_obs: 由pyg_to_nettack_format返回的格式数据
        masks (dict): PyG原生的train/val/test mask（numpy格式）
        gpu_id (int/None): TensorFlow GPU配置，None为CPU
        attack_node (int): 待攻击的节点ID（原始节点ID，需在test集内）
    Returns:
        nettack_ready (dict): 包含攻击所需的所有关键参数
    """
    # 1. 筛选最大连通分量（LCC），对齐原代码逻辑
    lcc = utils.largest_connected_components(_A_obs)  # 原始节点ID的LCC索引数组
    lcc_mask = np.zeros(_A_obs.shape[0], dtype=bool)
    lcc_mask[lcc] = True  # 原始节点的LCC掩码（方便后续映射）

    # 2. 对邻接矩阵/特征/标签进行LCC筛选
    _A_obs = _A_obs[lcc][:, lcc]
    _X_obs = _X_obs[lcc].astype('float32')
    _z_obs = _z_obs[lcc]

    # 3. 数据校验（避免后续攻击报错，可选但推荐）
    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "邻接矩阵必须是对称的"
    assert _A_obs.max() == 1, "邻接矩阵必须是无权图"
    assert _A_obs.sum(0).A1.min() > 0, "图中不能包含孤立节点"

    # 4. 关键参数计算
    _N = _A_obs.shape[0]
    _K = _z_obs.max() + 1
    _Z_obs = np.eye(_K)[_z_obs]  # 标签one-hot编码
    _An = utils.preprocess_graph(_A_obs)  # 邻接矩阵预处理（归一化）
    sizes = [16, _K]  # GCN模型层尺寸（和原代码一致）
    degrees = _A_obs.sum(0).A1  # 各节点的度（LCC筛选后）

    # ======================== 核心修改：从PyG mask提取节点划分（适配LCC） ========================
    # 步骤1：提取原始节点集中的train/val/test节点索引
    original_train_nodes = np.where(masks["train_mask"])[0]
    original_val_nodes = np.where(masks["val_mask"])[0]
    original_test_nodes = np.where(masks["test_mask"])[0]

    # 步骤2：映射到LCC筛选后的节点索引（关键：解决LCC后节点ID变化问题）
    # 构建原始节点ID → LCC节点ID的映射字典
    original_to_lcc = {original_node: lcc_node for lcc_node, original_node in enumerate(lcc)}
    # 仅保留在LCC中的节点，并转换为LCC内的节点ID
    split_train = [original_to_lcc[node] for node in original_train_nodes if node in original_to_lcc]
    split_val = [original_to_lcc[node] for node in original_val_nodes if node in original_to_lcc]
    split_unlabeled = [original_to_lcc[node] for node in original_test_nodes if node in original_to_lcc]  # Nettack中无标签集对应PyG的test集

    # 步骤3：将攻击节点（原始ID）映射到LCC内的ID，并验证有效性
    assert attack_node in original_to_lcc, f"攻击节点{attack_node}不在LCC连通分量中"
    lcc_attack_node = original_to_lcc[attack_node]
    assert lcc_attack_node in split_unlabeled, f"攻击节点{attack_node}必须在PyG的test集（Nettack无标签集）中"

    # 5. 训练原代理模型（TensorFlow GCN，不改动核心逻辑）
    surrogate_model = GCN.GCN(sizes, _An, _X_obs, with_relu=False, name="surrogate", gpu_id=gpu_id)
    surrogate_model.train(split_train, split_val, _Z_obs)
    W1 = surrogate_model.W1.eval(session=surrogate_model.session)
    W2 = surrogate_model.W2.eval(session=surrogate_model.session)

    # 6. 封装返回所有关键参数（便于后续调用，避免全局变量泛滥）
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
        "split_train": split_train,  # PyG mask提取的训练集（LCC内ID）
        "split_val": split_val,      # PyG mask提取的验证集（LCC内ID）
        "split_unlabeled": split_unlabeled,  # PyG mask提取的测试集（LCC内ID）
        "attack_node": lcc_attack_node,  # 映射后的攻击节点（LCC内ID）
        "original_attack_node": attack_node,  # 保留原始攻击节点ID（便于核对）
        "W1": W1,
        "W2": W2,
        "gpu_id": gpu_id,
        "original_to_lcc": original_to_lcc,  # 新增：原始ID→LCC ID映射（批量攻击用）
        "lcc": lcc  # 新增：LCC节点列表（批量攻击用）
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
        attack_success (bool): 攻击是否成功（模型预测标签与真实标签不一致）
    """
    # 1. 提取准备参数
    A_obs = nettack_ready["A_obs"]
    X_obs = nettack_ready["X_obs"]
    z_obs = nettack_ready["z_obs"]
    W1 = nettack_ready["W1"]
    W2 = nettack_ready["W2"]
    u = nettack_ready["attack_node"]
    degrees = nettack_ready["degrees"]
    original_attack_node = nettack_ready["original_attack_node"]

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
    nettack = ntk.Nettack(A_obs, X_obs, z_obs, W1, W2, u, verbose=False)  # 批量攻击时关闭verbose
    nettack.reset()
    nettack.attack_surrogate(
        n_perturbations,
        perturb_structure=perturb_structure,
        perturb_features=perturb_features,
        direct=direct_attack,
        n_influencers=n_influencers
    )

    # 4. 判定攻击是否成功（核心：攻击后模型预测标签≠真实标签）
    # 重新训练模型并预测攻击节点
    gcn_after = GCN.GCN(nettack_ready["sizes"], nettack.adj_preprocessed, nettack.X_obs.tocsr(), 
                        f"gcn_after_{original_attack_node}", gpu_id=nettack_ready["gpu_id"])
    gcn_after.train(nettack_ready["split_train"], nettack_ready["split_val"], nettack_ready["Z_obs"])
    pred_prob = gcn_after.predictions.eval(
        session=gcn_after.session,
        feed_dict={gcn_after.node_ids: [u]}
    )[0]
    pred_label = np.argmax(pred_prob)
    true_label = z_obs[u]
    attack_success = (pred_label != true_label)

    # 打印单节点攻击结果（可选）
    print(f"节点{original_attack_node}（LCC ID:{u}）- 真实标签:{true_label}, 攻击后预测标签:{pred_label}, 攻击成功:{attack_success}")

    return nettack, attack_config, attack_success

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
    plt.savefig("nettack_attack_result.png", dpi=300)

# ======================== 新增封装5：Nettack扰动数据 → PyG Data对象（核心功能） ========================
def nettack_to_pyg_format(nettack, original_data_info, return_full_graph=True):
    """
    将Nettack攻击后的扰动数据转换回PyG的Data对象格式
    Args:
        nettack (ntk.Nettack): 执行完攻击的Nettack对象（包含扰动结果）
        original_data_info (dict): 原始数据信息（从prepare_nettack返回）
        return_full_graph (bool): 是否还原为原始规模图（补全非LCC节点），False返回LCC规模图
    Returns:
        pyg_data_perturbed (torch_geometric.data.Data): 扰动后的PyG Data对象
    """
    # 1. 提取Nettack扰动后的核心数据（scipy格式）
    # 扰动后的邻接矩阵（结构扰动）
    A_pert_scipy = nettack.A_pert.tocsr()
    # 扰动后的特征矩阵（特征扰动）
    X_pert_scipy = nettack.X_obs.tocsr()
    # 扰动后的标签（LCC内，无扰动，仅用于还原）
    z_pert_lcc = nettack.z_obs

    # 2. 核心转换：scipy稀疏矩阵 → PyG张量格式
    # 邻接矩阵 → edge_index（PyG标准边索引格式）
    edge_index_pert, _ = from_scipy_sparse_matrix(A_pert_scipy)
    # 特征矩阵 → x（PyG节点特征张量）
    x_pert = torch.tensor(X_pert_scipy.toarray(), dtype=torch.float32)
    # 标签 → y（PyG标签张量）
    y_pert_lcc = torch.tensor(z_pert_lcc, dtype=torch.long).unsqueeze(1)

    # 3. 还原mask（LCC内，基于原始mask映射）
    lcc = original_data_info["lcc"]
    original_masks = original_data_info["original_masks"]
    original_to_lcc = original_data_info["original_to_lcc"]

    # 提取原始mask并映射到LCC内
    lcc_train_mask = np.zeros(A_pert_scipy.shape[0], dtype=bool)
    lcc_val_mask = np.zeros(A_pert_scipy.shape[0], dtype=bool)
    lcc_test_mask = np.zeros(A_pert_scipy.shape[0], dtype=bool)

    original_train_nodes = np.where(original_masks["train_mask"])[0]
    original_val_nodes = np.where(original_masks["val_mask"])[0]
    original_test_nodes = np.where(original_masks["test_mask"])[0]

    # 填充LCC内的mask
    for node in original_train_nodes:
        if node in original_to_lcc:
            lcc_train_mask[original_to_lcc[node]] = True
    for node in original_val_nodes:
        if node in original_to_lcc:
            lcc_val_mask[original_to_lcc[node]] = True
    for node in original_test_nodes:
        if node in original_to_lcc:
            lcc_test_mask[original_to_lcc[node]] = True

    # 转换为PyG张量格式mask
    train_mask_pert = torch.tensor(lcc_train_mask, dtype=torch.bool)
    val_mask_pert = torch.tensor(lcc_val_mask, dtype=torch.bool)
    test_mask_pert = torch.tensor(lcc_test_mask, dtype=torch.bool)

    # 4. 构建LCC规模的PyG Data对象
    pyg_data_lcc = Data(
        x=x_pert,
        edge_index=edge_index_pert,
        y=y_pert_lcc,
        train_mask=train_mask_pert,
        val_mask=val_mask_pert,
        test_mask=test_mask_pert,
        num_nodes=x_pert.shape[0]
    )

    # 5. 可选：还原为原始规模图（补全非LCC节点，保持与原始图节点数一致）
    if return_full_graph:
        original_node_count = original_data_info["original_node_count"]
        original_z_obs = original_data_info["original_z_obs"]

        # 补全特征矩阵（非LCC节点用0填充）
        x_full = torch.zeros((original_node_count, x_pert.shape[1]), dtype=torch.float32)
        x_full[lcc] = x_pert

        # 补全标签（非LCC节点保留原始标签）
        y_full = torch.tensor(original_z_obs, dtype=torch.long).unsqueeze(1)

        # 补全mask（非LCC节点设为False）
        train_mask_full = torch.zeros(original_node_count, dtype=torch.bool)
        val_mask_full = torch.zeros(original_node_count, dtype=torch.bool)
        test_mask_full = torch.zeros(original_node_count, dtype=torch.bool)
        train_mask_full[lcc] = train_mask_pert
        val_mask_full[lcc] = val_mask_pert
        test_mask_full[lcc] = test_mask_pert

        # 构建原始规模的PyG Data对象
        pyg_data_perturbed = Data(
            x=x_full,
            edge_index=edge_index_pert,  # 仅LCC内有边，非LCC节点为孤立节点
            y=y_full,
            train_mask=train_mask_full,
            val_mask=val_mask_full,
            test_mask=test_mask_full,
            num_nodes=original_node_count
        )
    else:
        pyg_data_perturbed = pyg_data_lcc

    # 6. 验证转换结果
    print(f"\n扰动数据转换为PyG格式完成：")
    print(f"- 节点数：{pyg_data_perturbed.num_nodes}")
    print(f"- 特征维度：{pyg_data_perturbed.x.shape}")
    print(f"- 边数：{pyg_data_perturbed.edge_index.shape[1]}")
    print(f"- 训练集大小：{pyg_data_perturbed.train_mask.sum().item()}")

    return pyg_data_perturbed


# ======================== 新增封装5：批量攻击测试集节点并统计成功率 ========================
def batch_attack_test_nodes(_A_obs, _X_obs, _z_obs, masks, gpu_id=None, 
                           perturb_structure=True, perturb_features=True,
                           sample_ratio=1.0):
    """
    批量攻击测试集节点，统计攻击成功率
    Args:
        _A_obs/_X_obs/_z_obs/masks: 由pyg_to_nettack_format返回的格式数据
        gpu_id (int/None): GPU配置
        perturb_structure/perturb_features: 是否扰动结构/特征
        sample_ratio (float): 测试集采样比例（0~1，避免测试集过大时耗时过久）
    Returns:
        attack_summary (dict): 攻击汇总结果（总节点数、成功数、成功率、失败节点列表等）
    """
    # 1. 提取所有测试集节点（原始ID）
    all_test_nodes = np.where(masks["test_mask"])[0]
    # 采样（可选，测试集过小时可全量，过大时采样）
    if sample_ratio < 1.0:
        sample_size = int(len(all_test_nodes) * sample_ratio)
        test_nodes = np.random.choice(all_test_nodes, size=sample_size, replace=False)
    else:
        test_nodes = all_test_nodes
    total_nodes = len(test_nodes)
    success_count = 0
    failed_nodes = []
    success_nodes = []

    print(f"\n开始批量攻击测试集节点：共{total_nodes}个节点（采样比例{sample_ratio}）")
    print("="*80)

    # 2. 遍历测试集节点执行攻击
    for idx, attack_node in enumerate(test_nodes):
        print(f"\n【进度{idx+1}/{total_nodes}】开始攻击节点{attack_node}")
        try:
            # 攻击准备
            nettack_ready = prepare_nettack(
                _A_obs, _X_obs, _z_obs, masks,
                gpu_id=gpu_id,
                attack_node=attack_node
            )
            # 执行攻击并判断是否成功
            _, _, attack_success = run_nettack(
                nettack_ready,
                perturb_structure=perturb_structure,
                perturb_features=perturb_features
            )
            # 统计结果
            if attack_success:
                success_count += 1
                success_nodes.append(attack_node)
            else:
                failed_nodes.append(attack_node)
        except Exception as e:
            print(f"节点{attack_node}攻击失败，原因：{str(e)}")
            failed_nodes.append(attack_node)

    # 3. 计算成功率并汇总结果
    success_rate = success_count / total_nodes * 100 if total_nodes > 0 else 0.0
    attack_summary = {
        "total_test_nodes_sampled": total_nodes,
        "success_count": success_count,
        "failed_count": len(failed_nodes),
        "success_rate(%)": round(success_rate, 2),
        "success_nodes": success_nodes,
        "failed_nodes": failed_nodes,
        "perturb_config": {
            "perturb_structure": perturb_structure,
            "perturb_features": perturb_features
        }
    }

    # 4. 打印汇总结果
    print("\n" + "="*80)
    print("批量攻击结果汇总：")
    print(f"- 采样测试集节点数：{total_nodes}")
    print(f"- 攻击成功节点数：{success_count}")
    print(f"- 攻击失败节点数：{len(failed_nodes)}")
    print(f"- 攻击成功率：{success_rate:.2f}%")
    print(f"- 成功节点列表：{success_nodes[:20]}..." if len(success_nodes) > 20 else f"- 成功节点列表：{success_nodes}")
    print(f"- 失败节点列表：{failed_nodes[:20]}..." if len(failed_nodes) > 20 else f"- 失败节点列表：{failed_nodes}")
    print("="*80)

    return attack_summary

# ======================== 主函数：支持单节点攻击+批量攻击 ========================
def main():
    """主函数：串联所有封装模块，支持单节点攻击可视化/批量攻击统计成功率"""
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
    

    # 1. 基础配置
    GPU_ID = None  # 如需GPU，设置为0/1等
    PERTURB_STRUCTURE = True  # 是否扰动结构
    PERTURB_FEATURES = True  # 是否扰动特征
    RETRAIN_ITERS = 5  # 单节点可视化时的重复训练次数
    SAMPLE_RATIO = 0.1  # 批量攻击时的测试集采样比例（建议先设0.1快速测试）

    # 2. 数据格式转换
    _A_obs, _X_obs, _z_obs, masks = pyg_to_nettack_format(data)

    # 3. 选择执行模式：批量攻击（统计成功率） or 单节点攻击（可视化）
    RUN_MODE = "batch"  # "batch" 批量攻击统计成功率 / "single" 单节点攻击可视化

    if RUN_MODE == "batch":
        # 批量攻击测试集节点并统计成功率
        attack_summary = batch_attack_test_nodes(
            _A_obs, _X_obs, _z_obs, masks,
            gpu_id=GPU_ID,
            perturb_structure=PERTURB_STRUCTURE,
            perturb_features=PERTURB_FEATURES,
            sample_ratio=SAMPLE_RATIO
        )
        # 可选：保存结果到文件
        np.save("nettack_batch_attack_summary.npy", attack_summary)
        print("批量攻击结果已保存到 nettack_batch_attack_summary.npy")

    elif RUN_MODE == "single":
        # 单节点攻击并可视化
        test_nodes = np.where(masks["test_mask"])[0]
        ATTACK_NODE = test_nodes[0]  # 待攻击节点（选第一个测试节点）
        # 攻击准备
        nettack_ready = prepare_nettack(
            _A_obs, _X_obs, _z_obs, masks,
            gpu_id=GPU_ID,
            attack_node=ATTACK_NODE
        )
        # 执行攻击
        nettack, attack_config = run_nettack(
            nettack_ready,
            perturb_structure=PERTURB_STRUCTURE,
            perturb_features=PERTURB_FEATURES
        )[:2]  # 单节点模式忽略attack_success返回值
        # 评估与可视化
        evaluate_and_visualize(nettack_ready, nettack, attack_config, retrain_iters=RETRAIN_ITERS)

# ======================== 一键运行入口 ========================
if __name__ == "__main__":
    # 设置随机种子保证可复现
    SEED = 15  # 随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    main()
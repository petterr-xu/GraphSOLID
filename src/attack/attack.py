import numpy as np
from tqdm import tqdm
import tensorflow as tf
import scipy.sparse as sp

from src.utils import graphbuilder
from ..dataset.abstract_dataset import AbstractDataModule
from src.gnn_meta_attack.metattack import meta_gradient_attack as mtk, utils

# class random_attacker():
#     pass

class Metattacker():
    def __init__(self, dataset_module:AbstractDataModule, share_perturbations, classifier, re_trainings=5, device=0, train_iters = 200):
        super().__init__()
        self.dataset_module = dataset_module
        self.GPU_ID = device
        self.share_perturbations = share_perturbations
        self.train_iters = train_iters
        self.RE_TRAININGS = re_trainings
        self.DTYPE = tf.float32
        self.ENFORCE_LL_CONSTRAINT = False

        self.classifier = classifier

    def poison(self):
        """
        向测试集投毒，并测试投毒后的测试结果
        """
        loader = self.dataset_module.test_dataloader()
        all_accuracies_clean = []
        all_accuracies_atk = []
        for data in loader:
            _A_obs, _X_obs, _z_obs, _Z_obs, _N, _K, data_mask = graphbuilder.pyg2matrix(data)
            split_train, split_val, split_unlabeled = graphbuilder.split_dataset_by_pyg_mask(data_mask)
            modified_adjacency = self._run_meta_attack_on_single_graph(
                _A_obs, _X_obs, _Z_obs, _N, _K, split_train, split_unlabeled, attack_variant="A-Train"
            )
            accuracies_clean, accuracies_atk = self._evaluate_accuracy_on_single_graph(
                _A_obs, modified_adjacency, _X_obs, _Z_obs, _z_obs, split_train, split_unlabeled
            )
            all_accuracies_clean.extend(accuracies_clean)
            all_accuracies_atk.extend(accuracies_atk)
        return all_accuracies_clean, all_accuracies_atk
        

    def _evaluate_accuracy_on_single_graph(self, _A_obs, modified_adjacency, _X_obs, _Z_obs, _z_obs, split_train, split_unlabeled):
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
        gcn_before_attack = mtk.GCNSparse(sp.csr_matrix(_A_obs), _X_obs, _Z_obs, hidden_sizes, gpu_id=self.GPU_ID)
        gcn_before_attack.build(with_relu=True)
        accuracies_clean = []
        for _it in tqdm(range(self.RE_TRAININGS), desc="Evaluating clean accuracy"):
            gcn_before_attack.train(split_train, initialize=True, display=False)
            accuracy_clean = (gcn_before_attack.logits.eval(session=gcn_before_attack.session).argmax(1) == _z_obs)[split_unlabeled].mean()
            accuracies_clean.append(accuracy_clean)

        # 2. 攻击后准确率
        gcn_after_attack = mtk.GCNSparse(sp.csr_matrix(modified_adjacency), _X_obs, _Z_obs, hidden_sizes, gpu_id=self.GPU_ID)
        gcn_after_attack.build(with_relu=True)
        accuracies_atk = []
        for _it in tqdm(range(self.RE_TRAININGS), desc="Evaluating attacked accuracy"):
            gcn_after_attack.train(split_train, initialize=True, display=False)
            accuracy_atk = (gcn_after_attack.logits.eval(session=gcn_after_attack.session).argmax(1) == _z_obs)[split_unlabeled].mean()
            accuracies_atk.append(accuracy_atk)

        return accuracies_clean, accuracies_atk

    def _run_meta_attack_on_single_graph(self, _A_obs, _X_obs, _Z_obs, _N, _K, split_train, split_unlabeled, attack_variant):
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
        :param share_perturbations: 扰动比例
        :return: modified_adjacency（扰动后的邻接矩阵）
        """
        # 1. 初始化并训练代理GCN
        hidden_sizes = [16]
        surrogate = mtk.GCNSparse(_A_obs, _X_obs, _Z_obs, hidden_sizes, gpu_id=self.GPU_ID)
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
        perturbations = int(self.share_perturbations * (_A_obs.sum() // 2))

        # 4. 初始化攻击器
        if approximate_meta_gradient:
            gcn_attack = mtk.GNNMetaApprox(
                _A_obs, _X_obs, labels_self_training, hidden_sizes,
                gpu_id=self.GPU_ID, _lambda=lambda_, train_iters=self.train_iters, dtype=self.DTYPE
            )
        else:
            gcn_attack = mtk.GNNMeta(
                _A_obs, _X_obs.astype("float32"), labels_self_training, hidden_sizes,
                gpu_id=self.GPU_ID, attack_features=False, train_iters=self.train_iters, dtype=self.DTYPE
            )

        # 5. 执行攻击
        gcn_attack.build()
        gcn_attack.make_loss(ll_constraint=self.ENFORCE_LL_CONSTRAINT)
        if approximate_meta_gradient:
            gcn_attack.attack(perturbations, split_train, split_unlabeled, idx_attack)
        else:
            gcn_attack.attack(perturbations, split_train, idx_attack)

        # 6. 获取扰动后的邻接矩阵
        modified_adjacency = gcn_attack.modified_adjacency.eval(session=gcn_attack.session)
        return modified_adjacency

# class nettack():
#     pass

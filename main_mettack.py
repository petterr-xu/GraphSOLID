import os
import torch
import hydra
import warnings
import numpy as np
import seaborn as sns
import os.path as osp
import tensorflow as tf
import scipy.sparse as sp
from args import parse_args
from omegaconf import DictConfig
from matplotlib import pyplot as plt


from src.attack.attacker import Metattacker
from src.utils import VNG_utils, graphbuilder
from src.nettack.nettack import nettack as ntk
from src.gnn_meta_attack.metattack import utils as metattack_utils
from src.utils.hetero_dataset_util import GraphDataLoader
from src.gnn_meta_attack.metattack import meta_gradient_attack as mtk
from src.DiGress.src import utils as digress_utils
from src.DiGress.src.diffusion_model_discrete import DiscreteDenoisingDiffusion 
# 1. 屏蔽 TensorFlow C++ 层面的日志 (3 = 仅致命错误)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# 2. 屏蔽 Python 库级别的所有警告 (如 Scipy, Numpy 的弃用警告)
warnings.filterwarnings("ignore")
# 禁用TF Eager Execution（原代码要求）
tf.compat.v1.disable_eager_execution()
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# ===================== 2. 全局配置 =====================
SEED = 15
SHARE_PERTURBATIONS = 0.15  # 扰动边的比例
DTYPE = tf.float32  # 内存不足可换tf.float16
ATTACK_VARIANT = "Meta-Self"  # 攻击变体（可选：Meta-Train/Meta-Self/A-Meta-Train等）
ENFORCE_LL_CONSTRAINT = False

def load_imb_data(dataset, imb_ratio = 0, keep_edge=True,device='cpu'):
    root_path = osp.dirname(osp.realpath(__file__))
    loader = GraphDataLoader()
    data_path = osp.join(root_path, 'data', dataset, 'data', dataset + '.mat')
    cnfg_path = osp.join(root_path, 'data', dataset, 'meta', dataset + '.json')
    hetero_ctx = loader.load_from_config(cnfg_path, data_path)
    target = hetero_ctx.target_node  # 'review' 或 'user'
    data = hetero_ctx.g.to(device)
    n_feat = hetero_ctx.n_features
    n_cls = hetero_ctx.n_classes
    print(data)
    if imb_ratio == 0:
        return hetero_ctx
    
    max_n=500
    if dataset in ['YelpChi', 'Amazon-Products']:
        data_train_mask, data_val_mask, data_test_mask = data[target].train_mask.clone(), data[target].val_mask.clone(), data[target].test_mask.clone()
        stats = data[target].y[data_train_mask]
        n_data = []
        for i in range(n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        idx_info = VNG_utils.get_idx_info(data[target].y, n_cls, data_train_mask)
        class_num_list = n_data
        print("num of class in original training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
        class_num_list, data_train_mask, _, edge_mask_dict = graphbuilder.make_hetero_longtailed_data_remove(data, target, n_data, n_cls, imb_ratio, data_train_mask.clone(), max_n)
        # 更新 HeteroData
        hetero_ctx.g[hetero_ctx.target_node].train_mask = data_train_mask
        # 更新边索引 (可选，取决于是否想物理删除边)
        if not keep_edge:
            for etype, mask in edge_mask_dict.items():
                hetero_ctx.g[etype].edge_index = hetero_ctx.g[etype].edge_index[:, mask]
        print("num of class in LT-training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
        minority_mask = class_num_list < (sum(class_num_list)/n_cls)
        minority_class = [i for i in range(n_cls) if minority_mask[i]]
        print("minority classes {}".format(minority_class))
    else:
        raise NotImplementedError("Not implemented for dataset {}".format(dataset))
    
    return hetero_ctx

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

@hydra.main(version_base='1.3', config_path='./configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    hetero_data = load_imb_data(dataset_config["name"])
    if dataset_config["name"] in ['YelpChi', 'Amazon-Products']:
        from src.dataset.YelpChi_dataset import YelpChihDataModule, YelpChiDatasetInfos
        from src.DiGress.src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from src.DiGress.src.analysis.visualization import NonMolecularVisualization
        from src.DiGress.src.analysis.spectre_utils import YelpChiSamplingMetrics
        from src.DiGress.src.diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
        from src.DiGress.src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
        datamodule = YelpChihDataModule(cfg, hetero_data.g)
        if(dataset_config["name"]=='YelpChi'):
            sampling_metrics = YelpChiSamplingMetrics(datamodule,cfg)
        else:
            sampling_metrics = None # todo

        # dataset_infos = YelpChiSubgraphDatasetInfos(datamodule, cfg)
        dataset_infos = YelpChiDatasetInfos(datamodule, cfg)
        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = None # NonMolecularVisualization()

        '''
        todo: extra features for hetero graph
        '''
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        # dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
        #                                         domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    digress_utils.create_folders(cfg)
    path = '/root/autodl-tmp/outputs/2026-01-26/16-51-37-graph-tf-model/checkpoints/graph-tf-model/last-v1.ckpt'
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(path, **model_kwargs)
    model = model.to('cuda:0')
    
    gpu = cfg.general.gpus
    if gpu == 0:
        gpuid = None
    else:
        gpuid = gpu - 1
    metattacker = Metattacker(datamodule,perturb_ratio=SHARE_PERTURBATIONS,re_trainings=5,device=gpuid,train_iters = 200)
    accuracies_clean, accuracies_atk = metattacker.poison()
    # 打印关键结果
    print(f"Clean Accuracy (mean±std): {np.mean(accuracies_clean):.4f} ± {np.std(accuracies_clean):.4f}")
    print(f"Attacked Accuracy (mean±std): {np.mean(accuracies_atk):.4f} ± {np.std(accuracies_atk):.4f}")
    # 步骤5：绘制并保存结果图
    plot_accuracy(accuracies_clean, accuracies_atk)


if __name__ == '__main__':
    # 固定随机种子（可选）
    np.random.seed(SEED)
    tf.compat.v1.set_random_seed(SEED)
    # 执行主流程
    main()

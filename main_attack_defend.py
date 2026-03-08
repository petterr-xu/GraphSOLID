import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import csv
import torch
import hydra
import random
import warnings
import numpy as np
import seaborn as sns
import os.path as osp
import tensorflow as tf
import scipy.sparse as sp
from args import parse_args
from omegaconf import DictConfig
from matplotlib import pyplot as plt

from src import loss_fn
from src.models import HeteroNN, classifier_engine
from src.attack.attacker import Metattacker, RandomAttacker, MintaAttacker
from src.attack.pipeline import DefaultPipeline
from src.attack.surrogate_pipeline import SurrogateAttackPipeline
from src.attack.defender import DiffusionPurifyDefender
from src.utils import VNG_utils, graphbuilder
from src.utils.hetero_dataset_util import GraphDataLoader
from src.DiGress.src import utils as digress_utils
from src.DiGress.src.diffusion_model_discrete import DiscreteDenoisingDiffusion 
tf.get_logger().setLevel("ERROR")
# 禁用TF Eager Execution（原代码要求）
tf.compat.v1.disable_eager_execution()
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# ===================== 2. 全局配置 =====================
DEFAULT_SEED = 15
SHARE_PERTURBATIONS = 0.15  # 扰动边的比例
DTYPE = tf.float32  # 内存不足可换tf.float16
ATTACK_VARIANT = "Meta-Self"  # 攻击变体（可选：Meta-Train/Meta-Self/A-Meta-Train等）
ENFORCE_LL_CONSTRAINT = False


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tf.compat.v1.set_random_seed(seed)


def _is_number(v):
    return isinstance(v, (int, float, np.number)) and not isinstance(v, bool)


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path, rows):
    if rows is None or len(rows) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary_rows(trial_rows):
    if trial_rows is None or len(trial_rows) == 0:
        return []
    numeric_keys = set()
    excluded = {"trial", "seed"}
    for row in trial_rows:
        for k, v in row.items():
            if k not in excluded and _is_number(v):
                numeric_keys.add(k)
    out = []
    for k in sorted(numeric_keys):
        vals = np.array([_to_float(r.get(k, float("nan"))) for r in trial_rows], dtype=float)
        non_nan = vals[~np.isnan(vals)]
        out.append(
            {
                "metric": k,
                "mean": float(np.nanmean(vals)) if non_nan.size > 0 else float("nan"),
                "var": float(np.nanvar(vals)) if non_nan.size > 0 else float("nan"),
                "count_non_nan": int(non_nan.size),
                "count_trials": int(len(trial_rows)),
            }
        )
    return out


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

@hydra.main(version_base='1.3', config_path='./configs', config_name='config')
def main(cfg: DictConfig):
    run_seed = int(getattr(cfg.train, "seed", DEFAULT_SEED))
    set_global_seed(run_seed)

    dataset_config = cfg["dataset"]
    hetero_data = load_imb_data(dataset_config["name"])
    if dataset_config["name"] in ['YelpChi', 'Amazon-Products']:
        from src.dataset.YelpChi_multiview_dataset import YelpChihDataModule, YelpChiDatasetInfos
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
    path = '/root/autodl-tmp/outputs/curr/graph-tf-model/checkpoints/graph-tf-model/last-v1.ckpt'
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(path, **model_kwargs)
    
    gpu = cfg.general.gpus
    if gpu == 0:
        gpuid = None
        device = 'cpu'
    else:
        gpuid = gpu - 1
        device = f'cuda:{gpuid}'
    model = model.to(device)
    target = cfg.dataset.target
    nclass = hetero_data.n_classes
    print(hetero_data.g.metadata())

    trial_runs = int(getattr(cfg.general, "trial_runs", 1))
    trial_base_seed = int(getattr(cfg.general, "trial_base_seed", run_seed))
    save_trial_csv = bool(getattr(cfg.general, "save_trial_csv", True))
    trial_csv_name = str(getattr(cfg.general, "trial_csv_name", "attack_defend_per_trial.csv"))
    summary_csv_name = str(getattr(cfg.general, "summary_csv_name", "attack_defend_summary.csv"))

    trial_rows = []
    last_result = None

    for trial_idx in range(trial_runs):
        trial_seed = trial_base_seed + trial_idx
        set_global_seed(trial_seed)

        if cfg.general.attack_method == 'metattack':
            attacker = Metattacker(datamodule,perturb_ratio=cfg.general.general_edge_perturb_ratio,re_trainings=5,device=gpuid,train_iters = 200)
        elif cfg.general.attack_method == 'random':
            attacker = RandomAttacker(datamodule, perturb_ratio=cfg.general.general_edge_perturb_ratio)
        elif cfg.general.attack_method == 'minta':
            minta_positive_label = int(getattr(cfg.general, "minta_positive_label", 1))
            attacker = MintaAttacker(
                datamodule,
                perturb_ratio=cfg.general.sub_edge_perturb_ratio,
                ctrl_nodes_size=cfg.general.ctrl_nodes_size,
                target_label=minta_positive_label,
                only_attack_correctly_detected=bool(getattr(cfg.general, "minta_only_attack_correctly_detected", True)),
                surrogate_epochs=int(getattr(cfg.general, "minta_surrogate_epochs", 50)),
                surrogate_early_stop_patience=int(getattr(cfg.general, "minta_surrogate_patience", 10)),
                surrogate_early_stop_min_delta=float(getattr(cfg.general, "minta_surrogate_min_delta", 1e-3)),
                enable_feature_perturb = False, # 目前仅攻击结构
                device=device,
            )
        else:
            raise NotImplementedError("Unknown attack method {}".format(cfg.general.attack_method))

        diffusionDefender = DiffusionPurifyDefender(
            diffusion_steps=cfg.general.purify_steps,
            diffusion_model=model,
            metapaths=cfg.dataset.metapaths,
            target_node_type=cfg.dataset.target,
        )

        classifier = HeteroNN.HeteroGNN_classifier(
            net=cfg.general.net,
            target_node=target,
            metadata=hetero_data.g.metadata(),
            nhid=cfg.general.feat_dim,
            nclass=nclass,
            nlayer=cfg.general.n_layers,
            dropout=0.5,
        ).to(device)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        train_mask = hetero_data.g[target].train_mask
        class_count = torch.bincount(hetero_data.g[target].y[train_mask].view(-1), minlength=nclass).to(device,torch.float)
        classifier_criterion = loss_fn.IMB_LOSS("ce",nclass,class_count.detach().cpu().numpy(),device=device)
        cl_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            classifier_optimizer,
            mode='min',
            factor = 0.5,
            patience = 100,
            verbose=False,
        )
        minta_victim_engine = str(getattr(cfg.general, "minta_victim_engine", "hetero")).lower()
        if minta_victim_engine == "dense_surrogate":
            input_dim = int(hetero_data.g[target].x.size(-1))
            same_type_edge_types = [et for et in hetero_data.g.edge_types if et[0] == target and et[2] == target]
            classifier_eg = classifier_engine.MintaSurrogateEngine(
                input_dim=input_dim,
                hidden_dim=cfg.general.feat_dim,
                num_classes=nclass,
                target_node=cfg.dataset.target,
                edge_types_for_adj=same_type_edge_types,
                device=device,
            ).to(device)
        else:
            # Default: use the same hetero classifier family as victim for MintA evaluation.
            classifier_eg = classifier_engine.HeteroClassifierEngine(
                model=classifier,
                optimizer=classifier_optimizer,
                criterion=classifier_criterion,
                scheduler=cl_scheduler,
                target_node=cfg.dataset.target,
                device=device,
            ).to(device)

        minta_pipeline = SurrogateAttackPipeline(
            dataset_module=datamodule,
            attacker=attacker,
            classifier_engine=classifier_eg,
            defender=diffusionDefender,
            device=device,
        )
        result = minta_pipeline.evasion_then_defend(
            split='test',
            train_split=getattr(cfg.general, "minta_train_split", "train"),
            train_epochs=int(getattr(cfg.general, "minta_victim_epochs", 200)),
            eval_split='test',
            log_every=int(getattr(cfg.general, "minta_log_every", 0)),
            positive_label=int(getattr(cfg.general, "minta_positive_label", 1)),
            early_stop_patience=int(getattr(cfg.general, "minta_victim_patience", 10)),
            early_stop_min_delta=float(getattr(cfg.general, "minta_victim_min_delta", 1e-3)),
            train_log_interval=int(getattr(cfg.general, "minta_train_log_interval", 10)),
        )
        last_result = result

        metrics = result.get("metrics", {})
        trial_row = {"trial": int(trial_idx), "seed": int(trial_seed)}
        for k, v in metrics.items():
            trial_row[k] = _to_float(v) if _is_number(v) else v
        trial_rows.append(trial_row)
        print(f"[trial {trial_idx + 1}/{trial_runs}] seed={trial_seed} metrics={metrics}")

    if last_result is not None:
        print(last_result["metrics"])

    if save_trial_csv and len(trial_rows) > 0:
        out_dir = cfg.general.output_dir if hasattr(cfg.general, "output_dir") else "./outputs"
        trial_csv_path = os.path.join(out_dir, trial_csv_name)
        summary_csv_path = os.path.join(out_dir, summary_csv_name)
        _write_csv(trial_csv_path, trial_rows)
        _write_csv(summary_csv_path, _build_summary_rows(trial_rows))
        print(f"[trial_csv] saved per-trial metrics to {trial_csv_path}")
        print(f"[trial_csv] saved summary(mean/var) to {summary_csv_path}")


if __name__ == '__main__':
    main()

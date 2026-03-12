import sys
import os
# 将 src 目录加入路径，这样 Python 就能找到 src 下的模块
sys.path.append(os.path.join(os.getcwd(), "src"))
import rdkit
import os
import copy
import hydra
import torch
import random
import warnings
import statistics
import numpy as np
import os.path as osp
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch_geometric.utils import train_test_split_edges,negative_sampling

from src import solid
from args import parse_args
from src.utils import VNG_utils, graphbuilder
from solid_trainer import SolidTrainer
from src.utils.hetero_dataset_util import GraphDataLoader
from src.DiGress.src import utils
from src.DiGress.src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.models import gnn,sage,edge_learner,teacher,diffusion,mlp,HeteroNN
from src.denoise import unet
warnings.filterwarnings("ignore")

os.environ["WANDB_MODE"] = "disabled"

args = parse_args()
print(args)
reweight = False
timestamp_format = "%Y%m%d_%H%M%S"


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
    dataset_config = cfg["dataset"]
    hetero_data = load_imb_data(dataset_config["name"])
    if dataset_config["name"] in ['YelpChi', 'Amazon-Products']:
        from src.dataset.YelpChi_dataset import YelpChihDataModule, YelpChiDatasetInfos
        from src.dataset.AmazonProducts_dataset import AmPdDataModule, AmPdDatasetInfos
        from src.DiGress.src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from src.DiGress.src.analysis.visualization import NonMolecularVisualization
        from src.DiGress.src.analysis.spectre_utils import YelpChiSamplingMetrics, AmPdSamplingMetrics
        from src.DiGress.src.diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
        from src.DiGress.src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
        if(dataset_config["name"]=='YelpChi'):
            datamodule = YelpChihDataModule(cfg, hetero_data.g)
            sampling_metrics = YelpChiSamplingMetrics(datamodule,cfg)
            dataset_infos = YelpChiDatasetInfos(datamodule, cfg)
        elif(dataset_config["name"]=='Amazon-Products'):
            datamodule = AmPdDataModule(cfg, hetero_data.g)
            sampling_metrics = AmPdSamplingMetrics(datamodule, cfg)
            dataset_infos = AmPdDatasetInfos(datamodule, cfg)
        else:
            raise NotImplementedError("Unknown dataset {}".format(dataset_config["name"]))


        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization()

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

    utils.create_folders(cfg)
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                    #   strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    if cfg.general.name not in ['debug', 'test']:
        trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    main()

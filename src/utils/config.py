from dataclasses import dataclass
import dgl
import torch.nn as nn
@dataclass
class TrainConfig:
    learning_rate:float
    epochs:int
    batch_size:int
    save_cp:int
    patience:int

@dataclass
class GraphDatasetConfig:
    graph_id:str
    graph_dataset:dgl.DGLGraph
    file_path:str = None
    split_schedule:dict = None
    load_cache_file:bool = False
    padding:tuple = (0,0,0,0)
    dgl_feature_field_name:str = "feat"
    num_classes:int = 7
    amp_scale:float = 1

@dataclass
class DiffusionConfig(TrainConfig):
    T:int = 1000
    beta:tuple = (0.0001, 0.02)
    beta_schedule:str = "lin" #{"lin","exp","quad"}
    guidance_drop_prob:float = 0.15
    guidance:float = 6
    SMOTE_aug:bool = False
    SMOTE_kneighbors:int = 5
    is_confidence_guide:bool = True
    teacher_model:nn.Module = None
    temperature:float = 10.0

@dataclass
class ClassifierConfig(TrainConfig):
    criterion:nn.Module = None# 用于训练分类器的损失函数
    noise_aug:bool = False
    T:int = 1000 # 用于生产噪声样本训练分类器
    SMOTE_aug:bool = False
    
@dataclass
class UnetConfig:
    vector_channels: int = 1
    feature_length: int = 1440
    n_channels: int = 32
    n_length: int = 100
    num_class: int = 10
    class_embedding_channel:int = 4 * 32
    time_embedding_channel:int = 4 * 32
    ch_mults: tuple = (1, 2, 4)
    is_attn:tuple = (False, False, True)
    n_blocks: int = 2

@dataclass
class EncoderConfig(TrainConfig):
    input_size: int = 1433
    latent_size: int = 10


@dataclass
class VAEConfig(EncoderConfig):
    kl_weight: float = 0.5
    n_block: int = 4

@dataclass
class GraphEncoderConfig(EncoderConfig):
    layers:int = 3
    gnn:str = "gcn"
    feature_field_name:str = "feat"
    dataset:dgl.DGLGraph = None
    graph_file:str = None
    beta:float = 4. # 联合损失的加权系数
    alpha_lr:float = 0.001 # 中心损失的学习率
    padding:tuple=(0,0,0,0)
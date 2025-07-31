import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import diffusion,gnn,gsl
from utils.config import GraphDatasetConfig,DiffusionConfig,UnetConfig,GraphEncoderConfig

class GNN_Encoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.gnn_encoder = gnn.JointGNN(args.layers,args.gnn_type,args.gnn_in_feats,args.gnn_out_feats)
        self.edge_learner = gsl.EdgeLearner(args.gnn_out_feats)
        self.diffusion_model = diffusion.GDDPMblock(args.eps_model,args.beta,args.n_steps,args.device)
    def forward(self,g:dgl.DGLGraph,x:torch.Tensor):
        h,recon_feat = self.gnn_encoder(g,x)
        edge_score = self.edge_learner(g,h)
        return recon_feat,edge_score

def train_cg_vng(graph_dataset_config:GraphDatasetConfig, diffusion_config:DiffusionConfig, denoise_config:UnetConfig, gnn_encoder_config:GraphEncoderConfig, device="cuda:0"):
    pass

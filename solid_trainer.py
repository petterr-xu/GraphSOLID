import copy
import torch
import random
import warnings
import statistics
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import train_test_split_edges,negative_sampling
from sklearn.metrics import balanced_accuracy_score, f1_score,classification_report,roc_auc_score, accuracy_score, recall_score

from args import parse_args
from src import solid,loss_fn
from src.data import TabDataset
from src.utils import VNG_utils,tab_dataset_util
from src.TabDiff.tabdiff.metrics import TabMetrics
from src.TabDiff.tabdiff.modules.main_modules import UniModMLP
from src.TabDiff.tabdiff.modules.main_modules import Model
from src.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from src.models import gnn,sage,gcn,gat,edge_learner,teacher,diffusion,mlp
from src.denoise import unet
import src.utils.graphbuilder

timestamp_format = "%Y%m%d_%H%M%S"
root_path = osp.dirname(osp.realpath(__file__))
class SolidTrainer:
    def __init__(self, tabgraph:TabDataset, data_train_mask, data_val_mask, diffusion,teacher:nn.Module, edge_learner:nn.Module, classifier:nn.Module,
                  diff_lr=1e-4,tearch_lr=1e-3,el_lr=1e-3, cl_lr=1e-3, diff_bs = 64, device='cuda:0'):
        self.tabgraph = tabgraph
        self.data_train_mask = data_train_mask
        self.data_val_mask = data_val_mask
        self.diff_bs = diff_bs

        self.diffusion = diffusion
        self.teacher = teacher
        self.edge_learner = edge_learner
        self.classifier = classifier
        self.data = tabgraph.graph.to(device)
        self.device = device

        self.teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=tearch_lr)

        self.dif_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=diff_lr)
    
    def train_teacher_oneloop(self):
        teacher_model = self.teacher
        teacher_optimizer = self.teacher_optimizer
        
        teacher_model.train()
        teacher_optimizer.zero_grad()
        inputs = self.data.x.to(self.device)
        logits = teacher_model(inputs[self.data_train_mask])
        loss = F.cross_entropy(logits,self.data.y[self.data_train_mask])
        loss.backward()
        teacher_optimizer.step()
        with torch.no_grad():
            teacher_model.eval()
            logits = teacher_model(inputs[self.data_val_mask])
            val_loss = F.cross_entropy(logits,self.data.y[self.data_val_mask])
        return val_loss
    
    def train_teacher(self, epochs):
        best_loss = float('inf')
        patience = 5
        patience_count = 0
        patience_beta = 1e-3
        with tqdm(total=epochs, desc="Teacher Training Progress") as pbar:
            for e in range(epochs):
                val_loss = self.train_teacher_oneloop()
                if val_loss < (best_loss - patience_beta):
                    best_loss = val_loss
                    patience_count = 0
                else: patience_count += 1
                pbar.set_postfix({
                    'Val Loss': f'{val_loss:.4f}', 
                    'Patience': f'{patience_count}/{patience}'
                })
                pbar.update(1)
                if patience_count >= patience:
                    pbar.write(f"Early stopping at epoch {e+1}")
                    pbar.close()
                    break
    
    def train_tabdiff_oneloop(self,args):
        device = self.device
        n_cls = self.tabgraph.n_labels
        train_feat_data = self.data.x[self.data_train_mask]
        train_label_data = self.data.y[self.data_train_mask]
        eval_feat = self.data.x[self.data_val_mask]
        eval_label = self.data.y[self.data_val_mask]
        loss_value = 0.0
        class_dis = VNG_utils.class_dis(self.data.y[self.data_train_mask],n_cls)
        class_dis = class_dis / sum(class_dis)
        class_mask = VNG_utils.dis_based_class_mask(train_label_data,class_dis,n_cls,train_label_data.shape[0],args.guidance_drop_prob,adjustment_factor=args.adjustment_factor,device=device)
        class_mask = class_mask[:, None].repeat(1,n_cls).to(torch.int32)
        class_mask = (-1*(1-class_mask))
        # print("{} guidance, hard label of guidance is:".format(sum(class_mask)))
        # print(train_label_data[class_mask].argmax(1))
        train_dataset = TensorDataset(train_feat_data, train_label_data, class_mask)
        data_loader = DataLoader(train_dataset, self.diff_bs, shuffle=True)
        for inputs,targets,c_mask in data_loader:
            targets = F.one_hot(targets,num_classes=n_cls)
            soft_labels = self.teacher_model.softmax_with_temperature(inputs,args.temperature)
            targets = soft_labels * (1. - args.hard_factor) + targets * args.hard_factor
            # print(inputs.shape)
            self.dif_optimizer.zero_grad()
            targets = targets.to(device)
            # 按照一定的概率将guidance置空，由此只用一个backbone训练出适用于两种情况（有无条件）的模型
            dloss, closs = self.diffusion.mixed_loss(inputs,targets,c_mask.to(torch.int32))
            loss = args.dloss_weight * dloss + args.closs_weight * closs
            loss.backward()
            self.dif_optimizer.step()
            loss_value += loss.item()
        val_class_mask = VNG_utils.dis_based_class_mask(eval_label,class_dis,n_cls,eval_label.shape[0],args.guidance_drop_prob,adjustment_factor=args.adjustment_factor,device=device)
        eval_data = [eval_feat,eval_label,val_class_mask.to(torch.int32)]
        val_loss = self.eval_diffusion_model(eval_data,args)
        return val_loss

    @torch.no_grad()
    def eval_diffusion_model(self,eval_data,args):
        device = self.device
        n_cls = self.tabgraph.n_labels
        self.diffusion.eval()
        with torch.no_grad():
            feat_dataset = TensorDataset(eval_data[0],eval_data[1],eval_data[2])
            test_data_loader = DataLoader(feat_dataset, batch_size=32, shuffle=True)
            data_size = len(test_data_loader.dataset)
            loss_value = 0.0
            for inputs,targets,c_mask in test_data_loader:
                inputs = inputs.to(device)
                targets = F.one_hot(targets,num_classes=n_cls)
                soft_labels = self.teacher_model.softmax_with_temperature(inputs,args.temperature)
                targets = soft_labels * (1. - args.hard_factor) + targets * args.hard_factor
                targets = targets.to(device)
                # class_mask = (torch.rand(targets.shape[0]) < 0.15).to(device,torch.int32)
                # c_mask = torch.ones_like(c_mask,device=device)
                dloss, closs = self.diffusion.mixed_loss(inputs,targets,c_mask.to(torch.int32))
                loss = args.dloss_weight * dloss + args.closs_weight * closs
                loss_value += loss.item()
        return loss_value / data_size
    
    def train_diffusion(self,args):
        best_loss = float('inf')
        patience = 5
        patience_count = 0
        patience_beta = 2e-4
        dif_epoch = 1000
        with tqdm(total=dif_epoch, desc="Diffusion Training") as pbar:
            for e in range(dif_epoch):
                val_loss = self.train_tabdiff_oneloop()
                if val_loss < (best_loss - patience_beta):
                    best_loss = val_loss
                    patience_count = 0
                else: patience_count += 1
                pbar.set_postfix({
                    'Val Loss': f'{val_loss:.4f}', 
                    'Best Loss': f'{best_loss:.4f}',
                    'Patience': f'{patience_count}/{patience}'
                })
                pbar.update(1)
                if (e+1) % 10 == 0:
                    ts = datetime.now().strftime(timestamp_format)
                    ckpt_path = osp.join(root_path, "ckpt","tabdiff",args.dataset,"tabdiff_" + args.dataset+"_"+ts+f"_e{e}_"+".pth")
                    VNG_utils.save(self.diffusion,ckpt_path)
                if patience_count >= patience:
                    pbar.write(f"Early stopping at epoch {e+1}")
                    pbar.close()
                    break
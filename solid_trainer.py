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

from src import solid,loss_fn
from src.data import TabDataset
from src.utils import VNG_utils
from src.utils.hetero_dataset_util import HeteroGraphContext

timestamp_format = "%Y%m%d_%H%M%S"
root_path = osp.dirname(osp.realpath(__file__))
class SolidTrainer:
    def __init__(self, ctx: HeteroGraphContext, diffusion: nn.Module, teacher: nn.Module, edge_learner: nn.Module,  
                    classifier: nn.Module, encoder: nn.Module = None,
                    diff_lr=1e-4, tearch_lr=1e-3, el_lr=1e-3, cl_lr=1e-3, en_lr=1e-3, cent_lr=1e-3, 
                    n_hid=512, diff_bs=64, r=None, plot=False, save_model=True, device='cuda:0'):
            
        self.ctx = ctx
        self.target = ctx.target_node
        self.device = device
        
        self.data = ctx.g.to(device)
        self.edge_index_dict = self.data.edge_index_dict
        self.data_train_mask = self.data[self.target].train_mask
        self.data_val_mask = self.data[self.target].val_mask
        self.data_test_mask = self.data[self.target].test_mask
        
        self.diff_bs = diff_bs
        self.repeatition = r
        self.plot = plot
        self.save_model = save_model
        
        self.diffusion = diffusion.to(device)
        self.teacher = teacher.to(device)
        self.decoder = edge_learner.to(device)
        self.classifier = classifier.to(device)
        self.encoder = encoder.to(device) if encoder else None

        # 5. 优化器初始化 (逻辑不变)
        self.teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=tearch_lr)
        self.dif_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=diff_lr)
        self.de_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=el_lr)
        
        if self.encoder is not None:
            self.en_optimizer = torch.optim.Adam(encoder.parameters(), lr=en_lr)
            # 使用 ctx.num_classes 动态设置类别
            self.centloss_criterion = loss_fn.CenterLoss(self.ctx.num_classes, n_hid).to(device)
            self.centloss_optimizer = torch.optim.Adam(self.centloss_criterion.parameters(), lr=cent_lr)

        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=cl_lr)
        self.classifier_criterion = loss_fn.CrossEntropy().to(device)

    def cent_pretrain_oneloop(self,args):
        device = self.device
        decoder = self.decoder
        decoder.train()
        self.encoder.train()
        self.centloss_criterion.train()
        self.en_optimizer.zero_grad()
        self.centloss_optimizer.zero_grad()
        neg_edge_index = negative_sampling(
            edge_index=self.data.train_pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=self.data.train_pos_edge_index.size(1))
        edge_labels = torch.cat([torch.ones(self.data.train_pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))]).to(self.device)
        emb = self.encoder(self.data.x, self.edge_index[:,self.train_edge_mask], None)
        cent_loss = self.centloss_criterion(emb[self.data_train_mask],self.data.y[self.data_train_mask])
        pos_edge_scores = self.decoder(emb,self.data.train_pos_edge_index)
        neg_edge_scores = self.decoder(emb, neg_edge_index)
        edge_scores = torch.cat([pos_edge_scores,neg_edge_scores],dim=0)
        de_loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels)
        loss = args.w_con_loss * cent_loss+ de_loss
        loss.backward()
        # for param in centloss_criterion.parameters():
        #     param.grad.data *= (1./args.w_con_loss)
        with torch.no_grad():
            self.encoder.eval()
            self.centloss_criterion.eval()
            self.decoder.eval()
            emb = self.encoder(self.data.x, self.edge_index[:,self.train_edge_mask], None)
            val_cent_loss = self.centloss_criterion(emb[self.data_val_mask],self.data.y[self.data_val_mask])
            val_pos_edge_scores = self.decoder(emb, self.data.val_pos_edge_index)
            val_neg_edge_scores = self.decoder(emb, self.data.val_neg_edge_index.to(self.device))
            val_edge_scores = torch.cat([val_pos_edge_scores,val_neg_edge_scores],dim=0)
            val_edge_labels = torch.cat([torch.ones(self.data.val_pos_edge_index.size(1)), torch.zeros(self.data.val_neg_edge_index.size(1))]).to(self.device)
            val_recon_loss = F.binary_cross_entropy_with_logits(val_edge_scores, val_edge_labels)
            val_loss = args.w_con_loss * val_cent_loss + val_recon_loss
        self.en_optimizer.step()
        # cent_scheduler.step(val_cent_loss)
        self.centloss_optimizer.step()
        # de_optimizer.step()
        self.de_scheduler.step(val_recon_loss)
        return val_loss,val_cent_loss, val_recon_loss

    def cent_pretrain(self, args):
        best_loss = float('inf')
        patience = 20
        patience_count = 0
        patience_beta = 1e-3
        pre_epoch = 2000
        with tqdm(total=pre_epoch, desc="Pre-train") as pbar:
            for e in range(pre_epoch):
                val_loss, con_loss, recon_loss = self.cent_pretrain_oneloop(args)
                if val_loss < (best_loss - patience_beta):
                    best_loss = val_loss
                    patience_count = 0
                else: patience_count += 1
                pbar.set_postfix({
                    'Val Loss': f'{val_loss:.4f}', 
                    'Con Loss': f'{con_loss:.4f}', 
                    'Recon Loss': f'{recon_loss:.4f}',
                    'Patience': f'{patience_count}/{patience}'
                })
                pbar.update(1)
                if patience_count >= patience:
                    pbar.write(f"Early stopping at epoch {e+1}")
                    pbar.close()
                    break

        self.encoder.eval()
        embbeddings = self.encoder(self.data.x, self.edge_index[:,self.train_edge_mask], None).detach()
        emb_data = copy.deepcopy(self.data)
        emb_data.x = embbeddings
        self.emb_data = emb_data
        return emb_data

    def cover_data_with_emb(self):
        assert self.emb_data is not None, "Please run cent_pretrain() before calling this method."
        target = self.target
        new_embeddings = self.emb_data[target].x
        self.data[target].x = new_embeddings
        self.ctx.g[target].x = new_embeddings
        
        
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
    
    def train_teacher(self, epochs, skip = False, ckpt_path = None, ckpt_save_epoch = 0):
        if skip:
            assert ckpt_path is not None, "Please provide a valid checkpoint path to load the teacher model."
            VNG_utils.load(self.teacher, ckpt_path)
            print(f"Loaded teacher model from {ckpt_path}")
            return
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
                if ckpt_save_epoch > 0 and ((e+1) % ckpt_save_epoch == 0):
                    ts = datetime.now().strftime(timestamp_format)
                    path = osp.join(root_path, "ckpt","teacher",self.tabgraph.name,"teacher_" + self.tabgraph.name+"_"+ts+f"_e{e}_"+".pth")
                    VNG_utils.save(self.teacher,path)
                if patience_count >= patience:
                    pbar.write(f"Early stopping at epoch {e+1}")
                    pbar.close()
                    break

    @torch.no_grad()
    def teacher_test(self, x, y):
        """
        use teacher model to predict and evaluate the performance on given x and y
        
        :param self: 
        :param x: feature matrix
        :param y: labels
        :return: accuracy, f1 score, recall, classification report
        """
        self.teacher.eval()
        with torch.no_grad():
            logits = self.teacher(x.to(self.device))
            pred = logits.max(1)[1]
            y_pred = pred.cpu().numpy()
            y_true = y.cpu().numpy()
            acc = pred.eq(y.to(self.device)).sum().item() / y.shape[0]
            f1 = f1_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average=None)
            measure_result = classification_report(y_true, y_pred,digits=4, zero_division=np.nan)
        return acc, f1, recall, measure_result
    
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
            soft_labels = self.teacher.softmax_with_temperature(inputs,args.temperature)
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
                soft_labels = self.teacher.softmax_with_temperature(inputs,args.temperature)
                targets = soft_labels * (1. - args.hard_factor) + targets * args.hard_factor
                targets = targets.to(device)
                # class_mask = (torch.rand(targets.shape[0]) < 0.15).to(device,torch.int32)
                # c_mask = torch.ones_like(c_mask,device=device)
                dloss, closs = self.diffusion.mixed_loss(inputs,targets,c_mask.to(torch.int32))
                loss = args.dloss_weight * dloss + args.closs_weight * closs
                loss_value += loss.item()
        return loss_value / data_size
    
    def train_diffusion(self,args, skip = False, ckpt_path = None, ckpt_save_epoch = 10):
        if skip:
            assert ckpt_path is not None, "Please provide a valid checkpoint path to load the diffusion model."
            VNG_utils.load(self.diffusion, ckpt_path)
            print(f"Loaded diffusion model from {ckpt_path}")
            return
        best_loss = float('inf')
        patience = 5
        patience_count = 0
        patience_beta = 2e-4
        dif_epoch = 1000
        with tqdm(total=dif_epoch, desc="Diffusion Training") as pbar:
            for e in range(dif_epoch):
                val_loss = self.train_tabdiff_oneloop(args)
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
                if ckpt_save_epoch > 0 and (e+1) % ckpt_save_epoch == 0:
                    ts = datetime.now().strftime(timestamp_format)
                    ckpt_path = osp.join(root_path, "ckpt","tabdiff",args.dataset,"tabdiff_" + args.dataset+"_"+ts+f"_e{e}_"+".pth")
                    VNG_utils.save(self.diffusion,ckpt_path)
                if patience_count >= patience:
                    pbar.write(f"Early stopping at epoch {e+1}")
                    pbar.close()
                    break
    
    def train_edge_learner_oneloop(self):
        device = self.device
        decoder = self.decoder
        decoder.train()
        neg_edge_index = negative_sampling(
            edge_index=self.data.train_pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=self.data.train_pos_edge_index.size(1))
        edge_labels = torch.cat([torch.ones(self.data.train_pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))]).to(device)
        self.de_optimizer.zero_grad()
        pos_edge_scores = decoder(self.data.x,self.data.train_pos_edge_index)
        neg_edge_scores = decoder(self.data.x, neg_edge_index)
        edge_scores = torch.cat([pos_edge_scores,neg_edge_scores],dim=0)
        de_loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels)
        de_loss.backward()
        # for param in centloss_criterion.parameters():
        #     param.grad.data *= (1./args.w_con_loss)
        with torch.no_grad():
            decoder.eval()
            val_pos_edge_scores = decoder(self.data.x, self.data.val_pos_edge_index)
            val_neg_edge_scores = decoder(self.data.x, self.data.val_neg_edge_index.to(device))
            val_edge_scores = torch.cat([val_pos_edge_scores,val_neg_edge_scores],dim=0)
            val_edge_labels = torch.cat([torch.ones(self.data.val_pos_edge_index.size(1)), torch.zeros(self.data.val_neg_edge_index.size(1))]).to(device)
            val_recon_loss = F.binary_cross_entropy_with_logits(val_edge_scores, val_edge_labels)
        self.de_scheduler.step(val_recon_loss)
        return val_recon_loss

    def train_edge_learner(self, epochs = 2000, skip = False, ckpt_path = None, ckpt_save_epoch = 10):
        if skip:
            assert ckpt_path is not None, "Please provide a valid checkpoint path to load the edge learner model."
            VNG_utils.load(self.decoder, ckpt_path)
            print(f"Loaded edge learner model from {ckpt_path}")
            return
        best_loss = float('inf')
        patience = 20
        patience_count = 0
        patience_beta = 1e-3
        with tqdm(total=epochs, desc="Decoder Training") as pbar:
            for e in range(epochs):
                val_loss = self.train_edge_learner_oneloop()
                if val_loss < (best_loss - patience_beta):
                    best_loss = val_loss
                    patience_count = 0
                else: patience_count += 1
                pbar.set_postfix({
                    'Val Loss': f'{val_loss:.4f}', 
                    'Patience': f'{patience_count}/{patience}'
                })
                pbar.update(1)
                if ckpt_save_epoch > 0 and ((e+1) % ckpt_save_epoch == 0):
                    ts = datetime.now().strftime(timestamp_format)
                    ckpt_path = osp.join(root_path, "ckpt","decoder",self.tabgraph.name,"decoder_" + self.tabgraph.name+"_"+ts+f"_e{e}_"+".pth")
                    VNG_utils.save(self.decoder,ckpt_path)
                if patience_count >= patience:
                    pbar.write(f"Early stopping at epoch {e+1}")
                    pbar.close()
                    break

    def train_classifier_centloss(self):
        pass

    def train_classifier_vanilla_oneloop(self, weights=None):
        device = self.device
        self.classifier.train()
        self.classifier_optimizer.zero_grad()
        self.aug_data.x = self.aug_data.x.to(device)
        self.aug_data.y = self.aug_data.y.to(device)
        output = self.classifier(self.aug_data.x, self.edge_index_aug[:,self.train_edge_mask_aug], None)
        self.classifier_criterion(output[self.data_train_mask_aug], self.aug_data.y[self.data_train_mask_aug], weight=weights).backward()
        with torch.no_grad():
            self.classifier.eval()
            output = self.classifier(self.aug_data.x, self.edge_index_aug[:,self.train_edge_mask_aug], None)
            val_loss= F.cross_entropy(output[self.data_val_mask_aug], self.aug_data.y[self.data_val_mask_aug])

        self.classifier_optimizer.step()
        self.cl_scheduler.step(val_loss)

    def metric_classifier(self):
        self.classifier.eval()
        logits = self.classifier(self.aug_data.x, self.edge_index_aug[:,self.train_edge_mask_aug], None,)
        accs, baccs, f1s = [], [], []

        for i, mask in enumerate([self.data_train_mask_aug, self.data_val_mask_aug, self.data_test_mask_aug]):
            pred = logits[mask].max(1)[1]
            y_pred = pred.cpu().numpy()
            y_true = self.aug_data.y[mask].cpu().numpy()
            acc = pred.eq(self.aug_data.y[mask]).sum().item() / mask.sum().item()
            bacc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average=None)
            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)
            measure_result = classification_report(y_true, y_pred,digits=4, zero_division=np.nan)
        return accs, baccs, f1s, measure_result, recall
    
    def train_classifier_vanilla(self, epochs = 1000, weights=None):
        best_val_acc = test_acc = best_val_f1 = best_val_bacc = best_val_acc_f1 = -1
        best_measure = None
        # 初始化保存数据的列表
        val_acc_f1_list = []
        test_acc_f1_list = []
        val_f1_list = []
        val_acc_list = []
        tmp_test_acc_list = []
        tmp_test_f1_list = []

        with tqdm(total=epochs, desc="Classifier Training Progress") as pbar:
            for e in range(epochs):
                self.train_classifier_vanilla_oneloop()
                accs, bacc, f1s, measure_result, recall = self.metric_classifier()
                train_acc, val_acc, tmp_test_acc = accs
                train_f1, val_f1, tmp_test_f1 = f1s
                val_acc_f1 = (bacc[1] + val_f1) / 2.
                test_acc_f1 = (tmp_test_acc + tmp_test_f1) / 2.
            
                # 保存每轮的数值
                val_acc_f1_list.append(val_acc_f1)
                test_acc_f1_list.append(test_acc_f1)
                val_f1_list.append(val_f1)
                val_acc_list.append(val_acc)
                tmp_test_acc_list.append(tmp_test_acc)
                tmp_test_f1_list.append(tmp_test_f1)

                if val_f1 > best_val_f1:
                    best_val_bacc = bacc[1]
                    best_val_acc_f1 = val_acc_f1
                    best_measure = measure_result
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    test_acc = tmp_test_acc
                    test_bacc = bacc[2]
                    test_f1 = f1s[2]
                    best_recall = recall
                pbar.set_postfix({
                            'Accuracy': f'{val_acc:.4f}/{best_val_acc:.4f}', 
                            'F1 score': f'{val_f1:.4f}/{best_val_f1:.4f}'
                        })
                pbar.update(1)
        if(self.plot):
            VNG_utils.plot_val_test_acc_f1(val_acc_f1_list, test_acc_f1_list,title='acc_f1_{}'.format(self.repeatition))
            VNG_utils.plot_val_acc_f1(val_f1_list, val_acc_list,title='val_{}'.format(self.repeatition))
            VNG_utils.plot_tmp_test_acc_f1(tmp_test_acc_list, tmp_test_f1_list, title='test_{}'.format(self.repeatition))

        minority_recall = best_recall[self.minority_mask]
        majority_recall = best_recall[~self.minority_mask]
        return best_val_acc, best_val_f1, test_acc, test_bacc, test_f1, best_measure, minority_recall, majority_recall
    

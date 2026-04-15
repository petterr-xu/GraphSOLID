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
                    classifier: nn.Module, encoder: nn.Module, minority_mask,
                    diff_lr=1e-4, tearch_lr=1e-3, el_lr=1e-3, cl_lr=1e-3, en_lr=1e-3, cent_lr=1e-3, 
                    n_hid=512, diff_bs=64, r=None, plot=False, save_model=True, loss_type='ce', device='cuda:0'):
            
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

        self.minority_mask = minority_mask
        # 优化器初始化
        self.teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=tearch_lr)
        self.dif_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=diff_lr)
        self.de_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=el_lr)

        self.de_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.de_optimizer, mode='min',
                                                                factor = 0.8,
                                                                patience = 100,
                                                                verbose=False)
        
        if self.encoder is not None:
            self.en_optimizer = torch.optim.Adam(encoder.parameters(), lr=en_lr)
            # 使用 ctx.num_classes 动态设置类别
            self.centloss_criterion = loss_fn.CenterLoss(self.ctx.n_classes, n_hid).to(device)
            self.centloss_optimizer = torch.optim.Adam(self.centloss_criterion.parameters(), lr=cent_lr)

        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=cl_lr)
        self.cl_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.classifier_optimizer, mode='min',
                                                                factor = 0.5,
                                                                patience = 100,
                                                                verbose=False)
        factor_focal = 2.0
        factor_cb = 0.9999
        class_count = torch.bincount(self.data[self.target].y[self.data_train_mask].view(-1), minlength=self.ctx.n_classes).to(self.device,torch.float)
        if loss_type == "re":
            self.classifier_criterion = loss_fn.IMB_LOSS("re-weight",self.ctx.n_classes,class_count.detach().cpu().numpy(),device=self.device)
        elif loss_type == "ce":
            self.classifier_criterion = loss_fn.IMB_LOSS("ce",self.ctx.n_classes,class_count.detach().cpu().numpy(),device=self.device)
        elif loss_type == "cb":
            self.classifier_criterion = loss_fn.IMB_LOSS("cb-softmax",self.ctx.n_classes,class_count.detach().cpu().numpy(),factor_cb,device=self.device)
            # classifier_criterion = criterion.compute
        elif loss_type == "focal":
            self.classifier_criterion = loss_fn.IMB_LOSS("focal",self.ctx.n_classes,class_count.detach().cpu().numpy(),factor_focal,device=self.device)
            # classifier_criterion = criterion.compute
        else:
            raise Exception("No Implentation Loss")

    def _build_emb_data(self):
        self.encoder.eval()
        with torch.no_grad():
            emb_dict = self.encoder(self.data.x_dict, self.data.edge_index_dict)
            emb_data = copy.deepcopy(self.data)
            for ntype, embedding in emb_dict.items():
                emb_data[ntype].x = embedding.detach()
            self.emb_data = emb_data
        return emb_data

    def _hetero_recon_loss(self, emb_dict, split):
        total_loss = 0
        for edge_type in self.data.edge_types:
            src_type, _, dst_type = edge_type
            pos_edge_index = getattr(self.data[edge_type], f"{split}_pos_edge_index")
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=(self.data[src_type].num_nodes, self.data[dst_type].num_nodes),
                num_neg_samples=pos_edge_index.size(1)
            )
            pos_scores = self.decoder(emb_dict, pos_edge_index, edge_type)
            neg_scores = self.decoder(emb_dict, neg_edge_index, edge_type)
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.size(0), device=self.device),
                torch.zeros(neg_scores.size(0), device=self.device)
            ])
            total_loss += F.binary_cross_entropy_with_logits(scores, labels)
        return total_loss

    def cent_pretrain_oneloop(self, args):
        device = self.device
        target = self.target
        
        self.decoder.train()
        self.encoder.train()
        self.centloss_criterion.train()
        self.en_optimizer.zero_grad()
        self.centloss_optimizer.zero_grad()
        self.de_optimizer.zero_grad()

        emb_dict = self.encoder(self.data.x_dict, self.data.edge_index_dict)
        cent_loss = self.centloss_criterion(emb_dict[target][self.data_train_mask], 
                                           self.data[target].y[self.data_train_mask])

        total_de_loss = self._hetero_recon_loss(emb_dict, "train")

        # 4. 反向传播
        loss = args.w_con_loss * cent_loss + total_de_loss
        loss.backward()
        # 5. 验证部分 (逻辑类似，使用 val_data)
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            self.centloss_criterion.eval()
            v_emb_dict = self.encoder(self.data.x_dict, self.data.edge_index_dict)
            val_y = self.data[target].y[self.data_val_mask]
            val_cent_loss = self.centloss_criterion(v_emb_dict[target][self.data_val_mask], val_y)
            val_recon_loss = self._hetero_recon_loss(v_emb_dict, "val")
            
            val_loss = args.w_con_loss * val_cent_loss + val_recon_loss
        
        self.en_optimizer.step()
        self.centloss_optimizer.step()
        self.de_optimizer.step()
        self.de_scheduler.step(val_recon_loss)
            
        return val_loss, val_cent_loss, val_recon_loss

    def cent_pretrain(self, args, skip = False, ckpt_path:dict = None, ckpt_save_epoch = 0):
        if skip:
            assert ckpt_path is not None, "Please provide a valid checkpoint path to load the pre-trained encoder model."
            VNG_utils.load(self.encoder, ckpt_path['encoder'])
            VNG_utils.load(self.decoder, ckpt_path['decoder'])
            print(f"Loaded pre-trained encoder, decoder and centloss model from {ckpt_path}")
        else:
            best_loss = float('inf')
            patience = 5
            patience_count = 0
            patience_beta = 1e-3
            pre_epoch = 2000
            total_val_loss, total_con_loss, total_recon_loss = [],[],[]
            with tqdm(total=pre_epoch, desc="Pre-train") as pbar:
                for e in range(pre_epoch):
                    val_loss, con_loss, recon_loss = self.cent_pretrain_oneloop(args)
                    total_val_loss.append(val_loss.item())
                    total_con_loss.append(con_loss.item())
                    total_recon_loss.append(recon_loss.item())
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
                    if ckpt_save_epoch > 0 and ((e+1) % ckpt_save_epoch == 0):
                        ts = datetime.now().strftime(timestamp_format)
                        encoder_path = osp.join(root_path, "ckpt","encoder",self.ctx.name,"encoder_" + self.ctx.name+"_"+ts+f"_e{e}_"+".pth")
                        VNG_utils.save(self.encoder,encoder_path)
                        decoder_path = osp.join(root_path, "ckpt","decoder",self.ctx.name,"decoder_" + self.ctx.name+"_"+ts+f"_e{e}_"+".pth")
                        VNG_utils.save(self.decoder,decoder_path)

                    if patience_count >= patience:
                        pbar.write(f"Early stopping at epoch {e+1}")
                        pbar.close()
                        break
            if ckpt_save_epoch > 0:
                encoder_path = osp.join(root_path, "ckpt","encoder",self.ctx.name,"encoder_" + self.ctx.name+"_"+ts+".pth")
                VNG_utils.save(self.encoder,encoder_path)
                decoder_path = osp.join(root_path, "ckpt","decoder",self.ctx.name,"decoder_" + self.ctx.name+"_"+ts+".pth")
                VNG_utils.save(self.decoder,decoder_path)

        return self._build_emb_data()

    def train_teacher_encoder_joint_oneloop(self, args):
        target = self.target
        cls_weight = getattr(args, "w_cls_loss", 1.0)
        con_weight = getattr(args, "w_con_loss", 1.0)
        recon_weight = getattr(args, "w_recon_loss", 1.0)

        self.encoder.train()
        self.decoder.train()
        self.teacher.train()
        self.centloss_criterion.train()
        self.en_optimizer.zero_grad()
        self.de_optimizer.zero_grad()
        self.teacher_optimizer.zero_grad()
        self.centloss_optimizer.zero_grad()

        emb_dict = self.encoder(self.data.x_dict, self.data.edge_index_dict)
        train_emb = emb_dict[target][self.data_train_mask]
        train_y = self.data[target].y[self.data_train_mask]
        train_logits = self.teacher(train_emb)
        cls_loss = F.cross_entropy(train_logits, train_y)
        cent_loss = self.centloss_criterion(train_emb, train_y)
        recon_loss = self._hetero_recon_loss(emb_dict, "train")
        loss = cls_weight * cls_loss + con_weight * cent_loss + recon_weight * recon_loss
        loss.backward()
        self.en_optimizer.step()
        self.de_optimizer.step()
        self.teacher_optimizer.step()
        self.centloss_optimizer.step()

        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            self.teacher.eval()
            self.centloss_criterion.eval()
            val_emb_dict = self.encoder(self.data.x_dict, self.data.edge_index_dict)
            val_emb = val_emb_dict[target][self.data_val_mask]
            val_y = self.data[target].y[self.data_val_mask]
            val_logits = self.teacher(val_emb)
            val_cls_loss = F.cross_entropy(val_logits, val_y)
            val_cent_loss = self.centloss_criterion(val_emb, val_y)
            val_recon_loss = self._hetero_recon_loss(val_emb_dict, "val")
            val_loss = cls_weight * val_cls_loss + con_weight * val_cent_loss + recon_weight * val_recon_loss
        self.de_scheduler.step(val_recon_loss)

        return val_loss, val_cls_loss, val_cent_loss, val_recon_loss

    def train_teacher_encoder_joint(self, args, epochs=None, skip=False, ckpt_path:dict=None, ckpt_save_epoch=0):
        if skip:
            assert ckpt_path is not None, "Please provide checkpoint paths for encoder, decoder and teacher."
            VNG_utils.load(self.encoder, ckpt_path['encoder'])
            VNG_utils.load(self.decoder, ckpt_path['decoder'])
            VNG_utils.load(self.teacher, ckpt_path['teacher'])
            print(f"Loaded joint-trained encoder, decoder and teacher from {ckpt_path}")
            return self._build_emb_data()

        best_loss = float('inf')
        patience = getattr(args, "joint_patience", 5)
        patience_count = 0
        patience_beta = getattr(args, "joint_patience_beta", 1e-3)
        arg_epochs = getattr(args, "joint_epochs", None)
        train_epochs = epochs if epochs is not None else (arg_epochs if arg_epochs else getattr(args, "epochs", 2000))
        ts = datetime.now().strftime(timestamp_format)
        with tqdm(total=train_epochs, desc="Joint Teacher-Encoder Training") as pbar:
            for e in range(train_epochs):
                val_loss, cls_loss, con_loss, recon_loss = self.train_teacher_encoder_joint_oneloop(args)
                if val_loss < (best_loss - patience_beta):
                    best_loss = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                pbar.set_postfix({
                    'Val Loss': f'{val_loss:.4f}',
                    'Cls Loss': f'{cls_loss:.4f}',
                    'Con Loss': f'{con_loss:.4f}',
                    'Recon Loss': f'{recon_loss:.4f}',
                    'Patience': f'{patience_count}/{patience}'
                })
                pbar.update(1)
                if ckpt_save_epoch > 0 and ((e + 1) % ckpt_save_epoch == 0):
                    ts = datetime.now().strftime(timestamp_format)
                    encoder_path = osp.join(root_path, "ckpt", "encoder", self.ctx.name, "joint_encoder_" + self.ctx.name + "_" + ts + f"_e{e}_" + ".pth")
                    decoder_path = osp.join(root_path, "ckpt", "decoder", self.ctx.name, "joint_decoder_" + self.ctx.name + "_" + ts + f"_e{e}_" + ".pth")
                    teacher_path = osp.join(root_path, "ckpt", "teacher", self.ctx.name, "joint_teacher_" + self.ctx.name + "_" + ts + f"_e{e}_" + ".pth")
                    VNG_utils.save(self.encoder, encoder_path)
                    VNG_utils.save(self.decoder, decoder_path)
                    VNG_utils.save(self.teacher, teacher_path)
                if patience_count >= patience:
                    pbar.write(f"Early stopping at epoch {e+1}")
                    pbar.close()
                    break

        if ckpt_save_epoch > 0:
            encoder_path = osp.join(root_path, "ckpt", "encoder", self.ctx.name, "joint_encoder_" + self.ctx.name + "_" + ts + ".pth")
            decoder_path = osp.join(root_path, "ckpt", "decoder", self.ctx.name, "joint_decoder_" + self.ctx.name + "_" + ts + ".pth")
            teacher_path = osp.join(root_path, "ckpt", "teacher", self.ctx.name, "joint_teacher_" + self.ctx.name + "_" + ts + ".pth")
            VNG_utils.save(self.encoder, encoder_path)
            VNG_utils.save(self.decoder, decoder_path)
            VNG_utils.save(self.teacher, teacher_path)

        return self._build_emb_data()
    
    @DeprecationWarning
    def cover_data_with_emb(self):
        assert self.emb_data is not None, "Please run cent_pretrain() before calling this method."
        target = self.target
        new_embeddings = self.emb_data[target].x
        self.data[target].x = new_embeddings
        self.ctx.g[target].x = new_embeddings

    def update_data(self, data):
        assert data is not None, "Please run cent_pretrain() before calling this method."
        self.data = data.to(self.device)
        self.ctx.g = data.to(self.device)
        self.edge_index_dict = self.data.edge_index_dict
        self.data_train_mask = self.data[self.target].train_mask
        self.data_val_mask = self.data[self.target].val_mask
        self.data_test_mask = self.data[self.target].test_mask


        
        
    def train_teacher_oneloop(self):
        teacher_model = self.teacher
        teacher_optimizer = self.teacher_optimizer
        
        teacher_model.train()
        teacher_optimizer.zero_grad()
        inputs = self.data[self.target].x.to(self.device)
        logits = teacher_model(inputs[self.data_train_mask])
        loss = F.cross_entropy(logits,self.data[self.target].y[self.data_train_mask])
        loss.backward()
        teacher_optimizer.step()
        with torch.no_grad():
            teacher_model.eval()
            logits = teacher_model(inputs[self.data_val_mask])
            val_loss = F.cross_entropy(logits,self.data[self.target].y[self.data_val_mask])
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
                    path = osp.join(root_path, "ckpt","teacher",self.ctx.name,"teacher_" + self.ctx.name+"_"+ts+f"_e{e}_"+".pth")
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
    
    def train_diff_oneloop(self,args):
        device = self.device
        n_cls = self.ctx.n_classes
        train_feat_data = self.data[self.target].x[self.data_train_mask]
        train_label_data = self.data[self.target].y[self.data_train_mask]
        eval_feat = self.data[self.target].x[self.data_val_mask]
        eval_label = self.data[self.target].y[self.data_val_mask]
        loss_value = 0.0
        class_dis = VNG_utils.class_dis(self.data[self.target].y[self.data_train_mask],n_cls)
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
            # 去噪前调整inputs的维度
            inputs = torch.unsqueeze(inputs,dim=1)
            # 按照一定的概率将guidance置空，由此只用一个backbone训练出适用于两种情况（有无条件）的模型
            loss = self.diffusion.loss(inputs,targets,c_mask.to(torch.int32),args.padding)
            loss.backward()
            self.dif_optimizer.step()
            loss_value += loss.item()
        val_class_mask = VNG_utils.dis_based_class_mask(eval_label,class_dis,n_cls,eval_label.shape[0],args.guidance_drop_prob,adjustment_factor=args.adjustment_factor,device=device)
        val_class_mask = val_class_mask[:, None].repeat(1,n_cls).to(torch.int32)
        val_class_mask = (-1*(1-val_class_mask))
        eval_data = [eval_feat,eval_label,val_class_mask.to(torch.int32)]
        val_loss = self.eval_diffusion_model(eval_data,args)
        return val_loss

    @torch.no_grad()
    def eval_diffusion_model(self,eval_data,args):
        device = self.device
        n_cls = self.ctx.n_classes
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
                inputs = torch.unsqueeze(inputs,dim=1)
                loss = self.diffusion.loss(inputs,targets,c_mask.to(torch.int32), args.padding)
                loss_value += loss.item()
        return loss_value / data_size
    
    def train_diffusion(self,args, skip = False, ckpt_path = None, ckpt_save_epoch = 0):
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
                val_loss = self.train_diff_oneloop(args)
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
                    ckpt_path = osp.join(root_path, "ckpt","diffusion",self.ctx.name,"diffusion_" + self.ctx.name+"_"+ts+f"_e{e}_"+".pth")
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
    
    @DeprecationWarning
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
                    ckpt_path = osp.join(root_path, "ckpt","decoder",self.ctx.name,"decoder_" + self.ctx.name+"_"+ts+f"_e{e}_"+".pth")
                    VNG_utils.save(self.decoder,ckpt_path)
                if patience_count >= patience:
                    pbar.write(f"Early stopping at epoch {e+1}")
                    pbar.close()
                    break

    def train_classifier_centloss(self):
        pass

    def train_classifier_vanilla_oneloop(self, weights=None):
        device = self.device
        target = self.target
        
        self.classifier.train()
        self.classifier_optimizer.zero_grad()
        
        # 调用 HeteroGNN_classifier，传入字典格式的数据
        logits = self.classifier(self.data.x_dict, self.data.edge_index_dict)
        
        # 提取目标节点的标签和掩码
        labels = self.data[target].y
        train_mask = self.data[target].train_mask
        val_mask = self.data[target].val_mask
        
        # 计算训练损失
        loss = self.classifier_criterion.compute(logits[train_mask], labels[train_mask])
        loss.backward()
        self.classifier_optimizer.step()

        # 验证步骤
        with torch.no_grad():
            self.classifier.eval()
            output = self.classifier(self.data.x_dict, self.data.edge_index_dict)
            val_loss = self.classifier_criterion.compute(output[val_mask], labels[val_mask])
        self.classifier_optimizer.step()
        self.cl_scheduler.step(val_loss)
        return loss.item(), val_loss.item()

    def metric_classifier(self):
        self.classifier.eval()
        target = self.target
        
        with torch.no_grad():
            # 获取全图预测结果
            logits = self.classifier(self.data.x_dict, self.data.edge_index_dict)
        
        accs, baccs, f1s = [], [], []
        labels = self.data[target].y
        
        # 对应：训练集、验证集、测试集
        masks = [
            self.data[target].train_mask, 
            self.data[target].val_mask, 
            self.data[target].test_mask
        ]

        for i, mask in enumerate(masks):
            # 过滤出当前 mask 对应的预测和真值
            mask_logits = logits[mask]
            mask_labels = labels[mask]
            
            pred = mask_logits.max(1)[1]
            y_pred = pred.cpu().numpy()
            y_true = mask_labels.cpu().numpy()
            
            # 计算指标
            acc = pred.eq(mask_labels).sum().item() / mask.sum().item()
            bacc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)
            
            # 仅在测试集时生成详细报告和召回率
            if i == 2:
                measure_result = classification_report(y_true, y_pred, digits=4, zero_division=np.nan)
                recall = recall_score(y_true, y_pred, average=None)
                
        return accs, baccs, f1s, measure_result, recall
    
    def train_classifier_vanilla(self, epochs = 800, weights=None):
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
                self.train_classifier_vanilla_oneloop(weights)
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

                if val_acc_f1 > best_val_acc_f1:
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
        return best_val_acc, best_val_f1, test_acc, test_bacc, test_f1, best_measure, best_recall, minority_recall, majority_recall
    

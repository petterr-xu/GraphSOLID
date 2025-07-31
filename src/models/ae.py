import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import os
import datetime
from typing import Tuple,Optional
from . import mlp
from ..utils import VNG_utils
from ..utils.VNG_utils import load_cora_raw
from ..utils.config import VAEConfig,GraphDatasetConfig

class ClassEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(ClassEmbedding, self).__init__()
        """
        计算得到class embedding,作为条件生成的引导信息；包含两层全连接层。
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.encode(x.to(torch.float32))
  
class VAEDecodeLayer(nn.Module):
    def __init__(self, inputs_size, outputs_size, class_size, drop:int = 0.1):
        super().__init__()
        # self.class_emb = ClassEmbedding(class_size,inputs_size)
        # self.concate = nn.Linear(inputs_size+inputs_size,inputs_size)
        self.mlp = mlp.res_MLP(inputs_size,outputs_size)
        self.norm = nn.BatchNorm1d(outputs_size)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(drop)
    def forward(self,inputs,class_):
        # class_emb = self.class_emb(class_)
        # h = self.concate(torch.cat((inputs,class_emb),dim=1))
        h = self.norm(self.mlp(inputs))
        outputs = self.drop(self.act(h))
        return outputs

class VAEEncodeLayer(nn.Module):
    def __init__(self, inputs_size, outputs_size, drop:int = 0.1):
        super().__init__()
        self.mlp = mlp.res_MLP(inputs_size,outputs_size)
        self.norm = nn.BatchNorm1d(outputs_size)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(drop)
    def forward(self,inputs):
        h = self.norm(self.mlp(inputs))
        outputs = self.drop(self.act(h))
        return outputs

class VAEEncoder(nn.Module):
    def __init__(self,feature_size:int,latent_size:list) -> None:
        super().__init__()
        self.inlayer = nn.Linear(feature_size,feature_size)
        self.encoder = nn.Sequential()
        outputs_size = inputs_size = feature_size
        for outputs_size in latent_size:
            self.encoder.append(VAEEncodeLayer(inputs_size,outputs_size,drop=0.2))
            inputs_size = outputs_size
        
        # skip connection
        self.shortcut = nn.Linear(feature_size,latent_size[-1])
        
    def forward(self,inputs):
        x = self.inlayer(inputs)
        emb = self.encoder(x)
        emb += self.shortcut(inputs)
        # mean = self.mean_encoder(emb)
        # logvar = self.var_encoder(emb)
        # eps = torch.randn_like(emb)
        # std = torch.exp(logvar / 2)
        # z = eps * std + mean
        return emb# mean,logvar,z
    
class VAEDecoder(nn.Module):
    def __init__(self,feature_size:int,latent_size:list,class_size,drop:int = 0.3) -> None:
        super().__init__()
        self.mean_var_drop = nn.Dropout(drop)
        self.mean_encoder = nn.Linear(latent_size[-1],latent_size[-1])
        self.var_encoder = nn.Linear(latent_size[-1],latent_size[-1])
        self.decoder = nn.Sequential()
        inputs_size = latent_size[-1]
        self.class_emb = ClassEmbedding(class_size,inputs_size)
        self.concate = nn.Linear(inputs_size+inputs_size,inputs_size)
        for rec_size in reversed(latent_size[:-1]):
            self.decoder.append(VAEDecodeLayer(inputs_size,rec_size,class_size,drop))
            inputs_size = rec_size
        # 还原为原长度
        self.decoder.append(VAEDecodeLayer(inputs_size,feature_size,class_size,drop))
        # skip connection
        self.shortcut = nn.Linear(latent_size[-1],feature_size)
        # 输出层
        # self.final_act = nn.Sigmoid()
        self.final_drop1 = nn.Dropout(0.3)
        self.outlayer1 = nn.Linear(feature_size,feature_size)
        self.final_drop2 = nn.Dropout(0.3)
        self.outlayer2 = nn.Linear(feature_size,feature_size)

    def forward(self,latent_emb:torch.Tensor,class_):
        inputs = latent_emb
        mean = self.mean_encoder(self.mean_var_drop(latent_emb))
        logvar = self.var_encoder(self.mean_var_drop(latent_emb))
        eps = torch.randn_like(latent_emb)
        std = torch.exp(logvar / 2)
        latent_emb = eps * std + mean
        # 将标签信息嵌入隐层空间辅助解码
        # class_emb = self.class_emb(class_)
        # latent_emb = self.concate(torch.cat((latent_emb,class_emb),dim=1))
        for m in self.decoder:
            latent_emb = m(latent_emb,None)
        latent_emb += self.shortcut(inputs)
        out = self.outlayer1(self.final_drop1(latent_emb))
        reconstruct_y = self.outlayer2(self.final_drop2(out))
        return mean,logvar,F.sigmoid(reconstruct_y) # sigmoid激活函数将输出限定在0-1之间，用于和BCE损失函数配合使用，修改时请注意
        # return F.sigmoid(reconstruct_y)
    
class VAE(nn.Module):
    def __init__(self,feature_size:int,latent_size:list,class_size) -> None:
        super().__init__()
        self.encoder = VAEEncoder(feature_size,latent_size)
        self.decoder = VAEDecoder(feature_size,latent_size,class_size,drop=0.3)
    def forward(self,inputs,class_):
        z = self.encoder(inputs)
        mean,logvar,reconstruct_y = self.decoder(z,class_)
        return reconstruct_y, mean, logvar
    
    def encode(self,inputs):
        with torch.no_grad():
            self.encoder.eval()
            return self.encoder(inputs) # mean,logvar,z
    
    def decode(self,latent_emb,class_)->Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor,torch.Tensor]: mean,logvar,z
        """
        with torch.no_grad():
            self.decoder.eval()
            return self.decoder(latent_emb,class_)
    
    def loss_fn(inputs:torch.Tensor,targets:torch.Tensor,mean:torch.Tensor,logvar:torch.Tensor,kl_weight:float,padding:tuple):
        batch_size = inputs.shape[0]
        # 可能需要考虑去除padding
        left_pad,right_pad,top_pad,bottom_pad = padding
        # recons_loss = F.cosine_embedding_loss(inputs[:,left_pad:-right_pad], targets[:,left_pad:-right_pad],torch.ones(batch_size).to(inputs.device))
        # recons_loss = F.mse_loss(inputs[:,left_pad:-right_pad], targets[:,left_pad:-right_pad])
        recons_loss = F.binary_cross_entropy(inputs[:,left_pad:-right_pad], targets[:,left_pad:-right_pad])
        # recons_loss = F.l1_loss(inputs[:,left_pad:-right_pad], targets[:,left_pad:-right_pad])
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
        loss = recons_loss + kl_loss * kl_weight
        return loss,recons_loss,kl_loss
    
    def train_vae(model_config:VAEConfig,dataset_config:GraphDatasetConfig,device="cuda:0"):
        kl_weight = model_config.kl_weight
        batch_size = model_config.batch_size
        epochs = model_config.epochs
        feature_length = model_config.feature_length
        latent_size = model_config.latent_size
        lr = model_config.learning_rate
        n_block = model_config.n_block
        save_cp = model_config.save_cp

        graph_dataset = dataset_config.graph_dataset.capitalize()
        load_cache_file = dataset_config.load_cache_file
        padding = dataset_config.padding # cora padding = (0,7,0,0)
        dgl_feature_field_name = dataset_config.dgl_feature_field_name
        amp_scale = dataset_config.amp_scale

        file_path = r"CGDM-Im\\history_data\\encoder_model_checkpoint\\".replace("\\",os.sep)
        graph_load = {"Cora":load_cora_raw}[graph_dataset]
        graph = graph_load(load_cache_file)
        graph.ndata[dgl_feature_field_name] = graph.ndata[dgl_feature_field_name] * amp_scale
        train_feat_data = graph.ndata[dgl_feature_field_name][graph.ndata["train_mask"]]
        train_label_data = graph.ndata["label"][graph.ndata["train_mask"]]
        feat_dataset = TensorDataset(train_feat_data,train_label_data)
        data_loader = DataLoader(feat_dataset, batch_size, shuffle=True)
        len_train_data = len(data_loader.dataset)
        
        eval_feat = graph.ndata[dgl_feature_field_name][graph.ndata["val_mask"]].to(device)
        eval_label = graph.ndata["label"][graph.ndata["val_mask"]].to(device)
        eval_data = [eval_feat,eval_label]
        # eval_data = graph.ndata[dgl_feature_field_name][graph.ndata["val_mask"]].to(device)

        hidden_size = []
        for b in range(n_block-1):
            l = b + 1
            hidden_size.append(int(feature_length - float(l)/n_block*(feature_length-latent_size)))
        hidden_size.append(latent_size)
        class_size = dataset_config.num_classes
        model = VAE(feature_length,hidden_size,class_size)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr)
        
        for epoch in range(epochs):
            loss_value = np.array([0.0,0.0,0.0])
            for inputs,targets in data_loader:
                inputs = F.pad(inputs,pad=padding,mode="constant",value=0).to(device)
                targets = targets.to(device)
                # inputs = torch.unsqueeze(inputs,dim=1)
                # print(inputs.shape)
                optimizer.zero_grad()
                rec_feature, mean, logvar = model(inputs,targets)
                loss,recons_loss,kl_loss = VAE.loss_fn(rec_feature, inputs, mean, logvar, kl_weight, padding)
                loss.backward()
                optimizer.step()
                loss_value += np.array([loss.item(),recons_loss.item(),kl_loss.item()])
            loss_sum = loss_value / len_train_data
            if epoch % 10 == 0:
                eval_loss = VAE.eval_model(model,eval_data,kl_weight=0,padding=padding)
                print('Epoch [{}], Train Loss {:.6f} = {:.6f} + w*{:.6f}, Eval Loss {:.6f} '.format(epoch+1,loss_sum[0],loss_sum[1],loss_sum[2],eval_loss)+ datetime.datetime.now().strftime('%H:%M:%S'))
            if save_cp and (epoch + 1) % save_cp == 0:
                VNG_utils.save(model, file_path + 'VAE'+graph_dataset+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'e'+str(epoch+1)+'.pth')
            model.train()
            assert model.training , 'grad disable! stop train.'
        VNG_utils.save(model, file_path + 'VAE'+graph_dataset+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'.pth')
        return model
    
    def eval_model(model:nn.Module,eval_data,kl_weight, padding):
        model.eval()
        with torch.no_grad():
            feat_dataset = TensorDataset(eval_data[0],eval_data[1])
            test_data_loader = DataLoader(feat_dataset, batch_size=32, shuffle=True)
            eval_size = len(test_data_loader.dataset)
            loss_value = 0.0
            for inputs,targets in test_data_loader:
                inputs = F.pad(inputs,pad=padding,mode="constant",value=0)
                rec_feature, mean, logvar = model(inputs,targets)
                loss,recons_loss,kl_loss = VAE.loss_fn(rec_feature, inputs, mean, logvar, kl_weight, padding)
                loss_value += loss.item()
            loss_value = loss_value / eval_size
        return loss_value


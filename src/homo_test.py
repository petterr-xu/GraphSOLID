import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import os
import sys
import datetime
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from denoise import simple_unet
from models import ae,gcn,mlp,classifier,gsl,diffusion
from denoise import unet_vector as unet
import utils.VNG_utils as VNG_utils
from utils.config import DiffusionConfig,GraphDatasetConfig,ClassifierConfig,UnetConfig

def train_and_sample(train_device="cuda:0",test_device="cuda:0"):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    graph_config = GraphDatasetConfig(graph_dataset="Cora_emb",
                                      file_path=None,
                                      load_cache_file=True,
                                      padding=(0,0,0,0),
                                      dgl_feature_field_name="feat",
                                      num_classes=7,
                                      amp_scale=1)
    # vae:ae.VAE = torch.load("CGDM-Im\\history_data\\encoder_model_checkpoint\\VAECora06-18_16_56.pth".replace("\\",os.sep))
    diffusion_config = DiffusionConfig(T=1500,
                                       beta=(1e-4, 2e-2),
                                       beta_schedule="lin",
                                       SMOTE_aug=True,
                                       learning_rate=5e-5,
                                       is_latent_diffusion=False,
                                       latent_encoder=None,
                                       epochs=500,
                                       batch_size=128,
                                       save_cp=0)
    
    unet_config = UnetConfig(vector_channels=1,
                             feature_length=128,
                             n_channels=32,
                             n_length=128,
                             num_class=7,
                             class_embedding_channel=4*32,
                             time_embedding_channel=4*32,
                             ch_mults=(1, 2, 4),
                             n_blocks=2,
                             is_attn=(True, True, True),
                             )
    _,_,[diffusion_model,_] = diffusion.train(graph_config,diffusion_config,unet_config,device=train_device)
    return diffusion_model
    # diffusion_block_cp = r"CGDM-Im\\history_data\\diffusion_model_checkpoint\\DiffusionModelCora06-05_12_59.pth".replace("\\",os.sep)
    # diffusion_model = torch.load(diffusion_block_cp,map_location=test_device)

    classifier_block_cp = r"CGDM-Im\\history_data\\guidance_classifier_model_checkpoint\\GCModelCora_emb06-18_20_35.pth".replace("\\",os.sep)
    classifier_block = torch.load(classifier_block_cp,map_location=test_device)
    # classifier_block = classifier_block.to(test_device)
    diffusion_model = diffusion_model.to(test_device)

    class_ = torch.Tensor([[0,0,0,0,0,1,0]])
    # loss_fun = cm.CGloss(1433,7)
    loss_fun = nn.CrossEntropyLoss()
    for i in range(20):
        num_samples = 10
        nodes_class = class_.expand([num_samples,-1]).to(test_device)
        # nodes_class = torch.argmax(nodes_class,dim=nodes_class.dim()-1)
        # print("generating No. {}/{} sample.".format(i+1,num_samples))
        torch.no_grad()
        x_t = torch.randn([num_samples,1,unet_config.feature_length]).to(test_device)
        # x_t = F.pad(x_t,pad=(0,7,0,0),mode="constant",value=0)
        diffusion_model.eval()
        classifier_block.eval()
        x0,frame = diffusion_model.sampling([None,loss_fun],classifier_scale_mode=0,guidance_scale=i
                                        ,x_t=x_t,y=nodes_class,save_frames=True,device=test_device)
        # if(diffusion_config.is_latent_diffusion):
        #     mean,logvar,x0 = vae.decode(x0.squeeze(1),nodes_class)
        #     print("\nmean\n{}\n var\n{};".format(mean[0],torch.exp(logvar[0])))
        #     x0 = x0.unsqueeze(1)
        VNG_utils.show_tensor_dis(x0[0],name="feature_dis_{}".format(i))
        pred:torch.Tensor = classifier_block(x0)
        class_pred = torch.argmax(pred,dim=pred.dim()-1)
        print(class_pred)
        VNG_utils.show_tensor_dis(class_pred,bins=range(0,8),name="class_dis_{}".format(i))
        # print("generated")
        # for x in frame:
        #     VNG_utils.show_tensor_dis(x[0])
        #     input()


def graph_augment(diffusion_model=None,device="cuda:0",save_graph=True):
    # graph = VNG_utils.load_cora_raw().to(device)
    graph = VNG_utils.load_cora_emb().to(device)
    # dis_matrix = GSL.neigh_dis(graph,7)
    dis = VNG_utils.node_class_dis(graph,graph.ndata["train_mask"])
    aug_size = torch.max(dis) - dis
    node_classes = torch.tensor([],dtype=torch.int32)
    for class_,class_size in enumerate(aug_size):
        node_classes = torch.cat((node_classes,torch.full([class_size],fill_value=class_)))
    node_classes = F.one_hot(node_classes).to(device,dtype=torch.int32)
    num_samples = node_classes.shape[0]
    x_t = torch.randn([num_samples,1,128]).to(device)
    # x_t = F.pad(x_t,pad=(0,7,0,0),mode="constant",value=0)
    
    if diffusion_model == None:
        diffusion_block_cp = "CGDM-Im\\history_data\\diffusion_model_checkpoint\\DiffusionModelCora_emb06-19_10_53.pth".replace("\\",os.sep)
        diffusion_model = torch.load(diffusion_block_cp,map_location=device)
    diffusion_model.to(device)
    # vae:ae.VAE = torch.load("CGDM-Im\\history_data\\encoder_model_checkpoint\\VAECora06-18_08_57.pth".replace("\\",os.sep),map_location=device)
    for i in range(0,10):
        guidance = i
        x0,frame = diffusion_model.sampling([None,None],classifier_scale_mode=0,guidance_scale=guidance
                                            ,x_t=x_t,y=node_classes,save_frames=True,device=device)
        # _,_,x0 = vae.decode(x0.squeeze(1),node_classes)
        # x0 = x0.unsqueeze(1)
        virtual_feat = x0.squeeze(1) # [:,0:128]
        v_information = {"feat":virtual_feat,"label":node_classes}
        # new_graph = GSL.sim_based_graph_gen(subgraph,virtual_feat,v_information)
        new_graph = gsl.sim_based_graph_gen(graph,virtual_feat,v_information,top_k=4)
        print("augmented graph generated {}\n {}".format(i,aug_size))
        if save_graph:
            file_path = "CGDM-Im\\dataset\\aug_graph\\CoraAugLDM_gs"+str(guidance)+".bin"
            dgl.save_graphs(file_path.replace("\\",os.sep),new_graph)
        sys.stdout = open('CGDM-Im\\log\\log{}.txt'.format(guidance).replace("\\",os.sep), 'w')
        print("\nguidance scale = {}:\n".format(guidance))
        gcn.train_GCN_classifier(new_graph)
        sys.stdout = sys.__stdout__
        print("gcn{}".format(guidance))


if __name__ == "__main__":
    # classifier.train_mlp_classifier()
    diffusion_model = train_and_sample("cuda:0","cuda:0")
    graph_augment(diffusion_model,save_graph=True)
    # train_classifier()




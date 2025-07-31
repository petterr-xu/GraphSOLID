import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import PIL

import os
import datetime

from utils.config import DiffusionConfig,UnetConfig
from denoise import unet_image as unet
from CG_VNG import GDDPMblock
import utils.VNG_utils as VNG_utils

def train(diffusion_config:DiffusionConfig,denoise_config:UnetConfig,device,padding):
    T = diffusion_config.T
    guidance_drop_prob = diffusion_config.guidance_drop_prob
    learning_rate = diffusion_config.learning_rate
    epochs = diffusion_config.epochs
    batch_size = diffusion_config.batch_size
    save_cp = diffusion_config.save_cp
    file_path = "CGDM-Im\\history_data\\diffusion_model_checkpoint\\".replace("\\",os.sep)

    transform = transforms.Compose([transforms.PILToTensor()])
    train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)  
    test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)  # train=True训练集，=False测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("{} steps in an epoch".format(len(train_loader)))
    # fig = plt.figure()
    # for i in range(12):
    #     plt.subplot(3, 4, i+1)
    #     plt.tight_layout()
    #     plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    #     plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.savefig("CGDM-Im\\history_data\\figure\\mnist_graph.png")


    eps_model = unet.UNet(denoise_config)
    model = GDDPMblock(eps_model,n_steps=T,device=device)
    model = model.to(device)

    optimizer = optim.Adam(model.eps_model.parameters(), lr=learning_rate)
    train_loss = []
    eval_loss = []
    for epoch in range(epochs):
        loss_value = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, targets = data
            # 对节点特征进行padding,以避免unet下采样中出现奇数纬度导致分辨率不匹配
            inputs = F.pad(inputs,pad=padding,mode="constant",value=0).to(device)
            # print(inputs.shape)
            optimizer.zero_grad()
            targets = targets.to(device)
            inputs = inputs.to(device)
            # 按照一定的概率将guidance置空，由此只用一个backbone训练出适用于两种情况（有无条件）的模型
            class_mask = (torch.rand(targets.size()) < guidance_drop_prob).to(device,torch.int32)
            loss = model.loss(inputs,targets,class_mask)
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
        train_loss.append(loss_value)
        if (epoch + 1) % 10 == 0:
            eval_loss_value = eval_model(model,test_loader,guidance_drop_prob,padding)
            print('Epoch [{}], Eval Loss: {:.4f}, Train Loss {:.4f} '.format(epoch+1,eval_loss_value,loss_value)+ datetime.datetime.now().strftime('%H:%M:%S'))
            eval_loss.append(eval_loss_value)
            model.train()
        if save_cp and ((epoch + 1) % save_cp == 0):
            torch.save(model, file_path + 'ImageDiffusionModel'+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'e'+str(epoch+1)+'.pth')

        assert model.training , 'grad disable! stop train.'
    try:
        torch.save(model, file_path + 'ImageDiffusionModel'+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'.pth')
        # torch.save(eps_model, file_path + 'DenoiseModel'+graph_dataset+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'.pth')
    except FileNotFoundError as fnf:
        print("MODEL NOT SAVED!")
    return train_loss,eval_loss,[model,eps_model]

def eval_model(model:nn.Module,eval_loader,guidance_drop_prob,padding,device="cuda:0"):
    model.eval()
    loss_value = 0.0
    for batch_idx, data in enumerate(eval_loader, 0):
        inputs, targets = data
        # 对节点特征进行padding,以避免unet下采样中出现奇数纬度导致分辨率不匹配
        inputs = F.pad(inputs,pad=padding,mode="constant",value=0).to(device)
        targets = targets.to(device)
        inputs = inputs.to(device)
        # 按照一定的概率将guidance置空，由此只用一个backbone训练出适用于两种情况（有无条件）的模型
        class_mask = (torch.rand(targets.size()) < guidance_drop_prob).to(device,torch.int32)
        loss = model.loss(inputs,targets,class_mask)
        loss_value += loss.item()
    return loss_value

def sampling(padding,device="cuda:0"):
    diffusion_block_cp = r"CGDM-Im\\history_data\\diffusion_model_checkpoint\\ImageDiffusionModel05-11_22_22e160.pth".replace("\\",os.sep)
    classifier_block_cp = r"CGDM-Im\\history_data\\guidance_classifier_model_checkpoint\\GCModelCora04-19_20_33.pth".replace("\\",os.sep)
    diffusion_model = torch.load(diffusion_block_cp,map_location=device)
    classifier_block = torch.load(classifier_block_cp,map_location=device)

    num_labels = torch.Tensor([0,1,2,3,4,5,6,7,8,9]).to(device,torch.int64)
    # classes = F.one_hot(num_labels,10).to(device)
    # loss_fun = cm.CGloss(1433,7)
    loss_fun = nn.CrossEntropyLoss()
    torch.no_grad()
    x_t = torch.randn([10,1,28,28]).to(device)
    x_t = F.pad(x_t,pad=padding,mode="constant",value=0)
    diffusion_model.eval()
    classifier_block.eval()
    guidance_scale=3
    x0,x_frame = diffusion_model.sampling([None,loss_fun],classifier_scale_mode=0,
                                    guidance_scale=guidance_scale,x_t=x_t,y=num_labels,save_frames=True,device=device)
    fig = plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(x0[i].squeeze(0)[2:-2,2:-2].to("cpu"), cmap='gray', interpolation='none')
        plt.title("Labels: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.savefig("CGDM-Im\\history_data\\figure\\numbers1.png".replace("\\",os.sep))
    plt.close()
    # 保存动画为 GIF 文件
    file_path = 'CGDM-Im\\history_data\\figure\\figureanimation'+str(guidance_scale)+'.gif'
    VNG_utils.diffusion_gif(x_frame,file_path)
        
        

if __name__ == "__main__":
    diffusion_config_dict = {"T":500,
                            "SMOTE_aug":False,
                            "learning_rate":0.00003,
                            "epochs":70,
                            "batch_size":256,
                            "save_cp":20}
    diffusion_config = DiffusionConfig(**diffusion_config_dict)
    unet_config_dict = {
            "vector_channels":1,
            "n_channels":32,
            "class_embedding_channel":4 * 32,
            "time_embedding_channel":4 * 32,
            "ch_mults":(1, 2, 4),
            "is_attn":(False, False,False),
            "n_blocks":2
    }
    unet_config = UnetConfig(**unet_config_dict)
    train(diffusion_config,unet_config,"cuda:0",(0,0,0,0))
    # sampling(padding=(0,0,0,0))
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.VNG_utils as VNG_utils
from torch.utils.data import TensorDataset, DataLoader

import os
import sys
import copy
import random
from sklearn.metrics import roc_auc_score

from models import gnn,gsl,diffusion,classifier,teacher
from utils.config import GraphDatasetConfig,DiffusionConfig,UnetConfig,GraphEncoderConfig

def train_mlp_teacher(g:dgl.DGLGraph,in_feats,num_class,layers,drop,lr,batch_size=128,epochs=500,patience=3,show_details=False,device="cuda:0"):
    with g.local_scope():
        model = teacher.MLPTeacher(in_feats,num_class,layers,drop).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr)
        labels = g.ndata['label']
        if labels.dim() < 2:
            labels = F.one_hot(labels,num_class)
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        best_val_loss = float("inf")
        patience_count = 0
        patiance_beta = 1e-2
        train_dataset = TensorDataset(g.ndata["feat"][train_mask],labels[train_mask])
        data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        for e in range(epochs):
            loss_value = 0.0
            model.train()
            for inputs,targets in data_loader:
                optimizer.zero_grad()
                logits = model(inputs)
                # pred = logits.argmax(1)
                loss = F.cross_entropy(logits,targets.to(torch.float32))
                loss.backward()
                optimizer.step()
                loss_value += loss.item()
            if e % 5 == 0:
                val_loss = val_mlp_teacher([g.ndata["feat"][val_mask],labels[val_mask]],model)
                if best_val_loss > (val_loss + patiance_beta):
                    best_val_loss = val_loss
                    patience_count = 0
                else: patience_count += 1
                if show_details:
                    print('In epoch {}, loss: {:.3f}, val loss: {:.3f} (best {:.3f}), patience {}/{}'.format(
                        e, loss_value, val_loss, best_val_loss,patience_count,patience))
                if patience_count >= patience:
                    break
        logits = model(g.ndata["feat"][test_mask])
        preds = logits.argmax(1)
        if show_details : VNG_utils.show_detailed_evaluation(labels[test_mask].argmax(1),preds)  
    return model
    

def val_mlp_teacher(val_data,model:nn.Module):
    model.eval()
    inputs,targets = val_data
    with torch.no_grad():
        logits = model(inputs)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits,targets.to(torch.float32))
    return loss.item()



def experiment1(eliminate_randomness:int=10,guidance=6,areg_guidance=6,train_device="cuda:0",out_stream="std"):
    """
    本实验的目的是验证本文提出的方法在类不平衡下的图神经网络节点分类任务上的有效性。

    实验设置：
        1. 训练集中多数类的样本数量设定为20，少数类的样本数量设定为：20*a; 其中 a 为不平衡率
        2. 为避免节点类数量不一致对评估造成影响，设定验证集中每一类节点的样本数量为30，测试集中每一类节点的样本数量为100
    """
    # split_schedule = {"test":[100,100,100,100,100,100,100],
    #                 "val":[30,30,30,30,30,30,30],
    #                 "train":[10,25,25,25,10,10,20]} # 选择类0 4 5作为少数类，不平衡率为0.25
    diffusion_config = DiffusionConfig(T=650,
                                       beta=(1e-4, 2e-2),
                                       beta_schedule="lin",
                                       guidance=areg_guidance,
                                       SMOTE_aug=False,
                                       SMOTE_kneighbors=5,
                                       patience=5,
                                       learning_rate=5e-5,
                                       guidance_drop_prob=0.1,
                                       is_confidence_guide=True,
                                       teacher_model=None,
                                       temperature=5.,
                                       epochs=600,
                                       batch_size=64,
                                       save_cp=0)
    unet_config = UnetConfig(vector_channels=1,
                             feature_length=736,
                             n_length=512,
                             n_channels=128,
                             num_class=7,
                             class_embedding_channel=8*128,
                             time_embedding_channel=8*128,
                             ch_mults=(1, 2, 4),
                             n_blocks=3,
                             is_attn=(False, True, True),
                             )
    cgvng_preds = []
    cgvng_labels = []
    cgvng_auc_avg = 0.0
    embed_preds = []
    embed_labels = []
    embed_auc_avg = 0.0
    baseline_preds = []
    baseline_labels = []
    baseline_auc_avg = 0.0
    if out_stream != "std":
        sys.stdout = open(('CGDM-Im\\log\\experiment1_{}'.format(out_stream)+'.txt').replace("\\",os.sep), 'w')
    for e in range(eliminate_randomness):
        print("experiment1 conducting {}/{}".format(e+1,eliminate_randomness))
        # graph = VNG_utils.load_cora_vanilla()
        # train_schedule = VNG_utils.imbalanced_train_schedule(graph,10,7,30,100,bias=488).tolist()
        # print(train_schedule)
        split_schedule = {"val":[30,30,30,30,30,30,30],
                        "train":[8,8,20,20,20,8,8]}
        # im_indices = random.sample(range(7),3)
        # print("imbalance label : ",im_indices)
        # split_schedule["train"] = [10 if idx in im_indices else val for idx, val in enumerate(split_schedule["train"])]# 随机选择3个类，使其训练样本不平衡
        graph = VNG_utils.cora_dataset_split(split_schedule).to(train_device) # 划分一个新的数据集
        print(graph)
        print("training gcn encoder")
        graphencoder_config = GraphEncoderConfig(learning_rate=1e-3,
                                                 epochs= 650,
                                                 batch_size=None,
                                                 save_cp=0,
                                                 patience=5,
                                                 dataset=graph,
                                                 gnn="sage",
                                                 layers=2,
                                                 beta=5e-4,
                                                 alpha_lr=1e-3,
                                                 input_size=None, # input_size 参数无需设置
                                                 latent_size=unet_config.feature_length, # 隐空间维度需要与去噪网络的输入保持一致
                                                 )
        gcn_encoder = gnn.train_GNN_encoder(graphencoder_config,show_detail=True)
        emb_graph = gnn.encode_graph(graph,"feat",gcn_encoder)
        print("training teacher model")
        mlp_teacher = train_mlp_teacher(emb_graph,unet_config.feature_length,unet_config.num_class,layers=1,drop=0.4,lr=1e-4)
        diffusion_config.teacher_model = mlp_teacher
        print("training graph structure learner")
        edge_learner = gsl.train_edge_learner(emb_graph)
        # VNG_utils.show_embedding_dis(emb_graph.ndata["feat"][emb_graph.ndata["test_mask"]],emb_graph.ndata["label"][emb_graph.ndata["test_mask"]],7)
        graph_config = GraphDatasetConfig(graph_id="Cora",
                                          graph_dataset=emb_graph,
                                        file_path=None,
                                        load_cache_file=True,
                                        padding=(0,0,0,0),
                                        dgl_feature_field_name="feat",
                                        num_classes=7)
        print("training diffusion model")
        _,_,[diffusion_model,_] = diffusion.train(graph_config,diffusion_config,unet_config,device=train_device)
        new_graph = connect_new_nodes(emb_graph,mlp_teacher,diffusion_config.temperature,edge_learner,diffusion_model,embedding_size=unet_config.feature_length,over_sample_rate=0.5,guidance=guidance,device=train_device)
        # graph_augment(emb_graph,diffusion_model,embedding_size=unet_config.feature_length,guidance=guidance,device=train_device)
        print("training downstream classifier")
        classifier_config = GraphEncoderConfig(learning_rate=1e-3,
                                                 epochs= 500,
                                                 batch_size=None,
                                                 save_cp=0,
                                                 patience=3,
                                                 dataset=emb_graph,
                                                 gnn="sage",
                                                 layers=3,
                                                 beta=1e-1,
                                                 alpha_lr=1e-3,
                                                 input_size=None, # input_size 参数无需设置
                                                 latent_size=unet_config.feature_length # 隐空间维度需要与去噪网络的输入保持一
                                                 )
        print("raw cora basiline:")
        classifier_config.dataset = graph
        _,baseline_pred,baseline_target,baseline_auc = gnn.train_GNN_classifier(classifier_config,device=train_device)
        print("emb-cora:")
        _,embed_pred,embed_target,embed_auc = gnn.train_GNN_classifier(classifier_config,device=train_device)
        print("class-guidance free diffusion augmented dataset:")
        classifier_config.dataset = new_graph
        _,cgvng_pred,cgvng_target,cgvng_auc = gnn.train_GNN_classifier(classifier_config,device=train_device)
        embed_preds.append(embed_pred)
        embed_labels.append(embed_target)
        embed_auc_avg += embed_auc
        cgvng_preds.append(cgvng_pred)
        cgvng_labels.append(cgvng_target)
        cgvng_auc_avg += cgvng_auc
        baseline_preds.append(baseline_pred)
        baseline_labels.append(baseline_target)
        baseline_auc_avg += baseline_auc
        torch.cuda.empty_cache()
    cgvng_preds = torch.stack(cgvng_preds, dim=0).view(-1)
    cgvng_labels = torch.stack(cgvng_labels, dim=0).view(-1)
    embed_preds = torch.stack(embed_preds, dim=0).view(-1)
    embed_labels = torch.stack(embed_labels, dim=0).view(-1)
    baseline_preds = torch.stack(baseline_preds, dim=0).view(-1)
    baseline_labels = torch.stack(baseline_labels, dim=0).view(-1)
    print("embedding graph performance:")
    VNG_utils.show_detailed_evaluation(embed_labels,embed_preds)
    embed_auc_avg /= eliminate_randomness
    print("AUC score = {:.4f}".format(embed_auc_avg))
    print("guidance scale = {} augmented dataset".format(guidance))
    VNG_utils.show_detailed_evaluation(cgvng_labels,cgvng_preds)
    cgvng_auc_avg /= eliminate_randomness
    print("AUC score = {:.4f}".format(cgvng_auc_avg))
    print("basiline performance:")
    VNG_utils.show_detailed_evaluation(baseline_labels,baseline_preds)
    baseline_auc_avg /= eliminate_randomness
    print("AUC score = {:.4f}".format(baseline_auc_avg))
    sys.stdout = sys.__stdout__


def connect_new_nodes_neismixup(graph:dgl.DGLGraph,teacher:nn.Module,temperature,diffusion_model,embedding_size,topk=3,guidance=6, over_sample_rate = None,save_graph=True,device="cuda:0"):
    virtual_feat, v_information = diffusion.softlabel_based_hard_nodes_sampling(graph,teacher,temperature,diffusion_model,embedding_size,guidance, over_sample_rate,device)
    new_graph = gsl.sim_based_nei_mixup(graph,virtual_feat,v_information,top_k=topk)
    return new_graph

def connect_new_nodes(graph:dgl.DGLGraph,teacher:nn.Module,temperature,edge_predictor,diffusion_model,embedding_size,topk=3,guidance=6, over_sample_rate = None,save_graph=True,device="cuda:0"):
    rand_topk_scale = 6
    virtual_feat, v_information = diffusion.softlabel_based_hard_nodes_sampling(graph,teacher,temperature,diffusion_model,embedding_size,guidance, over_sample_rate,device)
    # virtual_feat, v_information = diffusion.confidence_based_hard_nodes_sampling(graph,diffusion_model,embedding_size,5,guidance, over_sample_rate,device)
    num_existing_nodes = graph.number_of_nodes()
    num_virtual_nodes = len(virtual_feat)
    fc_graph = copy.deepcopy(graph)
    fc_graph.add_nodes(num_virtual_nodes,v_information)
    # connect virtual nodes to all nodes
    u_ids = torch.tensor(range(num_existing_nodes,num_existing_nodes+num_virtual_nodes),dtype=torch.int64,device=device)
    u = u_ids.view(-1,1).expand([num_virtual_nodes,num_existing_nodes]).reshape([1,-1]).squeeze(0)
    # u = u_ids.view(-1,1).expand([num_virtual_nodes,num_existing_nodes+num_virtual_nodes]).reshape([1,-1]).squeeze(0)
    # v_ids = torch.tensor(range(0,num_existing_nodes+num_virtual_nodes),dtype=torch.int64,device=device)
    v_ids = torch.tensor(range(0,num_existing_nodes),dtype=torch.int64,device=device)
    v = v_ids.repeat([1,num_virtual_nodes]).squeeze(0)
    # add directed edge
    fc_graph.add_edges(u,v)
    fc_graph.add_edges(v,u)
    # 计算新节点与所有现有节点的边概率
    edge_scores = edge_predictor.predict(fc_graph,fc_graph.ndata["feat"],threshold=0.8)
    """
    对于一个无向图，dgl仍然以有向图的形式存储，因此当处理的图数据是无向图时一条边实际上在dgl中被存储为两条有向边。
    以下代码假设处理的数据是无向图，因此会将dgl中对应的两条有向边（代表无向图中的一条边）的score值做均值处理，以合成一条无向边的score。
    """
    virtual_edge_scores_d = edge_scores[-u.shape[0]-v.shape[0]:]
    u_score = virtual_edge_scores_d[0:u.shape[0]]
    v_score = virtual_edge_scores_d[-u.shape[0]:]
    virtual_edge_scores = (u_score + v_score )/ 2
    virtual_edge_scores_matrix = virtual_edge_scores.reshape([num_virtual_nodes,-1]) # 该矩阵的每一行表示一个生成节点与其他节点之间存在边的概率值
    _, v_indx = virtual_edge_scores_matrix.topk(k=topk*rand_topk_scale, dim=1) # 筛选概率值排名topk*3的节点，再从中随机选择topk个节点作为目标节点
    v_nodes = []
    for candidate in v_indx:
        indices = random.sample(range(topk*rand_topk_scale),topk)
        v_nodes.append(candidate[indices])
    v_nodes = torch.stack(v_nodes).to(device)
    v = v_nodes.reshape([1,-1]).squeeze(0)
    u_ids = torch.tensor(range(num_existing_nodes,num_existing_nodes+num_virtual_nodes),dtype=torch.int64,device=device)
    u = u_ids.view(-1,1).expand([num_virtual_nodes,topk]).reshape([1,-1]).squeeze(0)
    
    new_graph = copy.deepcopy(graph)
    new_graph.add_nodes(num_virtual_nodes,v_information)
    new_graph.add_edges(u,v)
    new_graph.add_edges(v,u)
    new_graph.ndata["train_mask"][-num_virtual_nodes:] = True
    new_graph.ndata["test_mask"][-num_virtual_nodes:] = False
    new_graph.ndata["val_mask"][-num_virtual_nodes:] = False
    return new_graph

def graph_augment(graph:dgl.DGLGraph,diffusion_model,embedding_size,guidance=6, over_sample_rate = None,save_graph=True,device="cuda:0"):
    virtual_feat, v_information = diffusion.virtual_nodes_sampling(graph,diffusion_model,embedding_size,guidance, over_sample_rate,device)
    # new_graph = GSL.sim_based_graph_gen(subgraph,virtual_feat,v_information)
    # dis_matrix = gsl.neigh_dis(graph,7)
    # new_graph = gsl.dis_based_graph_gen(graph,dis_matrix,node_classes,v_information)
    new_graph = gsl.sim_based_graph_gen(graph,virtual_feat,v_information,top_k=3)
    if save_graph:
        file_path = "CGDM-Im\\dataset\\aug_graph\\CoraAugLDM_gs"+str(guidance)+".bin"
        dgl.save_graphs(file_path.replace("\\",os.sep),new_graph)
    return new_graph

if __name__ == "__main__":
    # guidance = 7.5
    # # experiment1(eliminate_randomness=20,guidance=guidance,areg_guidance=6)
    # experiment1(eliminate_randomness=5,guidance=guidance,areg_guidance=6)

    split_schedule = {"val":[30,30,30,30,30,30,30],
                    "train":[20,20,20,20,20,20,20]}
    # im_indices = random.sample(range(7),3)
    # print("imbalance label : ",im_indices)
    # split_schedule["train"] = [10 if idx in im_indices else val for idx, val in enumerate(split_schedule["train"])]# 随机选择3个类，使其训练样本不平衡
    graph = VNG_utils.cora_dataset_split(split_schedule).to("cuda:0") # 划分一个新的数据集
    classifier_config = GraphEncoderConfig(learning_rate=1e-3,
                                                epochs= 500,
                                                batch_size=None,
                                                save_cp=0,
                                                patience=3,
                                                dataset=graph,
                                                gnn="sage",
                                                layers=3,
                                                beta=1e-1,
                                                alpha_lr=1e-3,
                                                input_size=None, # input_size 参数无需设置
                                                latent_size=1433 # 隐空间维度需要与去噪网络的输入保持一
                                                )
    print("raw cora basiline:")
    classifier_config.dataset = graph
    _,baseline_pred,baseline_target,baseline_auc = gnn.train_GNN_classifier(classifier_config,device="cuda:0")

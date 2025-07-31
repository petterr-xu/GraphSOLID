import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch.conv as GNN
import random
import datetime
import numpy as np
from sklearn.metrics import classification_report

from . import gnn,mlp
from ..utils import VNG_utils
from ..utils.config import GraphEncoderConfig

class GCL(nn.Module):
    def __init__(self,in_feats,out_feats) -> None:
        super().__init__()
        self.graph_encoder = GNN_learner(in_feats,out_feats,layers=3,drop=0.4)
    def forward(self,graph,feature):
        g = self.graph_encoder(graph,feature)
        return g

    def cl_loss(x, x_aug, temperature=0.2, sym=True):
        # x : anchor
        # x_aug : learned feature
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1

def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

class GNN_learner(nn.Module):
    def __init__(self, in_feats, out_feats, layers, k):
        super(GNN_learner, self).__init__()
        self.gnn_learner = gnn.ew_GCN(in_feats,out_feats,layers)
        self.input_dim = in_feats
        self.output_dim = out_feats
        self.k = k
    #     self.param_init()

    # def param_init(self):
    #     for layer in self.layers:
    #         layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, graph, feature:torch.tensor):
        embeddings = self.gnn_learner(graph, feature)
        rows, cols, values = VNG_utils.knn_fast(embeddings, self.k, 1000)
        rows_ = torch.cat((rows, cols))
        cols_ = torch.cat((cols, rows))
        values_ = torch.cat((values, values))
        values_ = F.relu(values_)
        new_grpah = dgl.graph((rows_, cols_), num_nodes=feature.shape[0], device='cuda')
        new_grpah.edata['w'] = values_
        return new_grpah

class DotPredictor(nn.Module):
    def __init__(self,h_feats) -> None:
        self.h_feats = h_feats
        super().__init__()
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return F.sigmoid(graph.edata['score']).squeeze(1)
        
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
    def apply_edges(self, edges):#将节点u的'h'特征和节点v的'h'特征拼接起来然后传入MLP
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        score = F.sigmoid(self.W2(F.relu(self.W1(h))))
        return {"score": score.squeeze(1)}
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata["score"]
    
class EdgeLearner(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.predictar = MLPPredictor(in_feats)
    def forward(self, g, neg_g, x):
        h = x
        return self.predictar(g, h), self.predictar(neg_g, h)
    def predict(self,g,features,threshold=None):
        self.predictar.eval()
        with torch.no_grad():
            score = self.predictar(g,features)
        if threshold is not None:
            score = (score > threshold).to(torch.int32)
        return score

def edge_pred_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(scores.device)
    return F.binary_cross_entropy(scores, labels, reduction="mean")

def train_edge_learner(graph:dgl.DGLGraph,show_details = False,device="cuda:0"):
    graph = graph.to(device)
    dgl_field_name = "feat"
    in_feats = graph.ndata[dgl_field_name].shape[1]
    x = graph.ndata[dgl_field_name]
    h_feats = in_feats // 2
    lr = 1e-4
    epochs = 400
    patience = 5
    patience_count = 0
    best_val_acc = 0
    # [train_pos_g,val_pos_g,test_pos_g],[train_neg_g,val_neg_g,test_neg_g] = VNG_utils.edge_dataset_split(graph)
    [train_pos_g,val_pos_g,test_pos_g],[train_neg_g,val_neg_g,test_neg_g] = VNG_utils.random_edge_dataset_split(graph,{"train":400,"val":100,"test":100})
    model = EdgeLearner(in_feats).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    threshold = 0.35
    for e in range(epochs):
        model.train()
        pos_score, neg_score = model(train_pos_g.to(device),train_neg_g.to(device),x)
        loss = edge_pred_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if((e)%5 == 0):
            model.eval()
            val_pos_score, val_neg_score = model(val_pos_g.to(device),val_neg_g.to(device),x)
            val_loss = edge_pred_loss(val_pos_score, val_neg_score)
            preds = (torch.cat([val_pos_score, val_neg_score])>threshold).to(torch.int32)
            labels = torch.cat([torch.ones(val_pos_score.shape[0]), torch.zeros(val_neg_score.shape[0])]).to(device,torch.int32)
            report = classification_report(labels.detach().to("cpu"),preds.detach().to("cpu"),output_dict=True,zero_division=np.nan)
            if show_details : 
                print('In epoch {}, train loss: {:.4f}, val acc: {:.4f}(best {}) patience{}'.format(e+1, loss.item(), report["accuracy"], best_val_acc, patience_count))
            if(best_val_acc < report["accuracy"]):
                best_val_acc = report["accuracy"]
                patience_count = 0
            else:
                patience_count += 1
            if(patience_count > patience):
                # print(classification_report(labels.detach().to("cpu"),preds.detach().to("cpu"),zero_division=np.nan))
                break
    # VNG_utils.save(model,"CGDM-Im\history_data\gsl_model_checkpoint\GNN_structure_learner"+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+".pth")
    if show_details :
        print("test : ")
        test_pos_score, test_neg_score = model(test_pos_g.to(device),test_neg_g.to(device),x)
        preds = (torch.cat([test_pos_score, test_neg_score])>threshold).to(torch.int32)
        labels = torch.cat([torch.ones(test_pos_score.shape[0]), torch.zeros(test_neg_score.shape[0])]).to(device,torch.int32)
        report = classification_report(labels.detach().to("cpu"),preds.detach().to("cpu"),zero_division=np.nan)
        print(report)
    return model

class StructureLeaner(nn.Module):
    def __init__(self,layers,gnn_type,gnn_in_feats,gnn_out_feats):
        super().__init__()
        self.gnn_encoder = gnn.JointGNN(layers,gnn_type,gnn_in_feats,gnn_out_feats)
        self.edge_learner = EdgeLearner(gnn_out_feats)
    def forward(self,g:dgl.DGLGraph,x:torch.Tensor):
        h,recon_feat = self.gnn_encoder(g,x)
        edge_score = self.edge_learner(g,h)
        return h,recon_feat,edge_score

def train_sl(encoder_config:GraphEncoderConfig,show_detail = False, device="cuda:0"):
    file_path = "CGDM-Im\\history_data\\encoder_model_checkpoint\\".replace("\\",os.sep)
    if encoder_config.graph_file != None:
        graph_list,_ = dgl.load_graphs(encoder_config.graph_file)
        g = graph_list[0]
    else:
        g = encoder_config.dataset

    in_feats =  g.ndata["feat"].shape[-1] # = encoder_config.in_size
    classes = g.ndata["label"].shape[-1]
    out_feats = encoder_config.latent_size

    # training
    model = StructureLeaner(encoder_config.layers, encoder_config.gnn,in_feats,out_feats).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), encoder_config.learning_rate)
    centloss_criterion = gnn.CenterLoss(classes,out_feats).to(device)
    centloss_optimizer = torch.optim.Adam(centloss_criterion.parameters(), lr=encoder_config.alpha_lr)
    labels = g.ndata['label']
    [train_pos_g,val_pos_g,test_pos_g],[train_neg_g,val_neg_g,test_neg_g] = VNG_utils.random_edge_dataset_split(g,{"train":400,"val":100,"test":100})
    if labels.dim() > 1:
        labels = labels.argmax(1)
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    best_val_loss = float("inf")
    # best_val_acc = 0.
    patience_count = 0
    for e in range(encoder_config.epochs):
        # Forward
        embedding,recon,edge_score = model(g, g.ndata["feat"])
        # pred = logits.argmax(1)
        # classify_loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        recon_loss = F.mse_loss(recon[train_mask],g.ndata["feat"][train_mask]) 
        dis_loss = centloss_criterion(embedding[train_mask],labels[train_mask])
        loss = recon_loss + encoder_config.beta * dis_loss # 用参数alpha加权两个任务损失

        if (e % 5 == 0) :
            val_recon_loss = F.mse_loss(recon[val_mask],g.ndata["feat"][val_mask])  
            val_dis_loss = centloss_criterion(embedding[val_mask],labels[val_mask])
            val_loss = val_recon_loss + encoder_config.beta * val_dis_loss # 用参数alpha加权两个任务损失
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                patience_count = 0
            else:
                patience_count += 1
            if patience_count > encoder_config.patience: break
            if show_detail:
                print('In epoch {}, loss: {:.3f} = {:.3f} + {:.3f}*{:.3f}, val loss{}(best {:.3f}) patience{}'.format(
                        e+1, loss.item(), recon_loss.item(), encoder_config.beta, dis_loss.item(), val_loss, best_val_loss, patience_count))

        model_optimizer.zero_grad()
        centloss_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        centloss_optimizer.step()
        if encoder_config.save_cp and ((e + 1) % encoder_config.save_cp == 0):
            VNG_utils.save(model.gcn_encoder, file_path + 'GCNEncoder'+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'e'+str(e+1)+'.pth')
    if encoder_config.save_cp : VNG_utils.save(model.gcn_encoder, file_path + 'GCNEncoder'+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'.pth')
    return model

def GSL_loss(graph:dgl.DGLGraph,x,structure_learner,graph_encoder):
    z1 = graph_encoder(graph,x)
    learned_graph = structure_learner(graph,x)
    z2 = graph_encoder(learned_graph,z1)
    loss = GCL.cl_loss(z1,z2)
    return loss, learned_graph

def sim_based_graph_gen(ori_graph:dgl.DGLGraph,virtual_node_feature:torch.Tensor,virtual_node_information:dict
                        ,mask:torch.Tensor=None,feature_field_name:str="feat",top_k:int = 2):
    """给出基于特征相似度的视图

    Args:
        graph (dgl.DGLGraph): 原图
        mask (torch.Tensor): 参与相似度排名的节点掩码,默认所有节点都参与
        feature_field_name (str): DGLGraph特征的字段
        top_k (_type_): topk参数
        virtual_node_information (_type_): 生成节点需要添加的信息，用字典储存，比如{"feat":feature,"label":labels}
        virtual_node_feature (_type_): 生成节点的特征

    Returns:
        _type_: _description_
    """
    graph = copy.deepcopy(ori_graph)
    ori_feat = graph.ndata[feature_field_name]
    num_nodes = graph.num_nodes()
    assert (len(ori_feat) >= top_k),"k is larger than num_nodes"
    num_virtual_nodes = len(virtual_node_feature)
    graph.add_nodes(len(virtual_node_feature),virtual_node_information)
    sim_matrix = torch.mm(virtual_node_feature,ori_feat.t()).clamp(min=1e-4)
    if mask is not None:
        assert (mask.sum() >= top_k) , "mask is narrower than topk"
        sim_matrix = torch.einsum("vn,n->vn",sim_matrix,mask)
    _, v_indx = sim_matrix.topk(k=top_k, dim=-1)
    u = torch.tensor(range(num_nodes,num_nodes+num_virtual_nodes),device=virtual_node_feature.device).view([-1,1]).expand((num_virtual_nodes,top_k)).reshape([1,-1]).squeeze(0)
    v = v_indx.view(-1)
    # add directed edge
    graph.add_edges(u,v)
    graph.add_edges(v,u)
    graph.ndata["train_mask"][-num_virtual_nodes:] = True
    graph.ndata["test_mask"][-num_virtual_nodes:] = False
    graph.ndata["val_mask"][-num_virtual_nodes:] = False
    
    return graph

def sim_based_nei_mixup(ori_graph:dgl.DGLGraph,virtual_node_feature:torch.Tensor,virtual_node_information:dict
                        ,mask:torch.Tensor=None,feature_field_name:str="feat",top_k:int = 4, device = "cuda:0"):
    graph = copy.deepcopy(ori_graph)
    rand_topk_scale = 4
    ori_feat = graph.ndata[feature_field_name]
    num_nodes = graph.num_nodes()
    assert (len(ori_feat) >= top_k),"topk({}) is larger than num_nodes({})".format(top_k,len(ori_feat))
    num_virtual_nodes = len(virtual_node_feature)
    graph.add_nodes(len(virtual_node_feature),virtual_node_information)
    sim_matrix = torch.mm(virtual_node_feature,ori_feat.t()).clamp(min=1e-4)
    if mask is not None:
        assert (mask.sum() >= top_k) , "mask({}) is narrower than topk({})".format(mask.sum(),top_k)
        sim_matrix = torch.einsum("vn,n->vn",sim_matrix,mask)
    _, v_indx = sim_matrix.topk(k=2*rand_topk_scale, dim=-1)
    v_nodes = []
    for nodes in v_indx:
        start_nodes = nodes[random.sample(range(len(nodes)),3)]
        neis_g = dgl.in_subgraph(graph,start_nodes,relabel_nodes=True)
        neis_id = neis_g.ndata[dgl.NID] # [n for n in neis_g.ndata[dgl.NID] if n not in start_nodes]
        # neis_g.edata[dgl.EID]
        indices = random.sample(range(len(neis_id)),top_k)
        v_nodes.append(neis_id[indices])
    u = torch.tensor(range(num_nodes,num_nodes+num_virtual_nodes),device=virtual_node_feature.device).view([-1,1]).expand((num_virtual_nodes,top_k)).reshape([1,-1]).squeeze(0)
    v = torch.stack(v_nodes).to(device).view(-1)
    # add directed edge
    graph.add_edges(u,v)
    graph.add_edges(v,u)
    graph.ndata["train_mask"][-num_virtual_nodes:] = True
    graph.ndata["test_mask"][-num_virtual_nodes:] = False
    graph.ndata["val_mask"][-num_virtual_nodes:] = False
    
    return graph
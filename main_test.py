import copy
import torch
import random
import warnings
import statistics
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import train_test_split_edges,negative_sampling
from sklearn.metrics import balanced_accuracy_score, f1_score,classification_report,roc_auc_score, accuracy_score, recall_score

from args import parse_args
from src import solid,loss_fn
from src.utils import VNG_utils,tab_dataset_util
from src.TabDiff.tabdiff.metrics import TabMetrics
from src.TabDiff.tabdiff.modules.main_modules import UniModMLP
from src.TabDiff.tabdiff.modules.main_modules import Model
from src.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from src.models import gnn,sage,gcn,gat,edge_learner,teacher,diffusion,mlp
from src.denoise import unet
import src.utils.graphbuilder
warnings.filterwarnings("ignore")

# def pre_train():
#     decoder.train()
#     neg_edge_index = negative_sampling(
#         edge_index=data.train_pos_edge_index,
#         num_nodes=data.num_nodes,
#         num_neg_samples=data.train_pos_edge_index.size(1))
#     edge_labels = torch.cat([torch.ones(data.train_pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))]).to(device)
#     de_optimizer.zero_grad()
#     encoder.train()
#     centloss_criterion.train()
#     en_optimizer.zero_grad()
#     centloss_optimizer.zero_grad()
#     emb = encoder(data.x, edge_index[:,train_edge_mask], None)
#     cent_loss = centloss_criterion(emb[data_train_mask],data.y[data_train_mask])
#     pos_edge_scores = decoder(emb,data.train_pos_edge_index)
#     neg_edge_scores = decoder(emb, neg_edge_index)
#     edge_scores = torch.cat([pos_edge_scores,neg_edge_scores],dim=0)
#     de_loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels)
#     loss = args.w_con_loss * cent_loss+ de_loss
#     loss.backward()
#     # for param in centloss_criterion.parameters():
#     #     param.grad.data *= (1./args.w_con_loss)
#     with torch.no_grad():
#         encoder.eval()
#         centloss_criterion.eval()
#         decoder.eval()
#         emb = encoder(data.x, edge_index[:,train_edge_mask], None)
#         val_cent_loss = centloss_criterion(emb[data_val_mask],data.y[data_val_mask])
#         val_pos_edge_scores = decoder(emb, data.val_pos_edge_index)
#         val_neg_edge_scores = decoder(emb, data.val_neg_edge_index.to(device))
#         val_edge_scores = torch.cat([val_pos_edge_scores,val_neg_edge_scores],dim=0)
#         val_edge_labels = torch.cat([torch.ones(data.val_pos_edge_index.size(1)), torch.zeros(data.val_neg_edge_index.size(1))]).to(device)
#         val_recon_loss = F.binary_cross_entropy_with_logits(val_edge_scores, val_edge_labels)
#         val_loss = args.w_con_loss * val_cent_loss + val_recon_loss
#     en_optimizer.step()
#     # cent_scheduler.step(val_cent_loss)
#     centloss_optimizer.step()
#     # de_optimizer.step()
#     de_scheduler.step(val_recon_loss)
#     return val_loss,val_cent_loss, val_recon_loss

def train_teacher():
    teacher_model.train()
    teacher_optimizer.zero_grad()
    inputs = data.x.to(device)
    logits = teacher_model(inputs[data_train_mask])
    loss = F.cross_entropy(logits,data.y[data_train_mask])
    loss.backward()
    teacher_optimizer.step()
    with torch.no_grad():
        teacher_model.eval()
        logits = teacher_model(inputs[data_val_mask])
        val_loss = F.cross_entropy(logits,data.y[data_val_mask])
    return val_loss

# def train_diffusion_model():
#     train_feat_data = emb_data.x[data_train_mask]
#     train_label_data = emb_data.y[data_train_mask]

#     eval_feat = emb_data.x[data_val_mask]
#     eval_label = emb_data.y[data_val_mask]
#     class_mask = torch.zeros((train_feat_data.shape[0]),dtype=torch.bool,device=device)
#     train_dataset = TensorDataset(train_feat_data,train_label_data,class_mask)
#     data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
#     loss_value = 0.0
#     class_dis = VNG_utils.class_dis(emb_data.y[data_train_mask],n_cls)
#     class_dis = class_dis / sum(class_dis)
#     class_mask = VNG_utils.dis_based_class_mask(train_label_data,class_dis,n_cls,train_label_data.shape[0],args.guidance_drop_prob,adjustment_factor=args.adjustment_factor,device=device)
#     # print("{} guidance, hard label of guidance is:".format(sum(class_mask)))
#     # print(train_label_data[class_mask].argmax(1))
#     train_dataset.tensors = (train_feat_data, train_label_data, class_mask)
#     data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
#     for inputs,targets,c_mask in data_loader:
#         # 对节点特征进行padding,以避免unet下采样中出现奇数纬度导致分辨率不匹配
#         inputs = F.pad(inputs,pad=args.padding,mode="constant",value=0).to(device)
#         targets = F.one_hot(targets,num_classes=n_cls)
#         soft_labels = teacher_model.softmax_with_temperature(inputs,args.temperature)
#         targets = soft_labels * (1. - args.hard_factor) + targets * args.hard_factor
        
#         inputs = torch.unsqueeze(inputs,dim=1)
#         # print(inputs.shape)
#         dif_optimizer.zero_grad()
#         targets = targets.to(device)
#         # 按照一定的概率将guidance置空，由此只用一个backbone训练出适用于两种情况（有无条件）的模型
#         loss = diffusion_model.loss(inputs,targets,c_mask.to(torch.int32),args.padding)
#         loss.backward()
#         dif_optimizer.step()
#         loss_value += loss.item()
#     val_class_mask = VNG_utils.dis_based_class_mask(eval_label,class_dis,n_cls,eval_label.shape[0],args.guidance_drop_prob,adjustment_factor=args.adjustment_factor,device=device)
#     eval_data = [eval_feat,eval_label,val_class_mask.to(torch.int32)]
#     val_loss = eval_diffusion_model(eval_data)
#     return val_loss

def train_tabdiff():
    train_feat_data = data.x[data_train_mask]
    train_label_data = data.y[data_train_mask]
    eval_feat = data.x[data_val_mask]
    eval_label = data.y[data_val_mask]
    loss_value = 0.0
    class_dis = VNG_utils.class_dis(data.y[data_train_mask],n_cls)
    class_dis = class_dis / sum(class_dis)
    class_mask = VNG_utils.dis_based_class_mask(train_label_data,class_dis,n_cls,train_label_data.shape[0],args.guidance_drop_prob,adjustment_factor=args.adjustment_factor,device=device)
    class_mask = class_mask[:, None].repeat(1,n_cls).to(torch.int32)
    class_mask = (-1*(1-class_mask))
    # print("{} guidance, hard label of guidance is:".format(sum(class_mask)))
    # print(train_label_data[class_mask].argmax(1))
    train_dataset = TensorDataset(train_feat_data, train_label_data, class_mask)
    data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    for inputs,targets,c_mask in data_loader:
        targets = F.one_hot(targets,num_classes=n_cls)
        soft_labels = teacher_model.softmax_with_temperature(inputs,args.temperature)
        targets = soft_labels * (1. - args.hard_factor) + targets * args.hard_factor
        inputs = torch.unsqueeze(inputs,dim=1)
        # print(inputs.shape)
        dif_optimizer.zero_grad()
        targets = targets.to(device)
        # 按照一定的概率将guidance置空，由此只用一个backbone训练出适用于两种情况（有无条件）的模型
        dloss, closs = tab_diffusion.mixed_loss(inputs,targets,c_mask.to(torch.int32))
        loss = args.dloss_weight * dloss + args.closs_weight * closs
        loss.backward()
        dif_optimizer.step()
        loss_value += loss.item()
    val_class_mask = VNG_utils.dis_based_class_mask(eval_label,class_dis,n_cls,eval_label.shape[0],args.guidance_drop_prob,adjustment_factor=args.adjustment_factor,device=device)
    eval_data = [eval_feat,eval_label,val_class_mask.to(torch.int32)]
    val_loss = eval_diffusion_model(eval_data)
    return val_loss

@torch.no_grad()
def eval_diffusion_model(eval_data):
    tab_diffusion.eval()
    with torch.no_grad():
        feat_dataset = TensorDataset(eval_data[0],eval_data[1],eval_data[2])
        test_data_loader = DataLoader(feat_dataset, batch_size=32, shuffle=True)
        data_size = len(test_data_loader.dataset)
        loss_value = 0.0
        for inputs,targets,c_mask in test_data_loader:
            # 对节点特征进行padding以避免unet下采样中出现奇数纬度导致分辨率不匹配
            inputs = inputs.to(device)
            targets = F.one_hot(targets,num_classes=n_cls)
            soft_labels = teacher_model.softmax_with_temperature(inputs,args.temperature)
            targets = soft_labels * (1. - args.hard_factor) + targets * args.hard_factor
            inputs = torch.unsqueeze(inputs,dim=1)
            targets = targets.to(device)
            # class_mask = (torch.rand(targets.shape[0]) < 0.15).to(device,torch.int32)
            # c_mask = torch.ones_like(c_mask,device=device)
            dloss, closs = tab_diffusion.mixed_loss(inputs,targets,c_mask.to(torch.int32))
            loss = args.dloss_weight * dloss + args.closs_weight * closs
            loss_value += loss.item()
    return loss_value / data_size

# @torch.no_grad()
# def eval_diffusion_model(eval_data):
#     diffusion_model.eval()
#     with torch.no_grad():
#         feat_dataset = TensorDataset(eval_data[0],eval_data[1],eval_data[2])
#         test_data_loader = DataLoader(feat_dataset, batch_size=32, shuffle=True)
#         data_size = len(test_data_loader.dataset)
#         loss_value = 0.0
#         for inputs,targets,c_mask in test_data_loader:
#             # 对节点特征进行padding以避免unet下采样中出现奇数纬度导致分辨率不匹配
#             inputs = F.pad(inputs,pad=args.padding,mode="constant",value=0).to(device)
#             targets = F.one_hot(targets,num_classes=n_cls)
#             soft_labels = teacher_model.softmax_with_temperature(inputs,args.temperature)
#             targets = soft_labels * (1. - args.hard_factor) + targets * args.hard_factor
#             inputs = torch.unsqueeze(inputs,dim=1)
#             targets = targets.to(device)
#             # class_mask = (torch.rand(targets.shape[0]) < 0.15).to(device,torch.int32)
#             # c_mask = torch.ones_like(c_mask,device=device)
#             loss = diffusion_model.loss(inputs,targets,c_mask,args.padding)
#             loss_value += loss.item()
#     return loss_value / data_size

# def train_gnn_classifier():
#     classifier.train()
#     classifier_optimizer.zero_grad()
#     output = classifier(aug_data.x, new_edge_index[:,train_edge_mask], None)
#     classifier_criterion(output[data_train_mask], aug_data.y[data_train_mask], weight=weights).backward()
#     with torch.no_grad():
#         classifier.eval()
#         output = classifier(aug_data.x, new_edge_index[:,train_edge_mask], None)
#         val_loss= F.cross_entropy(output[data_val_mask], aug_data.y[data_val_mask])

#     classifier_optimizer.step()
#     scheduler.step(val_loss)

# @torch.no_grad()
# def test_gnn_classifier():
#     classifier.eval()
#     logits = classifier(aug_data.x, new_edge_index[:,train_edge_mask], None,)
#     accs, baccs, f1s = [], [], []

#     for i, mask in enumerate([data_train_mask, data_val_mask, data_test_mask]):
#         pred = logits[mask].max(1)[1]
#         y_pred = pred.cpu().numpy()
#         y_true = aug_data.y[mask].cpu().numpy()
#         acc = pred.eq(aug_data.y[mask]).sum().item() / mask.sum().item()
#         bacc = balanced_accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='macro')
#         recall = recall_score(y_true, y_pred, average=None)
#         accs.append(acc)
#         baccs.append(bacc)
#         f1s.append(f1)
#         measure_result = classification_report(y_true, y_pred,digits=4, zero_division=np.nan)
#     return accs, baccs, f1s, measure_result, recall


args = parse_args()
print(args)
reweight = False

device = args.device
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
tab_dataset = tab_dataset_util.load_tab_dataset_info(args.dataset, path, split_type='full')
dataset = tab_dataset.graph
data = tab_dataset.graph.to(device)
n_feat = tab_dataset.n_features
print(data)
n_cls = tab_dataset.n_labels
ori_edge_index = data.edge_index
data = train_test_split_edges(data)

repeatition = 5
max_n=500
avg_test_acc, avg_val_acc, avg_val_f1, avg_test_bacc, avg_test_f1 = [], [], [], [], []
mi_recall = []
ma_recall = []

for r in range(repeatition):
    args.seed = args.seed + 1
    ## Fix seed ##
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)

    edge_index = copy.deepcopy(ori_edge_index)
    if args.dataset in ['Cora','CiteSeer','PubMed']:
        data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
        stats = data.y[data_train_mask]
        n_data = []
        for i in range(n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        idx_info = VNG_utils.get_idx_info(data.y, n_cls, data_train_mask)
        class_num_list = n_data
        print("num of class in original training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
        class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = src.utils.graphbuilder.make_longtailed_data_remove(edge_index, data.y, n_data, n_cls, args.imb_ratio, data_train_mask.clone(), max_n)
        if args.keep_edge:
            train_edge_mask = torch.ones_like(train_edge_mask,dtype=torch.bool,device=train_edge_mask.device)
        print("num of class in LT-training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
        minority_mask = class_num_list < (sum(class_num_list)/n_cls)
        minority_class = [i for i in range(n_cls) if minority_mask[i]]
        print("minority classes {}".format(minority_class))
        print("number of edges {}".format(sum(train_edge_mask)))
    elif args.dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
        train_idx, valid_idx, test_idx, train_node = VNG_utils.get_step_split(imb_ratio=args.imb_ratio, \
                                                                    valid_each=int(data.x.shape[0] * 0.1 / n_cls), \
                                                                    labeling_ratio=0.1, \
                                                                    all_idx=list(range(data.x.shape[0])), \
                                                                    all_label=data.y.cpu().detach().numpy(), \
                                                                    nclass=n_cls)
        data_train_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        data_val_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        data_test_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        data_train_mask[train_idx] = True
        data_val_mask[valid_idx] = True
        data_test_mask[test_idx] = True
        train_idx = data_train_mask.nonzero().squeeze()
        train_edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool).to(device)

        class_num_list = [len(item) for item in train_node]
        idx_info = [torch.tensor(item) for item in train_node]
        print("num of class in step data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
        minority_mask = torch.tensor(class_num_list) < (sum(class_num_list)/n_cls)
        minority_class = [i for i in range(n_cls) if minority_mask[i]]
        print("minority classes {}".format(minority_class))
        print("number of edges {}".format(sum(train_edge_mask)))

    best_loss = float('inf')
    patience = 5
    patience_count = 0
    patience_beta = 1e-3
    teacher_model = teacher.MLPTeacher(n_feat,n_cls,layers=1,drop=0.4).to(device)
    teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
    with tqdm(total=args.epochs, desc="Teacher Training Progress") as pbar:
        for e in range(args.epochs):
            val_loss = train_teacher()
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



    # training process for diffusion model
    denoise_kwargs = {
        "d_numerical" : tab_dataset.num_numerical_features, 
        "categories" : tab_dataset.categories, 
        "num_layers" : args.denoise_layers, 
        "d_token" : args.d_token
    }

    denoise_backbone = UniModMLP(
        **denoise_kwargs
    )

    # model_kwargs = {}
    denoise_model = Model(denoise_backbone)
    denoise_model.to(device)

    diffusion_kwargs = {
        "noise_dist" : "uniform_t"
    }
    tab_diffusion = UnifiedCtimeDiffusion(
        num_classes=tab_dataset.categories,
        num_numerical_features=tab_dataset.num_numerical_features,
        denoise_fn=denoise_model,
        y_only_model=None,
        **diffusion_kwargs,
        device=device,
    )
    num_params = sum(p.numel() for p in tab_diffusion.parameters())
    print("The number of parameters = ", num_params)
    tab_diffusion.to(device)
    tab_diffusion.train()

    dif_optimizer = torch.optim.Adam(tab_diffusion.parameters(), lr=args.dif_lr)
    train_tabdiff()

    best_loss = float('inf')
    patience = 10
    patience_count = 0
    patience_beta = 2e-4
    dif_epoch = 1000
    with tqdm(total=dif_epoch, desc="Diffusion Training") as pbar:
        for e in range(dif_epoch):
            val_loss = train_tabdiff()
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


if repeatition == 1 : exit()
# ## Calculate statistics ##
# acc_CI =  (statistics.stdev(avg_test_acc) / (repeatition ** (1/2)))
# bacc_CI =  (statistics.stdev(avg_test_bacc) / (repeatition ** (1/2)))
# f1_CI =  (statistics.stdev(avg_test_f1) / (repeatition ** (1/2)))
# mi_recall_CI = (statistics.stdev(mi_recall) / (repeatition ** (1/2)))
# ma_recall_CI = (statistics.stdev(ma_recall) / (repeatition ** (1/2)))
# avg_acc = statistics.mean(avg_test_acc)
# avg_val_acc = statistics.mean(avg_val_acc)
# avg_val_f1 = statistics.mean(avg_val_f1)
# avg_bacc = statistics.mean(avg_test_bacc)
# avg_f1 = statistics.mean(avg_test_f1)
# avg_mi_recall = statistics.mean(mi_recall)
# avg_ma_recall = statistics.mean(ma_recall)


# avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f}, Mi recall {:.4f}+-{:.4f}, Ma recall {:.4f}+-{:.4f}'
# avg_log = avg_log.format(avg_acc ,acc_CI ,avg_bacc, bacc_CI, avg_f1, f1_CI, avg_val_acc, avg_val_f1,avg_mi_recall,mi_recall_CI,avg_ma_recall,ma_recall_CI)
# log = "{}".format(avg_log)
# print(log)

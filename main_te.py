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
from src.utils import VNG_utils
from src.models import gnn,sage,gcn,gat,edge_learner,teacher,diffusion,mlp
from src.denoise import unet
import src.utils.graphbuilder
warnings.filterwarnings("ignore")

def pre_train():
    decoder.train()
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    edge_labels = torch.cat([torch.ones(data.train_pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))]).to(device)
    de_optimizer.zero_grad()
    encoder.train()
    centloss_criterion.train()
    en_optimizer.zero_grad()
    centloss_optimizer.zero_grad()
    emb = encoder(data.x, edge_index[:,train_edge_mask], None)
    cent_loss = centloss_criterion(emb[data_train_mask],data.y[data_train_mask])
    pos_edge_scores = decoder(emb,data.train_pos_edge_index)
    neg_edge_scores = decoder(emb, neg_edge_index)
    edge_scores = torch.cat([pos_edge_scores,neg_edge_scores],dim=0)
    de_loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels)
    loss = args.w_con_loss * cent_loss+ de_loss
    loss.backward()
    # for param in centloss_criterion.parameters():
    #     param.grad.data *= (1./args.w_con_loss)
    with torch.no_grad():
        encoder.eval()
        centloss_criterion.eval()
        decoder.eval()
        emb = encoder(data.x, edge_index[:,train_edge_mask], None)
        val_cent_loss = centloss_criterion(emb[data_val_mask],data.y[data_val_mask])
        val_pos_edge_scores = decoder(emb, data.val_pos_edge_index)
        val_neg_edge_scores = decoder(emb, data.val_neg_edge_index.to(device))
        val_edge_scores = torch.cat([val_pos_edge_scores,val_neg_edge_scores],dim=0)
        val_edge_labels = torch.cat([torch.ones(data.val_pos_edge_index.size(1)), torch.zeros(data.val_neg_edge_index.size(1))]).to(device)
        val_recon_loss = F.binary_cross_entropy_with_logits(val_edge_scores, val_edge_labels)
        val_loss = args.w_con_loss * val_cent_loss + val_recon_loss
    en_optimizer.step()
    # cent_scheduler.step(val_cent_loss)
    centloss_optimizer.step()
    # de_optimizer.step()
    de_scheduler.step(val_recon_loss)
    return val_loss,val_cent_loss, val_recon_loss

def train_teacher():
    teacher_model.train()
    teacher_optimizer.zero_grad()
    inputs = F.pad(emb_data.x,pad=args.padding,mode="constant",value=0).to(device)
    logits = teacher_model(inputs[data_train_mask])
    loss = F.cross_entropy(logits,data.y[data_train_mask])
    loss.backward()
    teacher_optimizer.step()
    with torch.no_grad():
        teacher_model.eval()
        logits = teacher_model(inputs[data_val_mask])
        val_loss = F.cross_entropy(logits,data.y[data_val_mask])
    return val_loss

def train_diffusion_model():
    train_feat_data = emb_data.x[data_train_mask]
    train_label_data = emb_data.y[data_train_mask]

    eval_feat = emb_data.x[data_val_mask]
    eval_label = emb_data.y[data_val_mask]
    class_mask = torch.zeros((train_feat_data.shape[0]),dtype=torch.bool,device=device)
    train_dataset = TensorDataset(train_feat_data,train_label_data,class_mask)
    data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    loss_value = 0.0
    class_dis = VNG_utils.class_dis(emb_data.y[data_train_mask],n_cls)
    class_dis = class_dis / sum(class_dis)
    class_mask = VNG_utils.dis_based_class_mask(train_label_data,class_dis,n_cls,train_label_data.shape[0],args.guidance_drop_prob,adjustment_factor=args.adjustment_factor,device=device)
    # print("{} guidance, hard label of guidance is:".format(sum(class_mask)))
    # print(train_label_data[class_mask].argmax(1))
    train_dataset.tensors = (train_feat_data, train_label_data, class_mask)
    data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    for inputs,targets,c_mask in data_loader:
        # 对节点特征进行padding,以避免unet下采样中出现奇数纬度导致分辨率不匹配
        inputs = F.pad(inputs,pad=args.padding,mode="constant",value=0).to(device)
        targets = F.one_hot(targets,num_classes=n_cls)
        soft_labels = teacher_model.softmax_with_temperature(inputs,args.temperature)
        targets = soft_labels * (1. - args.hard_factor) + targets * args.hard_factor
        inputs = torch.unsqueeze(inputs,dim=1)
        # print(inputs.shape)
        dif_optimizer.zero_grad()
        targets = targets.to(device)
        # 按照一定的概率将guidance置空，由此只用一个backbone训练出适用于两种情况（有无条件）的模型
        loss = diffusion_model.loss(inputs,targets,c_mask.to(torch.int32),args.padding)
        loss.backward()
        dif_optimizer.step()
        loss_value += loss.item()
    val_class_mask = VNG_utils.dis_based_class_mask(eval_label,class_dis,n_cls,eval_label.shape[0],args.guidance_drop_prob,adjustment_factor=args.adjustment_factor,device=device)
    eval_data = [eval_feat,eval_label,val_class_mask.to(torch.int32)]
    val_loss = eval_diffusion_model(eval_data)
    return val_loss

@torch.no_grad()
def eval_diffusion_model(eval_data):
    diffusion_model.eval()
    with torch.no_grad():
        feat_dataset = TensorDataset(eval_data[0],eval_data[1],eval_data[2])
        test_data_loader = DataLoader(feat_dataset, batch_size=32, shuffle=True)
        data_size = len(test_data_loader.dataset)
        loss_value = 0.0
        for inputs,targets,c_mask in test_data_loader:
            # 对节点特征进行padding以避免unet下采样中出现奇数纬度导致分辨率不匹配
            inputs = F.pad(inputs,pad=args.padding,mode="constant",value=0).to(device)
            targets = F.one_hot(targets,num_classes=n_cls)
            soft_labels = teacher_model.softmax_with_temperature(inputs,args.temperature)
            targets = soft_labels * (1. - args.hard_factor) + targets * args.hard_factor
            inputs = torch.unsqueeze(inputs,dim=1)
            targets = targets.to(device)
            # class_mask = (torch.rand(targets.shape[0]) < 0.15).to(device,torch.int32)
            # c_mask = torch.ones_like(c_mask,device=device)
            loss = diffusion_model.loss(inputs,targets,c_mask,args.padding)
            loss_value += loss.item()
    return loss_value / data_size

def train_gnn_classifier():
    classifier.train()
    classifier_optimizer.zero_grad()
    output = classifier(aug_data.x, new_edge_index[:,train_edge_mask], None)
    classifier_criterion(output[data_train_mask], aug_data.y[data_train_mask], weight=weights).backward()
    with torch.no_grad():
        classifier.eval()
        output = classifier(aug_data.x, new_edge_index[:,train_edge_mask], None)
        val_loss= F.cross_entropy(output[data_val_mask], aug_data.y[data_val_mask])

    classifier_optimizer.step()
    scheduler.step(val_loss)

@torch.no_grad()
def test_gnn_classifier():
    classifier.eval()
    logits = classifier(aug_data.x, new_edge_index[:,train_edge_mask], None,)
    accs, baccs, f1s = [], [], []

    for i, mask in enumerate([data_train_mask, data_val_mask, data_test_mask]):
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = aug_data.y[mask].cpu().numpy()
        acc = pred.eq(aug_data.y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average=None)
        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
        measure_result = classification_report(y_true, y_pred,digits=4, zero_division=np.nan)
    return accs, baccs, f1s, measure_result, recall


args = parse_args()
print(args)
reweight = False

device = args.device
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
dataset = VNG_utils.get_dataset(args.dataset,path,split_type="full")
n_feat=dataset.num_features
data = dataset[0].to(device)
print(data)
n_cls = data.y.max().item() + 1
ori_edge_index = data.edge_index
data = train_test_split_edges(data)

repeatition = 5
max_n=500
overall_test_acc, overall_val_acc, overall_val_f1, overall_test_bacc, overall_test_f1 = [], [], [], [], []
overall_mi_recall = []
overall_ma_recall = []

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

    if reweight:
        # calculate weights
        class_count = torch.bincount(data.y[data_train_mask].view(-1), minlength=n_cls).to(device,torch.float)
        total_count = class_count.sum()
        weights = total_count / (n_cls * class_count)  # reverse weight
        weights[torch.isinf(weights)] = 0  # avoid inf
        weights = weights / torch.sum(weights)
    else:
        weights = None
    print(weights)

    if not args.is_vanilla:
        if not args.raw_space:
            
            # if args.net == "SAGE":
            #     if args.n_layers == 1:
            #         encoder = sage.GraphSAGE_single(n_feat,args.n_hid,args.n_hid,1,dropout=0.6).to(device)
            #     else:
            #         classifier = sage.GraphSAGE(n_feat,args.n_hid,args.n_hid,1,dropout=0.6).to(device)
            # elif args.net == "GCN":
            #     if args.n_layers == 1:
            #         classifier = gcn.GCN_single(n_feat,args.n_hid,args.n_hid,1,dropout=0.6).to(device)
            #     else:
            #         classifier = gcn.GCN(n_feat,args.n_hid,args.n_hid,1,dropout=0.6).to(device)
            # elif args.net == "GAT":
            #     if args.n_layers == 1:
            #         classifier = gat.GAT_single(n_feat,args.n_hid,args.n_hid,1,dropout=0.6).to(device)
            #     else:
            #         classifier = gat.GAT(n_feat,args.n_hid,args.n_hid,1,dropout=0.6).to(device)
            # else:
            #     raise NotImplementedError("Not Implemented Architecture!")

            if args.n_en_layers == 1:
                encoder = sage.GraphSAGE_single(n_feat,args.n_hid,args.n_hid,1,dropout=0.6).to(device)
            else:
                encoder = sage.GraphSAGE_res(n_feat,args.n_hid,args.n_hid,nlayer=args.n_en_layers,dropout=0.6).to(device)
                
            en_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.en_lr)
            decoder = edge_learner.EdgePredicter(args.n_hid).to(device)
            de_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.de_lr)
            de_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(de_optimizer, mode='min',
                                                                    factor = 0.8,
                                                                    patience = 100,
                                                                    verbose=False)
            centloss_criterion = loss_fn.CenterLoss(n_cls,args.n_hid,weight=None).to(device)
            centloss_optimizer = torch.optim.Adam(centloss_criterion.parameters(), lr=args.cent_lr)

            best_loss = float('inf')
            patience = 20
            patience_count = 0
            patience_beta = 1e-3
            pre_epoch = 2000
            with tqdm(total=pre_epoch, desc="Pre-train") as pbar:
                for e in range(pre_epoch):
                    val_loss, con_loss, recon_loss = pre_train()
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
            
            encoder.eval()
            embbeddings = encoder(data.x, edge_index[:,train_edge_mask], None).detach()
            emb_data = copy.deepcopy(data)
            emb_data.x = embbeddings
        else:
            args.n_hid = n_feat
            embbeddings = data.x
            emb_data = data
        if not args.wo_diffu_aug:
            left_pad,right_pad,top_pad,bottom_pad = args.padding
            denoise_nhid = args.n_hid + left_pad + right_pad
            # train teacher model
            best_loss = float('inf')
            patience = 5
            patience_count = 0
            patience_beta = 1e-3
            teacher_model = teacher.MLPTeacher(denoise_nhid,n_cls,layers=1,drop=0.4).to(device)
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
                "feature_length": denoise_nhid,
                "n_length": args.n_length,
                "n_channels": args.n_channels,
                "ch_mults": args.ch_mults,
                "is_attn": args.is_attn,
                "n_blocks": args.n_blocks,
                "class_channels": args.class_embedding_channel,
                "time_channels": args.time_embedding_channel,
                "num_class": n_cls,
            }
            eps_model = unet.UNet(**denoise_kwargs)
            # eps_model = unet_vector.UNet(denoise_config)
            if args.beta_schedule == "lin":
                beta = torch.linspace(args.beta_bound[0], args.beta_bound[1], args.T)
            elif args.beta_schedule == "exp":
                beta_exp = args.beta_bound[0] * (args.beta_bound[1] / args.beta_bound[0]) ** (np.arange(args.T) / args.T)
                beta = torch.tensor(beta_exp,dtype=torch.float32)
            elif args.beta_schedule == "quad":
                beta_quad = args.beta_bound[0] + (np.arange(args.T) / args.T) ** 2 * (args.beta_bound[1] - args.beta_bound[0])
                beta = torch.tensor(beta_quad,dtype=torch.float32)
            else:
                print("NO SUCH BETA SCHEDULE:"+args.beta_schedule)
                raise Exception
            diffusion_model = diffusion.GDDPMblock(eps_model,beta,n_steps=args.T,device=device).to(device)
            dif_optimizer = torch.optim.Adam(diffusion_model.eps_model.parameters(), lr=args.dif_lr)

            best_loss = float('inf')
            patience = 10
            patience_count = 0
            patience_beta = 2e-4
            dif_epoch = 1000
            with tqdm(total=dif_epoch, desc="Diffusion Training") as pbar:
                for e in range(dif_epoch):
                    val_loss = train_diffusion_model()
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
            v_information, src_idx = solid.softlabel_based_hard_nodes_sampling(embbeddings[data_train_mask],
                                                                emb_data.y[data_train_mask],
                                                                n_cls,
                                                                diffusion_model,
                                                                teacher_model,
                                                                args.temperature,
                                                                args.padding,
                                                                args.guidance,
                                                                args.hard_factor,
                                                                args.aug_mode,
                                                                is_hard_sample=(args.hard_factor == 1.),
                                                                is_beta_sampling=False)
            sampling_src_idx = (torch.nonzero(data_train_mask, as_tuple=True)[0])[src_idx]
            
            new_node_num = v_information['feat'].shape[0]
            print("{} new nodes".format(new_node_num))
            aug_data, edge_index, data_train_mask, train_edge_mask = solid.add_new_homo_nodes(emb_data,
                                                                            v_information['feat'],
                                                                            v_information['label'],
                                                                            decoder,
                                                                            edge_index,
                                                                            data_train_mask,
                                                                            train_edge_mask,
                                                                            device=device)
            if args.keep_edge:
                train_edge_mask = torch.ones_like(train_edge_mask,dtype=torch.bool,device=train_edge_mask.device)
            new_edge_index = edge_index
            data_val_mask = torch.cat([data_val_mask, torch.zeros(new_node_num, dtype=torch.bool, device=device)])
            data_test_mask = torch.cat([data_test_mask, torch.zeros(new_node_num, dtype=torch.bool, device=device)])
            aug_data.train_mask = data_train_mask
            aug_data.val_mask = data_val_mask
            aug_data.test_mask = data_test_mask
        else:
            new_edge_index = edge_index
            aug_data = emb_data
    elif args.is_vanilla:
        new_edge_index = edge_index
        args.n_hid = n_feat
        aug_data = data
    if args.save_data:
        VNG_utils.save(aug_data,"dataset//aug_graph//aug_data_" + args.dataset + ".pth")
    # # train classifier
    # if args.net == "SAGE":
    #     if args.n_layers == 1:
    #         classifier = sage.GraphSAGE_single(args.n_hid,n_cls,n_cls,args.n_layers,dropout=0.5)
    #     else:
    #         classifier = sage.GraphSAGE(args.n_hid,args.feat_dim,n_cls,args.n_layers,dropout=0.5)
    # elif args.net == "GCN":
    #     if args.n_layers == 1:
    #         classifier = gcn.GCN_single(args.n_hid,n_cls,n_cls,args.n_layers,dropout=0.5)
    #     else:
    #         classifier = gcn.GCN(args.n_hid,args.feat_dim,n_cls,args.n_layers,dropout=0.5)
    # elif args.net == "GAT":
    #     if args.n_layers == 1:
    #         classifier = gat.GAT_single(args.n_hid,n_cls,n_cls,args.n_layers,dropout=0.5)
    #     else:
    #         classifier = gat.GAT(args.n_hid,args.feat_dim,n_cls,args.n_layers,dropout=0.5)
    # elif args.net == "mlp":
    #     classifier = mlp.MLP_f(args.n_hid,n_cls,args.n_layers, drop=0.5)

    classifier = gnn.GNN_classifier(args.net, args.n_hid, args.feat_dim, n_cls, args.n_layers, dropout=0.5)
        
    classifier = classifier.to(device)
    classifier_criterion = loss_fn.CrossEntropy().to(device)
    classifier_optimizer = torch.optim.Adam([
        dict(params=classifier.reg_params, weight_decay=5e-4),
        dict(params=classifier.non_reg_params, weight_decay=0),], lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, mode='min',
                                                            factor = 0.5,
                                                            patience = 100,
                                                            verbose=False)
    best_val_acc = test_acc = best_val_f1 = best_val_bacc = best_val_acc_f1 = -1
    best_measure = None

    # 初始化保存数据的列表
    val_acc_f1_list = []
    test_acc_f1_list = []
    val_f1_list = []
    val_acc_list = []
    tmp_test_acc_list = []
    tmp_test_f1_list = []

    with tqdm(total=args.epochs, desc="Classifier Training Progress") as pbar:
        for e in range(args.epochs):
            train_gnn_classifier()
            accs, bacc, f1s, measure_result, recall = test_gnn_classifier()
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
    framework = "SOLID" if not args.is_vanilla else "vanilla"
    VNG_utils.plot_val_test_acc_f1(val_acc_f1_list, test_acc_f1_list,title='acc_f1_{}'.format(r))
    VNG_utils.plot_val_acc_f1(val_f1_list, val_acc_list,title='val_{}'.format(r))
    VNG_utils.plot_tmp_test_acc_f1(tmp_test_acc_list, tmp_test_f1_list, title='test_{}'.format(r))
    # if not args.is_vanilla:
    #     classifier.eval()
    #     # aug_data.x = classifier.gnn(aug_data.x, new_edge_index[:,train_edge_mask], None)
    #     VNG_utils.show_samples_dis(aug_data,f"history_data//figure//aug_"+args.dataset,new_node_num,n_cls)

    minority_recall = best_recall[minority_mask]
    majority_recall = best_recall[~minority_mask]
    overall_mi_recall.append(sum(minority_recall)/len(minority_recall))
    overall_ma_recall.append(sum(majority_recall)/len(majority_recall))
    print("mi recall {}, ma recall {}".format(sum(minority_recall)/len(minority_recall),sum(majority_recall)/len(majority_recall)))

    overall_val_acc.append(best_val_acc)
    overall_val_f1.append(best_val_f1)
    overall_test_acc.append(test_acc)
    overall_test_bacc.append(test_bacc)
    overall_test_f1.append(test_f1)
    print(best_measure)
    print('Test Acc: {:.4f}, BAcc: {:.4f}, F1: {:.4f}'.format(test_acc,test_bacc,test_f1))

if repeatition == 1 : exit()
## Calculate statistics ##
acc_CI =  (statistics.stdev(overall_test_acc) / (repeatition ** (1/2)))
val_acc_CI =  (statistics.stdev(overall_val_acc) / (repeatition ** (1/2)))
val_f1_CI =  (statistics.stdev(overall_val_f1) / (repeatition ** (1/2)))
bacc_CI =  (statistics.stdev(overall_test_bacc) / (repeatition ** (1/2)))
f1_CI =  (statistics.stdev(overall_test_f1) / (repeatition ** (1/2)))
mi_recall_CI = (statistics.stdev(overall_mi_recall) / (repeatition ** (1/2)))
ma_recall_CI = (statistics.stdev(overall_ma_recall) / (repeatition ** (1/2)))

avg_acc = statistics.mean(overall_test_acc)
avg_val_acc = statistics.mean(overall_val_acc)
avg_val_f1 = statistics.mean(overall_val_f1)
avg_bacc = statistics.mean(overall_test_bacc)
avg_f1 = statistics.mean(overall_test_f1)
avg_mi_recall = statistics.mean(overall_mi_recall)
avg_ma_recall = statistics.mean(overall_ma_recall)


avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, Val Acc: {:.4f} +- {:.4f}, Val F1: {:.4f} +- {:.4f}, Mi recall {:.4f}+-{:.4f}, Ma recall {:.4f}+-{:.4f}'
avg_log = avg_log.format(avg_acc, acc_CI, avg_bacc, bacc_CI, avg_f1, f1_CI, avg_val_acc, val_acc_CI, avg_val_f1, val_f1_CI, avg_mi_recall, mi_recall_CI, avg_ma_recall, ma_recall_CI)
log = "{}".format(avg_log)
print(log)

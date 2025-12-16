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
from sklearn.metrics import balanced_accuracy_score, f1_score,classification_report,roc_auc_score, accuracy_score

from args import parse_args
from src import solid,loss_fn
from src.utils import VNG_utils
from src.models import gnn,sage,gcn,gat,edge_learner,teacher,diffusion,mlp
from src.neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist
from src.denoise import unet
import src.utils.graphbuilder
warnings.filterwarnings("ignore")

def train_gnn_classifier():
    classifier.train()
    classifier_optimizer.zero_grad()
    output = classifier(data.x, edge_index, None)
    classifier_criterion.compute(output[data_train_mask], data.y[data_train_mask]).backward()
    with torch.no_grad():
        classifier.eval()
        output = classifier(data.x, edge_index, None)
        val_loss= F.cross_entropy(output[data_val_mask], data.y[data_val_mask])

    classifier_optimizer.step()
    scheduler.step(val_loss)

@torch.no_grad()
def test_gnn_classifier():
    classifier.eval()
    logits = classifier(data.x, edge_index[:,train_edge_mask], None,)
    accs, baccs, f1s = [], [], []

    for i, mask in enumerate([data_train_mask, data_val_mask, data_test_mask]):
        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        accs.append(acc)
        baccs.append(bacc)
        f1s.append(f1)
        measure_result = classification_report(y_true, y_pred,digits=4, zero_division=np.nan)
    return accs, baccs, f1s, measure_result


args = parse_args()
print(args)
loss_type = "re" #re ce cb focal
factor_focal = 2.0
factor_cb = 0.9999
max_n = 500

device = args.device
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
dataset = VNG_utils.get_dataset(args.dataset,path,split_type="full")
n_feat=dataset.num_features
data = dataset[0].to(device)
n_cls = data.y.max().item() + 1
ori_edge_index = data.edge_index
data = train_test_split_edges(data)

repeatition = 5
avg_test_acc, avg_val_acc, avg_val_f1, avg_test_bacc, avg_test_f1 = [], [], [], [], []
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
    data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
    edge_index = copy.deepcopy(ori_edge_index)
    stats = data.y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = VNG_utils.get_idx_info(data.y, n_cls, data_train_mask)
    class_num_list = n_data
    print("num of class in original training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))

    ## Construct a long-tailed graph ##
    # class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = VNG_utils.make_random_data_remove(edge_index, data.y, n_data, n_cls, train_num, data_train_mask.clone())
    # class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = VNG_utils.make_imb_data(edge_index, data.y, n_data, n_cls, args.imb_ratio, data_train_mask.clone())
    class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = src.utils.graphbuilder.make_longtailed_data_remove(edge_index, data.y, n_data, n_cls, args.imb_ratio, data_train_mask.clone(), max_n)
    print("num of class in LT-training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
    minority_mask = class_num_list < (sum(class_num_list)/n_cls)
    minority_class = [i for i in range(n_cls) if minority_mask[i]]
    print("minority classes {}".format(minority_class))
    class_count = torch.bincount(data.y[data_train_mask].view(-1), minlength=n_cls).to(device,torch.float)
    # if reweight:
    #     # calculate weights
    #     total_count = class_count.sum()
    #     weights = total_count / (n_cls * class_count)  # reverse weight
    #     weights[torch.isinf(weights)] = 0  # avoid inf
    #     weights = weights / torch.sum(weights)
    # else:
    #     weights = None
    # print(weights)
    classifier = gnn.GNN_classifier(args.net, n_feat, args.n_hid, n_cls, args.n_layers, dropout=0.5)
    classifier = classifier.to(device)
    if loss_type == "re":
        classifier_criterion = loss_fn.IMB_LOSS("re-weight",n_cls,class_count.detach().cpu().numpy(),device=device)
    elif loss_type == "ce":
        classifier_criterion = loss_fn.IMB_LOSS("ce",n_cls,class_count.detach().cpu().numpy(),device=device)
    elif loss_type == "cb":
        classifier_criterion = loss_fn.IMB_LOSS("cb-softmax",n_cls,class_count.detach().cpu().numpy(),factor_cb,device=device)
        # classifier_criterion = criterion.compute
    elif loss_type == "focal":
        classifier_criterion = loss_fn.IMB_LOSS("focal",n_cls,class_count.detach().cpu().numpy(),factor_focal,device=device)
        # classifier_criterion = criterion.compute
    else:
        raise Exception("No Implentation Loss")

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
            accs, bacc, f1s, measure_result = test_gnn_classifier()
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
            pbar.set_postfix({
                        'Accuracy': f'{val_acc:.4f}/{best_val_acc:.4f}', 
                        'F1 score': f'{val_f1:.4f}/{best_val_f1:.4f}'
                    })
            pbar.update(1)
    VNG_utils.plot_val_test_acc_f1(val_acc_f1_list, test_acc_f1_list,title='acc_f1_{}'.format(r))
    VNG_utils.plot_val_acc_f1(val_f1_list, val_acc_list,title='val_{}'.format(r))
    VNG_utils.plot_tmp_test_acc_f1(tmp_test_acc_list, tmp_test_f1_list, title='test_{}'.format(r))

    avg_val_acc.append(best_val_acc)
    avg_val_f1.append(best_val_f1)
    avg_test_acc.append(test_acc)
    avg_test_bacc.append(test_bacc)
    avg_test_f1.append(test_f1)
    print(best_measure)
    print('Test Acc: {:.4f}, BAcc: {:.4f}, F1: {:.4f}'.format(test_acc,test_bacc,test_f1))

if repeatition == 1 : exit()
## Calculate statistics ##
acc_CI =  (statistics.stdev(avg_test_acc) / (repeatition ** (1/2)))
bacc_CI =  (statistics.stdev(avg_test_bacc) / (repeatition ** (1/2)))
f1_CI =  (statistics.stdev(avg_test_f1) / (repeatition ** (1/2)))
avg_acc = statistics.mean(avg_test_acc)
avg_val_acc = statistics.mean(avg_val_acc)
avg_val_f1 = statistics.mean(avg_val_f1)
avg_bacc = statistics.mean(avg_test_bacc)
avg_f1 = statistics.mean(avg_test_f1)

avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f}'
avg_log = avg_log.format(avg_acc ,acc_CI ,avg_bacc, bacc_CI, avg_f1, f1_CI, avg_val_acc, avg_val_f1)
log = "{}".format(avg_log)
print(log)

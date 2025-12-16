import copy
import torch
import random
import warnings
import statistics
import numpy as np
import os.path as osp
from torch_geometric.utils import train_test_split_edges,negative_sampling

from solid_trainer import SolidTrainer
from args import parse_args
from src import solid
from src.utils import VNG_utils,tab_dataset_util
from src.TabDiff.tabdiff.modules.main_modules import UniModMLP
from src.TabDiff.tabdiff.modules.main_modules import Model
from src.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from src.models import gnn,sage,gcn,gat,edge_learner,teacher,diffusion,mlp
from src.denoise import unet
import src.utils.graphbuilder
warnings.filterwarnings("ignore")


args = parse_args()
print(args)
reweight = False
timestamp_format = "%Y%m%d_%H%M%S"

device = args.device
root_path = osp.dirname(osp.realpath(__file__))
data_path = osp.join(root_path, 'data', args.dataset)
tab_dataset = tab_dataset_util.load_tab_dataset_info(args.dataset, data_path, split_type='full')
dataset = tab_dataset.graph
data = tab_dataset.graph.to(device)
n_feat = tab_dataset.n_features
print(data)
n_cls = tab_dataset.n_labels
ori_edge_index = data.edge_index
data = train_test_split_edges(data)

repeatition = 1
max_n=500
overall_test_acc, overall_val_acc, overall_val_f1, overall_test_bacc, overall_test_f1 = [], [], [], [], []
overall_mi_recall = []
overall_ma_recall = []
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


    teacher_model = teacher.MLPTeacher(n_feat,n_cls,layers=1,drop=0.4).to(device)

    # definition of diffusion model
    denoise_kwargs = {
        "d_numerical" : tab_dataset.num_numerical_features, 
        "categories" : (tab_dataset.categories+1).tolist(), 
        "num_layers" : args.denoise_layers, 
        "d_token" : args.d_token
    }

    denoise_backbone = UniModMLP(
        **denoise_kwargs
    )
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
        num_timesteps=args.T,
        **diffusion_kwargs,
        device=device,
    )
    num_params = sum(p.numel() for p in tab_diffusion.parameters())
    print("The number of parameters = ", num_params)
    tab_diffusion.to(device)
    tab_diffusion.train()
    train_args = {
        "diff_lr" : 1e-5,
        "tearch_lr" : 1e-3,
        "el_lr" : 1e-3,
        "cl_lr" : 1e-3, 
        "diff_bs" : args.batch_size,
        "r" : repeatition,
        "device" : device
    }

    # definition of edge learner
    edge_decoder = edge_learner.EdgePredicter(n_feat).to(device)

    # definition of gnn classifier
    classifier = gnn.GNN_classifier(args.net, n_feat, args.n_hid, n_cls, args.n_layers, dropout=0.5).to(device)

    trainer = SolidTrainer(tab_dataset, 
                           data_train_mask, 
                           data_val_mask, 
                           tab_diffusion, 
                           teacher_model, 
                           edge_decoder, 
                           classifier, 
                           **train_args)
    
    trainer.train_teacher(epochs=args.epochs)
    trainer.train_edge_learner()
    trainer.train_diffusion(args, skip=True, ckpt_path=f"/home/xvwenduan/GraphSOLID/ckpt/tabdiff/Cora/tabdiff_Cora_20251214_200118_e19_.pth")

    v_information, src_idx = solid.softlabel_based_hard_nodes_tab_sampling(data.x[data_train_mask],
                                                        data.y[data_train_mask],
                                                        n_cls,
                                                        diffusion_model = tab_diffusion,
                                                        teacher = teacher_model,
                                                        temperature = args.temperature,
                                                        guidance_scale = args.guidance,
                                                        hard_factor = args.hard_factor,
                                                        aug_mode = args.aug_mode,
                                                        is_hard_sample = (args.hard_factor == 1.),
                                                        is_beta_sampling = False)
    
    # construct new nodes and edges then augment the graph
    new_node_num = v_information['feat'].shape[0]
    print("{} new nodes".format(new_node_num))
    _, _, _, report_on_gen_samples = trainer.teacher_test(v_information['feat'], v_information['label'])
    print("Performance on generated samples: ", report_on_gen_samples)

    aug_data, edge_index, data_train_mask, train_edge_mask = solid.add_new_nodes(data,
                                                                    v_information['feat'],
                                                                    v_information['label'],
                                                                    edge_decoder,
                                                                    edge_index,
                                                                    data_train_mask,
                                                                    train_edge_mask,
                                                                    device=device)
    data_val_mask = torch.cat([data_val_mask, torch.zeros(new_node_num, dtype=torch.bool, device=device)])
    data_test_mask = torch.cat([data_test_mask, torch.zeros(new_node_num, dtype=torch.bool, device=device)])
    aug_data.train_mask = data_train_mask
    aug_data.val_mask = data_val_mask
    aug_data.test_mask = data_test_mask
    # update trainer data
    trainer.aug_data = aug_data.to(device)
    trainer.data_train_mask_aug = data_train_mask.to(device)
    trainer.data_val_mask_aug = data_val_mask.to(device)
    trainer.edge_index_aug = edge_index.to(device)
    trainer.train_edge_mask_aug = train_edge_mask.to(device)
    trainer.data_test_mask_aug = data_test_mask.to(device)
    trainer.minority_mask = minority_mask
    # train gnn classifier on augmented graph
    best_val_acc, best_val_f1, test_acc, test_bacc, test_f1, best_measure, minority_recall, majority_recall = trainer.train_classifier_vanilla()

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

    VNG_utils.show_samples_dis(aug_data,f"history_data//figure//aug_"+args.dataset,new_node_num,n_cls)


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

import copy
import torch
import random
import warnings
import statistics
import numpy as np
import os.path as osp
from torch_geometric.utils import train_test_split_edges,negative_sampling

from src import solid
from args import parse_args
from src.utils import VNG_utils, graphbuilder
from solid_trainer import SolidTrainer
from src.utils.hetero_dataset_util import GraphDataLoader
from src.TabDiff.tabdiff.modules.main_modules import UniModMLP
from src.TabDiff.tabdiff.modules.main_modules import Model
from src.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from src.models import gnn,sage,edge_learner,teacher,diffusion,mlp,HeteroNN
from src.denoise import unet
warnings.filterwarnings("ignore")


args = parse_args()
print(args)
reweight = False
timestamp_format = "%Y%m%d_%H%M%S"

device = args.device
root_path = osp.dirname(osp.realpath(__file__))
loader = GraphDataLoader()
data_path = osp.join(root_path, 'data', args.dataset, 'data', args.dataset + '.mat')
cnfg_path = osp.join(root_path, 'data', args.dataset, 'meta', args.dataset + '.json')
hetero_ctx = loader.load_from_config(cnfg_path, data_path)
target = hetero_ctx.target_node  # 'review' 或 'user'
data = hetero_ctx.g.to(device)
n_feat = hetero_ctx.n_features
n_cls = hetero_ctx.n_classes
print(data)

repeatition = 1
max_n=500
overall_test_acc, overall_val_acc, overall_val_f1, overall_test_bacc, overall_test_f1 = [], [], [], [], []
overall_mi_recall = []
overall_ma_recall = []
mi_recall = []
ma_recall = []

if args.dataset in ['YelpChi', 'Amazon-Products']:
    data_train_mask, data_val_mask, data_test_mask = data[target].train_mask.clone(), data[target].val_mask.clone(), data[target].test_mask.clone()
    stats = data[target].y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = VNG_utils.get_idx_info(data[target].y, n_cls, data_train_mask)
    class_num_list = n_data
    print("num of class in original training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
    class_num_list, data_train_mask, _, edge_mask_dict = graphbuilder.make_hetero_longtailed_data_remove(data, target, n_data, n_cls, args.imb_ratio, data_train_mask.clone(), max_n)
    # 更新 HeteroData
    hetero_ctx.g[hetero_ctx.target_node].train_mask = data_train_mask
    # 更新边索引 (可选，取决于是否想物理删除边)
    if not args.keep_edge:
        for etype, mask in edge_mask_dict.items():
            hetero_ctx.g[etype].edge_index = hetero_ctx.g[etype].edge_index[:, mask]
    print("num of class in LT-training data: {} -> {}".format(class_num_list,sum(data_train_mask).item()))
    minority_mask = class_num_list < (sum(class_num_list)/n_cls)
    minority_class = [i for i in range(n_cls) if minority_mask[i]]
    print("minority classes {}".format(minority_class))
else:
    raise NotImplementedError("Not implemented for dataset {}".format(args.dataset))

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

    encoder = HeteroNN.HeteroSAGE(hetero_ctx.g.metadata(), args.n_hid, num_layers=args.n_en_layers).to(device)
    teacher_model = teacher.MLPTeacher(args.n_hid,n_cls,layers=1,drop=0.4).to(device)

    # definition of diffusion model
    denoise_nhid = args.n_hid
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

    num_params = sum(p.numel() for p in diffusion_model.parameters())
    print("The number of parameters = ", num_params)
    diffusion_model.to(device)
    train_args = {
        "diff_lr" : args.dif_lr,
        "tearch_lr" : args.teacher_lr,
        "el_lr" : args.de_lr,
        "cl_lr" : args.lr, 
        "diff_bs" : args.batch_size,
        "n_hid" : args.n_hid,
        "r" : repeatition,
        "device" : device
    }

    # definition of edge learner
    node_types = hetero_ctx.g.node_types
    edge_types = hetero_ctx.g.edge_types
    # 自动生成维度字典 (根据 ctx.g 的特征形状)
    node_dim_dict = {
        node_type: args.n_hid # hetero_ctx.g[node_type].x.shape[1] 
        for node_type in node_types
    }
    edge_decoder = edge_learner.HeteroEdgePredicter(
        node_types=node_types,
        edge_types=edge_types,
        node_dim_dict=node_dim_dict,
        n_hid=args.decoder_hid
    ).to(device)

    # definition of hetero-gnn classifier
    classifier = HeteroNN.HeteroGNN_classifier(net=args.net, target_node=target, metadata=hetero_ctx.g.metadata(), nhid=args.feat_dim, nclass=n_cls, nlayer=args.n_layers, dropout=0.5).to(device)
    trainer = SolidTrainer(hetero_ctx, 
                           diffusion_model, 
                           teacher_model, 
                           edge_decoder, 
                           classifier, 
                           encoder,
                           minority_mask,
                           **train_args)
    
    trainer.minority_mask = minority_mask
    cktp_path = {
        "encoder": "/home/xvwenduan/GraphSOLID/ckpt/encoder/YelpChi/encoder_YelpChi_20251231_190604_e109_.pth",
        "decoder": "/home/xvwenduan/GraphSOLID/ckpt/decoder/YelpChi/decoder_YelpChi_20251231_190604_e109_.pth"
    }
    emb_data = trainer.cent_pretrain(args, skip=True, ckpt_path=cktp_path, ckpt_save_epoch=0)
    # cover data with initial embeddings
    trainer.update_data(emb_data)
    trainer.train_teacher(epochs=args.epochs)
    trainer.train_diffusion(args,ckpt_save_epoch=0)

    emb_data = emb_data.to(device)
    v_information, src_idx = solid.softlabel_based_hard_nodes_sampling(emb_data[hetero_ctx.target].x[data_train_mask],
                                                        emb_data[hetero_ctx.target].y[data_train_mask],
                                                        n_cls,
                                                        diffusion_model = diffusion_model,
                                                        teacher = teacher_model,
                                                        args = args,
                                                        device=device)
    
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
    trainer.update_data(aug_data)
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

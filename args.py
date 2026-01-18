import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    ## basic setup
    parser.add_argument('--save_data', action='store_true', help='save augmented graph')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=100, help='seed')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed', 'Amazon-Computers', 'Amazon-Photo', 'YelpChi', 'Amazon-Products'], default='YelpChi', help='dataset name')
    parser.add_argument('--data_path', type=str, default='datasets/', help='data path')
    parser.add_argument('--imb_ratio', type=float, default=20, help='imbalance ratio')
    parser.add_argument('--keep_edge', action='store_true', help='do not remove edge of imb-class')
    parser.add_argument('--is_vanilla', action='store_true', help='vanilla classifier')
    ## classifier setup
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for classifier')
    parser.add_argument('--net', type=str, choices=['HeteroSAGE', 'HeteroGAT', 'RGCN', 'mlp'], default='HeteroSAGE', help='GNN backbone')
    parser.add_argument('--n_layers', type=int, default=1, help='the number of layers')
    parser.add_argument('--feat_dim', type=int, choices=[64,128,256,512], default=64, help='feature dimension')
    parser.add_argument('--epochs', type=int, default=1500, help='epochs')
    parser.add_argument('--loss_type', type=str, choices=['re', 'ce', 'cb', 'focal'], default='ce', help='loss type: re-weighting (re), class-balanced (cb), focal loss (focal), cross-entropy (ce)')
    ## encoder setup
    parser.add_argument('--en_lr', type=float, default=1e-2, help='learning rate for encoder')
    parser.add_argument('--n_hid',type=int,default=32)
    parser.add_argument('--n_en_layers',type=int,default=1)
    parser.add_argument('--w_con_loss',type=float,default=1e-1)
    parser.add_argument('--cent_lr', type=float, default=1e-2, help='learning rate for center loss optimizer')
    ## decoder setup
    parser.add_argument('--de_lr', type=float, default=1e-3, help='learning rate for decoder')
    parser.add_argument('--decoder_hid',type=int,default=32)

    ## teacher setup
    parser.add_argument('--teacher_lr', type=float, default=1e-3, help='learning rate for teacher model')

    ## diffusion setup
    parser.add_argument('--wo_diffu_aug', action='store_true', help=' deprecated diffusion generation.')
    parser.add_argument('--raw_space', action='store_true', help='use raw node feature for diffusion generation, instead lantent embedding.')
    parser.add_argument('--dif_lr', type=float, default=1e-4, help='learning rate for diffusion model')
    parser.add_argument('--padding', type=tuple, default=(0,0,0,0), help='padding for node features to avoid resolution mismatch caused by odd latitudes in unet downsampling')
    parser.add_argument('--T', type=int, default=1000, help='time steps for diffusion process')
    parser.add_argument('--beta_bound', type=tuple, default=(1e-4, 2e-2), help='lower bound and upper bound of beta')
    parser.add_argument('--beta_schedule', type=str, default='lin', help='beta schedule')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of diffusion model')
    parser.add_argument('--guidance_drop_prob', type=float, default=0.1, help='drop probability for class mask')
    parser.add_argument('--temperature', type=float, default=4., help='temperature for solft labels')
    parser.add_argument('--guidance', type=float, default=0.75, help='guidance')
    parser.add_argument('--hard_factor', type=float, default=0.5, help='factor that mixup soft labels and hard labels')
    parser.add_argument('--adjustment_factor', type=float, default=1, help='adjustment class distribution')
    parser.add_argument('--aug_mode', type=str, choices=['mean', 'ratio', 'max'], default='mean', help='augmentation mode')
    parser.add_argument('--n_length', type=int, default=512, help='project length')
    parser.add_argument('--n_channels', type=int, default=128, help='number of channel in the first layer of unet')
    parser.add_argument('--class_embedding_channel', type=int, default=128*8, help='class guidance channel')
    parser.add_argument('--time_embedding_channel', type=int, default=128*8, help='time step channel')
    parser.add_argument('--is_attn', type=tuple, default=(False, True, True), help='attention mechanism')
    parser.add_argument('--ch_mults', type=tuple, default=(1, 2, 4))
    parser.add_argument('--n_blocks', type=int, default=3, help='number of block of up and down sample in unet')
    parser.add_argument('--is_beta_sampling', action='store_false', help='use beta sampling strategy for diffusion training')

    ## tabdiff setup
    parser.add_argument('--dloss_weight', type=float, default=1.0, help='weight for discrate loss weight')
    parser.add_argument('--closs_weight', type=float, default=1.0, help='weight for continuous loss weight')
    parser.add_argument('--denoise_layers', type=int, default=3, help='number of layers for denoising transformer')
    parser.add_argument('--d_token', type=int, default=8, help='token dimension for denoising transformer')
    parser.add_argument('--edm_params', type=dict, default={"precond": True, "sigma_data": 1.0, "net_conditioning": "sigma"}, help='edm parameters')
    parser.add_argument('--sampler_params', type=dict, default={"stochastic_sampler": True, "second_order_correction": True})
    parser.add_argument('--noise_dist_params', type=dict, default={"P_mean": -1.2, "P_std": 1.2})
    parser.add_argument('--noise_schedule_params', type=dict, default={"sigma_min": 0.002, "sigma_max": 80, "rho": 7, "eps_max": 1e-3, "eps_min": 1e-5, "rho_init": 7.0, "rho_offset": 5.0, "k_init":-6.0, "k_offset":1.0})

    ## digress setup
    

    args = parser.parse_args()
    return args

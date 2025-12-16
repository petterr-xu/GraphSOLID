import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..utils import VNG_utils
class GDDPMblock(nn.Module):
    def __init__(self, eps_model: nn.Module, beta:torch.Tensor , n_steps: int, device: torch.device) -> None:
        super(GDDPMblock,self).__init__()
        self.eps_model = eps_model
        self.beta = beta.to(device) # torch.linspace(0.0001, 0.02, n_steps)
        # 根据beta设置alpha=1-beta
        self.alpha = 1. - self.beta
        # 计算alpha的连乘值alpha bar
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0.dim() > 3:
            gather = VNG_utils.gather_image
        elif x0.dim() > 2:
            gather = VNG_utils.gather
        else:
            gather = VNG_utils.gather_vector
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    
    def condition_guide(self, classifier_model:list, classifier_scale, xt: torch.tensor,t,y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = xt.detach().requires_grad_(True)
            # logits = self.classifier(x_in, t)
            classifier,loss_fun = classifier_model
            # classifier.eval()
            logits = classifier(x_in)
            # if len(logits.shape) < 3:
            # 	logits = torch.unsqueeze(logits,dim=1)
            loss = loss_fun(logits,torch.argmax(y,dim=len(y.shape)-1))
            grad = torch.autograd.grad(loss, x_in)[0] * classifier_scale
            # del x_in
        return grad
            # log_probs = F.log_softmax(logits,dim=-1)
            # selected = log_probs[range(len(logits)), y.view(-1)]
            # return torch.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale

    def p_sample(self,classifier_model:list,classifier_scale, guidance_scale, xt: torch.Tensor, t: torch.Tensor,y):
        with torch.no_grad(): # 停止记录梯度，避免爆显存
            # if y.dim() > 1:
            #     nodes_class = torch.argmax(y,dim=y.dim()-1)
            # else:
            nodes_class = y
            # classifier-free 条件生成的噪声预测
            cond_mask = torch.full_like(t,0,dtype=torch.int32)
            eps_theta_cond = self.eps_model(xt, t, nodes_class, cond_mask)
            # classifier-free 无生成的噪声预测
            uncond_mask = torch.full_like(t,1,dtype=torch.int32)
            eps_theta_uncond = self.eps_model(xt, t, nodes_class, uncond_mask)
            eps_theta = eps_theta_uncond + guidance_scale*(eps_theta_cond - eps_theta_uncond)
            # eps_theta = self.eps_model(xt, t, y)
            if xt.dim() > 3:
                gather = VNG_utils.gather_image
            elif xt.dim() > 2:
                gather = VNG_utils.gather
            else:
                gather = VNG_utils.gather_vector
            alpha_bar = gather(self.alpha_bar, t)
            alpha = gather(self.alpha, t)
            eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
            mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
            var = gather(self.sigma2, t)
            if classifier_scale != 0:
                cond_grad = self.condition_guide(classifier_model,classifier_scale,xt,t,y)
                mean_cond = mean+var*cond_grad
            else:
                mean_cond = mean
            eps = torch.randn(xt.shape, device=xt.device)
            # del eps_theta
        return mean_cond + (var ** .5) * eps
    
    def sampling(self,classifier_model,classifier_scale_mode,guidance_scale,x_t:torch.Tensor,y:torch.Tensor, padding,save_frames,device):
        """采样生成
        Args:
            classifier_model (list): 引导用的分类器=[classifier,loss_fun,classifier_loss_beta]
            classifier_scale_mode (_type_): 分类器梯度scale
            guidance_scale : classifier-free guidance 引导强度
            x_t (torch.Tensor): 输入噪声x_t.shape = [batch_size,in_channel,feat_length]
            y (torch.Tensor): onehot编码的类型引导 y.shape = [batch_size,num_classes]
            device (_type_): _description_
        Return:
            x_t: 生成的样本 x_t.shape = [batch_size,in_channel,feat_length]
            frames: 生成x_t的过程记录,每10个时间步记录一帧 frames.shape = [frame_num,batch_size,in_channel,feat_length]
        """
        frames = []
        assert x_t.shape[0] == y.shape[0],"check x_t and y shape"
        t_schedule = torch.arange(0,self.n_steps,1).flip(dims=[0])
        batch_size = x_t.shape[0]
        if classifier_scale_mode == "exp":
            max_scale = 3
            temperature = 100.0
            # 用指数函数调整不同时间步时classifier_scale的大小
            # t较大时（即去噪开始时）classifier_scale比较大，当样本基本成型（即t较小时），classifier_scale也较小（趋近于零）
            classifier_scale = (max_scale)*torch.exp(-(t_schedule)/temperature)
        else:
            classifier_scale = torch.full_like(t_schedule,classifier_scale_mode)
        for t in t_schedule:
            t_batch = torch.full([batch_size],t).to(device,torch.int64)
            x_t = self.p_sample(classifier_model,classifier_scale[t],guidance_scale,x_t,t_batch,y)
            if save_frames and (t + 1) % 10 == 0:
                frames.append(x_t)
        
        # 可能需要考虑去除padding
        left_pad,right_pad,top_pad,bottom_pad = padding
        if right_pad != 0:
            if x_t.dim() > 2:
                x_t = x_t[:,:,left_pad:-right_pad]
            else:
                x_t = x_t[:,left_pad:-right_pad]
        elif left_pad != 0:
            if x_t.dim() > 2:
                x_t = x_t[:,:,left_pad:]
            else:
                x_t = x_t[:,left_pad:]
                
        return x_t,frames

    def loss(self, x0: torch.Tensor, guidance: torch.Tensor, guidance_mask:torch.Tensor, padding:tuple, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0.to(torch.float32))
        xt = self.q_sample(x0, t, eps=noise )
        # print(xt.shape)
        eps_theta = self.eps_model(xt, t, guidance, guidance_mask)
        # print(eps_theta.shape)
        # 可能需要考虑去除padding
        left_pad,right_pad,top_pad,bottom_pad = padding
        if right_pad != 0:
            if noise.dim() > 2:
                return F.mse_loss(noise[:,:,left_pad:-right_pad], eps_theta[:,:,left_pad:-right_pad])
            else:
                return F.mse_loss(noise[:,left_pad:-right_pad], eps_theta[:,left_pad:-right_pad])
        elif left_pad != 0:
            if noise.dim() > 2:
                return F.mse_loss(noise[:,:,left_pad:], eps_theta[:,:,left_pad:])
            else:
                return F.mse_loss(noise[:,left_pad:], eps_theta[:,left_pad:])
        else:
            return F.mse_loss(noise, eps_theta)


# def virtual_nodes_sampling(graph:dgl.DGLGraph,diffusion_model,embedding_size,guidance=7.5, over_sample_rate = None,device="cuda:0"):
#     mlp_judger = classifier.train_mlp_classifier(graph)
#     dis = VNG_utils.node_class_dis(graph,graph.ndata["train_mask"])
#     if over_sample_rate is None:
#         aug_size = torch.max(dis) - dis
#     else:
#         aug_size = dis[dis != torch.max(dis)]*over_sample_rate
#     node_classes = torch.tensor([],dtype=torch.int32)
#     for class_,class_size in enumerate(aug_size):
#         node_classes = torch.cat((node_classes,torch.full([class_size],fill_value=class_)))
    
#     node_classes = F.one_hot(node_classes,7).to(device,dtype=torch.int32)
#     num_samples = node_classes.shape[0]
#     x_t = torch.randn([num_samples,1,embedding_size]).to(device)
#     diffusion_model.eval()
#     diffusion_model.to(device)
#     x0,frame = diffusion_model.sampling([None,None],classifier_scale_mode=0,guidance_scale=guidance
#                                         ,x_t=x_t,y=node_classes,save_frames=True,device=device)
#     virtual_feat = x0.squeeze(1) # [:,0:128]
#     v_information = {"feat":virtual_feat,"label":node_classes}
#     print("generated node feature quility:")
#     classifier.val_mlp_classifier(mlp_judger,virtual_feat,node_classes,device,show_detail=True)
#     return virtual_feat,v_information

# def softlabel_based_hard_nodes_sampling(graph:dgl.DGLGraph,teacher,temperature,diffusion_model,embedding_size,guidance=7.5, hard_factor = 0.5, over_sample_rate = None,device="cuda:0"):
#     confidence_filter_ratio=5
#     dis = VNG_utils.node_class_dis(graph,graph.ndata["train_mask"])
#     num_classes = len(dis)
#     soft_labels = teacher.softmax_with_temperature(graph.ndata['feat'][graph.ndata["train_mask"]],temperature)
#     hard_labels = graph.ndata['label'][graph.ndata["train_mask"]].argmax(1)
#     # confidence_dis = VNG_utils.confidence_dis(soft_labels,hard_labels,num_classes)
#     # if minority_class == None:
#     #     aug_mask = torch.ones([num_classes],dtype=torch.int32,device=device)
#     if over_sample_rate is None:
#         aug_size = torch.max(dis) - dis
#     else:
#         dis[dis == torch.max(dis)] = 0
#         aug_size = torch.tensor(dis*over_sample_rate,dtype=torch.int32)
#     # aug_size = aug_size * confidence_filter_ratio
#     # assert confidence_filter_ratio > 1, "parameter confidence_filter_ratio must larger than 1"
#     x0 = torch.tensor([],device=device)
#     labels = torch.tensor([],device=device)
#     for class_,class_aug_size in enumerate(aug_size):
#         if class_aug_size == 0 : continue
#         node_classes = torch.full([int(class_aug_size)],fill_value=class_,device=device)
#         # filter hard sample
#         class_mask = (hard_labels == class_)
#         nodes_confidence = 1-torch.index_select(soft_labels[class_mask], dim = 1, index=torch.tensor(class_,device=device)).view(-1)
#         _,indices = torch.topk(nodes_confidence,sum(class_mask)//2)
#         hard_samples = soft_labels[class_mask][indices]
#         mean_confidence = torch.mean(hard_samples,dim=0)
#         variance_confidence = mean_confidence / 100
#         print("mean ", mean_confidence)
#         print("var ", variance_confidence)
#         # assume that confidence follow Beta districution
#         alpha = mean_confidence * (mean_confidence * (1 - mean_confidence) / variance_confidence - 1)
#         beta_param = (1 - mean_confidence) * (mean_confidence * (1 - mean_confidence) / variance_confidence - 1)
#         print("alpha ", alpha)
#         print("beta ", beta_param)
#         confidence_sample = torch.distributions.Beta(alpha, beta_param).sample((class_aug_size,)).to(device)
#         confidence_guidance = confidence_sample / torch.sum(confidence_sample,dim=1,keepdim=True)
#         overall_guidance = confidence_guidance + F.one_hot(node_classes,num_classes) * hard_factor
#         x_t = torch.randn([class_aug_size,1,embedding_size]).to(device)
#         x0_v,_ = diffusion_model.sampling([None,None],classifier_scale_mode=0,guidance_scale=guidance
#                                             ,x_t=x_t,y=overall_guidance,save_frames=False,device=device)
#         if x0.numel() == 0:
#             x0 = x0_v
#             labels = F.one_hot(node_classes,num_classes)
#         else:
#             x0 = torch.concat([x0,x0_v],dim=0)
#             labels = torch.concat([labels,F.one_hot(node_classes,num_classes)],dim=0)
            
#     virtual_feat = x0.squeeze(1)
#     v_information = {"feat":torch.detach(virtual_feat),"label":torch.detach(labels.to(graph.ndata['label'].dtype))}
#     print("generated node feature quility:")
#     logits = teacher(virtual_feat)
#     preds = logits.argmax(1)
#     print(preds)
#     classifier.val_mlp_classifier(teacher,virtual_feat,labels.to(graph.ndata['label'].dtype),device,show_detail=True)
#     torch.cuda.empty_cache()
#     return virtual_feat,v_information

# def confidence_based_hard_nodes_sampling(graph:dgl.DGLGraph,diffusion_model,embedding_size,confidence_filter_ratio=5,guidance=7.5, over_sample_rate = None,device="cuda:0"):
#     mlp_judger = classifier.train_mlp_classifier(graph)
#     dis = VNG_utils.node_class_dis(graph,graph.ndata["train_mask"])
#     num_classes = len(dis)
#     if over_sample_rate is None:
#         aug_size = torch.max(dis) - dis
#     else:
#         dis[dis == torch.max(dis)] = 0
#         aug_size = torch.tensor(dis*over_sample_rate,dtype=torch.int32)
#     aug_size = aug_size * confidence_filter_ratio
#     assert confidence_filter_ratio > 1, "parameter confidence_filter_ratio must larger than 1"
#     node_classes = torch.tensor([],dtype=torch.int32)
#     for class_,class_size in enumerate(aug_size):
#         node_classes = torch.cat((node_classes,torch.full([int(class_size)],fill_value=class_)))
    
#     node_classes = F.one_hot(node_classes,7).to(device,dtype=torch.int32)
#     num_samples = node_classes.shape[0]
#     x_t = torch.randn([num_samples,1,embedding_size]).to(device)
#     diffusion_model.eval()
#     diffusion_model.to(device)
#     if num_samples < 1000:
#         x0,_ = diffusion_model.sampling([None,None],classifier_scale_mode=0,guidance_scale=guidance
#                                             ,x_t=x_t,y=node_classes,save_frames=False,device=device)
#     else:
#         x0 = torch.empty([],device=device)
#         generate_batch_size = 500
#         start_idx = 0
#         end_idx = 0
#         for i in range(num_samples//generate_batch_size + 1):
#             start_idx = i * generate_batch_size
#             if start_idx > num_samples : break
#             end_idx = num_samples if start_idx + generate_batch_size > num_samples else start_idx + generate_batch_size
#             x0_batch,_ = diffusion_model.sampling([None,None],classifier_scale_mode=0,guidance_scale=guidance
#                                             ,x_t=x_t[start_idx:end_idx],y=node_classes[start_idx:end_idx],save_frames=False,device=device)
#             if(i == 0):
#                 x0 = x0_batch
#             else:
#                 x0 = torch.concat([x0,x0_batch],dim=0)
            
#     virtual_feat = x0.squeeze(1) # [:,0:128]
#     logits = mlp_judger(virtual_feat)
#     start = 0
#     quality_feat = torch.tensor([],dtype=torch.float32,device=device)
#     quality_label = torch.tensor([],dtype=torch.int32,device=device)
#     for class_,class_size in enumerate(aug_size):
#         if(class_size != 0):
#             nodes_confidence = logits[start:start+class_size]
#             nodes_confidence = (nodes_confidence - torch.min(nodes_confidence,dim=1,keepdim=True).values) / torch.sum(nodes_confidence - torch.min(nodes_confidence,dim=1,keepdim=True).values,dim=1,keepdim=True)
#             nodes_confidence = 1-torch.index_select(nodes_confidence, 1, torch.tensor(class_,device=nodes_confidence.device)).view(-1)
#             # _,quality_indices = torch.topk(nodes_confidence,int(class_size / confidence_filter_ratio))
#             _,rank_indices = torch.sort(nodes_confidence,descending = True)
#             quality_start = 0#int(len(rank_indices)*4/5)
#             quality_indices = rank_indices[quality_start:quality_start+int(class_size / confidence_filter_ratio)]
#             quality_feat = torch.cat((quality_feat,virtual_feat[start:start+class_size][quality_indices]))
#             quality_label = torch.cat((quality_label,torch.full([int(class_size / confidence_filter_ratio)],fill_value=class_,device=device)))
#             start += class_size
#     v_information = {"feat":torch.detach(quality_feat),"label":torch.detach(F.one_hot(quality_label,num_classes).to(torch.int32))}
#     print("generated node feature quility:")
#     classifier.val_mlp_classifier(mlp_judger,torch.detach(v_information['feat']),torch.detach(v_information['label']),device,show_detail=True)
#     torch.cuda.empty_cache()
#     return quality_feat,v_information


# def train(graph_dataset_config:GraphDatasetConfig, diffusion_config:DiffusionConfig, denoise_config:UnetConfig, device="cuda:0"):
#     # 初始化模型训练的各个参数
#     if graph_dataset_config.file_path is None:
#         graph = graph_dataset_config.graph_dataset
#     else:
#         dgl_graph_list,_ = dgl.load_graphs(graph_dataset_config.file_path)
#         graph = dgl_graph_list[0]
    
#     padding = graph_dataset_config.padding # cora padding = (0,7,0,0)
#     dgl_feature_field_name = graph_dataset_config.dgl_feature_field_name

#     T = diffusion_config.T
#     beta_edge = diffusion_config.beta
#     guidance_drop_prob = diffusion_config.guidance_drop_prob
#     learning_rate = diffusion_config.learning_rate
#     epochs = diffusion_config.epochs
#     batch_size = diffusion_config.batch_size
#     save_cp = diffusion_config.save_cp

#     file_path = r"CGDM-Im\\history_data\\diffusion_model_checkpoint\\".replace("\\",os.sep)
#     # VAE用于将初始输入压缩到隐空间内(optional)
#     if diffusion_config.is_confidence_guide:
#         teacher_model = diffusion_config.teacher_model.to(device)
#     eps_model = unet.UNet(denoise_config)
#     # eps_model = unet_vector.UNet(denoise_config)
#     if diffusion_config.beta_schedule == "lin":
#         beta = torch.linspace(beta_edge[0], beta_edge[1], T)
#     elif diffusion_config.beta_schedule == "exp":
#         beta_exp = beta_edge[0] * (beta_edge[1] / beta_edge[0]) ** (np.arange(T) / T)
#         beta = torch.tensor(beta_exp,dtype=torch.float32)
#     elif diffusion_config.beta_schedule == "quad":
#         beta_quad = beta_edge[0] + (np.arange(T) / T) ** 2 * (beta_edge[1] - beta_edge[0])
#         beta = torch.tensor(beta_quad,dtype=torch.float32)
#     else:
#         print("NO SUCH BETA SCHEDULE:"+diffusion_config.beta_schedule)
#         raise Exception

#     model = GDDPMblock(eps_model,beta,n_steps=T,device=device)
#     model = model.to(device)

#     train_feat_data = graph.ndata[dgl_feature_field_name][graph.ndata["train_mask"]]
#     train_label_data = graph.ndata["label"][graph.ndata["train_mask"]]

#     eval_feat = graph.ndata[dgl_feature_field_name][graph.ndata["val_mask"]]
#     eval_label = graph.ndata["label"][graph.ndata["val_mask"]]
#     # eval_data = [eval_feat,eval_label]
#     # 对数据集进行SMOTE处理(optional)
#     if diffusion_config.SMOTE_aug :
#         sampler = SMOTE(k_neighbors=diffusion_config.SMOTE_kneighbors)
#         y_onehot = train_label_data
#         y = torch.argmax(y_onehot,y_onehot.dim()-1).to("cpu").numpy()
#         train_data_res,train_label_res = sampler.fit_resample(train_feat_data.to("cpu").numpy(),y)
#         train_feat_data = torch.tensor(train_data_res)
#         train_label_data = F.one_hot(torch.tensor(train_label_res),graph_dataset_config.num_classes)

#     class_mask = torch.zeros((train_feat_data.shape[0]),dtype=torch.bool,device=device)
#     train_dataset = TensorDataset(train_feat_data,train_label_data,class_mask)
#     data_loader = DataLoader(train_dataset, batch_size, shuffle=True)

#     optimizer = optim.Adam(model.eps_model.parameters(), lr=learning_rate)
#     train_loss = []
#     eval_loss = []
#     best_val_loss = float("inf")
#     patience_count = 0
#     patience_beta = 2e-4
#     for epoch in range(epochs):
#         loss_value = 0.0
#         class_dis = VNG_utils.node_class_dis(graph,mask=graph.ndata['train_mask'],num_classes=graph_dataset_config.num_classes)
#         class_dis = class_dis / sum(class_dis)
#         class_mask = VNG_utils.dis_based_class_mask(train_label_data,class_dis,graph_dataset_config.num_classes,train_label_data.shape[0],diffusion_config.guidance_drop_prob,adjustment_factor=0.5,device=device)
#         # print("{} guidance, hard label of guidance is:".format(sum(class_mask)))
#         # print(train_label_data[class_mask].argmax(1))
#         train_dataset.tensors = (train_feat_data, train_label_data, class_mask)
#         data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
#         for inputs,targets,c_mask in data_loader:
#             # 对节点特征进行padding,以避免unet下采样中出现奇数纬度导致分辨率不匹配
#             inputs = F.pad(inputs,pad=padding,mode="constant",value=0).to(device)
#             if diffusion_config.is_confidence_guide:
#                 soft_labels = teacher_model.softmax_with_temperature(inputs,diffusion_config.temperature)
#                 targets = soft_labels + targets / 2

#             inputs = torch.unsqueeze(inputs,dim=1)
#             # print(inputs.shape)
#             optimizer.zero_grad()
#             targets = targets.to(device)
#             # 按照一定的概率将guidance置空，由此只用一个backbone训练出适用于两种情况（有无条件）的模型
#             # c_mask = (torch.rand(targets.shape[0]) < guidance_drop_prob).to(device,torch.int32)
#             loss = model.loss(inputs,targets,c_mask.to(torch.int32),padding)
#             loss.backward()
#             optimizer.step()
#             loss_value += loss.item()
#         train_loss.append(loss_value)
#         if (epoch) % 5 == 0:
#             class_mask = VNG_utils.dis_based_class_mask(eval_label,class_dis,graph_dataset_config.num_classes,eval_label.shape[0],diffusion_config.guidance_drop_prob,adjustment_factor=0.5,device=device)
#             eval_data = [eval_feat,eval_label,class_mask.to(torch.int32)]
#             if diffusion_config.is_confidence_guide:
#                 eval_loss_value = eval_model(model,padding,eval_data,teacher_model,diffusion_config.temperature)
#             else:
#                 eval_loss_value = eval_model(model,padding,eval_data)
#             if eval_loss_value  < (best_val_loss-patience_beta):
#                 best_val_loss = eval_loss_value
#                 patience_count = 0
#             else:
#                 patience_count += 1
#             if patience_count >= diffusion_config.patience:
#                 break
#             print('Epoch [{}], Train Loss {:.4f}, Eval loss {:.4f}(best {:.4f}) patience{} '.format(
#                 epoch+1,loss_value/len(data_loader.dataset),eval_loss_value,best_val_loss,patience_count)+ datetime.datetime.now().strftime('%H:%M:%S'))
#             # eval_loss.append(eval_loss_value)
#         if save_cp and ((epoch + 1) % save_cp == 0):
#                 torch.save(model, file_path + 'DiffusionModel'+graph_dataset_config.graph_id+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'e'+str(epoch+1)+'.pth')
#         model.train()
#         assert model.training , 'grad disable! stop train.'
#     if save_cp:
#         VNG_utils.save(model, file_path + 'DiffusionModel'+graph_dataset_config.graph_id+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'.pth')

#     return train_loss,eval_loss,[model,eps_model]

# # def train_autoregression(graph_dataset_config:GraphDatasetConfig, diffusion_config:DiffusionConfig, denoise_config:UnetConfig, device="cuda:0"):
# #     # 初始化模型训练的各个参数
# #     if graph_dataset_config.file_path is None:
# #         graph = graph_dataset_config.graph_dataset
# #     else:
# #         dgl_graph_list,_ = dgl.load_graphs(graph_dataset_config.file_path)
# #         graph = dgl_graph_list[0]
# #     padding = graph_dataset_config.padding # cora padding = (0,7,0,0)
# #     dgl_feature_field_name = graph_dataset_config.dgl_feature_field_name

# #     T = diffusion_config.T
# #     beta_edge = diffusion_config.beta
# #     guidance_drop_prob = diffusion_config.guidance_drop_prob
# #     learning_rate = diffusion_config.learning_rate
# #     epochs = diffusion_config.epochs
# #     batch_size = diffusion_config.batch_size
# #     save_cp = diffusion_config.save_cp

# #     file_path = r"CGDM-Im\\history_data\\diffusion_model_checkpoint\\".replace("\\",os.sep)
# #     # VAE用于将初始输入压缩到隐空间内(optional)
# #     if diffusion_config.is_latent_diffusion:
# #         lantent_encoder:ae.VAE = diffusion_config.latent_encoder.to(device)
# #     eps_model = unet.UNet(denoise_config)
# #     # eps_model = unet_vector.UNet(denoise_config)
# #     if diffusion_config.beta_schedule == "lin":
# #         beta = torch.linspace(beta_edge[0], beta_edge[1], T)
# #     elif diffusion_config.beta_schedule == "exp":
# #         beta_exp = beta_edge[0] * (beta_edge[1] / beta_edge[0]) ** (np.arange(T) / T)
# #         beta = torch.tensor(beta_exp,dtype=torch.float32)
# #     elif diffusion_config.beta_schedule == "quad":
# #         beta_quad = beta_edge[0] + (np.arange(T) / T) ** 2 * (beta_edge[1] - beta_edge[0])
# #         beta = torch.tensor(beta_quad,dtype=torch.float32)
# #     else:
# #         print("NO SUCH BETA SCHEDULE:"+diffusion_config.beta_schedule)
# #         raise Exception

# #     model = GDDPMblock(eps_model,beta,n_steps=T,device=device)
# #     model = model.to(device)

# #     train_feat_data = graph.ndata[dgl_feature_field_name][graph.ndata["train_mask"]]
# #     train_label_data = graph.ndata["label"][graph.ndata["train_mask"]]

# #     eval_feat = graph.ndata[dgl_feature_field_name][graph.ndata["val_mask"]]
# #     eval_label = graph.ndata["label"][graph.ndata["val_mask"]]
# #     eval_data = [eval_feat,eval_label]
# #     # 对数据集进行SMOTE处理(optional)
# #     if diffusion_config.SMOTE_aug :
# #         sampler = SMOTE(k_neighbors=diffusion_config.SMOTE_kneighbors)
# #         y_onehot = train_label_data
# #         y = torch.argmax(y_onehot,y_onehot.dim()-1).to("cpu").numpy()
# #         train_data_res,train_label_res = sampler.fit_resample(train_feat_data.to("cpu").numpy(),y)
# #         train_feat_data = torch.tensor(train_data_res)
# #         train_label_data = F.one_hot(torch.tensor(train_label_res),graph_dataset_config.num_classes)

# #     train_dataset = TensorDataset(train_feat_data,train_label_data)
# #     data_loader = DataLoader(train_dataset, batch_size, shuffle=True)

# #     optimizer = optim.Adam(model.eps_model.parameters(), lr=learning_rate)
# #     train_loss = []
# #     eval_loss = []
# #     best_val_loss = float("inf")
# #     patience_count = 0
# #     patience_beta = 2e-4
# #     areg = 2
# #     for r in range(areg):
# #         for epoch in range(epochs):
# #             loss_value = 0.0
# #             for inputs,targets in data_loader:
# #                 # 对节点特征进行padding,以避免unet下采样中出现奇数纬度导致分辨率不匹配
# #                 inputs = F.pad(inputs,pad=padding,mode="constant",value=0).to(device)
# #                 if diffusion_config.is_latent_diffusion:
# #                     # 调用VAE的编码器计算隐空间嵌入
# #                     inputs = lantent_encoder.encode(inputs)
# #                 inputs = torch.unsqueeze(inputs,dim=1)
# #                 # print(inputs.shape)
# #                 optimizer.zero_grad()
# #                 targets = targets.to(device)
# #                 # 按照一定的概率将guidance置空，由此只用一个backbone训练出适用于两种情况（有无条件）的模型
# #                 class_mask = (torch.rand(targets.shape[0]) < guidance_drop_prob).to(device,torch.int32)
# #                 loss = model.loss(inputs,targets,class_mask,padding)
# #                 loss.backward()
# #                 optimizer.step()
# #                 loss_value += loss.item()
# #             train_loss.append(loss_value)
# #             if (epoch) % 5 == 0:
# #                 if diffusion_config.is_latent_diffusion:
# #                     eval_loss_value = eval_model(model,padding,eval_data,lantent_encoder)
# #                 else:
# #                     eval_loss_value = eval_model(model,padding,eval_data)
# #                 if eval_loss_value  < (best_val_loss-patience_beta):
# #                     best_val_loss = eval_loss_value
# #                     patience_count = 0
# #                 else:
# #                     patience_count += 1
# #                 if patience_count >= diffusion_config.patience:
# #                     break
# #                 print('Epoch [{}], Train Loss {:.4f}, Eval loss {:.4f}(best {:.4f}) patience{} '.format(
# #                     epoch+1,loss_value/len(data_loader.dataset),eval_loss_value,best_val_loss,patience_count)+ datetime.datetime.now().strftime('%H:%M:%S'))
# #                 # eval_loss.append(eval_loss_value)
# #             if save_cp and ((epoch + 1) % save_cp == 0):
# #                     torch.save(model, file_path + 'DiffusionModel'+graph_dataset_config.graph_id+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'e'+str(epoch+1)+'.pth')
# #             model.train()
# #             assert model.training , 'grad disable! stop training.'
# #         if r != areg-1: # autoregression training
# #             patience_count = 0
# #             best_val_loss = float("inf")
# #             _,v_information = virtual_nodes_sampling(graph,model,denoise_config.feature_length,guidance=6,over_sample_rate=1,device=device)
# #             train_feat_data_a = torch.concat([v_information['feat'],train_feat_data],dim=0)
# #             train_label_data_a = torch.concat([v_information['label'],train_label_data],dim=0)
# #             train_dataset = TensorDataset(train_feat_data_a,train_label_data_a)
# #             data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
# #     if save_cp:
# #         try:
# #             torch.save(model, file_path + 'DiffusionModel'+graph_dataset_config.graph_id+str(datetime.datetime.now().strftime("%m-%d_%H_%M"))+'.pth')
# #         except FileNotFoundError as fnf:
# #             print("MODEL NOT SAVED!")
# #     return train_loss,eval_loss,[model,eps_model]

# def eval_model(model:nn.Module,padding,eval_data,teacher:nn.Module=None,temperature=None,device="cuda:0"):
#     model.eval()
#     with torch.no_grad():
#         feat_dataset = TensorDataset(eval_data[0],eval_data[1],eval_data[2])
#         test_data_loader = DataLoader(feat_dataset, batch_size=32, shuffle=True)
#         data_size = len(test_data_loader.dataset)
#         loss_value = 0.0
#         for inputs,targets,c_mask in test_data_loader:
#             # 对节点特征进行padding以避免unet下采样中出现奇数纬度导致分辨率不匹配
#             inputs = F.pad(inputs,pad=padding,mode="constant",value=0).to(device)
#             if teacher != None:
#                 soft_labels = teacher.softmax_with_temperature(inputs,temperature)
#                 targets = soft_labels + targets / 2
#             inputs = torch.unsqueeze(inputs,dim=1)
#             targets = targets.to(device)
#             # class_mask = (torch.rand(targets.shape[0]) < 0.15).to(device,torch.int32)
#             # c_mask = torch.ones_like(c_mask,device=device)
#             loss = model.loss(inputs,targets,c_mask,padding)
#             loss_value += loss.item()
#     return loss_value / data_size


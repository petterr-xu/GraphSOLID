import torch
from torch import nn
from tqdm import tqdm
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

    def remove_padding(self, tensor: torch.Tensor, padding: tuple) -> torch.Tensor:
        """
        去除tensor的padding。
        Args:
            tensor: 输入tensor
            padding: (left_pad, right_pad, top_pad, bottom_pad)
        Returns:
            去除padding后的tensor
        """
        left_pad, right_pad, top_pad, bottom_pad = padding
        if tensor.dim() == 2:
            # 假设只处理最后一个维度
            start = left_pad
            end = tensor.shape[1] - right_pad if right_pad > 0 else None
            return tensor[:, start:end]
        elif tensor.dim() == 3:
            # 假设处理最后一个维度
            start = left_pad
            end = tensor.shape[2] - right_pad if right_pad > 0 else None
            return tensor[:, :, start:end]
        else:
            # 通用处理最后一个维度
            slices = [slice(None)] * tensor.dim()
            slices[-1] = slice(left_pad, -right_pad if right_pad > 0 else None)
            return tensor[tuple(slices)]

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
    
    def classifier_guide(self, classifier_model:list, classifier_scale, xt: torch.tensor,t,y=None):
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

    def p_sample(self, guidance_scale, xt: torch.Tensor, t: torch.Tensor,y):
        with torch.no_grad(): # 停止记录梯度，避免爆显存
            # if y.dim() > 1:
            #     nodes_class = torch.argmax(y,dim=y.dim()-1)
            # else:
            nodes_class = y
            # classifier-free 条件生成的噪声预测
            cond_mask = torch.full_like(t,0,dtype=torch.int32)[:,None]
            eps_theta_cond = self.eps_model(xt, t, nodes_class, cond_mask)
            # classifier-free 无生成的噪声预测
            uncond_mask = torch.full_like(t,1,dtype=torch.int32)[:,None]
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
            mean_cond = mean
            eps = torch.randn(xt.shape, device=xt.device)
            # del eps_theta
            nonzero_mask = (t != 0).float().view(-1, *([1] * (xt.dim() - 1)))
        return mean + nonzero_mask * (var ** 0.5) * eps

    def sampling(self,guidance_scale,x_t:torch.Tensor,y:torch.Tensor, padding=(0,0,0,0),save_frames=False, show_pbar=False,device="cuda:0"):
        """采样生成
        Args:
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
        
        pbar = tqdm(t_schedule, desc="Diffusion Sampling", disable=not show_pbar, leave=False)

        for t in pbar:
            t_batch = torch.full([batch_size], t).to(device, torch.int64)
            x_t = self.p_sample(guidance_scale, x_t, t_batch, y)
            if save_frames and (t + 1) % 10 == 0:
                frames.append(x_t)
            pbar.set_postfix({'t':f"{t.item()}/{self.n_steps}"})
        
        # 可能需要考虑去除padding
        x_t = self.remove_padding(x_t, padding)
                
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
        noise_unpadded = self.remove_padding(noise, padding)
        eps_theta_unpadded = self.remove_padding(eps_theta, padding)
        return F.mse_loss(noise_unpadded, eps_theta_unpadded)

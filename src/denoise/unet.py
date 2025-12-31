import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from ..utils.config import UnetConfig

from ..models.attention import MultiHeadAttention as AttentionBlock

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MultiLinearLayer1d(nn.Module):
    def __init__(self, in_features, out_channels, out_features):
        super(MultiLinearLayer1d, self).__init__()
        self.out_length = out_features
        # 初始化n个线性层，每个线性层的输出特征数为 out_channels ，输出长度为 out_features
        self.linear_layers = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(out_channels)])
        self.act = nn.GELU()

    def forward(self, x:torch.Tensor):
        # 应用每个线性层并获取输出
        outputs = [layer(x) for layer in self.linear_layers]
        # 将输出张量合并为一个形状为[batch_size, out_channels, out_features]的张量
        combined = torch.stack(outputs, dim=1)
        # # 重新排列维度以得到形状为[batch_size, out_channels, out_features]的张量
        outputs = combined.view(combined.size(0), -1, self.out_length)
        return self.act(outputs)

class TimeEmbedding(nn.Module):
    '''
    TimeEmbedding模块将把整型t,以Transformer函数式位置编码的方式,映射成向量,
    其shape为(batch_size, time_channel)
    '''
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)
    def forward(self, t: torch.Tensor):
        """
        Params:
            t: 维度(batch_size),整型时刻t
        """
        # 以下转换方法和Transformer的位置编码一致
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        # 输出维度(batch_size, time_channels)
        return emb
    
class ClassEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(ClassEmbedding, self).__init__()
        """
        计算得到class embedding,作为条件生成的引导信息；包含两层全连接层。
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.encode(x.to(torch.float32))

class ResidualBlock(nn.Module):
    """
    每一个Residual block都有两层CNN做特征提取
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 16, dropout: float = 0.1):
        """
        Params:
            in_channels:  输入向量的channel数量
            out_channels: 经过residual block后输出向量的channel数量
            time_channels:time_embedding的向量维度,例如t原来是个整型,值为1,表示时刻1,
                           现在要将其变成维度为(1, time_channels)的向量
            n_groups:     Group Norm中的超参
            dropout:      dropout rate
        """
        super().__init__()
        
        # 第一层卷积 = Group Norm + CNN
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # 第二层卷积 = Group Norm + CNN
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # 当in_c = out_c时,残差连接直接将输入输出相加；
        # 当in_c != out_c时,对输入数据做一次卷积,将其通道数变成和out_c一致,再和输出相加
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # t向量的维度time_channels可能不等于out_c,所以要对起做一次线性转换
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        # self.class_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Params:
            x: 输入数据xt,尺寸大小为(batch_size, in_channels, length)
            t: 输入数据t,尺寸大小为(batch_size, time_c)
        """
        # 1.输入数据先过一层卷积
        h = self.conv1(self.act1(self.norm1(x)))
        # print(h.shape)
        # 2. 对time_embedding向量,通过线性层使time_c变为out_c,再和输入数据的特征图相加
        h += self.time_emb(self.time_act(t))[:, :, None]
        # 3、过第二层卷积
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # 4、返回残差连接后的结果
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    """
    Down block,即Encoder中每一层的核心处理逻辑
    DownBlock = ResidualBlock + AttentionBlock
    """

    def __init__(self, in_channels: int, out_channels: int, length:int, time_channels: int, class_dims:int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.guidance_emb = ClassEmbedding(class_dims,in_channels)
        if has_attn:
            self.attn = AttentionBlock(length,num_heads=2)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_ : torch.Tensor):
        class_embedding = self.guidance_emb(class_)
        x += class_embedding[:,:,None]
        x = self.res(x, t)
        x = self.attn(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, length:int, time_channels: int, class_dims:int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        # self.guidance_emb = ClassEmbedding(class_dims,in_channels)
        if has_attn:
            self.attn = AttentionBlock(length,num_heads=2)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # class_embedding = self.guidance_emb(class_)
        # x += class_embedding[:,:,None,None]
        x = self.res(x, t)
        x = self.attn(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, length:int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(length,num_heads=1)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(n_channels, n_channels,4,2,1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_embedding: torch.Tensor = None):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_channels,3,2,1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,feature_length,n_length,n_channels,ch_mults,is_attn,n_blocks,class_channels,time_channels,num_class):
        """
        * `n_channels` is number of channels in the initial feature map that we transform the vector into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        self.num_class: int = num_class
        super(UNet,self).__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        # self.vector_proj = nn.Conv1d(vector_channels, n_channels, kernel_size=3, padding=1)
        self.vector_proj = nn.Sequential(
            nn.Linear(feature_length,n_length),
            nn.Conv1d(in_channels=1,out_channels=n_channels,kernel_size=3,padding=1)
        )
        # self.vector_proj = MultiLinearLayer1d(feature_length,n_channels,n_length)
        # self.vector_proj = nn.Sequential(
        #     nn.Linear(feature_length,n_length),
        #     nn.Conv1d(in_channels=1,out_channels=n_channels,kernel_size=3,padding=1)
        # )
        # Time embedding 层 将时间步t输出为channel为 `n_channels * 4` 的时间嵌入
        self.time_emb = TimeEmbedding(time_channels)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        emb_length = n_length
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, int(emb_length), time_channels, self.num_class, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
                emb_length /= 2

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, int(emb_length),time_channels)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, int(emb_length), time_channels, class_channels, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, int(emb_length), time_channels, class_channels, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))
                emb_length *= 2

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Sequential(
            nn.Conv1d(in_channels,out_channels=1,kernel_size=3,padding=1),
            nn.Linear(n_length,feature_length)
        )
        # self.final = nn.Linear(n_length*in_channels,feature_length)

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_: torch.Tensor, class_mask:torch.Tensor):
        """
        * x shape = [batch_size, in_channels, length]
        * t shape = [batch_size]
        * class_embedding shape = [batch_size, num_class]
        """
        # 获取time embeddings shape = [batch_size, time_channels]
        batch_size = x.shape[0]
        t = self.time_emb(t)
        # 随机mask掉一些样本的类型引导，通过这一步使模型同时具有条件生成和无条件生成的能力
        # class_mask = class_mask[:, None]
        # class_mask = class_mask.repeat(1,self.num_class)
        # class_mask = (-1*(1-class_mask))

        # # 注意数据是否需要进行onehot编码
        # class_ = nn.functional.one_hot(class_, num_classes=self.num_class)
        class_ = class_ * class_mask

        x = self.vector_proj(x)
        # print(x.shape)
        # `h`存储下采样中每一步的输出 用于skip connection
        h = [x]
        # Encode部分 下采样
        for m in self.down:
            x = m(x, t, class_)
            # print(x.shape)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Decode部分 上采样
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        f = self.act(self.norm(x))
        # return self.final(f.view(batch_size,-1)).unsqueeze(1)
        return self.final(f)

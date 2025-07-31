import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
import dgl
import dgl.data as data
from utils.config import UnetConfig


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

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

    def __init__(self, in_length: int, out_length: int, time_emb_length: int,
                 dropout: float = 0.1):
        """
        Params:
            in_channels:  输入向量的channel数量
            out_channels: 经过residual block后输出向量的channel数量
            time_channels:time_embedding的向量维度,例如t原来是个整型,值为1,表示时刻1,
                           现在要将其变成维度为(1, time_channels)的向量
            dropout:      dropout rate
        """
        super().__init__()
        
        # 第一层线性层
        self.norm1 = nn.BatchNorm1d(in_length)
        self.act1 = Swish()
        # self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(in_length, out_length)

        # 第二层线性层
        self.norm2 = nn.BatchNorm1d(out_length)
        self.act2 = Swish()
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.linear2 = nn.Linear(out_length, out_length)


        # 当in_c = out_c时,残差连接直接将输入输出相加；
        # 当in_c != out_c时,对输入数据做一次卷积,将其通道数变成和out_c一致,再和输出相加
        if in_length != out_length:
            # self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.shortcut = nn.Linear(in_length, out_length)
        else:
            self.shortcut = nn.Identity()

        # t向量的维度time_channels可能不等于out_c,所以要对起做一次线性转换
        self.time_emb = nn.Linear(time_emb_length, out_length)
        self.time_act = Swish()
        # self.class_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Params:
            x: 输入数据xt,尺寸大小为(batch_size, length)
            t: 输入数据t,尺寸大小为(batch_size, time_emb_length)
        """
        # 1.输入数据先过一层线性层
        h = self.linear1(self.act1(self.norm1(x)))
        # print(h.shape)
        # 2. 对time_embedding向量,通过线性层使time_c变为out_c,再和输入数据的特征图相加
        h += self.time_emb(self.time_act(t))
        # 3、过第二层卷积
        h = self.linear2(self.dropout(self.act2(self.norm2(h))))

        # 4、返回残差连接后的结果
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """
    Attention模块
    和Transformer中的multi-head attention原理及实现方式一致
    """

    def __init__(self, n_length: int, n_heads: int = 1, d_k: int = None):
        """
        Params:
            n_channels:等待做attention操作的特征图的channel数
            n_heads:   attention头数
            d_k:       每一个attention头处理的向量维度
            n_groups:  Group Norm超参数
        """
        super().__init__()

        # 一般而言,d_k = n_channels // n_heads,需保证n_channels能被n_heads整除
        if d_k is None:
            d_k = n_length
        # 定义Group Norm
        self.norm = nn.BatchNorm1d(n_length)
        # Multi-head attention层: 定义输入token分别和q,k,v矩阵相乘后的结果
        self.projection = nn.Linear(n_length, n_heads * d_k * 3)
        # MLP层
        self.output = nn.Linear(n_heads * d_k, n_length)
        
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        Params:
            x: 输入数据xt,尺寸大小为(batch_size, length)
            t: 输入数据t,尺寸大小为(batch_size, time_length)
        """
        # t并没有用到,但是为了和ResidualBlock定义方式一致,这里也引入了t
        _ = t
        # 获取shape
        batch_size, length = x.shape
        # 将输入数据的shape改为(batch_size, height*weight, n_channels)
        # 这三个维度分别等同于transformer输入中的(batch_size, seq_length, token_embedding)
        x = x.permute(0, 2, 1)
        # 计算输入过矩阵q,k,v的结果,self.projection通过矩阵计算,一次性把这三个结果出出来
        # 也就是qkv矩阵是三个结果的拼接
        # 其shape为:(batch_size, height*weight, n_heads, 3 * d_k)
        qkv = self.projection(x).view(batch_size, length, self.n_heads, 3 * self.d_k)
        # 将拼接结果切开,每一个结果的shape为(batch_size, height*weight, n_heads, d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # 以下是正常计算attention score的过程,不再做说明
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # 将结果reshape成(batch_size, height*weight,, n_heads * d_k)
        # 复习一下:n_heads * d_k = n_channels
        res = res.view(batch_size, length, self.n_heads * self.d_k)
        # MLP层,输出结果shape为(batch_size, height*weight,, n_channels)
        res = self.output(res)

        # 残差连接
        res += x

        # 将输出结果从序列形式还原成图像形式,
        # shape为(batch_size, n_channels, height, width)
        res = res.permute(0, 2, 1).view(batch_size, length)
        return res

class DownBlock(nn.Module):
    """
    Down block,即Encoder中每一层的核心处理逻辑
    DownBlock = ResidualBlock + AttentionBlock
    """

    def __init__(self, in_length: int, out_length: int, time_emb_length: int, class_dims:int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_length, out_length, time_emb_length)
        self.guidance_emb = ClassEmbedding(class_dims,in_length)
        if has_attn:
            self.attn = AttentionBlock(out_length)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_ : torch.Tensor):
        class_embedding = self.guidance_emb(class_)
        x += class_embedding
        x = self.res(x, t)
        x = self.attn(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_length: int, out_length: int, time_emb_length: int, class_dims:int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_length + out_length, out_length, time_emb_length)
        # self.guidance_emb = ClassEmbedding(class_dims,in_channels)
        if has_attn:
            self.attn = AttentionBlock(out_length)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # class_embedding = self.guidance_emb(class_)
        # x += class_embedding[:,:,None,None]
        x = self.res(x, t)
        x = self.attn(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, n_length: int, time_emb_length: int):
        super().__init__()
        self.res1 = ResidualBlock(n_length, n_length, time_emb_length)
        self.attn = AttentionBlock(n_length)
        self.res2 = ResidualBlock(n_length, n_length, time_emb_length)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        # x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    def __init__(self, n_length):
        super().__init__()
        self.trans = nn.Linear(n_length, n_length)

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_embedding: torch.Tensor = None):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.trans(x)


class Downsample(nn.Module):
    def __init__(self, n_length):
        super().__init__()
        self.trans = nn.Linear(n_length, n_length)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.trans(x)

class UNet(nn.Module):
    def __init__(self, unet_config:UnetConfig):
        """
        * `vector_channels` is the number of channels in the vector.
        * `n_channels` is number of channels in the initial feature map that we transform the vector into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        feature_length: int = unet_config.feature_length
        n_length: int = unet_config.n_channels
        ln_mults: tuple = unet_config.ch_mults
        is_attn: tuple = unet_config.is_attn
        n_blocks: int = unet_config.n_blocks
        class_guidance_length: int = unet_config.class_embedding_channel
        time_guidance_length:int = unet_config.time_embedding_channel
        self.num_class: int = unet_config.num_class
        super(UNet,self).__init__()

        # Number of resolutions
        n_resolutions = len(ln_mults)

        # Project image into feature map
        self.vector_proj = nn.Linear(feature_length, n_length)

        # Time embedding 层 将时间步t输出为channel为 `n_channels * 4` 的时间嵌入
        self.time_emb = TimeEmbedding(time_guidance_length)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_length = in_length = n_length
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_length = int(in_length * ln_mults[i])
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_length, out_length, time_guidance_length, self.num_class, is_attn[i]))
                in_length = out_length
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_length))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_length,time_guidance_length)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_length = out_length
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_length = in_length
            for _ in range(n_blocks):
                up.append(UpBlock(in_length, out_length, time_guidance_length, class_guidance_length, is_attn[i]))
            # Final block to reduce the number of channels
            out_length = int(in_length // ln_mults[i])
            up.append(UpBlock(in_length, out_length, time_guidance_length, class_guidance_length, is_attn[i]))
            in_length = out_length
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_length))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.BatchNorm1d(n_length)
        self.act = Swish()
        self.final = nn.Linear(in_length, feature_length)

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_: torch.Tensor, class_mask:torch.Tensor):
        """
        * x shape = [batch_size, length]
        * t shape = [batch_size]
        * class_embedding shape = [batch_size, num_class]
        """
        # 获取time embeddings shape = [batch_size, time_channels]
        t = self.time_emb(t)
        # 随机mask掉一些样本的类型引导，通过这一步使模型同时具有条件生成和无条件生成的能力
        class_mask = class_mask[:, None]
        class_mask = class_mask.repeat(1,self.num_class)
        class_mask = (-1*(1-class_mask))

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

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))



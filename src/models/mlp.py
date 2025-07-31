import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    一个简单的MLP模型
    """
    def __init__(self, input_size, output_size, layers, drop = 0.1):
        """
        Args:
            layers (_type_): 隐藏层数量（不包含输出）
            drop (float, optional): Defaults to 0.1.
        """
        super(MLP, self).__init__()
        hidden_size = torch.linspace(input_size,output_size,layers+2,dtype=torch.int32)[1:-1]
        in_size = input_size
        self.layers = nn.Sequential()
        for hidden in hidden_size:
            self.layers.add_module("linear{}-{}".format(in_size,hidden),nn.Linear(in_size,hidden))
            self.layers.add_module("act{}-{}".format(in_size,hidden),nn.ReLU())
            self.layers.add_module("drop{}-{}".format(in_size,hidden),nn.Dropout(drop))
            in_size = hidden
        self.fc = nn.Linear(in_size, output_size)

    def forward(self, x:torch.Tensor):
        h = self.layers(x)
        out = self.fc(h)
        return out

class res_MLP(nn.Module):
    """
    一个残差连接MLP模型

    Args:
        nn (_type_): _description_
    """
    def __init__(self,inputs_size:int,outputs_size:int,dropout):
        super().__init__()
        hidden_size = int((inputs_size+outputs_size) // 2)
        self.dropout_p = dropout
        self.layers = nn.Sequential(
            nn.Linear(inputs_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size,outputs_size),
            nn.GELU(),
        )
        self.outlayer = nn.Linear(outputs_size,outputs_size)
        # self.final_act = nn.GELU()
        if inputs_size == outputs_size:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(inputs_size,outputs_size)
    def forward(self,inputs):
        h = self.layers(inputs)
        outputs = self.outlayer(F.relu(self.shortcut(inputs)+h))
        return outputs
    

class MLP_f(nn.Module):
    """
    一个简单的MLP模型
    """
    def __init__(self, input_size, output_size, layers, drop = 0.1):
        """
        Args:
            layers (_type_): 隐藏层数量（不包含输出）
            drop (float, optional): Defaults to 0.1.
        """
        super(MLP_f, self).__init__()
        hidden_size = torch.linspace(input_size,output_size,layers+2,dtype=torch.int32)[1:-1]
        in_size = input_size
        self.models = nn.Sequential()
        for hidden in hidden_size:
            self.models.add_module("linear{}-{}".format(in_size,hidden),nn.Linear(in_size,hidden))
            self.models.add_module("act{}-{}".format(in_size,hidden),nn.ReLU())
            self.models.add_module("drop{}-{}".format(in_size,hidden),nn.Dropout(drop))
            in_size = hidden
        self.fc = nn.Linear(in_size, output_size)
        self.reg_params = self.models.parameters()
        self.non_reg_params = self.fc.parameters()

    def forward(self, x:torch.Tensor, edge=None, edge_weight=None):
        # 对每一个样本进行normalize
        # x = F.normalize(x,p=1,dim=x.dim()-1)
        h = self.models(x)
        out = self.fc(h)
        return out
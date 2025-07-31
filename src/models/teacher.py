import torch
import torch.nn as nn

from . import mlp
from ..utils import VNG_utils

class MLPTeacher(nn.Module):
    def __init__(self,in_feats,out_feats,layers,drop):
        super().__init__()
        self.model = mlp.MLP(in_feats,out_feats,layers,drop)
    def forward(self,x):
        logits = self.model(x)
        return logits
    def softmax_with_temperature(self, feats, temperature):
        self.eval()
        with torch.no_grad():
            logits = self.model(feats)
            softmax_output = VNG_utils.softmax_with_temperature(logits, temperature)
        return softmax_output



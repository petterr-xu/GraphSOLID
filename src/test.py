import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# import dgl.data as data
# import networkx as nx
# import numpy as np
# import dataclasses
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTEN,SMOTE
# import os
# import sys

# from denoise import unet
# from models import ae, gnn, gsl, graph_smote,diffusion
# from utils.config import GraphDatasetConfig,DiffusionConfig,ClassifierConfig,UnetConfig,VAEConfig,GraphEncoderConfig

import utils.VNG_utils as VNG_utils
import numpy as np
from random import sample
import matplotlib.pyplot as plt

x_g = torch.randn([4,16])
x_r = torch.randn([7,16])
fid_score = VNG_utils.fid(x_r,x_g)
print(fid_score)
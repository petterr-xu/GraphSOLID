import numpy as np
from typing import Optional, Dict, Any


ArrayDict = Dict[str, np.ndarray]
class TabDataset:
    name: str
    num_numerical_features: int # number of numerical features
    categories: Optional[np.ndarray] # number array of categories for each categorical feature
    n_labels: Optional[int]
    n_features: int
    graph: object # numerical features will be move to the top of feature matrix

class HeteroGraphDataset:
    name: str
    

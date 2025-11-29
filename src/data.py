import numpy as np
from typing import Optional, Dict, Any


ArrayDict = Dict[str, np.ndarray]
class TabDataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    int_col_idx_wrt_num: list
    y_info: Dict[str, Any]
    n_classes: Optional[int]
    raw_graph: object
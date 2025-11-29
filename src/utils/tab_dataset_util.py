import VNG_utils
from ..data import *

def load_tab_dataset_info(name, path, split_type='public') -> TabDataset:
    graph_dataset = VNG_utils.load_dataset(name, path, split_type)
    dataset = TabDataset(
        raw_graph = graph_dataset
    )
    return dataset

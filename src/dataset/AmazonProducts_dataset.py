from torch_geometric.data import HeteroData
from .hetero_dataset import HeteroDataset, MultiviewDataset, HeteroDatasetInfos, HeteroDataModule



class AmPdMultiviewDataset(MultiviewDataset):
    def __init__(self, stage, root, hetero_dataset, metapaths, target, transform=None, pre_transform=None):
        super().__init__(stage, root, hetero_dataset, metapaths, target, transform, pre_transform)


class AmPdHeteroDataset(HeteroDataset):
    def __init__(
        self,
        stage,
        root,
        original_data: HeteroData,
        target_node_type,
        num_parts=100,
        transform=None,
        pre_transform=None,
        strategy='HGT',
        node_val_ratio: float = 0.2,
        node_test_ratio: float = 0.4,
        rebalance_train_majority: bool = False,
        train_majority_ratio_cap: float = 0.85,
    ):
        super().__init__(stage,
                        root,
                        original_data,
                        target_node_type,
                        num_parts,
                        transform,
                        pre_transform,
                        strategy,
                        node_val_ratio,
                        node_test_ratio,
                        rebalance_train_majority,
                        train_majority_ratio_cap)

class AmPdDataModule(HeteroDataModule):
    def __init__(self, cfg, hetero_graph=None):
        super().__init__(cfg, hetero_graph)
        
class AmPdDatasetInfos(HeteroDatasetInfos):
    def __init__(self,datamodule, cfg):
        super().__init__(datamodule, cfg)

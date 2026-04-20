import os
import sys
import hydra
import torch
import warnings
import os.path as osp
import torch.nn.functional as F
from collections import defaultdict
from omegaconf import DictConfig
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.join(os.getcwd(), "src"))

from src.utils import VNG_utils, graphbuilder
from src.utils.hetero_dataset_util import GraphDataLoader
from src.DiGress.src import utils as digress_utils
from src.models.soft_multiview_classifier import SoftMultiViewClassifier

warnings.filterwarnings("ignore")


def load_imb_data(dataset, imb_ratio=0, keep_edge=True, device='cpu'):
    root_path = osp.dirname(osp.realpath(__file__))
    loader = GraphDataLoader()
    data_path = osp.join(root_path, 'data', dataset, 'data', dataset + '.mat')
    cnfg_path = osp.join(root_path, 'data', dataset, 'meta', dataset + '.json')
    hetero_ctx = loader.load_from_config(cnfg_path, data_path)
    target = hetero_ctx.target_node
    data = hetero_ctx.g.to(device)
    n_cls = hetero_ctx.n_classes

    if imb_ratio == 0:
        return hetero_ctx

    max_n = 500
    if dataset not in ['YelpChi', 'Amazon-Products']:
        raise NotImplementedError(f"Not implemented for dataset {dataset}")

    data_train_mask = data[target].train_mask.clone()
    stats = data[target].y[data_train_mask]
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    class_num_list, data_train_mask, _, edge_mask_dict = graphbuilder.make_hetero_longtailed_data_remove(
        data, target, n_data, n_cls, imb_ratio, data_train_mask.clone(), max_n
    )
    hetero_ctx.g[target].train_mask = data_train_mask
    if not keep_edge:
        for etype, mask in edge_mask_dict.items():
            hetero_ctx.g[etype].edge_index = hetero_ctx.g[etype].edge_index[:, mask]
    print("num of class in LT-training data: {} -> {}".format(class_num_list, sum(data_train_mask).item()))
    return hetero_ctx


def build_grouped_indices(view_dataset):
    groups = defaultdict(list)
    for idx in range(len(view_dataset)):
        sample = view_dataset[idx]
        parent_idx = int(sample.parent_hetero_idx)
        view_idx = int(sample.y.argmax().item()) if sample.y.dim() > 1 else int(sample.y.item())
        groups[parent_idx].append((view_idx, idx))
    ordered = []
    for _, entries in sorted(groups.items(), key=lambda item: item[0]):
        ordered.append([idx for _, idx in sorted(entries, key=lambda item: item[0])])
    return ordered


def load_view_bundle(view_dataset, grouped_indices, device):
    views = [view_dataset[idx] for idx in grouped_indices]
    x = views[0].x.float().unsqueeze(0).to(device)
    if hasattr(views[0], "node_y"):
        y = views[0].node_y.view(-1).long().unsqueeze(0).to(device)
    else:
        y = views[0].x.argmax(dim=-1).long().unsqueeze(0).to(device)

    edge_views = []
    node_mask = None
    for view in views:
        num_nodes = view.x.size(0)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=view.x.device)
        dense_data, dense_mask = digress_utils.to_dense(view.x, view.edge_index, view.edge_attr, batch)
        edge_views.append(dense_data.E[0])
        if node_mask is None:
            node_mask = dense_mask[0]

    edge_views = torch.stack(edge_views, dim=0).unsqueeze(0).float().to(device)
    node_mask = node_mask.unsqueeze(0).to(device)
    train_mask = views[0].train_mask.unsqueeze(0).bool().to(device)
    val_mask = views[0].val_mask.unsqueeze(0).bool().to(device)
    test_mask = views[0].test_mask.unsqueeze(0).bool().to(device)
    return x, edge_views, y, node_mask, train_mask, val_mask, test_mask


def run_epoch(model, view_dataset, grouped_indices, device, split, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_items = 0
    y_true = []
    y_pred = []

    for group in grouped_indices:
        x, edge_views, y, node_mask, train_mask, val_mask, test_mask = load_view_bundle(view_dataset, group, device)
        if split == "train":
            mask = train_mask & node_mask
        elif split == "val":
            mask = val_mask & node_mask
        elif split == "test":
            mask = test_mask & node_mask
        else:
            raise ValueError(f"Unknown split: {split}")

        if not mask.any():
            continue

        if is_train:
            optimizer.zero_grad()

        logits = model(x, edge_views, node_mask)
        loss = F.cross_entropy(logits[mask], y[mask])

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * int(mask.sum().item())
        total_items += int(mask.sum().item())
        preds = logits.argmax(dim=-1)
        y_true.extend(y[mask].detach().cpu().tolist())
        y_pred.extend(preds[mask].detach().cpu().tolist())

    mean_loss = total_loss / max(total_items, 1)
    if total_items == 0:
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0}
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {"loss": mean_loss, "acc": acc, "macro_f1": macro_f1}


@hydra.main(version_base='1.3', config_path='./configs', config_name='config')
def main(cfg: DictConfig):
    device = "cuda:0" if torch.cuda.is_available() and cfg.general.gpus > 0 else "cpu"
    dataset_name = cfg.dataset.name
    hetero_data = load_imb_data(
        dataset_name,
        imb_ratio=float(getattr(cfg.dataset, "imb_ratio", 0)),
        keep_edge=bool(getattr(cfg.dataset, "keep_edge", True)),
    )

    if dataset_name == 'YelpChi':
        from src.dataset.YelpChi_dataset import YelpChihDataModule
        datamodule = YelpChihDataModule(cfg, hetero_data.g)
    elif dataset_name == 'Amazon-Products':
        from src.dataset.AmazonProducts_dataset import AmPdDataModule
        datamodule = AmPdDataModule(cfg, hetero_data.g)
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")

    train_groups = build_grouped_indices(datamodule.train_dataset)
    val_groups = build_grouped_indices(datamodule.val_dataset)
    test_groups = build_grouped_indices(datamodule.test_dataset)

    use_input_x = bool(getattr(cfg.general, "task_classifier_use_input_x", False))
    input_dim = datamodule.train_dataset[0].x.size(-1) if use_input_x else 1
    model = SoftMultiViewClassifier(
        input_dim=input_dim,
        hidden_dim=int(getattr(cfg.general, "task_classifier_hidden_dim", 64)),
        num_classes=int(hetero_data.n_classes),
        num_layers=int(getattr(cfg.general, "task_classifier_num_layers", 2)),
        dropout=float(getattr(cfg.general, "task_classifier_dropout", 0.1)),
        use_input_x=use_input_x,
        fusion=str(getattr(cfg.general, "task_classifier_fusion", "attention")),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(getattr(cfg.general, "task_classifier_lr", 1e-3)),
        weight_decay=float(getattr(cfg.general, "task_classifier_weight_decay", 1e-4)),
    )

    train_device = next(model.parameters()).device
    best_val_f1 = -1.0
    best_path = getattr(cfg.general, "task_classifier_output", "./outputs/task_classifier.pt")
    patience = int(getattr(cfg.general, "task_classifier_patience", 10))
    patience_count = 0
    epochs = int(getattr(cfg.general, "task_classifier_epochs", 100))

    for epoch in range(epochs):
        train_metrics = run_epoch(model, datamodule.train_dataset, train_groups, train_device, "train", optimizer=optimizer)
        val_metrics = run_epoch(model, datamodule.val_dataset, val_groups, train_device, "val")
        print(
            f"[task-clf] epoch={epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_count = 0
            model.save_checkpoint(best_path)
        else:
            patience_count += 1

        if patience_count >= patience:
            print(f"[task-clf] early stop at epoch {epoch + 1}")
            break

    best_model = SoftMultiViewClassifier.load_checkpoint(best_path, map_location=train_device).to(train_device)
    test_metrics = run_epoch(best_model, datamodule.test_dataset, test_groups, train_device, "test")
    print(
        f"[task-clf] best_val_f1={best_val_f1:.4f} "
        f"test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['macro_f1']:.4f} "
        f"saved={best_path}"
    )


if __name__ == '__main__':
    main()

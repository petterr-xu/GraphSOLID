import dgl
import dgl.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from . import mlp
from ..utils import VNG_utils


def train_mlp_classifier(graph:dgl.DGLGraph,train_device="cuda:0",test_device="cuda:0",show=False,save=False):
    with graph.local_scope():
        train_feat_data = graph.ndata["feat"][graph.ndata["train_mask"]].to(train_device)
        train_label_data = graph.ndata["label"][graph.ndata["train_mask"]].to(train_device)
        train_dataset = TensorDataset(train_feat_data,train_label_data)
        data_loader = DataLoader(train_dataset, 16, shuffle=True)

        val_feat_data = graph.ndata["feat"][graph.ndata["val_mask"]]
        val_label_data = graph.ndata["label"][graph.ndata["val_mask"]]

        test_feat_data = graph.ndata["feat"][graph.ndata["test_mask"]]
        test_label_data = graph.ndata["label"][graph.ndata["test_mask"]]
        patience = 10
        patience_count = 0
        best_val_acc = 0

        model = mlp.MLP(input_size=train_feat_data.shape[-1],output_size=7,layers=3)
        model.to(train_device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(300):
            model.train()
            loss_value = 0.0
            for inputs,targets in data_loader:
                optimizer.zero_grad()
                pred = model(inputs)
                loss = F.cross_entropy(pred,targets.to(torch.float32))
                loss.backward()
                optimizer.step()
                loss_value += loss.item()
            if (epoch) % 5 == 0:
                val_acc = val_mlp_classifier(model,val_feat_data,val_label_data,val_device=test_device)
                if show: print('In epoch {}, loss: {:.3f},val acc: {:.3f}'.format(epoch, loss_value, val_acc))
                if(val_acc > best_val_acc):
                    best_val_acc = val_acc
                    patience_count = 0
                else: patience_count += 1
                if patience_count >= patience : break
        print("mlp classifier:")
        val_mlp_classifier(model,test_feat_data,test_label_data,test_device,show_detail=True)
        if save : VNG_utils.save(model,file_path="CGDM-Im\\history_data\\guidance_classifier_model_checkpoint\\mlp_classifier.pth")
    return model


def val_mlp_classifier(model:nn.Module,val_inputs,val_targets,val_device,show_detail=False):
    model.eval()
    model.to(val_device)
    logits = model(val_inputs.to(val_device))
    pred = logits.argmax(1)
    if show_detail:
        VNG_utils.show_detailed_evaluation(val_targets.argmax(1).to(val_device),pred)
    val_acc = (pred == val_targets.argmax(1).to(val_device)).float().mean()
    return val_acc
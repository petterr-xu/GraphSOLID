import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self, num_classes=7, feat_dim=128, weight=None, device="cuda:0"):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        if weight is None:
            weight = torch.full((num_classes,),fill_value=1/num_classes,dtype=torch.float,device=device)
        self.weight = weight

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(labels.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        if self.weight is None:
            loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        else:
            loss = (dist * self.weight[labels]).clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss
    

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target, weight=None, reduction='mean'):
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)
    
def focal_loss(labels, logits, alpha, gamma):

    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss,dim=1)

    return torch.mean(focal_loss)

class IMB_LOSS:
    def __init__(self,loss_name,n_cls,n_train,factor=None,device="cuda:0"):
        self.loss_name = loss_name
        self.device    = device
        self.cls_num   = n_cls
        
        train_size = n_train
        train_size_arr = np.array(train_size)
        train_size_mean = np.mean(train_size_arr)
        train_size_factor = train_size_mean / train_size_arr
        
        #alpha in re-weight
        self.factor_train = torch.from_numpy(train_size_factor).type(torch.FloatTensor)
        

        #beta in CB
        weights = torch.from_numpy(np.array([1.0 for _ in range(self.cls_num)])).float()

        if self.loss_name == 'focal':
            #gamma in focal
            self.factor_focal = factor
            weights = self.factor_train

        if self.loss_name == 'cb-softmax':
            beta = factor
            effective_num = 1.0 - np.power(beta, train_size_arr)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * self.cls_num
            weights = torch.tensor(weights).float()

        self.weights = weights.unsqueeze(0).to(device)



    def compute(self,pred,target):

        if self.loss_name == 'ce':
            return F.cross_entropy(pred,target,weight=None,reduction='mean')

        elif self.loss_name == 're-weight':
            return F.cross_entropy(pred,target,weight=self.factor_train.to(self.device),reduction='mean')

        elif self.loss_name == 'focal':
            labels_one_hot = F.one_hot(target, self.cls_num).type(torch.FloatTensor).to(self.device)
            weights = self.weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1,self.cls_num)

            return focal_loss(labels_one_hot,pred,weights,self.factor_focal)

        elif self.loss_name == 'cb-softmax':
            labels_one_hot = F.one_hot(target, self.cls_num).type(torch.FloatTensor).to(self.device)
            weights = self.weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1,self.cls_num)

            pred = pred.softmax(dim = 1)
            temp_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights,reduction='mean') 
            return temp_loss

        else:
            raise Exception("No Implentation Loss")

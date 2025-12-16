import torch
x_marginals = torch.tensor([1,2,3,4,5,6])
X_classes = len(x_marginals)
u_x = x_marginals.unsqueeze(0).expand(X_classes, -1).unsqueeze(0)
print(u_x)
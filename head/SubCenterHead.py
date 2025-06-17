import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class SubCenterHead(Module):
    def __init__(self, feat_dim, num_class, k=3):
        super(SubCenterHead, self).__init__()
        self.k = k
        self.num_class = num_class
        self.weight = Parameter(torch.Tensor(num_class * k, feat_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-1, 1)
        self.weight.data = F.normalize(self.weight.data, dim=1)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        weight = F.normalize(self.weight, dim=1)

        cosine = torch.matmul(x, weight.t())  # [B, num_class * k]
        cosine = cosine.view(-1, self.num_class, self.k)
        cosine, _ = torch.max(cosine, dim=2)  # [B, num_class]

        return cosine

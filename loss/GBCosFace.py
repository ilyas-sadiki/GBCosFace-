import numpy as np
import torch
import torch.cuda.comm
import torch.distributed as dist
from torch.distributed import ReduceOp, get_world_size
import numpy as np
import torch
import torch.nn as nn

class SmoothedCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        """
        logits: precomputed Original and Reversed terms [batchsize, 2]
        target: true class index = 0 (Original term)
        """
        n_classes = logits.size(1)
        # one-hot encode target
        target_probs = torch.nn.functional.one_hot(target, num_classes=n_classes).float()
        # label smoothing: q'(k|x) = (1-eps) δ_k,y + eps/num_classes
        target_probs = (1 - self.eps) * target_probs + self.eps / n_classes
        # compute loss directly without softmax
        loss = -(target_probs * logits).sum(dim=1).mean()
        return loss


class GBCosFace(nn.Module):
    def __init__(self, local_rank, eps, s=32, min_cos_v=0.5, max_cos_v=0.7,
                 margin=0.16, update_rate=0.01, alpha=0.15):
        super(GBCosFace, self).__init__()
        self.scale = s
        self.margin = margin
        self.cos_v = None
        self.min_cos_v = min_cos_v
        self.max_cos_v = max_cos_v
        self.target = torch.tensor([0], dtype=torch.long)  # Original term = class 0
        self.update_rate = update_rate
        self.alpha = alpha
        self.eps = eps
        self.label_smooth_loss = SmoothedCrossEntropy(eps=eps)
        self.local_rank = local_rank

    def forward(self, cos_theta, labels, **args):
        batchsize = cos_theta.size(0)

        # create mask for positive class
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.view(-1, 1), 1)
        index = index.bool()

        # positive similarities
        cos_p = cos_theta[index].unsqueeze(-1)
        cos_pm = cos_p - self.margin

        # negative similarities
        index_neg = torch.bitwise_not(index)
        cos_i = cos_theta[index_neg].view(batchsize, -1)

        # compute virtual cos
        cos_n = torch.logsumexp(cos_i * self.scale, dim=-1) / self.scale
        cos_n = cos_n.unsqueeze(-1)
        cos_v = (cos_p.detach() + cos_n.detach()) / 2
        cos_v_update = torch.mean(cos_v).reshape(1)

        if self.cos_v is None:
            self.cos_v = cos_v_update
        self.cos_v = torch.clamp(self.cos_v, self.min_cos_v, self.max_cos_v)

        delta = self.alpha * (self.cos_v - cos_v)
        cos_v_pred = cos_v + delta

        target = self.target.expand(batchsize).to(cos_theta.device)

        # === Original term O_i ===
        O_i = -0.5 * (
            torch.log(torch.exp(2*self.scale*(cos_pm)) / (torch.exp(2*self.scale*(cos_pm)) + torch.exp(2*self.scale*cos_v_pred))) +
            torch.log(torch.exp(2*self.scale*(cos_v_pred - self.margin)) / (torch.exp(2*self.scale*(cos_v_pred - self.margin)) + torch.exp(2*self.scale*cos_n)))
        )

        # === Reversed term R_i ===
        R_i = -0.5 * (
            torch.log(torch.exp(2*self.scale*cos_n) / (torch.exp(2*self.scale*cos_n) + torch.exp(2*self.scale*cos_v_pred))) +
            torch.log(torch.exp(2*self.scale*(cos_v_pred - self.margin)) / (torch.exp(2*self.scale*(cos_v_pred - self.margin)) + torch.exp(2*self.scale*cos_pm)))
        )

        # combine Original and Reversed terms into logits
        final_logits = torch.cat([O_i, R_i], dim=-1)  # [batchsize, 2]

        # === single call to label-smoothed loss ===
        total_loss = self.label_smooth_loss(final_logits, target)

        # update cos_v
        self.cos_v = (1 - self.update_rate) * self.cos_v + self.update_rate * cos_v_update

        return dict(
            loss=total_loss,
            cos_v=self.cos_v,
            delta=delta
        )


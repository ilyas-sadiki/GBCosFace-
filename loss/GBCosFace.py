import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.nn as nn

class SmoothedCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        n_classes = logits.size(1)
        target_probs = torch.nn.functional.one_hot(target, num_classes=n_classes).float()
        target_probs = (1 - self.eps) * target_probs + self.eps / n_classes
        loss = -(target_probs * logits).sum(dim=1)  # per-sample loss
        return loss  # mean handled later

class GBCosFace(nn.Module):
    def __init__(self, local_rank, eps=0.1, s=32, min_cos_v=0.5, max_cos_v=0.7,
                 margin=0.16, update_rate=0.01, alpha=0.15):
        super(GBCosFace, self).__init__()
        self.scale = s
        self.margin = margin
        self.cos_v = None
        self.min_cos_v = min_cos_v
        self.max_cos_v = max_cos_v
        self.target = torch.tensor([0], dtype=torch.long)
        self.update_rate = update_rate
        self.alpha = alpha
        self.eps = eps
        self.label_smooth_loss = SmoothedCrossEntropy(eps=eps)
        self.local_rank = local_rank

    def forward(self, cos_theta, labels, **args):
        batchsize = cos_theta.size(0)
        # mask positive class
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
        delta_for_log = torch.mean(torch.abs(delta))
        cos_v_pred = cos_v + delta

        target = self.target.expand(batchsize).to(cos_theta.device)

        # logits for pos/neg
        pos_logits = torch.cat([cos_pm, cos_v_pred], dim=-1)
        neg_logits = torch.cat([cos_v_pred - self.margin, cos_n], dim=-1)

        # label-smoothed losses
        pos_loss = self.label_smooth_loss(pos_logits, target).mean() / 2
        neg_loss = self.label_smooth_loss(neg_logits, target).mean() / 2

        # update cos_v
        self.cos_v = (1 - self.update_rate) * self.cos_v + self.update_rate * cos_v_update

        # DDP sync
        dist.all_reduce(self.cos_v, op=ReduceOp.SUM)
        self.cos_v /= dist.get_world_size()

        # extra logs
        delta_p = torch.softmax(2*self.scale * pos_logits.detach(), 1)
        delta_n = torch.softmax(2*self.scale * neg_logits.detach(), 1)
        delta_p = delta_p[:, 1]
        delta_n = torch.sum(delta_n[:, 1:], 1)
        delta_p_mean = torch.mean(delta_p)
        delta_n_mean = torch.mean(delta_n)

        return dict(
            loss_pos=pos_loss,
            loss_neg=neg_loss,
            cos_v=self.cos_v,
            delta_p=delta_p_mean,
            delta_n=delta_n_mean,
            delta=delta_for_log,
            update_rate=self.update_rate
        )

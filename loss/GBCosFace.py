import numpy as np
import torch
import torch.cuda.comm
import torch.distributed as dist
from torch.distributed import ReduceOp, get_world_size

class SmoothedCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        n_classes = logits.size(1)
        target_probs = torch.nn.functional.one_hot(target, num_classes=n_classes).float()
        target_probs = (1 - self.eps) * target_probs + self.eps / n_classes
        loss = -(target_probs * log_probs).sum(dim=1).mean()
        return loss

class GBCosFace(torch.nn.Module):
    def __init__(self, local_rank, eps, s=32, min_cos_v=0.5, max_cos_v=0.7, margin=0.16, update_rate=0.01, alpha=0.15):
        super(GBCosFace, self).__init__()
        self.scale = s
        self.margin = margin
        self.cos_v = None
        self.min_cos_v = min_cos_v
        self.max_cos_v = max_cos_v
        self.target = torch.from_numpy(np.array([0], np.int64))
        self.update_rate = update_rate
        self.alpha = alpha
        self.label_smooth_loss = SmoothedCrossEntropy(eps=eps)
        self.local_rank = local_rank
        self.eps = eps

    def forward(self, cos_theta, labels, **args):
        def cal_cos_n(pi, s):
            cos_n = torch.logsumexp(s*pi, dim=-1) / s
            return cos_n.unsqueeze(-1)

        update_rate = self.update_rate
        batchsize = cos_theta.size()[0]
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.bool()

        # pos cos similarities
        cos_p = cos_theta[index].unsqueeze(-1)
        cos_pm = cos_p - self.margin

        # neg cos similarities
        index_neg = torch.bitwise_not(index)
        cos_i = cos_theta[index_neg]
        cos_i = cos_i.view(batchsize, -1)

        # cal pv and update cos_v
        cos_n = cal_cos_n(cos_i, self.scale)
        cos_v = (cos_p.detach() + cos_n.detach()) / 2
        cos_v_update = torch.mean(cos_v).reshape(1)
        if self.cos_v is None:
            self.cos_v = cos_v_update
        self.cos_v = torch.clamp(self.cos_v, self.min_cos_v, self.max_cos_v)

        delta = self.alpha * (self.cos_v - cos_v)
        delta_for_log = torch.mean(torch.abs(delta))
        cos_v_pred = cos_v + delta

        # === Standard Terms ===
        std_pos_pred = torch.cat((cos_pm, cos_v_pred), -1)              # [py - m, pv]
        std_neg_pred = torch.cat((cos_v_pred - self.margin, cos_n), -1) # [pv - m, pn]
        target = self.target.expand(batchsize).to(cos_theta.device)
        std_pos_loss = self.label_smooth_loss(2 * self.scale * std_pos_pred, target.long())
        std_neg_loss = self.label_smooth_loss(2 * self.scale * std_neg_pred, target.long())
        std_loss = (1 - self.alpha) * (std_pos_loss + std_neg_loss) / 2

        # === Reversed Smoothed Terms ===
        rev_pos_pred = torch.cat((cos_n, cos_v_pred), -1)               # [pn, pv]
        rev_neg_pred = torch.cat((cos_v_pred - self.margin, cos_pm), -1) # [pv - m, py - m]
        rev_pos_loss = self.label_smooth_loss(2 * self.scale * rev_pos_pred, target.long())
        rev_neg_loss = self.label_smooth_loss(2 * self.scale * rev_neg_pred, target.long())
        rev_loss = self.alpha * (rev_pos_loss + rev_neg_loss) / 2

        total_loss = std_loss + rev_loss

        # update cos_v
        self.cos_v = (1 - self.update_rate) * self.cos_v + self.update_rate * cos_v_update

        # for debug log
        delta_p = torch.softmax(2*self.scale * std_pos_pred.detach(), 1)
        delta_n = torch.softmax(2*self.scale * std_neg_pred.detach(), 1)
        delta_p = delta_p[:, 1]
        delta_n = torch.sum(delta_n[:, 1:], 1)
        delta_p_mean = torch.mean(delta_p)
        delta_n_mean = torch.mean(delta_n)

        return dict(
            loss=total_loss,
            loss_pos=std_pos_loss,
            loss_neg=std_neg_loss,
            rev_pos=rev_pos_loss,
            rev_neg=rev_neg_loss,
            cos_v=self.cos_v,
            delta_p=delta_p_mean,
            delta_n=delta_n_mean,
            delta=delta_for_log,
            update_rate=update_rate
        )

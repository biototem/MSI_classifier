import torch
import torch.nn as nn


class WeightL1Loss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        super(WeightL1Loss, self).__init__()
        if weight is None:
            self.weight = [1, 1]
        else:
            self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        t = torch.abs(input - target)
        ret = torch.where(target < 1, t * self.weight[0], t * self.weight[1])
        ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret


class CEAddL1Loss(nn.Module):
    def __init__(self, weight=None, alpha=1.0, beta=1.0, reduction="mean"):
        super(CEAddL1Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        # self.L1 = nn.L1Loss()
        self.L1 = WeightL1Loss(weight=weight, reduction=reduction)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y):
        CE_loss = self.CE(x, y)
        x = x[:, -1]
        L1_loss = self.L1(x, y)
        loss = self.alpha * CE_loss + self.beta * L1_loss
        return loss

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLoss(torch.nn.Module, abc.ABC):

    def __init__(self, criteria: dict):
        super(MultiLoss, self).__init__()

        self.criterion_dict = {}
        self.losses_dict = {}
        self.num_criteria = 0
        self.total_loss = 0.

        for n, l in criteria.items():
            self.add_criterion(l, n)

    def add_criterion(self, criterion, criterion_name=None):
        criterion_name = "l_%s" % self.num_criteria if criterion_name is None else criterion_name
        self.add_module(criterion_name, criterion)
        self.criterion_dict[criterion_name] = criterion
        self.num_criteria += 1

    def item(self):
        return self.total_loss.item()

    @abc.abstractmethod
    def forward(self, inputs, targets: dict):
        self.losses_dict = {}
        for k in self.criterion_dict.keys():
            self.losses_dict[k] = self.criterion_dict[k](inputs[k], targets[k])

    def extra_forward(self, inputs, targets, suffix='', weight=1.):
        losses_dict = {}
        for k in self.criterion_dict.keys():
            losses_dict[k] = weight * self.criterion_dict[k](inputs[k], targets[k])
        self.add_extra_losses(losses_dict, suffix)

    def add_extra_loss(self, loss, loss_name: str):
        self.losses_dict[loss_name] = loss

    def add_extra_losses(self, losses_dict, suffix=''):
        for k, v in losses_dict.items():
            self.add_extra_loss(v, k if not suffix else f"{k}_{suffix}")

    def backward(self):
        self.total_loss.backward()

    def tensor(self):
        return self.total_loss


class UniformMultiLoss(MultiLoss):

    def forward(self, inputs, targets: dict):
        super(UniformMultiLoss, self).forward(inputs, targets)
        self.total_loss = 0.
        for k in self.losses_dict.keys():
            if not torch.isnan(self.losses_dict[k]):
                self.total_loss += self.losses_dict[k]
        return self

    def add_extra_loss(self, loss, loss_name):
        super(UniformMultiLoss, self).add_extra_loss(loss, loss_name)
        if not torch.isnan(loss):
            self.total_loss += loss


class UpsamplingCELoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean',
                 label_smoothing=0):
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, input, target):
        out_size = target.shape[-2:]
        if input.shape[0] != out_size[0] or input.shape[1] != out_size[1]:
            input = F.interpolate(input, size=out_size, mode='bilinear', align_corners=True)
        return super().forward(input, target.squeeze(dim=1).long())


# Adapted from https://github.com/hzhupku/SemiSeg-AEL/blob/747c7972a1aed589a8a62cdd98c3d1836c609735/semseg/utils/loss_helper.py#L99
class OhemCrossEntropy2D(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, **kwargs):
        super(OhemCrossEntropy2D, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, **kwargs)

    def forward(self, pred, target):
        out_size = target.shape[-2:]
        b, c, h, w = pred.size()
        if h != out_size[0] or w != out_size[1]:
            pred = F.interpolate(pred, size=out_size, mode='bilinear', align_corners=True)
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target.long(), torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                valid_mask = valid_mask * kept_mask

        target = target.long()
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

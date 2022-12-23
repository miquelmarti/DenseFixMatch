"""
Adapted from: https://github.com/IntelVCL/MultiObjectiveOptimization/
Which was in turn adapted from https://github.com/meetshah1995/pytorch-semseg/
"""

import abc
import statistics

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def class_metric_to_string(metric_values, total_entries):
    return "\n".join(["  Class #%d: %.3f on %d images" % (i, m, t) for i, m, t in
                      zip(range(len(total_entries)), metric_values, total_entries)
                      ]
                     )


class RunningMetric(abc.ABC):
    def __init__(self, null_target=None, descending=False):
        self.null_target = null_target
        self.descending = descending
        self.results = {}
        self.past_results = []
        self.num_updates = 0
        self.max_updates = None

    @abc.abstractmethod
    def reset(self):
        self.num_updates = 0

    @abc.abstractmethod
    def update(self, pred, gt):
        if self.max_updates is not None:
            pending_num_updates = self.max_updates - self.num_updates
            pred = pred[:pending_num_updates]
            gt = gt[:pending_num_updates]
        return pred, gt

    @abc.abstractmethod
    def compute_results(self):
        """Must set self.result to current results and append key metric to self.past_results"""
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def get_value(self):
        """Returns metric value, which has to match last stored in self.past_results"""
        pass

    def get_EMA(self):
        ema = pd.DataFrame(self.past_results).ewm(alpha=.1, adjust=True).mean()
        return float(ema.tail(1).to_numpy())

    def get_EMA_all(self):
        ema = pd.DataFrame(self.past_results).ewm(alpha=.1, adjust=True).mean()
        return ema.to_numpy().squeeze()


class RunningAccuracy(RunningMetric):
    def __init__(self, null_target, n_classes=None):
        super(RunningAccuracy, self).__init__(null_target, descending=False)
        self._n_classes = n_classes
        self.reset()

    def reset(self):
        self.correct = 0.0
        if self._n_classes:
            self.class_correct = None
            self.class_total = None
        super(RunningAccuracy, self).reset()

    def update(self, pred, gt):
        pred_shape = pred.shape[-2:]
        gt_shape = gt.shape[-2:]
        if len(gt_shape) > 1:
            if pred_shape[0] != gt_shape[0] or pred_shape[1] != gt_shape[1]:
                pred = F.interpolate(pred, size=gt_shape, mode='bilinear', align_corners=True)

        predictions = pred.max(dim=1)[1].squeeze()
        gt = gt.squeeze()
        correct = (predictions == gt)
        mask = gt != self.null_target
        self.correct += correct[mask].sum().item()
        self.num_updates += mask.sum().item()
        if self._n_classes:
            if self.class_correct is None:
                self.class_correct = torch.zeros(self._n_classes, device=pred.device)
            if self.class_total is None:
                self.class_total = torch.zeros(self._n_classes, device=pred.device)
            for label in range(self._n_classes):
                self.class_correct[label] += correct[gt == label].sum()
                self.class_total[label] += (gt == label).sum()

    def compute_results(self):
        acc = self.correct / self.num_updates * 100
        self.results['micro_acc'] = acc
        if self._n_classes:
            per_class_acc = torch.zeros(self._n_classes)
            for i, c, t in zip(range(self._n_classes), self.class_correct, self.class_total):
                per_class_acc[i] = c / t * 100
            self.results['per_class_acc'] = per_class_acc
            self.results['macro_acc'] = per_class_acc.sum().item() / self._n_classes
        self.past_results.append(self.get_value())

    def report_result(self, value):
        self.results['micro_acc'] = value
        self.past_results.append(self.get_value())

    def __str__(self):
        out = "%s on %d images:" % ("Accuracy", self.num_updates)
        out += "\n  Micro accuracy: %.3f %% " % self.results['micro_acc']
        if self._n_classes:
            if 'per_class_acc' in self.results:
                out += "\n" + class_metric_to_string(
                    self.results['per_class_acc'], self.class_total)
            if 'macro_acc' in self.results:
                out += "\n  Macro accuracy: %.3f %%" % self.results['macro_acc']
        return out

    def get_value(self):
        return self.results['micro_acc']


class RunningError(RunningAccuracy):
    def __init__(self, null_target, n_classes=None):
        super(RunningError, self).__init__(null_target, n_classes)
        self.descending = True

    def compute_results(self):
        err = 100 - self.correct / self.num_updates * 100
        self.results['micro_err'] = err
        if self._n_classes:
            per_class_err = np.zeros(self._n_classes)
            for i, c, t in zip(range(self._n_classes), self.class_correct, self.class_total):
                per_class_err[i] = 100 - c / t * 100
            self.results['per_class_err'] = per_class_err
            self.results['macro_err'] = per_class_err.sum() / self._n_classes
        self.past_results.append(self.get_value())

    def __str__(self):
        out = "%s on %d images:" % ("Top-1 error", self.num_updates)
        out += "\n  Micro top-1 error: %.3f %% " % self.results['micro_err']
        if self._n_classes:
            out += "\n" + class_metric_to_string(self.results['per_class_err'], self.class_total)
            out += "\n  Macro top-1 error: %.3f %%" % self.results['macro_err']
        return out

    def get_value(self):
        return self.results['micro_err']


class RunningIOU(RunningMetric):
    def __init__(self, null_target, n_classes, confusion_matrix=None, background_class=None):
        super(RunningIOU, self).__init__(null_target, descending=False)
        if confusion_matrix is None:
            self.confusion_matrix = RunningConfusionMatrix(null_target, n_classes)
        else:
            self.confusion_matrix = confusion_matrix
        self.reset()
        self.background_class = background_class

    def reset(self):
        super(RunningIOU, self).reset()
        self.confusion_matrix.reset()

    def update(self, pred, gt):
        pred, gt = super().update(pred, gt)
        pred_shape = pred.shape[-2:]
        gt_shape = gt.shape[-2:]
        if len(gt_shape) > 1:
            if pred_shape[0] != gt_shape[0] or pred_shape[1] != gt_shape[1]:
                pred = F.interpolate(pred, size=gt_shape, mode='bilinear', align_corners=True)
        self.confusion_matrix.update(pred, gt)
        self.num_updates = self.confusion_matrix.num_updates

    def compute_results(self):
        confusion_matrix = self.confusion_matrix.get_value()
        tp = torch.diag(confusion_matrix)
        per_class_positives = confusion_matrix.sum(dim=0)
        per_class_total = confusion_matrix.sum(dim=1)
        iou = tp / (per_class_positives + per_class_total - tp)
        self.results["confusion_matrix"] = confusion_matrix
        self.results["class_total"] = per_class_total
        self.results["IOU"] = iou
        if self.background_class is not None:
            iou = torch.cat((iou[:self.background_class], iou[self.background_class+1:]))
            tp = torch.cat((tp[:self.background_class], tp[self.background_class+1:]))
            per_class_total = torch.cat((per_class_total[:self.background_class],
                                         per_class_total[self.background_class+1:]))
            confusion_matrix = confusion_matrix.clone()
            confusion_matrix[:, self.background_class] = 0.
            confusion_matrix[self.background_class, :] = 0.
        self.results["mIOU"] = torch.nanmean(iou).item()
        self.results["micro_acc"] = tp.sum() / confusion_matrix.sum()
        self.results["macro_acc"] = (tp / per_class_total).nanmean()
        self.past_results.append(self.results["mIOU"])

    def __str__(self):
        out = "\n%s on %d images: \n" % ("IOU", self.results["class_total"].sum().item())
        out += class_metric_to_string(self.results["IOU"], self.results["class_total"])
        out += "\n  Mean IOU: %.3f " % self.results["mIOU"]
        out += "\n  Micro accuracy: %.3f %% " % self.results['micro_acc']
        out += "\n  Macro accuracy: %.3f %% " % self.results['macro_acc']
        out += f"\n  Number of updates: {self.num_updates}"
        return out

    def get_value(self):
        return self.results["mIOU"]


class RunningF1(RunningIOU):
    def compute_results(self):
        confusion_matrix = self.confusion_matrix.get_value()
        tp = np.diag(confusion_matrix)
        per_class_positives = confusion_matrix.sum(axis=0)
        per_class_total = confusion_matrix.sum(axis=1)
        f1 = 2 * tp / (per_class_positives + per_class_total)
        self.results["class_total"] = confusion_matrix.sum(axis=1)
        self.results["F1"] = f1
        self.results["mF1"] = np.nanmean(f1)
        self.past_results.append(self.results["mF1"])

    def __str__(self):
        out = "\n%s on %d images: \n" % ("F1", self.results["class_total"].sum())
        out += class_metric_to_string(self.results["F1"], self.results["class_total"]) + "\n"
        out += "\n  Mean F1: %.3f " % self.results["mF1"]
        return out

    def get_value(self):
        return self.results["mF1"]


class RunningConfusionMatrix(RunningMetric):
    def __init__(self, null_target, n_classes):
        super().__init__(null_target, descending=False)
        self._n_classes = n_classes
        self.reset()

    def reset(self):
        super().reset()
        self.confusion_matrix = None

    def _fast_hist(self, pred, gt):
        mask = gt != self.null_target
        hist = torch.bincount(
            self._n_classes * gt[mask].to(dtype=int) +
            pred[mask], minlength=self._n_classes ** 2).reshape(self._n_classes, self._n_classes)
        return hist

    def update(self, pred, gt):
        pred, gt = super().update(pred, gt)
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros(
                (self._n_classes, self._n_classes), device=pred.device)
        _pred = pred.max(dim=1)[1]
        for lp, lt in zip(_pred, gt):
            self.confusion_matrix += self._fast_hist(lp.flatten(), lt.flatten())
        self.num_updates += gt.size(0)

    def compute_results(self):
        confusion_matrix = self.confusion_matrix.get_value()
        self.results["confusion_matrix"] = confusion_matrix

    def __str__(self):
        out = "%s on %d images:" % ("Confusion matrix", self.num_updates)
        out += "\n %s" % self.confusion_matrix
        return out

    def get_value(self):
        return self.confusion_matrix


class JointMetric(abc.ABC):

    def __init__(self, metrics, use_ema=False):
        super().__init__()
        error_msg = "Metrics need to be all ascending or all descending"
        all_descending = all([m.descending for m in metrics.values()])
        all_ascending = all([not m.descending for m in metrics.values()])
        assert all_descending or all_ascending, error_msg
        self._metrics = metrics
        self.values = []
        self.use_ema = use_ema
        self.descending = all_descending

    def set_metrics(self, metrics):
        self._metrics = metrics

    def get_metric_values(self):
        metrics_values = []
        for m in self._metrics.values():
            val = m.get_EMA() if self.use_ema else m.get_value()
            metrics_values.append(val)
        return metrics_values

    def get_metrics_dict(self):
        return {t: (m.get_EMA() if self.use_ema else m.get_value(),
                    m.descending) for t, m in self._metrics.items()}

    @abc.abstractmethod
    def compute(self, epoch):
        pass

    def get_value(self):
        return self.values[-1]

    def is_new_best(self):
        if self.descending:
            return all([self.values[-1] < v for v in self.values[:-1]])
        else:
            return all([self.values[-1] > v for v in self.values[:-1]])


class ArithmeticMean(JointMetric):
    def compute(self, epoch):
        self.values.append(statistics.mean(self.get_metric_values()))
        return self.get_value()

import kornia
import torch
import wandb
from torch.nn import functional as F

from ssl_methods.ssl_module import SSLModule
from losses.losses import MultiLoss, OhemCrossEntropy2D
from utils.logging import log_fixmatch


class FixMatch(SSLModule):
    def __init__(self, null_target, consistency_weight_max=None, warmup_steps=None,
                 confidence_threshold=.95, use_all=False, log_quantities=False, log_samples=None,
                 ohem=False, dense=False):
        """Implementation of FixMatch consistency loss computation.
           https://arxiv.org/abs/2001.07685
        """
        super(FixMatch, self).__init__(null_target, consistency_weight_max, warmup_steps)
        if ohem:
            self.ssl_criterion = OhemCrossEntropy2D(
                reduction='none', ignore_index=null_target)
        else:
            self.ssl_criterion = torch.nn.CrossEntropyLoss(
                reduction='none', ignore_index=null_target)
        self.confidence_threshold = confidence_threshold
        self.use_all = use_all
        self.log_quantities = log_quantities
        self.log_samples = log_samples
        self.dense = dense
        self.loss = 0.

    def forward(self, inputs_s, outputs_w, outputs_s, labels, augmentation_module, loss,
                dataset_class, step=None, num_batches_between_logs=None):
        if not isinstance(labels, dict):
            assert isinstance(outputs_w, torch.Tensor) and isinstance(outputs_s, torch.Tensor) \
                and isinstance(labels, torch.Tensor), \
                "Both outputs and labels must be of the same type: tensors or dicts of tensors"
            outputs_w = {"t0": outputs_w}
            outputs_s = {"t0": outputs_s}
            labels = {"t0": labels}
            assert len(outputs_w["t0"]) == len(outputs_s["t0"]) == len(labels["t0"]), \
                "Outputs from weakly and strongly samples must be as long as the labels."
        else:
            assert isinstance(outputs_w, dict) and isinstance(outputs_w, dict) \
                and isinstance(labels, dict), \
                "Both outputs and labels must be of the same type: tensors or dicts of tensors"
            for k in outputs_w.keys():
                assert len(outputs_w[k]) == len(outputs_s[k]), \
                    "Outputs from weakly and strongly samples must be as long as the labels."

        # Get masks defining unlabeled samples
        unlabeled_masks = {}
        image_masks = {}
        for k in labels.keys():
            batch_size = labels[k].shape[0]
            unlabeled_masks[k] = torch.all(
                (labels[k] == self.null_target).reshape(batch_size, -1), dim=1)
            if self.use_all:
                image_masks[k] = torch.ones(batch_size).to(
                    dtype=torch.bool, device=labels[k].device)
            else:
                # Use only fully unlabeled images
                image_masks[k] = unlabeled_masks[k].clone()
        if wandb.run and self.log_quantities:
            for k in image_masks.keys():
                wandb.log({f"fixmatch/num_images/{k}": image_masks[k].sum()}, step=step)

        if not any([torch.any(m) for m in image_masks.values()]):
            return

        # Compute pseudolabels
        max_confidences = {}
        pseudo_targets = {}
        masks = {}
        with torch.no_grad():
            for k, o_ in outputs_w.items():
                o_ = o_.detach()
                if len(o_.shape) > 2:
                    o_ = F.interpolate(
                        o_, labels[k].shape[-2:], mode='bilinear', align_corners=True)

                # for classification tasks
                confs = F.softmax(o_, dim=1)
                max_confidences[k], pseudo_targets[k] = confs.max(dim=1)
                masks[k] = max_confidences[k].ge(self.confidence_threshold)
                max_confidences[k] = max_confidences[k].unsqueeze(dim=1)
                if len(masks[k].shape) > 2:
                    masks[k] = masks[k] & image_masks[k][:, None, None].expand(
                        -1, *masks[k].shape[-2:])
                else:
                    masks[k] = masks[k] & image_masks[k]

                if wandb.run and self.log_quantities:
                    wandb.log({
                        f"fixmatch/num_labels_over_th_labeled/{k}":
                            masks[k][~unlabeled_masks[k]].sum(),
                        f"fixmatch/avg_confidence_labeled/{k}":
                            max_confidences[k][~unlabeled_masks[k]].mean(),
                        f"fixmatch/num_labels_over_th_unlabeled/{k}":
                            masks[k][unlabeled_masks[k]].sum(),
                        f"fixmatch/avg_confidence_unlabeled/{k}":
                            max_confidences[k][unlabeled_masks[k]].mean()
                    }, step=step)

                pseudo_targets[k][~masks[k]] = -1.

        consistency_weight = self._get_consistency_weight(step)

        if wandb.run and self.log_quantities:
            wandb.log({"fixmatch/consistency_weight": consistency_weight}, step=step)

        for k in outputs_s.keys():
            if len(outputs_s[k].shape) > 2:
                outputs_s[k] = F.interpolate(
                    outputs_s[k], labels[k].shape[-2:], mode='bilinear', align_corners=True)
                pseudo_targets[k] = pseudo_targets[k].unsqueeze(dim=1)

        num_elements = {}
        if augmentation_module is not None and self.dense:
            pseudo_targets = augmentation_module.match(pseudo_targets)
            null_target_masks = augmentation_module.match(
                {k: torch.zeros_like(pseudo_targets[k]) for k in pseudo_targets.keys()},
                data_keys=[kornia.constants.DataKey.MASK])
            for k in null_target_masks.keys():
                num_elements[k] = (null_target_masks[k][image_masks[k]] != self.null_target).sum()

            if wandb.run and (self.log_samples is not None or self.log_quantities):
                max_confidences = augmentation_module.match(
                    max_confidences, data_keys=[kornia.constants.DataKey.MASK])

                for k in pseudo_targets.keys():
                    if k not in ['semantic']:
                        raise NotImplementedError

                    matched_pseudo_targets_ = pseudo_targets[k]
                    matched_pseudo_targets_ = matched_pseudo_targets_.to(dtype=torch.int).cpu()
                    num_matched_valid_targets_labeled = (
                        matched_pseudo_targets_[~unlabeled_masks[k].cpu()] != self.null_target).sum()
                    num_matched_valid_targets_unlabeled = (
                        matched_pseudo_targets_[unlabeled_masks[k].cpu()] != self.null_target).sum()

                    if self.log_samples is not None and (step + 1) % num_batches_between_logs == 0:
                        log_fixmatch(
                            k, matched_pseudo_targets_, max_confidences, inputs_s, outputs_s,
                            dataset_class, self.log_samples, batch_size, unlabeled_masks,
                            image_masks, step
                        )
                    if self.log_quantities:
                        wandb.log({
                            f'fixmatch/num_matched_valid_targets/labeled/{k}':
                                num_matched_valid_targets_labeled,
                            f'fixmatch/num_matched_valid_targets/unlabeled/{k}':
                                num_matched_valid_targets_unlabeled,
                            f'fixmatch/num_possible_targets/{k}':
                                num_elements[k]
                        }, step=step)
        else:
            for k in labels.keys():
                num_elements[k] = image_masks[k].sum()

        for k, out_s in outputs_s.items():
            if len(out_s.shape) > 2:
                pseudo_targets[k] = pseudo_targets[k].to(dtype=torch.long).squeeze(dim=1)

            uloss = self.ssl_criterion(out_s, pseudo_targets[k])
            uloss = uloss.sum() * consistency_weight / num_elements[k]

            if isinstance(loss, MultiLoss):
                loss.add_extra_loss(uloss, 'fixmatch_loss_%s' % k)
            else:
                self.loss += uloss
                loss += uloss

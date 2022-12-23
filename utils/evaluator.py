import copy

import torch
import tqdm

from losses import losses
from utils.logging import log_eval, get_logger, log_inputs_and_outputs
from utils import utils

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, eval_dl, running_metrics, joint_metric=None, summary=None, criteria=None,
                 accelerator=None, log_samples=None, num_logs=10):
        assert all([m is not None for m in running_metrics.values()])
        assert eval_dl is not None and len(running_metrics) > 0
        assert accelerator is not None, 'Accelerator is required!'

        self.eval_dl = accelerator.prepare(eval_dl)
        self.running_metrics = copy.deepcopy(running_metrics)
        for metric in self.running_metrics.values():
            metric.max_updates = len(eval_dl.dataset)
        if joint_metric is not None:
            self.joint_metric = copy.deepcopy(joint_metric)
            self.joint_metric.set_metrics(self.running_metrics)
        else:
            self.joint_metric = None
        self.summary = summary
        self.criteria = criteria
        self.accelerator = accelerator
        self.log_samples = log_samples
        self.num_batches_between_logs = max(1, len(self.eval_dl) // num_logs)

    @torch.inference_mode()
    def evaluate(self, model, step=0, epoch=0, eval_set=None):
        device = self.accelerator.device

        # Evaluation mode - disable dropout etc.
        model.eval()

        seen_batches = 0
        loss = None
        cum_loss = 0.0
        avg_loss = 0.0
        loss_dict = {}
        wandb_images = None

        for batch in tqdm.tqdm(self.eval_dl):
            batch = utils.nested_to(batch, device=device)

            if isinstance(batch, dict):
                inputs = batch['input']
                labels = batch['labels']
            else:
                inputs, labels = batch

            outputs = model(inputs)

            outputs = self.accelerator.gather(outputs)
            labels = self.accelerator.gather(labels)

            if self.accelerator.is_main_process:
                if seen_batches % self.num_batches_between_logs == 0 and self.log_samples:
                    wandb_images = log_inputs_and_outputs(
                        inputs, labels, outputs, step, self.eval_dl.dataset, self.log_samples,
                        'eval/', wandb_images=wandb_images)

                if self.criteria:
                    with self.accelerator.autocast():
                        loss = self.criteria(outputs, labels)
                    cum_loss += loss.item()
                    if isinstance(loss, losses.MultiLoss):
                        for k, v in loss.losses_dict.items():
                            if k in loss_dict:
                                loss_dict[k] += v
                            else:
                                loss_dict[k] = v

                for t in self.running_metrics.keys():
                    if isinstance(outputs, dict) and isinstance(labels, dict):
                        self.running_metrics[t].update(outputs[t], labels[t])
                    else:
                        self.running_metrics[t].update(outputs, labels)

            seen_batches += 1

        if self.accelerator.is_main_process:
            if self.criteria:
                avg_loss = cum_loss / seen_batches
                if isinstance(loss, losses.MultiLoss):
                    logger.info('Losses on evaluation:')
                    for k in loss_dict.keys():
                        loss_dict[k] /= seen_batches
                        logger.info('  %s: %.5f' % (k, loss_dict[k]))
                logger.info('Mean loss on evaluation: %.5f' % avg_loss)

            for t, m in self.running_metrics.items():
                m.compute_results()
                logger.info('[%s] %s' % (t, m))
                m.reset()

            if self.joint_metric:
                global_metric = self.joint_metric.compute(epoch)
                logger.info(
                    f'Joint metric{" EMA" if self.joint_metric.use_ema else ""} value: {global_metric:.3f}')
            else:
                global_metric = None

            log_eval(
                avg_loss, loss_dict, self.running_metrics, global_metric, step, epoch, self.summary,
                eval_set=eval_set, wandb_images=wandb_images)

        return self.joint_metric

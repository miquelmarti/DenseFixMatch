import copy
import math

import torch

from losses import losses
from utils import utils
from utils.logging import log_best, log_stats, log_learning_rate, log_train_metrics, get_logger, \
    log_inputs_and_outputs


logger = get_logger(__name__)


class Trainer:
    def __init__(self, model, train_dl, optimizer, criteria, run_dir, ckpt_manager,
                 summary=None, show_progress=True, num_logs=10,
                 ema_decay=0., running_metrics=None, lr_scheduler=None,
                 accelerator=None, batch_transform=None, log_samples=None):
        assert accelerator is not None, 'Accelerator is required!'
        self.ckpt_manager = ckpt_manager

        self.ema_decay = ema_decay
        ema = self._create_ema(model) if ema_decay > 0. else None
        self.model, self.optimizer, self.ema = accelerator.prepare(model, optimizer, ema)
        if self.ema is not None:
            self.ema.eval()

        self.train_dl = train_dl
        self.dataset = self.train_dl.dataset.dataset if isinstance(
            self.train_dl.dataset, torch.utils.data.Subset) else self.train_dl.dataset
        self.batch_transform = batch_transform

        self.criteria = criteria
        self.lr_scheduler = lr_scheduler

        self.run_dir = run_dir
        self.show_progress = show_progress
        self.summary = summary
        self.log_samples = log_samples

        if running_metrics is not None:
            assert isinstance(running_metrics, dict), "Running metrics must be a dict."
        self.running_metrics = running_metrics

        self.num_batches = len(train_dl)
        self.num_batches_between_logs = self.num_batches // num_logs \
            if self.num_batches > num_logs else 1

        self.accelerator = accelerator
        self.device = accelerator.device

    def _step(self, step, batch):
        batch = utils.nested_to(batch, device=self.device)

        if self.batch_transform is not None:
            batch = self.batch_transform(*batch)

        if isinstance(batch, dict):
            inputs = batch['input']
            labels = batch['labels']
        else:
            inputs, labels = batch

        # zero the parameter gradients
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        with self.accelerator.autocast():
            loss = self.criteria(outputs, labels)

        self._update_running_metrics(outputs, labels)

        self._log_inputs_and_outputs(inputs, labels, outputs, step)

        self._backward(loss)

        self.optimizer.step()

        return loss

    def train_epoch(self, epoch):

        # Train mode - enable dropout, etc.
        self.model.train()

        if self.show_progress and self.accelerator.is_local_main_process:
            self.running_loss = 0.
            self.running_loss_dict = {}
            self.seen_batches = 0

        if self.lr_scheduler is not None and self.accelerator.is_local_main_process:
            log_learning_rate(self.lr_scheduler.optimizer.param_groups, epoch * self.num_batches,
                              epoch, self.summary)

        for i, batch in enumerate(self.train_dl, start=0):
            step = epoch * self.num_batches + i
            loss = self._step(step, batch)

            if self.ema_decay > 0.:
                self._update_ema(step)
            if self.show_progress and self.accelerator.is_local_main_process:
                self._compute_and_show_progress(i, step, epoch, loss)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            if self.accelerator.is_local_main_process:
                logger.info("Stepping learning rate. New learning rate: "
                            f"{self.lr_scheduler.optimizer.param_groups[0]['lr']:.5f}")

        if self.running_metrics is not None and self.accelerator.is_local_main_process:
            for t, m in self.running_metrics.items():
                m.compute_results()
                logger.info('[%s] %s' % (t, m))
                m.reset()

            log_train_metrics(self.running_metrics, step, epoch, self.summary)

        return step

    def train(self, num_epochs, start_epoch=0, evaluator=None, patience_epochs=None,
              track_best=True, evaluate_on_ema=False, eval_epoch=1):
        self.num_epochs = num_epochs
        best_epoch = 0
        new_best = False
        global_metric = None
        if evaluate_on_ema:
            assert self.ema_decay > 0., 'Evaluation on EMA weights requires updates.'
        # If evaluator is a dict, assume two entries: val and test
        if isinstance(evaluator, dict):
            val_evaluator = evaluator["val"]
            test_evaluator = evaluator["test"]

        done_training = torch.zeros(1, dtype=torch.bool, device=self.accelerator.device)
        for epoch in range(start_epoch, num_epochs):

            step = self.train_epoch(epoch)

            self.accelerator.wait_for_everyone()
            do_evaluation = (epoch+1) % eval_epoch == 0 or epoch == num_epochs - 1
            evaluation_model = self.get_evaluation_model(evaluate_on_ema)
            if evaluator is not None and do_evaluation:
                if isinstance(evaluator, dict):
                    joint_metric = val_evaluator.evaluate(
                        evaluation_model, step, epoch, eval_set='val')
                    test_evaluator.evaluate(evaluation_model, step, epoch, eval_set='test')
                else:
                    joint_metric = evaluator.evaluate(evaluation_model, step, epoch)

                if self.accelerator.is_main_process:
                    global_metric = joint_metric.get_value()
                    if track_best:
                        new_best = joint_metric.is_new_best()
                        if new_best:
                            best_epoch = epoch
                            best_metrics = joint_metric.get_metrics_dict()
                            log_best(
                                global_metric, best_metrics, best_epoch, step, self.summary, logger)
                        elif patience_epochs is not None:
                            if epoch > best_epoch + patience_epochs:
                                logger.info(
                                    "Target metric not improving after %d epochs, stopping run...",
                                    patience_epochs)
                                done_training = ~ done_training
            else:
                new_best = False
                global_metric = None

            self.accelerator.wait_for_everyone()
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(done_training, src=0)
            if done_training.item() is True:
                logger.info("Done training...")
                break

            self._save_checkpoint(new_best, epoch, global_metric)

    def _backward(self, loss):
        if isinstance(loss, losses.MultiLoss):
            self.accelerator.backward(loss.tensor())
        else:
            self.accelerator.backward(loss)

    def _compute_and_show_progress(self, i, step, epoch, loss):
        self.running_loss += loss.item() if not math.isnan(loss.item()) else 0.
        if isinstance(loss, losses.MultiLoss):
            for k, v in loss.losses_dict.items():
                v = v if not math.isnan(v) else 0.
                if k in self.running_loss_dict:
                    self.running_loss_dict[k] += v
                else:
                    self.running_loss_dict[k] = v
        self.seen_batches += 1

        if (i+1) % self.num_batches_between_logs == 0:
            log_stats(
                self.running_loss, self.running_loss_dict, self.seen_batches, step,
                epoch, i, self.summary, logger)
            self.running_loss = 0.0
            self.running_loss_dict = {}
            self.seen_batches = 0

    def _log_inputs_and_outputs(self, inputs, labels, outputs, step):
        if self.accelerator.is_local_main_process and self.log_samples is not None and \
                (step + 1) % self.num_batches_between_logs == 0:
            log_inputs_and_outputs(inputs, labels, outputs, step, self.dataset, self.log_samples)

    def _create_ema(self, model):
        ema = copy.deepcopy(model)
        if 'ema_model_state_dict' in self.ckpt_manager._tmp_model_state_dicts:
            self.ckpt_manager.load_model_state_from_key(ema, 'ema_model_state_dict')
        return ema

    def _update_ema(self, step):
        assert self.ema is not None
        utils.update_ema(self.model, self.ema, self.ema_decay, step == 0)

    def _get_ema_model(self):
        return self.ema

    def _get_model(self):
        return self.model

    def get_evaluation_model(self, evaluate_on_ema):
        if evaluate_on_ema:
            eval_model = self._get_ema_model()
        else:
            eval_model = self._get_model()
        return self.accelerator.unwrap_model(eval_model)

    def _update_running_metrics(self, outputs, labels):
        if self.running_metrics is not None:
            outputs = self.accelerator.gather(outputs)
            labels = self.accelerator.gather(labels)
            with torch.inference_mode():
                for t in self.running_metrics.keys():
                    if isinstance(outputs, dict) and isinstance(labels, dict):
                        self.running_metrics[t].update(outputs[t], labels[t])
                    else:
                        self.running_metrics[t].update(outputs, labels)

    def _save_checkpoint(self, new_best, epoch, metric):
        if self.ckpt_manager and self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self._get_model())
            ema_model = self._get_ema_model()
            if ema_model:
                model = {
                    '': model,
                    'ema': self.accelerator.unwrap_model(ema_model)
                }
            self.ckpt_manager.checkpoint(
                model, new_best, epoch, metric,
                optimizer=self.optimizer,
                scaler=self.accelerator.scaler)


class FixMatchTrainer(Trainer):
    def __init__(self, model, train_dl, optimizer, criteria, run_dir, ckpt_manager,
                 summary=None, show_progress=True, num_logs=10, ema_decay=0,
                 running_metrics=None, fixmatch_module=None, lr_scheduler=None, accelerator=None,
                 batch_transform=None, log_samples=None, loss_on_strong=False,
                 loss_on_weak_and_strong=False, pseudotargets_on_ema=False):
        super().__init__(model, train_dl, optimizer, criteria, run_dir, ckpt_manager,
                         summary, show_progress, num_logs, ema_decay, running_metrics,
                         lr_scheduler, accelerator, batch_transform, log_samples)
        self.fixmatch_module = fixmatch_module
        self.loss_on_strong = loss_on_strong
        self.loss_on_weak_and_strong = loss_on_weak_and_strong
        self.pseudotargets_on_ema = pseudotargets_on_ema

    def _step(self, step, batch):
        batch = utils.nested_to(batch, device=self.device)

        if self.batch_transform is not None:
            batch = self.batch_transform(*batch)

        if isinstance(batch, dict):
            inputs = batch['input']
            labels = batch['labels']
        else:
            inputs, labels = batch

        # zero the parameter gradients
        self.optimizer.zero_grad()

        outputs = self.model(torch.cat((inputs['weak'], inputs['strong'])))
        if isinstance(outputs, dict):
            outputs_w = {}
            outputs_s = {}
            for k, out in outputs.items():
                outputs_w[k], outputs_s[k] = out.chunk(2)
        else:
            outputs_w, outputs_s = outputs.chunk(2)

        # compute supervised losses from outputs on weakly and/or strongly augmented samples
        with self.accelerator.autocast():
            if self.loss_on_strong:
                loss = self.criteria(outputs_s, labels['strong'])
            else:
                loss = self.criteria(outputs_w, labels['weak'])
            if self.loss_on_weak_and_strong:
                self.criteria.extra_forward(outputs_s, labels['strong'], 'strong')

        if self.loss_on_strong:
            self._update_running_metrics(outputs_s, labels['strong'])
        else:
            self._update_running_metrics(outputs_w, labels['weak'])

        self._log_inputs_and_outputs(
            inputs, labels, {'weak': outputs_w, 'strong': outputs_s}, step)

        with self.accelerator.autocast():
            if self.pseudotargets_on_ema:
                with torch.inference_mode():
                    outputs_w = self._get_ema_model()(inputs['weak'])
            self.fixmatch_module(inputs['strong'], outputs_w, outputs_s, labels['weak'],
                                 self.batch_transform, loss, self.dataset, step,
                                 self.num_batches_between_logs)

        self._backward(loss)
        self.optimizer.step()

        return loss

    def _compute_and_show_progress(self, i, step, epoch, loss):
        self.running_loss += loss.item() if not math.isnan(loss.item()) else 0.
        if isinstance(loss, losses.MultiLoss):
            for k, v in loss.losses_dict.items():
                v = v if not math.isnan(v) else 0.
                if k in self.running_loss_dict:
                    self.running_loss_dict[k] += v
                else:
                    self.running_loss_dict[k] = v
        else:
            v = self.fixmatch_module.loss if not math.isnan(self.fixmatch_module.loss) else 0.
            self.running_loss_dict['fixmatch_loss'] = v
        self.seen_batches += 1

        if (i+1) % self.num_batches_between_logs == 0:
            log_stats(
                self.running_loss, self.running_loss_dict, self.seen_batches, step,
                epoch, i, self.summary, logger)
            self.running_loss = 0.0
            self.running_loss_dict = {}
            self.fixmatch_module.loss = 0.0
            self.seen_batches = 0

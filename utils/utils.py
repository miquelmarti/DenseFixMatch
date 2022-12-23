import datetime
import os
import random
import subprocess
from contextlib import contextmanager

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from ssl_methods.fixmatch import FixMatch
from utils.trainer import FixMatchTrainer, Trainer
from utils.logging import get_logger


logger = get_logger(__name__)


def current_time():
    now = datetime.datetime.utcnow()
    return str(now.date()).replace('-', '') + '-' + str(now.time())


@contextmanager
def repository_directory():
    '''Context manager to switch to root of repository containing this file.

    [!] It is currently hardcoded given relative position of this file in the repository.
    '''
    prev_cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def log_code_version():
    try:
        branch_name = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8')
        revision_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8')
        git_diff = subprocess.check_output(['git', 'diff']).decode('utf-8')

        logger.info("-" * 20)
        logger.info("Code version information:")
        logger.info("-" * 20)
        logger.info("Branch: %s", branch_name)
        logger.info("Revision hash: %s", revision_hash)
        logger.info("-" * 20)
        logger.info(f"Diff from previous commit: \n {git_diff}")
        logger.info("-" * 20)
    except Exception:
        logger.info("Not in a git repository. Skipping...")


def set_random_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    logger.info('Set random, numpy and pytorch random seeds to %d', seed)


def _apply_no_decay(grouped_named_parameters, no_decay_layers=None):
    if no_decay_layers is None:
        no_decay_layers = ['bn', 'bias']
    new_grouped_named_parameters = []
    for param_group in grouped_named_parameters:
        decay_group = param_group.copy()
        decay_group['params'] = [(n, p) for n, p in param_group['params'] if not any(
            nd in n for nd in no_decay_layers)]
        no_decay_group = param_group.copy()
        no_decay_group['params'] = [(n, p) for n, p in param_group['params'] if any(
            nd in n for nd in no_decay_layers)]
        no_decay_group['weight_decay'] = 0.
        new_grouped_named_parameters.append(decay_group)
        new_grouped_named_parameters.append(no_decay_group)
    return new_grouped_named_parameters


def get_model_parameters(model, optim_cfg, no_decay=True):
    grouped_named_parameters = [{'params': [(n, p) for n, p in model.named_parameters()]}]

    if no_decay and 'weight_decay' in optim_cfg:
        grouped_named_parameters = _apply_no_decay(grouped_named_parameters)

    # remove names from params group
    for param_group in grouped_named_parameters:
        param_group['params'] = [p for _, p in param_group['params']]
    return grouped_named_parameters


def get_hydra_run_dir():
    return HydraConfig.get().run.dir


def get_trainer(cfg, model, train_dl, optimizer, criteria, run_dir, lr_scheduler,
                ckpt_manager, ssl_method=None, summary=None, show_progress=True, num_logs=10,
                running_metrics=None, accelerator=None, batch_transform=None):
    use_fixmatch = isinstance(ssl_method, FixMatch)
    ema_decay = cfg.training.ema_decay if 'ema_decay' in cfg.training else 0.
    num_logs = cfg.training.num_logs if 'num_logs' in cfg.training else num_logs
    trainer_class = FixMatchTrainer if use_fixmatch else Trainer
    log_samples = cfg.training.log_samples if 'log_samples' in cfg.training else None

    kwargs = {}
    if use_fixmatch:
        kwargs['loss_on_strong'] = cfg.training.get('loss_on_strong', False)
        kwargs['loss_on_weak_and_strong'] = cfg.training.get('loss_on_weak_and_strong', False)
        assert not (kwargs['loss_on_strong'] and kwargs['loss_on_weak_and_strong'])
        kwargs['pseudotargets_on_ema'] = cfg.training.get('pseudotargets_on_ema', False)
        kwargs['fixmatch_module'] = ssl_method

    return trainer_class(model, train_dl, optimizer, criteria, run_dir, ckpt_manager,
                         summary, show_progress, num_logs, ema_decay=ema_decay,
                         running_metrics=running_metrics, lr_scheduler=lr_scheduler,
                         accelerator=accelerator, batch_transform=batch_transform,
                         log_samples=log_samples, **kwargs)


def get_evaluator(cfg, eval_dl, metrics, joint_metric, summary, accelerator, criteria=None):
    if eval_dl is None:
        return None
    args = dict(eval_dl=eval_dl, running_metrics=metrics, joint_metric=joint_metric,
                summary=summary, accelerator=accelerator, criteria=criteria)
    return hydra.utils.instantiate(cfg.evaluation.evaluator, **args)


def get_device_from_model(model):
    devices = [p.device for p in model.parameters()]
    assert all([devices[0] == d for d in devices])
    return devices[0]


# EMA weights update from https://github.com/kekmodel/FixMatch-pytorch
# License notice:
# MIT License
# Copyright (c) 2019 Jungdae Kim, Qing Yu
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
@torch.no_grad()
def update_ema(model, ema, ema_decay, first_step=False):
    msd = model.state_dict()
    esd = ema.state_dict()
    for k in esd.keys():
        model_v = msd[k].detach()
        ema_v = esd[k]
        if first_step:
            esd[k].copy_(model_v)
        else:
            esd[k].copy_(ema_v * ema_decay + (1. - ema_decay) * model_v)


def use_wandb_artifact(cfg):
    if 'checkpoint' not in cfg:
        return
    elif 'path' not in cfg.checkpoint:
        return
    elif cfg.checkpoint.type != 'wandb_artifact':
        return
    if wandb.run is None:
        raise ValueError('Using W&B artifacts requires an active W&B run.')
    # path is a wandb path to the run that created the checkpoint
    run_path = cfg.checkpoint.path.split("/")
    if len(run_path) == 4:
        run_path, filename = run_path[:-1], run_path[-1]
    else:
        filename = None
    run_path[-1] = 'ckpts-' + run_path[-1] + ':latest'
    artifact = wandb.run.use_artifact('/'.join(run_path), type='checkpoint')
    if filename:
        file = artifact.get_path(filename).download()
    else:
        file = artifact.file()
    logger.info(f"Using W&B artifact {run_path} in {file}.")
    with open_dict(cfg):
        cfg.checkpoint.path = file
        if artifact.metadata and cfg.checkpoint.load_metadata:
            if 'seed' in artifact.metadata:
                cfg.seed = artifact.metadata['seed']
            if 'dataset' in artifact.metadata:
                cfg.dataset.seed = artifact.metadata['dataset']['seed']


class PolynomialLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, gamma, max_iter, last_epoch) -> None:
        def lr_lambda(x):
            return (1 - x / max_iter) ** gamma
        super().__init__(optimizer, lr_lambda, last_epoch)


def nested_to(input, **kwargs):
    "Apply 'to' recursively over lists or dicts of tensors or nn.Modules"
    if isinstance(input, dict):
        output = {}
        for k, v in input.items():
            output[k] = nested_to(v, **kwargs)
    elif isinstance(input, (list, tuple)):
        output = []
        for v in input:
            output.append(nested_to(v, **kwargs))
    else:
        output = input.to(**kwargs)
    return output

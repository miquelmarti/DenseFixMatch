from collections.abc import MutableSequence

import hydra
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

import dataloading
import losses
import metrics
import models
import ssl_methods
from utils import utils
from utils.ckpt_manager import CheckpointManager
from utils.evaluator import Evaluator
from utils.logging import get_logger

logger = get_logger('MAIN')


@hydra.main(config_path='../cfg', config_name='cifar10', version_base='1.1')
def train_and_evaluate(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        mixed_precision='fp16' if cfg.system.fp16 else 'no',
        cpu=cfg.system.cpu,
        dispatch_batches=True
    )
    # dispatch batches is currently required for the explicit setting batch sampler, others could work

    if accelerator.is_local_main_process:
        logger.info(f"Starting run. Output dir: {utils.get_hydra_run_dir()}")
        logger.info(f"Run config summary:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    with utils.repository_directory():
        # wandb requires current directory to be script location
        if 'wandb' in cfg and accelerator.is_main_process:
            wandb.init(name=cfg.wandb.run_name if 'run_name' in cfg.wandb else None,
                       config=OmegaConf.to_container(cfg, resolve=True),
                       project=cfg.wandb.project_name if 'project_name' in cfg.wandb else None,
                       entity=cfg.wandb.entity if 'entity' in cfg.wandb else None,
                       group=cfg.wandb.experiment if 'experiment' in cfg.wandb else None,
                       tags=cfg.wandb.tags if 'tags' in cfg.wandb else None,
                       dir=utils.get_hydra_run_dir())
        if accelerator.is_local_main_process:
            utils.log_code_version()
    utils.use_wandb_artifact(cfg)

    with accelerator.main_process_first():
        dataloaders = dataloading.get_dataloaders(cfg)

    if isinstance(dataloaders['train_dl'], tuple):
        train_dl, batch_transform = dataloaders['train_dl']
        if batch_transform is not None:
            batch_transform = batch_transform.to(accelerator.device)
    else:
        train_dl = dataloaders['train_dl']
        batch_transform = None
    train_dl = accelerator.prepare(train_dl)

    if 'evaluation' in cfg:
        if isinstance(cfg.evaluation.set, MutableSequence):
            eval_dl = {eval_set: dataloaders[eval_set + '_dl'] for eval_set in cfg.evaluation.set}
        elif cfg.evaluation.set in ['traindev', 'val', 'test']:
            eval_dl = dataloaders[cfg.evaluation.set + '_dl']
        else:
            raise ValueError(f"Wrong evaluation set {cfg.evaluation.set}")
        logger.info(f"Evaluating on {cfg.evaluation.set} set.")
    else:
        eval_dl = None

    # setting seed again for possibly different runs depending on initialization, etc.
    utils.set_random_seeds(cfg.seed)

    model = models.get_model(cfg)

    optimizer = hydra.utils.instantiate(
        cfg.training.optimizer,
        params=utils.get_model_parameters(model, cfg.training.optimizer),
        _convert_='partial')
    criteria = losses.get_criteria(cfg)

    ssl_method = ssl_methods.get_ssl_method(cfg, len(train_dl))

    metrics_dict, joint_metric = metrics.get_metrics(cfg)

    ckpt_manager = CheckpointManager(utils.get_hydra_run_dir(), cfg.ckpt_manager)
    start_epoch = ckpt_manager.configure_checkpoint_loading(
        cfg, model, optimizer, accelerator.scaler)
    lr_scheduler = hydra.utils.instantiate(
        cfg.training.lr_scheduler, optimizer=optimizer,
        last_epoch=start_epoch if start_epoch else -1)

    summary = SummaryWriter(log_dir=wandb.run.dir if wandb.run else utils.get_hydra_run_dir())

    # different seeds in each process to get different data augmentations
    utils.set_random_seeds(cfg.seed + accelerator.process_index)

    trainer = utils.get_trainer(
        cfg, model, train_dl, optimizer, criteria, utils.get_hydra_run_dir(),
        lr_scheduler, ckpt_manager, ssl_method=ssl_method, summary=summary,
        running_metrics=metrics_dict, accelerator=accelerator, batch_transform=batch_transform
    )

    if isinstance(eval_dl, dict):
        evaluator = {
            set: Evaluator(
                eval_dl[set], metrics_dict, joint_metric if set == 'val' else None, summary,
                criteria, accelerator, cfg.evaluation.evaluator.log_samples)
            for set in eval_dl.keys()
        }
    else:
        evaluator = Evaluator(
            eval_dl, metrics_dict, joint_metric, summary, criteria, accelerator,
            cfg.evaluation.evaluator.log_samples)

    trainer.train(
        num_epochs=cfg.training.num_epochs,
        start_epoch=start_epoch,
        evaluator=evaluator,
        track_best=cfg.evaluation.track_best,
        patience_epochs=cfg.evaluation.patience_epochs,
        evaluate_on_ema=cfg.evaluation.use_ema_weights,
        eval_epoch=cfg.evaluation.epoch_interval)

    summary.close()

    if wandb.run and accelerator.is_main_process:
        metadata = OmegaConf.to_container(cfg, resolve=True)
        trainer.ckpt_manager.save_as_wandb_artifact(metadata)


if __name__ == "__main__":
    train_and_evaluate()

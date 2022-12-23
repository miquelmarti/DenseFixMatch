from collections.abc import MutableSequence

import hydra
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

import dataloading
import models
import losses
import metrics
import utils

logger = utils.get_logger('MAIN')


@hydra.main(config_path='../cfg', config_name='cifar10', version_base='1.1')
def evaluate(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        mixed_precision='fp16' if cfg.system.fp16 else 'no',
        cpu=cfg.system.cpu,
        dispatch_batches=True
    )

    assert 'evaluation' in cfg, '[!] Evaluation run requires evaluation configuration.'
    assert 'checkpoint' in cfg, '[!] Evaluation run requires loading a checkpoint.'

    logger.info(f"Starting evaluation run. Output dir: {utils.get_hydra_run_dir()}")
    logger.info(f"Run config summary:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    with utils.repository_directory():
        # wandb requires current directory to be script location
        if 'wandb' in cfg:
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

    if isinstance(cfg.evaluation.set, MutableSequence):
        eval_dl = {eval_set: dataloaders[eval_set + '_dl'] for eval_set in cfg.evaluation.set}
    elif cfg.evaluation.set in ['traindev', 'val', 'test']:
        eval_dl = dataloaders[cfg.evaluation.set + '_dl']
    else:
        raise ValueError(f"Wrong evaluation set {cfg.evaluation.set}")
    logger.info(f"Evaluating on {cfg.evaluation.set} set.")

    criteria = losses.get_criteria(cfg)

    metrics_dict, joint_metric = metrics.get_metrics(cfg)

    summary = SummaryWriter(log_dir=utils.get_hydra_run_dir())

    # setting seed again for possibly different runs depending on initialization, etc.
    utils.set_random_seeds(cfg.seed)

    model = models.get_model(cfg)
    ckpt_manager = utils.CheckpointManager(utils.get_hydra_run_dir(), cfg.ckpt_manager)
    ckpt_manager.configure_checkpoint_loading(cfg, model)
    model = accelerator.prepare(model)

    if isinstance(eval_dl, dict):
        for set in eval_dl.keys():
            evaluator = utils.get_evaluator(
                cfg, eval_dl[set], metrics_dict, joint_metric, summary, criteria=criteria,
                accelerator=accelerator
            )
            evaluator.evaluate(model, eval_set=set)
    else:
        evaluator = utils.get_evaluator(
            cfg, eval_dl, metrics_dict, joint_metric, summary, criteria=criteria,
            accelerator=accelerator
        )
        evaluator.evaluate(model)

    summary.close()


if __name__ == "__main__":
    evaluate()

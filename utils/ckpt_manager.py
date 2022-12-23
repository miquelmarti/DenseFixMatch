import os

import torch
import wandb

from utils import utils
from utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    def __init__(self, run_dir, cfg):
        self.run_dir = run_dir
        self.best_ckpt_paths = []
        self.last_ckpt_paths = []
        self.last_is_best = False
        self.keep_all = cfg.keep_all
        self.keep_all_best = cfg.keep_all_best

        self._tmp_model_state_dicts = {}

    # Takes a model or dict of models to get checkpoints from
    def checkpoint(self, model, new_best, epoch, metric, optimizer, scaler):
        ckpt_path = self._save_checkpoint(epoch, model, optimizer, scaler, new_best, metric)

        if new_best and self.best_ckpt_paths and not self.keep_all and not self.keep_all_best:
            logger.info("Removing previous best checkpoint %s", self.best_ckpt_paths[-1])
            os.remove(self.best_ckpt_paths.pop())
        if self.last_ckpt_paths and not self.keep_all:
            last_ckpt_path = self.last_ckpt_paths.pop()
            if not self.last_is_best:
                logger.info("Removing previous last checkpoint %s", last_ckpt_path)
                os.remove(last_ckpt_path)

        self.last_is_best = new_best
        if new_best:
            self.best_ckpt_paths.append(ckpt_path)
        self.last_ckpt_paths.append(ckpt_path)

    def save_as_wandb_artifact(self, metadata):
        ckpts_artifact = wandb.Artifact(
            'ckpts-'+str(wandb.run.id), type='checkpoint', metadata=metadata)
        if self.best_ckpt_paths:
            for ckpt_path in self.best_ckpt_paths:
                wandb.save(ckpt_path)
                ckpts_artifact.add_file(ckpt_path, name=os.path.basename(ckpt_path))
            last_ckpt_paths = [ckpt_path for ckpt_path in self.last_ckpt_paths
                               if ckpt_path not in self.best_ckpt_paths]
        else:
            last_ckpt_paths = self.last_ckpt_paths
        for ckpt_path in last_ckpt_paths:
            wandb.save(ckpt_path)
            ckpts_artifact.add_file(ckpt_path, name=os.path.basename(ckpt_path))
        wandb.log_artifact(ckpts_artifact)

    def _save_checkpoint(self, epoch, model, optimizer, scaler, best=False, metric=None):
        filename = 'ckpt-%s-epoch:%d' % (utils.current_time(), epoch)
        if metric:
            filename += '-metric:%.3f' % (metric)
        if best:
            filename += '-best'
        path = os.path.join(self.run_dir, filename)

        checkpoint_dict = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_value': metric
        }

        if isinstance(model, dict):
            for name, m in model.items():
                key = 'model_state_dict'
                if len(name) > 0:
                    key = f'{name}_{key}'
                checkpoint_dict[key] = m.state_dict()
        else:
            checkpoint_dict['model_state_dict'] = model.state_dict()

        if scaler:
            checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

        torch.save(checkpoint_dict, path)
        logger.info('Saved model and optimizer states to %s', path)

        return path

    @staticmethod
    def _filter_model_state_dict(model_state_dict, filter):
        assert isinstance(filter, str)
        model_state_dict = {
            k.replace(filter, ''): v for k, v in model_state_dict.items() if filter in k}
        return model_state_dict

    def load_model_state_from_key(self, model, key):
        if key not in self._tmp_model_state_dicts:
            raise ValueError("Model state key not in ckpt")

        model_state_dict = self._tmp_model_state_dicts.pop(key)
        incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
        logger.info("Loaded model state from checkpoint.")

        if len(incompatible_keys.missing_keys) > 0:
            logger.warning("Missing keys: %s", str(incompatible_keys.missing_keys))
        if len(incompatible_keys.unexpected_keys) > 0:
            logger.warning("Unexpected keys: %s", str(incompatible_keys.unexpected_keys))

    def load_checkpoint(self, path, model, optimizer=None, scaler=None, device='cpu', filter=None,
                        main_model_key=None):
        ckpt = torch.load(path, map_location=device)

        # get all model_state_dicts
        model_state_keys = [k for k in ckpt.keys() if 'model_state_dict' in k]
        for key in model_state_keys:
            model_state_dict = ckpt.pop(key)
            if filter:
                model_state_dict = self._filter_model_state_dict(model_state_dict, filter)
            self._tmp_model_state_dicts[key] = model_state_dict

        # load for main model, others will be loaded later
        if main_model_key:
            assert main_model_key in model_state_keys
            self.load_model_state_from_key(model, main_model_key)
        else:
            self.load_model_state_from_key(model, 'model_state_dict')

        if optimizer is not None and ckpt.get('optimizer_state_dict'):
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            logger.info('Loaded optimizer state from checkpoint')

        if scaler is not None and ckpt.get('scaler_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            logger.info('Loaded scaler state from checkpoint')

        return ckpt['epoch']

    def configure_checkpoint_loading(self, config, model, optimizer=None, scaler=None):
        if 'checkpoint' not in config:
            return 0
        device = utils.get_device_from_model(model)
        pretrained = config.training.pretrained if 'pretrained' in config.training else False
        ckpt_path = config.checkpoint.path if 'path' in config.checkpoint else None
        if pretrained and ckpt_path:
            logger.warn('Loading a checkpoint overwrites pretrained model weights.')
        if ckpt_path is not None:
            assert os.path.isfile(ckpt_path), f'[!] Checkpoint path {ckpt_path} does not exist.'
            resume = config.checkpoint.resume if 'resume' in config.checkpoint else False
            filter = config.checkpoint.filter if 'filter' in config.checkpoint else None
            ckpt_epoch = self.load_checkpoint(
                ckpt_path, model, optimizer if resume else None, scaler if resume else None,
                device, filter=filter,
                main_model_key=config.checkpoint.key if 'key' in config.checkpoint else None)
            if resume:
                logger.info(f'Resuming from epoch #{ckpt_epoch + 1}')
                return ckpt_epoch + 1
        return 0

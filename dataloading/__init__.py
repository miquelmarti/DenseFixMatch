def get_dataloaders(cfg):
    import os

    from omegaconf import open_dict

    from utils import utils
    from dataloading import cifar, cityscapes, voc
    from dataloading.utils import Augmentation

    try:
        from utils.logging import get_logger
        logger = get_logger(__name__)
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)


    # setting seed to get always get same dataset split
    utils.set_random_seeds(cfg.dataset.seed)

    dataset_dir = os.path.join(cfg.system.datasets_dir, cfg.dataset.name)

    augmentation_type = Augmentation[cfg.dataset.augmentation.upper()]
    if cfg.get('ssl'):
        if cfg.ssl._target_ == 'ssl_methods.fixmatch.FixMatch':
            if augmentation_type != Augmentation.FIXMATCH:
                logger.warn('Using FixMatch requires FixMatch augmentation pipeline. Overwriting.')
            augmentation_type = Augmentation.FIXMATCH

    with open_dict(cfg):
        cfg.dataset.dataset_dir = dataset_dir
        cfg.dataset.augmentation_type = augmentation_type

    if hasattr(cfg.dataset, 'tasks'):
        tasks = cfg.get('tasks')
        declared_tasks = cfg.dataset.tasks.keys()
        if tasks is not None:
            error_msg = f'One or more tasks in: {tasks} not in dataset declared tasks: ' \
                f'{list(declared_tasks)}'
            assert all([t in declared_tasks for t in tasks]), error_msg
        else:
            tasks = list(declared_tasks)

    if cfg.dataset.name == 'cifar10' or cfg.dataset.name == 'cifar100':
        return cifar.get_dataloaders(cfg.dataset)
    elif cfg.dataset.name == 'cityscapes':
        return cityscapes.get_dataloaders(cfg.dataset, tasks)
    elif cfg.dataset.name == 'voc':
        return voc.get_dataloaders(cfg.dataset, tasks)
    else:
        raise NotImplementedError

import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import kornia.augmentation as K

from dataloading.transforms import BatchRandAugment, BatchTransformFixMatch
from dataloading.utils import (Augmentation, ExplicitSSLBatchSampler, get_train_loader,
                               split_train_val, split_train_val_class_balanced, apply_num_labels,
                               apply_num_labels_class_balanced)

try:
    from utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


_IMAGE_SIZE = (32, 32)


def get_dataloaders(cfg):
    if cfg.use_all_as_unlabeled:
        assert cfg.use_implicit_setting is False, \
            "[!] Cannot use all data as unlabeled in implicit setting."

    if cfg.name == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        dataset_class = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

    if cfg.use_implicit_setting or cfg.bs_unlabeled_factor is None:
        batch_size_unlabeled = None
        logger.info("Using implicit SSL setting.")
    elif cfg.bs_unlabeled_factor is not None:
        batch_size_unlabeled = int(round(cfg.batch_size * cfg.bs_unlabeled_factor))
        if batch_size_unlabeled > 0:
            logger.info("Using explicit SSL setting. Epoch refers to unlabeled epoch.")
            logger.info("Labeled epoch might finish more often than unlabeled epoch.")
        else:
            logger.info("Not using any unlabeled data.")
    else:
        logger.info("Not using any unlabeled data.")

    train_transform, batch_transform = _get_train_transform(cfg.augmentation_type, mean, std)

    eval_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = dataset_class(
        root=cfg.dataset_dir, train=True, download=cfg.download, transform=train_transform)
    val_dataset = dataset_class(
        root=cfg.dataset_dir, train=True, download=cfg.download, transform=eval_transform)
    test_dataset = dataset_class(
        root=cfg.dataset_dir, train=False, download=cfg.download, transform=eval_transform)

    train_indices = list(range(len(train_dataset)))
    if cfg.val_split_size is not None:
        if cfg.balanced:
            train_indices, val_indices = split_train_val_class_balanced(
                train_indices, cfg.val_split_size, range(len(train_dataset.classes)),
                train_dataset.targets, shuffle=cfg.shuffle)
        else:
            train_indices, val_indices = split_train_val(
                train_indices, cfg.val_split_size, shuffle=True)
    else:
        val_indices = None

    logger.info(f'Length training set: {len(train_indices)}')
    if val_indices is not None:
        logger.info(f'Length validation set: {len(val_indices)}')
    logger.info(f'Length test set: {len(test_dataset)}')

    train_labeled_indices = train_indices
    train_unlabeled_indices = None

    # Apply number of labels or label ratio
    if cfg.label_ratio is not None or cfg.num_labels is not None:
        assert not (cfg.label_ratio is not None and cfg.num_labels is not None), \
            '[!] Amount of labels in dataset specified either via label_ratios or num_labels.'

        if cfg.label_ratio is not None:
            error_msg = '[!] label ratio should be in the range (0, 1]'
            assert ((cfg.label_ratio > 0) and (cfg.label_ratio <= 1)), error_msg
            num_labels = int(np.floor(cfg.label_ratio * len(train_indices)))
        else:
            num_labels = cfg.num_labels

        error_msg = "[!] Number of labels must be 0 < num_labels <= len(dataset)"
        assert ((num_labels > 0) and (num_labels <= len(train_indices))), error_msg

        if cfg.balanced:
            train_labeled_indices, train_unlabeled_indices = apply_num_labels_class_balanced(
                num_labels, train_indices, range(len(train_dataset.classes)), train_dataset.targets)
        else:
            train_labeled_indices, train_unlabeled_indices = apply_num_labels(
                num_labels, train_indices)


    logger.info(f'Length labeled training set: {len(train_labeled_indices)}')
    logger.debug(f'Indices: {train_labeled_indices}')
    if train_unlabeled_indices is not None:
        train_dataset.targets = np.array(train_dataset.targets)
        train_dataset.targets[train_unlabeled_indices] = cfg.null_target
        if cfg.use_all_as_unlabeled and batch_size_unlabeled is not None:
            logger.info('Using all samples as unlabeled samples.')
            train_unlabeled_indices = train_indices
        logger.info(f'Length unlabeled training set: {len(train_unlabeled_indices)}')
        logger.debug(f'Indices: {train_unlabeled_indices}')

    # Prepare train dataloader
    if train_unlabeled_indices is not None and batch_size_unlabeled is not None:
        if batch_size_unlabeled > 0:
            train_dl = DataLoader(
                train_dataset, batch_sampler=ExplicitSSLBatchSampler(
                    train_labeled_indices, train_unlabeled_indices, cfg.batch_size,
                    batch_size_unlabeled),
                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                collate_fn=None, persistent_workers=True
            )
        else:
            train_dl = get_train_loader(
                cfg, dataset=Subset(train_dataset, train_labeled_indices),
                num_workers=cfg.num_workers, collate_fn=None, persistent_workers=True
            )
    else:
        train_dl = get_train_loader(
            cfg, Subset(train_dataset, train_indices), collate_fn=None, persistent_workers=True
        )

    # Prepare validation dataloader
    if val_indices is not None:
        val_dl = DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=cfg.eval_batch_size,
            num_workers=cfg.eval_num_workers,
            shuffle=False, pin_memory=cfg.pin_memory)
    else:
        val_dl = None

    # Prepare test dataloader
    test_dl = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.eval_num_workers,
        shuffle=False, pin_memory=cfg.pin_memory)

    return {
        'train_dl': (train_dl, batch_transform),
        'val_dl': val_dl,
        'test_dl': test_dl
    }


def _get_train_transform(augmentation_type, mean, std):
    if augmentation_type == Augmentation.NONE:
        return torchvision.transforms.ToTensor(), None
    elif augmentation_type == Augmentation.STD:
        pre_transform = torchvision.transforms.ToTensor()
        batch_transform = K.AugmentationSequential(
            K.RandomCrop(size=_IMAGE_SIZE, padding=4),
            K.RandomHorizontalFlip(),
            K.Normalize(mean, std),
        )
        return pre_transform, batch_transform
    elif augmentation_type == Augmentation.FIXMATCH:
        pre_transform = torchvision.transforms.ToTensor()
        batch_transform = BatchTransformFixMatch(2, 10, mean, std, use_crops=True,
                                                 image_size=_IMAGE_SIZE, padding=4,
                                                 always_cutout=True)
        return pre_transform, batch_transform
    elif augmentation_type == Augmentation.RANDAUG:
        pre_transform = torchvision.transforms.ToTensor()
        batch_transform = BatchRandAugment(2, 10, mean, std, use_crops=True,
                                           image_size=_IMAGE_SIZE, padding=4,
                                           always_cutout=True)
        return pre_transform, batch_transform
    else:
        raise ValueError(f"Augmentation type {augmentation_type} not supported.")

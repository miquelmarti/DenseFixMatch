import enum

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, SubsetRandomSampler, RandomSampler, \
    SequentialSampler, Sampler

try:
    from utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class Augmentation(enum.IntEnum):
    NONE = 1
    STD = 2
    RANDAUG = 3
    FIXMATCH = 4


class SSLJointDataloader:
    def __init__(self, labeled_loader, unlabeled_loader, num_iterations=None, null_target=-1):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.num_iterations = max(len(labeled_loader), len(unlabeled_loader)) \
            if num_iterations is None else num_iterations
        self.null_target = null_target
        self.dataset = labeled_loader.dataset

    def _concat_loaders_outputs(self, labeled, unlabeled):
        labeled_data, labels = labeled
        unlabeled_data, unlabeled_labels = unlabeled
        data = torch.cat((labeled_data, unlabeled_data))
        labels = torch.cat((labels, self.null_target * torch.ones_like(unlabeled_labels)))
        return data, labels

    def __iter__(self):
        labeled_iterator = iter(self.labeled_loader)
        unlabeled_iterator = iter(self.unlabeled_loader)
        for _ in range(self.num_iterations):
            try:
                labeled_batch = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(self.labeled_loader)
                labeled_batch = next(labeled_iterator)
            try:
                unlabeled_batch = next(unlabeled_iterator)
            except StopIteration:
                unlabeled_iterator = iter(self.unlabeled_loader)
                unlabeled_batch = next(unlabeled_iterator)
            yield self._concat_loaders_outputs(labeled_batch, unlabeled_batch)

    def __len__(self):
        return self.num_iterations


class ExplicitSSLBatchSampler(Sampler):
    def __init__(self, labeled_indices, unlabeled_indices, batch_size_labeled,
                 batch_size_unlabeled, drop_last=False, fill_batches=True):
        self.batch_size_labeled = batch_size_labeled
        self.batch_size_unlabeled = batch_size_unlabeled
        self.batch_size = batch_size_labeled + batch_size_unlabeled  # accelerate.BatchSamplerShard
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.labeled_sampler = BatchSampler(
            SubsetRandomSampler(labeled_indices), batch_size_labeled, drop_last)
        self.unlabeled_sampler = BatchSampler(
            SubsetRandomSampler(unlabeled_indices), batch_size_unlabeled, drop_last)
        self.labeled_is_shorter = len(labeled_indices) < len(unlabeled_indices)
        self.drop_last = drop_last
        self.fill_batches = fill_batches

    def __iter__(self):
        initial_labeled = None
        initial_unlabeled = None
        labeled_iterator = iter(self.labeled_sampler)
        unlabeled_iterator = iter(self.unlabeled_sampler)
        while True:
            try:
                labeled_batch = next(labeled_iterator)
                if not initial_labeled:
                    initial_labeled = labeled_batch
            except StopIteration:
                if self.labeled_is_shorter:
                    labeled_iterator = iter(self.labeled_sampler)
                    logger.debug("Repeating the labeled sampler.")
                    labeled_batch = next(labeled_iterator)
                    initial_labeled = labeled_batch
                else:
                    break
            try:
                unlabeled_batch = next(unlabeled_iterator)
                if not initial_unlabeled:
                    initial_unlabeled = unlabeled_batch
            except StopIteration:
                if not self.labeled_is_shorter:
                    unlabeled_iterator = iter(self.unlabeled_sampler)
                    logger.debug("Repeating the unlabeled sampler.")
                    unlabeled_batch = next(unlabeled_iterator)
                    initial_unlabeled = unlabeled_batch
                else:
                    break
            if len(labeled_batch) < self.batch_size_labeled and self.fill_batches:
                labeled_batch += initial_labeled[:self.batch_size_labeled-len(labeled_batch)]
            if len(unlabeled_batch) < self.batch_size_unlabeled and self.fill_batches:
                unlabeled_batch += initial_unlabeled[
                    :self.batch_size_unlabeled-len(unlabeled_batch)]
            logger.debug(f"Batch indices: {labeled_batch + unlabeled_batch}")
            yield labeled_batch + unlabeled_batch

    def __len__(self):
        num_iter = max(len(self.labeled_sampler), len(self.unlabeled_sampler))
        if self.drop_last:
            if self.labeled_is_shorter:
                return num_iter - 1 if (num_iter * self.batch_size_unlabeled) % len(
                    self.unlabeled_indices) != 0 else num_iter
            else:
                return num_iter - 1 if (num_iter * self.batch_size_labeled) % len(
                    self.labeled_indices) != 0 else num_iter
        else:
            return num_iter


def _class_balanced_split(original_indices, labels, classes, size_first):
    first_part = []
    second_part = []
    samples_per_class = size_first // len(classes)
    for class_ in classes:
        class_mask = labels[original_indices] == class_
        first_part.extend(original_indices[class_mask][:samples_per_class])
        second_part.extend(original_indices[class_mask][samples_per_class:])
    return first_part, second_part


def split_train_val(train_indices, valid_size, shuffle=True):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    if shuffle:
        np.random.shuffle(train_indices)
    split = int(np.floor(valid_size * len(train_indices)))

    train_indices, val_indices = train_indices[split:], train_indices[:split]

    train_indices = train_indices if len(train_indices) > 0 else None
    val_indices = val_indices if len(val_indices) > 0 else None

    return train_indices, val_indices


def split_train_val_class_balanced(train_indices, valid_size, classes, targets, shuffle=True):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    if shuffle:
        np.random.shuffle(train_indices)
    split = int(np.floor(valid_size * len(train_indices)))

    train_indices, val_indices = _class_balanced_split(
        np.array(train_indices), np.array(targets), classes, len(train_indices) - split)
    logger.info("Using class-balanced training and validation splits.")

    train_indices = train_indices if len(train_indices) > 0 else None
    val_indices = val_indices if len(val_indices) > 0 else None

    return train_indices, val_indices


def apply_num_labels(num_labels, train_indices):
    if num_labels < len(train_indices):
        train_labeled_indices = train_indices[:num_labels]
        train_unlabeled_indices = train_indices[num_labels:]
        return train_labeled_indices, train_unlabeled_indices
    else:
        return train_indices, []


def apply_num_labels_class_balanced(num_labels, train_indices, classes, targets):
    if num_labels < len(train_indices):
        logger.info('Using class-balanced labeled and unlabeled splits.')
        train_labeled_indices, train_unlabeled_indices = _class_balanced_split(
            np.array(train_indices), np.array(targets), classes, num_labels)
        return train_labeled_indices, train_unlabeled_indices
    else:
        return train_indices, None


def get_train_loader(cfg, dataset, num_workers=None, pin_memory=None, **kwargs):
    if cfg.shuffle:
        num_samples = int(round(len(dataset) * cfg.repeat)) if cfg.repeat is not None else None
        sampler = RandomSampler(dataset, num_samples=num_samples)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset=dataset, sampler=sampler, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=cfg.pin_memory if pin_memory is None else pin_memory, **kwargs)


# From https://github.com/charlesCXK/TorchSemiSeg/blob/main/exp.voc/voc8.res50v3%2B.CPS/dataloader.py
# MIT License

# Copyright (c) 2021 Chen XiaoKang

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
def get_class_colors(N):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    class_colors = cmap.tolist()
    return class_colors

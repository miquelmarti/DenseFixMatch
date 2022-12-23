# Extended from torchvision https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import collections
import copy
import os
import yaml
from collections import namedtuple
from xml.etree.ElementTree import Element as ET_Element
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torchvision
from kornia import augmentation as K
from kornia.constants import DataKey
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

from dataloading.utils import Augmentation, split_train_val, apply_num_labels, \
    ExplicitSSLBatchSampler, get_class_colors
from dataloading.transforms import BatchAugment, BatchRandAugment, BatchTransformFixMatch, \
    CropMode, RandomCrop

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

try:
    from utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

DATASET_YEAR_DICT = {
    "2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "base_dir": os.path.join("VOCdevkit", "VOC2012"),
    },
    "2011": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
        "filename": "VOCtrainval_25-May-2011.tar",
        "md5": "6c3384ef61512963050cb5d687e5bf1e",
        "base_dir": os.path.join("TrainVal", "VOCdevkit", "VOC2011"),
    },
    "2010": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        "filename": "VOCtrainval_03-May-2010.tar",
        "md5": "da459979d0c395079b5c75ee67908abb",
        "base_dir": os.path.join("VOCdevkit", "VOC2010"),
    },
    "2009": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
        "filename": "VOCtrainval_11-May-2009.tar",
        "md5": "a3e00b113cfcfebf17e343f59da3caa1",
        "base_dir": os.path.join("VOCdevkit", "VOC2009"),
    },
    "2008": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "2629fa636546599198acfcfbfcf1904a",
        "base_dir": os.path.join("VOCdevkit", "VOC2008"),
    },
    "2007": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "md5": "c52e279531787c972589f7e41ab4ae64",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
    "2007-test": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "filename": "VOCtest_06-Nov-2007.tar",
        "md5": "b6e924de25625d8de591ea690078ad9f",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
}

_target_types2data_keys = {
    'semantic': DataKey.MASK,
    'detection': DataKey.BBOX_XYXY,
}

VocClass = namedtuple('VocClass', ['name', 'id', 'color'])


class VOC(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or
            ``"val"``. If ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its
            target as entry and returns a transformed version.
    """

    semantic_ignore = 255
    class_names = [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]
    num_classes = len(class_names)
    class_colors = get_class_colors(num_classes)

    classes = [
        VocClass(name, i, color) for i, (name, color) in enumerate(zip(class_names, class_colors))
    ]
    trainid2name = {c.id: c.name for c in classes}

    _SPLITS_DIR = "Main"
    _SPLITS_DIR_SEGMENTATION = "Segmentation"

    _TARGET_DIR_SEMANTIC = "SegmentationClass"
    _TARGET_DIR_SEMANTIC_AUGMENTED = "SegmentationClassAug"
    _TARGET_FILE_EXT_SEMANTIC = ".png"

    _TARGET_DIR_DETECTION = "Annotations"
    _TARGET_FILE_EXT_DETECTION = ".xml"

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        transform=None,
        transforms=None,
        null_target=255,
        use_augmented_set=False,
        tasks: Optional[List[str]] = None,
        download: bool = False,
    ):
        super().__init__(
            root, transform=transform if transforms is None else None, transforms=transforms)
        if transforms is not None:
            self.transform = transform

        self.ignore_index = null_target

        valid_tasks = ['semantic']
        self.tasks = tasks if tasks is not None else ['semantic']
        if not isinstance(tasks, collections.abc.MutableSequence):
            self.tasks = [tasks]
        [verify_str_arg(value, "target_type", valid_tasks) for value in self.tasks]

        if year == "2007-test":
            if image_set == "test":
                logger.warn(
                    "Accessing the test image set of the year 2007 with year='2007-test' is "
                    "deprecated since 0.12 and will be removed in 0.14. "
                    "Please use the combination year='2007' and image_set='test' instead."
                )
                year = "2007"
            else:
                raise ValueError(
                    "In the test image set of the year 2007 only image_set='test' is allowed. "
                    "For all other image sets use year='2007' instead."
                )
        self.year = year

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)

        if download:
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to "
                               "download it")

        splits_dir_seg = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR_SEGMENTATION)
        split_f_seg = os.path.join(splits_dir_seg, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f_seg)) as f:
            file_names = [x.strip() for x in f.readlines()]

        if use_augmented_set and image_set == "train":
            target_dir_augmented = os.path.join(voc_root, self._TARGET_DIR_SEMANTIC_AUGMENTED)
            if not os.path.isdir(target_dir_augmented):
                raise RuntimeError("Augmented dataset not found or corrupted."
                                   "Download instructions: https://github.com/zhixuanli/segmentation-paper-reading-notes/blob/master/others/Summary%20of%20the%20semantic%20segmentation%20datasets.md#3-pascal-voc-2012-augmented-with-sbd")

            trainval_aug = [f.strip(".png") for f in os.listdir(target_dir_augmented)]
            splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR_SEGMENTATION)
            split_f = os.path.join(splits_dir, "val.txt")
            with open(os.path.join(split_f)) as f:
                file_names_val_seg = [x.strip() for x in f.readlines()]
            file_names_aug = sorted(
                list(set(trainval_aug) - set(file_names) - set(file_names_val_seg)))
            self.original_indices = list(range(len(file_names)))
            file_names.extend(file_names_aug)
        else:
            self.original_indices = None

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        self.targets = {}
        target_dir_semantic = os.path.join(voc_root,
                                           self._TARGET_DIR_SEMANTIC_AUGMENTED if use_augmented_set
                                           else self._TARGET_DIR_SEMANTIC)
        self.targets['semantic'] = [
            os.path.join(target_dir_semantic, x + self._TARGET_FILE_EXT_SEMANTIC)
            for x in file_names]

        target_dir_detection = os.path.join(voc_root, self._TARGET_DIR_DETECTION)
        self.targets['detection'] = [
            os.path.join(target_dir_detection, x + self._TARGET_FILE_EXT_DETECTION)
            for x in file_names]

    def __len__(self) -> int:
        return len(self.images)

    @property
    def masks(self) -> List[str]:
        return self.targets['semantic']

    @property
    def annotations(self) -> List[str]:
        return self.targets['detection']

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOC.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    @classmethod
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for cl in self.classes:
            r[temp == cl.id] = cl.color[0]
            g[temp == cl.id] = cl.color[1]
            b[temp == cl.id] = cl.color[2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        rgb[rgb == self.classes[0].id] = self.classes[0].color[0]
        rgb[np.logical_or(rgb > 1., rgb < 0.)] = 1.
        return rgb

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is a dict with labels for the selected tasks.
        """
        img = to_tensor(Image.open(self.images[index]).convert("RGB"))

        targets = {}
        for t in self.tasks:
            if t == 'semantic':
                if self.masks[index] is not None:
                    if os.path.exists(self.masks[index]):
                        targets[t] = to_tensor(Image.open(self.masks[index])) * 255.
                        targets[t][targets[t] == self.semantic_ignore] = self.ignore_index
                        continue
                target_size = [1, *img.shape[1:]]
                targets[t] = torch.ones(size=target_size) * self.ignore_index
            elif t == 'detection':
                if os.path.exists(self.annotations[index]):
                    targets[t] = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
                else:
                    targets[t] = {}
            else:
                raise ValueError(f'Task {t} not supported or does not exist.')

        if self.transform is not None:
            img = self.transform(img)

        if self.transforms is not None:
            img, targets['semantic'] = self.transforms(img, targets['semantic'])

        return img, targets


def get_dataloaders(cfg, tasks):
    train_transform, batch_transform = _get_train_transform(cfg, VOC.MEAN, VOC.STD, tasks)
    eval_transform = torchvision.transforms.Normalize(VOC.MEAN, VOC.STD)

    train_dataset = VOC(
        root=cfg.dataset_dir, year=cfg.year, image_set='train', tasks=tasks,
        null_target=cfg.null_target, transforms=train_transform, download=cfg.download,
        use_augmented_set=cfg.use_augmented_set
    )

    # split of train set for traindev evaluation set
    traindev_dataset = copy.deepcopy(train_dataset)
    traindev_dataset.transform = eval_transform

    val_dataset = VOC(
        root=cfg.dataset_dir, year=cfg.year, image_set='val', tasks=tasks,
        null_target=cfg.null_target, transform=eval_transform, transforms=train_transform,
        download=cfg.download
    )

    if cfg.year == '2007':
        test_dataset = VOC(
            root=cfg.dataset_dir, year=cfg.year, image_set='test', tasks=tasks,
            null_target=cfg.null_target, transform=eval_transform, transforms=train_transform,
            download=cfg.download
        )
    else:
        test_dataset = None

    labeled_indices, unlabeled_indices = _get_fully_annotated_sample_indices(train_dataset.targets)

    if cfg.traindev_split_size > 0.:
        traindev_split_path = os.path.join(
            cfg.dataset_dir, f'traindev_split_size_{cfg.traindev_split_size}_s{cfg.seed}'
            f'{"_aug" if cfg.use_augmented_set and cfg.traindev_split_on_full_set else ""}.yaml')
        if os.path.exists(traindev_split_path):
            with open(traindev_split_path, 'r') as f:
                train_idx = yaml.load(f, yaml.Loader)
            traindev_idx = list(set(labeled_indices) - set(train_idx))
        else:
            train_idx, traindev_idx = split_train_val(
                labeled_indices, cfg.traindev_split_size, cfg.shuffle)
            with open(traindev_split_path, 'w') as f:
                yaml.dump(train_idx, f)
    else:
        if cfg.shuffle:
            np.random.shuffle(labeled_indices)
        train_idx = labeled_indices
        traindev_idx = []

    # Select ratio or number of labels on specific tasks
    if any([task_cfg.num_labels is not None or task_cfg.label_ratio is not None for task_cfg in
            cfg.tasks.values()]):
        train_idx_set = set(train_idx)
        unlabeled_indices_set = set(unlabeled_indices)
        for task, task_cfg in cfg.tasks.items():
            assert not (task_cfg.num_labels is not None and task_cfg.label_ratio is not None)
            if task_cfg.label_ratio is not None:
                error_msg = '[!] label ratio should be in the range (0, 1]'
                assert ((task_cfg.label_ratio > 0) and (task_cfg.label_ratio <= 1)), error_msg
                num_labels = int(np.floor(task_cfg.label_ratio * len(train_idx)))
            else:
                num_labels = task_cfg.num_labels

            error_msg = "[!] Number of labels must be 0 < num_labels <= len(dataset)"
            assert ((num_labels > 0) and (num_labels <= len(train_idx))), error_msg

            label_split_path = os.path.join(
                cfg.dataset_dir,
                f'labeled_{task}_{num_labels}_tdevsplit_{cfg.traindev_split_size}'
                f'_s{cfg.seed}.yaml')
            if os.path.exists(label_split_path):
                with open(label_split_path, 'r') as f:
                    train_labeled_idx = yaml.load(f, yaml.Loader)
                train_unlabeled_idx = list(train_idx_set - set(train_labeled_idx))
            else:
                if train_dataset.original_indices:
                    if cfg.shuffle:
                        np.random.shuffle(train_dataset.original_indices)
                    train_labeled_idx, train_unlabeled_idx = apply_num_labels(
                        num_labels, train_dataset.original_indices)
                    train_unlabeled_idx += list(
                        train_idx_set - set(train_dataset.original_indices))
                else:
                    train_labeled_idx, train_unlabeled_idx = apply_num_labels(
                        num_labels, train_idx)
                with open(label_split_path, 'w') as f:
                    yaml.dump(train_labeled_idx, f)

            targets = np.array(train_dataset.targets[task], dtype=object)
            targets[train_unlabeled_idx] = None
            train_dataset.targets[task] = targets.tolist()

            train_idx_set -= set(train_unlabeled_idx)
            unlabeled_indices_set.update(train_unlabeled_idx)
        train_idx = list(train_idx_set)
        unlabeled_indices = list(unlabeled_indices_set)

    logger.info("Available samples:")
    logger.info(f"  {len(train_idx)} fully labeled for training.")
    logger.info(f"  {len(unlabeled_indices)} unlabeled or partially labeled samples.")
    logger.info(f"  {len(traindev_idx)} fully labeled samples for evaluation in traindev split.")

    bs_unlabeled_factor = cfg.get('bs_unlabeled_factor', False)
    if cfg.use_implicit_setting:
        train_idx.extend(unlabeled_indices)
        logger.info("Loading unlabeled samples in implicit setting.")
    elif bs_unlabeled_factor:
        logger.info(f"Loading unlabeled samples in explicit setting with factor"
                    f" {bs_unlabeled_factor}.")
    else:
        logger.info("Using only fully labeled samples")

    if cfg.shuffle:
        np.random.shuffle(train_idx)

    logger.info(f"Total train set length: {len(train_idx)}.")
    logger.info(f"Total traindev set length: {len(traindev_idx)}.")
    logger.info(f"Total validation set length: {len(val_dataset)}.")
    if test_dataset is not None:
        logger.info(f"Total test set length: {len(test_dataset)}.")

    if bs_unlabeled_factor:
        assert not cfg.use_implicit_setting and bs_unlabeled_factor > 0
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=ExplicitSSLBatchSampler(
                train_idx, unlabeled_indices, cfg.batch_size, cfg.batch_size*bs_unlabeled_factor),
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            collate_fn=None,
            persistent_workers=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, train_idx), batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            collate_fn=None,
            persistent_workers=True
        )

    traindev_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(traindev_dataset, traindev_idx), batch_size=cfg.eval_batch_size,
        shuffle=False, num_workers=cfg.eval_num_workers, pin_memory=cfg.pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.eval_num_workers, pin_memory=cfg.pin_memory
    )

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
            num_workers=cfg.eval_num_workers, pin_memory=cfg.pin_memory
        )

    return {
        'train_dl': (train_loader, batch_transform),
        'traindev_dl': traindev_loader,
        'val_dl': val_loader,
        'test_dl': test_loader if test_dataset is not None else None
    }


def _get_fully_annotated_sample_indices(targets):
    labeled_indices = []
    unlabeled_indices = []
    for i, tgts in enumerate(zip(*targets.values())):
        if all([os.path.exists(p) for p in tgts]):
            labeled_indices.append(i)
        else:
            unlabeled_indices.append(i)
    return labeled_indices, unlabeled_indices


def _get_train_transform(cfg, mean, std, target_types=None):
    data_keys = [DataKey.INPUT]
    if target_types is not None:
        data_keys += [_target_types2data_keys[t] for t in target_types]

    pre_transform = RandomCrop((cfg.image_size[0], cfg.image_size[1]), pad_if_needed=True)

    if cfg.augmentation_type == Augmentation.FIXMATCH:
        crop_mode = CropMode[cfg.crop_mode.upper()] if cfg.get(
            'crop_mode') else CropMode.OVERLAPPING
        batch_transform = BatchTransformFixMatch(
            2, 10, mean, std, use_crops=True, image_size=cfg.image_size, always_cutout=True,
            data_keys=data_keys, crop_mode=crop_mode)
        return pre_transform, batch_transform
    elif cfg.augmentation_type == Augmentation.RANDAUG:
        batch_transform = BatchRandAugment(
            2, 10, mean, std, use_crops=True, image_size=cfg.image_size, always_cutout=True,
            data_keys=data_keys)
        return pre_transform, batch_transform
    elif cfg.augmentation_type == Augmentation.NONE:
        return pre_transform, None
    elif cfg.augmentation_type == Augmentation.STD:
        transforms = [
            K.AugmentationSequential(
                *(
                    K.RandomCrop(
                        (int(round(cfg.image_size[0] * s)), int(round(cfg.image_size[1] * s))),
                        pad_if_needed=True, cropping_mode='resample'
                    ) for s in (.5, .75, 1, 1.25, 1.5, 2)
                ),
                data_keys=data_keys, random_apply=1
            ),  # simulated random resize + crop with steps
            K.Resize(cfg.image_size)
        ]
        return pre_transform, BatchAugment(mean=mean, std=std, use_hflips=True, use_crops=False,
                                           data_keys=data_keys, transforms=transforms)
    else:
        raise ValueError(f"Augmentation type {cfg.augmentation_type} not supported.")

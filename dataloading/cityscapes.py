# Dataset class extended from torchvision https://github.com/pytorch/vision/blob/master/torchvision/datasets/cityscapes.py
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

# Label encoding from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py
# MIT License

# Copyright (c) 2017 Meet Pragnesh Shah

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

import copy
import glob
import json
import os
from collections import namedtuple
from collections.abc import MutableSequence
import yaml

import imageio
import numpy as np
import torch
import torchvision
from kornia import augmentation as K
from kornia.constants import DataKey
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset

from dataloading.utils import Augmentation, split_train_val, apply_num_labels, \
    ExplicitSSLBatchSampler
from dataloading.transforms import BatchAugment, BatchRandAugment, BatchTransformFixMatch, CropMode

try:
    from utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

_ORIGINAL_IMAGE_SIZE = (1024, 2048)
_MAX_BOXES = 120

_target_types2data_keys = {
    'instance': DataKey.MASK,
    'semantic': DataKey.MASK,
    'detection': DataKey.BBOX_XYXY,
    'disparity': DataKey.MASK
}


class Cityscapes(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if
            mode="gtFine" otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``gtFine`` or ``gtCoarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``,
            ``polygon``, ``color`` or ``detection``. Can also be a list to output a dict with all
            specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its
            target as entry and returns a transformed version.
        use_sequence (bool, optional): Whether to use the extra samples from the whole sequences.
        expand_with_extra (bool, optional): Whether to extend the training set with the extra
            samples from train_extra together with the train samples. Extra samples are only
            annotated with ``gtCoarse`` quality, by default not used.
        use_coarse (bool, optional): Only has effect when expand_with_extra is True. In such case,
            uses the coarse annotations available in train extra.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass',
                                 ['name', 'id', 'train_id', 'category', 'category_id',
                                  'has_instances', 'ignore_in_eval', 'color'])
    semantic_ignore = 255
    classes = [
        CityscapesClass('unlabeled', 0, semantic_ignore, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, semantic_ignore, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, semantic_ignore, 'void', 0, False, True,
                        (0, 0, 0)),
        CityscapesClass('out of roi', 3, semantic_ignore, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, semantic_ignore, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, semantic_ignore, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, semantic_ignore, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, semantic_ignore, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, semantic_ignore, 'flat', 1, False, True,
                        (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, semantic_ignore, 'construction', 2, False, True,
                        (180, 165, 180)),
        CityscapesClass('bridge', 15, semantic_ignore, 'construction', 2, False, True,
                        (150, 100, 100)),
        CityscapesClass('tunnel', 16, semantic_ignore, 'construction', 2, False, True,
                        (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, semantic_ignore, 'object', 3, False, True,
                        (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, semantic_ignore, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, semantic_ignore, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', 34, semantic_ignore, 'vehicle', 7, False, True,
                        (0, 0, 142)),
    ]
    max_disparity = 192
    n_classes = 19
    id2trainid = {c.id: c.train_id for c in classes}
    trainid2color = {c.train_id: c.color for c in classes}
    trainid2name = {c.train_id: c.name for c in classes}

    # Values from https://github.com/charlesCXK/TorchSemiSeg/blob/main/exp.city/city8.res50v3%2B.CPS%2BCutMix/config.py
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, root, split='train', mode='fine', target_type=None, transform=None,
                 null_target=255, use_sequences=False, extend_with_extra=False, use_coarse=False):
        super(Cityscapes, self).__init__(root)
        self.mode = 'gtCoarse' if mode == 'coarse' else 'gtFine'
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.ignore_index = null_target
        self.transform = transform

        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.disparity_dir = os.path.join(self.root, 'disparity', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, MutableSequence):
            self.target_type = [target_type]
        [
            verify_str_arg(value, "target_type",
                           ("instance", "semantic", "disparity", "detection"))
            for value in self.target_type
        ]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(
                    self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root,
                                              '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError(
                    'Dataset not found or incomplete. Please make sure all required folders for'
                    ' the specified "split" and "mode" are inside the "root" directory')

        if 'disparity' in self.target_type and not os.path.isdir(self.disparity_dir):
            raise RuntimeError(
                'Disparity directory not found. Please make sure all required folders for the'
                ' specified "split" and "mode" are inside the "root" directory')

        image_files = sorted(glob.glob(os.path.join(self.images_dir, "*", "*.png"),
                                       recursive=True))
        if extend_with_extra:
            assert split == 'train' and mode == 'fine', \
                'Extend_with_coarse only for fine mode and train split'
            train_extra_image_files = sorted(glob.glob(os.path.join(*os.path.split(
                self.images_dir)[:-1], "train_extra", "*", "*.png"), recursive=True))
            image_files += train_extra_image_files
        if use_sequences:
            assert split == 'train', 'use_sequences to be used only for train split'
            sequences_dir = self.images_dir.replace("leftImg8bit", "leftImg8bit_sequence")
            train_sequence_image_files = sorted(
                glob.glob(os.path.join(sequences_dir, "*", "*.png"), recursive=True))
            # We want to remove images in sequence already present in the standard train split
            tmp_image_files = [
                x.replace('/leftImg8bit/', '/leftImg8bit_sequence/') for x in image_files]
            unique_sequence_image_files = list(set(train_sequence_image_files).difference(
                set(tmp_image_files)))

            image_files += unique_sequence_image_files

        target_type = self.target_type.copy()
        if 'detection' in self.target_type and 'instance' not in self.target_type:
            target_type.append('instance')
        if 'instance' in target_type and 'semantic' not in self.target_type:
            target_type.append('semantic')

        target_files = {}
        for t in target_type:
            if t == 'disparity':
                target_files[t] = list(
                    map(lambda x: x.replace("leftImg8bit", "disparity"), image_files))
            else:
                if self.mode == "gtCoarse" or use_coarse:
                    target_files[t] = [
                        x.replace("leftImg8bit", "gtCoarse" if "train_extra" in x else "gtFine"
                                  ).replace(".", "_{}.".format(self._get_target_suffix(t)))
                        if 'sequence' not in x else None for x in image_files]
                else:
                    target_files[t] = [
                        x.replace("leftImg8bit", "gtFine").replace(
                            ".png", "_{}.png".format(self._get_target_suffix(t)))
                        if ('train_extra' not in x and 'sequence' not in x) else None
                        for x in image_files]

        self.images = image_files
        self.targets = target_files

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a
                list with more than one item. Otherwise target is a json object if
                target_type="polygon", else the image segmentation.
        """
        image = to_tensor(Image.open(self.images[index]))

        targets = {}
        for t in self.target_type:
            if t == 'polygon':
                raise NotImplementedError('Polygon annotations not supported at the moment.')
            elif t == 'disparity':
                targets[t] = to_tensor(
                    imageio.imread(self.targets[t][index]).astype(np.float32))
                mask = targets['disparity'] != 0.
                targets['disparity'][mask] = (targets['disparity'][mask] - 1.) / 256.
                targets['disparity'][~mask] = self.ignore_index
            else:
                if self.targets[t][index] is not None:
                    if t != 'detection':
                        targets[t] = to_tensor(Image.open(self.targets[t][index]))
                    # Instance and detection processing requires loading the semantic labels
                    if (t == 'instance' or t == 'detection') and \
                            'semantic' not in self.target_type:
                        targets['semantic'] = to_tensor(
                            Image.open(self.targets['semantic'][index]))
                    # Detection processing requires loading the instance ids
                    if t == 'detection' and 'instance' not in self.target_type:
                        targets['instance'] = to_tensor(
                            Image.open(self.targets['instance'][index]))
                else:
                    target_shape = [1, *image.shape[1:]]
                    if t != 'detection':
                        targets[t] = torch.ones(size=target_shape) * self.ignore_index
                    if (t == 'instance' or t == 'detection') and \
                            'semantic' not in self.target_type:
                        targets['semantic'] = torch.ones(size=target_shape) * self.ignore_index
                    if t == 'detection' and 'instance' not in self.target_type:
                        targets['instance'] = torch.ones(size=target_shape) * self.ignore_index

        if 'semantic' in targets:
            targets['semantic'] = self.encode_segmap(targets['semantic'])
            targets['semantic'][targets['semantic'] == self.semantic_ignore] = self.ignore_index
        if 'instance' in targets:
            targets['instance'][targets['semantic'] == self.ignore_index] = self.ignore_index
            targets['instance'] = self.encode_instancemap(targets['instance'])
        if 'detection' in self.target_type:
            assert 'instance' in targets, 'Detection targets require loading instance targets'
            targets['detection'] = self.encode_bboxes(targets['instance'])
        if 'instance' in targets:
            if 'semantic' not in self.target_type:
                del targets['semantic']
            if 'instance' not in self.target_type:
                del targets['instance']

        # Pre-transforms to images only
        if self.transform is not None:
            image = self.transform(image)

        return image, targets

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, target_type):
        target2suffix = {
            'instance': 'instanceIds',
            'detection': 'instanceIds',
            'semantic': 'labelIds',
            'color': 'color',
        }
        return target2suffix[target_type]

    def encode_segmap(self, mask):
        mask[mask != self.ignore_index] *= 255
        for cl in self.classes:
            mask[mask == cl.id] = cl.train_id
        return mask

    def encode_instancemap(self, ins):
        ins = ins.type(dtype=torch.float32)
        ins[ins < 1000] = self.ignore_index  # To avoid grouped objects without instance id
        return ins

    def encode_bboxes(self, ins_map):
        obj_ids = torch.unique(ins_map)[1:]
        masks = ins_map == obj_ids[:, None, None]
        labels = [self.id2trainid[id.item() // 1000] for id in obj_ids]
        bboxes = torchvision.ops.masks_to_boxes(masks)  # Returns bboxes in XYminXYmax format
        bboxes_padded = torch.ones((_MAX_BOXES, 4)) * self.ignore_index
        if len(labels) > _MAX_BOXES:
            logger.warn(f"Number of objects = {len(labels)}, larger than _MAX_BOXES={_MAX_BOXES}")
        bboxes_padded[:len(labels)] = bboxes[:_MAX_BOXES]
        labels_padded = torch.ones((_MAX_BOXES)) * self.ignore_index
        labels_padded[:len(labels)] = torch.as_tensor(labels[:_MAX_BOXES])
        return {'boxes': bboxes_padded, 'labels': labels_padded}

    @classmethod
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for cl in self.classes:
            if not cl.ignore_in_eval:
                r[temp == cl.train_id] = cl.color[0]
                g[temp == cl.train_id] = cl.color[1]
                b[temp == cl.train_id] = cl.color[2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        rgb[rgb == self.classes[0].train_id] = self.classes[0].color[0]
        rgb[np.logical_or(rgb > 1., rgb < 0.)] = 1.
        return rgb

    @classmethod
    def decode_insmap(self, insmap):
        # ins = torch.zeros(size=insmap.shape[1:])
        # for i in range(insmap.shape[0]):
        #     ins[insmap[i] == 1] = i
        unique_ids = insmap.unique()[1:]
        for i, uid in enumerate(unique_ids):
            insmap[insmap == uid] = i
        return insmap


def get_dataloaders(cfg, tasks):
    train_transform, batch_transform = _get_train_transform(
        cfg, Cityscapes.MEAN, Cityscapes.STD, tasks)
    eval_transform = torchvision.transforms.Normalize(
        Cityscapes.MEAN, Cityscapes.STD)

    train_dataset = Cityscapes(
        root=cfg.dataset_dir, split='train', target_type=tasks,
        null_target=cfg.null_target, use_sequences=cfg.use_sequences, mode=cfg.mode,
        extend_with_extra=cfg.extend_with_extra, use_coarse=cfg.use_coarse,
        transform=train_transform
    )

    # split of train set for traindev evaluation set
    traindev_dataset = copy.deepcopy(train_dataset)
    traindev_dataset.transform = eval_transform

    # The official validation set split
    val_dataset = Cityscapes(
        root=cfg.dataset_dir, split='val', target_type=tasks,
        null_target=cfg.null_target, transform=eval_transform
    )

    # the test set, with dummy labels, for qualitative evaluation or submission
    test_dataset = Cityscapes(
        root=cfg.dataset_dir, split='test', target_type=tasks,
        null_target=cfg.null_target, transform=eval_transform
    )

    (labeled_indices, coarsely_labeled_indices,
     unlabeled_indices) = _get_fully_annotated_sample_indices(train_dataset.targets)

    if cfg.traindev_split_size > 0.:
        traindev_split_path = os.path.join(
            cfg.dataset_dir, f'traindev_split_size_{cfg.traindev_split_size}_s{cfg.seed}.yaml')
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
                f'labeled_{task}_{num_labels}_tdevsplit'
                f'_{cfg.traindev_split_size}_s{cfg.seed}.yaml')
            if os.path.exists(label_split_path):
                with open(label_split_path, 'r') as f:
                    train_labeled_idx = yaml.load(f, yaml.Loader)
                train_unlabeled_idx = list(train_idx_set - set(train_labeled_idx))
            else:
                train_labeled_idx, train_unlabeled_idx = apply_num_labels(num_labels, train_idx)
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
    logger.info(f"  {len(coarsely_labeled_indices)} coarsely labeled samples.")
    logger.info(f"  {len(unlabeled_indices)} unlabeled or partially labeled samples.")
    logger.info(f"  {len(traindev_idx)} fully labeled samples for evaluation in traindev split.")

    bs_unlabeled_factor = cfg.get('bs_unlabeled_factor', False)
    if cfg.use_coarse:
        train_idx.extend(coarsely_labeled_indices)
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
    logger.info(f"Total test set length: {len(test_dataset)}.")

    if bs_unlabeled_factor:
        assert not cfg.use_implicit_setting and bs_unlabeled_factor > 0
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=ExplicitSSLBatchSampler(
                train_idx, unlabeled_indices, cfg.batch_size, cfg.batch_size*bs_unlabeled_factor),
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            collate_fn=None, persistent_workers=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, train_idx), batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            collate_fn=None, persistent_workers=True
        )

    traindev_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(traindev_dataset, traindev_idx), batch_size=cfg.eval_batch_size,
        shuffle=False, num_workers=cfg.eval_num_workers, pin_memory=cfg.pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.eval_num_workers, pin_memory=cfg.pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.eval_num_workers, pin_memory=cfg.pin_memory
    )

    return {
        'train_dl': (train_loader, batch_transform),
        'traindev_dl': traindev_loader,
        'val_dl': val_loader,
        'test_dl': test_loader
    }


def _get_fully_annotated_sample_indices(targets):
    labeled_indices = []
    coarsely_labeled_indices = []
    unlabeled_indices = []
    for i, tgts in enumerate(zip(*targets.values())):
        if all(tgts):
            if 'train_extra' in tgts[0]:
                coarsely_labeled_indices.append(i)
            else:
                labeled_indices.append(i)
        else:
            unlabeled_indices.append(i)
    return labeled_indices, coarsely_labeled_indices, unlabeled_indices


def _get_train_transform(cfg, mean, std, target_types=None):
    data_keys = [DataKey.INPUT]
    if target_types is not None:
        data_keys += [_target_types2data_keys[t] for t in target_types]

    if cfg.augmentation_type == Augmentation.FIXMATCH:
        crop_mode = CropMode[cfg.crop_mode.upper()] if cfg.get('crop_mode') \
            else CropMode.OVERLAPPING
        pre_transform = None
        batch_transform = BatchTransformFixMatch(
            2, 10, mean, std, use_crops=True, image_size=cfg.image_size, always_cutout=True,
            data_keys=data_keys, crop_mode=crop_mode)
        return pre_transform, batch_transform
    elif cfg.augmentation_type == Augmentation.RANDAUG:
        pre_transform = None
        batch_transform = BatchRandAugment(
            2, 10, mean, std, use_crops=True, image_size=cfg.image_size, always_cutout=True,
            data_keys=data_keys)
        return pre_transform, batch_transform
    elif cfg.augmentation_type == Augmentation.NONE:
        return None, None
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
        return None, BatchAugment(mean=mean, std=std, use_hflips=True, use_crops=False,
                                  data_keys=data_keys, transforms=transforms)
    else:
        raise ValueError(f"Augmentation type {cfg.augmentation_type} not supported.")

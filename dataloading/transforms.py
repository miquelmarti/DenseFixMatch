import enum
from typing import Any, Union, Dict, Optional

import kornia
import kornia.augmentation as K
from kornia.constants import DataKey
import torch
from torch import nn
from torchvision import transforms

PARAMETER_MAX = 10


class CropMode(enum.IntEnum):
    SAME = 1
    OVERLAPPING = 2
    ANY = 3


class AutoContrastKornia(kornia.augmentation._2d.intensity.base.IntensityAugmentationBase2D):
    r"""Maximize (normalize) image contrast.

    Args:
        p: Probability to apply auto contrast.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_contrast`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> autocontrast = AutoContrast(p=1.)
        >>> autocontrast(input)
    """

    def __init__(
        self, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor],
        flags: Optional[Dict[str, Any]] = None, transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return kornia.enhance.normalize_min_max(input)


class BatchAugment(nn.Module):
    def __init__(self,
                 mean: Union[tuple, list, torch.tensor] = None,
                 std: Union[tuple, list, torch.tensor] = None,
                 transforms: Union[list, dict] = None,
                 data_keys: list = None,
                 use_crops: bool = False,
                 image_size: tuple = None,
                 padding: int = None,
                 use_hflips: bool = True):
        super().__init__()

        self.image_size = image_size

        if mean is not None and std is not None:
            if not isinstance(mean, torch.Tensor):
                self.mean = torch.Tensor(mean)
            else:
                self.mean = mean
            if not isinstance(std, torch.Tensor):
                self.std = torch.Tensor(std)
            else:
                self.std = std

        self.use_crops = use_crops
        if use_crops:
            assert len(image_size) == 2, 'Invalid `image_size`. Must be a tuple of form (h, w)'
            if not isinstance(image_size, tuple):
                image_size = tuple(image_size)
            self.crop = K.RandomCrop(image_size, padding=padding, pad_if_needed=True, fill=0,
                                     cropping_mode='resample')

        if mean is not None and std is not None:
            self.normalize = K.Normalize(self.mean, self.std)
        else:
            self.normalize = None

        self.transforms = transforms

        self.use_hflips = use_hflips

        self.data_keys = data_keys

        self.setup()

    def setup(self):
        if isinstance(self.transforms, dict):
            self.transforms = list(self.transforms.values())
        if self.transforms is not None:
            assert isinstance(self.transforms, list)

        data_keys = self.data_keys if self.data_keys is not None else [DataKey.INPUT]
        tfms = []
        if self.use_crops:
            tfms.append(self.crop)
        if self.transforms is not None:
            tfms.extend(self.transforms)
        if self.use_hflips:
            tfms.append(K.RandomHorizontalFlip())
        if self.normalize is not None:
            tfms.append(self.normalize)
        self.transform = K.AugmentationSequential(*tfms, data_keys=data_keys)

    @torch.no_grad()
    def forward(self, x, y):
        """
        Applies transforms on the batch.

        Args:
            x (torch.Tensor): Batch of input images.
            y (List[torch.Tensor]): Lists of batch of labels. Requires data_keys with same order.
        """
        if y is None:
            return self.transform(x)

        if self.data_keys:
            inputs = [x]
            for d, y_ in zip(self.data_keys[1:], y.values()):
                if d == DataKey.BBOX or d == DataKey.BBOX_XYWH or d == DataKey.BBOX_XYXY:
                    inputs.append(y_['boxes'])
                else:
                    inputs.append(y_ + 1.)  # Shift all labels for zero padding to work

            out = self.transform(*inputs, data_keys=self.data_keys)

            x_out = out[0]
            y_out = {}
            for d, k, o in zip(self.data_keys[1:], y.keys(), out[1:]):
                if d == DataKey.BBOX or d == DataKey.BBOX_XYWH or d == DataKey.BBOX_XYXY:
                    y_out[k] = {'boxes': o, 'labels': y[k]['labels']}
                else:
                    y_out[k] = o - 1.  # Bring back labels to original range, padding becomes -1

            return x_out, y_out
        else:
            return self.transform(x), y


class BatchRandAugment(nn.Module):
    def __init__(self,
                 N_TFMS: int,
                 MAGN: int,
                 mean: Union[tuple, list, torch.tensor] = None,
                 std: Union[tuple, list, torch.tensor] = None,
                 data_keys: list = None,
                 use_crops: bool = False,
                 image_size: tuple = None,
                 padding: int = None,
                 always_cutout: bool = False,
                 use_hflips: bool = True,
                 disable_translations: bool = False):
        super().__init__()

        self.N_TFMS = N_TFMS
        self.MAGN = MAGN

        self.image_size = image_size

        if mean is not None and std is not None:
            if not isinstance(mean, torch.Tensor):
                self.mean = torch.Tensor(mean)
            else:
                self.mean = mean
            if not isinstance(std, torch.Tensor):
                self.std = torch.Tensor(std)

        self.use_crops = use_crops
        if use_crops:
            assert len(image_size) == 2, 'Invalid `image_size`. Must be a tuple of form (h, w)'
            if not isinstance(image_size, tuple):
                image_size = tuple(image_size)
            self.crop = K.RandomCrop(image_size, padding=padding, pad_if_needed=True, fill=0,
                                     cropping_mode='resample')

        if mean is not None and std is not None:
            self.normalize = K.Normalize(self.mean, self.std)
        else:
            self.normalize = None

        self.transforms = randaug_transforms_pool(MAGN)

        if disable_translations:
            self.transforms.pop('TranslateX')
            self.transforms.pop('TranslateY')

        if always_cutout:
            self.cutout = self.transforms.pop(
                'Cutout', K.RandomErasing(scale=(0.1, 0.3), ratio=(1, 1), value=.5, p=1.))
        else:
            self.cutout = False

        self.use_hflips = use_hflips

        self.data_keys = data_keys

        self.setup()

    def setup(self):
        self.transforms = list(self.transforms.values())

        data_keys = self.data_keys if self.data_keys is not None else [DataKey.INPUT]
        tfms = []
        if self.use_crops:
            tfms.append(self.crop)
        tfms.append(
            K.AugmentationSequential(
                *self.transforms, random_apply=self.N_TFMS, data_keys=data_keys
            )
        )
        if self.use_hflips:
            tfms.append(K.RandomHorizontalFlip())
        if self.cutout:
            tfms.append(self.cutout)
        if self.normalize is not None:
            tfms.append(self.normalize)
        self.transform = K.AugmentationSequential(*tfms, data_keys=data_keys)

    @torch.no_grad()
    def forward(self, x, y=None):
        """
        Applies transforms on the batch.

        Args:
            x (torch.Tensor): Batch of input images.
            y (List[torch.Tensor]): Lists of batch of labels. Requires data_keys with same order.
        """
        if y is None:
            return self.transform(x)

        if self.data_keys:
            inputs = [x]
            for d, y_ in zip(self.data_keys[1:], y.values()):
                if d == DataKey.BBOX or d == DataKey.BBOX_XYWH or d == DataKey.BBOX_XYXY:
                    inputs.append(y_['boxes'])
                else:
                    inputs.append(y_ + 1.)  # Shift all labels for zero padding to work

            out = self.transform(*inputs, data_keys=self.data_keys)

            x_out = out[0]
            y_out = {}
            for d, k, o in zip(self.data_keys[1:], y.keys(), out[1:]):
                if d == DataKey.BBOX or d == DataKey.BBOX_XYWH or d == DataKey.BBOX_XYXY:
                    y_out[k] = {'boxes': o, 'labels': y[k]['labels']}
                else:
                    y_out[k] = o - 1.  # Bring back labels to original range, padding becomes -1

            return x_out, y_out
        else:
            return self.transform(x), y


class BatchTransformFixMatch(BatchRandAugment):
    def __init__(self,
                 N_TFMS: int,
                 MAGN: int,
                 mean: Union[tuple, list, torch.tensor] = None,
                 std: Union[tuple, list, torch.tensor] = None,
                 data_keys: list = None,
                 use_crops: bool = False,
                 image_size: tuple = None,
                 padding: int = None,
                 always_cutout: bool = False,
                 use_hflips: bool = True,
                 crop_mode: CropMode = CropMode.OVERLAPPING):
        super().__init__(
            N_TFMS, MAGN, mean, std, data_keys, use_crops, image_size, padding, always_cutout,
            use_hflips, disable_translations=use_crops)
        self.crop_mode = crop_mode

    def setup(self):
        super().setup()

        data_keys = self.data_keys if self.data_keys is not None else [DataKey.INPUT]
        tfms = []
        if self.use_crops:
            tfms.append(self.crop)
        if self.use_hflips:
            tfms.append(K.RandomHorizontalFlip())
        if self.normalize is not None:
            tfms.append(self.normalize)
        self.weak_transform = K.AugmentationSequential(*tfms, data_keys=data_keys)

    @torch.no_grad()
    def forward(self, x, y=None):
        """
        Applies weak and strong transforms on the batch

        Args:
            x (torch.Tensor): Batch of input images.
            y (torch.Tensor): Batch of labels.
        """
        if self.data_keys:
            inputs = [x]
            for d, y_ in zip(self.data_keys[1:], y.values()):
                if d == DataKey.BBOX or d == DataKey.BBOX_XYWH or d == DataKey.BBOX_XYXY:
                    inputs.append(y_['boxes'])
                else:
                    inputs.append(y_ + 1.)  # Shift all labels for zero padding to work

            out_w = self.weak_transform(*inputs, data_keys=self.data_keys)

            params_s = self.transform.forward_parameters(inputs[0].shape)

            if self.crop_mode == CropMode.SAME:
                params_s[0] = self.weak_transform._params[0]  # Same random crop
            elif self.crop_mode == CropMode.OVERLAPPING:
                crop_factor = 1.
                crop_src = self.weak_transform._params[0].data['src']
                bias = (torch.rand(crop_src.shape[0], crop_src.shape[2]) - .5) * crop_factor
                bias = torch.round(bias * torch.Tensor(self.image_size))
                bias = bias.unsqueeze(dim=1).expand_as(params_s[0].data['src'])
                new_crop_src = crop_src + bias
                params_s[0].data['src'] = new_crop_src
            elif self.crop_mode == CropMode.ANY:
                pass  # Do not modify the crop parameters
            out_s = self.transform(*inputs, params=params_s, data_keys=self.data_keys)

            if len(self.data_keys) > 1:
                input_s = out_s[0]
                input_w = out_w[0]
            else:
                input_s = out_s
                input_w = out_w

            labels_s = {}
            labels_w = {}
            for i, (d, k) in enumerate(zip(self.data_keys[1:], y.keys())):
                if d == DataKey.BBOX or d == DataKey.BBOX_XYWH or d == DataKey.BBOX_XYXY:
                    labels_s[k] = {'boxes': out_s[i+1], 'labels': y[k]['labels']}
                    labels_w[k] = {'boxes': out_w[i+1], 'labels': y[k]['labels']}
                else:
                    # Bring back labels to original range, padding becomes -1
                    labels_s[k] = out_s[i+1] - 1.
                    labels_w[k] = out_w[i+1] - 1.

            out_dict = {
                'input': {'weak': input_w, 'strong': input_s}
            }
            if len(self.data_keys) > 1:
                out_dict['labels'] = {'weak': labels_w, 'strong': labels_s}
        else:
            # For image-only augmentation
            out_dict = {
                'input':
                    {
                        'weak': self.weak_transform(x),
                        'strong': self.transform(x)
                    },
            }
            if y is not None:
                # For image-level labels, otherwise data keys have to be defined
                out_dict['labels'] = {
                    'weak': y,
                    'strong': y
                }
        return out_dict

    def match(self, x, data_keys=None):
        data_keys = self.data_keys[1:] if data_keys is None else data_keys

        labels = (x[k] + 1. for k in x.keys())

        out = self.weak_transform.inverse(*labels, data_keys=data_keys)
        if isinstance(out, torch.Tensor):
            out = [out]
        out = self.transform(*out, params=self.transform._params, data_keys=data_keys)
        if isinstance(out, torch.Tensor):
            out = [out]

        y_out = {}
        for d, k, o in zip(self.data_keys[1:], x.keys(), out):
            if d == DataKey.BBOX or d == DataKey.BBOX_XYWH or d == DataKey.BBOX_XYXY:
                y_out[k] = {'boxes': o, 'labels': x[k]['labels']}
            else:
                y_out[k] = o - 1.  # Bring back labels to original range, padding becomes -1

        return y_out


def randaug_transforms_pool(magnitude=10):
    # Original RandAug pool https://arxiv.org/pdf/1909.13719.pdf
    diff_mag = magnitude * 0.9 / 10
    return {
        'Identity': K.ColorJitter(p=0.),
        'AutoContrast': AutoContrastKornia(p=1.),
        'Equalize': K.RandomEqualize(p=1.),
        'Invert': K.RandomInvert(p=1.),
        'Solarize': K.RandomSolarize(thresholds=(1-magnitude/10, 1), additions=0., p=1.),
        'Posterize': K.RandomPosterize(bits=8-round(magnitude*4/10), p=1.),
        'Color': K.ColorJitter(saturation=(1 - diff_mag, 1 + diff_mag), p=1.),
        'Contrast': K.ColorJitter(contrast=(1 - diff_mag, 1 + diff_mag), p=1.),
        'Brightness': K.ColorJitter(brightness=(1 - diff_mag, 1 + diff_mag), p=1.),
        'Sharpness': K.RandomSharpness(sharpness=(1 - diff_mag, 1 + diff_mag), p=1.),
        'ShearX': K.RandomAffine(degrees=0., shear=magnitude*17/10, p=1.),  # shear factor .3
        'ShearY': K.RandomAffine(
            degrees=0., shear=(0, 0, -magnitude*17/10, magnitude*17/10), p=1.),
        'TranslateX': K.RandomAffine(degrees=0., translate=(magnitude*.15/10, 0), p=1.),
        'TranslateY': K.RandomAffine(degrees=0., translate=(0, magnitude*.15/10), p=1.),
        'Rotate': K.RandomAffine(degrees=magnitude*30/10, p=1.),
        'Cutout': K.RandomErasing(scale=(magnitude*0.1/10, magnitude*0.3/10), ratio=(1, 1),
            value=.5, p=1.)
    }


class RandomCrop(transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, fill_target=-1,
                 padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.fill_target = fill_target

    def forward(self, img, target):
        if self.padding is not None:
            img = transforms.functional.pad(img, self.padding, self.fill, self.padding_mode)
            target = transforms.functional.pad(target, self.padding, self.fill_target,
                                               self.padding_mode)

        width, height = transforms.functional.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = transforms.functional.pad(img, padding, self.fill, self.padding_mode)
            target = transforms.functional.pad(target, padding, self.fill_target,
                                               self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = transforms.functional.pad(img, padding, self.fill, self.padding_mode)
            target = transforms.functional.pad(target, padding, self.fill_target,
                                               self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return transforms.functional.crop(img, i, j, h, w), \
            transforms.functional.crop(target, i, j, h, w)

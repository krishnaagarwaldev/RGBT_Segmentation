"""Synchronized multimodal augmentations and preprocessing for RGB + Thermal segmentation.

Ensures:
- Discrete integer class mask IDs (0..8) are strictly preserved using NEAREST interpolation.
- Geometric transforms (RandomHorizontalFlip, RandomRescale, RandomCrop) are applied identically
  across RGB, Thermal, and Mask.
- Modality-specific photometric transforms (ColorJitter) are applied only to RGB.
- RGB and Thermal channels are independently normalized with appropriate statistics.
"""
import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any


class Compose:
    """Compose multiple synchronized transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb: Image.Image, thermal: Image.Image, mask: Image.Image):
        for t in self.transforms:
            rgb, thermal, mask = t(rgb, thermal, mask)
        return rgb, thermal, mask


class RandomHorizontalFlip:
    """Synchronously flip RGB, Thermal, and Mask horizontally with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, rgb: Image.Image, thermal: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            rgb = TF.hflip(rgb)
            thermal = TF.hflip(thermal)
            mask = TF.hflip(mask)
        return rgb, thermal, mask


class RandomScaleCrop:
    """Synchronous random scaling followed by crop or pad to target size.
    
    CRITICAL: Mask interpolation is strictly InterpolationMode.NEAREST.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (480, 640),
        scale_range: Tuple[float, float] = (0.75, 1.25),
        pad_fill_mask: int = 0
    ):
        self.target_height, self.target_width = target_size
        self.scale_min, self.scale_max = scale_range
        self.pad_fill_mask = pad_fill_mask

    def __call__(self, rgb: Image.Image, thermal: Image.Image, mask: Image.Image):
        # 1. Random scaling
        scale = random.uniform(self.scale_min, self.scale_max)
        orig_w, orig_h = rgb.size
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # Bilinear for images, strictly NEAREST for mask
        rgb = TF.resize(rgb, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        thermal = TF.resize(thermal, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)

        # 2. Pad if smaller than target size
        pad_h = max(0, self.target_height - new_h)
        pad_w = max(0, self.target_width - new_w)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # [left, top, right, bottom]
            padding = [pad_left, pad_top, pad_right, pad_bottom]
            rgb = TF.pad(rgb, padding, fill=0, padding_mode="constant")
            thermal = TF.pad(thermal, padding, fill=0, padding_mode="constant")
            mask = TF.pad(mask, padding, fill=self.pad_fill_mask, padding_mode="constant")
            new_w += pad_w
            new_h += pad_h

        # 3. Random crop to target size
        if new_h > self.target_height or new_w > self.target_width:
            top = random.randint(0, new_h - self.target_height)
            left = random.randint(0, new_w - self.target_width)
            rgb = TF.crop(rgb, top, left, self.target_height, self.target_width)
            thermal = TF.crop(thermal, top, left, self.target_height, self.target_width)
            mask = TF.crop(mask, top, left, self.target_height, self.target_width)

        return rgb, thermal, mask


class ColorJitterRGB:
    """Apply photometric color jitter strictly to the RGB stream (thermal is untouched)."""
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, rgb: Image.Image, thermal: Image.Image, mask: Image.Image):
        if self.brightness > 0 and random.random() < 0.5:
            factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            rgb = TF.adjust_brightness(rgb, factor)
        if self.contrast > 0 and random.random() < 0.5:
            factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            rgb = TF.adjust_contrast(rgb, factor)
        if self.saturation > 0 and random.random() < 0.5:
            factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            rgb = TF.adjust_saturation(rgb, factor)
        return rgb, thermal, mask


class ToTensorNormalize:
    """Convert RGB, Thermal, and Mask PIL Images to PyTorch tensors and normalize.
    
    Outputs:
        rgb_tensor: float32 tensor of shape (3, H, W), normalized.
        thermal_tensor: float32 tensor of shape (1, H, W), normalized.
        mask_tensor: int64 tensor of shape (H, W), with exact class IDs preserved.
    """
    def __init__(
        self,
        rgb_mean: Tuple[float, float, float] = (0.2215, 0.2586, 0.2299),
        rgb_std: Tuple[float, float, float] = (0.2280, 0.2318, 0.2367),
        thermal_mean: Tuple[float] = (0.3954,),
        thermal_std: Tuple[float] = (0.1848,)
    ):
        self.rgb_mean = list(rgb_mean)
        self.rgb_std = list(rgb_std)
        self.thermal_mean = list(thermal_mean)
        self.thermal_std = list(thermal_std)

    def __call__(self, rgb: Image.Image, thermal: Image.Image, mask: Image.Image):
        # Convert RGB to (3, H, W) in [0, 1]
        rgb_t = TF.to_tensor(rgb)
        rgb_t = TF.normalize(rgb_t, mean=self.rgb_mean, std=self.rgb_std)

        # Convert Thermal to (1, H, W) in [0, 1]
        th_t = TF.to_tensor(thermal)
        th_t = TF.normalize(th_t, mean=self.thermal_mean, std=self.thermal_std)

        # Convert Mask to (H, W) torch.long without normalization (exact integer IDs)
        mask_np = np.array(mask, dtype=np.int64)
        mask_t = torch.from_numpy(mask_np)

        return rgb_t, th_t, mask_t


def get_train_transforms(
    target_size: Tuple[int, int] = (480, 640),
    hflip_p: float = 0.5,
    scale_range: Tuple[float, float] = (0.75, 1.25),
    jitter_brightness: float = 0.2,
    jitter_contrast: float = 0.2,
    jitter_saturation: float = 0.2,
    rgb_mean: Tuple[float, float, float] = (0.2215, 0.2586, 0.2299),
    rgb_std: Tuple[float, float, float] = (0.2280, 0.2318, 0.2367),
    thermal_mean: Tuple[float] = (0.3954,),
    thermal_std: Tuple[float] = (0.1848,)
) -> Compose:
    """Build training transform pipeline."""
    return Compose([
        RandomHorizontalFlip(p=hflip_p),
        RandomScaleCrop(target_size=target_size, scale_range=scale_range),
        ColorJitterRGB(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation
        ),
        ToTensorNormalize(
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            thermal_mean=thermal_mean,
            thermal_std=thermal_std
        )
    ])


def get_val_transforms(
    rgb_mean: Tuple[float, float, float] = (0.2215, 0.2586, 0.2299),
    rgb_std: Tuple[float, float, float] = (0.2280, 0.2318, 0.2367),
    thermal_mean: Tuple[float] = (0.3954,),
    thermal_std: Tuple[float] = (0.1848,)
) -> Compose:
    """Build validation/testing transform pipeline (deterministic, no augmentation)."""
    return Compose([
        ToTensorNormalize(
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            thermal_mean=thermal_mean,
            thermal_std=thermal_std
        )
    ])

"""PyTorch Dataset for RGB + Thermal Semantic Segmentation on ir_seg_dataset (MFNet).

Handles:
- 4-channel PNG images in images/ (Channels 0..2 = RGB, Channel 3 = Thermal).
- 1-channel PNG ground-truth masks in labels/ (Values 0..8).
- Train/Val/Test split files (train.txt, val.txt, test.txt, test_day.txt, test_night.txt).
- Resolves '_flip' entries in train.txt gracefully by loading base image and flipping.
- Verifies image dimensions, channel counts, and mask ID preservation.
"""
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, List, Dict, Any, Tuple


class RGBTDataset(Dataset):
    """Multispectral RGB-Thermal Semantic Segmentation Dataset."""

    def __init__(
        self,
        dataset_root: str,
        split_file: str,
        transforms: Optional[Callable] = None,
        use_unique_stems: bool = False,
        images_subdir: str = "images",
        labels_subdir: str = "labels"
    ):
        """
        Args:
            dataset_root: Path to ir_seg_dataset directory.
            split_file: Path to split text file (e.g., train.txt, val.txt, test.txt).
            transforms: Callable transform accepting (rgb, thermal, mask) and returning tensors.
            use_unique_stems: If True, strips '_flip' duplicates so each base sample is loaded once
                              (recommended for training with dynamic online random augmentations).
            images_subdir: Subdirectory containing 4-channel PNG images.
            labels_subdir: Subdirectory containing mask PNG images.
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.images_dir = os.path.join(dataset_root, images_subdir)
        self.labels_dir = os.path.join(dataset_root, labels_subdir)
        self.transforms = transforms

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        # Read sample stems from split file
        with open(split_file, "r") as f:
            raw_lines = [l.strip() for l in f if l.strip()]

        if use_unique_stems:
            # Strip '_flip' suffixes to retain only unique physical base images
            seen = set()
            unique_lines = []
            for l in raw_lines:
                base = l[:-5] if l.endswith("_flip") else l
                if base not in seen:
                    seen.add(base)
                    unique_lines.append(base)
            self.samples = unique_lines
        else:
            self.samples = raw_lines

        # Verify dataset integrity for this split
        self._verify_samples()

    def _verify_samples(self):
        """Verify that every sample has an existing base image and label mask."""
        missing = []
        for sample in self.samples:
            base_stem = sample[:-5] if sample.endswith("_flip") else sample
            img_path = os.path.join(self.images_dir, f"{base_stem}.png")
            lbl_path = os.path.join(self.labels_dir, f"{base_stem}.png")
            if not os.path.exists(img_path):
                missing.append(f"Image: {img_path}")
            if not os.path.exists(lbl_path):
                missing.append(f"Label: {lbl_path}")
        if missing:
            raise FileNotFoundError(
                f"Dataset integrity check failed! {len(missing)} missing files:\n"
                + "\n".join(missing[:10])
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_name = self.samples[idx]
        is_flip = sample_name.endswith("_flip")
        base_stem = sample_name[:-5] if is_flip else sample_name

        img_path = os.path.join(self.images_dir, f"{base_stem}.png")
        lbl_path = os.path.join(self.labels_dir, f"{base_stem}.png")

        # 1. Load 4-channel RGBA image
        with Image.open(img_path) as full_img:
            # Ensure RGBA format (ch0..2 = RGB, ch3 = Thermal)
            if full_img.mode != "RGBA":
                full_img = full_img.convert("RGBA")
            img_np = np.array(full_img)

        rgb_np = img_np[:, :, :3]   # (H, W, 3)
        th_np = img_np[:, :, 3]     # (H, W)

        rgb_pil = Image.fromarray(rgb_np, mode="RGB")
        th_pil = Image.fromarray(th_np, mode="L")

        # 2. Load Ground-Truth Segmentation Mask
        with Image.open(lbl_path) as lbl_img:
            # Ensure single-channel grayscale mask
            if lbl_img.mode != "L":
                lbl_img = lbl_img.convert("L")
            mask_pil = lbl_img.copy()

        # 3. If split item is explicitly '_flip' and no dynamic hflip was used, flip now
        if is_flip:
            rgb_pil = rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)
            th_pil = th_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

        # 4. Apply synchronized transforms
        if self.transforms is not None:
            rgb_t, th_t, mask_t = self.transforms(rgb_pil, th_pil, mask_pil)
        else:
            # Default fallback to tensors
            rgb_t = torch.from_numpy(np.array(rgb_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            th_t = torch.from_numpy(np.array(th_pil, dtype=np.float32)[np.newaxis, :, :] / 255.0)
            mask_t = torch.from_numpy(np.array(mask_pil, dtype=np.int64))

        return {
            "rgb": rgb_t,           # Tensor (3, H, W)
            "thermal": th_t,       # Tensor (1, H, W)
            "mask": mask_t,         # Tensor (H, W) with int class IDs 0..8
            "name": sample_name     # String identifier
        }

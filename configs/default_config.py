"""Default configuration for CSFNet-1 (STDC1) RGBT semantic segmentation on ir_seg_dataset."""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DatasetConfig:
    # Paths (relative to workspace root or absolute)
    dataset_root: str = "ir_seg_dataset"
    images_dir: str = os.path.join("ir_seg_dataset", "images")
    labels_dir: str = os.path.join("ir_seg_dataset", "labels")
    
    # Split files
    train_split: str = os.path.join("ir_seg_dataset", "train.txt")
    val_split: str = os.path.join("ir_seg_dataset", "val.txt")
    test_split: str = os.path.join("ir_seg_dataset", "test.txt")
    test_day_split: str = os.path.join("ir_seg_dataset", "test_day.txt")
    test_night_split: str = os.path.join("ir_seg_dataset", "test_night.txt")
    
    # Image properties
    image_height: int = 480
    image_width: int = 640
    rgb_channels: int = 3
    thermal_channels: int = 1
    
    # Classes
    num_classes: int = 9
    class_names: List[str] = field(default_factory=lambda: [
        "unlabeled",   # 0
        "car",         # 1
        "person",      # 2
        "bike",        # 3
        "curve",       # 4
        "car_stop",    # 5
        "guardrail",   # 6
        "color_cone",  # 7
        "bump"         # 8
    ])
    
    # Normalization statistics computed across ir_seg_dataset train split
    # RGB Mean: [0.2215, 0.2586, 0.2299], Std: [0.2280, 0.2318, 0.2367]
    # Thermal Mean: [0.3954], Std: [0.1848]
    rgb_mean: Tuple[float, float, float] = (0.2215, 0.2586, 0.2299)
    rgb_std: Tuple[float, float, float] = (0.2280, 0.2318, 0.2367)
    thermal_mean: Tuple[float] = (0.3954,)
    thermal_std: Tuple[float] = (0.1848,)


@dataclass
class AugmentationConfig:
    # Synchronized geometric augmentations
    horizontal_flip_prob: float = 0.5
    random_scale_range: Tuple[float, float] = (0.75, 1.25)
    random_crop_size: Optional[Tuple[int, int]] = (480, 640)
    
    # Modality-specific photometric augmentations (applied only to RGB)
    rgb_brightness: float = 0.2
    rgb_contrast: float = 0.2
    rgb_saturation: float = 0.2


@dataclass
class ModelConfig:
    architecture: str = "CSFNet-2"
    backbone: str = "STDCNet1446"  # STDC2
    num_classes: int = 9
    dataset: str = "MFNet"        # Configures pooling sizes: [(16, 24), (8, 12), (4, 6), (2, 3), (5, 5)]
    act_type: str = "relu"
    pretrain: bool = True
    backbone_path: Optional[str] = "pretrained/STDCNet1446_76.47.tar"


@dataclass
class LossConfig:
    # Options: "ce", "weighted_ce", "focal", "dice", "combined_ce_dice", "combined_focal_dice"
    loss_type: str = "combined_ce_dice"
    
    # Weighting strategy for CE: "none", "median_frequency", "enet_log", "power_smoothed"
    class_weight_strategy: str = "enet_log"
    dice_weight: float = 0.5      # Weight multiplier lambda for Dice Loss in combined losses
    focal_gamma: float = 2.0      # Focusing parameter for Focal Loss
    ignore_index: Optional[int] = None  # Class 0 is evaluated in MFNet, set to None


@dataclass
class TrainingConfig:
    batch_size: int = 4           # Fits comfortably in 6GB VRAM (RTX 3050 Laptop)
    val_batch_size: int = 4
    num_workers: int = 2          # Windows-safe multiprocessing
    epochs: int = 150
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    optimizer: str = "adamw"      # "adamw" or "sgd"
    momentum: float = 0.9         # For SGD
    lr_scheduler: str = "cosine"  # "cosine" or "poly"
    poly_power: float = 0.9
    
    # Mixed precision & hardware
    use_amp: bool = True
    device: str = "cuda"
    seed: int = 42
    
    # Checkpoints & Early Stopping
    output_dir: str = "runs"
    experiment_name: str = "csfnet2_pretrained"
    save_interval: int = 5
    early_stopping_patience: int = 25
    early_stopping_metric: str = "mIoU"  # Monitor mIoU, not pixel accuracy



@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> Config:
    return Config()

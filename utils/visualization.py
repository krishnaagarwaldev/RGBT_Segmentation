"""Visualization utilities for RGBT segmentation: palettes, side-by-side plots, and training curves."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for command line execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any


# 9 Curated, distinct colors for MFNet categories:
# 0: unlabeled   -> Dark Slate Gray [40, 40, 40]
# 1: car         -> Crimson Red     [220, 20, 60]
# 2: person      -> Royal Blue      [65, 105, 225]
# 3: bike        -> Goldenrod       [218, 165, 32]
# 4: curve       -> Purple          [138, 43, 226]
# 5: car_stop    -> Orange Red      [255, 69, 0]
# 6: guardrail   -> Cyan            [0, 206, 209]
# 7: color_cone  -> Coral           [255, 127, 80]
# 8: bump        -> Lime Green      [50, 205, 50]
CLASS_COLORS = np.array([
    [40, 40, 40],       # 0: unlabeled
    [220, 20, 60],      # 1: car
    [65, 105, 225],     # 2: person
    [218, 165, 32],     # 3: bike
    [138, 43, 226],     # 4: curve
    [255, 69, 0],       # 5: car_stop
    [0, 206, 209],      # 6: guardrail
    [255, 127, 80],     # 7: color_cone
    [50, 205, 50],      # 8: bump
], dtype=np.uint8)

CLASS_NAMES = [
    "unlabeled", "car", "person", "bike",
    "curve", "car_stop", "guardrail", "color_cone", "bump"
]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map integer class mask (H, W) to RGB colored mask (H, W, 3)."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c_id in range(len(CLASS_COLORS)):
        color_mask[mask == c_id] = CLASS_COLORS[c_id]
    return color_mask


def denormalize_rgb(
    rgb_tensor: np.ndarray,
    mean: Tuple[float, float, float] = (0.2215, 0.2586, 0.2299),
    std: Tuple[float, float, float] = (0.2280, 0.2318, 0.2367)
) -> np.ndarray:
    """Convert (3, H, W) normalized tensor to (H, W, 3) uint8 image in [0, 255]."""
    rgb = rgb_tensor.transpose(1, 2, 0).copy()
    for c in range(3):
        rgb[:, :, c] = rgb[:, :, c] * std[c] + mean[c]
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def denormalize_thermal(
    th_tensor: np.ndarray,
    mean: Tuple[float] = (0.3954,),
    std: Tuple[float] = (0.1848,)
) -> np.ndarray:
    """Convert (1, H, W) normalized tensor to (H, W) uint8 thermal array in [0, 255]."""
    th = th_tensor[0].copy()
    th = th * std[0] + mean[0]
    th = np.clip(th * 255.0, 0, 255).astype(np.uint8)
    return th


def create_comparison_plot(
    rgb_img: np.ndarray,
    thermal_img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    title: Optional[str] = None
):
    """Generate a clean 4-panel side-by-side comparison figure with color legend.

    Panels: [RGB Image | Thermal (Inferno) | Ground Truth Mask | Predicted Mask]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

    # 1. RGB
    axes[0].imshow(rgb_img)
    axes[0].set_title("Visible (RGB)", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # 2. Thermal
    axes[1].imshow(thermal_img, cmap="inferno")
    axes[1].set_title("Thermal (IR)", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # 3. Ground Truth
    gt_colored = colorize_mask(gt_mask)
    axes[2].imshow(gt_colored)
    axes[2].set_title("Ground Truth Mask", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # 4. Prediction
    pred_colored = colorize_mask(pred_mask)
    axes[3].imshow(pred_colored)
    axes[3].set_title("CSFNet-1 Prediction", fontsize=13, fontweight="bold")
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    # Class legend underneath
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=f"{i}: {CLASS_NAMES[i]}")
        for i in range(len(CLASS_NAMES))
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(CLASS_NAMES),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02)
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: Dict[str, List[float]], save_dir: str):
    """Plot and save loss, mIoU, and pixel accuracy training curves.

    Generates:
        - {save_dir}/loss_curve.png
        - {save_dir}/miou_curve.png
        - {save_dir}/accuracy_curve.png
        - {save_dir}/training_dashboard.png
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

    if len(epochs) == 0:
        return

    # 1. Training & Validation Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss Curve", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close(fig)

    # 2. Validation mIoU Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if "val_mIoU" in history:
        ax.plot(epochs, [v * 100 for v in history["val_mIoU"]], label="Val mIoU (All 9 Classes)", color="#2ca02c", linewidth=2)
    if "val_mIoU_fg" in history:
        ax.plot(epochs, [v * 100 for v in history["val_mIoU_fg"]], label="Val mIoU (Foreground Classes 1-8)", color="#9467bd", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mIoU (%)", fontsize=12)
    ax.set_title("Validation Mean IoU (mIoU) Progression", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "miou_curve.png"))
    plt.close(fig)

    # 3. Overall 4-panel Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # Subplot 1: Loss
    if "train_loss" in history:
        axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4")
    if "val_loss" in history:
        axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e")
    axes[0, 0].set_title("Loss Evolution")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    # Subplot 2: mIoU
    if "val_mIoU" in history:
        axes[0, 1].plot(epochs, [v * 100 for v in history["val_mIoU"]], label="Overall mIoU", color="#2ca02c")
    if "val_mIoU_fg" in history:
        axes[0, 1].plot(epochs, [v * 100 for v in history["val_mIoU_fg"]], label="Foreground mIoU", color="#9467bd", linestyle="--")
    axes[0, 1].set_title("Mean IoU Progression (%)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("mIoU (%)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    # Subplot 3: Pixel Accuracy
    if "val_pixel_acc" in history:
        axes[1, 0].plot(epochs, [v * 100 for v in history["val_pixel_acc"]], label="Pixel Accuracy", color="#d62728")
        axes[1, 0].set_title("Pixel Accuracy Progression (%)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)
        axes[1, 0].legend()

    # Subplot 4: Learning Rate
    if "lr" in history:
        axes[1, 1].plot(epochs, history["lr"], label="Learning Rate", color="#8c564b")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("LR")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)
        axes[1, 1].legend()

    fig.suptitle("CSFNet-1 (STDC1) RGBT Training Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "training_dashboard.png"))
    plt.close(fig)

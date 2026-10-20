"""Evaluation metrics for RGBT Semantic Segmentation.

Tracks:
- Overall Pixel Accuracy
- Per-class IoU for all 9 classes (0..8)
- Mean IoU (mIoU) across all 9 classes
- Foreground Mean IoU (mIoU_fg) across classes 1..8
- Complete 9x9 Confusion Matrix
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union

CLASS_NAMES = [
    "unlabeled",   # 0
    "car",         # 1
    "person",      # 2
    "bike",        # 3
    "curve",       # 4
    "car_stop",    # 5
    "guardrail",   # 6
    "color_cone",  # 7
    "bump"         # 8
]


class SegmentationMetrics:
    """Streaming metric accumulator computing confusion matrix, per-class IoU, and mIoU."""

    def __init__(self, num_classes: int = 9, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None else CLASS_NAMES[:num_classes]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        """Reset the confusion matrix to zero."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def _fast_hist(self, label_true: np.ndarray, label_pred: np.ndarray) -> np.ndarray:
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(np.int64) + label_pred[mask].astype(np.int64),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, preds: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]):
        """Accumulate predictions and ground-truth targets.

        Args:
            preds: (B, H, W) or (H, W) array of predicted class IDs.
            targets: (B, H, W) or (H, W) array of ground-truth class IDs.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds = preds.flatten()
        targets = targets.flatten()
        self.confusion_matrix += self._fast_hist(targets, preds)

    def compute(self) -> Dict[str, Any]:
        """Compute pixel accuracy, per-class IoU, mIoU, and foreground mIoU.

        Returns:
            Dict containing:
                'pixel_accuracy': float
                'class_iou': Dict[str, float]
                'mIoU': float (overall across all classes 0..8)
                'mIoU_fg': float (foreground classes 1..8)
                'confusion_matrix': np.ndarray (num_classes, num_classes)
        """
        hist = self.confusion_matrix
        total = hist.sum()
        if total == 0:
            return {
                "pixel_accuracy": 0.0,
                "class_iou": {name: 0.0 for name in self.class_names},
                "mIoU": 0.0,
                "mIoU_fg": 0.0,
                "confusion_matrix": hist
            }

        pixel_accuracy = float(np.diag(hist).sum() / total)

        denominator = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iu = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            if denominator[i] > 0:
                iu[i] = hist[i, i] / denominator[i]
            else:
                iu[i] = float("nan")

        class_iou = {
            self.class_names[i]: float(iu[i]) if not np.isnan(iu[i]) else 0.0
            for i in range(self.num_classes)
        }

        # Overall mIoU (including unlabeled 0)
        valid_iu = iu[~np.isnan(iu)]
        mean_iou = float(np.mean(valid_iu)) if len(valid_iu) > 0 else 0.0

        # Foreground mIoU (classes 1..8)
        fg_iu = iu[1:][~np.isnan(iu[1:])]
        mean_iou_fg = float(np.mean(fg_iu)) if len(fg_iu) > 0 else 0.0

        return {
            "pixel_accuracy": pixel_accuracy,
            "class_iou": class_iou,
            "mIoU": mean_iou,
            "mIoU_fg": mean_iou_fg,
            "confusion_matrix": hist
        }

    def get_summary_table(self) -> str:
        """Return formatted ASCII table of metrics."""
        results = self.compute()
        lines = []
        lines.append(f"{'Class ID':<8} {'Class Name':<15} {'IoU (%)':<10}")
        lines.append("-" * 35)

        for i, name in enumerate(self.class_names):
            iou_val = results["class_iou"][name] * 100
            lines.append(f"{i:<8} {name:<15} {iou_val:<10.2f}")

        lines.append("-" * 35)
        lines.append(f"Pixel Accuracy   : {results['pixel_accuracy'] * 100:.2f}%")
        lines.append(f"Mean IoU (all 9) : {results['mIoU'] * 100:.2f}%")
        lines.append(f"Foreground mIoU  : {results['mIoU_fg'] * 100:.2f}%")
        return "\n".join(lines)



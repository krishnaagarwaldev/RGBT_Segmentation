"""Loss functions for class-imbalanced RGBT semantic segmentation.

Includes:
1. Weighted CrossEntropy Loss (with principled weight smoothing).
2. Multi-Class Dice Loss (scale-invariant regional IoU optimization).
3. Multi-Class Focal Loss (down-weights well-classified majority background pixels).
4. Combined Loss (CrossEntropy/Focal + Dice Loss).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

from data.class_weights import compute_class_weights


class MultiClassDiceLoss(nn.Module):
    """Soft Multi-Class Dice Loss for semantic segmentation.
    
    Directly optimizes region overlap (IoU proxy), inherently robust to class imbalance.
    """
    def __init__(self, smooth: float = 1e-5, ignore_index: Optional[int] = None):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw unnormalized model predictions.
            targets: (B, H, W) integer ground-truth class IDs.
        """
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets: (B, C, H, W)
        targets_one_hot = F.one_hot(targets.clamp(min=0, max=num_classes - 1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.ignore_index is not None and 0 <= self.ignore_index < num_classes:
            valid_mask = torch.ones(num_classes, dtype=torch.bool, device=logits.device)
            valid_mask[self.ignore_index] = False
            dice = dice[valid_mask]

        loss = 1.0 - torch.mean(dice)
        return loss


class FocalLoss(nn.Module):
    """Multi-Class Focal Loss.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy background examples so model focuses on hard minority classes.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index if ignore_index is not None else -100

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Principled combination of Cross-Entropy / Focal and Multi-Class Dice Loss.
    
    Loss = L_ce + dice_weight * L_dice
    """
    def __init__(
        self,
        ce_weight: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        ignore_index: Optional[int] = None
    ):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.dice_loss = MultiClassDiceLoss(ignore_index=ignore_index)
        
        if use_focal:
            self.base_loss = FocalLoss(gamma=focal_gamma, weight=ce_weight, ignore_index=ignore_index)
        else:
            ig = ignore_index if ignore_index is not None else -100
            self.base_loss = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ig)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return base + (self.dice_weight * dice)


def build_loss(
    loss_type: str = "combined_ce_dice",
    weight_strategy: str = "enet_log",
    dice_weight: float = 0.5,
    focal_gamma: float = 2.0,
    ignore_index: Optional[int] = None,
    device: str = "cuda"
) -> nn.Module:
    """Factory function for instantiating the selected loss module."""
    class_weights = compute_class_weights(strategy=weight_strategy)
    if device == "cuda" and torch.cuda.is_available():
        class_weights = class_weights.cuda()

    if loss_type == "ce":
        ig = ignore_index if ignore_index is not None else -100
        return nn.CrossEntropyLoss(ignore_index=ig)

    elif loss_type == "weighted_ce":
        ig = ignore_index if ignore_index is not None else -100
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ig)

    elif loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights, ignore_index=ignore_index)

    elif loss_type == "dice":
        return MultiClassDiceLoss(ignore_index=ignore_index)

    elif loss_type == "combined_ce_dice":
        return CombinedLoss(
            ce_weight=class_weights,
            dice_weight=dice_weight,
            use_focal=False,
            ignore_index=ignore_index
        )

    elif loss_type == "combined_focal_dice":
        return CombinedLoss(
            ce_weight=class_weights,
            dice_weight=dice_weight,
            use_focal=True,
            focal_gamma=focal_gamma,
            ignore_index=ignore_index
        )

    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

"""Class distribution analysis and loss weighting strategies for MFNet / ir_seg_dataset.

Class Pixel Frequencies (computed across 784 train images, 240,844,800 pixels):
  0: unlabeled   221,252,408 (91.8651%)
  1: car          10,902,426 ( 4.5267%)
  2: person        3,324,364 ( 1.3803%)
  3: bike          1,027,788 ( 0.4267%)
  4: curve         1,548,814 ( 0.6431%)
  5: car_stop      1,313,666 ( 0.5454%)
  6: guardrail       363,618 ( 0.1510%)
  7: color_cone      422,237 ( 0.1753%)
  8: bump            689,479 ( 0.2863%)

Why Raw Inverse-Frequency Weights Cause Instability:
  Raw inverse frequency gives:
    w_0 = 1.09, w_1 = 22.1, w_2 = 72.4, w_3 = 234.3, w_6 (guardrail) = 661.9.
  A weight of 660x produces massive gradient spikes during backpropagation whenever
  guardrail pixels are encountered. This causes gradient explosions, unstable loss
  dynamics, and an overwhelming surge in false-positive predictions.

Principled Weighting Strategies Implemented:
  1. ENet Logarithmic Weighting:
     w_c = 1 / ln(c + f_c)
     Smooths out extremes while giving controlled emphasis to rare classes.
  2. Median Frequency Balancing:
     w_c = median(f) / f_c, clipped to [0.1, 10.0].
  3. Power-Law Smoothed Weighting:
     w_c = (1 / f_c)^0.5, normalized so sum(w) = num_classes.
  4. Combined Loss Strategy:
     Pairing smoothed CrossEntropy with Multi-Class Dice Loss.
     Dice Loss is inherently scale-invariant, directly optimizing regional overlap
     (intersection over union) without requiring extreme manual weights.
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple


# Empirical pixel counts and frequencies on ir_seg_dataset training split
TRAIN_PIXEL_COUNTS = [
    221252408,  # 0: unlabeled
    10902426,   # 1: car
    3324364,    # 2: person
    1027788,    # 3: bike
    1548814,    # 4: curve
    1313666,    # 5: car_stop
    363618,     # 6: guardrail
    422237,     # 7: color_cone
    689479      # 8: bump
]

TOTAL_TRAIN_PIXELS = sum(TRAIN_PIXEL_COUNTS)  # 240,844,800
TRAIN_FREQUENCIES = [cnt / TOTAL_TRAIN_PIXELS for cnt in TRAIN_PIXEL_COUNTS]

CLASS_NAMES = [
    "unlabeled",
    "car",
    "person",
    "bike",
    "curve",
    "car_stop",
    "guardrail",
    "color_cone",
    "bump"
]


def compute_class_weights(
    strategy: str = "enet_log",
    c_param: float = 1.02,
    power: float = 0.5,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
    normalize: bool = True
) -> torch.Tensor:
    """Compute normalized class weights based on a chosen strategy.

    Args:
        strategy: 'none', 'enet_log', 'median_frequency', 'power_smoothed', 'inverse_frequency'.
        c_param: Constant c for ENet log weighting: 1 / ln(c + f_c).
        power: Power exponent for power-law smoothing.
        min_weight: Lower clipping bound.
        max_weight: Upper clipping bound.
        normalize: Whether to normalize weights so their mean is 1.0.

    Returns:
        torch.Tensor of shape (num_classes,) as float32.
    """
    freqs = np.array(TRAIN_FREQUENCIES, dtype=np.float64)
    num_classes = len(freqs)

    if strategy == "none" or strategy is None:
        weights = np.ones(num_classes, dtype=np.float32)
        return torch.from_numpy(weights)

    elif strategy == "enet_log":
        # ENet style: w_c = 1 / ln(c + f_c)
        weights = 1.0 / np.log(c_param + freqs)

    elif strategy == "median_frequency":
        # Median frequency balancing: w_c = median(f) / f_c
        med = np.median(freqs)
        weights = med / (freqs + 1e-7)

    elif strategy == "power_smoothed":
        # Power-law smoothed: w_c = (1 / f_c)^power
        weights = (1.0 / (freqs + 1e-7)) ** power

    elif strategy == "inverse_frequency":
        # Raw inverse frequency (with clipping for safety)
        weights = 1.0 / (freqs + 1e-7)

    else:
        raise ValueError(f"Unknown class weighting strategy: {strategy}")

    # Clip weights to prevent numeric instability
    weights = np.clip(weights, min_weight, max_weight)

    if normalize:
        weights = weights / np.mean(weights)

    return torch.tensor(weights, dtype=torch.float32)


def get_class_distribution_summary() -> str:
    """Return a formatted string table of class distributions and computed weights."""
    header = f"{'ID':<4} {'Class':<12} {'Train Count':<14} {'Freq (%)':<10} {'ENet Log':<10} {'Med Freq':<10} {'Power-0.5':<10}"
    divider = "-" * len(header)
    lines = [header, divider]

    w_enet = compute_class_weights("enet_log").numpy()
    w_med = compute_class_weights("median_frequency").numpy()
    w_pow = compute_class_weights("power_smoothed").numpy()

    for idx, name in enumerate(CLASS_NAMES):
        cnt = TRAIN_PIXEL_COUNTS[idx]
        freq = TRAIN_FREQUENCIES[idx] * 100
        line = (
            f"{idx:<4} {name:<12} {cnt:<14,d} {freq:<10.3f} "
            f"{w_enet[idx]:<10.3f} {w_med[idx]:<10.3f} {w_pow[idx]:<10.3f}"
        )
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    print(get_class_distribution_summary())

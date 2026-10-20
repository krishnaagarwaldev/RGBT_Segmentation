"""Cosine Similarity Attention Fusion Module (CS-AFM) and Decoder Fusion for CSFNet.

References:
"Cosine Similarity Fusion Network for Real-Time RGB-X Semantic Segmentation of Driving Scenes"
Uses Cosine Similarity between multi-modal feature vectors to calculate adaptive attention
weights without heavy parameters, enabling ultra-fast real-time multimodal fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FusionAttention(nn.Module):
    """Cosine Similarity Attention Fusion Module (CS-AFM)."""
    def __init__(self, num_channel: int, out_channel: int, pool_size: Tuple[int, int]):
        super(FusionAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.conv1 = nn.Conv2d(num_channel, out_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, num_channel, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial pooling to pool_size and reshape for cosine similarity
        rgb_pooled = self.pool(rgb).view(rgb.size(0), rgb.size(1), -1)
        depth_pooled = self.pool(depth).view(depth.size(0), depth.size(1), -1)

        # Compute cosine similarity across channels
        similarity_vector = self.cos(rgb_pooled, depth_pooled).view(rgb.size(0), rgb.size(1), 1, 1)

        # 1x1 convs to generate attention weight map
        feat = F.relu(self.bn1(self.conv1(similarity_vector)), inplace=True)
        feat = self.conv2(feat)
        weight = self.activation(feat)

        return weight, 1.0 - weight


class EncoderFusion(nn.Module):
    """Encoder cross-modal fusion stage using CS-AFM."""
    def __init__(
        self,
        num_channel: int,
        hid_channels: int,
        pool_size: Tuple[int, int],
        skip_connection: bool = True,
        last_fusion: bool = False
    ):
        super(EncoderFusion, self).__init__()
        self.attention = FusionAttention(num_channel, hid_channels, pool_size)
        self.skip_connection = skip_connection
        self.last_fusion = last_fusion

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        wd, wr = self.attention(rgb, depth)
        w_rgb = rgb.mul(wr)
        w_depth = depth.mul(wd)

        if self.skip_connection and not self.last_fusion:
            rgb = rgb + w_depth
            depth = depth + w_rgb
            sc = w_rgb + w_depth
            return rgb, depth, sc

        if not self.skip_connection and not self.last_fusion:
            rgb = rgb + w_depth
            depth = depth + w_rgb
            return rgb, depth

        if self.skip_connection and self.last_fusion:
            # Transitions the dual-branch into a single-branch
            sc = w_rgb + w_depth
            return sc


class DecoderFusion(nn.Module):
    """Decoder multi-level feature fusion."""
    def __init__(self, num_channel: int, hid_channels: int, pool_size: Tuple[int, int]):
        super(DecoderFusion, self).__init__()
        self.attention = FusionAttention(num_channel, hid_channels, pool_size)

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.attention(x_high, x_low)
        return (alpha * x_high) + (beta * x_low)

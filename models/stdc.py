"""STDC1 (STDCNet813) Backbone for CSFNet-1.

Reference: Short-Term Dense Concatenate Network (STDC)
Specifically STDCNet813 with CatBottleneck blocks, base channels 64, layers=[2, 2, 2].
"""
import os
import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional


class ConvX(nn.Module):
    """Standard Convolution-BatchNorm-ReLU block."""
    def __init__(self, in_planes: int, out_planes: int, kernel: int = 3, stride: int = 1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class CatBottleneck(nn.Module):
    """Short-Term Dense Concatenate (STDC) bottleneck block with concatenation."""
    def __init__(self, in_planes: int, out_planes: int, block_num: int = 4, stride: int = 1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.ModuleList()
        self.stride = stride
        
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False
                ),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1)))
                )
            else:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx)))
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


def make_stdc_layers(in_channels: int, base: int = 64, layers: List[int] = [2, 2, 2], block_num: int = 4) -> nn.Sequential:
    """Build multi-stage STDC backbone layers."""
    features = []
    features.append(ConvX(in_channels, base // 2, 3, 2))  # Stage 1: stride 2
    features.append(ConvX(base // 2, base, 3, 2))         # Stage 2: stride 4

    # Stages 3, 4, 5
    for i, layer in enumerate(layers):
        for j in range(layer):
            if i == 0 and j == 0:
                features.append(CatBottleneck(base, base * 4, block_num, 2))
            elif j == 0:
                features.append(CatBottleneck(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
            else:
                features.append(CatBottleneck(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

    return nn.Sequential(*features)


def load_pretrained_stdc_weights(
    features_rgb: nn.Sequential,
    features_thermal: Optional[nn.Sequential] = None,
    weight_path: str = "pretrained/STDCNet1446_76.47.tar"
) -> Tuple[int, int]:
    """Load official ImageNet pretrained weights into STDC RGB and Thermal streams.
    
    For RGB stream:
        Direct 1-to-1 transfer of all feature layers.
    For Thermal stream:
        Initializes first conv layer by taking the mean across the 3 RGB channels,
        and directly transfers all subsequent feature layers.
    
    Returns:
        (matched_rgb, matched_thermal): counts of loaded parameters
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Pretrained weights not found at: {weight_path}")

    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Strip prefixes like 'features.' if needed
    cleaned_dict = {}
    for k, v in state_dict.items():
        if k.startswith("features."):
            cleaned_dict[k[len("features."):]] = v
        else:
            cleaned_dict[k] = v

    # 1. Load RGB stream
    rgb_dict = features_rgb.state_dict()
    matched_rgb = 0
    rgb_updates = {}
    for k, v in rgb_dict.items():
        if k in cleaned_dict and cleaned_dict[k].shape == v.shape:
            rgb_updates[k] = cleaned_dict[k]
            matched_rgb += 1

    rgb_dict.update(rgb_updates)
    features_rgb.load_state_dict(rgb_dict)

    # 2. Load Thermal stream (if provided)
    matched_thermal = 0
    if features_thermal is not None:
        thermal_dict = features_thermal.state_dict()
        thermal_updates = {}
        for k, v in thermal_dict.items():
            if k == "0.conv.weight":
                # First layer has 1 channel instead of 3: average across RGB input channels
                pretrained_w = cleaned_dict.get(k)
                if pretrained_w is not None and pretrained_w.dim() == 4:
                    thermal_updates[k] = pretrained_w.mean(dim=1, keepdim=True)
                    matched_thermal += 1
            elif k in cleaned_dict and cleaned_dict[k].shape == v.shape:
                thermal_updates[k] = cleaned_dict[k]
                matched_thermal += 1

        thermal_dict.update(thermal_updates)
        features_thermal.load_state_dict(thermal_dict)

    return matched_rgb, matched_thermal


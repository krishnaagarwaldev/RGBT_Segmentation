"""Complete CSFNet-1 (STDC1) Architecture for RGBT Semantic Segmentation.

Architecture Highlights:
1. Dual-to-Single Branch Encoder:
   - Stage 1 (stride 2): RGB (3 ch) & Thermal (1 ch) extracted by STDC layers; fused via CS-AFM (f1).
   - Stage 2 (stride 4): Extracted by STDC layers; fused via CS-AFM (f2).
   - Stage 3 (stride 8): Extracted by STDC layers; merged into single branch via CS-AFM (f3).
   - Stages 4 (stride 16) & 5 (stride 32): Single-branch high-level semantic representation.
2. Context Module (CM):
   - Multi-kernel asymmetric strip pooling to capture wide horizontal & vertical driving context.
3. Lightweight Decoder:
   - Progressive bilateral upsampling and cross-level fusion with skip connections.
   - Outputs logits (B, num_classes, H, W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any

from models.stdc import ConvX, make_stdc_layers, load_pretrained_stdc_weights
from models.fusion import EncoderFusion, DecoderFusion


class ConvBNAct(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU block with flexible kernel/padding."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False
    ):
        if isinstance(kernel_size, (list, tuple)):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        else:
            padding = (kernel_size - 1) // 2 * dilation

        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ContextModule(nn.Module):
    """Context Module (CM) capturing multi-scale strip pooling context."""
    def __init__(self, in_channels: int = 1024, out_channels: int = 32, pooling_size: Tuple[int, int] = (5, 5)):
        super(ContextModule, self).__init__()
        hid_channels = int(in_channels // 4)

        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(pooling_size),
            ConvBNAct(in_channels, hid_channels, kernel_size=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hid_channels, 64, kernel_size=(1, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hid_channels, 64, kernel_size=(4, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()[2:]
        p = self.pool1(x)

        x1 = F.interpolate(self.conv2(p), size=size, mode="bilinear", align_corners=True)
        x2 = F.interpolate(self.conv3(p), size=size, mode="bilinear", align_corners=True)
        out = self.conv_out(x1 + x2)
        return out


class Decoder(nn.Module):
    """Lightweight progressive decoder for CSFNet."""
    def __init__(
        self,
        decoder_channels: List[int] = [32, 64, 128, 32, 9],
        pool_out: List[Tuple[int, int]] = [(16, 24), (8, 12), (4, 6), (2, 3), (5, 5)]
    ):
        super(Decoder, self).__init__()
        self.CBR1 = ConvBNAct(decoder_channels[0], decoder_channels[0])
        self.CBR2 = ConvBNAct(decoder_channels[0], decoder_channels[1])
        self.CBR3 = ConvBNAct(decoder_channels[1], decoder_channels[2])
        self.CBR4 = ConvBNAct(decoder_channels[2], decoder_channels[3])
        self.CBR5 = ConvBNAct(decoder_channels[3], decoder_channels[4])

        self.D_fusion1 = DecoderFusion(32, 16, pool_out[3])  # (2, 3)
        self.D_fusion2 = DecoderFusion(64, 32, pool_out[2])  # (4, 6)
        self.D_fusion3 = DecoderFusion(32, 16, pool_out[0])  # (16, 24)

    def forward(self, x1: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor) -> torch.Tensor:
        x = self.CBR1(x5)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.D_fusion1(x, x4)

        x = self.CBR2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.D_fusion2(x, x3)

        x = self.CBR3(x)
        x = self.CBR4(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        x = self.D_fusion3(x, x1)

        x = self.CBR5(x)
        out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return out


class DualEncoderSTDC(nn.Module):
    """Dual-to-Single Branch Encoder supporting both STDC1 and STDC2 backbones."""
    def __init__(
        self,
        base: int = 64,
        layers: List[int] = [2, 2, 2],
        block_num: int = 4,
        pool_out: List[Tuple[int, int]] = [(16, 24), (8, 12), (4, 6), (2, 3), (5, 5)]
    ):
        super(DualEncoderSTDC, self).__init__()
        self.layers = layers
        # RGB stream (3 channels)
        self.features_rgb = make_stdc_layers(in_channels=3, base=base, layers=layers, block_num=block_num)
        # Thermal stream (1 channel)
        self.features_thermal = make_stdc_layers(in_channels=1, base=base, layers=layers, block_num=block_num)

        # Slice stages dynamically according to layer configuration
        l1, l2, l3 = layers
        self.x2 = nn.Sequential(self.features_rgb[:1])
        self.x4 = nn.Sequential(self.features_rgb[1:2])
        self.x8 = nn.Sequential(self.features_rgb[2 : 2 + l1])
        self.x16 = nn.Sequential(self.features_rgb[2 + l1 : 2 + l1 + l2])
        self.x32 = nn.Sequential(self.features_rgb[2 + l1 + l2 :])

        self.x2d = nn.Sequential(self.features_thermal[:1])
        self.x4d = nn.Sequential(self.features_thermal[1:2])
        self.x8d = nn.Sequential(self.features_thermal[2 : 2 + l1])

        # CS-AFM cross-modal fusions
        self.f1 = EncoderFusion(32, 16, pool_out[0], skip_connection=True, last_fusion=False)
        self.f2 = EncoderFusion(64, 32, pool_out[1], skip_connection=False, last_fusion=False)
        self.f3 = EncoderFusion(256, 128, pool_out[2], skip_connection=True, last_fusion=True)

        # 1x1 conv agents for skip dimension matching
        self.agent1 = ConvBNAct(32, 32, kernel_size=1)
        self.agent3 = ConvBNAct(256, 64, kernel_size=1)
        self.agent4 = ConvBNAct(512, 32, kernel_size=1)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stage 1: stride 2
        rgb = self.x2(rgb)
        thermal = self.x2d(thermal)
        rgb, thermal, s1 = self.f1(rgb, thermal)
        x1 = self.agent1(s1)

        # Stage 2: stride 4
        rgb = self.x4(rgb)
        thermal = self.x4d(thermal)
        rgb, thermal = self.f2(rgb, thermal)

        # Stage 3: stride 8 -> merges into single branch
        rgb = self.x8(rgb)
        thermal = self.x8d(thermal)
        s3 = self.f3(rgb, thermal)
        x3 = self.agent3(s3)

        # Stage 4: stride 16 (single branch)
        s4 = self.x16(s3)
        x4 = self.agent4(s4)

        # Stage 5: stride 32 (single branch)
        x5 = self.x32(s4)

        return x1, x3, x4, x5


# Alias for backward compatibility
DualEncoderSTDC1 = DualEncoderSTDC


class CSFNet(nn.Module):
    """Universal CSFNet model supporting CSFNet-1 (STDC1) and CSFNet-2 (STDC2)."""
    def __init__(
        self,
        backbone: str = "STDCNet1446",
        num_classes: int = 9,
        pool_out: List[Tuple[int, int]] = [(16, 24), (8, 12), (4, 6), (2, 3), (5, 5)],
        pretrained: bool = False,
        pretrained_path: Optional[str] = None
    ):
        super(CSFNet, self).__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Select backbone configuration
        if backbone.lower() in ["stdcnet1446", "stdc2", "csfnet2", "csfnet-2"]:
            layers = [4, 5, 3]
            default_weight = "pretrained/STDCNet1446_76.47.tar"
        else:
            layers = [2, 2, 2]
            default_weight = "pretrained/STDCNet813M_73.91.tar"

        decoder_channels = [32, 64, 128, 32, num_classes]

        self.encoder = DualEncoderSTDC(base=64, layers=layers, block_num=4, pool_out=pool_out)
        self.CM = ContextModule(in_channels=1024, out_channels=decoder_channels[0], pooling_size=pool_out[4])
        self.decoder = Decoder(decoder_channels=decoder_channels, pool_out=pool_out)

        self._init_weights()

        if pretrained:
            path_to_load = pretrained_path if pretrained_path else default_weight
            matched_rgb, matched_th = load_pretrained_stdc_weights(
                self.encoder.features_rgb,
                self.encoder.features_thermal,
                weight_path=path_to_load
            )
            print(f"[CSFNet] Loaded ImageNet pretrained backbone from '{path_to_load}' "
                  f"(RGB: {matched_rgb} params, Thermal: {matched_th} params)")

    def _init_weights(self):
        """Initialize weights with He/Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: Tensor of shape (B, 3, H, W).
            thermal: Tensor of shape (B, 1, H, W).
        Returns:
            logits: Tensor of shape (B, num_classes, H, W).
        """
        x1, x3, x4, x5 = self.encoder(rgb, thermal)
        x5 = self.CM(x5)
        logits = self.decoder(x1, x3, x4, x5)
        return logits


# Alias for backward compatibility
CSFNet1 = CSFNet


def build_csfnet(
    backbone: str = "STDCNet1446",
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    num_classes: int = 9,
    dataset: str = "MFNet"
) -> CSFNet:
    """Factory to build CSFNet-1 or CSFNet-2."""
    if dataset == "MFNet":
        pool_out = [(16, 24), (8, 12), (4, 6), (2, 3), (5, 5)]
    else:
        pool_out = [(16, 32), (8, 16), (4, 8), (2, 4), (4, 8)]

    return CSFNet(
        backbone=backbone,
        num_classes=num_classes,
        pool_out=pool_out,
        pretrained=pretrained,
        pretrained_path=pretrained_path
    )


def build_csfnet1(num_classes: int = 9, dataset: str = "MFNet") -> CSFNet:
    """Backward compatibility helper for CSFNet-1 from scratch."""
    return build_csfnet(backbone="STDCNet813", pretrained=False, num_classes=num_classes, dataset=dataset)


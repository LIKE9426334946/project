import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

from models.attention import ASPP, ConvBNReLU


class UNetResNet34Attn(nn.Module):
    def __init__(
        self,
        num_classes: int = 4, #! 类别为4
        in_channels: int = 3,
        pretrained: bool = True,
        use_scse: bool = True,
        use_aspp: bool = True,
    ):
        super().__init__()
        _ = use_scse
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        # DeepLabV3+ encoder head
        self.aspp = ASPP(512, 256) if use_aspp else ConvBNReLU(512, 256)

        # Low-level feature projection (from encoder1)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # DeepLabV3+ decoder
        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256),
            ConvBNReLU(256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        e0 = self.stem(x)
        e1 = self.encoder1(self.maxpool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        high = self.aspp(e4)
        high = F.interpolate(high, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_level_proj(e1)
        x = self.decoder(torch.cat([high, low], dim=1))
        x = self.classifier(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

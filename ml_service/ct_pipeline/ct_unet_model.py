# ct_unet_model.py
"""
UNet model for lung or nodule segmentation on CT slices.
This is a clean, production-level PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Building Blocks
# -------------------------

class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    """Upsample → Concatenate → DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Padding in case sizes differ due to pooling rounding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1×1 convolution"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out(x)


# -------------------------
# UNet Model
# -------------------------

class UNet(nn.Module):
    """Standard 2D UNet for segmentation"""

    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.in_conv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.out_conv(x)
        return logits


# -------------------------
# Quick Test
# -------------------------

if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=1)
    x = torch.randn(1, 1, 512, 512)  # Example CT slice
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)

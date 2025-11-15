# ct_classifier.py
"""
3D CNN classifier for lung cancer probability from CT scans.

Input:
    1 × D × H × W CT volume (resampled, normalized)

Output:
    Cancer risk score (0–1)

Model:
    3D ResNet-like architecture (lightweight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# Basic 3D CNN Block
# -------------------------------------------------------

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# -------------------------------------------------------
# 3D ResNet Classifier
# -------------------------------------------------------

class CT3DClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(CT3DClassifier, self).__init__()

        self.layer1 = BasicBlock3D(1, 32, stride=2)
        self.layer2 = BasicBlock3D(32, 64, stride=2)
        self.layer3 = BasicBlock3D(64, 128, stride=2)
        self.layer4 = BasicBlock3D(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x shape: (B, 1, D, H, W)
        """
        x = self.layer1(x)  # (B, 32, D/2, ...)
        x = self.layer2(x)  # (B, 64, ...)
        x = self.layer3(x)  # (B, 128, ...)
        x = self.layer4(x)  # (B, 256, ...)

        x = self.avgpool(x)  # (B, 256, 1, 1, 1)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return torch.sigmoid(x)  # cancer probability (0–1)


# -------------------------------------------------------
# Quick Test
# -------------------------------------------------------

if __name__ == "__main__":
    print("Testing 3D CT Classifier...")

    model = CT3DClassifier()
    example = torch.randn(1, 1, 64, 128, 128)  # (batch, channel, depth, height, width)

    out = model(example)
    print("Output shape:", out.shape)
    print("Cancer probability:", out.item())

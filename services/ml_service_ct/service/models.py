# services/ml_service_ct/service/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, embedding_dim=512):
        super().__init__()
        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.enc2 = ConvBlock3D(base_ch, base_ch*2)
        self.enc3 = ConvBlock3D(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool3d(2)
        self.bottle = ConvBlock3D(base_ch*4, base_ch*8)
        self.up3 = UpBlock3D(base_ch*8, base_ch*4)
        self.up2 = UpBlock3D(base_ch*4, base_ch*2)
        self.up1 = UpBlock3D(base_ch*2, base_ch)
        # segmentation head
        self.seg_head = nn.Conv3d(base_ch, 1, 1)
        # malignancy/classifier head for each detected region (we'll use pooled features)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_ch*8, embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, 4), # 4 classes
            nn.Softmax(dim=1)
        )
        # global embedding
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_ch*8, embedding_dim),
            nn.ReLU()
        )
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottle(self.pool(e3))
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        seg = torch.sigmoid(self.seg_head(d1))
        cls = self.class_head(b)
        emb = self.embedding_head(b)
        return {"seg": seg, "class_probs": cls, "embedding": emb}

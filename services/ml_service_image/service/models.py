import torch
import torch.nn as nn
from torchvision import models


class ImageCancerModel(nn.Module):
    """
    Simple ResNet-18 classifier for PNG/JPG lung cancer detection.
    Classes:
      0: normal
      1: suspicious
      2: malignant
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

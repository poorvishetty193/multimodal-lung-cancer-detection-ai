# models.py
import torch
import torch.nn as nn
from torchvision import models

class Classifier(nn.Module):
    def __init__(self, n_classes=4, pretrained=True):
        super().__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.model(x)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, n_classes=4):
    model = Classifier(n_classes=n_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

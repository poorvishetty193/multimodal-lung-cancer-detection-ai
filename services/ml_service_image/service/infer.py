# infer.py
import torch
from torchvision import transforms
from PIL import Image
from models import load_model

def infer(model_path, img_path, classes, img_size=224):
    model = load_model(model_path, n_classes=len(classes))

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze()

    return {cls: float(probs[i]) for i, cls in enumerate(classes)}

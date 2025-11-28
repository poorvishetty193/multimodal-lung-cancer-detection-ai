import torch
from PIL import Image
from torchvision import transforms
from models import ImageCancerModel

LABELS = ["normal", "suspicious", "malignant"]


def load_model(model_path="image_model.pth"):
    model = ImageCancerModel(num_classes=len(LABELS))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict_image(image_path: str, model=None):
    if model is None:
        model = load_model()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    result = {
        "prediction": LABELS[int(probs.argmax())],
        "probabilities": {
            label: float(p) for label, p in zip(LABELS, probs)
        }
    }
    return result

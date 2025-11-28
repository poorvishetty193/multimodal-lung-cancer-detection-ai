import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from models import ImageCancerModel


def train_model(data_dir="data", epochs=5, lr=1e-4, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ImageCancerModel(num_classes=len(dataset.classes)).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = F.cross_entropy(preds, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "image_model.pth")
    print("Model saved → image_model.pth")


if __name__ == "__main__":
    train_model()

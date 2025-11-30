# train.py
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import Classifier, save_model
from utils import get_transforms


def make_dataloaders(data_root, img_size=224, batch_size=16, num_workers=2):
    train_tf, infer_tf = get_transforms(img_size)

    train_path = os.path.join(data_root, "train")
    valid_path = os.path.join(data_root, "valid")

    if not os.path.exists(train_path):
        raise RuntimeError(f"Training folder not found: {train_path}")

    if not os.path.exists(valid_path):
        raise RuntimeError(f"Validation folder not found: {valid_path}")

    train_ds = ImageFolder(train_path, transform=train_tf)
    valid_ds = ImageFolder(valid_path, transform=infer_tf)

    print("\nDetected classes:", train_ds.classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, valid_loader, train_ds.classes


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, valid_loader, classes = make_dataloaders(
        args.data, args.img_size, args.batch_size, args.num_workers
    )

    n_classes = len(classes)
    model = Classifier(n_classes=n_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n------ Epoch {epoch}/{args.epochs} ------")

        # --------------------- TRAIN ---------------------
        model.train()
        total_train_loss = 0

        for imgs, labels in tqdm(train_loader, desc="Training"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * imgs.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        # --------------------- VALIDATION ---------------------
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(valid_loader, desc="Validating"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = total_val_loss / len(valid_loader.dataset)
        val_acc = correct / total

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # --------------------- SAVE BEST ---------------------
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, "image_classifier.pt")
            save_model(model, save_path)
            print("✔ Saved BEST model →", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to Data folder")
    parser.add_argument("--out-dir", default="models", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()
    train(args)

"""
train_disease_model.py
──────────────────────
Train the Green Gram / Rice leaf disease CNN (ResNet18, transfer learning).

DATASET SETUP (do this first):
─────────────────────────────
Option A — Rice Disease Dataset (recommended for green gram, similar diseases):
    Download: https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset
    After download, your folder should look like:
        dataset/
            Bacterial Leaf Blight/   (images here)
            Brown Spot/
            Healthy/
            Leaf Blast/

Option B — PlantVillage (larger, more classes):
    Download: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
    Use the rice/legume subset.

Usage:
    pip install torch torchvision scikit-learn Pillow
    python train_disease_model.py --data ./dataset --epochs 15

Output:
    ml_models/disease_model.pth          ← paste here
    ml_models/disease_model_accuracy.txt ← accuracy stored here
    ml_models/disease_classes.txt        ← class names (order matters!)
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = Path(__file__).resolve().parent / 'ml_models'


def train(data_dir, epochs=15, batch_size=32, lr=1e-4):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"ERROR: Dataset folder not found: {data_dir}")
        print("Download from: https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset")
        return

    # ── Data transforms ────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = ImageFolder(data_dir, transform=train_transform)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)
    print(f"\n[1/5] Dataset loaded")
    print(f"      Classes ({num_classes}): {class_names}")
    print(f"      Total images: {len(full_dataset)}")

    # ── Train/val split 80/20 ──────────────────────────────────────────────
    val_size   = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    # Apply val transform to val set
    val_set.dataset = ImageFolder(data_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Device: {device}")

    # ── Model: ResNet18 with transfer learning ──────────────────────────────
    print("\n[2/5] Loading ResNet18 (pretrained on ImageNet)...")
    model = models.resnet18(pretrained=True)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace final FC layer — only this trains in phase 1
    model.fc = nn.Linear(512, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Phase 1: train only the final layer (5 epochs)
    optimizer_phase1 = torch.optim.Adam(model.fc.parameters(), lr=lr)

    # Phase 2: unfreeze all and fine-tune with lower LR
    optimizer_phase2 = torch.optim.Adam(model.parameters(), lr=lr * 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=epochs)

    best_acc = 0.0

    print(f"\n[3/5] Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()

        # Phase 1: first 5 epochs — only fine-tune FC layer
        if epoch < 5:
            optimizer = optimizer_phase1
            if epoch == 0:
                print("      Phase 1: Training final layer only (epochs 1–5)")
        else:
            # Unfreeze all at epoch 5
            if epoch == 5:
                print("      Phase 2: Fine-tuning entire network (epochs 6+)")
                for param in model.parameters():
                    param.requires_grad = True
            optimizer = optimizer_phase2

        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch >= 5:
            scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(device)
                outputs = model(images)
                preds   = torch.argmax(outputs, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        acc = round(accuracy_score(all_labels, all_preds) * 100, 1)
        avg_loss = round(running_loss / len(train_loader), 4)
        print(f"      Epoch {epoch+1:2d}/{epochs} — Loss: {avg_loss:.4f} | Val Accuracy: {acc}%")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_DIR / 'disease_model_best.pth')

    # Final evaluation
    print(f"\n[4/5] Final evaluation (best model: {best_acc}%)")
    model.load_state_dict(torch.load(MODEL_DIR / 'disease_model_best.pth', map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds  = torch.argmax(model(images), dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ── Save final model ───────────────────────────────────────────────────
    print(f"[5/5] Saving model...")
    MODEL_DIR.mkdir(exist_ok=True)

    # Final model file (this is what Django loads)
    torch.save(model.state_dict(), MODEL_DIR / 'disease_model.pth')
    (MODEL_DIR / 'disease_model_accuracy.txt').write_text(str(best_acc))
    (MODEL_DIR / 'disease_classes.txt').write_text('\n'.join(class_names))

    print(f"\n✅ Done!")
    print(f"   Model saved    : ml_models/disease_model.pth")
    print(f"   Accuracy       : {best_acc}%")
    print(f"   Classes file   : ml_models/disease_classes.txt")
    print(f"\n   ⚠️  Update DISEASE_CLASSES list in agriculture/ml_utils.py")
    print(f"      to match the order in disease_classes.txt\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='./dataset',
                        help='Path to dataset folder (subfolders = class names)')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch',  type=int, default=32)
    parser.add_argument('--lr',     type=float, default=1e-4)
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch, args.lr)

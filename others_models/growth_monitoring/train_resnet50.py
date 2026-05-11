# train_resnet50.py
# ---------------------------------------------------------
# Train ResNet50 for Rice Leaf Disease Detection
# ---------------------------------------------------------

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score

MODEL_DIR = Path(__file__).resolve().parent / 'ml_models'


def train(data_dir, epochs=5, batch_size=16, lr=1e-4):

    data_dir = Path(data_dir)

    print("\n[1/5] Loading Dataset...")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    full_dataset = ImageFolder(data_dir, transform=train_transform)

    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"      Classes: {class_names}")
    print(f"      Images : {len(full_dataset)}")

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size]
    )

    val_set.dataset = ImageFolder(
        data_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"      Device : {device}")

    # ---------------------------------------------------------
    # ResNet50
    # ---------------------------------------------------------

    print("\n[2/5] Loading ResNet50...")

    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=lr
    )

    best_acc = 0.0

    print(f"\n[3/5] Training for {epochs} epochs...")

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                outputs = model(images)

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = round(
            accuracy_score(all_labels, all_preds) * 100,
            1
        )

        avg_loss = round(
            running_loss / len(train_loader),
            4
        )

        print(
            f"      Epoch {epoch+1}/{epochs}"
            f" - Loss: {avg_loss}"
            f" - Val Accuracy: {acc}%"
        )

        if acc > best_acc:

            best_acc = acc

            MODEL_DIR.mkdir(exist_ok=True)

            torch.save(
                model.state_dict(),
                MODEL_DIR / 'resnet50_best.pth'
            )

    print(f"\n[4/5] Best Accuracy: {best_acc}%")

    print("\n[5/5] Saving Complete Model...")

    torch.save(
        model.state_dict(),
        MODEL_DIR / 'resnet50_final.pth'
    )

    print("\n✅ Training Complete!")
    print(f"   Model Saved : {MODEL_DIR/'resnet50_final.pth'}")
    print(f"   Best Accuracy : {best_acc}%\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        default='./dataset'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5
    )

    args = parser.parse_args()

    train(args.data, args.epochs)
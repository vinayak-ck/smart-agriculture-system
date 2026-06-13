import argparse
from pathlib import Path

import torch
import torch.nn as nn
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

from torchvision import transforms
from torchvision import datasets
from torchvision import models

from torchvision.models import ResNet18_Weights

from torch.utils.data import DataLoader
from torch.utils.data import random_split

MODEL_DIR = Path("ml_models")


def train(data_dir, epochs=30, batch_size=32, lr=1e-4):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    full_dataset = datasets.ImageFolder(
        data_dir
    )

    class_names = full_dataset.classes

    print("\nClasses Found:", len(class_names))
    print("Total Images:", len(full_dataset))

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_indices, val_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size]
    )

    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(
            data_dir,
            transform=train_transform
        ),
        train_indices.indices
    )

    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(
            data_dir,
            transform=val_transform
        ),
        val_indices.indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    num_classes = len(class_names)

    print("\nLoading ResNet18...")

    weights = ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(
            model.fc.in_features,
            num_classes
        )
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_fc = torch.optim.Adam(
        model.fc.parameters(),
        lr=lr
    )

    optimizer_all = torch.optim.Adam(
        model.parameters(),
        lr=lr * 0.1
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_all,
        mode='max',
        patience=2,
        factor=0.5
    )

    best_acc = 0
    patience = 5
    early_stop_counter = 0

    print("\nTraining Started\n")

    for epoch in range(epochs):

        if epoch == 3:
            print("\nUnfreezing entire network...\n")

            for param in model.parameters():
                param.requires_grad = True

        optimizer = (
            optimizer_fc
            if epoch < 3
            else optimizer_all
        )

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, preds = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    preds == labels
                ).sum().item()

        accuracy = (
            100 * correct / total
        )

        avg_loss = (
            running_loss /
            len(train_loader)
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Val Acc: {accuracy:.2f}%"
        )

        if epoch >= 3:
            scheduler.step(accuracy)

        if accuracy > best_acc:

            best_acc = accuracy

            early_stop_counter = 0

            MODEL_DIR.mkdir(
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_DIR /
                "disease_model_best.pth"
            )

        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:

            print(
                "\nEarly stopping triggered."
            )

            break

    print(
        f"\nBest Accuracy: "
        f"{best_acc:.2f}%"
    )

    model.load_state_dict(
        torch.load(
            MODEL_DIR /
            "disease_model_best.pth",
            map_location=device
        )
    )

    torch.save(
        model.state_dict(),
        MODEL_DIR /
        "disease_model.pth"
    )

    with open(
        MODEL_DIR /
        "disease_classes.txt",
        "w"
    ) as f:

        for cls in class_names:
            f.write(cls + "\n")

    with open(
        MODEL_DIR /
        "disease_model_accuracy.txt",
        "w"
    ) as f:

        f.write(
            str(
                round(best_acc, 2)
            )
        )

    print("\nModel Saved Successfully")
    print("Classes:", len(class_names))
    print("Best Accuracy:", best_acc)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="dataset/color"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    args = parser.parse_args()

    train(
        args.data,
        args.epochs,
        args.batch,
        args.lr
    )
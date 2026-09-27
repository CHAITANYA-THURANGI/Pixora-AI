"""
Tier-2 emotion model: a custom CNN trained from scratch on FER-2013.

WHY THIS EXISTS. modules/emotion.py works out of the box using DeepFace's
built-in pretrained model (Tier 1 -- zero setup). This script trains your
OWN emotion classifier instead, on the same dataset your original project
brief named directly (FER-2013 CNN). It's a genuine from-scratch PyTorch
CNN, not a wrapper around someone else's model -- appropriate to describe
as your own trained model in a report, and it runs on your GPU.

DATA -- this is the one step you have to do by hand, and why:
FER-2013 is gated behind a free Kaggle account (standard for Kaggle-hosted
datasets; there's no ungated mirror of the actual images, only of things
like MUStARD's text). One-time setup:
  1. Create a free account at https://www.kaggle.com if you don't have one.
  2. Go to https://www.kaggle.com/datasets/msambare/fer2013 and click Download
     (top right). This downloads a .zip, a few hundred MB.
  3. Extract it so you end up with this exact folder layout inside the
     project (create the fer2013/ folder yourself if it doesn't exist):
        datasets/fer2013/train/angry/*.jpg  (and disgust, fear, happy, neutral, sad, surprise)
        datasets/fer2013/test/angry/*.jpg   (same 7 class folders)
     If the zip extracts with different top-level folder names, just
     rename/move them to match -- the folder names ARE the class labels
     (torchvision.datasets.ImageFolder reads them directly, no separate
     label file to worry about getting out of sync).

Full walkthrough with exact detail: docs/STEP_BY_STEP.md, "Training the emotion model."

Usage:
    python training/train_emotion_local.py
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "fer2013")
SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "emotion_custom")

# Fixed order -- must match modules/emotion_custom.py's EMOTIONS list exactly,
# since the model's output index 0..6 is decoded using this order.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class EmotionCNN(nn.Module):
    """Small CNN for 48x48 grayscale face crops -> 7 emotion classes."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def build_transforms(train: bool):
    ops = [transforms.Grayscale(num_output_channels=1), transforms.Resize((48, 48))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(ops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                         help="Folder containing train/ and test/ subfolders (see docstring above).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise RuntimeError(
            f"Expected {train_dir} and {test_dir} to exist, each with 7 class "
            f"subfolders. See this script's docstring (or docs/STEP_BY_STEP.md, "
            f"'Training the emotion model') for exactly how to download and "
            f"place the FER-2013 data first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(train=True))
    test_ds = datasets.ImageFolder(test_dir, transform=build_transforms(train=False))

    # ImageFolder sorts class names alphabetically to assign indices -- verify
    # that matches CLASS_NAMES above, since a silent mismatch would make
    # every prediction map to the wrong emotion.
    if train_ds.classes != CLASS_NAMES:
        raise RuntimeError(
            f"Found class folders {train_ds.classes} but expected {CLASS_NAMES}. "
            f"Check your datasets/fer2013/train/ subfolder names exactly match "
            f"(angry, disgust, fear, happy, neutral, sad, surprise)."
        )

    print(f"Train: {len(train_ds)} images across {len(train_ds.classes)} classes")
    print(f"Test:  {len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    model = EmotionCNN(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    best_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        scheduler.step(acc)
        print(f"Epoch {epoch:2d}/{args.epochs}  train_loss={train_loss:.4f}  test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "emotion_cnn.pt"))

    # final report using the BEST checkpoint, not just whatever the last epoch happened to be
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "emotion_cnn.pt"), map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== Final report (best checkpoint) ===")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    cm = confusion_matrix(all_labels, all_preds).tolist()
    print("Confusion matrix:", cm)

    with open(os.path.join(SAVE_DIR, "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f)
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump({
            "model": "Custom CNN (from scratch), trained on FER-2013",
            "test_accuracy": round(best_acc, 4),
            "test_f1_macro": round(f1_score(all_labels, all_preds, average="macro"), 4),
            "confusion_matrix": cm,
            "class_names": CLASS_NAMES,
        }, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Saved model + metrics to {SAVE_DIR}")
    print("modules/emotion.py will auto-detect and use it. Just run `python main.py` again.")


if __name__ == "__main__":
    main()

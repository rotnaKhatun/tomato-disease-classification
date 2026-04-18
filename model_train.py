# =========================
# Libraries
# =========================
import copy
import os
import random
import shutil
import time
from IPython.display import FileLink
from tqdm import tqdm

from collections import Counter
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lime import lime_image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.quantization
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.models as models
from torchvision import datasets, transforms

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skimage.segmentation import mark_boundaries

try:
    import umap.umap_ as umap
    has_umap = True
except ImportError:
    has_umap = False


# =========================
# Device Selection
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# =========================
# Dataset Preparation
# =========================

# ---- Custom Dataset Definition ----
class TomatoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        
        for cls in self.classes:
            class_path = os.path.join(root_dir, cls)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label 


# ---- Dataset Split and DataLoader Instantiation ----
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

full_train_dataset = TomatoDataset(
    root_dir="/kaggle/input/datasets/kaustubhb999/tomatoleaf/tomato/train",
    transform=train_transform
)

train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

test_dataset = TomatoDataset(
    root_dir="/kaggle/input/datasets/kaustubhb999/tomatoleaf/tomato/val",
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 


# =========================
# Model Preparation
# =========================
num_classes = len(train_dataset.dataset.classes)

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, num_classes
)

model = model.to(device)

print(model)


# =========================
# Training and Validation
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4
)

num_epochs = 100
patience = 5
best_val_loss = float("inf")
early_stop_counter = 0
start_epoch = 0

checkpoint_path = "checkpoint.pth"
best_model_path = "best_model.pth"

train_acc_history = []
val_acc_history = []

if os.path.exists(checkpoint_path):

    print("Loading checkpoint...")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    early_stop_counter = checkpoint["early_stop_counter"]

    train_acc_history = checkpoint["train_acc_history"]
    val_acc_history = checkpoint["val_acc_history"]

    print(f"Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, num_epochs):

    print(f"\nEpoch {epoch+1}/{num_epochs}")

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader)

    for images, labels in train_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        train_bar.set_description(f"Train Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)
    train_acc = correct / total

    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    val_bar = tqdm(val_loader)

    with torch.no_grad():

        for images, labels in val_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            val_bar.set_description(f"Val Loss: {loss.item():.4f}")

    val_loss /= len(val_loader)
    val_acc = correct / total

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "early_stop_counter": early_stop_counter,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history
    }, checkpoint_path)

    if val_loss < best_val_loss:
        print("Saving best model...")
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), best_model_path)

    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered")
        break


# =========================
# Testing and Evaluation
# =========================
num_classes = len(test_dataset.classes)

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, num_classes
)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

class_names = test_dataset.classes

print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))

cm = confusion_matrix(all_labels, all_preds)

plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

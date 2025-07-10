# q3_model_training.py

import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MulticlassF1Score

# ---------- Setup ----------
os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# ---------- Load labeled dataset ----------
df = pd.read_csv("outputs/q2_labeled_dataset.csv")

# Build label encoder
class_names = sorted(df['class_name'].unique())
class_to_idx = {label: idx for idx, label in enumerate(class_names)}
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_names)

df["label_idx"] = df["class_name"].map(class_to_idx)

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ---------- Custom Dataset ----------
class LandCoverDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row["filename"])).convert("RGB")
        label = row["label_idx"]
        if self.transform:
            image = self.transform(image)
        return image, label, row["filename"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = LandCoverDataset(train_df, "data/rgb", transform)
test_dataset = LandCoverDataset(test_df, "data/rgb", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ---------- Load ResNet18 ----------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------- Train ----------
print(" Training model...")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader):.4f}")

# ---------- Evaluate ----------
print(" Evaluating model...")
model.eval()
all_preds = []
all_labels = []
all_filenames = []

with torch.no_grad():
    for images, labels, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_filenames.extend(filenames)

# ---------- Custom F1 ----------
f1_custom = f1_score(all_labels, all_preds, average='macro')
print(f" Custom F1 (macro): {f1_custom:.4f}")

# ---------- torchmetrics F1 ----------
f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
with torch.no_grad():
    all_preds_tensor = torch.tensor(all_preds).to(device)
    all_labels_tensor = torch.tensor(all_labels).to(device)
    f1_torchmetrics = f1_metric(all_preds_tensor, all_labels_tensor).item()
print(f" Torchmetrics F1 (macro): {f1_torchmetrics:.4f}")

# ---------- Confusion Matrix ----------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/q3_confusion_matrix.png")
plt.close()
print(" Saved confusion matrix: outputs/q3_confusion_matrix.png")

# ---------- Save predictions ----------
results_df = pd.DataFrame({
    "filename": all_filenames,
    "true_label": [idx_to_class[i] for i in all_labels],
    "predicted_label": [idx_to_class[i] for i in all_preds]
})
results_df.to_csv("outputs/q3_model_results.csv", index=False)
print(" Saved predictions: outputs/q3_model_results.csv")

# ---------- Plot correct and incorrect ----------
correct = results_df[results_df["true_label"] == results_df["predicted_label"]].head(5)
incorrect = results_df[results_df["true_label"] != results_df["predicted_label"]].head(5)

def plot_examples(df_subset, title):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for ax, (_, row) in zip(axs, df_subset.iterrows()):
        img = Image.open(os.path.join("data/rgb", row["filename"]))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{row['true_label']} → {row['predicted_label']}", fontsize=8)
    plt.suptitle(title)
    return fig

fig1 = plot_examples(correct, " Correct Predictions")
fig2 = plot_examples(incorrect, " Incorrect Predictions")

fig1.savefig("outputs/q3_correct_predictions.png")
fig2.savefig("outputs/q3_incorrect_predictions.png")
plt.close('all')
print(" Saved prediction examples to outputs/")

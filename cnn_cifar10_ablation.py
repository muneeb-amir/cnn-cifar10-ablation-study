# CIFAR-10 CNN + Ablation Study (single cell)
# Requirements: torch, torchvision, datasets, sklearn, matplotlib, tqdm, pandas, numpy
# Paste and run in Kaggle notebook (GPU). Uses multiple GPUs via nn.DataParallel if available.

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# ---------------------------
# Reproducibility & device
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| CUDA devices:", torch.cuda.device_count())

# ---------------------------
# 1) Dataset Preparation
# ---------------------------
# CIFAR-10 normalization values
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

# Transforms: augment train, simple normalization for test
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Load HF dataset once
hf_ds = load_dataset("cifar10")  # downloads automatically in Kaggle

# Wrap HuggingFace split into PyTorch Dataset
class HFCIFARDataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = np.array(item["img"])         # PIL-like image -> np.array
        label = int(item["label"])
        if self.transform:
            img = self.transform(img)
        return img, label

# Helper to make dataloaders given batch_size
def make_dataloaders(batch_size=32, num_workers=2):
    train_ds = HFCIFARDataset(hf_ds["train"], transform=train_transform)
    test_ds  = HFCIFARDataset(hf_ds["test"], transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# Class labels for plotting
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ---------------------------
# 2) CNN Model (flexible)
# ---------------------------
class FlexibleCNN(nn.Module):
    """
    A simple configurable CNN:
    - num_layers: number of conv blocks (conv -> bn -> relu), every second block applies MaxPool
    - base_filters: filters for first conv, doubles roughly every layer until a cap
    """
    def __init__(self, num_layers=5, base_filters=32, num_classes=10, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = 3
        f = base_filters
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_ch, f, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(f))
            layers.append(nn.ReLU(inplace=True))
            # Add pooling every 2 blocks to reduce spatial dims gradually
            if (i % 2) == 1:
                layers.append(nn.MaxPool2d(2))
            in_ch = f
            f = min(f * 2, 512)
        self.features = nn.Sequential(*layers)
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.globalpool(x)
        x = self.classifier(x)
        return x

# Utility to wrap model in DataParallel if >1 GPU
def prepare_model_for_training(model):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Wrapped model with DataParallel. Using GPUs:", torch.cuda.device_count())
    return model.to(device)

# ---------------------------
# Training & Evaluation helpers
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / total
    train_error = 1.0 - (correct / total)
    return avg_loss, train_error

def evaluate_model(model, loader):
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    acc = accuracy_score(labels_all, preds_all)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='weighted', zero_division=0)
    cm = confusion_matrix(labels_all, preds_all)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "cm": cm, "preds": preds_all, "labels": labels_all}

def plot_train_history(losses, errors, title="Training"):
    epochs = np.arange(1, len(losses)+1)
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, losses, marker='o', label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, errors, marker='x', label="Train Error", color="tab:red")
    ax2.set_ylabel("Error")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(title)
    plt.show()

def show_confusion_matrix(cm, classes=CLASSES, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45); ax.set_yticklabels(classes)
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label"); plt.title(title)
    plt.tight_layout()
    plt.show()

# ---------------------------
# 3) Baseline training (30 epochs)
# ---------------------------
def run_training_experiment(name="experiment", num_layers=5, base_filters=32, lr=1e-3, batch_size=32, epochs=30, verbose=True):
    # Create data loaders for this run
    train_loader, test_loader = make_dataloaders(batch_size=batch_size)
    # Build & prepare model
    model = FlexibleCNN(num_layers=num_layers, base_filters=base_filters)
    model = prepare_model_for_training(model)
    # Optimizer & loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//3), gamma=0.1)
    # Training loop
    train_losses = []
    train_errors = []
    for ep in range(epochs):
        loss, err = train_one_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(loss)
        train_errors.append(err)
        scheduler.step()
        if verbose:
            print(f"{name} | Epoch {ep+1}/{epochs} -> loss: {loss:.4f}, train_error: {err:.4f}")
    # Evaluate
    metrics = evaluate_model(model, test_loader)
    # Plot training curves and confusion matrix
    plot_train_history(train_losses, train_errors, title=f"{name} Training Loss & Error")
    show_confusion_matrix(metrics["cm"], title=f"{name} Confusion Matrix")
    return model, metrics, {"losses": train_losses, "errors": train_errors}

# Run baseline
BASE_NUM_LAYERS = 5
BASE_BASE_FILTERS = 32
BASE_LR = 1e-3
BASE_BS = 32
BASE_EPOCHS = 30

print("\n--- Running baseline model (30 epochs) ---")
baseline_model, baseline_metrics, baseline_history = run_training_experiment(
    name="Baseline",
    num_layers=BASE_NUM_LAYERS,
    base_filters=BASE_BASE_FILTERS,
    lr=BASE_LR,
    batch_size=BASE_BS,
    epochs=BASE_EPOCHS
)

# Show baseline metrics table
baseline_row = ["Baseline", baseline_metrics["acc"], baseline_metrics["precision"], baseline_metrics["recall"], baseline_metrics["f1"]]
print("Baseline metrics (Accuracy, Precision, Recall, F1):")
print(baseline_row)

# ---------------------------
# 4) Feature map extraction & visualization
# ---------------------------
def extract_feature_maps(model, sample_input, layer_indices=None, max_maps=6):
    """
    Run the sample_input through model.features layer-by-layer and capture activations
    at indices in layer_indices. sample_input is a single tensor (C,H,W) already transformed.
    """
    model_cpu = model.module if isinstance(model, nn.DataParallel) else model
    model_cpu.eval()
    x = sample_input.unsqueeze(0).to(device)
    acts = []
    cur = x
    for idx, layer in enumerate(model_cpu.features):
        cur = layer(cur)
        if layer_indices and idx in layer_indices:
            acts.append((idx, cur.detach().cpu().squeeze(0)))
    return acts

def plot_feature_maps_from_acts(acts, max_maps=6):
    for idx, feat in acts:
        n_ch = feat.shape[0]
        n_show = min(n_ch, max_maps)
        fig, axs = plt.subplots(1, n_show, figsize=(n_show*2, 2))
        for i in range(n_show):
            ax = axs[i] if n_show>1 else axs
            fmap = feat[i].numpy()
            # Normalize for visualization
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-9)
            ax.imshow(fmap, interpolation='nearest')
            ax.axis('off')
        plt.suptitle(f'Feature maps at features index {idx} (showing {n_show})')
        plt.show()

# Pick a test sample (transformed) for visualization
test_loader_for_vis = make_dataloaders(batch_size=1)[0]
sample_img, sample_label = next(iter(test_loader_for_vis))
sample_img = sample_img[0]  # tensor (C,H,W)
print("\nVisualizing feature maps for a sample test image (label:", CLASSES[sample_label], ")")
# Choose a few layer indices that correspond to Conv/ReLU positions (we inspect several indexes)
layer_indices_to_show = [0, 2, 4, 6, 8]  # these are indices inside model.features
acts_baseline = extract_feature_maps(baseline_model, sample_img, layer_indices=layer_indices_to_show)
plot_feature_maps_from_acts(acts_baseline, max_maps=6)

# Short discussion printed (you can expand in your report)
print("\nFeature map interpretation hints:")
print("- Early conv layers (first few indices) typically show edge and color blob detectors.")
print("- Middle layers often capture textures and simple patterns.")
print("- Deeper layers (closer to classifier) show stronger class-specific patterns or combinations of shapes.\n")

# ---------------------------
# 5) Ablation study (one factor at a time)
# ---------------------------
print("\n--- Starting ablation study (one-factor-at-a-time) ---")
ablation_results = []

# A) Learning Rate variations
lr_values = [0.001, 0.01, 0.1]
print("\nAblation: Learning Rate (10 epochs each)")
for lr in lr_values:
    _, metrics, _ = run_training_experiment(
        name=f"Ablation_LR_{lr}",
        num_layers=BASE_NUM_LAYERS,
        base_filters=BASE_BASE_FILTERS,
        lr=lr,
        batch_size=BASE_BS,
        epochs=10,
        verbose=False
    )
    print(f"LR={lr} -> acc={metrics['acc']:.4f}, prec={metrics['precision']:.4f}, rec={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
    ablation_results.append(("LR", lr, metrics))

# B) Batch Size variations
bs_values = [16, 32, 64]
print("\nAblation: Batch Size (10 epochs each)")
for bs in bs_values:
    _, metrics, _ = run_training_experiment(
        name=f"Ablation_BS_{bs}",
        num_layers=BASE_NUM_LAYERS,
        base_filters=BASE_BASE_FILTERS,
        lr=BASE_LR,
        batch_size=bs,
        epochs=10,
        verbose=False
    )
    print(f"BS={bs} -> acc={metrics['acc']:.4f}, prec={metrics['precision']:.4f}, rec={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
    ablation_results.append(("BS", bs, metrics))

# C) Number of convolutional filters variations (base_filters)
filter_values = [16, 32, 64]
print("\nAblation: Number of Filters (10 epochs each)")
for f in filter_values:
    _, metrics, _ = run_training_experiment(
        name=f"Ablation_Filters_{f}",
        num_layers=BASE_NUM_LAYERS,
        base_filters=f,
        lr=BASE_LR,
        batch_size=BASE_BS,
        epochs=10,
        verbose=False
    )
    print(f"Filters={f} -> acc={metrics['acc']:.4f}, prec={metrics['precision']:.4f}, rec={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
    ablation_results.append(("Filters", f, metrics))

# D) Number of layers variations
layers_values = [3, 5, 7]
print("\nAblation: Number of Layers (10 epochs each)")
for nl in layers_values:
    _, metrics, _ = run_training_experiment(
        name=f"Ablation_Layers_{nl}",
        num_layers=nl,
        base_filters=BASE_BASE_FILTERS,
        lr=BASE_LR,
        batch_size=BASE_BS,
        epochs=10,
        verbose=False
    )
    print(f"Layers={nl} -> acc={metrics['acc']:.4f}, prec={metrics['precision']:.4f}, rec={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
    ablation_results.append(("Layers", nl, metrics))

# ---------------------------
# Aggregate ablation results and pick best single run
# ---------------------------
# Build dataframe of ablation runs
rows = []
for typ, val, met in ablation_results:
    rows.append({"Type": typ, "Value": val, "Accuracy": met["acc"], "Precision": met["precision"], "Recall": met["recall"], "F1": met["f1"]})
ablation_df = pd.DataFrame(rows)
print("\nAblation summary (one-factor-at-a-time):")
display(ablation_df.sort_values("Accuracy", ascending=False).reset_index(drop=True))

# Pick best row by Accuracy
best_row = ablation_df.loc[ablation_df["Accuracy"].idxmax()]
best_type = best_row["Type"]
best_value = best_row["Value"]
print(f"\nBest single ablation result -> Type: {best_type}, Value: {best_value}, Accuracy: {best_row['Accuracy']:.4f}")

# Assemble best hyperparameters by applying the best changed hyperparameter on top of baseline
best_hparams = {
    "num_layers": BASE_NUM_LAYERS,
    "base_filters": BASE_BASE_FILTERS,
    "lr": BASE_LR,
    "batch_size": BASE_BS
}
if best_type == "LR":
    best_hparams["lr"] = float(best_value)
elif best_type == "BS":
    best_hparams["batch_size"] = int(best_value)
elif best_type == "Filters":
    best_hparams["base_filters"] = int(best_value)
elif best_type == "Layers":
    best_hparams["num_layers"] = int(best_value)

print("\nBest hyperparameters to retrain (applied on baseline):", best_hparams)

# ---------------------------
# 6) Retrain with best hyperparameters for 30 epochs and compare
# ---------------------------
print("\n--- Retraining best hyperparameters for 30 epochs ---")
best_model, best_metrics, best_history = run_training_experiment(
    name="Best_Retrain",
    num_layers=best_hparams["num_layers"],
    base_filters=best_hparams["base_filters"],
    lr=best_hparams["lr"],
    batch_size=best_hparams["batch_size"],
    epochs=30,
    verbose=True
)

# Build final comparison table: Baseline vs Best Retrained vs Top Ablation run
final_rows = [
    {"Model": "Baseline", "Accuracy": baseline_metrics["acc"], "Precision": baseline_metrics["precision"], "Recall": baseline_metrics["recall"], "F1-Score": baseline_metrics["f1"]},
    {"Model": f"Best_Retrain ({best_type}={best_value})", "Accuracy": best_metrics["acc"], "Precision": best_metrics["precision"], "Recall": best_metrics["recall"], "F1-Score": best_metrics["f1"]},
]

# Also include the best single ablation run (10-epoch run) for reference
final_rows.append({"Model": f"Top_Ablation_{best_type}_{best_value}_10ep", "Accuracy": best_row["Accuracy"], "Precision": best_row["Precision"], "Recall": best_row["Recall"], "F1-Score": best_row["F1"]})

final_df = pd.DataFrame(final_rows)
print("\nFINAL Comparison Table (Baseline vs Tuned):")
display(final_df)

# Also show confusion matrices side-by-side for Baseline and Best_Retrain
print("\nBaseline confusion matrix:")
show_confusion_matrix(baseline_metrics["cm"], title="Baseline Confusion Matrix")
print("\nBest retrained confusion matrix:")
show_confusion_matrix(best_metrics["cm"], title=f"Best Retrain ({best_type}={best_value}) Confusion Matrix")

# Feature maps on best model as well (optional)
print("\nFeature maps for best retrained model:")
acts_best = extract_feature_maps(best_model, sample_img, layer_indices=layer_indices_to_show)
plot_feature_maps_from_acts(acts_best, max_maps=6)

# ---------------------------
# Done
# ---------------------------
print("\nAll steps complete. Summary:")
print("- Dataset preparation: loaded from Hugging Face and preprocessed.")
print("- Baseline trained for 30 epochs. Training error plotted.")
print("- Evaluation: confusion matrices + metrics (Accuracy, Precision, Recall, F1).")
print("- Feature maps extracted and visualized for baseline and best model.")
print("- Ablation (one-factor-at-a-time) performed for LR, Batch Size, Filters, Layers (10 epochs each).")
print("- Best hyperparameter setting retrained for 30 epochs and compared to baseline.")

# End of full script

from config import CFG
from utils.setup_config import setup_config
from utils.create_folds import create_folds
from utils.pytorch import train_model, evaluate_model, predict 

from models.fcn import FCN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat

import numpy as np
import pandas as pd
from pathlib import Path
import os
import copy
import matplotlib.pyplot as plt

# --- Setup ---

setup_config()
torch.manual_seed(CFG.seed)
device = CFG.device

# --- Import Dataset ---

base_dir = Path(CFG.data_root)
data_path = base_dir / CFG.data_mat
label_path = base_dir / CFG.label_csv

data = loadmat(data_path)['dataset_all_time']
data = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1) # (num_samples, num_timesteps, num_features)
print("data:", data.shape, data.dtype)

# label_map = {
#     'mutant-1': 'mutant-1',
#     'mutant-2': 'mutant-2',
#     'wild':     'wild'
# }

label_map = {
    'mutant-1': 'mutant',
    'mutant-2': 'mutant',
    'wild':     'wild'
}

le = LabelEncoder()
labels = pd.read_csv(label_path)
labels = labels.iloc[:, 0].map(label_map)
labels = le.fit_transform(labels)
labels = torch.tensor(labels, dtype=torch.int64)
print("labels:", labels.shape, labels.dtype)
print("label encoding:", dict(zip(le.classes_, range(len(le.classes_)))))

CFG.num_classes = len(le.classes_)
print("num_classes:", CFG.num_classes)
print()

# --- Create Folds and Train/Test Splits ---

folds = create_folds(
    data, 
    labels, 
    num_folds=5, 
    train_size=0.8,
    val_size=0.0,
    test_size=0.2,
    seed=CFG.seed,
    shuffle=True
)

for idx, fold in enumerate(folds):
    print(f"Fold {idx}:")
    
    print("\tTrain data:", fold['train_data'].shape, fold['train_data'].dtype)
    print("\tTrain label:", fold['train_labels'].shape, fold['train_labels'].dtype)
    
    if fold['val_data'] is not None:
        print("\tVal data:", fold['val_data'.shape], fold['val_data'].dtype)
        print("\tVal label:", fold['val_labels'.shape], fold['val_labels'].dtype)
    
    print("\tTest data:", fold['test_data'].shape, fold['test_data'].dtype)
    print("\tTest label:", fold['test_labels'].shape, fold['test_labels'].dtype)
print()

# --- Data Preprocessing ---

for idx, fold in enumerate(folds):
    # add to device
    fold['train_data'] = fold['train_data'].to(device)
    fold['train_labels'] = fold['train_labels'].to(device)
    fold['test_data'] = fold['test_data'].to(device)
    fold['test_labels'] = fold['test_labels'].to(device)
    
    # normalize per feature on train data statitistics
    train_mean = fold['train_data'].mean(dim=(0,1), keepdim=True)
    train_std = fold['train_data'].std(dim=(0,1), keepdim=True)

    fold['train_data'] = (fold['train_data'] - train_mean) / train_std
    fold['test_data'] = (fold['test_data'] - train_mean) / train_std

# --- Prepare Dataloaders ---

fold_loaders = []
for idx, fold in enumerate(folds):
    train_dataset = TensorDataset(fold['train_data'], fold['train_labels'])
    test_dataset = TensorDataset(fold['test_data'], fold['test_labels'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)

    fold_loaders.append({
        "train_loader": train_loader,
        "test_loader": test_loader
    })

# --- Model Training ---

CFG.output_dir = Path(CFG.output_dir) / CFG.model_name
os.makedirs(CFG.output_dir, exist_ok=True)

model_args = {
    "in_channels":   CFG.in_channels,
    "hidden_size":  CFG.hidden_size,
    "kernel_size":  CFG.kernel_size,
    "output_size":  CFG.num_classes,
    "dropout":      CFG.dropout,
}
model = FCN(**model_args)

# Select either "best_train_loss" or "last_epoch".
checkpoint_policy = "last_epoch"

if checkpoint_policy not in {"best_train_loss", "last_epoch"}:
    raise ValueError(
        "checkpoint_policy must be 'best_train_loss' or 'last_epoch'."
    )

# Directory for loss plots.
plot_dir = Path("results") / "fcn"
plot_dir.mkdir(parents=True, exist_ok=True)

class_names = list(le.classes_)

label_encoding = dict(
    zip(class_names, range(len(class_names)))
)

fold_accuracies = []
fold_class_accuracies = []
fold_test_losses = []
fold_checkpoint_epochs = []

all_train_losses = []
all_test_losses = []


for fold_idx, fold in enumerate(fold_loaders):
    fold_number = fold_idx + 1

    print("\n" + "=" * 65)
    print(
        f"Fold {fold_number}/{len(fold_loaders)} "
        f"(Segment Classifier)"
    )
    print("=" * 65)

    train_loader = fold["train_loader"]
    test_loader = fold["test_loader"]

    # Create a fresh model for every fold.
    model = FCN(**model_args).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        eps=1e-8,
    )

    train_losses = []
    test_losses = []

    checkpoint_epoch = 0
    checkpoint_train_loss = float("inf")
    checkpoint_state_dict = None

    for epoch in range(CFG.epochs):
        train_loss = train_model(
            model,
            train_loader,
            CFG.criterion,
            optimizer,
            device,
        )

        test_loss, test_acc, test_class_acc = evaluate_model(
            model,
            test_loader,
            CFG.criterion,
            device,
            num_classes=CFG.num_classes,
        )

        train_loss = float(train_loss)
        test_loss = float(test_loss)
        test_acc = float(test_acc)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"Epoch {epoch + 1:03d}/{CFG.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

        if not np.isfinite(train_loss) or not np.isfinite(test_loss):
            print(
                f"Non-finite loss at epoch {epoch + 1}. "
                "Stopping this fold."
            )
            break

        should_checkpoint = (
            checkpoint_policy == "last_epoch"
            or train_loss < checkpoint_train_loss
        )

        if should_checkpoint:
            checkpoint_epoch = epoch + 1
            checkpoint_train_loss = train_loss

            checkpoint_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if checkpoint_state_dict is None:
        raise RuntimeError(
            f"Fold {fold_number} did not complete a valid epoch."
        )

    # Restore the selected checkpoint.
    model.load_state_dict(checkpoint_state_dict)
    model.to(device)

    # Evaluate the selected checkpoint.
    (
        checkpoint_test_loss,
        checkpoint_test_acc,
        checkpoint_class_acc,
    ) = evaluate_model(
        model,
        test_loader,
        CFG.criterion,
        device,
        num_classes=CFG.num_classes,
    )

    checkpoint_test_loss = float(checkpoint_test_loss)
    checkpoint_test_acc = float(checkpoint_test_acc)

    checkpoint_class_acc = np.asarray(
        checkpoint_class_acc,
        dtype=float,
    )

    fold_accuracies.append(checkpoint_test_acc)
    fold_class_accuracies.append(checkpoint_class_acc)
    fold_test_losses.append(checkpoint_test_loss)
    fold_checkpoint_epochs.append(checkpoint_epoch)

    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)

    print("-" * 65)
    print(
        f"Checkpoint Epoch: {checkpoint_epoch} | "
        f"Train Loss: {checkpoint_train_loss:.4f} | "
        f"Test Loss: {checkpoint_test_loss:.4f} | "
        f"Test Acc: {checkpoint_test_acc:.2f}%"
    )

    # --------------------------------------------------------
    # Save train and test loss plot
    # --------------------------------------------------------

    epochs_completed = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs_completed,
        train_losses,
        label="Train Loss",
        linewidth=2,
    )

    ax.plot(
        epochs_completed,
        test_losses,
        label="Test Loss",
        linewidth=2,
    )

    ax.axvline(
        checkpoint_epoch,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Checkpoint Epoch {checkpoint_epoch}",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Fold {fold_number}: Train and Test Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    plot_path = (
        plot_dir
        / f"{CFG.model_name}_fold_{fold_number:02d}_loss.png"
    )

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved plot: {plot_path}")

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    save_path = (
        Path(CFG.output_dir)
        / f"{CFG.model_name}_fold_{fold_number:02d}_checkpoint.pt"
    )

    torch.save(
        {
            "fold_idx": fold_idx,
            "fold_number": fold_number,
            "epoch": checkpoint_epoch,
            "checkpoint_policy": checkpoint_policy,

            "model_state_dict": checkpoint_state_dict,

            "train_loss": checkpoint_train_loss,
            "test_loss": checkpoint_test_loss,
            "test_acc": checkpoint_test_acc,
            "test_class_acc": checkpoint_class_acc.tolist(),

            "train_losses": train_losses,
            "test_losses": test_losses,

            "model_name": CFG.model_name,
            "model_args": model_args,
            "num_classes": CFG.num_classes,
            "class_names": class_names,
            "label_encoding": label_encoding,

            "optimizer_name": "Adam",
            "optimizer_args": {
                "lr": CFG.learning_rate,
                "weight_decay": CFG.weight_decay,
                "eps": 1e-8,
            },

            "training_args": {
                "epochs": CFG.epochs,
                "batch_size": CFG.batch_size,
                "seed": CFG.seed,
            },
        },
        save_path,
    )

    print(f"Saved checkpoint: {save_path}")


# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================

fold_accuracies = np.asarray(
    fold_accuracies,
    dtype=float,
)

fold_class_accuracies = np.asarray(
    fold_class_accuracies,
    dtype=float,
)

fold_test_losses = np.asarray(
    fold_test_losses,
    dtype=float,
)

print("\n" + "=" * 65)
print("Cross-Validation Statistics (Segment Classifier)")
print("=" * 65)
print("Fold Results")

for fold_number, (
    accuracy,
    test_loss,
    checkpoint_epoch,
) in enumerate(
    zip(
        fold_accuracies,
        fold_test_losses,
        fold_checkpoint_epochs,
    ),
    start=1,
):
    print(
        f"Fold {fold_number:2d}: "
        f"Epoch {checkpoint_epoch:3d} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Accuracy: {accuracy:.2f}%"
    )

print("\n" + "=" * 42)

print(
    f"{CFG.model_name:<18}"
    f"{'Mean (%)':>11}"
    f"{'Std (%)':>11}"
)

print("-" * 42)

for class_idx, class_name in enumerate(class_names):
    class_values = fold_class_accuracies[:, class_idx]

    print(
        f"{class_name:<18}"
        f"{np.nanmean(class_values):>11.2f}"
        f"{np.nanstd(class_values):>11.2f}"
    )

print(
    f"{'Overall':<18}"
    f"{np.nanmean(fold_accuracies):>11.2f}"
    f"{np.nanstd(fold_accuracies):>11.2f}"
)

print("=" * 42)
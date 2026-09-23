
import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt

from models import create_model

def train_model(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    device
):
    """
    Trains model and returns loss
    """
    model.train()
    running_loss = 0
    
    for inputs, labels in train_loader:
        # use device
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # go forward through model and find loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backpropagation and optimize
        optimizer.zero_grad() # zero gradient buffers
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients

        optimizer.step()
        
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

# evaluates the model and returns loss and accuracy
def evaluate_model(
    model, 
    test_loader, 
    criterion, 
    device, 
    num_classes
):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    # per-class tracking
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # update per-class stats
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()

                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    # overall accuracy
    overall_accuracy = 100 * correct / total
    epoch_loss = running_loss / len(test_loader)

    # per-class accuracy
    class_accuracy = [
        100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(num_classes)
    ]

    return epoch_loss, overall_accuracy, class_accuracy

def predict(model, input_loader, device):
    """
    Predicts class probabilities for the input data using the trained model.
    Returns a tensor of shape (num_samples, num_classes) with the predicted probabilities.
    """
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in input_loader:
            inputs = inputs.to(device, non_blocking=True)

            outputs = model(inputs) # (batch_size, num_classes)
            probabilities = torch.softmax(outputs, dim=1)

            all_probabilities.append(probabilities)
            all_labels.append(labels)

    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_probabilities, all_labels

def create_loss(loss_name):
    losses = {
        "cross_entropy": nn.CrossEntropyLoss,
    }

    if loss_name not in losses:
        raise ValueError(f"Unsupported loss: {loss_name}")

    return losses[loss_name]()

def _save_loss_plot(
    train_losses,
    test_losses,
    checkpoint_epoch,
    fold_number,
    model_name,
    output_dir,
):
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, test_losses, label="Test Loss")

    ax.axvline(
        checkpoint_epoch,
        color="tab:red",
        linestyle="--",
        label=f"Checkpoint Epoch {checkpoint_epoch}",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Fold {fold_number}: Train and Test Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    plot_path = (
        output_dir
        / f"{model_name}_fold_{fold_number:02d}_loss.png"
    )

    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_checkpoint(
    result,
    cfg,
    class_names,
    fold_idx,
    output_dir,
):
    checkpoint_dir = Path(f"checkpoints\{cfg.model_name}")
    checkpoint_path = checkpoint_dir / f"fold_{fold_idx + 1:02d}_checkpoint.pt"

    torch.save(
        {
            "fold_idx": fold_idx,
            "fold_number": fold_idx + 1,
            "epoch": result["checkpoint_epoch"],
            "model_state_dict": result["model_state_dict"],
            "train_loss": result["train_loss"],
            "test_loss": result["test_loss"],
            "test_accuracy": result["test_accuracy"],
            "class_accuracy": result["class_accuracy"].tolist(),
            "train_losses": result["train_losses"],
            "test_losses": result["test_losses"],
            "model_name": cfg.model_name,
            "num_classes": cfg.num_classes,
            "class_names": class_names,
        },
        checkpoint_path,
    )

def train_fold(
    cfg,
    fold,
    fold_number,
    device,
):
    model = create_model(cfg).to(device)
    criterion = create_loss(cfg.loss)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eps=1e-8,
    )

    train_losses = []
    test_losses = []

    checkpoint_epoch = 0
    checkpoint_train_loss = float("inf")
    checkpoint_state = None

    for epoch in range(cfg.epochs):
        train_loss = train_model(
            model,
            fold["train"],
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy, _ = evaluate_model(
            model,
            fold["test"],
            criterion,
            device,
            num_classes=cfg.num_classes,
        )

        train_loss = float(train_loss)
        test_loss = float(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"Fold {fold_number} | "
            f"Epoch {epoch + 1:03d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {float(test_accuracy):.2f}%"
        )

        if not np.isfinite(train_loss):
            break

        # Last-epoch policy.
        checkpoint_epoch = epoch + 1
        checkpoint_train_loss = train_loss
        checkpoint_state = deepcopy(model.state_dict())

    if checkpoint_state is None:
        raise RuntimeError(
            f"Fold {fold_number} did not complete a valid epoch."
        )

    model.load_state_dict(checkpoint_state)

    test_loss, test_accuracy, class_accuracy = evaluate_model(
        model,
        fold["test"],
        criterion,
        device,
        num_classes=cfg.num_classes,
    )

    return {
        "checkpoint_epoch": checkpoint_epoch,
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in checkpoint_state.items()
        },
        "train_loss": checkpoint_train_loss,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_accuracy": np.asarray(class_accuracy, dtype=float),
        "train_losses": train_losses,
        "test_losses": test_losses,
    }

def run_cross_validation(
    cfg,
    fold_loaders,
    class_names,
    device,
    output_dir,
):
    results = []

    for fold_idx, fold in enumerate(fold_loaders):
        fold_number = fold_idx + 1

        result = train_fold(
            cfg=cfg,
            fold=fold,
            fold_number=fold_number,
            device=device,
        )

        _save_loss_plot(
            train_losses=result["train_losses"],
            test_losses=result["test_losses"],
            checkpoint_epoch=result["checkpoint_epoch"],
            fold_number=fold_number,
            model_name=cfg.model_name,
            output_dir=output_dir,
        )

        _save_checkpoint(
            result=result,
            cfg=cfg,
            class_names=class_names,
            fold_idx=fold_idx,
            output_dir=output_dir,
        )

        results.append(result)

    print_cross_validation_summary(
        results,
        class_names,
        cfg.model_name,
    )

    return results

def print_cross_validation_summary(
    results,
    class_names,
    model_name,
):
    fold_accuracies = np.asarray(
        [result["test_accuracy"] for result in results]
    )

    fold_class_accuracies = np.asarray(
        [result["class_accuracy"] for result in results]
    )

    print("\nCross-Validation Statistics")
    print("-" * 42)

    for fold_number, result in enumerate(results, start=1):
        print(
            f"Fold {fold_number:2d}: "
            f"Epoch {result['checkpoint_epoch']:3d} | "
            f"Test Loss: {result['test_loss']:.4f} | "
            f"Test Accuracy: {result['test_accuracy']:.2f}%"
        )

    print("\n" + "-" * 42)
    print(f"{model_name:<18}{'Mean (%)':>11}{'Std (%)':>11}")

    for class_idx, class_name in enumerate(class_names):
        values = fold_class_accuracies[:, class_idx]

        print(
            f"{class_name:<18}"
            f"{np.nanmean(values):>11.2f}"
            f"{np.nanstd(values):>11.2f}"
        )

    print(
        f"{'Overall':<18}"
        f"{np.nanmean(fold_accuracies):>11.2f}"
        f"{np.nanstd(fold_accuracies):>11.2f}"
    )
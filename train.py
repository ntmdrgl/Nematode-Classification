from utils.dataset import load_dataset, load_unsplit_dataset, create_split 
from utils.dataset import create_training_loader, create_testing_loader, load_unsplit_dataset

from utils.training import train_model, evaluate_model

from models import create_model

import math
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    data_path = f"data/{cfg.dataset}.npz"

    X_train, y_train, X_test, y_test = load_dataset(data_path)

    X_train, y_train, X_val, y_val = create_split(
        X_train,
        y_train,
        fraction=0.15/(1-0.15),
        seed=cfg.seed
    )

    print("Train data:", X_train.shape)
    print("Train labels:", y_train.shape)
    print("Valid data:", X_val.shape)
    print("Valid labels:", y_val.shape)
    print("Test data:", X_test.shape)
    print("Test labels:", y_test.shape, "\n")

    train_loader, stats = create_training_loader(X_train, y_train, cfg.batch_size)
    val_loader = create_testing_loader(X_val, y_val, stats, cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(cfg.model, cfg.model_params).to(device)
    
    if cfg.class_weighted_loss:
        counts = torch.bincount(
            y_train,
            minlength=cfg.model_params.num_classes,
        ).float()

        if (counts == 0).any():
            raise ValueError("Every class must appear in the training subset.")

        weights = counts.sum() / (len(counts) * counts)
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eps=1e-8
    )

    checkpoint_dir = Path("checkpoints") / f"{cfg.model}_{cfg.dataset}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{cfg.model}_{cfg.dataset}_best_val_loss.pt"

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_model(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy, _ = evaluate_model(
            model, val_loader, criterion, device, cfg.model_params.num_classes
        )

        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}.")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_accuracy={val_accuracy:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "normalization": {
                        key: value.detach().cpu() for key, value in stats.items()
                    },
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(
                    f"Early stopping: validation loss has not improved for "
                    f"{cfg.patience} consecutive epochs."
                )
                break

    # Figure avoids opening a GUI window during training.
    figure = Figure(figsize=(8, 5))
    axes = figure.subplots()
    epochs = range(1, len(history["train_loss"]) + 1)
    axes.plot(epochs, history["train_loss"], label="Training loss")
    axes.plot(epochs, history["val_loss"], label="Validation loss")
    axes.axvline(best_epoch, color="gray", linestyle="--", label="Best checkpoint")
    axes.set(
        title="Training and validation loss",
        xlabel="Epoch",
        ylabel="Cross-entropy loss",
    )
    axes.legend()
    figure.tight_layout()
    plot_path = checkpoint_dir / "loss_curve.png"
    figure.savefig(plot_path, dpi=150)
    print(f"Saved loss curve: {plot_path}")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()

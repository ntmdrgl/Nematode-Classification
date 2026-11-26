"""Shared checkpoint and test-data loading for evaluation scripts."""

from pathlib import Path

from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import OmegaConf
import torch

from models import create_model
from utils.batched_dataset import MappedDataset, mapped_loader
from utils.openworm_io import feature_array


EIGENWORMS_CLASS_NAMES = {
    "1": "N2 (wild type)", "2": "goa-1", "3": "unc-1",
    "4": "unc-38", "5": "unc-63",
}


def load_evaluation(cfg):
    """Load a split NPZ and its trained model, using saved model parameters."""
    data_path = Path(to_absolute_path(cfg.get("data_path") or f"data/{cfg.dataset}.npz"))
    dataset_name = data_path.stem
    run_name = f"{cfg.model}_{dataset_name}"
    checkpoint_path = Path(to_absolute_path("checkpoints")) / run_name / f"{run_name}_best_val_loss.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_cfg = OmegaConf.create(checkpoint["config"])
    num_classes = int(saved_cfg.model_params.num_classes)

    with np.load(data_path, allow_pickle=False) as data:
        labels = data["y_test"]
        class_names = (data["classes"].astype(str) if "classes" in data
                       else np.arange(num_classes).astype(str))

    features = feature_array(data_path, "X_test")
    shape = features.shape
    del features
    if (len(shape) != 3 or any(size == 0 for size in shape)
            or shape[1] != saved_cfg.model_params.in_channels):
        raise ValueError("Test features must have shape (samples, checkpoint input channels, timesteps).")
    if labels.ndim != 1 or len(labels) != shape[0]:
        raise ValueError("Test inputs and labels must have the same nonzero sample count.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Test labels must be integer class indices.")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"Test labels must be between 0 and {num_classes - 1}.")
    if class_names.ndim != 1 or len(class_names) != num_classes:
        raise ValueError("Dataset classes do not match the checkpoint's class count.")
    class_names = class_names.tolist()
    if dataset_name.lower() == "eigenworms":
        class_names = [EIGENWORMS_CLASS_NAMES.get(name, name) for name in class_names]

    stats = checkpoint["normalization"]
    for name in ("mean", "std"):
        if tuple(stats[name].shape) != (1, shape[1], 1):
            raise ValueError(f"Checkpoint normalization {name} does not match the test features.")
        if not torch.isfinite(stats[name]).all():
            raise ValueError(f"Checkpoint normalization {name} contains non-finite values.")
    if (stats["std"] <= 0).any():
        raise ValueError("Checkpoint normalization standard deviations must be positive.")

    dataset = MappedDataset(data_path, "X_test", labels, stats)
    loader = mapped_loader(dataset, cfg.batch_size, num_workers=cfg.get("num_workers", 0))
    model = create_model(saved_cfg.model, saved_cfg.model_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, loader, class_names, saved_cfg, dataset_name

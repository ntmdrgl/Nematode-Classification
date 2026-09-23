import torch

from pathlib import Path
from omegaconf import OmegaConf

from utils.config import load_config
from utils.data_pipeline import load_dataset, create_fold_loaders
from utils.training import run_cross_validation

def create_device(device_name):
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(device_name)

cfg = load_config()
torch.manual_seed(cfg.seed)
device = create_device(cfg.device)
print("device:", device)

# load and preprocess data

# LABEL_MAP = {
#     "mutant_1": "mutant",
#     "mutant_2": "mutant",
#     "wild": "wild",
# }

LABEL_MAP = {
    "mutant_1": "mutant_1",
    "mutant_2": "mutant_2",
    "wild": "wild",
}

# LABEL_MAP = {
#     "aq2947": "aq2947",
#     "cb1112": "cb1112",
#     "cb5": "cb5",
#     "n2_herma": "n2_herma",
#     "n2_male": "n2_male",
#     "ow953": "ow953",
# }

data, labels, class_names = load_dataset(cfg, LABEL_MAP)

cfg.seq_length = data.shape[1]
cfg.in_channels = data.shape[2]
cfg.num_classes = len(class_names)

print("data:", data.shape, data.dtype)
print("labels:", labels.shape, labels.dtype)
print("classes:", class_names, "\n")

print("Data contains NaN:", torch.isnan(data).any().item())
print("Data contains Inf:", torch.isinf(data).any().item())
print("Data minimum:", data.min().item())
print("Data maximum:", data.max().item(), "\n")

# prepare folds and dataloaders
fold_loaders = create_fold_loaders(
    data=data,
    labels=labels,
    cfg=cfg,
)
print()

output_dir = Path(cfg.output_dir) / cfg.model_name
output_dir.mkdir(parents=True, exist_ok=True)

OmegaConf.save(cfg, output_dir / "config.yaml")

# cross validation training loop
run_cross_validation(
    cfg=cfg,
    fold_loaders=fold_loaders,
    class_names=class_names,
    device=device,
    output_dir=output_dir,
)
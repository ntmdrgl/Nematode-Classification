from utils.dataset import load_dataset, create_training_loader, create_testing_loader

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Config:")
    print(OmegaConf.to_yaml(cfg))

    X_train, y_train, X_test, y_test = load_dataset(cfg.datapath)

    print("Train data:", X_train.shape)
    print("Train labels:", y_train.shape)
    print("Test data:", X_test.shape)
    print("Train labels:", y_test.shape, "\n")

    train_loader, stats = create_training_loader(X_train, y_train, cfg.batch_size)
    test_loader = create_testing_loader(X_test, y_test, stats, cfg.batch_size)

if __name__ == "__main__":
    main()
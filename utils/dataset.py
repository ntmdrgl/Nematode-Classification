
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(data_path):
    with np.load(data_path, allow_pickle=False) as data:
        X_train = torch.as_tensor(data["X_train"], dtype=torch.float32)
        y_train = torch.as_tensor(data["y_train"], dtype=torch.long)
        X_test = torch.as_tensor(data["X_test"], dtype=torch.float32)
        y_test = torch.as_tensor(data["y_test"], dtype=torch.long)

    return X_train, y_train, X_test, y_test

def create_training_loader(
    X_train,
    y_train,
    batch_size: 16,
):
    mean = X_train.mean(dim=(0, 2), keepdim=True)
    std = X_train.std(dim=(0, 2), correction=0, keepdim=True).clamp_min(1e-8)
    
    stats = {
        "mean": mean,
        "std": std,
    }

    X_train = (X_train - stats["mean"]) / stats["std"]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, stats

def create_testing_loader(
    X_test,
    y_test,
    stats,
    batch_size: 16,
):
    X_test = (X_test - stats["mean"]) / stats["std"]

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
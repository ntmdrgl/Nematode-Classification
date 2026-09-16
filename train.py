from config import CFG
from utils.setup_config import setup_config
from utils.create_folds import create_folds

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat

import numpy as np
import pandas as pd
import os
from pathlib import Path

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

# --- Train / Test Splits ---

folds = create_folds(
    data, 
    labels, 
    num_folds=5, 
    train_size=0.8,
    val_size=0,
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
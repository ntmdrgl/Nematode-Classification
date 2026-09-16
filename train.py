from config import CFG
from utils.setup_config import setup_config

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

# --- Train / Test Splits ---

splits = None
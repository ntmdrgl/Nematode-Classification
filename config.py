from types import SimpleNamespace
import torch

CFG = SimpleNamespace(
    # General
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    seed=42,

    # Model 
    model_name=None,
    in_channels=None,
    hidden_size=None,
    kernel_size=None,
    dropout=None,

    # Training
    epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    weight_decay=1e-2,
    patience=5,

    # Dataset 
    num_classes=None,
    num_folds=5,
    segment_size=None,

    # Data paths
    data_root="data",
    data_mat="nematode_dataset_data.mat",
    label_csv="nematode_dataset_label.csv",

    # Output / Logging
)
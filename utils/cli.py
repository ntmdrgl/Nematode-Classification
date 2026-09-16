import argparse
from config import CFG

def parse_args():
    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument("--model", type=str, default=CFG.model_name)
    parser.add_argument("--data_root", type=str, default=CFG.data_root)
    parser.add_argument("--data", type=str, default=CFG.data_mat)
    parser.add_argument("--label", type=str, default=CFG.label_csv)

    # Model parameters
    parser.add_argument("--in_channels", type=str, default=CFG.in_channels)
    parser.add_argument("--hidden_size", type=str, default=CFG.hidden_size)
    parser.add_argument("--kernel_size", type=str, default=CFG.kernel_size)
    parser.add_argument("--dropout", type=str, default=CFG.dropout)

    # Training
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--learning_rate", type=float, default=CFG.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--patience", type=int, default=CFG.patience)

    # Dataset
    parser.add_argument("--segment_size", type=int, default=CFG.segment_size)
    parser.add_argument("--num_folds", type=int, default=CFG.num_folds)

    # Output / Logging

    return parser.parse_args()
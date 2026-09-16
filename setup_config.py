import argparse
from config import CFG
from cli import parse_args
from model_zoo import MODEL_ZOO

def setup_config():
    # Setup Config from Command Line arguments and Model Zoo
    args = parse_args()

    # Main arguments
    CFG.model_name      = args.model
    CFG.dataset_root    = args.data_root
    CFG.data_mat        = args.data
    CFG.label_csv       = args.label

    # Architecture parameters
    defaults = MODEL_ZOO.get(CFG.model_name, {})

    CFG.in_channels     = args.in_channels or defaults.get("in_channels")
    CFG.hidden_size     = args.hidden_size or defaults.get("hidden_size")
    CFG.kernel_size     = args.kernel_size or defaults.get("kernel_size")
    CFG.dropout         = args.dropout or defaults.get("dropout")

    # Training
    CFG.epochs          = args.epochs
    CFG.batch_size      = args.batch_size
    CFG.learning_rate   = args.learning_rate
    CFG.weight_decay    = args.weight_decay
    CFG.patience        = args.patience

    # Dataset 
    CFG.segment_size    = args.segment_size

if __name__ == "__main__":
    setup_config()

    print(f"model: {CFG.model_name}")
    print(f"in_channels: {CFG.in_channels}")
    print(f"hidden_size: {CFG.hidden_size}")
    print(f"kernel_size: {CFG.kernel_size}")
    print(f"dropout: {CFG.dropout}")

    print(f"\nepochs: {CFG.epochs}")
    print(f"batch_size: {CFG.batch_size}")
    print(f"learning_rate: {CFG.learning_rate}")
    print(f"weight_decay: {CFG.weight_decay}")
    print(f"patience: {CFG.patience}")

    print(f"\nsegment_size: {CFG.segment_size}")
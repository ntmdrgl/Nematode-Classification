import argparse
from config import CFG
from utils.cli import parse_args
from models.model_zoo import MODEL_ZOO

def setup_config():
    # Setup Config from Command Line arguments and Model Zoo
    args = parse_args()

    # Main arguments
    CFG.model_name      = args.model
    CFG.data_root       = args.data_root
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

    print("model:", CFG.model_name)
    print("in_channels:", CFG.in_channels)
    print("hidden_size:", CFG.hidden_size)
    print("kernel_size:", CFG.kernel_size)
    print("dropout:", CFG.dropout)

    print("\nepochs:", CFG.epochs)
    print("batch_size:", CFG.batch_size)
    print("learning_rate:", CFG.learning_rate)
    print("weight_decay:", CFG.weight_decay)
    print("patience:", CFG.patience)

    print("\nsegment_size:", CFG.segment_size)
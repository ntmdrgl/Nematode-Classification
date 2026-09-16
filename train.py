from config import CFG
from setup_config import setup_config

import numpy as np
import os

import torch
import torch.nn as nn

# Setup
setup_config()
torch.manual_seed(CFG.seed)

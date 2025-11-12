"""Configuration constants for the CMI project."""

from pathlib import Path

# Training mode
TRAIN = True

# Directory paths
RAW_DIR = Path("data")
PRETRAINED_DIR = Path("output")
EXPORT_DIR = Path("output")
EXPORT_DIR.mkdir(exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40
SEED = 42


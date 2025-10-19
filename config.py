import os
import torch

# ===== PATHS =====
DATA_DIR = "data/raw"
INDEX_CSV = "data/index.csv"
CHECKPOINT_DIR = "data/checkpoints"
RESULTS_DIR = "data/results"
DATA_ROOT = "data/raw"
OUTPUT_DIR = "data/outputs"

# ===== DATA PARAMS =====
PATCH_SIZE = 128
STRIDE = 64
WINDOW_T = 3  # Neighbors on each side
SEQUENCE_LENGTH = 5  # was 6
LOOKBACK = 5         # past frames only

# Synthetic mask generation
SYNTH_MASK_COVERAGE = (0.10, 0.40)  # 10-40% synthetic cloud coverage
MIN_WATER_PIXELS = 5000  # Minimum water pixels required in image

# ===== MODEL PARAMS =====
HIDDEN_DIMS = [32, 32, 32]
KERNEL_SIZE = 3
INPUT_CHANNELS = 7  # frame + mask + 6 present_bits

# ===== TRAINING PARAMS =====
BATCH_SIZE = 8
EPOCHS = 60
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

ALPHA = 0.8  # L1 weight in loss
SSIM_WIN = 7
LR = 1e-3

CONTEXT_DROPOUT_P = 0.25
USE_DIST_WEIGHT = True
MIN_MASK_COVERAGE = 0.25

LOSS_ALPHA = 0.8
SSIM_WINDOW = 7
USE_DIST_WEIGHTS = True

# ===== SPLITS =====
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ===== DEVICE =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_GAP_DAYS = 7 # Skip if any neighbor is >7 days away









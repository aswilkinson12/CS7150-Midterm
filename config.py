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
PATCH_SIZE = 128  # Fits well in 250x250 images
STRIDE = 64       # 50% overlap for evaluation

# Temporal settings
SEQUENCE_LENGTH = 5  # Number of past frames
LOOKBACK = 5         # Past frames only (t-5 to t-1)
MAX_GAP_DAYS = 7     # Maximum gap between frames

# Cloud filtering
MAX_TARGET_CLOUD_COVERAGE = 0.95  # Discard targets >95% cloudy
MIN_WATER_PIXELS = 2000  # Minimum visible water (lower for 250x250)

# Synthetic mask generation
SYNTH_MASK_COVERAGE = (0.10, 0.40)  # 10-40% synthetic cloud coverage

# ===== MODEL PARAMS =====
HIDDEN_DIMS = [32, 32, 32]
KERNEL_SIZE = 3
INPUT_CHANNELS = 7  # frame + mask + 5 present_bits

# ===== TRAINING PARAMS =====
BATCH_SIZE = 8
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# ===== SPLITS =====
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ===== DEVICE =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
from pathlib import Path

# dataset & dataLoader
DATA_DIR = Path("raw")    
SEQ_LEN = 3
IMG_SIZE = (256, 256)   
BATCH_SIZE = 4       
NUM_WORKERS = 4

# dodel configuration
INPUT_DIM = 3
HIDDEN_DIM = 64
N_LAYERS = 3
KERNEL_SIZE = (3, 3)

# training
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 20
DEVICE = "cuda"     
ALPHA = 0.8         # weight for hybrid MSSIMLoss
DROPOUT = 0.1

# paths
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT = CHECKPOINT_DIR / "convlstm_best.pth"
RESULTS_DIR = Path("results")

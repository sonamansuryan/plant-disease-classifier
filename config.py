from pathlib import Path
import torch


class Config:
    TRAIN_DIR  = Path("data/train")
    VAL_DIR    = Path("data/val")
    OUTPUT_DIR = Path("outputs")

    BACKBONE = "efficientnet_b4"
    DROPOUT  = 0.2

    IMG_SIZE        = 380
    BATCH_SIZE      = 32
    NUM_EPOCHS      = 50
    LR              = 3e-4
    WEIGHT_DECAY    = 1e-4
    NUM_WORKERS     = 2
    PATIENCE        = 15
    LABEL_SMOOTHING = 0.1
    USE_MIXUP       = False
    MIXUP_ALPHA     = 0.3

    USE_WANDB     = True
    WANDB_PROJECT = "plant-disease-clf"

    SEED   = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP    = True


CFG = Config()
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

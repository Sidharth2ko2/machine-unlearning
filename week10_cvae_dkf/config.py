"""Week 10: Conditional β-VAE DKF for CIFAR-100 multi-class forgetting."""
import os
import torch

DATASET       = "cifar100"
NUM_CLASSES   = 100
FORGET_CLASSES = list(range(10))
RETAIN_CLASSES = [c for c in range(NUM_CLASSES) if c not in FORGET_CLASSES]

DATA_DIR       = "../week9_cifar100_resnet50/data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"

BATCH_SIZE  = 64
NUM_WORKERS = 0

LR_ORIGINAL  = 0.1
LR_FINETUNE  = 0.01
LR_NEGGRAD   = 1e-4
LR_DKF       = 5e-5
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4

EPOCHS_ORIGINAL  = 100
EPOCHS_RETRAIN   = 100
EPOCHS_UNLEARN   = 10
DKF_EPOCHS       = 5
VAE_PRETRAIN_EPOCHS = 15

NEGGRAD_ALPHA = 0.5
LATENT_DIM    = 128
EMBED_DIM     = 64       # class embedding size added in Week 10
BETA          = 4.0
TEMPERATURE   = 0.07

LAMBDA_RETAIN = 10.0
LAMBDA_FORGET = 0.06
LAMBDA_C      = 0.01
LAMBDA_ALIGN  = 0.05

# V2 enhanced hyperparameters — same base LR/epochs as v1, just adds retain KD
LR_DKF_V2        = 5e-5   # same as v1 (1e-4 caused overshoot)
DKF_EPOCHS_V2    = 8      # slightly more than v1's 5
LAMBDA_FORGET_V2 = 0.06   # same as v1
LAMBDA_KD        = 1.0    # retain knowledge distillation weight (small, targeted)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_dirs():
    for path in (DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR):
        os.makedirs(path, exist_ok=True)

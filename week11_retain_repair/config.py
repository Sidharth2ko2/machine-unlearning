"""Week 11-A: Two-stage unlearning — C-DKF v2 + guarded retain repair."""
import os
import torch

DATASET        = "cifar100"
NUM_CLASSES    = 100
FORGET_CLASSES = list(range(10))
RETAIN_CLASSES = [c for c in range(NUM_CLASSES) if c not in FORGET_CLASSES]

DATA_DIR       = "../week9_cifar100_resnet50/data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"

# Week10 paths — loaded read-only, never written
W10_CKPT_DIR   = "../week10_cvae_dkf/checkpoints"
CDKF_V2_CKPT   = os.path.join(W10_CKPT_DIR, "cdkf_v2_resnet50_cifar100_emb64_kd1.0.pth")
ORIGINAL_CKPT  = os.path.join(W10_CKPT_DIR, "original_resnet50_cifar100.pth")
RETRAIN_CKPT   = os.path.join(W10_CKPT_DIR, "retrain_resnet50_cifar100.pth")
CVAE_CKPT      = os.path.join(W10_CKPT_DIR, "cvae_cifar100_e15_b4_emb64.pth")

BATCH_SIZE  = 64
NUM_WORKERS = 0

# Conditional VAE params (must match week10)
LATENT_DIM = 128
EMBED_DIM  = 64
BETA       = 4.0
TEMPERATURE = 0.07

# 11-A static repair hyperparameters
REPAIR_EPOCHS    = 3
LR_REPAIR        = 1e-4
LAMBDA_RETAIN    = 10.0
LAMBDA_KD        = 2.0
LAMBDA_GUARD     = 0.5

# 11-B adaptive repair uses same LR as 11-A — guard ramps adaptively, not LR drops
REPAIR_EPOCHS_STRONG = 3
LR_REPAIR_STRONG     = 1e-4
LAMBDA_GUARD_STRONG  = 2.0

# 11-B adaptive guard (AG-C-DKF)
# Base starts at same level as 11-A so retain gains are preserved.
# Guard only ramps up when p_forget rises above tau.
ALPHA_ADAPTIVE       = 5.0    # ramp-up speed
TAU                  = 0.05   # trigger threshold — fires when forget confidence > 5%
LAMBDA_GUARD_BASE    = 0.5    # same starting point as 11-A
LAMBDA_GUARD_MAX     = 3.0    # cap


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_dirs():
    for path in (CHECKPOINT_DIR, RESULTS_DIR):
        os.makedirs(path, exist_ok=True)

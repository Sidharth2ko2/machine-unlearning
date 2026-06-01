"""
Week 11-A: Guarded retain repair stage.

Takes a C-DKF v2 student (already unlearned) and runs a short repair phase
that restores retain utility without undoing the forgetting.

Repair loss:
    L = λ_retain * CE(student(x_r), y_r)
      + λ_kd     * KL(student(x_r) || teacher(x_r))   # soft label anchor on retain
      + λ_guard  * KL(student(x_f) || teacher(x_cf))  # forget suppression guard

The guard term reuses the conditional VAE counterfactuals from C-DKF.
Without it, even 3 epochs of retain fine-tuning can partially relearn
forget classes because the student weights still carry pre-unlearning signal.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    EMBED_DIM,
    LAMBDA_GUARD,
    LAMBDA_KD,
    LAMBDA_RETAIN,
    LATENT_DIM,
    LR_REPAIR,
    NUM_CLASSES,
    REPAIR_EPOCHS,
    BETA,
    TEMPERATURE,
    CVAE_CKPT,
)
from beta_vae_conditional import CondBetaVAE
from model_utils import get_features_and_logits


def _cycle(loader):
    while True:
        yield from loader


def _load_cvae(device):
    vae = CondBetaVAE(LATENT_DIM, LATENT_DIM, NUM_CLASSES, BETA, EMBED_DIM).to(device)
    vae.load_state_dict(torch.load(CVAE_CKPT, map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"[Repair] Loaded Cbeta-VAE <- {CVAE_CKPT}")
    return vae


def repair(
    student,
    teacher,
    retain_loader,
    forget_loader,
    device,
    epochs      = REPAIR_EPOCHS,
    lr          = LR_REPAIR,
    lambda_retain = LAMBDA_RETAIN,
    lambda_kd     = LAMBDA_KD,
    lambda_guard  = LAMBDA_GUARD,
    use_amp       = False,
):
    """
    student : C-DKF v2 model (will be modified in-place and returned)
    teacher : original model (frozen throughout)
    """
    vae = _load_cvae(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    criterion    = nn.CrossEntropyLoss()
    optimizer    = optim.Adam(student.parameters(), lr=lr)
    forget_cycle = _cycle(forget_loader)
    amp_enabled  = use_amp and device.type == "cuda"
    scaler       = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(
        f"\n[Repair] {epochs} epochs  lr={lr}  "
        f"lam_retain={lambda_retain}  lam_kd={lambda_kd}  lam_guard={lambda_guard}"
    )

    for epoch in tqdm(range(1, epochs + 1), desc="Repair", unit="epoch"):
        student.train()
        totals = {"ce": 0.0, "kd": 0.0, "guard": 0.0}
        steps  = 0

        for x_r, y_r in retain_loader:
            x_f, y_f = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f, y_f = x_r[:b], y_r[:b], x_f[:b], y_f[:b]

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                # Teacher soft targets for retain KD
                _, teacher_r_logits = get_features_and_logits(teacher, x_r)
                teacher_r_soft      = F.softmax(teacher_r_logits, dim=1)

                # Conditional VAE counterfactual for forget guard
                _, x_cf, _, _, _, _, _, _ = vae(x_f, x_r, y_f, y_r)
                _, teacher_cf_logits      = get_features_and_logits(teacher, x_cf)
                teacher_cf_soft           = F.softmax(teacher_cf_logits, dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _, logits_r = get_features_and_logits(student, x_r)
                _, logits_f = get_features_and_logits(student, x_f)

                # Retain: hard CE + soft KD
                loss_ce = lambda_retain * criterion(logits_r, y_r)
                loss_kd = lambda_kd * F.kl_div(
                    F.log_softmax(logits_r, dim=1),
                    teacher_r_soft,
                    reduction="batchmean",
                )

                # Forget guard: keep forget predictions aligned to counterfactual teacher
                loss_guard = lambda_guard * F.kl_div(
                    F.log_softmax(logits_f, dim=1),
                    teacher_cf_soft,
                    reduction="batchmean",
                )

                loss = loss_ce + loss_kd + loss_guard

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            totals["ce"]    += loss_ce.item()
            totals["kd"]    += loss_kd.item()
            totals["guard"] += loss_guard.item()
            steps += 1

        msg = "  ".join(f"{k}={v/max(steps,1):.4f}" for k, v in totals.items())
        tqdm.write(f"Epoch {epoch}/{epochs}  {msg}")

    out = os.path.join(CHECKPOINT_DIR, "cdkf_v2_repaired.pth")
    torch.save(student.state_dict(), out)
    print(f"\n[Repair] Saved -> {out}")
    return student

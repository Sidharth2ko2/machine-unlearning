"""
Week 11-B: AG-C-DKF — Adaptive Guarded Conditional DKF repair.

Extends the static repair (11-A) with a dynamic forget guard that
strengthens automatically when forget-class confidence rises in a batch.

Repair loss:
    p_forget   = mean softmax prob of forget class on x_f (batch proxy)
    lam_guard  = lam_base * (1 + alpha * clamp(p_forget - tau, 0)) capped at lam_max

    L = lam_retain * CE(student(x_r), y_r)
      + lam_kd     * KL(student(x_r) || teacher(x_r))
      + lam_guard  * KL(student(x_f) || teacher(x_cf))   <- adaptive weight

Key design (from ChatGPT review):
  p_forget is detached before computing lam_guard so the adaptive
  weight acts as a controller, not a differentiable path. Without
  detach, gradients flow through the weight itself causing instability.

Novelty claim:
  "Unlike static repair, AG-C-DKF treats forgetting as a dynamic
   constraint — counterfactual suppression increases only when
   forget-class confidence rises above threshold tau."
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import (
    ALPHA_ADAPTIVE,
    BETA,
    CHECKPOINT_DIR,
    CVAE_CKPT,
    EMBED_DIM,
    LAMBDA_GUARD_BASE,
    LAMBDA_GUARD_MAX,
    LAMBDA_KD,
    LAMBDA_RETAIN,
    LATENT_DIM,
    LR_REPAIR_STRONG,
    NUM_CLASSES,
    REPAIR_EPOCHS_STRONG,
    TAU,
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
    print(f"[AG-C-DKF] Loaded Cbeta-VAE <- {CVAE_CKPT}")
    return vae


def adaptive_repair(
    student,
    teacher,
    retain_loader,
    forget_loader,
    device,
    epochs          = REPAIR_EPOCHS_STRONG,
    lr              = LR_REPAIR_STRONG,
    lambda_retain   = LAMBDA_RETAIN,
    lambda_kd       = LAMBDA_KD,
    lambda_guard_base = LAMBDA_GUARD_BASE,
    lambda_guard_max  = LAMBDA_GUARD_MAX,
    alpha           = ALPHA_ADAPTIVE,
    tau             = TAU,
    use_amp         = False,
):
    """
    AG-C-DKF adaptive repair.

    student  : C-DKF v2 checkpoint (modified in-place and returned)
    teacher  : original model (frozen)
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
        f"\n[AG-C-DKF] {epochs} epochs  lr={lr}  "
        f"lam_guard_base={lambda_guard_base}  alpha={alpha}  tau={tau}  "
        f"lam_guard_max={lambda_guard_max}"
    )

    for epoch in tqdm(range(1, epochs + 1), desc="AG-C-DKF", unit="epoch"):
        student.train()
        totals = {"ce": 0.0, "kd": 0.0, "guard": 0.0, "lam_guard": 0.0}
        steps  = 0

        for x_r, y_r in retain_loader:
            x_f, y_f = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f, y_f = x_r[:b], y_r[:b], x_f[:b], y_f[:b]

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                _, teacher_r_logits  = get_features_and_logits(teacher, x_r)
                teacher_r_soft       = F.softmax(teacher_r_logits, dim=1)

                _, x_cf, _, _, _, _, _, _ = vae(x_f, x_r, y_f, y_r)
                _, teacher_cf_logits      = get_features_and_logits(teacher, x_cf)
                teacher_cf_soft           = F.softmax(teacher_cf_logits, dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _, logits_r = get_features_and_logits(student, x_r)
                _, logits_f = get_features_and_logits(student, x_f)

                # Batch-level forget-class confidence proxy
                p_forget          = F.softmax(logits_f, dim=1) \
                                     .gather(1, y_f.unsqueeze(1)).mean()
                p_forget_detached = p_forget.detach()

                # Adaptive guard weight — detached so it acts as controller only
                lam_guard = lambda_guard_base * (
                    1.0 + alpha * torch.clamp(p_forget_detached - tau, min=0.0)
                )
                lam_guard = torch.clamp(lam_guard, max=lambda_guard_max)

                loss_ce = lambda_retain * criterion(logits_r, y_r)
                loss_kd = lambda_kd * F.kl_div(
                    F.log_softmax(logits_r, dim=1),
                    teacher_r_soft,
                    reduction="batchmean",
                )
                loss_guard = lam_guard * F.kl_div(
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

            totals["ce"]       += loss_ce.item()
            totals["kd"]       += loss_kd.item()
            totals["guard"]    += loss_guard.item()
            totals["lam_guard"] += lam_guard.item()
            steps += 1

        avg = {k: v / max(steps, 1) for k, v in totals.items()}
        tqdm.write(
            f"Epoch {epoch}/{epochs}  "
            f"ce={avg['ce']:.4f}  kd={avg['kd']:.4f}  "
            f"guard={avg['guard']:.4f}  avg_lam_guard={avg['lam_guard']:.3f}"
        )

    out = os.path.join(CHECKPOINT_DIR, "agcdkf_repaired.pth")
    torch.save(student.state_dict(), out)
    print(f"\n[AG-C-DKF] Saved -> {out}")
    return student

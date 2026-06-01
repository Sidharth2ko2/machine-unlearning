"""
Week 10 training methods: DKF, RA-DKF, E-RA-DKF, C-DKF, CE-RA-DKF.

C-DKF and CE-RA-DKF use the Conditional β-VAE (CondBetaVAE) which takes
class labels as input, generating class-specific counterfactuals rather than
blurring across all 10 forget classes simultaneously.

method choices:
  "dkf"      - base DKF with unconditional VAE (Week 9 baseline)
  "radkf"    - DKF + normalized MSE alignment (Week 9 baseline)
  "eradkf"   - DKF + cosine alignment + detached contrast (Week 9 baseline)
  "cdkf"     - DKF with Conditional β-VAE  (Week 10 new)
  "ceradkf"  - E-RA-DKF with Conditional β-VAE  (Week 10 new, full stack)
"""
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from beta_vae_conditional import CondBetaVAE
from config import (
    BETA,
    CHECKPOINT_DIR,
    DKF_EPOCHS,
    DKF_EPOCHS_V2,
    EMBED_DIM,
    LATENT_DIM,
    LAMBDA_ALIGN,
    LAMBDA_C,
    LAMBDA_FORGET,
    LAMBDA_FORGET_V2,
    LAMBDA_KD,
    LAMBDA_RETAIN,
    LR_DKF,
    LR_DKF_V2,
    NUM_CLASSES,
    TEMPERATURE,
    VAE_PRETRAIN_EPOCHS,
)
from model_utils import get_features, get_features_and_logits

# Unconditional VAE — kept for Week-9-equivalent baseline comparison
from beta_vae_unconditional import BetaVAE


def _cycle(loader):
    while True:
        yield from loader


def contrastive_loss(z_anchor, z_pos, z_neg, temperature=TEMPERATURE):
    z_anchor = F.normalize(z_anchor, dim=1)
    z_pos    = F.normalize(z_pos,    dim=1)
    z_neg    = F.normalize(z_neg,    dim=1)
    pos_sim  = (z_anchor * z_pos).sum(dim=1, keepdim=True) / temperature
    neg_sim  = torch.mm(z_anchor, z_neg.t()) / temperature
    logits   = torch.cat([pos_sim, neg_sim], dim=1)
    labels   = torch.zeros(logits.size(0), dtype=torch.long, device=z_anchor.device)
    return F.cross_entropy(logits, labels)


def cosine_alignment_loss(z_student, z_teacher):
    z_s = F.normalize(z_student, dim=1)
    z_t = F.normalize(z_teacher, dim=1)
    return 1.0 - F.cosine_similarity(z_s, z_t, dim=1).mean()


# ── VAE loading helpers ────────────────────────────────────────────────────────

def _load_or_train_unconditional_vae(forget_loader, retain_loader, device, epochs, lr):
    vae  = BetaVAE(LATENT_DIM, LATENT_DIM, NUM_CLASSES, BETA).to(device)
    path = os.path.join(CHECKPOINT_DIR, f"vae_cifar100_e{epochs}_b{int(BETA)}.pth")
    if os.path.exists(path):
        vae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        vae.eval()
        return vae
    _train_vae(vae, forget_loader, retain_loader, device, epochs, lr, path, "VAE (uncond)")
    return vae


def _load_or_train_conditional_vae(forget_loader, retain_loader, device, epochs, lr):
    vae  = CondBetaVAE(LATENT_DIM, LATENT_DIM, NUM_CLASSES, BETA, EMBED_DIM).to(device)
    path = os.path.join(CHECKPOINT_DIR, f"cvae_cifar100_e{epochs}_b{int(BETA)}_emb{EMBED_DIM}.pth")
    if os.path.exists(path):
        vae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        vae.eval()
        return vae
    _train_vae(vae, forget_loader, retain_loader, device, epochs, lr, path, "Cβ-VAE (cond)")
    return vae


def _train_vae(vae, forget_loader, retain_loader, device, epochs, lr, save_path, desc):
    optimizer    = optim.Adam(vae.parameters(), lr=lr)
    forget_cycle = _cycle(forget_loader)
    for _ in tqdm(range(epochs), desc=desc, unit="epoch"):
        vae.train()
        for x_r, y_r in retain_loader:
            x_f, y_f = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f, y_f = x_r[:b], y_r[:b], x_f[:b], y_f[:b]
            optimizer.zero_grad()
            loss, _ = vae.compute_loss(x_f, x_r, y_r, y_f)
            loss.backward()
            optimizer.step()
    torch.save(vae.state_dict(), save_path)
    vae.eval()


# ── Student training ───────────────────────────────────────────────────────────

def train_student(
    original_model,
    forget_loader,
    retain_loader,
    device,
    method,
    vae_epochs            = VAE_PRETRAIN_EPOCHS,
    student_epochs        = DKF_EPOCHS,
    lr                    = LR_DKF,
    lambda_retain         = LAMBDA_RETAIN,
    lambda_forget         = LAMBDA_FORGET,
    lambda_c              = LAMBDA_C,
    lambda_align          = LAMBDA_ALIGN,
    lambda_kd             = 0.0,
    max_batches_per_epoch = None,
    use_amp               = False,
):
    """
    method:
      "dkf"        — retain CE + forget KL + contrastive  (unconditional VAE)
      "radkf"      — DKF + normalized MSE alignment        (unconditional VAE)
      "eradkf"     — DKF + cosine alignment + detach       (unconditional VAE)
      "cdkf"       — DKF with conditional VAE              (Week 10)
      "ceradkf"    — E-RA-DKF with conditional VAE         (Week 10)
      "cdkf_v2"    — C-DKF + retain KD + tuned HPs         (Week 10 v2)
      "ceradkf_v2" — CE-RA-DKF + retain KD + tuned HPs     (Week 10 v2)
    """
    assert method in {"dkf", "radkf", "eradkf", "cdkf", "ceradkf", "cdkf_v2", "ceradkf_v2"}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    conditional = method in {"cdkf", "ceradkf", "cdkf_v2", "ceradkf_v2"}
    if conditional:
        vae = _load_or_train_conditional_vae(forget_loader, retain_loader, device, vae_epochs, lr)
    else:
        vae = _load_or_train_unconditional_vae(forget_loader, retain_loader, device, vae_epochs, lr)
    for p in vae.parameters():
        p.requires_grad = False

    teacher = copy.deepcopy(original_model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student   = copy.deepcopy(original_model).to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # detach retain negatives in contrastive for eradkf / ceradkf variants
    detach_retain = method in {"eradkf", "ceradkf", "ceradkf_v2"}

    amp_enabled = use_amp and device.type == "cuda"
    scaler      = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    forget_cycle       = _cycle(forget_loader)
    batches_per_epoch  = len(retain_loader)
    if max_batches_per_epoch is not None:
        batches_per_epoch = min(max_batches_per_epoch, batches_per_epoch)

    desc = method.upper()
    for epoch in tqdm(range(student_epochs), desc=desc, unit="epoch"):
        student.train()
        retain_iter = iter(retain_loader)
        inner = tqdm(range(batches_per_epoch), desc=f"{desc} {epoch+1}/{student_epochs}", leave=False)

        for _ in inner:
            try:
                x_r, y_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                x_r, y_r   = next(retain_iter)

            x_f, y_f = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f, y_f = x_r[:b], y_r[:b], x_f[:b], y_f[:b]

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                if conditional:
                    _, x_cf, _, _, _, _, _, _ = vae(x_f, x_r, y_f, y_r)
                else:
                    _, x_cf, _, _, _, _, _, _ = vae(x_f, x_r)

                z_teacher_r, teacher_r_logits = get_features_and_logits(teacher, x_r)
                _, teacher_cf_logits = get_features_and_logits(teacher, x_cf)
                teacher_cf      = F.softmax(teacher_cf_logits, dim=1)
                teacher_r_soft  = F.softmax(teacher_r_logits, dim=1)
                z_teacher_r     = F.normalize(z_teacher_r, dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                z_r, logits_r = get_features_and_logits(student, x_r)
                z_f, logits_f = get_features_and_logits(student, x_f)
                z_cf          = get_features(student, x_cf)

                z_r_contrast = z_r.detach() if detach_retain else z_r

                loss_retain   = lambda_retain * criterion(logits_r, y_r)
                loss_forget   = lambda_forget * F.kl_div(
                    F.log_softmax(logits_f, dim=1), teacher_cf, reduction="batchmean"
                )
                loss_contrast = lambda_c * contrastive_loss(z_f, z_cf, z_r_contrast)

                # retain KD: match student soft distribution to teacher on retain samples
                loss_kd = lambda_kd * F.kl_div(
                    F.log_softmax(logits_r, dim=1),
                    teacher_r_soft,
                    reduction="batchmean",
                ) if lambda_kd > 0 else 0.0

                loss = loss_retain + loss_forget + loss_contrast + loss_kd

                if method == "radkf":
                    loss = loss + lambda_align * F.mse_loss(F.normalize(z_r, dim=1), z_teacher_r)
                elif method in {"eradkf", "ceradkf", "ceradkf_v2"}:
                    loss = loss + lambda_align * cosine_alignment_loss(z_r, z_teacher_r)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

    name_map = {
        "dkf":        "dkf_resnet50_cifar100.pth",
        "radkf":      f"radkf_resnet50_cifar100_la_{lambda_align}.pth",
        "eradkf":     f"eradkf_resnet50_cifar100_la_{lambda_align}_lf_{lambda_forget}.pth",
        "cdkf":       f"cdkf_resnet50_cifar100_emb{EMBED_DIM}.pth",
        "ceradkf":    f"ceradkf_resnet50_cifar100_emb{EMBED_DIM}_la_{lambda_align}_lf_{lambda_forget}.pth",
        "cdkf_v2":    f"cdkf_v2_resnet50_cifar100_emb{EMBED_DIM}_kd{lambda_kd}.pth",
        "ceradkf_v2": f"ceradkf_v2_resnet50_cifar100_emb{EMBED_DIM}_kd{lambda_kd}_lf_{lambda_forget}.pth",
    }
    torch.save(student.state_dict(), os.path.join(CHECKPOINT_DIR, name_map[method]))
    return student

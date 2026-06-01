"""
Week 11-A sweep: find best (repair_epochs, lambda_guard) combination.

Grid:
  repair_epochs : 1, 2, 3
  lambda_guard  : 0.5, 0.75, 1.0
  lr            : 1e-4  (fixed)

Runs incrementally per lambda_guard value — trains epoch-by-epoch and
evaluates after each, so 3 training chains produce all 9 results.

Usage:
    cd week11_retain_repair
    python sweep_11a.py --dkf-amp
"""
import argparse
import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from beta_vae_conditional import CondBetaVAE
from config import (
    BETA, CDKF_V2_CKPT, CHECKPOINT_DIR, CVAE_CKPT,
    EMBED_DIM, FORGET_CLASSES, LAMBDA_KD, LAMBDA_RETAIN,
    LATENT_DIM, NUM_CLASSES, ORIGINAL_CKPT, RESULTS_DIR,
    RETRAIN_CKPT, get_device, setup_dirs,
)
from data_utils import class_names, get_all_loaders
from evaluate import evaluate_model, evaluate_shared_knowledge, print_results_table
from model_utils import build_resnet50, get_features_and_logits


SWEEP_EPOCHS  = [1, 2, 3]
SWEEP_GUARDS  = [0.5, 0.75, 1.0]
LR_SWEEP      = 1e-4


def _cycle(loader):
    while True:
        yield from loader


def load_model(path, device):
    model = build_resnet50()
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model.to(device)


def load_cvae(device):
    vae = CondBetaVAE(LATENT_DIM, LATENT_DIM, NUM_CLASSES, BETA, EMBED_DIM).to(device)
    vae.load_state_dict(torch.load(CVAE_CKPT, map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def run_one_epoch(student, teacher, vae, retain_loader, forget_loader,
                  device, lambda_guard, optimizer, scaler, amp_enabled):
    """Train one repair epoch and return loss dict."""
    criterion    = nn.CrossEntropyLoss()
    forget_cycle = _cycle(forget_loader)
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
            _, teacher_r_logits = get_features_and_logits(teacher, x_r)
            teacher_r_soft      = F.softmax(teacher_r_logits, dim=1)
            _, x_cf, _, _, _, _, _, _ = vae(x_f, x_r, y_f, y_r)
            _, teacher_cf_logits      = get_features_and_logits(teacher, x_cf)
            teacher_cf_soft           = F.softmax(teacher_cf_logits, dim=1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            _, logits_r = get_features_and_logits(student, x_r)
            _, logits_f = get_features_and_logits(student, x_f)

            loss_ce    = LAMBDA_RETAIN * criterion(logits_r, y_r)
            loss_kd    = LAMBDA_KD * F.kl_div(
                F.log_softmax(logits_r, dim=1), teacher_r_soft, reduction="batchmean"
            )
            loss_guard = lambda_guard * F.kl_div(
                F.log_softmax(logits_f, dim=1), teacher_cf_soft, reduction="batchmean"
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

    return {k: v / max(steps, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dkf-amp", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    setup_dirs()
    device = get_device()
    print(f"[Device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    names = class_names(download=args.download)
    print("[Forget]", ", ".join(f"{i}:{names[i]}" for i in FORGET_CLASSES))

    loaders = get_all_loaders(download=args.download)
    amp     = args.dkf_amp and device.type == "cuda"

    # ── Load references ────────────────────────────────────────────────────────
    teacher       = load_model(ORIGINAL_CKPT, device)
    teacher.eval()
    retrain_model = load_model(RETRAIN_CKPT, device)
    retrain_ref   = evaluate_model(retrain_model, loaders, device)
    cdkf_v2       = load_model(CDKF_V2_CKPT, device)
    vae           = load_cvae(device)

    shared_kw = dict(retrain_metrics=retrain_ref, max_samples=5000)
    all_results = {}

    # Evaluate C-DKF v2 baseline once
    all_results["C-DKF v2"] = evaluate_shared_knowledge(teacher, cdkf_v2, loaders, device, **shared_kw)
    print(f"\n[Baseline] C-DKF v2  Acc_Dr={all_results['C-DKF v2']['Acc_Dr']:.2f}%  "
          f"Acc_Df={all_results['C-DKF v2']['Acc_Df']:.2f}%  "
          f"Avg.Gap={all_results['C-DKF v2'].get('Avg.Gap', 0):.2f}")

    # ── Sweep ──────────────────────────────────────────────────────────────────
    best_key, best_gap = None, float("inf")

    for lg in SWEEP_GUARDS:
        print(f"\n{'='*60}")
        print(f"[Sweep] lambda_guard={lg}  lr={LR_SWEEP}")
        print(f"{'='*60}")

        student   = copy.deepcopy(cdkf_v2)
        optimizer = optim.Adam(student.parameters(), lr=LR_SWEEP)
        scaler    = torch.cuda.amp.GradScaler(enabled=amp)

        for ep in tqdm(SWEEP_EPOCHS, desc=f"guard={lg}", unit="epoch"):
            losses = run_one_epoch(
                student, teacher, vae,
                loaders["retain"], loaders["forget"],
                device, lg, optimizer, scaler, amp,
            )
            tqdm.write(
                f"  ep={ep}  ce={losses['ce']:.4f}  "
                f"kd={losses['kd']:.4f}  guard={losses['guard']:.4f}"
            )

            key     = f"ep{ep}_lg{lg}"
            metrics = evaluate_shared_knowledge(teacher, student, loaders, device, **shared_kw)
            all_results[key] = {**metrics, "epochs": ep, "lambda_guard": lg}

            gap = metrics.get("Avg.Gap", float("inf"))
            print(
                f"  [{key}]  Acc_Dr={metrics['Acc_Dr']:.2f}%  "
                f"Acc_Df={metrics['Acc_Df']:.2f}%  "
                f"Acc_val={metrics['Acc_val']:.2f}%  "
                f"MIA={metrics['MIA']:.2f}  Avg.Gap={gap:.3f}"
            )
            if gap < best_gap:
                best_gap = gap
                best_key = key
                torch.save(
                    student.state_dict(),
                    os.path.join(CHECKPOINT_DIR, "sweep_best_repair.pth"),
                )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Sweep] Best config: {best_key}  Avg.Gap={best_gap:.3f}")
    print(f"{'='*60}")

    # Print clean comparison table
    display = {"C-DKF v2": all_results["C-DKF v2"]}
    for lg in SWEEP_GUARDS:
        for ep in SWEEP_EPOCHS:
            k = f"ep{ep}_lg{lg}"
            display[f"ep={ep} lg={lg}"] = all_results[k]
    print_results_table(display)

    out = os.path.join(RESULTS_DIR, "week11_sweep_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] {out}")
    print(f"[Best checkpoint] ./checkpoints/sweep_best_repair.pth  ({best_key})")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

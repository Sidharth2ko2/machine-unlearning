"""
Week 11 runner: static repair (11-A), strong static repair, and AG-C-DKF adaptive repair (11-B).

Usage:
    cd week11_retain_repair

    # 11-A only (already done)
    python run_experiments.py --stages static --reuse-repaired --dkf-amp

    # 11-B strong static guard
    python run_experiments.py --stages static_strong --dkf-amp

    # 11-B adaptive guard (AG-C-DKF)
    python run_experiments.py --stages adaptive --dkf-amp

    # All at once
    python run_experiments.py --stages static static_strong adaptive --dkf-amp
"""
import argparse
import copy
import json
import os

import torch

from config import (
    ALPHA_ADAPTIVE,
    CDKF_V2_CKPT,
    CHECKPOINT_DIR,
    FORGET_CLASSES,
    LAMBDA_GUARD,
    LAMBDA_GUARD_BASE,
    LAMBDA_GUARD_MAX,
    LAMBDA_GUARD_STRONG,
    LAMBDA_KD,
    LAMBDA_RETAIN,
    LR_REPAIR,
    LR_REPAIR_STRONG,
    ORIGINAL_CKPT,
    REPAIR_EPOCHS,
    REPAIR_EPOCHS_STRONG,
    RESULTS_DIR,
    RETRAIN_CKPT,
    TAU,
    get_device,
    setup_dirs,
)
from data_utils import class_names, get_all_loaders
from evaluate import evaluate_model, evaluate_shared_knowledge, print_results_table
from model_utils import build_resnet50
from repair import repair
from adaptive_repair import adaptive_repair


def load_model(path, device):
    model = build_resnet50()
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model.to(device)


def maybe_repair(label, ckpt_path, repair_fn, reuse, device):
    if reuse and os.path.exists(ckpt_path):
        print(f"[Load] {label} <- {ckpt_path}")
        return load_model(ckpt_path, device)
    print(f"[Train] {label}")
    return repair_fn()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["adaptive"],
                        choices=["static", "static_strong", "adaptive"])
    parser.add_argument("--reuse-repaired",    action="store_true")
    parser.add_argument("--dkf-amp",           action="store_true")
    parser.add_argument("--download",          action="store_true")
    parser.add_argument("--max-eval-samples",  type=int, default=None)
    # static override flags
    parser.add_argument("--repair-epochs",     type=int,   default=REPAIR_EPOCHS)
    parser.add_argument("--lr-repair",         type=float, default=LR_REPAIR)
    parser.add_argument("--lambda-guard",      type=float, default=LAMBDA_GUARD)
    # adaptive override flags
    parser.add_argument("--repair-epochs-strong", type=int,   default=REPAIR_EPOCHS_STRONG)
    parser.add_argument("--lr-repair-strong",     type=float, default=LR_REPAIR_STRONG)
    parser.add_argument("--lambda-guard-base",    type=float, default=LAMBDA_GUARD_BASE)
    parser.add_argument("--lambda-guard-max",     type=float, default=LAMBDA_GUARD_MAX)
    parser.add_argument("--alpha",                type=float, default=ALPHA_ADAPTIVE)
    parser.add_argument("--tau",                  type=float, default=TAU)
    args = parser.parse_args()

    setup_dirs()
    device = get_device()
    print(f"[Device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    names = class_names(download=args.download)
    print("[Forget]", ", ".join(f"{i}:{names[i]}" for i in FORGET_CLASSES))

    loaders = get_all_loaders(download=args.download)
    stages  = set(args.stages)

    # ── Load teacher and references ────────────────────────────────────────────
    print(f"[Load] Original <- {ORIGINAL_CKPT}")
    teacher = load_model(ORIGINAL_CKPT, device)
    teacher.eval()

    print(f"[Load] Retrain  <- {RETRAIN_CKPT}")
    retrain_model   = load_model(RETRAIN_CKPT, device)
    retrain_metrics = evaluate_model(retrain_model, loaders, device,
                                     max_samples=args.max_eval_samples)

    print(f"[Load] C-DKF v2 <- {CDKF_V2_CKPT}")
    cdkf_v2 = load_model(CDKF_V2_CKPT, device)

    shared_kw = dict(retrain_metrics=retrain_metrics, max_samples=5000)
    results   = {}
    results["Retrain"]  = retrain_metrics
    results["C-DKF v2"] = evaluate_shared_knowledge(teacher, cdkf_v2, loaders, device, **shared_kw)
    print("[Baseline] C-DKF v2 evaluated.")

    # ── 11-A static repair ─────────────────────────────────────────────────────
    if "static" in stages:
        static_path = os.path.join(CHECKPOINT_DIR, "cdkf_v2_repaired.pth")
        repaired_static = maybe_repair(
            "Static repair", static_path,
            lambda: repair(
                student       = copy.deepcopy(cdkf_v2),
                teacher       = teacher,
                retain_loader = loaders["retain"],
                forget_loader = loaders["forget"],
                device        = device,
                epochs        = args.repair_epochs,
                lr            = args.lr_repair,
                lambda_retain = LAMBDA_RETAIN,
                lambda_kd     = LAMBDA_KD,
                lambda_guard  = args.lambda_guard,
                use_amp       = args.dkf_amp,
            ),
            args.reuse_repaired, device,
        )
        results["C-DKF v2 + Repair"] = evaluate_shared_knowledge(
            teacher, repaired_static, loaders, device, **shared_kw
        )

    # ── 11-B strong static repair ──────────────────────────────────────────────
    if "static_strong" in stages:
        strong_path = os.path.join(CHECKPOINT_DIR, "cdkf_v2_repaired_strong.pth")
        repaired_strong = maybe_repair(
            "Strong static repair", strong_path,
            lambda: repair(
                student       = copy.deepcopy(cdkf_v2),
                teacher       = teacher,
                retain_loader = loaders["retain"],
                forget_loader = loaders["forget"],
                device        = device,
                epochs        = args.repair_epochs_strong,
                lr            = args.lr_repair_strong,
                lambda_retain = LAMBDA_RETAIN,
                lambda_kd     = LAMBDA_KD,
                lambda_guard  = LAMBDA_GUARD_STRONG,
                use_amp       = args.dkf_amp,
            ),
            args.reuse_repaired, device,
        )
        results["C-DKF v2 + Strong Repair"] = evaluate_shared_knowledge(
            teacher, repaired_strong, loaders, device, **shared_kw
        )

    # ── 11-B adaptive repair (AG-C-DKF) ───────────────────────────────────────
    if "adaptive" in stages:
        adaptive_path = os.path.join(CHECKPOINT_DIR, "agcdkf_repaired.pth")
        repaired_adaptive = maybe_repair(
            "AG-C-DKF adaptive repair", adaptive_path,
            lambda: adaptive_repair(
                student           = copy.deepcopy(cdkf_v2),
                teacher           = teacher,
                retain_loader     = loaders["retain"],
                forget_loader     = loaders["forget"],
                device            = device,
                epochs            = args.repair_epochs_strong,
                lr                = args.lr_repair_strong,
                lambda_retain     = LAMBDA_RETAIN,
                lambda_kd         = LAMBDA_KD,
                lambda_guard_base = args.lambda_guard_base,
                lambda_guard_max  = args.lambda_guard_max,
                alpha             = args.alpha,
                tau               = args.tau,
                use_amp           = args.dkf_amp,
            ),
            args.reuse_repaired, device,
        )
        results["AG-C-DKF"] = evaluate_shared_knowledge(
            teacher, repaired_adaptive, loaders, device, **shared_kw
        )

    # ── Output ─────────────────────────────────────────────────────────────────
    print_results_table(results)
    out = os.path.join(RESULTS_DIR, "week11_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

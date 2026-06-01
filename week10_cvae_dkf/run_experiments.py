"""
Week 10 runner: Conditional β-VAE DKF for CIFAR-100 multi-class forgetting.

Compares Week-9 baselines (DKF, E-RA-DKF) against Week-10 methods
(C-DKF, CE-RA-DKF) that use the class-conditioned VAE.

Usage:
    cd week10_cvae_dkf
    python run_experiments.py --stages all --reuse-checkpoints

Stages:
    original   — train / load original ResNet-50
    baselines  — Retrain, Fine-tune, NegGrad
    dkf        — Week 9 DKF (unconditional VAE, for comparison)
    eradkf     — Week 9 E-RA-DKF (unconditional VAE, for comparison)
    cdkf       — Week 10 C-DKF (conditional VAE)
    ceradkf    — Week 10 CE-RA-DKF (conditional VAE + cosine align + detach)
    eval       — print table and save JSON
"""
import argparse
import json
import os
import sys

import torch

from baselines import finetune, negative_gradient, retrain, train_original
from config import (
    CHECKPOINT_DIR,
    DKF_EPOCHS,
    DKF_EPOCHS_V2,
    EPOCHS_ORIGINAL,
    EPOCHS_RETRAIN,
    EPOCHS_UNLEARN,
    FORGET_CLASSES,
    LAMBDA_ALIGN,
    LAMBDA_FORGET,
    LAMBDA_FORGET_V2,
    LAMBDA_KD,
    LR_DKF_V2,
    RESULTS_DIR,
    get_device,
    setup_dirs,
)
from data_utils import class_names, get_all_loaders
from evaluate import evaluate_model, evaluate_shared_knowledge, print_results_table
from methods import train_student
from model_utils import build_resnet50


def load_model(path, device):
    model = build_resnet50()
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model.to(device)


def original_path():
    return os.path.join(CHECKPOINT_DIR, "original_resnet50_cifar100.pth")


def ckpt(name):
    return os.path.join(CHECKPOINT_DIR, name)


def maybe_load_or_train(label, path, train_fn, reuse, device):
    if reuse and os.path.exists(path):
        print(f"[Load]  {label} <- {path}")
        return load_model(path, device)
    print(f"[Train] {label}")
    return train_fn()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["all"],
                        choices=["all", "original", "baselines",
                                 "dkf", "eradkf", "cdkf", "ceradkf",
                                 "cdkfv2", "ceradkfv2", "eval"])
    parser.add_argument("--reuse-checkpoints",    action="store_true")
    parser.add_argument("--download",             action="store_true")
    parser.add_argument("--original-epochs",  type=int,   default=EPOCHS_ORIGINAL)
    parser.add_argument("--retrain-epochs",   type=int,   default=EPOCHS_RETRAIN)
    parser.add_argument("--unlearn-epochs",   type=int,   default=EPOCHS_UNLEARN)
    parser.add_argument("--student-epochs",   type=int,   default=DKF_EPOCHS)
    parser.add_argument("--lambda-align",     type=float, default=LAMBDA_ALIGN)
    parser.add_argument("--lambda-forget",    type=float, default=LAMBDA_FORGET)
    parser.add_argument("--dkf-batches-per-epoch", type=int, default=None)
    parser.add_argument("--dkf-amp",              action="store_true")
    parser.add_argument("--max-eval-samples",     type=int,   default=None)
    parser.add_argument("--lambda-kd",            type=float, default=LAMBDA_KD)
    parser.add_argument("--student-epochs-v2",    type=int,   default=DKF_EPOCHS_V2)
    parser.add_argument("--lr-v2",                type=float, default=LR_DKF_V2)
    parser.add_argument("--lambda-forget-v2",     type=float, default=LAMBDA_FORGET_V2)
    args = parser.parse_args()

    setup_dirs()
    device = get_device()
    stages = set(args.stages)
    run_all = "all" in stages

    print(f"[Device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    names = class_names(download=args.download)
    print("[Forget]", ", ".join(f"{i}:{names[i]}" for i in FORGET_CLASSES))

    loaders = get_all_loaders(download=args.download)
    results = {}

    # ── Original model ─────────────────────────────────────────────────────────
    needs_original = run_all or bool(stages.intersection(
        {"original", "baselines", "dkf", "eradkf", "cdkf", "ceradkf", "cdkfv2", "ceradkfv2", "eval"}
    ))
    if needs_original:
        if args.reuse_checkpoints and os.path.exists(original_path()):
            original = load_model(original_path(), device)
        else:
            original = train_original(
                loaders["train"], loaders["test"], device,
                epochs=args.original_epochs, resume=args.reuse_checkpoints,
            )
        results["Original"] = evaluate_model(original, loaders, device, max_samples=args.max_eval_samples)

    # ── Baselines ──────────────────────────────────────────────────────────────
    if run_all or "baselines" in stages:
        retrain_model = maybe_load_or_train(
            "Retrain", ckpt("retrain_resnet50_cifar100.pth"),
            lambda: retrain(loaders["retain"], device, epochs=args.retrain_epochs),
            args.reuse_checkpoints, device,
        )
        results["Retrain"] = evaluate_model(retrain_model, loaders, device, max_samples=args.max_eval_samples)

        ft_model = maybe_load_or_train(
            "Fine-tune", ckpt("finetune_resnet50_cifar100.pth"),
            lambda: finetune(original, loaders["retain"], device, epochs=args.unlearn_epochs),
            args.reuse_checkpoints, device,
        )
        results["Fine-tune"] = evaluate_model(ft_model, loaders, device, max_samples=args.max_eval_samples)

        ng_model = maybe_load_or_train(
            "NegGrad", ckpt("neggrad_resnet50_cifar100.pth"),
            lambda: negative_gradient(
                original, loaders["forget"], loaders["retain"], device, epochs=args.unlearn_epochs
            ),
            args.reuse_checkpoints, device,
        )
        results["NegGrad"] = evaluate_model(ng_model, loaders, device, max_samples=args.max_eval_samples)

    retrain_ref = results.get("Retrain")
    if retrain_ref is None and os.path.exists(ckpt("retrain_resnet50_cifar100.pth")):
        retrain_ref = evaluate_model(
            load_model(ckpt("retrain_resnet50_cifar100.pth"), device),
            loaders, device, max_samples=args.max_eval_samples,
        )
        results["Retrain"] = retrain_ref

    shared_kw = dict(retrain_metrics=retrain_ref, max_samples=5000)

    # ── DKF (Week 9 unconditional baseline) ────────────────────────────────────
    if run_all or "dkf" in stages:
        dkf_model = maybe_load_or_train(
            "DKF", ckpt("dkf_resnet50_cifar100.pth"),
            lambda: train_student(
                original, loaders["forget"], loaders["retain"], device, method="dkf",
                student_epochs=args.student_epochs,
                max_batches_per_epoch=args.dkf_batches_per_epoch, use_amp=args.dkf_amp,
            ),
            args.reuse_checkpoints, device,
        )
        results["DKF"] = evaluate_shared_knowledge(original, dkf_model, loaders, device, **shared_kw)

    # ── E-RA-DKF (Week 9 unconditional baseline) ───────────────────────────────
    if run_all or "eradkf" in stages:
        eradkf_name = f"eradkf_resnet50_cifar100_la_{args.lambda_align}_lf_{args.lambda_forget}.pth"
        eradkf_model = maybe_load_or_train(
            "E-RA-DKF", ckpt(eradkf_name),
            lambda: train_student(
                original, loaders["forget"], loaders["retain"], device, method="eradkf",
                student_epochs=args.student_epochs,
                lambda_align=args.lambda_align, lambda_forget=args.lambda_forget,
                max_batches_per_epoch=args.dkf_batches_per_epoch, use_amp=args.dkf_amp,
            ),
            args.reuse_checkpoints, device,
        )
        results["E-RA-DKF"] = evaluate_shared_knowledge(original, eradkf_model, loaders, device, **shared_kw)

    # ── C-DKF (Week 10 — conditional VAE) ─────────────────────────────────────
    if run_all or "cdkf" in stages:
        from config import EMBED_DIM
        cdkf_name  = f"cdkf_resnet50_cifar100_emb{EMBED_DIM}.pth"
        cdkf_model = maybe_load_or_train(
            "C-DKF", ckpt(cdkf_name),
            lambda: train_student(
                original, loaders["forget"], loaders["retain"], device, method="cdkf",
                student_epochs=args.student_epochs,
                max_batches_per_epoch=args.dkf_batches_per_epoch, use_amp=args.dkf_amp,
            ),
            args.reuse_checkpoints, device,
        )
        results["C-DKF"] = evaluate_shared_knowledge(original, cdkf_model, loaders, device, **shared_kw)

    # ── CE-RA-DKF (Week 10 — conditional VAE + cosine align + detach) ──────────
    if run_all or "ceradkf" in stages:
        from config import EMBED_DIM
        ceradkf_name  = f"ceradkf_resnet50_cifar100_emb{EMBED_DIM}_la_{args.lambda_align}_lf_{args.lambda_forget}.pth"
        ceradkf_model = maybe_load_or_train(
            "CE-RA-DKF", ckpt(ceradkf_name),
            lambda: train_student(
                original, loaders["forget"], loaders["retain"], device, method="ceradkf",
                student_epochs=args.student_epochs,
                lambda_align=args.lambda_align, lambda_forget=args.lambda_forget,
                max_batches_per_epoch=args.dkf_batches_per_epoch, use_amp=args.dkf_amp,
            ),
            args.reuse_checkpoints, device,
        )
        results["CE-RA-DKF"] = evaluate_shared_knowledge(original, ceradkf_model, loaders, device, **shared_kw)

    # ── C-DKF v2 (conditional VAE + retain KD + tuned HPs) ────────────────────
    if run_all or "cdkfv2" in stages:
        from config import EMBED_DIM
        cdkf_v2_name  = f"cdkf_v2_resnet50_cifar100_emb{EMBED_DIM}_kd{args.lambda_kd}.pth"
        cdkf_v2_model = maybe_load_or_train(
            "C-DKF v2", ckpt(cdkf_v2_name),
            lambda: train_student(
                original, loaders["forget"], loaders["retain"], device,
                method="cdkf_v2",
                student_epochs=args.student_epochs_v2,
                lr=args.lr_v2,
                lambda_forget=args.lambda_forget_v2,
                lambda_kd=args.lambda_kd,
                max_batches_per_epoch=args.dkf_batches_per_epoch,
                use_amp=args.dkf_amp,
            ),
            args.reuse_checkpoints, device,
        )
        results["C-DKF v2"] = evaluate_shared_knowledge(original, cdkf_v2_model, loaders, device, **shared_kw)

    # ── CE-RA-DKF v2 (full stack + retain KD + tuned HPs) ─────────────────────
    if run_all or "ceradkfv2" in stages:
        from config import EMBED_DIM
        ceradkf_v2_name  = f"ceradkf_v2_resnet50_cifar100_emb{EMBED_DIM}_kd{args.lambda_kd}_lf_{args.lambda_forget_v2}.pth"
        ceradkf_v2_model = maybe_load_or_train(
            "CE-RA-DKF v2", ckpt(ceradkf_v2_name),
            lambda: train_student(
                original, loaders["forget"], loaders["retain"], device,
                method="ceradkf_v2",
                student_epochs=args.student_epochs_v2,
                lr=args.lr_v2,
                lambda_forget=args.lambda_forget_v2,
                lambda_align=args.lambda_align,
                lambda_kd=args.lambda_kd,
                max_batches_per_epoch=args.dkf_batches_per_epoch,
                use_amp=args.dkf_amp,
            ),
            args.reuse_checkpoints, device,
        )
        results["CE-RA-DKF v2"] = evaluate_shared_knowledge(original, ceradkf_v2_model, loaders, device, **shared_kw)

    # ── Output ─────────────────────────────────────────────────────────────────
    print_results_table(results)
    out = os.path.join(RESULTS_DIR, "week10_cvae_dkf_v2_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    # run from the week10_cvae_dkf directory so relative paths resolve correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

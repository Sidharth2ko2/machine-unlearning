"""Week 9 runner: CIFAR-100 + ResNet-50, forget 10 classes."""
import argparse
import json
import os

import torch

from baselines import finetune, negative_gradient, retrain, train_original
from config import (
    CHECKPOINT_DIR,
    DKF_EPOCHS,
    EPOCHS_ORIGINAL,
    EPOCHS_RETRAIN,
    EPOCHS_UNLEARN,
    FORGET_CLASSES,
    LAMBDA_ALIGN,
    LAMBDA_FORGET,
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
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model.to(device)


def original_path():
    return os.path.join(CHECKPOINT_DIR, "original_resnet50_cifar100.pth")


def ckpt_path(name):
    return os.path.join(CHECKPOINT_DIR, name)


def maybe_train_or_load(name, path, train_fn, reuse, device):
    if reuse and os.path.exists(path):
        print(f"[Loading] {name} <- {path}")
        return load_model(path, device)
    print(f"[Training] {name}")
    return train_fn()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["all"],
                        choices=["all", "original", "baselines", "dkf", "radkf", "eradkf", "eval"])
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--original-epochs", type=int, default=EPOCHS_ORIGINAL)
    parser.add_argument("--retrain-epochs", type=int, default=EPOCHS_RETRAIN)
    parser.add_argument("--unlearn-epochs", type=int, default=EPOCHS_UNLEARN)
    parser.add_argument("--student-epochs", type=int, default=DKF_EPOCHS)
    parser.add_argument("--lambda-align", type=float, default=LAMBDA_ALIGN)
    parser.add_argument("--lambda-forget", type=float, default=LAMBDA_FORGET)
    parser.add_argument("--dkf-batches-per-epoch", type=int, default=None,
                        help="Limit DKF-family retain batches per epoch. Useful on smaller GPUs.")
    parser.add_argument("--dkf-amp", action="store_true",
                        help="Use CUDA mixed precision for DKF-family training.")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="Debug only. Leave unset for final full evaluation.")
    args = parser.parse_args()

    setup_dirs()
    device = get_device()
    stages = set(args.stages)
    run_all = "all" in stages

    print(f"[Device] {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    names = class_names(download=args.download)
    print("[Forget classes]", ", ".join(f"{i}:{names[i]}" for i in FORGET_CLASSES))

    loaders = get_all_loaders(download=args.download)
    results = {}

    if run_all or "original" in stages or "baselines" in stages or "dkf" in stages or "radkf" in stages or "eradkf" in stages or "eval" in stages:
        if args.reuse_checkpoints and os.path.exists(original_path()):
            original = load_model(original_path(), device)
        else:
            original = train_original(
                loaders["train"],
                loaders["test"],
                device,
                epochs=args.original_epochs,
                resume=args.reuse_checkpoints,
            )
        results["Original"] = evaluate_model(original, loaders, device, max_samples=args.max_eval_samples)

    if run_all or "baselines" in stages:
        retrain_model = maybe_train_or_load(
            "Retrain",
            ckpt_path("retrain_resnet50_cifar100.pth"),
            lambda: retrain(loaders["retain"], device, epochs=args.retrain_epochs),
            args.reuse_checkpoints,
            device,
        )
        results["Retrain"] = evaluate_model(retrain_model, loaders, device, max_samples=args.max_eval_samples)

        ft_model = maybe_train_or_load(
            "Fine-tune",
            ckpt_path("finetune_resnet50_cifar100.pth"),
            lambda: finetune(original, loaders["retain"], device, epochs=args.unlearn_epochs),
            args.reuse_checkpoints,
            device,
        )
        results["Fine-tune"] = evaluate_model(ft_model, loaders, device, max_samples=args.max_eval_samples)

        ng_model = maybe_train_or_load(
            "NegGrad",
            ckpt_path("neggrad_resnet50_cifar100.pth"),
            lambda: negative_gradient(original, loaders["forget"], loaders["retain"], device, epochs=args.unlearn_epochs),
            args.reuse_checkpoints,
            device,
        )
        results["NegGrad"] = evaluate_model(ng_model, loaders, device, max_samples=args.max_eval_samples)

    retrain_ref = results.get("Retrain")
    if retrain_ref is None and os.path.exists(ckpt_path("retrain_resnet50_cifar100.pth")):
        retrain_ref = evaluate_model(load_model(ckpt_path("retrain_resnet50_cifar100.pth"), device), loaders, device, max_samples=args.max_eval_samples)
        results["Retrain"] = retrain_ref

    if run_all or "dkf" in stages:
        dkf_model = maybe_train_or_load(
            "DKF",
            ckpt_path("dkf_resnet50_cifar100.pth"),
            lambda: train_student(
                original,
                loaders["forget"],
                loaders["retain"],
                device,
                method="dkf",
                student_epochs=args.student_epochs,
                max_batches_per_epoch=args.dkf_batches_per_epoch,
                use_amp=args.dkf_amp,
            ),
            args.reuse_checkpoints,
            device,
        )
        results["DKF"] = evaluate_shared_knowledge(original, dkf_model, loaders, device, retrain_metrics=retrain_ref, max_samples=5000)

    if run_all or "radkf" in stages:
        radkf_model = maybe_train_or_load(
            "RA-DKF",
            ckpt_path(f"radkf_resnet50_cifar100_la_{args.lambda_align}.pth"),
            lambda: train_student(
                original,
                loaders["forget"],
                loaders["retain"],
                device,
                method="radkf",
                student_epochs=args.student_epochs,
                lambda_align=args.lambda_align,
                max_batches_per_epoch=args.dkf_batches_per_epoch,
                use_amp=args.dkf_amp,
            ),
            args.reuse_checkpoints,
            device,
        )
        results["RA-DKF"] = evaluate_shared_knowledge(original, radkf_model, loaders, device, retrain_metrics=retrain_ref, max_samples=5000)

    if run_all or "eradkf" in stages:
        eradkf_model = maybe_train_or_load(
            "E-RA-DKF",
            ckpt_path(f"eradkf_resnet50_cifar100_la_{args.lambda_align}_lf_{args.lambda_forget}.pth"),
            lambda: train_student(
                original,
                loaders["forget"],
                loaders["retain"],
                device,
                method="eradkf",
                student_epochs=args.student_epochs,
                lambda_align=args.lambda_align,
                lambda_forget=args.lambda_forget,
                max_batches_per_epoch=args.dkf_batches_per_epoch,
                use_amp=args.dkf_amp,
            ),
            args.reuse_checkpoints,
            device,
        )
        results["E-RA-DKF"] = evaluate_shared_knowledge(original, eradkf_model, loaders, device, retrain_metrics=retrain_ref, max_samples=5000)

    print_results_table(results)
    out = os.path.join(RESULTS_DIR, "week9_cifar100_resnet50_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()

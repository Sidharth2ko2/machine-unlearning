"""Baseline unlearning methods for Week 10 (identical to Week 9)."""
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    EPOCHS_RETRAIN,
    EPOCHS_UNLEARN,
    LR_FINETUNE,
    LR_NEGGRAD,
    LR_ORIGINAL,
    MOMENTUM,
    NEGGRAD_ALPHA,
    WEIGHT_DECAY,
)
from model_utils import build_resnet50


def _cycle(loader):
    while True:
        yield from loader


def train_original(train_loader, test_loader, device, epochs, resume=False):
    model     = build_resnet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_ORIGINAL, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    path      = os.path.join(CHECKPOINT_DIR, "original_resnet50_cifar100.pth")
    start_epoch, best_acc = 1, 0.0

    if resume and os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt["test_acc"]
        for _ in range(ckpt["epoch"]):
            scheduler.step()

    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Original", unit="epoch"):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        acc = _accuracy(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "test_acc": acc}, path)
        tqdm.write(f"Epoch {epoch:3d}/{epochs}  test={acc:.2f}%  best={best_acc:.2f}%")
    return model


@torch.no_grad()
def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(total, 1)


def retrain(retain_loader, device, epochs=EPOCHS_RETRAIN):
    model     = build_resnet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_ORIGINAL, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in tqdm(range(epochs), desc="Retrain", unit="epoch"):
        model.train()
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "retrain_resnet50_cifar100.pth"))
    return model


def finetune(original_model, retain_loader, device, epochs=EPOCHS_UNLEARN):
    model     = copy.deepcopy(original_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_FINETUNE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    for _ in tqdm(range(epochs), desc="Fine-tune", unit="epoch"):
        model.train()
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "finetune_resnet50_cifar100.pth"))
    return model


def negative_gradient(original_model, forget_loader, retain_loader, device, epochs=EPOCHS_UNLEARN, alpha=NEGGRAD_ALPHA):
    model     = copy.deepcopy(original_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_NEGGRAD, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    forget_cycle = _cycle(forget_loader)
    for _ in tqdm(range(epochs), desc="NegGrad", unit="epoch"):
        model.train()
        for x_r, y_r in retain_loader:
            x_f, y_f = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f, y_f = x_r[:b], y_r[:b], x_f[:b], y_f[:b]
            optimizer.zero_grad()
            loss = criterion(model(x_r), y_r) - alpha * criterion(model(x_f), y_f)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "neggrad_resnet50_cifar100.pth"))
    return model

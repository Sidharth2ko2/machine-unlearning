"""Evaluation utilities for Week 10 — identical to Week 9."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

from model_utils import get_features


@torch.no_grad()
def accuracy(model, loader, device, max_samples=None):
    model.eval()
    correct = total = 0
    for x, y in loader:
        if max_samples is not None:
            remaining = max_samples - total
            if remaining <= 0:
                break
            x, y = x[:remaining], y[:remaining]
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def per_sample_loss(model, loader, device, max_samples=None):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses, seen = [], 0
    for x, y in loader:
        if max_samples is not None:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            x, y = x[:remaining], y[:remaining]
        x, y = x.to(device), y.to(device)
        losses.extend(criterion(model(x), y).cpu().numpy())
        seen += y.size(0)
    return np.array(losses)


def membership_inference_attack(model, forget_loader, forget_class_test_loader, device, max_samples=None):
    forget_losses = per_sample_loss(model, forget_loader,            device, max_samples=max_samples)
    test_losses   = per_sample_loss(model, forget_class_test_loader, device, max_samples=max_samples)
    n = min(len(forget_losses), len(test_losses))
    if n < 20:
        return 50.0
    x_mia = np.concatenate([forget_losses[:n], test_losses[:n]]).reshape(-1, 1)
    y_mia = np.concatenate([np.ones(n), np.zeros(n)])
    clf   = LogisticRegression(max_iter=1000)
    clf.fit(x_mia, y_mia)
    return accuracy_score(y_mia, clf.predict(x_mia)) * 100.0


def evaluate_model(model, loaders, device, max_samples=None):
    return {
        "Acc_Dr":  accuracy(model, loaders["retain"], device, max_samples=max_samples),
        "Acc_Df":  accuracy(model, loaders["forget"], device, max_samples=max_samples),
        "Acc_val": accuracy(model, loaders["test"],   device, max_samples=max_samples),
        "MIA": membership_inference_attack(
            model, loaders["forget"], loaders["forget_class_test"], device, max_samples=max_samples
        ),
    }


def avg_gap(metrics, retrain_metrics):
    keys = ("Acc_Dr", "Acc_Df", "Acc_val", "MIA")
    return sum(abs(metrics[k] - retrain_metrics[k]) for k in keys) / len(keys)


@torch.no_grad()
def feature_drift_metrics(teacher, student, loader, device, max_samples=5000):
    teacher.eval(); student.eval()
    zt_list, zs_list, seen = [], [], 0
    for x, _ in loader:
        remaining = max_samples - seen
        if remaining <= 0:
            break
        x = x[:remaining].to(device)
        zt_list.append(get_features(teacher, x).cpu())
        zs_list.append(get_features(student, x).cpu())
        seen += x.size(0)
    z_t = F.normalize(torch.cat(zt_list, dim=0), dim=1)
    z_s = F.normalize(torch.cat(zs_list, dim=0), dim=1)
    cosine = F.cosine_similarity(z_t, z_s, dim=1)
    return {
        "Retain_Feature_Cosine": float(cosine.mean()),
        "Retain_Feature_Drift":  float(1.0 - cosine.mean()),
        "Retain_Feature_MSE":    float(F.mse_loss(z_s, z_t)),
    }


@torch.no_grad()
def agreement_and_kl(teacher, student, loader, device, max_samples=5000):
    teacher.eval(); student.eval()
    total = agree = 0
    kl_sum = 0.0
    for x, _ in loader:
        remaining = max_samples - total
        if remaining <= 0:
            break
        x = x[:remaining].to(device)
        t_logits = teacher(x)
        s_logits = student(x)
        agree  += (t_logits.argmax(1) == s_logits.argmax(1)).sum().item()
        kl_sum += F.kl_div(
            F.log_softmax(s_logits, dim=1),
            F.softmax(t_logits, dim=1),
            reduction="batchmean",
        ).item() * x.size(0)
        total += x.size(0)
    return {
        "Retain_Agreement": 100.0 * agree / max(total, 1),
        "Retain_KL":        kl_sum / max(total, 1),
    }


@torch.no_grad()
def clustering_metrics(model, loader, device, max_samples=5000):
    model.eval()
    features, labels, seen = [], [], 0
    for x, y in loader:
        remaining = max_samples - seen
        if remaining <= 0:
            break
        x = x[:remaining].to(device)
        features.append(get_features(model, x).cpu())
        labels.append(y[:remaining].cpu())
        seen += x.size(0)
    z = F.normalize(torch.cat(features, dim=0), dim=1).numpy()
    y = torch.cat(labels, dim=0).numpy()
    n_clusters = len(np.unique(y))
    if n_clusters < 2:
        return {"Retain_ARI": 0.0, "Retain_NMI": 0.0}
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(z)
    return {
        "Retain_ARI": float(adjusted_rand_score(y, pred)),
        "Retain_NMI": float(normalized_mutual_info_score(y, pred)),
    }


def evaluate_shared_knowledge(teacher, student, loaders, device, retrain_metrics=None, max_samples=5000):
    out = evaluate_model(student, loaders, device)
    out.update(feature_drift_metrics(teacher, student, loaders["retain"], device, max_samples))
    out.update(agreement_and_kl(teacher, student, loaders["retain"], device, max_samples))
    out.update(clustering_metrics(student, loaders["retain"], device, max_samples))
    if retrain_metrics is not None:
        out["Avg.Gap"] = avg_gap(out, retrain_metrics)
    return out


def print_results_table(results):
    retrain = results.get("Retrain")
    header  = f"{'Method':<25} {'Acc_Dr':>8} {'Acc_Df':>8} {'Acc_val':>8} {'MIA':>8} {'AvgGap':>8} {'RetDrift':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in results.items():
        gap   = metrics.get("Avg.Gap")
        if gap is None and retrain and name != "Retrain":
            gap = avg_gap(metrics, retrain)
        gap_s   = "   --  " if gap   is None else f"{gap:7.2f}%"
        drift_s = "   --  " if metrics.get("Retain_Feature_Drift") is None else f"{metrics['Retain_Feature_Drift']:9.4f}"
        print(
            f"{name:<25} {metrics['Acc_Dr']:7.2f}%  {metrics['Acc_Df']:7.2f}%  "
            f"{metrics['Acc_val']:7.2f}%  {metrics['MIA']:7.2f}%  {gap_s}  {drift_s}"
        )

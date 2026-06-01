"""CIFAR-100 loaders with 10-class forget split."""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from config import BATCH_SIZE, DATA_DIR, FORGET_CLASSES, NUM_WORKERS


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_tf, test_tf


def get_datasets(download=True):
    train_tf, test_tf = get_transforms()
    train_ds = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=True,  download=download, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=False, download=download, transform=test_tf)
    return train_ds, test_ds


def split_forget_retain(train_ds):
    targets    = np.array(train_ds.targets)
    forget_idx = np.where(np.isin(targets, FORGET_CLASSES))[0]
    retain_idx = np.where(~np.isin(targets, FORGET_CLASSES))[0]
    return Subset(train_ds, forget_idx), Subset(train_ds, retain_idx)


def get_all_loaders(batch_size=BATCH_SIZE, download=True):
    train_ds, test_ds = get_datasets(download=download)
    forget_ds, retain_ds = split_forget_retain(train_ds)

    test_targets      = np.array(test_ds.targets)
    forget_test_idx   = np.where(np.isin(test_targets, FORGET_CLASSES))[0]

    return {
        "train":  DataLoader(train_ds,  batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS),
        "test":   DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS),
        "forget": DataLoader(forget_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS),
        "retain": DataLoader(retain_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS),
        "forget_class_test": DataLoader(
            Subset(test_ds, forget_test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        ),
    }


def class_names(download=True):
    train_ds, _ = get_datasets(download=download)
    return train_ds.classes

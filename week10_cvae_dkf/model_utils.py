"""Model helpers for CIFAR-sized ResNet-50."""
import torch.nn as nn
from torchvision.models import resnet50

from config import NUM_CLASSES


def build_resnet50(num_classes=NUM_CLASSES):
    """
    ResNet-50 adapted for 32x32 CIFAR images.

    Standard ImageNet ResNet uses 7x7 stride-2 conv + maxpool, which is too
    aggressive for CIFAR. The CIFAR adaptation uses 3x3 stride-1 conv and no
    initial maxpool, a common setup for small images.
    """
    model = resnet50(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_features(model, x):
    """Return 2048-dim avgpool features from ResNet-50."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return x.flatten(1)


def get_features_and_logits(model, x):
    """Return avgpool features and classifier logits from one ResNet pass."""
    features = get_features(model, x)
    logits   = model.fc(features)
    return features, logits

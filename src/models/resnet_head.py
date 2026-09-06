"""Frozen ResNet-18 backbone + small trainable classifier head.

IMPORTANT design decision (mirrors the approved system design):
the pretrained ResNet-18 backbone is frozen (``requires_grad=False`` and the
module is kept in ``eval()`` so BatchNorm statistics never move).  Only the
512->C classifier head is ever trained, aggregated, or unlearned.  This is
what makes the "Head-only KAF" contribution meaningful and keeps every
experiment CPU-feasible.

The backbone is applied at the MedMNIST native 28x28 resolution (see README,
"Known issues / challenges" — ImageNet backbones are trained at 224x224).
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_backbone() -> nn.Module:
    """Pretrained ImageNet ResNet-18, frozen, with the head chopped off."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()          # forward(x) -> (B, 512)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone


def build_head(n_classes: int, seed: int = 0) -> nn.Linear:
    """512 -> n_classes linear classifier with deterministic init."""
    g = torch.Generator().manual_seed(seed)
    head = nn.Linear(512, n_classes)
    head.reset_parameters()  # reset() draws from torch's global RNG...
    # ...so reseed explicitly for reproducibility.
    nn.init.kaiming_uniform_(head.weight, a=5 ** 0.5, generator=g)
    nn.init.zeros_(head.bias)
    return head


def head_params_count(head: nn.Linear) -> int:
    return sum(p.numel() for p in head.parameters())

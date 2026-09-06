"""One federated client: local head training on its own feature tuples."""
from __future__ import annotations

import torch
from torch import nn

from src.models.resnet_head import build_head


def train_head_local(
    head_state: dict,
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str = "cpu",
    weight_decay: float = 0.0,
    optimizer: str = "adam",
) -> dict:
    """Train a classifier head for ``epochs`` epochs on local (feature,label) tuples.

    ``head_state`` is the *received global* head — we warm-start from it,
    which is exactly what FedAvg does.  Only head parameters are optimised.
    Returns the updated head state_dict.
    """
    head = build_head(n_classes, seed=seed).to(device)
    head.load_state_dict(head_state)
    head.train()

    if optimizer.lower() == "sgd":
        opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    rng = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=rng)
        for i in range(0, len(y), batch_size):
            bidx = perm[i : i + batch_size]
            xb, yb = X[bidx].to(device), y[bidx].to(device)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in head.state_dict().items()}

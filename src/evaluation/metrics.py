"""Utility, cost, and per-client metrics."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.models.resnet_head import build_head


@torch.no_grad()
def _predict_np(head_state: dict, X, n_classes: int) -> np.ndarray:
    head = build_head(n_classes, seed=0)
    head.load_state_dict({k: v for k, v in head_state.items()})
    head.eval()
    return head(X).argmax(1).numpy()


def weighted_f1(head_state: dict, X, y, n_classes: int, labels=None) -> float:
    pred = _predict_np(head_state, X, n_classes)
    return float(f1_score(y.numpy(), pred, average="weighted", labels=labels or None))


def per_client_f1(head_state: dict, client_Xy: list[tuple], n_classes: int) -> list[float]:
    """Weighted F1 per retained client on that client's LOCAL validation set."""
    out = []
    for X, y in client_Xy:
        if len(y) == 0:
            out.append(float("nan"))
            continue
        pred = _predict_np(head_state, X, n_classes)
        out.append(float(f1_score(y.numpy(), pred, average="weighted")))
    return out


def communication_bytes_proxy(n_rounds: int, n_clients: int, head_params: int) -> float:
    """Rough proxy: rounds x participating clients x bytes for one fp32 head."""
    return float(n_rounds) * float(n_clients) * float(head_params) * 4

"""FedAvg server: aggregation loop over simulated hospitals.

Only the classifier-head parameters are exchanged/aggregated (the backbone is
frozen and never leaves each client's feature cache).  Communication volume is
therefore proportional to the head size — we report the round count and a byte
proxy in the evaluation module.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

from src.federated.client import train_head_local
from src.models.resnet_head import build_head


def fedavg_fit(
    client_data: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    num_rounds: int,
    local_epochs: int,
    clients_per_round: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str = "cpu",
    init_head_state: Optional[dict] = None,
    log_fn: Callable[[int, dict], None] | None = None,
    eval_every: int = 5,
    eval_fn: Optional[Callable[[dict], dict]] = None,
    client_ids: Optional[list[int]] = None,
) -> tuple[dict, dict]:
    """Run FedAvg and return ``(final_head_state, history)``.

    ``client_data[k] = (X_k, y_k)`` — feature matrix + labels for client k.
    ``client_ids`` optionally restricts which clients participate (used by
    retrain / fine-tune after a client has left).  ``eval_fn(head_state)``
    returns a metrics dict logged every ``eval_every`` rounds.
    """
    rng = np.random.default_rng(seed)
    if init_head_state is None:
        head = build_head(n_classes, seed=seed)
        head_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    else:
        head_state = {k: v.clone() for k, v in init_head_state.items()}

    ids = list(range(len(client_data))) if client_ids is None else list(client_ids)
    sizes = {k: len(client_data[k][0]) for k in ids}
    total = float(sum(sizes.values()))
    history: dict[str, list] = {"round": [], "loss": [], "eval": []}

    t_start = time.time()
    for r in range(1, num_rounds + 1):
        chosen = ids if clients_per_round >= len(ids) else list(rng.choice(ids, clients_per_round, replace=False))
        updates, losses = [], []
        for k in chosen:
            X_k = torch.as_tensor(client_data[k][0]).float()
            y_k = torch.as_tensor(client_data[k][1]).long()
            new_state = train_head_local(
                head_state, X_k, y_k, n_classes,
                epochs=local_epochs, batch_size=batch_size, lr=lr,
                seed=int(rng.integers(0, 2**31)), device=device,
            )
            updates.append((k, new_state))
            losses.append(float(torch.nn.functional.cross_entropy(
                torch.as_tensor(client_data[k][0]).float() @ new_state["weight"].t() + new_state["bias"],
                torch.as_tensor(client_data[k][1]).long()
            ).item()))

        # Weighted averaging by local sample count (FedAvg).
        avg = {}
        for key in updates[0][1].keys():
            w = torch.zeros_like(updates[0][1][key])
            for k, st in updates:
                w = w + (sizes[k] / total) * st[key]
            avg[key] = w
        head_state = avg

        if log_fn is not None or (eval_fn is not None and (r % eval_every == 0 or r == num_rounds)):
            ev = eval_fn(head_state) if eval_fn and (r % eval_every == 0 or r == num_rounds) else {}
            history["round"].append(r)
            history["loss"].append(float(np.mean(losses)))
            history["eval"].append(ev)
            if log_fn:
                log_fn(r, {"mean_local_loss": float(np.mean(losses)), **ev})

    return head_state, {"history": history, "wall_seconds": time.time() - t_start}


def evaluate_head(
    head_state: dict, X: torch.Tensor, y: torch.Tensor, n_classes: int
) -> dict:
    """Accuracy / CE of a head state on a feature matrix (no gradients)."""
    with torch.no_grad():
        head = build_head(n_classes, seed=0)
        head.load_state_dict({k: v for k, v in head_state.items()})
        logits = head(X)
        loss = torch.nn.functional.cross_entropy(logits, y).item()
        acc = (logits.argmax(1) == y).float().mean().item()
    return {"ce": loss, "acc": acc}

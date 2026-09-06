"""Method 1 — Full retraining (gold standard).

Re-runs the exact FedAvg procedure from scratch using only the N-1 retained
clients.  This is the upper bound on forgetting quality AND on cost: whatever
an approximate method removes, retraining-at-zero sets the target, and every
other method is reported relative to its cost.

The only difference from the baseline run is the client set, so we reuse
``fedavg_fit`` unchanged with ``client_ids`` = all clients except the removed
one and no warm-start head.
"""
from __future__ import annotations

from src.federated.server import fedavg_fit


def retrain(
    client_data: list,
    removed_client_id: int,
    n_classes: int,
    fc: dict,   # fedavg config section
    seed: int,
    device: str = "cpu",
    log_fn=None,
    eval_fn=None,
):
    retained = [k for k in range(len(client_data)) if k != removed_client_id]
    return fedavg_fit(
        client_data, n_classes,
        num_rounds=fc["num_rounds"], local_epochs=fc["local_epochs"],
        clients_per_round=fc["clients_per_round"], batch_size=fc["batch_size"],
        lr=fc["lr"], weight_decay=fc.get("weight_decay", 0.0), optimizer=fc.get("optimizer", "adam"),
        seed=seed, device=device,
        init_head_state=None,
        log_fn=log_fn, eval_every=fc.get("eval_every", 5) or fc.get("eval_every", 5),
        eval_fn=eval_fn, client_ids=retained,
    )

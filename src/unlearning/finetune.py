"""Method 2 — Fine-tune on the retained clients.

Starts from the fully-trained baseline head and continues FedAvg for a few
extra rounds using only the N-1 retained clients.  Cheap and fast, but the
baseline's memory of the removed client persists -> it is *expected* to forget
poorly.  Reporting that honest "bad" number is the point of including it as a
baseline comparator.
"""
from __future__ import annotations

from src.federated.server import fedavg_fit


def finetune(
    client_data: list,
    removed_client_id: int,
    n_classes: int,
    baseline_head_state: dict,
    fc: dict,
    extra_rounds: int,
    seed: int,
    device: str = "cpu",
    log_fn=None,
    eval_fn=None,
):
    retained = [k for k in range(len(client_data)) if k != removed_client_id]
    return fedavg_fit(
        client_data, n_classes,
        num_rounds=extra_rounds, local_epochs=fc["local_epochs"],
        clients_per_round=fc["clients_per_round"], batch_size=fc["batch_size"],
        lr=fc["lr"], weight_decay=fc.get("weight_decay", 0.0), optimizer=fc.get("optimizer", "adam"),
        seed=seed, device=device,
        init_head_state=baseline_head_state,
        log_fn=log_fn, eval_every=extra_rounds, eval_fn=eval_fn, client_ids=retained,
    )

"""Dirichlet non-IID client partitioning over MedMNIST tuples.

Each record is a complete (image, label) tuple: partitioning assigns whole
tuples (by their index in the medmnist training split), never labels alone.

Every client's allocated tuples are further split into a local *training* set
and a local *validation* set (``local_val_fraction``).  The local validation
tuples are NEVER trained on by any method — they exist so the membership-
inference attack has "definitely never trained" samples drawn from the SAME
distribution as the removed client's training data (a fair non-member pool).
"""
from __future__ import annotations

import numpy as np

from src.data.loaders import load_arrays


def client_counts_dirichlet(labels: np.ndarray, num_clients: int, alpha: float, rng) -> np.ndarray:
    """Draw per-client class proportions from Dirichlet(alpha).

    Returns a (num_clients, n_classes) weight matrix (rows sum to 1).
    ``alpha -> inf`` behaves like IID; ``alpha -> 0`` maximally skews clients.
    """
    n_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    props = rng.dirichlet([alpha] * n_classes, size=num_clients)
    # Zero-out proportions for classes the split does not actually contain.
    props *= (counts > 0)[None, :]
    row_sums = props.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    props /= row_sums
    return props


def dirichlet_partition(
    dataset: str,
    num_clients: int,
    alpha: float,
    seed: int,
    cap_per_client: int | None = None,
    local_val_fraction: float = 0.2,
) -> dict:
    """Partition the medmnist TRAIN split into ``num_clients`` hospitals.

    Returns a dict:
        "n_classes", "class_names", "test_*",
        "clients": list of dicts with keys
            "train_idx", "val_idx"   (indexes into the medmnist train split)
            "class_dist"             (per-class counts of the whole allocation)
    """
    rng = np.random.default_rng(seed)
    imgs, labels, n_classes, class_names = load_arrays(dataset, "train")

    props = client_counts_dirichlet(labels, num_clients, alpha, rng)
    per_class = [np.flatnonzero(labels == c) for c in range(n_classes)]

    alloc: list[list[int]] = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        idx_c = per_class[c]
        if len(idx_c) == 0:
            continue
        w = props[:, c]
        w = w / w.sum()                       # exact normalisation
        counts = rng.multinomial(len(idx_c), w)
        pool = idx_c.copy()
        rng.shuffle(pool)
        p = 0
        for k in range(num_clients):
            alloc[k].extend(pool[p : p + int(counts[k])].tolist())
            p += int(counts[k])

    clients = []
    for k in range(num_clients):
        idx = np.asarray(alloc[k], dtype=np.int64)
        if cap_per_client is not None and len(idx) > cap_per_client:
            idx = idx[:cap_per_client]
        n_val = max(1, int(round(len(idx) * local_val_fraction)))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        dist = np.bincount(labels[idx], minlength=n_classes)
        clients.append({"train_idx": train_idx, "val_idx": val_idx, "class_dist": dist})

    # Global held-out test split (never distributed to any client).
    timgs, tlabels, _, _ = load_arrays(dataset, "test")
    return {
        "n_classes": n_classes,
        "class_names": class_names,
        "clients": clients,
        "test": {"idx": np.arange(len(timgs))},
        "client_class_dists": np.stack([c["class_dist"] for c in clients]),
    }

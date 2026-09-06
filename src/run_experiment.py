"""Single-dataset experiment: baseline + 3 unlearning methods + full metrics.

Reproduces scope Section 5 for ONE dataset.  ``experiments/run_all_scoped.py``
calls this for both datasets and aggregates the CSV + plots.

Returns a list of row-dicts (one per method) plus per-dataset training history.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from src.config import dataset_config
from src.data.loaders import extract_features, load_arrays
from src.data.partition import dirichlet_partition
from src.evaluation.deleted_class_probe import deleted_client_probe
from src.evaluation.metrics import communication_bytes_proxy, weighted_f1
from src.evaluation.mia import mia_auroc
from src.federated.server import fedavg_fit, evaluate_head
from src.models.resnet_head import build_backbone, build_head, head_params_count
from src.unlearning.finetune import finetune
from src.unlearning.head_only_kaf import head_only_kaf
from src.unlearning.retrain import retrain


class _Logger:
    def __init__(self, tag: str, quiet: bool = False):
        self.tag, self.quiet = tag, quiet

    def __call__(self, stage, info):
        if not self.quiet:
            print(f"[{self.tag}] {stage}: {info}")


def run_dataset(dataset: str, cfg: dict | None = None, quiet: bool = False):
    cfg = dataset_config(dataset, cfg)
    device = cfg["device"]
    seed = cfg["seed"]
    N = cfg["num_clients"]
    rem = cfg["removal_client_id"]
    alpha = cfg["alpha"]
    fc = cfg["fedavg"]
    dc = cfg["data"]
    mc = cfg["mia"]
    cap = dc.get("path_train_cap_per_client") if dataset == "path" else None

    log = _Logger(dataset, quiet=quiet)

    par = dirichlet_partition(dataset, N, alpha, seed, cap_per_client=cap,
                              local_val_fraction=dc["local_val_fraction"])
    n_classes = par["n_classes"]

    print(f"\n===== {dataset.upper()} : {N} clients, alpha={alpha}, "
          f"cap/client={cap}, remove client {rem} =====")

    # People may want the raw arrays; reload labels only (imgs stay in cache path).
    _, train_labels, _, class_names = load_arrays(dataset, "train")
    _, test_labels, _, _ = load_arrays(dataset, "test")

    # ---- Build the set of tuples we actually need features for ----------
    per_client = par["clients"]
    all_idx = np.concatenate([np.concatenate([c["train_idx"], c["val_idx"]]) for c in per_client])
    pos = {int(idx): i for i, idx in enumerate(all_idx)}
    test_idx = par["test"]["idx"]

    backbone = build_backbone()
    feats_all = extract_features(dataset, "train", all_idx, backbone, device=device)
    feats_test = extract_features(dataset, "test", test_idx, backbone, device=device)
    del backbone

    # ---- Per-client tensors ----------------------------------------------
    client_data = []            # (X_train, y_train) per client k
    client_val = []             # (X_val, y_val) per client k  (never trained)
    for c in per_client:
        tr = c["train_idx"]
        vl = c["val_idx"]
        Xtr = torch.from_numpy(feats_all[[pos[int(i)] for i in tr]]).float()
        ytr = torch.from_numpy(train_labels[tr]).long()
        Xvl = torch.from_numpy(feats_all[[pos[int(i)] for i in vl]]).float()
        yvl = torch.from_numpy(train_labels[vl]).long()
        client_data.append((Xtr, ytr))
        client_val.append((Xvl, yvl))

    X_test = torch.from_numpy(feats_test).float()
    y_test = torch.from_numpy(test_labels[test_idx]).long()

    # Standardise the 512-d features using global client-train statistics.
    # The frozen backbone's features are large-magnitude (~hundreds); a linear
    # head directly on them is over-confident and collapses under FedAvg's
    # weight averaging across skewed clients. Z-scoring stabilises the head
    # and is documented in README (Known issues).
    all_train = torch.cat([X for X, _ in client_data], dim=0)
    mu = all_train.mean(dim=0, keepdim=True)
    sd = all_train.std(dim=0, unbiased=False, keepdim=True) + 1e-8
    scale = lambda t: (t - mu) / sd
    client_data = [(scale(X), y) for X, y in client_data]
    client_val = [(scale(X), y) for X, y in client_val]
    X_test = scale(X_test)

    head_params = head_params_count(build_head(n_classes, seed=seed))

    def eval_fn(head_state: dict, tag: str = "eval"):
        ev = evaluate_head(head_state, X_test, y_test, n_classes)
        log(tag, {k: round(v, 4) for k, v in ev.items()})
        return ev

    # ===================== BASELINE ======================================
    print("[baseline] training FedAvg ...")
    t0 = time.time()
    base_head, base_stats = fedavg_fit(
        client_data, n_classes,
        num_rounds=fc["num_rounds"], local_epochs=fc["local_epochs"],
        clients_per_round=fc["clients_per_round"], batch_size=fc["batch_size"],
        lr=fc["lr"], weight_decay=fc.get("weight_decay", 0.0), optimizer=fc.get("optimizer", "adam"),
        seed=seed, device=device,
        log_fn=log, eval_every=cfg["eval"]["log_every"], eval_fn=eval_fn,
    )
    base_train_time = time.time() - t0
    base_f1 = weighted_f1(base_head, X_test, y_test, n_classes)

    # ===================== UNLEARNING =====================================
    mX_mem, my_mem = client_data[rem]
    rem_val_X, rem_val_y = client_val[rem]      # removed client's held-out (KAF target + probe)

    # Class-matched NON-members for MIA.  Sampling the removed client's train
    # tuples (members) vs its local-val tuples (non-members) has a class/
    # covariate confound that makes AUROC ~constant and uninformative (seen in
    # a first run: retrain AUROC == baseline AUROC).  Instead, non-members are
    # drawn from the GLOBAL TEST set, matched to the members' class counts, so
    # the only signal left is genuine training-membership.  See README,
    # "Known issues / challenges".
    from collections import Counter
    rem_class_cnt = Counter(my_mem.tolist())
    g = torch.Generator().manual_seed(seed)
    nm_idx = []
    for c, nc in rem_class_cnt.items():
        cand = (y_test == c).nonzero().flatten()
        if len(cand) == 0:
            continue
        k = min(int(nc), len(cand))
        nm_idx.append(cand[torch.randperm(len(cand), generator=g)[:k]])
    nm_idx = torch.cat(nm_idx) if nm_idx else torch.tensor([], dtype=torch.long)
    nm_X = X_test[nm_idx]
    nm_y = y_test[nm_idx]

    retained = [k for k in range(N) if k != rem]

    methods = {}
    results = []

    def score(head, n_rounds, n_clients_used, unlearn_time, use_nm: tuple | None = None):
        non_x, non_y = use_nm if use_nm is not None else (nm_X, nm_y)
        auroc = mia_auroc(head if head is not None else base_head, mX_mem, my_mem,
                          non_x, non_y, n_classes, test_fraction=mc["test_fraction"],
                          seed=seed)
        f1 = weighted_f1(head, X_test, y_test, n_classes)
        probe = deleted_client_probe(head, rem_val_X, rem_val_y, n_classes)
        comm = communication_bytes_proxy(n_rounds, n_clients_used, head_params)
        return {"mia_auroc": auroc, "retained_f1": f1, "probe_acc": probe["accuracy"],
                "n_rounds": n_rounds, "comm_bytes": comm, "unlearn_time_s": unlearn_time}

    # 0) Baseline (all clients, no removal) — reference point
    results.append({
        "method": "baseline", "dataset": dataset,
        **score(base_head, fc["num_rounds"], N, base_train_time),
    })

    # 1) Retrain (gold)
    print("[retrain] full FedAvg on retained clients ...")
    t0 = time.time()
    rhead, rstat = retrain(client_data, rem, n_classes, fc, seed, device=device,
                           log_fn=log, eval_fn=lambda hs: eval_fn(hs, "retrain"))
    rt = time.time() - t0
    # score needs the *post* model for MIA, so pass rhead
    r = score(rhead, fc["num_rounds"], N - 1, rt); r["method"] = "retrain"; r["dataset"] = dataset
    results.append(r)

    # 2) Fine-tune
    print("[finetune] warm-start, few rounds on retained ...")
    t0 = time.time()
    fhead, fstat = finetune(client_data, rem, n_classes, base_head, fc,
                            cfg["finetune"]["extra_rounds"], seed, device=device,
                            log_fn=log, eval_fn=lambda hs: eval_fn(hs, "finetune"))
    ft = time.time() - t0
    r = score(fhead, cfg["finetune"]["extra_rounds"], N - 1, ft); r["method"] = "finetune"; r["dataset"] = dataset
    results.append(r)

    # 3) Head-only KAF (ours)
    print("[head_only_kaf] prior + forget on the removed client's held-out ...")
    t0 = time.time()
    khead, kstat = head_only_kaf(
        client_data, rem, n_classes, base_head, rem_val_X, rem_val_y,
        cfg["head_only_kaf"], fc, seed, device=device,
        log_fn=log, eval_fn=lambda hs: eval_fn(hs, "kaf"),
    )
    kt = time.time() - t0
    n_rounds = cfg["head_only_kaf"]["epochs"] + cfg["head_only_kaf"]["stabilize_rounds"]
    r = score(khead, n_rounds, N - 1, kt); r["method"] = "head_only_kaf"; r["dataset"] = dataset
    results.append(r)

    history = {"baseline": base_stats, "retrain": {}, "finetune": {}, "kaf": {}}
    return results, history

"""Membership-inference attack (MIA) for measuring forgetting.

Standard setup used throughout the unlearning literature (Shokri et al.):
a simple attacker is trained to tell apart *members* (tuples the model was
trained on) from *non-members* (tuples the model never saw) using only the
model's outputs (here: cross-entropy loss, max softmax confidence, and output
entropy).  AUROC ~= 0.5 means the attacker cannot distinguish members from
non-members -> good forgetting.  AUROC -> 1.0 means the model still memorizes
the deleted data -> poor forgetting.

After a client is removed we re-run the attack with:
    members    = the removed client's TRAINING tuples (are they still leaked?)
    non-members= the removed client's LOCAL-VALIDATION tuples (same class
                 distribution, but never trained on -> a fair held-out pool).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.models.resnet_head import build_head


@torch.no_grad()
def score_features(
    head_state: dict, X: torch.Tensor, y: torch.Tensor, n_classes: int
) -> np.ndarray:
    """Per-sample attack features: [-CE, max-softmax-confidence, entropy]."""
    head = build_head(n_classes, seed=0)
    head.load_state_dict({k: v for k, v in head_state.items()})
    head.eval()
    logits = head(X)
    ce = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    p = torch.softmax(logits, dim=1)
    conf = p.max(dim=1).values
    ent = -(p * torch.log(p + 1e-12)).sum(dim=1)
    return torch.stack([-ce, conf, ent], dim=1).numpy()


def mia_auroc(
    head_state: dict,
    member_X: torch.Tensor, member_y: torch.Tensor,
    nonmember_X: torch.Tensor, nonmember_y: torch.Tensor,
    n_classes: int,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> float:
    """Train an LR attacker and report held-out AUROC. Lower = better forgetting."""
    fm = score_features(head_state, member_X, member_y, n_classes)
    fn = score_features(head_state, nonmember_X, nonmember_y, n_classes)
    X = np.vstack([fm, fn])
    y = np.concatenate([np.zeros(len(fm)), np.ones(len(fn))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_fraction, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, proba))

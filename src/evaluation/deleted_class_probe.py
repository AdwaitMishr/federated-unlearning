"""Deleted-class probe (secondary forgetting signal for client-level removal).

For pure *client-level* removal there is no single "deleted class".  We
instead probe residual memory of the REMOVED client's *local class
distribution*: how confidently / accurately does the unlearned model still
recognise the tuples that came from the hospital that left?

Interpretation: a higher residual accuracy on the removed client's held-out
tuples suggests the model still "remembers" that site's distribution.
This is reported as a secondary signal alongside the MIA AUROC.  If the
removed client's local classes heavily overlap the retained clients' classes
(which is common when alpha=0.3, N=5), the probe is expected to stay fairly
high even for a perfect retrain — so read it relative to the Retrain baseline,
not as an absolute.
"""
from __future__ import annotations

import torch

from src.models.resnet_head import build_head

SKIP_REASON = (
    "Deleted-class probe reports residual accuracy/confidence on the removed "
    "client's local-valid set. We keep it as a secondary signal; for client-level "
    "removal with overlapping classes it is most meaningful relative to Retrain."
)


@torch.no_grad()
def deleted_client_probe(
    head_state: dict,
    removed_X: torch.Tensor, removed_y: torch.Tensor, n_classes: int
) -> dict:
    if len(removed_y) == 0:
        return {"accuracy": float("nan"), "avg_conf": float("nan"), "count": 0}
    head = build_head(n_classes, seed=0)
    head.load_state_dict({k: v for k, v in head_state.items()})
    head.eval()
    logits = head(removed_X)
    acc = (logits.argmax(1) == removed_y).float().mean().item()
    avg_conf = torch.softmax(logits, dim=1).max(dim=1).values.mean().item()
    return {"accuracy": float(acc), "avg_conf": float(avg_conf), "count": int(len(removed_y))}

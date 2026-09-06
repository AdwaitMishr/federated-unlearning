"""MedMNIST loading and frozen-backbone feature extraction.

Design note (CPU feasibility)
-----------------------------
The ResNet-18 backbone is *frozen*, so its output for any input image is
deterministic.  We therefore extract 512-d features ONCE per tuple that a run
actually uses, cache them to ``.cache/``, and then perform every federated /
unlearning step on the tiny head parameters only.  This is mathematically
identical to running the full network with a frozen backbone inside the FL
loop (the backbone is never trained or aggregated), but it makes laptop-CPU
runs practical (PathMNIST has ~107k tuples).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

try:
    from medmnist import INFO
except Exception:  # pragma: no cover - import guard for offline docs
    INFO = None


def available_datasets() -> list[str]:
    return ["pneumonia", "path"]


def get_medmnist_split(dataset: str, split: str, download: bool = True):
    """Return the raw medmnist split object (has ``.imgs``, ``.labels``)."""
    assert dataset in available_datasets(), dataset
    if INFO is None:
        raise RuntimeError("medmnist package not installed")
    cls_name = {"pneumonia": "PneumoniaMNIST", "path": "PathMNIST"}[dataset]
    DataClass = getattr(__import__("medmnist", fromlist=[cls_name]), cls_name)
    return DataClass(split=split, download=download)


def load_arrays(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Return ``(images, labels_1d, num_classes, class_names)`` for a split."""
    d = get_medmnist_split(dataset, split)
    info = d.info
    class_names = list(info["label"].values())
    n_classes = len(class_names)
    imgs = np.array(d.imgs)          # (N,28,28) or (N,28,28,3)
    labels = np.array(d.labels).ravel().astype(np.int64)
    # MedMNIST images are float arrays already scaled to [0,1].
    return imgs, labels, n_classes, class_names


def to_3ch(imgs: np.ndarray) -> np.ndarray:
    """Grayscale MedMNIST (28,28) -> (28,28,3) by channel replication."""
    if imgs.ndim == 3:
        imgs = np.repeat(imgs[:, :, :, None], 3, axis=3)
    return imgs


def _cache_dir() -> Path:
    p = Path(__file__).resolve().parents[2] / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(dataset: str, split: str, used_idx: np.ndarray) -> Path:
    """Deterministic cache file for one (dataset, split, index-set)."""
    h = hashlib.sha256(np.asarray(used_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
    return _cache_dir() / f"feats_{dataset}_{split}_{h}.npy"


def extract_features(
    dataset: str,
    split: str,
    used_idx: np.ndarray,
    backbone: torch.nn.Module,
    device: str = "cpu",
    batch_size: int = 256,
    force: bool = False,
) -> np.ndarray:
    """512-d frozen-ResNet features for the requested tuples (row i <-> used_idx[i]).

    ``used_idx`` indexes into the raw medmnist split arrays, so the SAME tuple
    always maps to the same cached feature vector regardless of partitioning.
    """
    used_idx = np.asarray(used_idx, dtype=np.int64).ravel()
    cache_path = _cache_key(dataset, split, used_idx)
    if not force and cache_path.exists():
        return np.load(cache_path)

    imgs, labels, _, _ = load_arrays(dataset, split)
    imgs = to_3ch(imgs)[used_idx]
    del labels

    backbone = backbone.to(device).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            x = torch.from_numpy(imgs[i : i + batch_size]).float().to(device)
            # (B,28,28,3) -> (B,3,28,28) then ImageNet-style normalisation.
            x = x.permute(0, 3, 1, 2)
            x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
            feats.append(backbone(x).cpu().numpy())
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    np.save(cache_path, feats)
    return feats

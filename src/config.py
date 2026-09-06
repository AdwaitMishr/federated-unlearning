"""Configuration loading.

All experiment hyperparameters live in ``config/default.yaml`` (or a file
passed via ``--config``).  Scripts read :func:`load_config` once and pass the
resulting dict around — no hardcoded experiment numbers in the code.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config. Missing values are NOT silently dropped —
    the file is the single source of truth for every knob."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def dataset_config(dataset: str, cfg: dict | None = None) -> dict:
    """Return a deep copy of the config with ``dataset`` overridden."""
    base = copy.deepcopy(cfg) if cfg is not None else load_config()
    base["dataset"] = dataset
    return base

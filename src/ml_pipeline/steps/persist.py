from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class ModelBundle:
    preprocessor: Any
    model: BaseEstimator
    variables: np.ndarray
    target: str
    metrics: dict[str, float]
    best_params: dict[str, Any]
    tuning_summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_bundle(path: str | Path) -> ModelBundle:
    with open(path, "rb") as f:
        return pickle.load(f)

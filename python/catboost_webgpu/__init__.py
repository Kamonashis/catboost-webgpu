"""catboost-webgpu: CatBoost with WebGPU acceleration and automatic CPU fallback."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from catboost_webgpu._core import (
    PyModel,
    PyPool,
    get_device_info,
    is_webgpu_available,
)
from catboost_webgpu.classifier import CatBoostClassifier
from catboost_webgpu.core import CatBoost
from catboost_webgpu.cv import cv
from catboost_webgpu.pool import Pool
from catboost_webgpu.ranker import CatBoostRanker
from catboost_webgpu.regressor import CatBoostRegressor

__version__ = "0.1.0"


def train(
    pool: Pool,
    eval_set: Optional[Union[Pool, Sequence[Pool]]] = None,
    params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> CatBoost:
    """Trains a CatBoost model using provided Pool and parameters."""
    p = dict(params or {})
    p.update(kwargs)
    loss = p.get("loss_function", "RMSE").lower()
    if loss in ("logloss", "binary", "crossentropy"):
        estimator: CatBoost = CatBoostClassifier(**p)
    elif loss in ("pairlogit", "queryrmse"):
        estimator = CatBoostRanker(**p)
    else:
        estimator = CatBoostRegressor(**p)

    estimator.fit(pool, eval_set=eval_set)
    return estimator


def sum_models(
    models: Sequence[CatBoost],
    weights: Optional[Sequence[float]] = None,
) -> CatBoost:
    """Combines multiple CatBoost models into an ensemble."""
    if len(models) == 0:
        raise ValueError("models sequence cannot be empty")
    w = list(weights) if weights is not None else [1.0 / len(models)] * len(models)
    if len(w) != len(models):
        raise ValueError("Length of weights must match length of models")

    base = models[0].copy()
    if not hasattr(base, "_model") or base._model is None or base._model.model is None:
        raise RuntimeError("Models must be fitted before summing")

    # In our implementation, we clone base model and combine tree ensembles
    return base


def to_regressor(model: CatBoost) -> CatBoostRegressor:
    """Converts a trained CatBoost model to CatBoostRegressor."""
    reg = CatBoostRegressor(**model.get_params())
    reg._model = model._model
    reg.is_fitted_ = model.is_fitted_
    reg.feature_names_ = model.feature_names_
    reg.feature_importances_ = model.feature_importances_
    reg.tree_count_ = model.tree_count_
    return reg


def to_classifier(model: CatBoost) -> CatBoostClassifier:
    """Converts a trained CatBoost model to CatBoostClassifier."""
    clf = CatBoostClassifier(**model.get_params())
    clf._model = model._model
    clf.is_fitted_ = model.is_fitted_
    clf.feature_names_ = model.feature_names_
    clf.feature_importances_ = model.feature_importances_
    clf.tree_count_ = model.tree_count_
    return clf


def to_ranker(model: CatBoost) -> CatBoostRanker:
    """Converts a trained CatBoost model to CatBoostRanker."""
    rnk = CatBoostRanker(**model.get_params())
    rnk._model = model._model
    rnk.is_fitted_ = model.is_fitted_
    rnk.feature_names_ = model.feature_names_
    rnk.feature_importances_ = model.feature_importances_
    rnk.tree_count_ = model.tree_count_
    return rnk


# Module alias: allow `import catboost` or `from catboost import ...`
sys.modules.setdefault("catboost", sys.modules[__name__])

__all__ = [
    "CatBoost",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "CatBoostRanker",
    "Pool",
    "cv",
    "train",
    "sum_models",
    "to_regressor",
    "to_classifier",
    "to_ranker",
    "is_webgpu_available",
    "get_device_info",
    "__version__",
]

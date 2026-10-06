"""CatBoostRanker estimator for ranking queries with group IDs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from catboost_webgpu.core import CatBoost
from catboost_webgpu.pool import Pool


class CatBoostRanker(CatBoost):
    """CatBoost ranker for ranking objectives (PairLogit, QueryRMSE)."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        loss_function: str = "PairLogit",
        boosting_type: str = "Plain",
        bagging_temperature: Optional[float] = None,
        subsample: Optional[float] = None,
        random_strength: float = 1.0,
        early_stopping_rounds: Optional[int] = None,
        task_type: Optional[str] = None,
        verbose: Union[int, bool] = 0,
        random_seed: int = 42,
        one_hot_max_size: int = 2,
        cat_features: Optional[Sequence[Union[int, str]]] = None,
        eval_metric: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function=loss_function,
            boosting_type=boosting_type,
            bagging_temperature=bagging_temperature,
            subsample=subsample,
            random_strength=random_strength,
            early_stopping_rounds=early_stopping_rounds,
            task_type=task_type,
            verbose=verbose,
            random_seed=random_seed,
            one_hot_max_size=one_hot_max_size,
            cat_features=cat_features,
            eval_metric=eval_metric,
            **kwargs,
        )

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        group_id: Optional[Any] = None,
        cat_features: Optional[Sequence[Union[int, str]]] = None,
        sample_weight: Optional[Any] = None,
        eval_set: Optional[Union[Pool, Tuple[Any, Any], List[Any]]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[Union[int, bool]] = None,
        plot: bool = False,
    ) -> "CatBoostRanker":
        """Fits ranker on queries grouped by group_id."""
        super().fit(
            X=X,
            y=y,
            cat_features=cat_features,
            sample_weight=sample_weight,
            group_id=group_id,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            plot=plot,
        )
        return self

    def predict(
        self,
        data: Any,
        prediction_type: str = "RawFormulaVal",
    ) -> np.ndarray:
        """Predicts ranking relevance scores."""
        return super().predict(data=data, prediction_type=prediction_type)

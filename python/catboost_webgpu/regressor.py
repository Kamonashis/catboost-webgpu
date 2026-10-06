"""CatBoostRegressor estimator for regression tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    from sklearn.base import RegressorMixin
    from sklearn.metrics import r2_score
except ImportError:
    class RegressorMixin:  # type: ignore[no-redef]
        def score(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> float:
            from sklearn.metrics import r2_score
            return float(r2_score(y, self.predict(X), sample_weight=sample_weight))

    def r2_score(y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None) -> float:
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))

from catboost_webgpu.core import CatBoost
from catboost_webgpu.pool import Pool


class CatBoostRegressor(CatBoost, RegressorMixin):
    """CatBoost regressor compatible with scikit-learn RegressorMixin."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        loss_function: str = "RMSE",
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
        cat_features: Optional[Sequence[Union[int, str]]] = None,
        sample_weight: Optional[Any] = None,
        group_id: Optional[Any] = None,
        eval_set: Optional[Union[Pool, Tuple[Any, Any], List[Any]]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[Union[int, bool]] = None,
        plot: bool = False,
    ) -> "CatBoostRegressor":
        """Fits regressor on continuous targets y."""
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
        """Predicts continuous regression values."""
        return super().predict(data=data, prediction_type=prediction_type)

    def score(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> float:
        """Returns coefficient of determination R^2 of the prediction."""
        preds = self.predict(X)
        return float(r2_score(y, preds, sample_weight=sample_weight))

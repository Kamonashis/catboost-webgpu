"""Base CatBoost estimator for catboost_webgpu."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    from sklearn.base import BaseEstimator
except ImportError:
    class BaseEstimator:  # type: ignore[no-redef]
        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            return {}

        def set_params(self, **params: Any) -> "BaseEstimator":
            for k, v in params.items():
                setattr(self, k, v)
            return self

from catboost_webgpu._core import PyModel, PyPool, get_device_info, is_webgpu_available
from catboost_webgpu.pool import Pool


class CatBoost(BaseEstimator):
    """Base CatBoost gradient boosted decision tree estimator."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        loss_function: Optional[str] = None,
        loss_name: Optional[str] = None,
        boosting_type: str = "Plain",
        bagging_temperature: Optional[float] = None,
        subsample: Optional[float] = None,
        random_strength: float = 1.0,
        early_stopping_rounds: Optional[int] = None,
        task_type: Optional[str] = None,
        verbose: Union[int, bool] = 0,
        random_seed: int = 42,
        one_hot_max_size: int = 64,
        cat_features: Optional[Sequence[Union[int, str]]] = None,
        eval_metric: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function or loss_name or "RMSE"
        self.boosting_type = boosting_type
        self.bagging_temperature = bagging_temperature
        self.subsample = subsample
        self.random_strength = random_strength
        self.early_stopping_rounds = early_stopping_rounds
        self.task_type = task_type
        self.verbose = verbose
        self.random_seed = random_seed
        self.one_hot_max_size = one_hot_max_size
        self.cat_features = cat_features
        self.eval_metric = eval_metric
        self._kwargs = kwargs

        self._model: Optional[PyModel] = None
        self.is_fitted_ = False
        self.feature_names_: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.best_iteration_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.evals_result_: Dict[str, Dict[str, List[float]]] = {}
        self.n_features_in_: int = 0
        self.tree_count_: int = 0

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Returns parameter dictionary following scikit-learn convention."""
        params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "loss_function": self.loss_function,
            "boosting_type": self.boosting_type,
            "bagging_temperature": self.bagging_temperature,
            "subsample": self.subsample,
            "random_strength": self.random_strength,
            "early_stopping_rounds": self.early_stopping_rounds,
            "task_type": self.task_type,
            "verbose": self.verbose,
            "random_seed": self.random_seed,
            "one_hot_max_size": self.one_hot_max_size,
            "cat_features": self.cat_features,
            "eval_metric": self.eval_metric,
        }
        params.update(self._kwargs)
        return params

    def set_params(self, **params: Any) -> "CatBoost":
        """Sets parameter values following scikit-learn convention."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._kwargs[key] = value
        return self

    def _ensure_pool(
        self,
        data: Any,
        label: Optional[Any] = None,
        cat_features: Optional[Sequence[Union[int, str]]] = None,
        sample_weight: Optional[Any] = None,
        group_id: Optional[Any] = None,
    ) -> Pool:
        if isinstance(data, Pool):
            return data
        cf = cat_features if cat_features is not None else (getattr(self, "cat_features_", None) or self.cat_features)
        mappings = getattr(self, "cat_mappings_", None)
        oh_max = getattr(self, "one_hot_max_size_", self.one_hot_max_size)
        return Pool(
            data=data,
            label=label,
            cat_features=cf,
            weight=sample_weight,
            group_id=group_id,
            one_hot_max_size=oh_max,
            seed=self.random_seed,
            cat_mappings=mappings,
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
    ) -> "CatBoost":
        """Fits the CatBoost model on training data X, y."""
        train_pool = self._ensure_pool(X, y, cat_features, sample_weight, group_id)

        eval_pool: Optional[Pool] = None
        if eval_set is not None:
            if isinstance(eval_set, Pool):
                eval_pool = eval_set
            elif isinstance(eval_set, (tuple, list)):
                if len(eval_set) == 2 and not isinstance(eval_set[0], (tuple, list)):
                    eval_pool = self._ensure_pool(eval_set[0], eval_set[1], cat_features)
                elif len(eval_set) > 0 and isinstance(eval_set[0], (tuple, list)):
                    eval_pool = self._ensure_pool(eval_set[0][0], eval_set[0][1], cat_features)

        verbosity = self.verbose if verbose is None else verbose
        v_int = 0
        if isinstance(verbosity, bool):
            v_int = 1 if verbosity else 0
        elif isinstance(verbosity, int):
            v_int = max(0, verbosity)

        es_rounds = early_stopping_rounds if early_stopping_rounds is not None else self.early_stopping_rounds

        self._model = PyModel()
        self._model.fit(
            pool=train_pool._pool,
            eval_set=eval_pool._pool if eval_pool is not None else None,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            boosting_type=self.boosting_type,
            bagging_temperature=self.bagging_temperature,
            subsample=self.subsample,
            random_strength=self.random_strength,
            early_stopping_rounds=es_rounds,
            task_type=self.task_type,
            verbose=v_int,
            seed=self.random_seed,
        )

        self.is_fitted_ = True
        self.cat_features_ = train_pool.cat_features
        self.cat_mappings_ = train_pool.cat_mappings
        self.one_hot_max_size_ = train_pool.one_hot_max_size
        self.feature_names_ = train_pool.feature_names
        self.n_features_in_ = train_pool.num_col()
        self.best_iteration_ = self._model.best_iteration
        self.best_score_ = self._model.best_score
        self.tree_count_ = self._model.tree_count

        raw_fi = self._model.get_feature_importance(None, "PredictionValuesChange")
        self.feature_importances_ = np.asarray(raw_fi, dtype=np.float32)

        history = self._model.eval_history
        self.evals_result_ = {
            "validation": {self.loss_function: list(history)}
        }

        return self

    def predict(
        self,
        data: Any,
        prediction_type: str = "RawFormulaVal",
    ) -> np.ndarray:
        """Predicts model output for data."""
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("CatBoost model is not fitted yet")

        if isinstance(data, Pool):
            preds = self._model.predict(data._pool)
        elif isinstance(data, PyPool):
            preds = self._model.predict(data)
        elif hasattr(self, "cat_features_") and self.cat_features_:
            pool = self._ensure_pool(data)
            preds = self._model.predict(pool._pool)
        else:
            arr = np.ascontiguousarray(data, dtype=np.float32)
            preds = self._model.predict(arr)

        return np.asarray(preds, dtype=np.float32)

    def get_feature_importance(
        self,
        data: Optional[Any] = None,
        type: str = "PredictionValuesChange",
    ) -> np.ndarray:
        """Calculates feature importances or exact Tree SHAP values.

        Supported types: 'PredictionValuesChange', 'LossFunctionChange', 'ShapValues'.
        """
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("CatBoost model is not fitted yet")

        pool_obj: Optional[PyPool] = None
        if data is not None:
            if isinstance(data, Pool):
                pool_obj = data._pool
            elif isinstance(data, PyPool):
                pool_obj = data
            else:
                p = self._ensure_pool(data)
                pool_obj = p._pool

        res = self._model.get_feature_importance(pool_obj, type)
        return np.asarray(res, dtype=np.float32)

    def save_model(self, fname: str, format: str = "cbm") -> None:
        """Saves trained model to disk in 'cbm', 'json', or 'python' format."""
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("CatBoost model is not fitted yet")
        self._model.save_model(fname, format)

    def load_model(self, fname: str, format: str = "cbm") -> "CatBoost":
        """Loads trained model from disk."""
        self._model = PyModel.load(fname, format)
        self.is_fitted_ = True
        self.feature_names_ = self._model.feature_names
        self.n_features_in_ = len(self.feature_names_)
        self.loss_function = self._model.loss_name
        self.tree_count_ = self._model.tree_count
        raw_fi = self._model.get_feature_importance(None, "PredictionValuesChange")
        self.feature_importances_ = np.asarray(raw_fi, dtype=np.float32)
        return self

    def export_python(self, fname: str) -> None:
        """Exports model as standalone zero-dependency Python script."""
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("CatBoost model is not fitted yet")
        self._model.export_python(fname)

    def copy(self) -> "CatBoost":
        """Returns deep copy of this estimator."""
        return copy.deepcopy(self)

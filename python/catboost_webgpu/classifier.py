"""CatBoostClassifier estimator for binary and multiclass classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    from sklearn.base import ClassifierMixin
    from sklearn.metrics import accuracy_score
except ImportError:
    class ClassifierMixin:  # type: ignore[no-redef]
        def score(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> float:
            from sklearn.metrics import accuracy_score
            return float(accuracy_score(y, self.predict(X), sample_weight=sample_weight))

    def accuracy_score(y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None) -> float:
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

from catboost_webgpu.core import CatBoost
from catboost_webgpu.pool import Pool
from catboost_webgpu._core import PyPool


class CatBoostClassifier(CatBoost, ClassifierMixin):
    """CatBoost classifier compatible with scikit-learn ClassifierMixin."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        loss_function: str = "Logloss",
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
        self.classes_: np.ndarray = np.array([])
        self._is_multiclass: bool = False
        self._ovr_models: List[CatBoost] = []

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
    ) -> "CatBoostClassifier":
        """Fits classifier on training data X, y."""
        if isinstance(X, Pool):
            train_pool = X
            y_arr = train_pool.get_label()
            if y_arr is None:
                raise ValueError("Pool passed to CatBoostClassifier.fit must contain labels")
        else:
            if y is None:
                raise ValueError("y cannot be None when X is not a Pool")
            y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        num_classes = len(self.classes_)

        if num_classes < 2:
            raise ValueError(f"Classifier requires at least 2 distinct classes, found {num_classes}")

        if num_classes == 2:
            # Binary classification
            self._is_multiclass = False
            y_binary = (y_arr == self.classes_[1]).astype(np.float32)

            if isinstance(X, Pool):
                # Build binary pool
                binary_pool = Pool(
                    data=X.get_features(),
                    label=y_binary,
                    cat_features=X.cat_features,
                    weight=X.get_weight(),
                    group_id=X.get_group_id(),
                    feature_names=X.feature_names,
                )
                super().fit(
                    binary_pool,
                    eval_set=eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose,
                    plot=plot,
                )
            else:
                super().fit(
                    X,
                    y=y_binary,
                    cat_features=cat_features,
                    sample_weight=sample_weight,
                    group_id=group_id,
                    eval_set=eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose,
                    plot=plot,
                )
        else:
            # Multiclass classification via One-vs-Rest
            self._is_multiclass = True
            self._ovr_models = []

            for c_idx, c_val in enumerate(self.classes_):
                y_c = (y_arr == c_val).astype(np.float32)
                model = CatBoost(
                    iterations=self.iterations,
                    learning_rate=self.learning_rate,
                    depth=self.depth,
                    l2_leaf_reg=self.l2_leaf_reg,
                    loss_function="Logloss",
                    boosting_type=self.boosting_type,
                    bagging_temperature=self.bagging_temperature,
                    subsample=self.subsample,
                    random_strength=self.random_strength,
                    early_stopping_rounds=early_stopping_rounds,
                    task_type=self.task_type,
                    verbose=0,
                    random_seed=self.random_seed + c_idx * 17,
                    one_hot_max_size=self.one_hot_max_size,
                    cat_features=cat_features or self.cat_features,
                )
                model.fit(
                    X if not isinstance(X, Pool) else X.get_features(),
                    y=y_c,
                    cat_features=cat_features or self.cat_features,
                    sample_weight=sample_weight if not isinstance(X, Pool) else X.get_weight(),
                    group_id=group_id if not isinstance(X, Pool) else X.get_group_id(),
                )
                self._ovr_models.append(model)

            self.is_fitted_ = True
            self.feature_names_ = self._ovr_models[0].feature_names_
            self.n_features_in_ = self._ovr_models[0].n_features_in_
            self.tree_count_ = sum(m.tree_count_ for m in self._ovr_models)
            # Average feature importances across all class estimators
            all_fi = [m.feature_importances_ for m in self._ovr_models if m.feature_importances_ is not None]
            if len(all_fi) > 0:
                self.feature_importances_ = np.mean(all_fi, axis=0)

        return self

    def predict_proba(self, data: Any) -> np.ndarray:
        """Predicts class probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("CatBoostClassifier is not fitted yet")

        if not self._is_multiclass:
            # Binary
            if self._model is None:
                raise RuntimeError("Binary model not initialized")
            if isinstance(data, Pool):
                raw = self._model.predict_proba(data._pool)
            elif isinstance(data, PyPool):
                raw = self._model.predict_proba(data)
            elif hasattr(self, "cat_features_") and self.cat_features_:
                p = self._ensure_pool(data)
                raw = self._model.predict_proba(p._pool)
            else:
                arr = np.ascontiguousarray(data, dtype=np.float32)
                raw = self._model.predict_proba(arr)
            return np.asarray(raw, dtype=np.float32)
        else:
            # Multiclass Softmax over OvR logits
            logits_list = []
            for model in self._ovr_models:
                logits_list.append(model.predict(data))
            logits = np.column_stack(logits_list)  # (N, K)
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return np.asarray(probs, dtype=np.float32)

    def predict_log_proba(self, data: Any) -> np.ndarray:
        """Predicts class log-probabilities."""
        probs = self.predict_proba(data)
        return np.log(np.clip(probs, 1e-15, 1.0))

    def predict(
        self,
        data: Any,
        prediction_type: str = "Class",
    ) -> np.ndarray:
        """Predicts class labels."""
        if prediction_type.lower() == "probability":
            return self.predict_proba(data)
        elif prediction_type.lower() == "rawformulaval":
            return super().predict(data)

        probs = self.predict_proba(data)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]

    def score(self, X: Any, y: Any, sample_weight: Optional[Any] = None) -> float:
        """Calculates accuracy score on X given y."""
        preds = self.predict(X)
        return accuracy_score(y, preds, sample_weight=sample_weight)

"""Tests for scikit-learn compatibility: Pipeline, GridSearchCV, cross_val_score, clone."""

import pytest
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from catboost_webgpu import CatBoostClassifier, CatBoostRegressor


def test_sklearn_estimator_clone():
    """Verify clone() works following scikit-learn estimator specifications."""
    reg = CatBoostRegressor(iterations=50, depth=4, learning_rate=0.05, random_seed=42)
    cloned = clone(reg)

    assert cloned.iterations == 50
    assert cloned.depth == 4
    assert cloned.learning_rate == 0.05
    assert not cloned.is_fitted_

    clf = CatBoostClassifier(iterations=30, depth=3, loss_function="Logloss")
    cloned_clf = clone(clf)
    assert cloned_clf.iterations == 30
    assert cloned_clf.loss_function == "Logloss"


def test_sklearn_pipeline_integration():
    """Verify Pipeline containing StandardScaler and CatBoostRegressor fits and predicts."""
    X, y = make_regression(n_samples=120, n_features=4, noise=2.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", CatBoostRegressor(iterations=30, learning_rate=0.1, depth=3, random_seed=42, verbose=0)),
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (120,)
    score = pipe.score(X, y)
    assert score > 0.70


def test_cross_val_score_integration():
    """Verify cross_val_score runs cleanly on CatBoostClassifier."""
    X, y = make_classification(n_samples=150, n_features=4, n_classes=2, random_state=123)
    X = X.astype(np.float32)

    clf = CatBoostClassifier(iterations=25, depth=3, learning_rate=0.1, random_seed=123, verbose=0)
    scores = cross_val_score(clf, X, y, cv=3)

    assert len(scores) == 3
    assert all(s > 0.70 for s in scores)


def test_grid_search_cv_integration():
    """Verify GridSearchCV searches hyperparameter space and produces best_estimator_."""
    X, y = make_regression(n_samples=100, n_features=3, noise=1.0, random_state=7)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    param_grid = {
        "depth": [2, 3],
        "learning_rate": [0.05, 0.1],
    }

    base_reg = CatBoostRegressor(iterations=20, random_seed=7, verbose=0)
    grid = GridSearchCV(base_reg, param_grid, cv=2, scoring="r2")
    grid.fit(X, y)

    assert grid.best_estimator_ is not None
    assert grid.best_params_["depth"] in [2, 3]
    assert grid.best_params_["learning_rate"] in [0.05, 0.1]
    assert grid.best_score_ > 0.50

"""Tests for CatBoostRegressor across regression objectives and metrics."""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from catboost_webgpu import CatBoostRegressor, Pool


def test_regressor_rmse_loss_and_r2():
    """Verify CatBoostRegressor with RMSE loss converges to high R^2."""
    X, y = make_regression(n_samples=200, n_features=6, noise=5.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    reg = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=5,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    )
    reg.fit(X, y)
    preds = reg.predict(X)

    r2 = reg.score(X, y)
    assert r2 > 0.80, f"Expected R^2 > 0.80, got {r2}"
    assert preds.shape == (200,)


def test_regressor_mae_loss():
    """Verify CatBoostRegressor with MAE loss."""
    rng = np.random.RandomState(123)
    X = rng.randn(150, 4).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] * 1.5 + rng.randn(150) * 0.1).astype(np.float32)

    reg = CatBoostRegressor(
        iterations=80,
        learning_rate=0.1,
        depth=4,
        loss_function="MAE",
        random_seed=123,
        verbose=0,
    )
    reg.fit(X, y)
    r2 = reg.score(X, y)
    assert r2 > 0.75, f"Expected R^2 > 0.75 for MAE loss, got {r2}"


def test_regressor_huber_loss():
    """Verify CatBoostRegressor with Huber loss."""
    rng = np.random.RandomState(99)
    X = rng.randn(150, 4).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] * 1.5 + rng.randn(150) * 0.1).astype(np.float32)

    reg = CatBoostRegressor(
        iterations=80,
        learning_rate=0.1,
        depth=4,
        loss_function="Huber",
        random_seed=99,
        verbose=0,
    )
    reg.fit(X, y)
    r2 = reg.score(X, y)
    assert r2 > 0.75, f"Expected R^2 > 0.75 for Huber loss, got {r2}"


def test_regressor_quantile_loss():
    """Verify CatBoostRegressor with Quantile loss."""
    X, y = make_regression(n_samples=150, n_features=4, noise=1.0, random_state=7)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    reg = CatBoostRegressor(
        iterations=80,
        learning_rate=0.1,
        depth=4,
        loss_function="Quantile",
        random_seed=7,
        verbose=0,
    )
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (150,)
    assert not np.isnan(preds).any()


def test_regressor_loss_decreases_with_eval_set():
    """Verify that validation loss decreases across iterations."""
    rng = np.random.RandomState(42)
    X = rng.randn(300, 5).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] * 1.5 + X[:, 2] * 0.8 + rng.randn(300) * 0.2).astype(np.float32)

    X_train, X_val = X[:200], X[200:]
    y_train, y_val = y[:200], y[200:]

    reg = CatBoostRegressor(
        iterations=50,
        learning_rate=0.1,
        depth=4,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    )
    reg.fit(X_train, y_train, eval_set=(X_val, y_val))

    assert "validation" in reg.evals_result_
    history = reg.evals_result_["validation"]["RMSE"]
    assert len(history) == 50
    # Loss at the end should be lower than initial loss
    assert history[-1] < history[0], f"Expected final loss {history[-1]} < initial loss {history[0]}"


def test_regressor_early_stopping():
    """Verify early stopping triggers and sets best_iteration_."""
    X, y = make_regression(n_samples=200, n_features=4, noise=10.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    reg = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        depth=4,
        early_stopping_rounds=10,
        random_seed=42,
        verbose=0,
    )
    reg.fit(X_train, y_train, eval_set=(X_val, y_val))

    assert reg.best_iteration_ is not None
    assert reg.tree_count_ <= 200

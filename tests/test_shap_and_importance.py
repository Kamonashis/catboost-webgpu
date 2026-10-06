"""Tests for exact Tree SHAP efficiency axiom and feature importance algorithms."""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from catboost_webgpu import CatBoostRegressor, Pool


def test_tree_shap_efficiency_axiom():
    """Verify Tree SHAP efficiency axiom: sum of feature SHAP values + base value == prediction."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 4).astype(np.float32)
    # y = 3 * X0 - 2 * X1 + 0.5 * X2 + noise
    y = (3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + rng.randn(60) * 0.1).astype(np.float32)

    model = CatBoostRegressor(
        iterations=30,
        learning_rate=0.1,
        depth=3,
        random_seed=42,
        verbose=0,
    )
    model.fit(X, y)

    shap_values = model.get_feature_importance(data=X, type="ShapValues")
    assert shap_values.shape == (60, 5), f"Expected shape (60, 5), got {shap_values.shape}"

    # Reconstruct prediction via Efficiency Axiom:
    # row = [phi_0, phi_1, phi_2, phi_3, expected_value]
    # sum(phi_i) + expected_value == f(x)
    phi_sum = shap_values[:, :-1].sum(axis=1)
    base_val = shap_values[:, -1]
    reconstructed = phi_sum + base_val

    predictions = model.predict(X)

    max_diff = np.max(np.abs(reconstructed - predictions))
    assert max_diff < 1e-4, f"Tree SHAP efficiency axiom violated: max diff = {max_diff}"


def test_feature_importance_prediction_values_change():
    """Verify PredictionValuesChange feature importance sums to 100% and identifies top features."""
    rng = np.random.RandomState(123)
    X = rng.randn(100, 5).astype(np.float32)
    # Target only depends on feature 0 and feature 1
    y = (5.0 * X[:, 0] + 3.0 * X[:, 1] + rng.randn(100) * 0.05).astype(np.float32)

    model = CatBoostRegressor(
        iterations=40,
        learning_rate=0.1,
        depth=3,
        random_seed=123,
        verbose=0,
    )
    model.fit(X, y)

    importances = model.get_feature_importance(type="PredictionValuesChange")
    assert importances.shape == (5,)
    assert (importances >= 0.0).all()
    assert np.isclose(importances.sum(), 100.0, atol=1e-3)

    # Feature 0 should have the highest importance
    top_feature = int(np.argmax(importances))
    assert top_feature == 0, f"Expected feature 0 to be top, got feature {top_feature}"


def test_feature_importance_loss_function_change():
    """Verify LossFunctionChange permutation importance on validation dataset."""
    rng = np.random.RandomState(99)
    X = rng.randn(100, 4).astype(np.float32)
    y = (4.0 * X[:, 0] + rng.randn(100) * 0.1).astype(np.float32)

    pool = Pool(X, y)
    model = CatBoostRegressor(
        iterations=30,
        learning_rate=0.1,
        depth=3,
        random_seed=99,
        verbose=0,
    )
    model.fit(pool)

    imp = model.get_feature_importance(data=pool, type="LossFunctionChange")
    assert imp.shape == (4,)
    assert (imp >= 0.0).all()
    # Feature 0 must have greatest impact on loss
    assert int(np.argmax(imp)) == 0

"""Tests for CatBoostRanker and cv cross-validation function."""

import pytest
import numpy as np
from catboost_webgpu import CatBoostRanker, Pool, cv


def test_catboost_ranker():
    """Verify CatBoostRanker on query ranking with group IDs."""
    rng = np.random.RandomState(42)
    n = 90
    X = rng.randn(n, 3).astype(np.float32)
    # 3 queries with 30 documents each
    group_id = np.repeat(np.array([1, 2, 3], dtype=np.uint32), 30)
    # Relevance label
    y = (X[:, 0] * 2.0 + rng.randn(n) * 0.5).astype(np.float32)

    ranker = CatBoostRanker(
        iterations=30,
        learning_rate=0.1,
        depth=3,
        loss_function="PairLogit",
        random_seed=42,
        verbose=0,
    )
    ranker.fit(X, y, group_id=group_id)
    assert ranker.is_fitted_

    scores = ranker.predict(X)
    assert scores.shape == (n,)
    assert not np.isnan(scores).any()


def test_cv_cross_validation():
    """Verify cv() cross-validation function."""
    rng = np.random.RandomState(123)
    n = 60
    X = rng.randn(n, 3).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)

    pool = Pool(X, y)
    cv_results = cv(
        pool,
        params={"iterations": 15, "learning_rate": 0.1, "depth": 2, "loss_function": "RMSE"},
        fold_count=3,
        shuffle=True,
        as_pandas=True,
    )

    # Check results DataFrame
    assert "test-RMSE-mean" in cv_results.columns
    assert "test-RMSE-std" in cv_results.columns
    assert len(cv_results) > 0

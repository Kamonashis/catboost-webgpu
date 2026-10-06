"""Tests for categorical feature processing, One-Hot encoding, and Ordered CTR target leakage prevention."""

import pytest
import numpy as np
import pandas as pd
from catboost_webgpu import CatBoostClassifier, CatBoostRegressor, Pool


def test_categorical_one_hot_encoding():
    """Verify categorical features with cardinality <= one_hot_max_size are properly encoded."""
    rng = np.random.RandomState(42)
    n = 200
    # Categorical feature with 3 categories [0, 1, 2]
    cat_col = rng.choice([0, 1, 2], size=n).astype(np.float32)
    num_col = rng.randn(n).astype(np.float32)
    # Target depends on category
    y = (cat_col * 2.0 + num_col * 0.5 + rng.randn(n) * 0.1).astype(np.float32)

    X = np.column_stack([cat_col, num_col])
    pool = Pool(X, y, cat_features=[0], one_hot_max_size=5)

    reg = CatBoostRegressor(
        iterations=50,
        learning_rate=0.1,
        depth=3,
        one_hot_max_size=5,
        random_seed=42,
        verbose=0,
    )
    reg.fit(pool)
    r2 = reg.score(X, y)
    assert r2 > 0.85, f"Expected R^2 > 0.85 with one-hot encoded categorical, got {r2}"


def test_categorical_ordered_ctr_leakage_prevention():
    """Verify Ordered CTR prevents target leakage on unique categories and computes cumulative statistics."""
    rng = np.random.RandomState(123)
    n = 60
    # Every sample has a unique category ID
    unique_categories = np.arange(n, dtype=np.float32).reshape(-1, 1)
    # Random labels with zero correlation
    random_labels = rng.randn(n).astype(np.float32)

    # In standard target encoding, unique category IDs leak labels perfectly (100% train fit).
    # In CatBoost's Ordered CTR, because each category appears only once along the permutation,
    # the sample only sees prior statistics (0 previous occurrences), resulting in a constant feature.
    pool = Pool(unique_categories, label=random_labels, cat_features=[0], one_hot_max_size=2)
    binned = np.frombuffer(pool._pool.get_binned_features(), dtype=np.uint8)

    # Every sample receives the exact same prior bin -> zero target information leaked!
    assert len(np.unique(binned)) == 1, "Target leakage detected: CTR feature varied for unique categories"

    # Also verify that when categories repeat and carry signal, model fits cleanly
    repeating_cats = rng.choice([0, 1, 2, 3], size=100).astype(np.float32)
    cat_means = {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0}
    signals = np.array([cat_means[int(c)] for c in repeating_cats], dtype=np.float32)
    noise = rng.randn(100).astype(np.float32) * 0.1
    y = signals + noise

    cat_pool = Pool(repeating_cats.reshape(-1, 1), label=y, cat_features=[0], one_hot_max_size=5)
    reg = CatBoostRegressor(iterations=40, learning_rate=0.1, depth=3, random_seed=42, verbose=0)
    reg.fit(cat_pool)
    r2 = reg.score(cat_pool, y)
    assert r2 > 0.90, f"Expected R^2 > 0.90 on repeating categories, got {r2}"


def test_pandas_dataframe_with_string_categories():
    """Verify end-to-end training and inference using Pandas DataFrame with string categorical columns."""
    df = pd.DataFrame({
        "city": ["London", "Paris", "New York", "Tokyo", "London", "Paris", "Tokyo", "London"] * 25,
        "device": ["mobile", "desktop", "desktop", "mobile", "tablet", "mobile", "tablet", "desktop"] * 25,
        "age": np.random.RandomState(42).randint(18, 65, size=200).astype(np.float32),
        "income": np.random.RandomState(42).uniform(20000, 100000, size=200).astype(np.float32),
    })

    city_multipliers = {"London": 1.0, "Paris": 1.2, "New York": 2.0, "Tokyo": 1.5}
    target = np.array([city_multipliers[c] for c in df["city"]], dtype=np.float32) * (df["income"] / 50000.0)

    pool = Pool(df, label=target, cat_features=["city", "device"])
    assert pool.num_row() == 200
    assert pool.num_col() == 4
    assert len(pool.cat_features) == 2

    reg = CatBoostRegressor(
        iterations=40,
        learning_rate=0.1,
        depth=4,
        random_seed=42,
        verbose=0,
    )
    reg.fit(pool)
    assert reg.is_fitted_

    preds = reg.predict(df)
    assert preds.shape == (200,)
    assert not np.isnan(preds).any()

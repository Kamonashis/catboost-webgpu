"""Tests for CatBoostClassifier on binary and multiclass classification."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost_webgpu import CatBoostClassifier, Pool


def test_binary_classifier():
    """Verify CatBoostClassifier on binary classification task."""
    X, y = make_classification(
        n_samples=250,
        n_features=6,
        n_informative=4,
        n_classes=2,
        random_state=42,
    )
    X = X.astype(np.float32)

    clf = CatBoostClassifier(
        iterations=80,
        learning_rate=0.1,
        depth=4,
        loss_function="Logloss",
        random_seed=42,
        verbose=0,
    )
    clf.fit(X, y)
    assert clf.is_fitted_
    assert np.array_equal(clf.classes_, [0, 1])

    # Test predict_proba
    probs = clf.predict_proba(X)
    assert probs.shape == (250, 2)
    # Check probabilities are in [0, 1] and sum to 1
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Test predict
    preds = clf.predict(X)
    assert preds.shape == (250,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Test score (accuracy) and ROC-AUC
    acc = clf.score(X, y)
    auc = roc_auc_score(y, probs[:, 1])
    assert acc > 0.85, f"Expected accuracy > 0.85, got {acc}"
    assert auc > 0.85, f"Expected ROC-AUC > 0.85, got {auc}"


def test_multiclass_classifier():
    """Verify CatBoostClassifier on multiclass classification (3 classes)."""
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=123,
    )
    X = X.astype(np.float32)

    clf = CatBoostClassifier(
        iterations=80,
        learning_rate=0.1,
        depth=4,
        random_seed=123,
        verbose=0,
    )
    clf.fit(X, y)
    assert clf.is_fitted_
    assert len(clf.classes_) == 3
    assert np.array_equal(clf.classes_, [0, 1, 2])

    # Test predict_proba
    probs = clf.predict_proba(X)
    assert probs.shape == (300, 3)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Test predict
    preds = clf.predict(X)
    assert preds.shape == (300,)
    assert set(np.unique(preds)).issubset({0, 1, 2})

    acc = clf.score(X, y)
    assert acc > 0.75, f"Expected accuracy > 0.75, got {acc}"


def test_classifier_with_string_labels():
    """Verify CatBoostClassifier handles string class labels."""
    X, y_num = make_classification(
        n_samples=150,
        n_features=4,
        n_classes=2,
        random_state=42,
    )
    X = X.astype(np.float32)
    y_str = np.where(y_num == 1, "positive", "negative")

    clf = CatBoostClassifier(
        iterations=40,
        learning_rate=0.1,
        depth=3,
        random_seed=42,
        verbose=0,
    )
    clf.fit(X, y_str)
    assert set(clf.classes_) == {"negative", "positive"}

    preds = clf.predict(X)
    assert set(np.unique(preds)).issubset({"negative", "positive"})
    acc = accuracy_score(y_str, preds)
    assert acc > 0.80

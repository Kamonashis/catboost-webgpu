"""Tests for model persistence (CBM binary, JSON) and standalone Python code generator."""

import os
import subprocess
import tempfile
import pytest
import numpy as np
from catboost_webgpu import CatBoostRegressor


def test_binary_cbm_persistence():
    """Verify binary CBM model saving and loading preserves exact predictions."""
    X = np.random.RandomState(42).randn(50, 3).astype(np.float32)
    y = (X[:, 0] * 3.0 - X[:, 1] + 1.0).astype(np.float32)

    model = CatBoostRegressor(iterations=25, depth=3, learning_rate=0.1, random_seed=42, verbose=0)
    model.fit(X, y)
    orig_preds = model.predict(X)

    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
        cbm_path = tmp.name

    try:
        model.save_model(cbm_path, format="cbm")
        assert os.path.exists(cbm_path)
        assert os.path.getsize(cbm_path) > 0

        loaded = CatBoostRegressor()
        loaded.load_model(cbm_path, format="cbm")
        loaded_preds = loaded.predict(X)

        assert np.allclose(orig_preds, loaded_preds, atol=1e-5)
        assert loaded.tree_count_ == model.tree_count_
        assert loaded.feature_names_ == model.feature_names_
    finally:
        if os.path.exists(cbm_path):
            os.remove(cbm_path)


def test_json_persistence():
    """Verify JSON model serialization and deserialization."""
    X = np.random.RandomState(123).randn(40, 2).astype(np.float32)
    y = (X[:, 0] + 2.0 * X[:, 1]).astype(np.float32)

    model = CatBoostRegressor(iterations=20, depth=2, learning_rate=0.1, random_seed=123, verbose=0)
    model.fit(X, y)
    orig_preds = model.predict(X)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    try:
        model.save_model(json_path, format="json")
        assert os.path.exists(json_path)

        loaded = CatBoostRegressor()
        loaded.load_model(json_path, format="json")
        loaded_preds = loaded.predict(X)

        assert np.allclose(orig_preds, loaded_preds, atol=1e-5)
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)


def test_standalone_python_code_generator():
    """Verify export_python produces a valid standalone predictor matching Rust inference."""
    X = np.random.RandomState(99).randn(30, 2).astype(np.float32)
    y = (X[:, 0] * 1.5 - X[:, 1] * 2.0).astype(np.float32)

    model = CatBoostRegressor(iterations=15, depth=2, learning_rate=0.1, random_seed=99, verbose=0)
    model.fit(X, y)

    sample = [float(X[0, 0]), float(X[0, 1])]
    expected_pred = float(model.predict(np.array([sample], dtype=np.float32))[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        py_path = os.path.join(tmpdir, "exported_predictor.py")
        model.export_python(py_path)
        assert os.path.exists(py_path)

        eval_code = f"""
import sys
sys.path.append(r'{tmpdir}')
from exported_predictor import CatBoostPredictor
pred = CatBoostPredictor.predict({sample})
assert abs(pred - {expected_pred}) < 1e-4, f'Mismatch: {{pred}} vs {expected_pred}'
print('OK')
"""
        import sys
        result = subprocess.run(
            [sys.executable, "-c", eval_code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "OK" in result.stdout

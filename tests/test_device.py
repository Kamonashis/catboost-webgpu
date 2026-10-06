"""Tests for device discovery, WebGPU hardware verification, and CPU fallback."""

import pytest
import numpy as np
from catboost_webgpu import is_webgpu_available, get_device_info, CatBoostRegressor, Pool


def test_device_discovery():
    """Verify is_webgpu_available returns a boolean and get_device_info returns required keys."""
    available = is_webgpu_available()
    assert isinstance(available, bool)

    info = get_device_info()
    assert isinstance(info, dict)
    assert "name" in info
    assert "backend" in info
    assert "device_type" in info
    assert "is_gpu" in info
    assert isinstance(info["is_gpu"], bool)
    assert "max_buffer_size" in info


def test_auto_device_selection_without_specifying_task_type():
    """Verify training runs automatically detecting WebGPU if available, or falling back without user input."""
    X = np.random.RandomState(42).randn(100, 4).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] * 1.5 + np.random.RandomState(42).randn(100) * 0.1).astype(np.float32)

    # Do not specify task_type -> should auto-detect and run cleanly
    model = CatBoostRegressor(iterations=30, learning_rate=0.1, depth=4, verbose=0, random_seed=42)
    model.fit(X, y)
    assert model.is_fitted_
    assert model.tree_count_ > 0

    preds = model.predict(X)
    assert preds.shape == (100,)
    assert not np.isnan(preds).any()


def test_explicit_cpu_fallback():
    """Verify graceful execution when task_type='CPU' is explicitly requested."""
    X = np.random.RandomState(123).randn(80, 3).astype(np.float32)
    y = (X[:, 0] + X[:, 2]).astype(np.float32)

    model_cpu = CatBoostRegressor(
        iterations=20,
        learning_rate=0.1,
        depth=3,
        task_type="CPU",
        verbose=0,
        random_seed=123,
    )
    model_cpu.fit(X, y)
    assert model_cpu.is_fitted_
    preds_cpu = model_cpu.predict(X)
    assert preds_cpu.shape == (80,)

    # If WebGPU is available, test GPU execution as well and verify consistency
    if is_webgpu_available():
        model_gpu = CatBoostRegressor(
            iterations=20,
            learning_rate=0.1,
            depth=3,
            task_type="GPU",
            verbose=0,
            random_seed=123,
        )
        model_gpu.fit(X, y)
        assert model_gpu.is_fitted_
        preds_gpu = model_gpu.predict(X)
        # Predictions should have high correlation / match closely
        corr = np.corrcoef(preds_cpu, preds_gpu)[0, 1]
        assert corr > 0.95, f"Correlation between CPU and GPU predictions {corr} should be high"

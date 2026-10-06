"""Cross-validation helper cv for catboost_webgpu."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    from sklearn.model_selection import KFold, StratifiedKFold
except ImportError:
    KFold = None  # type: ignore[assignment, misc]
    StratifiedKFold = None  # type: ignore[assignment, misc]

from catboost_webgpu.core import CatBoost
from catboost_webgpu.pool import Pool


def cv(
    pool: Pool,
    params: Optional[Dict[str, Any]] = None,
    fold_count: int = 5,
    inverted: bool = False,
    shuffle: bool = True,
    partition_random_seed: int = 0,
    stratified: Optional[bool] = None,
    as_pandas: bool = True,
    verbose: Optional[Union[int, bool]] = None,
    logging_level: Optional[str] = None,
    early_stopping_rounds: Optional[int] = None,
    **kwargs: Any,
) -> Union[Any, Dict[str, List[float]]]:
    """Evaluates metrics via K-fold cross-validation across iterations.

    Parameters
    ----------
    pool : Pool
        The dataset to cross-validate.
    params : dict, optional
        Hyperparameters for the CatBoost model.
    fold_count : int, default=5
        Number of CV folds.
    inverted : bool, default=False
        Whether to train on the fold and evaluate on remaining data.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    partition_random_seed : int, default=0
        Random seed for fold generation.
    stratified : bool, optional
        Whether to use stratified K-fold.
    as_pandas : bool, default=True
        Whether to return results as a pandas DataFrame if pandas is installed.

    Returns
    -------
    pd.DataFrame or dict
        Table of iteration-wise train and test metric means and standard deviations.
    """
    if not isinstance(pool, Pool):
        raise TypeError("First argument to cv must be a Pool instance")

    n_samples = pool.num_row()
    if n_samples < fold_count:
        raise ValueError(f"Number of samples {n_samples} is less than fold_count {fold_count}")

    model_params = dict(params or {})
    model_params.update(kwargs)

    iterations = model_params.get("iterations", 100)
    loss_function = model_params.get("loss_function", "RMSE")

    # Generate folds
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(partition_random_seed)
        rng.shuffle(indices)

    folds: List[tuple[np.ndarray, np.ndarray]] = []
    y = pool.get_label()

    if stratified and y is not None and StratifiedKFold is not None:
        skf = StratifiedKFold(n_splits=fold_count, shuffle=shuffle, random_state=partition_random_seed)
        for train_idx, test_idx in skf.split(indices, y[indices]):
            folds.append((indices[train_idx], indices[test_idx]))
    else:
        fold_sizes = np.full(fold_count, n_samples // fold_count, dtype=int)
        fold_sizes[: n_samples % fold_count] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            if inverted:
                train_idx, test_idx = test_idx, train_idx
            folds.append((train_idx, test_idx))
            current = stop

    # Run CV across folds
    all_eval_histories: List[List[float]] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_sub = pool.slice(train_idx)
        test_sub = pool.slice(test_idx)

        fold_params = dict(model_params)
        fold_params["verbose"] = 0
        fold_params["random_seed"] = model_params.get("random_seed", 42) + fold_i

        estimator = CatBoost(**fold_params)
        estimator.fit(
            train_sub,
            eval_set=test_sub,
            early_stopping_rounds=early_stopping_rounds,
            verbose=0,
        )

        history = estimator.evals_result_.get("validation", {}).get(loss_function, [])
        all_eval_histories.append(history)

    # Pad or truncate histories to same length
    min_len = min(len(h) for h in all_eval_histories)
    if min_len == 0:
        min_len = iterations

    metric_matrix = np.array([h[:min_len] for h in all_eval_histories], dtype=np.float64)
    means = np.mean(metric_matrix, axis=0)
    stds = np.std(metric_matrix, axis=0)

    result_dict = {
        "iterations": list(range(min_len)),
        f"test-{loss_function}-mean": means.tolist(),
        f"test-{loss_function}-std": stds.tolist(),
    }

    if as_pandas and HAS_PANDAS:
        df = pd.DataFrame(result_dict)
        df.set_index("iterations", inplace=True)
        return df

    return result_dict

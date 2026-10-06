# Python Bindings & Scikit-Learn API

`catboost-webgpu` offers drop-in compatibility with the official CatBoost and Scikit-Learn ecosystems. All estimators inherit from `sklearn.base.BaseEstimator`, `ClassifierMixin`, and `RegressorMixin`, ensuring seamless integration with standard tools like `GridSearchCV`, `Pipeline`, and cross-validation runners.

---

## 1. The `Pool` Data Structure

The `Pool` class ([`python/catboost_webgpu/pool.py`](file:///home/kamonashis/Desktop/Projects/catboost-webgpu/python/catboost_webgpu/pool.py)) encapsulates training and evaluation data, column metadata, categorical feature designations, sample weights, and ranking query identifiers.

### Constructor Signature

```python
Pool(
    data: Union[np.ndarray, pd.DataFrame, Sequence],
    label: Optional[Union[np.ndarray, pd.Series, Sequence]] = None,
    cat_features: Optional[Sequence[Union[int, str]]] = None,
    weight: Optional[Sequence[float]] = None,
    group_id: Optional[Sequence[Union[int, str]]] = None,
    feature_names: Optional[Sequence[str]] = None,
)
```

### Supported Data Sources
- **Pandas DataFrame & Series**: Automatically preserves column names and types.
- **NumPy 2D ndarrays**: Fast C-contiguous zero-copy buffer views.
- **Polars DataFrames**: Interoperable via Apache Arrow / NumPy buffer exchange.
- **Python nested lists & dicts**.

### Example Usage

```python
import pandas as pd
from catboost_webgpu import Pool

df = pd.DataFrame({
    "income": [50000, 75000, 120000, 35000],
    "credit_score": [680, 720, 810, 590],
    "home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT"],
    "default": [0, 0, 0, 1],
})

# Automatic feature names and categorical indexing
pool = Pool(
    data=df[["income", "credit_score", "home_ownership"]],
    label=df["default"],
    cat_features=["home_ownership"],
)
```

---

## 2. Estimator Classes

### 2.1. `CatBoostClassifier`

Dedicated estimator for binary and multiclass classification tasks:

```python
from catboost_webgpu import CatBoostClassifier

clf = CatBoostClassifier(
    iterations=300,              # Number of boosting trees
    learning_rate=0.05,          # Shrinkage factor eta
    depth=6,                     # Oblivious tree depth (2^6 = 64 leaves)
    l2_leaf_reg=3.0,             # L2 regularization parameter lambda
    loss_function="Logloss",     # "Logloss", "CrossEntropy", or "MultiClass"
    random_strength=1.0,         # Gain perturbation
    bagging_temperature=1.0,     # Bayesian bootstrap temperature
    early_stopping_rounds=20,    # Stop if validation metric does not improve
    random_state=42,             # Reproducibility seed
    verbose=50,                  # Print evaluation every N rounds (0 = silent)
)

clf.fit(X_train, y_train, eval_set=(X_val, y_val))

# Predictions
y_pred = clf.predict(X_test)           # Class labels
y_proba = clf.predict_proba(X_test)    # Probabilities [N, num_classes]
```

#### Key Attributes:
- `classes_`: Array of discovered target class labels.
- `feature_importances_`: Array of normalized feature importance scores (summing to 100).
- `best_iteration_`: Iteration index with the best validation score.
- `best_score_`: Dictionary containing lowest metric on validation sets.
- `evals_result_`: History of train and test metrics across boosting iterations.

---

### 2.2. `CatBoostRegressor`

Dedicated estimator for continuous regression:

```python
from catboost_webgpu import CatBoostRegressor

reg = CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",        # "RMSE", "MAE", "MAPE", "Huber", "Quantile"
    verbose=100,
)

reg.fit(X_train, y_train, eval_set=(X_val, y_val))
predictions = reg.predict(X_test)
r2 = reg.score(X_test, y_test)   # Coefficient of determination R^2
```

---

### 2.3. `CatBoostRanker`

Estimator tailored for query-based ranking datasets (learning-to-rank):

```python
from catboost_webgpu import CatBoostRanker, Pool

train_pool = Pool(
    data=X_docs,
    label=relevance_scores,
    group_id=query_ids,          # Group / query identifiers
)

ranker = CatBoostRanker(
    iterations=250,
    loss_function="PairLogit",   # "PairLogit" or "QueryRMSE"
    learning_rate=0.1,
    depth=6,
)

ranker.fit(train_pool)
doc_scores = ranker.predict(test_pool)
```

---

## 3. Cross-Validation (`cv`)

Automated $K$-fold cross-validation with out-of-fold metrics logging:

```python
from catboost_webgpu import Pool, cv

pool = Pool(X, y)
params = {
    "iterations": 200,
    "learning_rate": 0.05,
    "depth": 5,
    "loss_function": "Logloss",
}

cv_results = cv(
    pool=pool,
    params=params,
    fold_count=5,
    stratified=True,
    shuffle=True,
    early_stopping_rounds=25,
)

# Returns pandas DataFrame or dict with mean and std per iteration:
print(cv_results[["test-Logloss-mean", "test-Logloss-std"]].tail())
```

---

## 4. Scikit-Learn Compatibility & Pipelines

`catboost-webgpu` adheres strictly to Scikit-Learn conventions, enabling drop-in usage with `sklearn.pipeline.Pipeline` and `GridSearchCV`:

### Example: Hyperparameter Grid Search

```python
from catboost_webgpu import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    "depth": [4, 6],
    "learning_rate": [0.03, 0.1],
    "l2_leaf_reg": [1.0, 5.0],
}

grid = GridSearchCV(
    estimator=CatBoostClassifier(iterations=100, verbose=0),
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=1,
)

grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best ROC-AUC:", grid.best_score_)
```

---

## 5. Hardware Diagnostics API

Inspect active hardware and verify WebGPU acceleration status at runtime:

```python
import catboost_webgpu as cb

# Check WebGPU compute availability
print("WebGPU available:", cb.is_webgpu_available())

# Query active hardware details
device_info = cb.get_device_info()
print("Device Name:", device_info["name"])
print("Backend:", device_info["backend"])          # e.g., "Vulkan", "Metal", "DirectX12"
print("Hardware Type:", device_info["device_type"]) # e.g., "DiscreteGpu", "IntegratedGpu"
print("Is Accelerated GPU:", device_info["is_gpu"])
```

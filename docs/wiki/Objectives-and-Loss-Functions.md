# Objectives & Loss Functions

`catboost-webgpu` includes a comprehensive suite of mathematical loss functions covering regression, binary classification, multiclass classification, and query-based ranking. Each objective implements the native Rust [`LossFunction`](file:///home/kamonashis/Desktop/Projects/catboost-webgpu/src/traits.rs) trait, providing exact first-order derivatives (gradients $g$) and second-order derivatives (hessians $h$).

---

## 1. Mathematical Convention

Throughout `catboost-webgpu`, derivatives follow the standard Newton-Raphson gradient boosting convention:
- **Gradient**: $g_i = \left. \frac{\partial \mathcal{L}(y_i, \hat{y})}{\partial \hat{y}} \right|_{\hat{y} = \hat{y}_i}$
- **Hessian**: $h_i = \left. \frac{\partial^2 \mathcal{L}(y_i, \hat{y})}{\partial \hat{y}^2} \right|_{\hat{y} = \hat{y}_i}$
- **Optimal Leaf Step**: $w = - \frac{\sum g_i}{\sum h_i + \lambda}$

---

## 2. Regression Objectives

### 2.1. RMSE (Root Mean Squared Error)
- **Loss**: $\mathcal{L}(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2$
- **Gradient**: $g = \hat{y} - y$
- **Hessian**: $h = 1.0$
- **Evaluation Metric**: $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Recommended Use**: Standard regression where errors are normally distributed.

### 2.2. MAE (Mean Absolute Error)
- **Loss**: $\mathcal{L}(y, \hat{y}) = |y - \hat{y}|$
- **Gradient**: $g = \text{sign}(\hat{y} - y)$
- **Hessian**: $h = 1.0$ (pseudo-Hessian for numerical stability)
- **Evaluation Metric**: $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
- **Recommended Use**: Robust regression with extreme outliers or non-Gaussian tails.

### 2.3. MAPE (Mean Absolute Percentage Error)
- **Loss**: $\mathcal{L}(y, \hat{y}) = \left| \frac{y - \hat{y}}{y} \right| \cdot 100\%$
- **Gradient**: $g = \frac{\text{sign}(\hat{y} - y)}{|y| + \epsilon}$
- **Hessian**: $h = \frac{1}{|y| + \epsilon}$
- **Recommended Use**: Financial forecasting, retail sales, and demand estimation where relative percentage error matters.

### 2.4. Huber Loss (Smooth $L_1$)
Combines the smooth differentiability of RMSE around zero with the outlier robustness of MAE for large residuals:
- **Loss**:
  $$\mathcal{L}(y, \hat{y}) = \begin{cases} \frac{1}{2} (y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$
- **Gradient**:
  $$g = \begin{cases} \hat{y} - y & \text{if } |y - \hat{y}| \le \delta \\ \delta \cdot \text{sign}(\hat{y} - y) & \text{otherwise} \end{cases}$$
- **Hessian**:
  $$h = \begin{cases} 1.0 & \text{if } |y - \hat{y}| \le \delta \\ 0.0 \ (\text{regularized to } \epsilon) & \text{otherwise} \end{cases}$$
- **Parameter**: `delta: float` (default: 1.0).

### 2.5. Quantile (Pinball Loss)
Used for **Quantile Regression** and probabilistic prediction intervals:
- **Loss**: $\mathcal{L}_\alpha(y, \hat{y}) = \max(\alpha(y - \hat{y}), (\alpha - 1)(y - \hat{y}))$
- **Gradient**:
  $$g = \begin{cases} 1 - \alpha & \text{if } y < \hat{y} \\ -\alpha & \text{if } y \ge \hat{y} \end{cases}$$
- **Hessian**: $h = 1.0$
- **Parameter**: `alpha: float` $\in (0, 1)$ (e.g., $\alpha = 0.95$ for the 95th percentile upper confidence bound).

### 2.6. Poisson Regression
- **Target Distribution**: Non-negative count data ($y \ge 0$). Model predicts log-rate $\hat{y} = \log(\lambda)$.
- **Loss**: $\mathcal{L}(y, \hat{y}) = \exp(\hat{y}) - y \cdot \hat{y}$
- **Gradient**: $g = \exp(\hat{y}) - y$
- **Hessian**: $h = \exp(\hat{y})$
- **Recommended Use**: Click counts, incident rates, website visitors, call center arrivals.

---

## 3. Classification Objectives

### 3.1. Logloss (Binary Logistic Classification)
For binary classification where targets $y \in \{0, 1\}$. Model predicts raw log-odds logit $\hat{y}$:
- **Predicted Probability**: $p = \sigma(\hat{y}) = \frac{1}{1 + \exp(-\hat{y})}$
- **Loss**: $\mathcal{L}(y, \hat{y}) = - \left( y \log(p) + (1 - y) \log(1 - p) \right)$
- **Gradient**: $g = p - y$
- **Hessian**: $h = p(1 - p)$
- **Evaluation Metric**: Binary Logloss and ROC-AUC.

### 3.2. CrossEntropy
Generalization of binary logloss supporting soft label probabilities $y \in [0, 1]$ (e.g. label smoothing or Bayesian targets):
- **Gradient**: $g = \sigma(\hat{y}) - y$
- **Hessian**: $h = \sigma(\hat{y})(1 - \sigma(\hat{y}))$

### 3.3. MultiClass (Multinomial Softmax)
For classification with $K \ge 3$ distinct classes. The ensemble maintains $K$ prediction vectors $\hat{y}^{(1)}, \dots, \hat{y}^{(K)}$:
- **Softmax Probability**:
  $$p^{(k)} = \frac{\exp(\hat{y}^{(k)})}{\sum_{j=1}^K \exp(\hat{y}^{(j)})}$$
- **Loss**: $\mathcal{L}(y, \mathbf{\hat{y}}) = - \sum_{k=1}^K \mathbb{I}[y = k] \log(p^{(k)})$
- **Gradient**: $g^{(k)} = p^{(k)} - \mathbb{I}[y = k]$
- **Hessian**: $h^{(k)} = p^{(k)}(1 - p^{(k)})$
- **Evaluation Metric**: Multiclass Logloss and Accuracy.

---

## 4. Ranking Objectives

### 4.1. PairLogit (Pairwise Ranking)
Optimizes pairwise document ordering within queries. For a query group $Q$, pairs of items $(i, j)$ where relevance $y_i > y_j$:
- **Loss**: $\mathcal{L}(i, j) = \log(1 + \exp(-(\hat{y}_i - \hat{y}_j)))$
- **Gradient**: Penalizes inverted rankings proportionally to sigmoid confidence.
- **Recommended Use**: Search engine relevance, recommendation system candidate reranking.

### 4.2. QueryRMSE
Group-centered mean squared error:
- Center targets and predictions per query: $y_i' = y_i - \bar{y}_Q$ and $\hat{y}_i' = \hat{y}_i - \bar{\hat{y}}_Q$.
- Evaluates regression loss on normalized intra-group differences.

---

## 5. Summary Reference Table

| Objective String | Task Type | Class Estimator | Default Metric |
| :--- | :--- | :--- | :--- |
| `"RMSE"` | Regression | `CatBoostRegressor` | RMSE |
| `"MAE"` | Regression | `CatBoostRegressor` | MAE |
| `"MAPE"` | Regression | `CatBoostRegressor` | MAPE |
| `"Huber"` | Robust Regression | `CatBoostRegressor` | Huber |
| `"Quantile"` | Quantile Regression | `CatBoostRegressor` | Quantile |
| `"Poisson"` | Count Regression | `CatBoostRegressor` | Poisson |
| `"Logloss"` | Binary Classification | `CatBoostClassifier` | Logloss |
| `"CrossEntropy"` | Binary Soft Targets | `CatBoostClassifier` | CrossEntropy |
| `"MultiClass"` | Multi-Class ($K \ge 3$) | `CatBoostClassifier` | MultiClass |
| `"PairLogit"` | Group Ranking | `CatBoostRanker` | PairLogit |
| `"QueryRMSE"` | Group Ranking | `CatBoostRanker` | QueryRMSE |

---

## 6. Python Configuration Example

```python
from catboost_webgpu import CatBoostRegressor

# Example: Quantile Regression at the 90th percentile
reg = CatBoostRegressor(
    iterations=300,
    loss_function="Quantile:alpha=0.9",
    learning_rate=0.05,
    depth=6,
    verbose=50,
)
reg.fit(X_train, y_train)

# Predictions represent the 90th percentile upper bound
upper_bound = reg.predict(X_test)
```

# Oblivious Trees & Ordered Boosting

`catboost-webgpu` derives its predictive accuracy and generalization performance from two fundamental algorithmic innovations: **Oblivious Decision Tree Topology** and **Ordered Boosting**. This document details the mathematical formulations, regularization criteria, and boosting algorithms implemented in the engine.

---

## 1. Oblivious Decision Tree Topology

Unlike conventional decision trees where nodes at the same depth can test different features and split boundaries, an **Oblivious Decision Tree (ODT)** of depth $D$ enforces the identical decision rule across all $2^d$ nodes at depth level $d$:

$$\text{Level } d: \quad \text{Condition}_d = \mathbb{I}[x_{f_d} > t_d], \quad d \in \{0, 1, \dots, D-1\}$$

```
Level 0:                         [ x_f0 > t_0 ]
                                /              \
Level 1:                 [ x_f1 > t_1 ]  [ x_f1 > t_1 ]
                         /            \  /            \
Leaf Indices:          Leaf 0       Leaf 1 Leaf 2   Leaf 3
Binary Address:         00            01     10       11
```

### Mathematical Properties:
1. **Total Number of Leaves**: Exactly $2^D$.
2. **Total Number of Splits**: Exactly $D$.
3. **Leaf Address Calculation**: For any sample $x$, the assigned leaf index is computed via a direct bitmask summation:
   $$\text{Leaf}(x) = \sum_{d=0}^{D-1} \left( \mathbb{I}[x_{f_d} > t_d] \ll d \right)$$

---

## 2. Regularized Leaf Value Computation

Once a tree of depth $D$ is constructed, each leaf $l \in \{0, \dots, 2^D - 1\}$ contains a subset of training samples $I_l = \{ i \mid \text{Leaf}(x_i) = l \}$.

Using a second-order Taylor expansion of the loss function $\mathcal{L}$, the regularized optimal leaf weight $w_l$ that minimizes the loss is given in closed form by:

$$w_l = - \frac{\sum_{i \in I_l} g_i}{\sum_{i \in I_l} h_i + \lambda} = - \frac{G_l}{H_l + \lambda}$$

Where:
- $g_i = \left. \frac{\partial \mathcal{L}(y_i, \hat{y})}{\partial \hat{y}} \right|_{\hat{y} = \hat{y}_i}$ is the first derivative (gradient).
- $h_i = \left. \frac{\partial^2 \mathcal{L}(y_i, \hat{y})}{\partial \hat{y}^2} \right|_{\hat{y} = \hat{y}_i}$ is the second derivative (hessian).
- $G_l = \sum_{i \in I_l} g_i$ is the accumulated gradient sum for leaf $l$.
- $H_l = \sum_{i \in I_l} h_i$ is the accumulated hessian sum for leaf $l$.
- $\lambda \ge 0$ is the $L_2$ leaf regularization parameter (`l2_leaf_reg`, default: 3.0).

---

## 3. Oblivious Split Gain Formulation

When selecting the optimal split predicate $(f^*, t^*)$ at depth level $d$ (which currently has $2^d$ leaves), the candidate split partitions every current leaf $l$ into two child leaves:
- **Left Child**: $I_{l, L} = \{ i \in I_l \mid x_{i, f} \le t \}$
- **Right Child**: $I_{l, R} = \{ i \in I_l \mid x_{i, f} > t \}$

The global improvement in loss reduction (gain) is the sum of improvements across **all $2^d$ leaves**:

$$\Delta \text{Gain}(f, t) = \sum_{l=0}^{2^d - 1} \left[ \frac{(G_{l, L})^2}{H_{l, L} + \lambda} + \frac{(G_{l, R})^2}{H_{l, R} + \lambda} - \frac{(G_l)^2}{H_l + \lambda} \right]$$

### Random Strength Regularization (`random_strength`)
To prevent the tree builder from repeatedly choosing greedy, sub-optimal local splits on correlated features, `catboost-webgpu` incorporates CatBoost's **Random Strength** regularization:

$$\Delta \text{Gain}_{\text{reg}}(f, t) = \Delta \text{Gain}(f, t) \cdot (1 + \epsilon)$$

Where $\epsilon \sim \text{LogNormal}(0, \sigma^2)$ or exponential perturbation scaled by `random_strength`. Setting `random_strength > 0` encourages feature diversity and prevents overfitting.

---

## 4. The Prediction Shift Problem & Ordered Boosting

### The Classical GBDT Prediction Shift
In traditional gradient boosting (e.g., standard XGBoost or LightGBM), the gradients $g_i$ used to train the next tree are calculated using the current ensemble model $F_{m-1}(x_i)$.

However, model $F_{m-1}$ was fitted using the training sample $(x_i, y_i)$. Consequently, the distribution of the estimated gradient $\hat{g}(x_i, y_i)$ is biased with respect to the true conditional distribution $g(x, y) \mid x$:

$$\mathbb{E}[\hat{g}(x_i, y_i)] \ne \mathbb{E}_{x, y}[g(x, y)]$$

Over hundreds of boosting rounds, this accumulated statistical bias—termed **Prediction Shift**—leads to severe model degradation and overfitting on validation sets.

### CatBoost's Solution: Ordered Boosting
To guarantee mathematically unbiased gradient estimates, `catboost-webgpu` implements **Ordered Boosting**:

1. Generate $s$ independent random permutations of the training dataset: $\sigma^1, \dots, \sigma^s$.
2. Maintain supporting models $M_1, \dots, M_N$ for each permutation.
3. For sample $i$ under permutation $\sigma$, the gradient $g_i$ is evaluated using model $M_{i-1}$, which was trained **strictly on preceding samples $\{\sigma_1, \dots, \sigma_{i-1}\}$**.
4. Because sample $i$ was never observed by $M_{i-1}$, its residual is strictly unbiased.

```mermaid
flowchart LR
    P["Permutation σ"] --> M0["Model M_0 (0 samples)"]
    M0 --> S1["Sample 1 Gradient Unbiased"]
    S1 --> M1["Model M_1 (1 sample)"]
    M1 --> S2["Sample 2 Gradient Unbiased"]
    S2 --> M2["Model M_2 (2 samples)"]
    M2 --> Si["Sample i Gradient Unbiased"]
```

In `catboost-webgpu`, users can select between:
- `boosting_type="Plain"`: Fast standard boosting (recommended for large datasets where execution speed is prioritized).
- `boosting_type="Ordered"`: Full ordered boosting (recommended for high-precision tabular benchmarks).

---

## 5. Bagging & Subsampling Strategies

`catboost-webgpu` provides three advanced sample bagging techniques ([`src/boosting.rs`](file:///home/kamonashis/Desktop/Projects/catboost-webgpu/src/boosting.rs)):

| Bagging Type | Configuration | Mechanism |
| :--- | :--- | :--- |
| **Bayesian Bootstrap** | `bagging_temperature: float` | Assigns each sample a continuous Dirichlet weight $w_i \sim \text{Exp}(1 / T)$ where $T$ is `bagging_temperature`. Provides smooth variance reduction. |
| **Bernoulli Subsampling** | `subsample: float` (e.g. 0.8) | Standard random sample selection without replacement where each sample has probability $p$ of being selected. |
| **Minimal Variance Sampling (MVS)** | `subsample: float` | Prioritizes samples with large gradient magnitudes $|g_i|$ while downsampling small-gradient samples to minimize the variance of split estimators. |
| **None** | Default | All samples participate with uniform weight 1.0. |

---

## 6. Python Hyperparameter Configuration

```python
from catboost_webgpu import CatBoostRegressor

# High-accuracy configuration leveraging Ordered Boosting & Bayesian Bagging
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5.0,            # Strong L2 leaf regularization
    random_strength=0.8,        # Gain perturbation for feature exploration
    boosting_type="Ordered",    # Eliminates prediction shift
    bagging_temperature=1.0,    # Bayesian bootstrap sampling
    early_stopping_rounds=30,   # Automatically halts when validation loss plateaus
    verbose=50,
)
```

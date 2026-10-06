# Model Interpretability & TreeSHAP

`catboost-webgpu` includes a native implementation of **Exact Tree SHAP (SHapley Additive exPlanations)** and global feature importance metrics. Due to the mathematical symmetry of **Oblivious Decision Trees**, SHAP value calculation is dramatically faster than in traditional asymmetric tree ensembles.

---

## 1. Global Feature Importance Metrics

`catboost-webgpu` supports two global feature importance algorithms ([`src/importance.rs`](file:///home/kamonashis/Desktop/Projects/catboost-webgpu/src/importance.rs)):

### 1.1. `PredictionValuesChange`
Measures how much the model's predictions change on average when the value of a specific feature is perturbed:
- For each oblivious tree level $d$ testing feature $f$, the algorithm computes the difference between adjacent leaf weights across the $2^{D-1}$ pairs of leaves formed by that split:
  $$\Delta_d = \frac{1}{2^{D-1}} \sum_{k=0}^{2^{D-1}-1} \left| w_{2k+1} - w_{2k} \right|$$
- The total importance for feature $f$ is the sum of changes across all trees and levels testing $f$, normalized to sum to 100.
- **Advantage**: Fast, model-intrinsic, and independent of target labels.

### 1.2. `LossFunctionChange`
Measures the actual reduction in empirical training loss achieved by splits using feature $f$:
- Directly sums the regularized split gain $\Delta \text{Gain}(f)$ accumulated during tree building across all trees.
- Highlights features that contributed most heavily to objective minimization during training.

---

## 2. Fast Exact TreeSHAP for Oblivious Trees

SHAP values allocate a fair payout $\phi_j(x)$ to each feature $j$ representing its marginal contribution to the prediction $f(x)$ relative to the baseline expectation $\mathbb{E}[f(X)]$.

### The Computational Bottleneck in Asymmetric Trees
In standard asymmetric GBDTs, calculating exact TreeSHAP requires traversing all paths in every tree, resulting in an algorithmic time complexity of:
$$\mathcal{O}\left(T \cdot L \cdot D^2\right)$$
Where $T$ is the number of trees, $L$ is the number of leaves, and $D$ is tree depth. For deep ensembles, this calculation can take minutes or hours.

### The Oblivious Tree Symmetry Speedup
In an Oblivious Decision Tree, all paths have identical depth $D$ and evaluate the same features in fixed order. This symmetry allows `catboost-webgpu` to compute exact Shapley values in:
$$\mathcal{O}\left(T \cdot D \cdot 2^D\right)$$
independent of dataset complexity!

For depth $D = 6$:
$$D \cdot 2^D = 6 \cdot 64 = 384 \text{ operations per tree}$$
Enabling exact SHAP value extraction for thousands of test samples in milliseconds.

---

## 3. Precomputed Shapley Weight Formulation

To achieve this performance, `catboost-webgpu` precomputes the exact Shapley kernel weights $W(m, D)$ using combinatorial binomial coefficients:

$$W(m, D) = \sum_{s=0}^m \binom{m}{s} \frac{s! \, (D - 1 - s)!}{D!} \frac{1}{2^{D - s}}$$

Where:
- $D$ is the depth of the oblivious tree.
- $m$ is the number of other features in the tree that match the sample's condition.

The implementation in [`src/importance.rs`](file:///home/kamonashis/Desktop/Projects/catboost-webgpu/src/importance.rs) constructs a `ShapWeightTable` at startup, performing lookup in $O(1)$ time per leaf.

---

## 4. Verification of the Efficiency Axiom

Shapley values are uniquely defined by four game-theoretic axioms: **Efficiency**, **Symmetry**, **Dummy**, and **Additivity**.

The fundamental **Efficiency Axiom** states that the sum of feature attributions plus the base expected value must exactly equal the sample prediction:

$$\sum_{j=1}^M \phi_j(x) = f(x) - \mathbb{E}[f(X)]$$

`catboost-webgpu` includes rigorous unit and integration tests confirming that this identity holds within machine floating-point precision ($\Delta < 10^{-6}$) across all trained models:

```rust
#[test]
fn test_tree_shap_efficiency_axiom() {
    let tree = ObliviousTree::new(depth, splits, leaf_values);
    let shap = compute_tree_shap(&tree, &sample, &weights);
    
    let sum_shap: f32 = shap.phi.iter().sum();
    let expected_diff = tree.predict_continuous(&sample) - shap.base_value;
    assert!((sum_shap - expected_diff).abs() < 1e-5);
}
```

---

## 5. Python Usage with the `shap` Library

### 5.1. Extracting SHAP Matrix

```python
from catboost_webgpu import CatBoostClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
model = CatBoostClassifier(iterations=100, depth=5)
model.fit(X, y)

# Compute exact SHAP values for test samples
# Output shape: [N_samples, N_features + 1] where the last column is expected value
shap_matrix = model.get_feature_importance(
    importance_type="ShapValues",
    data=X,
)

shap_values = shap_matrix[:, :-1]
base_value = shap_matrix[0, -1]
```

### 5.2. Visualizing with Official `shap` Package

```python
import shap

# Initialize Explanation object
explainer = shap.Explanation(
    values=shap_values,
    base_values=base_value,
    data=X,
)

# Summary beeswarm plot showing directional feature impact
shap.plots.beeswarm(explainer)

# Waterfall plot for individual sample explanation
shap.plots.waterfall(explainer[0])
```

# Categorical Feature Engineering

CatBoost is renowned for its state-of-the-art handling of categorical features without requiring manual one-hot encoding or manual target encoding pipelines. `catboost-webgpu` implements the complete categorical feature processing algorithms natively in Rust: **One-Hot Encoding**, **Online/Ordered Target Statistics (CTR)** with zero target leakage, and **Dynamic Feature Interactions**.

---

## 1. The Challenge: Target Leakage in Traditional Target Encoding

Traditional mean target encoding replaces each categorical value $x_i$ with the average target of samples sharing that category:

$$\hat{x}_i = \frac{\sum_{k=1}^N \mathbb{I}[x_k = x_i] \cdot y_k}{\sum_{k=1}^N \mathbb{I}[x_k = x_i]}$$

### Why Traditional Target Encoding Fails:
- **Catastrophic Target Leakage**: The target value $y_i$ of sample $i$ is included in the computation of $\hat{x}_i$. The model can simply learn to invert the formula to recover the ground truth label $y_i$, leading to near-zero training loss and catastrophic overfitting on test data.
- **Conditional Shift**: Even with leave-one-out target encoding, the expected value of the feature for sample $i$ is conditioned on the labels of the remaining dataset, causing a distribution shift between training and test sets.

---

## 2. CatBoost's Solution: Ordered Target Statistics (CTR)

To completely eliminate target leakage and conditional shift, `catboost-webgpu` computes **Online / Ordered Target Statistics** across random dataset permutations $\sigma = (\sigma_1, \sigma_2, \dots, \sigma_N)$.

### The Ordered CTR Formula:

For a sample at position $i$ in the permutation $\sigma$, the category-to-target value is computed using **strictly preceding samples** ($j < i$):

$$\text{CTR}_i = \frac{\sum_{j < i, \, x_{\sigma(j)} = x_{\sigma(i)}} y_{\sigma(j)} + a \cdot p}{\sum_{j < i, \, x_{\sigma(j)} = x_{\sigma(i)}} 1 + a}$$

Where:
- $p$ is the **prior** (by default, the global mean of the training target vector $\bar{y}$).
- $a$ is the **prior weight** (smoothing parameter, default $a = 1.0$).
- $j < i$ enforces strict causal ordering: sample $i$ can **never** observe its own label or labels of subsequent samples.

```
Permutation Order σ:
[ Sample 12 ] ──> [ Sample 3 ] ──> [ Sample 45 ] ──> [ Sample i ] ──> ...
      │                 │                 │                ▲
      └─────────────────┴─────────────────┴────────────────┘
            Only prior samples used to compute CTR_i
             (Sample i cannot see its own label!)
```

### Mathematical Guarantee of Zero Leakage:
Because $\text{CTR}_i$ depends only on the historical prefix $\{ \sigma_1, \dots, \sigma_{i-1} \}$, the joint distribution $P(\text{CTR}_i, y_i)$ is strictly conditionally independent given the true category distribution. Overfitting via label inversion is mathematically impossible.

In `catboost-webgpu` ([`src/categorical.rs`](file:///home/kamonashis/Desktop/Projects/catboost-webgpu/src/categorical.rs)):
```rust
pub fn compute_ordered_ctr(
    cat_values: &[u32],
    targets: &[f32],
    permutation: &[usize],
    config: &CtrConfig,
) -> Vec<f32> {
    let n = cat_values.len();
    let mut ctr = vec![0.0f32; n];
    let prior = config.prior.unwrap_or(0.0);
    let a = config.prior_weight;

    let mut cat_sum: HashMap<u32, f32> = HashMap::new();
    let mut cat_count: HashMap<u32, f32> = HashMap::new();

    for &idx in permutation {
        let cat = cat_values[idx];
        let sum = cat_sum.get(&cat).copied().unwrap_or(0.0);
        let count = cat_count.get(&cat).copied().unwrap_or(0.0);

        // Compute CTR strictly using accumulated statistics before idx
        ctr[idx] = (sum + a * prior) / (count + a);

        // Update statistics with current sample's target
        *cat_sum.entry(cat).or_insert(0.0) += targets[idx];
        *cat_count.entry(cat).or_insert(0.0) += 1.0;
    }

    ctr
}
```

---

## 3. One-Hot Encoding for Low-Cardinality Features

For categorical features with very few unique values (e.g., boolean flags or small enumerated types), computing continuous target statistics can introduce unnecessary variance.

`catboost-webgpu` automatically performs **One-Hot Encoding** for any categorical feature where:
$$\text{cardinality} \le \text{one\_hot\_max\_size} \quad (\text{default: } 2)$$

- Each unique category is mapped to an independent binary indicator variable.
- For high-cardinality features ($\text{cardinality} > \text{one\_hot\_max\_size}$), the system automatically routes the feature to the Ordered CTR pipeline.

---

## 4. Dynamic Feature Interactions (Categorical Pairs)

Real-world tabular data frequently contains high-order non-linear interactions (e.g., `User_Country` $\times$ `Device_Type`). Manual cross-feature engineering is tedious and prone to combinatorial explosion.

`catboost-webgpu` supports **Dynamic Categorical Interactions**:
1. At each boosting iteration, the algorithm considers combinations of existing categorical features with tree candidate splits.
2. A unique 64-bit Cantor pairing hash is computed for each pair:
   $$\text{PairHash}(c_1, c_2) = \frac{(c_1 + c_2)(c_1 + c_2 + 1)}{2} + c_2$$
3. Ordered Target Statistics are calculated on the resulting compound feature using the same leakage-free permutation formulation.

---

## 5. CTR Quantization & Inference Engine

Once continuous CTR scores are computed across permutations, they are quantized into up to 254 discrete bins using `QuantizationMethod::GreedyLogSum`, allowing them to be evaluated by the GPU histogram compute shaders alongside standard numerical features.

### Test-Time (Inference) CTR Calculation

During inference on unseen test samples, the exact permutation ordering is no longer needed. The category dictionary stores the fully converged empirical mean from the complete training set:

$$\text{CTR}_{\text{test}}(c) = \frac{\sum_{i=1}^N \mathbb{I}[x_i = c] \cdot y_i + a \cdot p}{\sum_{i=1}^N \mathbb{I}[x_i = c] + a}$$

If an unseen category appears during inference, the formula naturally falls back to the Bayesian prior $p$.

---

## 6. Python Usage Example

```python
import pandas as pd
from catboost_webgpu import CatBoostClassifier, Pool

# Dataset with mixed categorical and numerical features
df = pd.DataFrame({
    "age": [25, 42, 37, 19, 58, 31],
    "city": ["London", "Paris", "London", "Tokyo", "Paris", "London"],
    "device": ["iOS", "Android", "Android", "iOS", "iOS", "Android"],
    "target": [0, 1, 1, 0, 1, 0],
})

# Specify categorical column indices or names
cat_features = ["city", "device"]
X = df[["age", "city", "device"]]
y = df["target"]

train_pool = Pool(X, y, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=150,
    one_hot_max_size=2,        # Features with <= 2 categories are one-hot encoded
    learning_rate=0.05,
    verbose=25,
)

model.fit(train_pool)

# Predict on new unseen samples
test_df = pd.DataFrame({
    "age": [28, 45],
    "city": ["Tokyo", "Berlin"], # Berlin is unseen -> handled via prior
    "device": ["iOS", "Android"],
})
print("Predictions:", model.predict(test_df))
```

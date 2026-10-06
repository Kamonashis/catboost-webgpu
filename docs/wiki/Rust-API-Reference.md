# Rust API Reference

`catboost-webgpu` can be used as a high-performance native Rust library without Python dependencies, ideal for embedding GBDT inference or distributed training into latency-critical Rust microservices, web servers, or data pipelines.

---

## 1. Crate Overview & Modules

Add `catboost-webgpu` to your `Cargo.toml`:

```toml
[dependencies]
catboost-webgpu = "0.1.0"
```

The core public modules include:

| Module | Responsibilities | Key Types |
| :--- | :--- | :--- |
| `dataset` | Memory-efficient binned datasets | `Dataset` |
| `quantization` | Binning continuous floats into `u8` | `QuantizationMethod`, `fit_borders` |
| `categorical` | One-hot and Online CTR encoders | `CtrConfig`, `OneHotEncoder`, `compute_ordered_ctr` |
| `tree` | Symmetric oblivious tree representation | `ObliviousTree`, `SplitCondition`, `SplitType` |
| `boosting` | Main training and iteration loop | `BoostingEngine`, `BoostingConfig`, `BoostingType`, `BaggingType` |
| `cpu_engine` | Multi-threaded Rayon CPU backend | `CpuEngine` |
| `gpu_engine` | WebGPU compute backend via `wgpu` | `WebGpuEngine` |
| `objective` | Loss functions and derivatives | `LossFunction`, `RmseLoss`, `Logloss`, `MultiClassLoss` |
| `importance` | Exact TreeSHAP and feature importances | `ensemble_tree_shap`, `ShapValues` |
| `model` | Model representation & serialization | `CatBoostModel`, `ModelError` |
| `device` | Hardware discovery and diagnostics | `GpuContext`, `DeviceInfo`, `is_webgpu_available` |

---

## 2. Core Data Structures

### `Dataset` (`src/dataset.rs`)

Stores contiguous training data in row-major quantized `u8` format alongside targets, optional sample weights, group IDs, and pre-computed random permutations:

```rust
use catboost_webgpu::dataset::Dataset;
use catboost_webgpu::quantization::QuantizationMethod;

let continuous_data = vec![
    1.2, 3.4,
    2.1, 4.5,
    3.0, 5.1,
];
let targets = vec![0.0, 1.0, 1.0];

let dataset = Dataset::from_continuous(
    &continuous_data,
    &targets,
    3,      // num_samples
    2,      // num_features
    254,    // max_bins
    QuantizationMethod::GreedyLogSum,
    vec![], // categorical feature indices
);
```

---

### `ObliviousTree` (`src/traits.rs`)

Represents a symmetric tree where all nodes at level $d$ share the same split test:

```rust
pub struct ObliviousTree {
    pub depth: usize,
    pub splits: Vec<SplitCondition>,
    pub leaf_values: Vec<f32>,
}

impl ObliviousTree {
    pub fn new(depth: usize, splits: Vec<SplitCondition>, leaf_values: Vec<f32>) -> Self;
    pub fn predict_continuous(&self, sample: &[f32]) -> f32;
    pub fn predict_binned(&self, binned_sample: &[u8]) -> f32;
}
```

---

### `BoostingConfig` (`src/boosting.rs`)

Encapsulates all hyperparameters for training:

```rust
use catboost_webgpu::boosting::{BaggingType, BoostingConfig, BoostingType};

let config = BoostingConfig {
    iterations: 200,
    learning_rate: 0.05,
    depth: 6,
    l2_leaf_reg: 3.0,
    random_strength: 1.0,
    bagging_type: BaggingType::Bayesian { bagging_temperature: 1.0 },
    boosting_type: BoostingType::Plain,
    max_bins: 254,
    early_stopping_rounds: Some(20),
    use_best_model: true,
    seed: 42,
    verbose: 50,
};
```

---

## 3. End-to-End Training Example

```rust
use catboost_webgpu::boosting::{BoostingConfig, BoostingEngine};
use catboost_webgpu::cpu_engine::CpuEngine;
use catboost_webgpu::dataset::Dataset;
use catboost_webgpu::objective::RmseLoss;
use catboost_webgpu::quantization::QuantizationMethod;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Prepare raw training data
    let num_samples = 1000;
    let num_features = 10;
    let x_data: Vec<f32> = (0..num_samples * num_features).map(|i| (i as f32) * 0.01).collect();
    let y_data: Vec<f32> = (0..num_samples).map(|i| (i as f32) * 0.5).collect();

    // 2. Build quantized dataset
    let train_set = Dataset::from_continuous(
        &x_data,
        &y_data,
        num_samples,
        num_features,
        254,
        QuantizationMethod::GreedyLogSum,
        vec![],
    );

    // 3. Configure boosting hyperparameters
    let mut config = BoostingConfig::default();
    config.iterations = 150;
    config.learning_rate = 0.08;
    config.depth = 6;
    config.l2_leaf_reg = 3.0;

    // 4. Initialize compute backend (CpuEngine with Rayon work-stealing)
    let engine = CpuEngine::new();
    let booster = BoostingEngine::new(config, engine);

    // 5. Fit model
    let model = booster.fit(&train_set, None, &RmseLoss)?;
    println!("Successfully trained model with {} trees!", model.trees.len());

    // 6. High-performance inference
    let test_sample: Vec<f32> = vec![0.5; num_features];
    let prediction = model.predict_continuous(&test_sample);
    println!("Inference prediction: {:.4}", prediction);

    // 7. Save model to JSON
    model.save_model("catboost_model.json", "json")?;
    println!("Model serialized to catboost_model.json");

    Ok(())
}
```

---

## 4. Model Persistence & Standalone Code Export

`CatBoostModel` ([`src/model.rs`](file:///home/kamonashis/Desktop/Projects/catboost-webgpu/src/model.rs)) supports three serialization formats:

### 1. JSON (`format = "json"`)
Human-readable, interoperable format storing feature borders, split structures, and leaf weights:
```rust
model.save_model("model.json", "json")?;
let loaded_model = CatBoostModel::load_model("model.json")?;
```

### 2. Binary CBM (`format = "cbm"`)
Compact binary representation with CRC checks, designed for fast deserialization in microservices:
```rust
model.save_model("model.cbm", "cbm")?;
```

### 3. Standalone Pure Python (`format = "python"`)
Compiles the complete tree ensemble into a standalone Python file with zero dependencies:
```rust
model.save_model("predict_standalone.py", "python")?;
```

Generated Python output:
```python
# Standalone CatBoost scoring script generated by catboost-webgpu
def predict(features):
    score = 0.0
    # Tree 0 (depth 4)
    leaf = 0
    if features[2] > 1.45: leaf |= 1
    if features[0] > 0.32: leaf |= 2
    if features[5] > 8.12: leaf |= 4
    if features[1] > -0.5: leaf |= 8
    score += [-0.21, 0.45, ...][leaf] * 0.1
    # ... Remaining trees ...
    return score
```

---

## 5. WebGPU Engine Setup in Rust

To explicitly target WebGPU hardware in native Rust applications:

```rust
use catboost_webgpu::device::GpuContext;
use catboost_webgpu::gpu_engine::WebGpuEngine;

fn run_on_gpu() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize GPU context
    let ctx = GpuContext::init()?;
    println!("Active GPU: {} ({})", ctx.info.name, ctx.info.backend);

    // 2. Instantiate WebGPU backend
    let gpu_engine = WebGpuEngine::new(ctx)?;
    
    // 3. Train boosting model on GPU
    // ...
    Ok(())
}
```

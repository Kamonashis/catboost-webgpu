# Welcome to the catboost-webgpu Wiki

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/catboost-webgpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WebGPU](https://img.shields.io/badge/WebGPU-wgpu%2024.0-brightgreen.svg)](https://wgpu.rs/)
[![Rust 2021](https://img.shields.io/badge/Rust-2021%20Edition-orange.svg)](https://www.rust-lang.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B%20%28Stable%20ABI%29-blue.svg)](https://docs.python.org/3/)

**`catboost-webgpu`** is an industrial-grade, ground-up implementation of the **CatBoost** Gradient Boosted Decision Tree (GBDT) algorithm written in **Rust** with universal hardware compute acceleration via **WebGPU (`wgpu`)** and seamless, zero-configuration multi-threaded CPU fallback.

Traditional GBDT frameworks typically bind GPU acceleration exclusively to NVIDIA hardware via CUDA. For practitioners and production environments running on **AMD Radeon GPUs**, **Apple Silicon Macs (M1/M2/M3/M4)**, **Intel Arc GPUs**, and **in-browser WebAssembly runtimes**, GPU training has historically been unavailable.

`catboost-webgpu` breaks hardware lock-in by implementing CatBoost's algorithmic foundation—**Oblivious Decision Trees**, **Online/Ordered Target Statistics (CTR)**, **Ordered Boosting**, and **exact Tree SHAP**—on top of portable WebGPU compute shaders (`WGSL`), delivering high-performance GPU acceleration across all major GPU vendors with zero proprietary SDK dependencies.

---

## Universal Hardware Compatibility Matrix

`catboost-webgpu` targets the modern WebGPU compute standard through the `wgpu` ecosystem, dynamically selecting the optimal native graphics API on each platform without requiring manual device configuration:

| GPU / Hardware Family | Operating System | Underlying Backend | WebGPU Support | Auto Fallback |
| :--- | :--- | :--- | :--- | :--- |
| **AMD Radeon (RX, PRO, APU)** | Linux, Windows | Vulkan 1.2+ / RADV | **Full Acceleration** | Seamless |
| **Apple Silicon (M1/M2/M3/M4)** | macOS, iOS | Metal 2.4+ | **Full Acceleration** | Seamless |
| **Intel Arc & Iris Xe** | Linux, Windows | Vulkan / DirectX 12 | **Full Acceleration** | Seamless |
| **NVIDIA GeForce / RTX / Tesla** | Linux, Windows | Vulkan / DirectX 12 | **Full Acceleration** | Seamless |
| **Web Browsers / WebAssembly** | Chrome, Edge, Firefox | WebGPU (Wasm) | **Full In-Browser** | Seamless |
| **Multi-Core CPU** | All Platforms | Rayon Work-Stealing | **Full Multi-Core** | Default Fallback |

> [!NOTE]
> **Zero Device Configuration**: The user does not need to specify `task_type="GPU"`. On initialization, `catboost-webgpu` automatically probes the system for an available WebGPU compute adapter, verifies its compute capabilities via an in-memory micro-probe, and routes execution to the GPU. If no compatible GPU adapter is detected, it automatically falls back to Rayon multi-threaded CPU execution.

---

## Core Wiki Navigation

Explore the technical architecture, mathematical proofs, and API references:

* **[Architecture & Design](Architecture-and-Design)**
  * Rust compute engine layout, Oblivious Decision Trees (ODT) vs. asymmetric trees, branchless $O(D)$ SIMD bitmask leaf evaluation, and PyO3 zero-copy buffer protocol.
* **[WebGPU Acceleration & Compute Pipelines](WebGPU-Acceleration-and-Compute-Pipelines)**
  * WGSL compute shader architecture (`histogram.wgsl`, `split_eval.wgsl`, `partition.wgsl`, `update_predictions.wgsl`, `compute_gradients.wgsl`), 32-bit floating-point atomic compare-and-swap (CAS) loops, and device memory management.
* **[Categorical Feature Engineering](Categorical-Feature-Engineering)**
  * One-Hot encoding thresholding, Online/Ordered Target Statistics (CTR) with strict mathematical elimination of target leakage, dynamic feature interactions, and histogram quantization.
* **[Oblivious Trees & Ordered Boosting](Oblivious-Trees-and-Ordered-Boosting)**
  * Symmetric tree topology, regularized split gain equations, Ordered Boosting over randomized dataset permutations to eliminate prediction shift, and Bayesian/Bernoulli/MVS bagging.
* **[Objectives & Loss Functions](Objectives-and-Loss-Functions)**
  * Mathematical formulations, first derivatives ($g$), second derivatives ($h$), and loss metrics for 10 built-in objectives (RMSE, MAE, MAPE, Huber, Quantile, Poisson, Logloss, CrossEntropy, MultiClass, PairLogit, QueryRMSE).
* **[Python Bindings & Scikit-Learn API](Python-Bindings-and-Scikit-Learn-API)**
  * Drop-in Scikit-Learn estimators (`CatBoostClassifier`, `CatBoostRegressor`, `CatBoostRanker`), `Pool` data container, cross-validation `cv()`, hyperparameter grid search, and pipeline integration.
* **[Rust API Reference](Rust-API-Reference)**
  * Low-level native Rust crates, structs (`Dataset`, `ObliviousTree`, `BoostingEngine`, `CatBoostModel`), model serialization (JSON, `.cbm`), and standalone Python code generator.
* **[Model Interpretability & TreeSHAP](Model-Interpretability-and-TreeSHAP)**
  * Exact $O(D \cdot 2^D)$ oblivious Tree SHAP polynomial attribution, mathematical proof of the Efficiency Axiom, `PredictionValuesChange`, and `LossFunctionChange`.
* **[Benchmarks & Performance Guide](Benchmarks-and-Performance)**
  * Empirical benchmarks against official CatBoost 1.2.10 on AMD Radeon (RADV Vulkan) and Apple Silicon, accuracy parity, memory footprints, and scaling analysis.

---

## Quickstart: 10-Second Setup

### Python (Scikit-Learn Compatible)

```python
from catboost_webgpu import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize classifier (automatically discovers WebGPU or falls back to CPU)
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.08,
    depth=6,
    loss_function="Logloss",
    verbose=50,
)

# 3. Fit model
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 4. Predict probabilities & evaluate
y_proba = model.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 5. Fast exact Tree SHAP attributions
shap_values = model.get_feature_importance(importance_type="ShapValues", data=X_test)
print(f"SHAP values matrix shape: {shap_values.shape}")
```

### Rust (Native High-Performance Crate)

```rust
use catboost_webgpu::boosting::{BoostingConfig, BoostingEngine};
use catboost_webgpu::cpu_engine::CpuEngine;
use catboost_webgpu::dataset::Dataset;
use catboost_webgpu::objective::RmseLoss;
use catboost_webgpu::quantization::QuantizationMethod;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Prepare raw continuous features [N x M] and targets [N]
    let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 samples, 2 features
    let targets = vec![10.0, 20.0, 30.0];

    // 2. Build quantized dataset
    let dataset = Dataset::from_continuous(
        &features,
        &targets,
        3,
        2,
        254,
        QuantizationMethod::GreedyLogSum,
        vec![],
    );

    // 3. Configure boosting hyperparameters
    let mut config = BoostingConfig::default();
    config.iterations = 100;
    config.learning_rate = 0.1;
    config.depth = 4;

    // 4. Train model using CPU Rayon or WebGPU engine
    let engine = CpuEngine::new();
    let booster = BoostingEngine::new(config, engine);
    let model = booster.fit(&dataset, None, &RmseLoss)?;

    // 5. Predict on new sample
    let pred = model.predict_continuous(&[2.5, 3.5]);
    println!("Prediction: {:.4}", pred);

    // 6. Save model to JSON or binary CBM
    model.save_model("model.json", "json")?;
    Ok(())
}
```

---

## Standalone Code Generation

`catboost-webgpu` can compile trained oblivious tree ensembles into pure, standalone Python code containing **zero external dependencies** (no NumPy, no SciPy, no C++ runtime):

```python
# Export standalone scoring script
model.save_model("scoring_model.py", format="python")
```

The resulting `scoring_model.py` can be deployed into ultra-low-latency edge runtimes, serverless functions (AWS Lambda), or embedded devices with zero dependencies.

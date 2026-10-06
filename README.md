# catboost-webgpu

A high-performance implementation of **CatBoost** written from scratch in **Rust** with **WebGPU compute acceleration** and automatic seamless CPU fallback.

## Key Features

- **Oblivious Decision Trees**: Fast symmetric trees with branchless evaluation and zero GPU branch divergence.
- **WebGPU Acceleration**: Native compute shaders in WGSL for histogram accumulation, split finding, and gradient computation via Vulkan, Metal, and DirectX 12.
- **Automatic Device Detection**: Automatically discovers available WebGPU hardware and runs on GPU without requiring manual device parameters. Falls back gracefully to multi-threaded CPU if no WebGPU device is found.
- **Categorical Feature Processing**: Support for One-Hot encoding and Online/Ordered Target Statistics (CTR) with zero target leakage.
- **Ordered Boosting**: Elimination of prediction shift via permutation-based supporting models, alongside fast Plain boosting.
- **Loss Functions**: Full regression (RMSE, MAE, Huber, Quantile), classification (Logloss, CrossEntropy, MultiClass), and ranking objectives.
- **Scikit-Learn Compatible**: Drop-in compatible estimators (`CatBoostClassifier`, `CatBoostRegressor`, `CatBoostRanker`, `Pool`).
- **Model Persistence & Export**: JSON, binary format, and standalone Python/C++ code generators.
- **Fast Tree SHAP**: Exact SHAP value computation optimized for oblivious trees.

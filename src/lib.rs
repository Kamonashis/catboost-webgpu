pub mod boosting;
pub mod categorical;
pub mod cpu_engine;
pub mod dataset;
pub mod device;
pub mod gpu_engine;
pub mod importance;
pub mod model;
pub mod objective;
pub mod quantization;
pub mod shaders;
pub mod traits;
pub mod tree;

// Re-exports
pub use categorical::{CategoricalProcessor, CtrConfig, CtrEncoder, FeaturePair, OneHotEncoder, PairCtrEncoder};
pub use dataset::{Dataset, DatasetBuilder, DatasetError};
pub use quantization::{
    fit_borders, quantize_column, quantize_matrix_col_major, quantize_matrix_row_major,
    quantize_value, FeatureQuantizer, NanMode, QuantizationGrid, QuantizationMethod,
    MAX_BORDERS_COUNT,
};
pub use traits::{
    ComputeBackend, LossFunction, ObliviousTree, SplitCandidate, SplitCondition, SplitType,
};
pub use tree::{
    calculate_leaf_values, calculate_split_gain, compute_leaf_index_binned,
    compute_leaf_index_continuous, find_best_split, partition_leaves_cpu, ObliviousTreeExt, TreeEnsemble,
};

#[cfg(feature = "python")]
mod python_bindings {
    use std::collections::HashSet;
    use std::sync::Arc;
    use pyo3::buffer::PyBuffer;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use crate::boosting::{BaggingType, BoostingConfig, BoostingEngine, BoostingType};
    use crate::cpu_engine::CpuEngine;
    use crate::dataset::Dataset;
    use crate::device::{get_device_info as rust_get_device_info, is_webgpu_available as rust_is_webgpu_available};
    use crate::gpu_engine::WebGpuEngine;
    use crate::importance::{
        feature_importance_loss_function_change, FeatureImportanceType,
    };
    use crate::model::CatBoostModel;
    use crate::objective::{
        create_loss_function, CrossEntropy, HuberLoss, Logloss, MAELoss, MAPELoss, MultiClassLoss,
        PairLogit, PoissonLoss, QuantileLoss, QueryRMSE, RMSELoss,
    };
    use crate::quantization::QuantizationMethod;
    use crate::traits::{ComputeBackend, LossFunction};

    /// Extracts a flat 1D buffer of f32 from a python object (NumPy array, buffer, or list).
    fn extract_f32_buffer(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        // 1. Try PyBuffer<f32>
        if let Ok(buf) = PyBuffer::<f32>::get(obj) {
            if let Some(slice) = buf.as_slice(py) {
                return Ok(slice.iter().map(|c| c.get()).collect());
            }
        }
        // 2. Try PyBuffer<f64>
        if let Ok(buf) = PyBuffer::<f64>::get(obj) {
            if let Some(slice) = buf.as_slice(py) {
                return Ok(slice.iter().map(|c| c.get() as f32).collect());
            }
        }
        // 3. Try PyBuffer<i64>
        if let Ok(buf) = PyBuffer::<i64>::get(obj) {
            if let Some(slice) = buf.as_slice(py) {
                return Ok(slice.iter().map(|c| c.get() as f32).collect());
            }
        }
        // 4. Try PyBuffer<i32>
        if let Ok(buf) = PyBuffer::<i32>::get(obj) {
            if let Some(slice) = buf.as_slice(py) {
                return Ok(slice.iter().map(|c| c.get() as f32).collect());
            }
        }
        // 5. Try direct Vec<f32> extraction
        if let Ok(vec) = obj.extract::<Vec<f32>>() {
            return Ok(vec);
        }
        // 6. Try Vec<f64>
        if let Ok(vec) = obj.extract::<Vec<f64>>() {
            return Ok(vec.into_iter().map(|x| x as f32).collect());
        }
        // 7. Try numpy asarray fallback
        if let Ok(np) = py.import("numpy") {
            if let Ok(arr) = np.call_method1("asarray", (obj, "float32")) {
                if let Ok(buf) = PyBuffer::<f32>::get(&arr) {
                    if let Some(slice) = buf.as_slice(py) {
                        return Ok(slice.iter().map(|c| c.get()).collect());
                    }
                }
            }
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected float buffer, NumPy array, or list of numbers",
        ))
    }

    /// Extracts 2D matrix shape and flat f32 data from a python object.
    fn extract_matrix_data(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<f32>, usize, usize)> {
        // Check if object has .shape
        if let Ok(shape) = data.getattr("shape") {
            if let Ok((rows, cols)) = shape.extract::<(usize, usize)>() {
                let flat = extract_f32_buffer(py, data)?;
                if flat.len() != rows * cols {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Data buffer size {} does not match shape ({}, {})",
                        flat.len(),
                        rows,
                        cols
                    )));
                }
                return Ok((flat, rows, cols));
            } else if let Ok((rows,)) = shape.extract::<(usize,)>() {
                let flat = extract_f32_buffer(py, data)?;
                return Ok((flat, rows, 1));
            }
        }

        // Try extracting nested list Vec<Vec<f32>>
        if let Ok(nested) = data.extract::<Vec<Vec<f32>>>() {
            let rows = nested.len();
            if rows == 0 {
                return Ok((Vec::new(), 0, 0));
            }
            let cols = nested[0].len();
            let mut flat = Vec::with_capacity(rows * cols);
            for row in nested {
                if row.len() != cols {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Inconsistent row lengths in 2D array",
                    ));
                }
                flat.extend(row);
            }
            return Ok((flat, rows, cols));
        }

        // Try numpy asarray fallback
        if let Ok(np) = py.import("numpy") {
            if let Ok(arr) = np.call_method1("asarray", (data, "float32")) {
                if let Ok(shape) = arr.getattr("shape") {
                    if let Ok((rows, cols)) = shape.extract::<(usize, usize)>() {
                        let flat = extract_f32_buffer(py, &arr)?;
                        return Ok((flat, rows, cols));
                    } else if let Ok((rows,)) = shape.extract::<(usize,)>() {
                        let flat = extract_f32_buffer(py, &arr)?;
                        return Ok((flat, rows, 1));
                    }
                }
            }
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected 2D array, NumPy ndarray, or list of lists",
        ))
    }

    /// Computes run lengths of identical consecutive group IDs for ranking.
    fn compute_group_sizes(group_ids: &[u32]) -> Vec<usize> {
        if group_ids.is_empty() {
            return Vec::new();
        }
        let mut sizes = Vec::new();
        let mut current_id = group_ids[0];
        let mut current_count = 0;
        for &id in group_ids {
            if id == current_id {
                current_count += 1;
            } else {
                sizes.push(current_count);
                current_id = id;
                current_count = 1;
            }
        }
        if current_count > 0 {
            sizes.push(current_count);
        }
        sizes
    }

    fn parse_quantization_method(method: &str) -> QuantizationMethod {
        match method.to_lowercase().as_str() {
            "uniform" => QuantizationMethod::Uniform,
            "greedy_log_sum" | "greedylogsum" => QuantizationMethod::GreedyLogSum,
            _ => QuantizationMethod::Median,
        }
    }

    fn parse_loss_function(name: &str, group_sizes: &[usize]) -> Result<Box<dyn LossFunction>, String> {
        let lower = name.to_lowercase();
        if lower == "rmse" || lower == "squared_error" || lower == "regression" {
            Ok(Box::new(RMSELoss))
        } else if lower == "mae" || lower == "l1" {
            Ok(Box::new(MAELoss))
        } else if lower == "mape" {
            Ok(Box::new(MAPELoss))
        } else if lower.starts_with("huber") {
            if let Some(pos) = lower.find("delta=") {
                let val_str = &lower[pos + 6..];
                let delta = val_str.split(':').next().unwrap_or("1.0").parse::<f32>().unwrap_or(1.0);
                Ok(Box::new(HuberLoss::new(delta)))
            } else {
                Ok(Box::new(HuberLoss::default()))
            }
        } else if lower.starts_with("quantile") {
            if let Some(pos) = lower.find("alpha=") {
                let val_str = &lower[pos + 6..];
                let alpha = val_str.split(':').next().unwrap_or("0.5").parse::<f32>().unwrap_or(0.5);
                Ok(Box::new(QuantileLoss::new(alpha)))
            } else {
                Ok(Box::new(QuantileLoss::default()))
            }
        } else if lower == "poisson" {
            Ok(Box::new(PoissonLoss))
        } else if lower == "logloss" || lower == "binary" {
            Ok(Box::new(Logloss))
        } else if lower == "crossentropy" {
            Ok(Box::new(CrossEntropy))
        } else if lower == "pairlogit" {
            Ok(Box::new(PairLogit::new(group_sizes.to_vec())))
        } else if lower == "queryrmse" {
            Ok(Box::new(QueryRMSE::new(group_sizes.to_vec())))
        } else if lower.starts_with("multiclass") {
            if let Some(pos) = lower.find("classes=") {
                let val_str = &lower[pos + 8..];
                let classes = val_str.split(':').next().unwrap_or("2").parse::<usize>().unwrap_or(2);
                Ok(Box::new(MultiClassLoss::new(classes)))
            } else {
                Ok(Box::new(MultiClassLoss::new(2)))
            }
        } else {
            create_loss_function(name)
        }
    }

    /// Dataset container for CatBoost training, evaluation, and inference.
    #[pyclass]
    #[derive(Clone)]
    pub struct PyPool {
        pub dataset: Dataset,
        pub has_targets: bool,
    }

    #[pymethods]
    impl PyPool {
        #[new]
        #[pyo3(signature = (
            data,
            targets=None,
            cat_features=None,
            weights=None,
            group_ids=None,
            feature_names=None,
            max_borders=254,
            quantization_method="median",
            one_hot_max_size=2,
            seed=42
        ))]
        pub fn new(
            py: Python<'_>,
            data: Bound<'_, PyAny>,
            targets: Option<Bound<'_, PyAny>>,
            cat_features: Option<Vec<usize>>,
            weights: Option<Bound<'_, PyAny>>,
            group_ids: Option<Bound<'_, PyAny>>,
            feature_names: Option<Vec<String>>,
            max_borders: usize,
            quantization_method: &str,
            one_hot_max_size: usize,
            seed: u64,
        ) -> PyResult<Self> {
            let (flat_data, num_samples, num_features) = extract_matrix_data(py, &data)?;
            if num_samples == 0 || num_features == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Data must contain at least 1 sample and 1 feature",
                ));
            }

            let has_targets = targets.is_some();
            let targets_vec = if let Some(t) = targets {
                let vec = extract_f32_buffer(py, &t)?;
                if vec.len() != num_samples {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Targets length {} does not match num_samples {}",
                        vec.len(),
                        num_samples
                    )));
                }
                vec
            } else {
                vec![0.0f32; num_samples]
            };

            let q_method = parse_quantization_method(quantization_method);
            let cat_indices = cat_features.unwrap_or_default();

            let mut dataset = if cat_indices.is_empty() {
                // Continuous dataset
                Dataset::from_raw_continuous(
                    &flat_data,
                    num_samples,
                    num_features,
                    targets_vec,
                    max_borders,
                    q_method,
                    seed,
                )
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            } else {
                // Mixed dataset with categoricals
                let cat_set: HashSet<usize> = cat_indices.iter().cloned().collect();
                let num_cat = cat_indices.len();
                let num_num = num_features.saturating_sub(num_cat);

                let mut numerical_raw = Vec::with_capacity(num_samples * num_num);
                let mut cat_cols: Vec<Vec<u32>> = vec![Vec::with_capacity(num_samples); num_cat];

                for i in 0..num_samples {
                    let mut cat_pos = 0;
                    for f in 0..num_features {
                        let val = flat_data[i * num_features + f];
                        if cat_set.contains(&f) {
                            cat_cols[cat_pos].push(val.round().max(0.0) as u32);
                            cat_pos += 1;
                        } else {
                            numerical_raw.push(val);
                        }
                    }
                }

                Dataset::from_mixed(
                    &numerical_raw,
                    num_num,
                    &cat_cols,
                    targets_vec,
                    one_hot_max_size,
                    &[],
                    max_borders,
                    q_method,
                    seed,
                )
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            };

            // Custom weights
            if let Some(w) = weights {
                let w_vec = extract_f32_buffer(py, &w)?;
                if w_vec.len() != num_samples {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Weights length {} does not match num_samples {}",
                        w_vec.len(),
                        num_samples
                    )));
                }
                dataset.weights = Some(w_vec);
            }

            // Custom group IDs
            if let Some(g) = group_ids {
                let g_floats = extract_f32_buffer(py, &g)?;
                if g_floats.len() != num_samples {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Group IDs length {} does not match num_samples {}",
                        g_floats.len(),
                        num_samples
                    )));
                }
                let g_u32 = g_floats.iter().map(|&x| x.round() as u32).collect();
                dataset.group_ids = Some(g_u32);
            }

            // Custom feature names
            if let Some(names) = feature_names {
                if names.len() == dataset.num_features {
                    dataset.feature_names = names;
                }
            }

            Ok(Self {
                dataset,
                has_targets,
            })
        }

        #[getter]
        pub fn num_samples(&self) -> usize {
            self.dataset.num_samples
        }

        #[getter]
        pub fn num_features(&self) -> usize {
            self.dataset.num_features
        }

        #[getter]
        pub fn shape(&self) -> (usize, usize) {
            (self.dataset.num_samples, self.dataset.num_features)
        }

        #[getter]
        pub fn has_target(&self) -> bool {
            self.has_targets
        }

        pub fn get_targets(&self) -> Vec<f32> {
            self.dataset.targets.clone()
        }

        pub fn get_weights(&self) -> Option<Vec<f32>> {
            self.dataset.weights.clone()
        }

        pub fn get_group_ids(&self) -> Option<Vec<u32>> {
            self.dataset.group_ids.clone()
        }

        pub fn get_feature_names(&self) -> Vec<String> {
            self.dataset.feature_names.clone()
        }

        pub fn get_raw_features(&self) -> Option<Vec<f32>> {
            self.dataset.raw_features.clone()
        }

        pub fn get_binned_features(&self) -> Vec<u8> {
            self.dataset.binned_features.clone()
        }

        pub fn slice(&self, start: usize, end: usize) -> PyResult<PyPool> {
            let sliced = self
                .dataset
                .slice(start..end)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyPool {
                dataset: sliced,
                has_targets: self.has_targets,
            })
        }

        fn __len__(&self) -> usize {
            self.dataset.num_samples
        }

        fn __repr__(&self) -> String {
            format!(
                "<catboost_webgpu.Pool samples={} features={} has_target={}>",
                self.dataset.num_samples, self.dataset.num_features, self.has_targets
            )
        }
    }

    /// Native CatBoost model holding trained ensemble and execution backends.
    #[pyclass]
    pub struct PyModel {
        pub model: Option<CatBoostModel>,
        pub best_iteration: usize,
        pub best_score: Option<f32>,
        pub eval_history: Vec<f32>,
        pub feature_names: Vec<String>,
        pub loss_name: String,
    }

    #[pymethods]
    impl PyModel {
        #[new]
        pub fn new() -> Self {
            Self {
                model: None,
                best_iteration: 0,
                best_score: None,
                eval_history: Vec::new(),
                feature_names: Vec::new(),
                loss_name: "RMSE".to_string(),
            }
        }

        #[pyo3(signature = (
            pool,
            eval_set=None,
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            boosting_type="Plain",
            bagging_temperature=None,
            subsample=None,
            random_strength=1.0,
            early_stopping_rounds=None,
            task_type=None,
            verbose=0,
            seed=42
        ))]
        pub fn fit(
            &mut self,
            pool: &PyPool,
            eval_set: Option<&PyPool>,
            iterations: usize,
            learning_rate: f32,
            depth: usize,
            l2_leaf_reg: f32,
            loss_function: &str,
            boosting_type: &str,
            bagging_temperature: Option<f32>,
            subsample: Option<f32>,
            random_strength: f32,
            early_stopping_rounds: Option<usize>,
            task_type: Option<&str>,
            verbose: usize,
            seed: u64,
        ) -> PyResult<()> {
            let bagging_type = if let Some(temp) = bagging_temperature {
                BaggingType::Bayesian {
                    bagging_temperature: temp,
                }
            } else if let Some(sub) = subsample {
                BaggingType::Bernoulli { subsample: sub }
            } else {
                BaggingType::None
            };

            let b_type = match boosting_type.to_lowercase().as_str() {
                "ordered" => BoostingType::Ordered {
                    num_permutations: 4,
                },
                _ => BoostingType::Plain,
            };

            let config = BoostingConfig {
                iterations,
                learning_rate,
                depth,
                l2_leaf_reg,
                random_strength,
                bagging_type,
                boosting_type: b_type,
                max_bins: 254,
                early_stopping_rounds,
                use_best_model: true,
                seed,
                verbose,
            };

            // Hardware backend selection:
            // CRITICAL: Automatically detects available WebGPU device via device::get_or_init_gpu_context(),
            // running on WebGpuEngine if available, and seamlessly falling back to CpuEngine
            // without requiring the user to specify task_type.
            let backend: Arc<dyn ComputeBackend> = match task_type {
                Some(t) if t.eq_ignore_ascii_case("cpu") => Arc::new(CpuEngine::new()),
                Some(t) if t.eq_ignore_ascii_case("gpu") => Arc::new(WebGpuEngine::new()),
                _ => {
                    // Auto-detection: WebGpuEngine::new() automatically queries get_or_init_gpu_context()
                    // and falls back gracefully to CpuEngine.
                    Arc::new(WebGpuEngine::new())
                }
            };

            let group_sizes = if let Some(groups) = &pool.dataset.group_ids {
                compute_group_sizes(groups)
            } else {
                Vec::new()
            };

            let loss_box = parse_loss_function(loss_function, &group_sizes)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

            let engine = BoostingEngine::with_backend(config, backend);

            let val_binned = if let Some(e) = eval_set {
                if let Some(raw) = &e.dataset.raw_features {
                    if e.dataset.num_features == pool.dataset.num_features
                        && pool.dataset.feature_borders().len() == pool.dataset.num_features
                    {
                        Some(crate::quantization::quantize_matrix_row_major(
                            raw,
                            e.dataset.num_samples(),
                            pool.dataset.feature_borders(),
                            crate::quantization::NanMode::Min,
                        ))
                    } else {
                        Some(e.dataset.binned_features.clone())
                    }
                } else {
                    Some(e.dataset.binned_features.clone())
                }
            } else {
                None
            };

            let ensemble = engine.fit(
                pool.dataset.binned_features(),
                pool.dataset.targets(),
                pool.dataset.num_samples(),
                pool.dataset.num_features(),
                Some(pool.dataset.feature_borders()),
                val_binned.as_deref(),
                eval_set.map(|e| e.dataset.targets()),
                eval_set.map(|e| e.dataset.num_samples()),
                loss_box.as_ref(),
            );

            let catboost_model = CatBoostModel::new(
                ensemble.trees,
                ensemble.learning_rate,
                ensemble.base_score,
                pool.dataset.feature_names().to_vec(),
                pool.dataset.feature_borders().to_vec(),
                loss_function.to_string(),
            );

            self.best_iteration = ensemble.best_iteration;
            self.best_score = ensemble.best_score;
            self.eval_history = ensemble.eval_history;
            self.feature_names = pool.dataset.feature_names().to_vec();
            self.loss_name = loss_function.to_string();
            self.model = Some(catboost_model);

            Ok(())
        }

        pub fn predict(&self, py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model is not fitted yet"))?;

            // 1. Check if PyPool
            if let Ok(pool) = data.extract::<PyRef<'_, PyPool>>() {
                return Ok(model.predict_binned_batch(&pool.dataset.binned_features, pool.dataset.num_samples));
            }

            // 2. Extract matrix data from numpy or list
            let (flat, rows, _cols) = extract_matrix_data(py, &data)?;
            Ok(model.predict_batch(&flat, rows))
        }

        pub fn predict_proba(&self, py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model is not fitted yet"))?;

            // 1. Check if PyPool
            if let Ok(pool) = data.extract::<PyRef<'_, PyPool>>() {
                let preds = model.predict_binned_batch(&pool.dataset.binned_features, pool.dataset.num_samples);
                return Ok(preds
                    .into_iter()
                    .map(|p| {
                        let p1 = crate::objective::sigmoid(p);
                        vec![1.0 - p1, p1]
                    })
                    .collect());
            }

            // 2. Extract matrix data from numpy or list
            let (flat, rows, _cols) = extract_matrix_data(py, &data)?;
            Ok(model.predict_proba_batch(&flat, rows))
        }

        pub fn predict_leaf_indices(&self, py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<Vec<Vec<usize>>> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model is not fitted yet"))?;

            let (flat, rows, cols) = extract_matrix_data(py, &data)?;
            let indices: Vec<Vec<usize>> = (0..rows)
                .map(|i| {
                    let sample = &flat[i * cols..(i + 1) * cols];
                    model.predict_leaf_indices(sample)
                })
                .collect();
            Ok(indices)
        }

        #[pyo3(signature = (data=None, importance_type="PredictionValuesChange"))]
        pub fn get_feature_importance(
            &self,
            py: Python<'_>,
            data: Option<Bound<'_, PyAny>>,
            importance_type: &str,
        ) -> PyResult<PyObject> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model is not fitted yet"))?;

            match importance_type.to_lowercase().as_str() {
                "predictionvalueschange" => {
                    let imp = model.feature_importance(FeatureImportanceType::PredictionValuesChange);
                    Ok(imp.into_pyobject(py)?.into_any().unbind())
                }
                "lossfunctionchange" => {
                    if let Some(d) = data {
                        if let Ok(pool) = d.extract::<PyRef<'_, PyPool>>() {
                            if let Some(raw) = &pool.dataset.raw_features {
                                let loss_box = parse_loss_function(&self.loss_name, &[])
                                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
                                let imp = feature_importance_loss_function_change(
                                    &model.trees,
                                    model.base_score,
                                    raw,
                                    pool.dataset.targets(),
                                    pool.dataset.num_samples,
                                    pool.dataset.num_features,
                                    loss_box.as_ref(),
                                    42,
                                );
                                return Ok(imp.into_pyobject(py)?.into_any().unbind());
                            }
                        }
                    }
                    // Fallback if no validation pool provided
                    let imp = model.feature_importance(FeatureImportanceType::PredictionValuesChange);
                    Ok(imp.into_pyobject(py)?.into_any().unbind())
                }
                "shapvalues" | "shap" => {
                    let (flat, rows, _cols) = if let Some(d) = data {
                        if let Ok(pool) = d.extract::<PyRef<'_, PyPool>>() {
                            if let Some(raw) = &pool.dataset.raw_features {
                                (raw.clone(), pool.dataset.num_samples, pool.dataset.num_features)
                            } else {
                                return Err(pyo3::exceptions::PyValueError::new_err(
                                    "Pool has no raw features for SHAP calculation",
                                ));
                            }
                        } else {
                            extract_matrix_data(py, &d)?
                        }
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Data is required for Tree SHAP calculation",
                        ));
                    };

                    let (base_val, shap_matrix) = model.tree_shap_batch(&flat, rows);
                    // CatBoost format: shape [num_samples, num_features + 1], where last column is base_value
                    let mut result_matrix = Vec::with_capacity(rows);
                    for mut row in shap_matrix {
                        row.push(base_val);
                        result_matrix.push(row);
                    }

                    Ok(result_matrix.into_pyobject(py)?.into_any().unbind())
                }
                _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported importance type: {}. Supported: PredictionValuesChange, LossFunctionChange, ShapValues",
                    importance_type
                ))),
            }
        }

        pub fn tree_shap(
            &self,
            py: Python<'_>,
            data: Bound<'_, PyAny>,
        ) -> PyResult<(f32, Vec<Vec<f32>>)> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model is not fitted yet"))?;

            let (flat, rows, _cols) = extract_matrix_data(py, &data)?;
            let (base_val, shap_matrix) = model.tree_shap_batch(&flat, rows);
            Ok((base_val, shap_matrix))
        }

        #[pyo3(signature = (path, format="cbm"))]
        pub fn save_model(&self, path: &str, format: &str) -> PyResult<()> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model is not fitted yet"))?;
            model
                .save_model(path, format)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            Ok(())
        }

        #[pyo3(signature = (path, format="cbm"))]
        pub fn load_model(&mut self, path: &str, format: &str) -> PyResult<()> {
            let loaded = CatBoostModel::load_model(path, format)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            self.feature_names = loaded.feature_names.clone();
            self.loss_name = loaded.loss_name.clone();
            self.model = Some(loaded);
            Ok(())
        }

        #[staticmethod]
        #[pyo3(signature = (path, format="cbm"))]
        pub fn load(path: &str, format: &str) -> PyResult<Self> {
            let loaded = CatBoostModel::load_model(path, format)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            Ok(Self {
                feature_names: loaded.feature_names.clone(),
                loss_name: loaded.loss_name.clone(),
                model: Some(loaded),
                best_iteration: 0,
                best_score: None,
                eval_history: Vec::new(),
            })
        }

        pub fn export_python(&self, path: &str) -> PyResult<()> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model is not fitted yet"))?;
            model
                .export_python(path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            Ok(())
        }

        #[getter]
        pub fn best_iteration(&self) -> usize {
            self.best_iteration
        }

        #[getter]
        pub fn best_score(&self) -> Option<f32> {
            self.best_score
        }

        #[getter]
        pub fn eval_history(&self) -> Vec<f32> {
            self.eval_history.clone()
        }

        #[getter]
        pub fn feature_names(&self) -> Vec<String> {
            self.feature_names.clone()
        }

        #[getter]
        pub fn tree_count(&self) -> usize {
            self.model.as_ref().map(|m| m.trees.len()).unwrap_or(0)
        }

        #[getter]
        pub fn is_fitted(&self) -> bool {
            self.model.is_some()
        }

        #[getter]
        pub fn loss_name(&self) -> String {
            self.loss_name.clone()
        }

        fn __repr__(&self) -> String {
            if let Some(m) = &self.model {
                format!(
                    "<catboost_webgpu.PyModel trees={} loss='{}' features={}>",
                    m.trees.len(),
                    self.loss_name,
                    m.feature_names.len()
                )
            } else {
                "<catboost_webgpu.PyModel (unfitted)>".to_string()
            }
        }
    }

    #[pyfunction]
    pub fn is_webgpu_available() -> bool {
        rust_is_webgpu_available()
    }

    #[pyfunction]
    pub fn get_device_info(py: Python<'_>) -> PyResult<PyObject> {
        let info = rust_get_device_info();
        let dict = PyDict::new(py);
        dict.set_item("name", info.name)?;
        dict.set_item("backend", info.backend)?;
        dict.set_item("device_type", info.device_type)?;
        dict.set_item("is_gpu", info.is_gpu)?;
        dict.set_item("max_buffer_size", info.max_buffer_size)?;
        dict.set_item(
            "max_compute_workgroup_size_x",
            info.max_compute_workgroup_size_x,
        )?;
        dict.set_item(
            "max_compute_invocations_per_workgroup",
            info.max_compute_invocations_per_workgroup,
        )?;
        dict.set_item(
            "max_storage_buffer_binding_size",
            info.max_storage_buffer_binding_size,
        )?;
        Ok(dict.into())
    }

    #[pymodule]
    pub fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyPool>()?;
        m.add_class::<PyModel>()?;
        m.add_function(wrap_pyfunction!(is_webgpu_available, m)?)?;
        m.add_function(wrap_pyfunction!(get_device_info, m)?)?;
        Ok(())
    }
}

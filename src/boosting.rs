use std::sync::Arc;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use crate::cpu_engine::CpuEngine;
use crate::traits::{ComputeBackend, LossFunction, ObliviousTree, SplitCondition, SplitType};

/// Subsampling and bagging strategies supported by the boosting engine.
#[derive(Debug, Clone)]
pub enum BaggingType {
    /// No bagging; all samples used with weight 1.0.
    None,
    /// Standard Bernoulli subsampling with inclusion probability `subsample` in (0, 1].
    Bernoulli { subsample: f32 },
    /// Bayesian bootstrap with weights sampled from Exponential distribution.
    /// `bagging_temperature` = 1.0 gives standard Dirichlet/Bayesian weights.
    Bayesian { bagging_temperature: f32 },
    /// Minimal Variance Sampling (MVS) targeting ratio `subsample`.
    /// Samples with higher gradient magnitudes are prioritized to minimize variance.
    MVS { subsample: f32 },
}

impl Default for BaggingType {
    fn default() -> Self {
        BaggingType::None
    }
}

/// Boosting mode: standard Plain boosting vs. CatBoost's permutation-based Ordered boosting.
#[derive(Debug, Clone)]
pub enum BoostingType {
    Plain,
    Ordered { num_permutations: usize },
}

impl Default for BoostingType {
    fn default() -> Self {
        BoostingType::Plain
    }
}

/// Hyperparameters and settings for gradient boosting.
#[derive(Debug, Clone)]
pub struct BoostingConfig {
    pub iterations: usize,
    pub learning_rate: f32,
    pub depth: usize,
    pub l2_leaf_reg: f32,
    pub random_strength: f32,
    pub bagging_type: BaggingType,
    pub boosting_type: BoostingType,
    pub max_bins: usize,
    pub early_stopping_rounds: Option<usize>,
    pub use_best_model: bool,
    pub seed: u64,
    pub verbose: usize,
}

impl Default for BoostingConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            learning_rate: 0.1,
            depth: 6,
            l2_leaf_reg: 3.0,
            random_strength: 1.0,
            bagging_type: BaggingType::None,
            boosting_type: BoostingType::Plain,
            max_bins: 254,
            early_stopping_rounds: None,
            use_best_model: true,
            seed: 42,
            verbose: 0,
        }
    }
}

/// Result of training containing the learned trees, iteration metrics, and metadata.
#[derive(Debug, Clone)]
pub struct TrainedEnsemble {
    pub trees: Vec<ObliviousTree>,
    pub learning_rate: f32,
    pub base_score: f32,
    pub best_iteration: usize,
    pub best_score: Option<f32>,
    pub eval_history: Vec<f32>,
}

/// Gradient boosting orchestrator.
pub struct BoostingEngine {
    pub config: BoostingConfig,
    pub backend: Arc<dyn ComputeBackend>,
    pub cpu_engine: CpuEngine,
}

impl BoostingEngine {
    pub fn new(config: BoostingConfig) -> Self {
        let cpu_engine = CpuEngine::new();
        Self {
            config,
            backend: Arc::new(cpu_engine.clone()),
            cpu_engine,
        }
    }

    pub fn with_backend(config: BoostingConfig, backend: Arc<dyn ComputeBackend>) -> Self {
        Self {
            config,
            backend,
            cpu_engine: CpuEngine::new(),
        }
    }

    /// Generates bagging weights for samples for the current iteration.
    fn generate_bagging_weights(
        &self,
        bagging_type: &BaggingType,
        gradients: &[f32],
        num_samples: usize,
        rng: &mut impl Rng,
    ) -> Vec<f32> {
        match bagging_type {
            BaggingType::None => vec![1.0; num_samples],
            BaggingType::Bernoulli { subsample } => {
                let p = subsample.clamp(0.01, 1.0);
                (0..num_samples)
                    .map(|_| if rng.gen::<f32>() < p { 1.0 } else { 0.0 })
                    .collect()
            }
            BaggingType::Bayesian { bagging_temperature } => {
                let temp = bagging_temperature.max(0.0);
                if temp < 1e-6 {
                    vec![1.0; num_samples]
                } else {
                    (0..num_samples)
                        .map(|_| {
                            let u: f32 = rng.gen::<f32>().max(1e-7);
                            let exp_val = -u.ln();
                            exp_val.powf(temp)
                        })
                        .collect()
                }
            }
            BaggingType::MVS { subsample } => {
                let target_k = (subsample.clamp(0.01, 1.0) * num_samples as f32).max(1.0);
                let magnitudes: Vec<f32> = gradients.iter().map(|g| g.abs()).collect();
                let max_mag = magnitudes.iter().cloned().fold(0.0f32, f32::max);

                if max_mag < 1e-8 {
                    return vec![1.0; num_samples];
                }

                // Binary search threshold mu such that sum(min(1.0, mag / mu)) == target_k
                let mut low = 1e-8f32;
                let mut high = max_mag;
                let mut mu = high;

                for _ in 0..30 {
                    let mid = (low + high) * 0.5;
                    let expected_k: f32 = magnitudes
                        .iter()
                        .map(|&m| (m / mid).min(1.0))
                        .sum();
                    if expected_k > target_k {
                        low = mid;
                    } else {
                        high = mid;
                        mu = mid;
                    }
                }

                magnitudes
                    .iter()
                    .map(|&m| {
                        let prob = (m / mu).clamp(0.0, 1.0);
                        if prob >= 1.0 {
                            1.0
                        } else if rng.gen::<f32>() < prob {
                            1.0 / prob.max(1e-5)
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
        }
    }

    /// Fits gradient boosted oblivious trees on the given binned training dataset.
    pub fn fit(
        &self,
        binned_train: &[u8],
        train_targets: &[f32],
        num_train_samples: usize,
        num_features: usize,
        feature_borders: Option<&[Vec<f32>]>,
        binned_eval: Option<&[u8]>,
        eval_targets: Option<&[f32]>,
        num_eval_samples: Option<usize>,
        loss: &dyn LossFunction,
    ) -> TrainedEnsemble {
        assert_eq!(binned_train.len(), num_train_samples * num_features);
        assert_eq!(train_targets.len(), num_train_samples);

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);

        // Initial base prediction score
        let base_score = 0.0f32;
        let mut train_preds = vec![base_score; num_train_samples];

        let has_eval = binned_eval.is_some() && eval_targets.is_some() && num_eval_samples.is_some();
        let eval_n = num_eval_samples.unwrap_or(0);
        let mut eval_preds = if has_eval {
            vec![base_score; eval_n]
        } else {
            Vec::new()
        };

        let mut trees = Vec::with_capacity(self.config.iterations);
        let mut eval_history = Vec::with_capacity(self.config.iterations);

        let lower_better = loss.lower_is_better();
        let mut best_score = if lower_better { f32::INFINITY } else { f32::NEG_INFINITY };
        let mut best_iteration = 0usize;
        let mut rounds_without_improvement = 0usize;

        // Set up permutations for Ordered boosting if requested
        let is_ordered = matches!(self.config.boosting_type, BoostingType::Ordered { .. });
        let num_perms = match self.config.boosting_type {
            BoostingType::Ordered { num_permutations } => num_permutations.max(1),
            BoostingType::Plain => 1,
        };

        let mut permutations = Vec::with_capacity(num_perms);
        let mut supporting_preds = Vec::with_capacity(num_perms);

        if is_ordered {
            for _ in 0..num_perms {
                let mut perm: Vec<usize> = (0..num_train_samples).collect();
                perm.shuffle(&mut rng);
                permutations.push(perm);
                supporting_preds.push(vec![base_score; num_train_samples]);
            }
        }

        let mut gradients = vec![0.0f32; num_train_samples];
        let mut hessians = vec![0.0f32; num_train_samples];

        for it in 0..self.config.iterations {
            let iter_seed = self.config.seed.wrapping_add((it as u64) * 99991 + 7);

            // Compute gradients
            if is_ordered {
                let perm_idx = it % num_perms;
                // Gradients computed from supporting predictions for unbiased structure selection
                loss.compute_gradients_hessians(
                    train_targets,
                    &supporting_preds[perm_idx],
                    &mut gradients,
                    &mut hessians,
                );
            } else {
                loss.compute_gradients_hessians(
                    train_targets,
                    &train_preds,
                    &mut gradients,
                    &mut hessians,
                );
            }

            // Generate bagging weights
            let weights = self.generate_bagging_weights(
                &self.config.bagging_type,
                &gradients,
                num_train_samples,
                &mut rng,
            );

            // Apply bagging weights to gradients and hessians for tree structure finding
            let mut weighted_g = vec![0.0f32; num_train_samples];
            let mut weighted_h = vec![0.0f32; num_train_samples];
            for i in 0..num_train_samples {
                weighted_g[i] = gradients[i] * weights[i];
                weighted_h[i] = (hessians[i] * weights[i]).max(1e-12);
            }

            // Build oblivious tree splits of specified depth
            let depth = self.config.depth;
            let mut splits = Vec::with_capacity(depth);
            let mut leaf_indices = vec![0u32; num_train_samples];

            for d in 0..depth {
                let num_leaves = 1 << d;
                let histograms = self.backend.compute_histograms(
                    binned_train,
                    &leaf_indices,
                    &weighted_g,
                    &weighted_h,
                    num_train_samples,
                    num_features,
                    num_leaves,
                    self.config.max_bins,
                );

                let best_cand = self.cpu_engine.find_best_split(
                    &histograms,
                    num_features,
                    num_leaves,
                    self.config.max_bins,
                    self.config.l2_leaf_reg,
                    self.config.random_strength,
                    iter_seed.wrapping_add(d as u64 * 31),
                );

                let (feat_idx, bin_thresh) = if let Some(cand) = best_cand {
                    (cand.feature_idx, cand.bin_threshold)
                } else {
                    (0, 0)
                };

                let continuous_threshold = if let Some(borders) = feature_borders {
                    if feat_idx < borders.len() && (bin_thresh as usize) < borders[feat_idx].len() {
                        borders[feat_idx][bin_thresh as usize]
                    } else {
                        bin_thresh as f32
                    }
                } else {
                    bin_thresh as f32
                };

                splits.push(SplitCondition {
                    feature_idx: feat_idx,
                    bin_threshold: bin_thresh,
                    continuous_threshold,
                    split_type: SplitType::Numerical,
                });

                // Partition leaves in-place
                self.backend.partition_leaves(
                    binned_train,
                    &mut leaf_indices,
                    feat_idx,
                    bin_thresh,
                    d,
                    num_train_samples,
                    num_features,
                );
            }

            // Compute main leaf values using current training gradients
            let final_num_leaves = 1 << depth;
            let leaf_values = if is_ordered {
                // For Ordered boosting, final leaf values are computed from current train_preds
                let mut main_g = vec![0.0f32; num_train_samples];
                let mut main_h = vec![0.0f32; num_train_samples];
                loss.compute_gradients_hessians(
                    train_targets,
                    &train_preds,
                    &mut main_g,
                    &mut main_h,
                );
                self.cpu_engine.compute_leaf_values(
                    &leaf_indices,
                    &main_g,
                    &main_h,
                    final_num_leaves,
                    self.config.l2_leaf_reg,
                )
            } else {
                self.cpu_engine.compute_leaf_values(
                    &leaf_indices,
                    &weighted_g,
                    &weighted_h,
                    final_num_leaves,
                    self.config.l2_leaf_reg,
                )
            };

            // Update main predictions
            self.backend.update_predictions(
                &mut train_preds,
                &leaf_indices,
                &leaf_values,
                self.config.learning_rate,
            );

            // In Ordered boosting, update supporting predictions via prefix sums along the permutation
            if is_ordered {
                for p_idx in 0..num_perms {
                    let perm = &permutations[p_idx];
                    let supp = &mut supporting_preds[p_idx];

                    let mut prefix_g = vec![0.0f32; final_num_leaves];
                    let mut prefix_h = vec![0.0f32; final_num_leaves];

                    for &sample_idx in perm {
                        let leaf = leaf_indices[sample_idx] as usize;
                        let denom = prefix_h[leaf] + self.config.l2_leaf_reg;
                        let val = if denom > 1e-12 {
                            -prefix_g[leaf] / denom
                        } else {
                            0.0
                        };

                        supp[sample_idx] += self.config.learning_rate * val;

                        prefix_g[leaf] += gradients[sample_idx];
                        prefix_h[leaf] += hessians[sample_idx];
                    }
                }
            }

            let tree = ObliviousTree::new(depth, splits, leaf_values);
            trees.push(tree.clone());

            // Evaluation tracking on validation set
            if has_eval {
                let b_eval = binned_eval.unwrap();
                let e_targets = eval_targets.unwrap();

                // Partition eval leaf indices
                let mut eval_leaf_indices = vec![0u32; eval_n];
                for (d, split) in tree.splits.iter().enumerate() {
                    self.backend.partition_leaves(
                        b_eval,
                        &mut eval_leaf_indices,
                        split.feature_idx,
                        split.bin_threshold,
                        d,
                        eval_n,
                        num_features,
                    );
                }

                self.backend.update_predictions(
                    &mut eval_preds,
                    &eval_leaf_indices,
                    &tree.leaf_values,
                    self.config.learning_rate,
                );

                let eval_metric = loss.evaluate_metric(e_targets, &eval_preds);
                eval_history.push(eval_metric);

                let is_improvement = if lower_better {
                    eval_metric < best_score - 1e-7
                } else {
                    eval_metric > best_score + 1e-7
                };

                if is_improvement {
                    best_score = eval_metric;
                    best_iteration = it;
                    rounds_without_improvement = 0;
                } else {
                    rounds_without_improvement += 1;
                }

                if self.config.verbose > 0 && (it + 1) % self.config.verbose == 0 {
                    println!(
                        "Iteration [{:4}/{}]: eval {} = {:.6} (best = {:.6} at it {})",
                        it + 1,
                        self.config.iterations,
                        loss.name(),
                        eval_metric,
                        best_score,
                        best_iteration
                    );
                }

                // Check early stopping
                if let Some(es_rounds) = self.config.early_stopping_rounds {
                    if rounds_without_improvement >= es_rounds {
                        if self.config.verbose > 0 {
                            println!(
                                "Early stopping triggered after {} rounds without improvement at iteration {}.",
                                rounds_without_improvement,
                                it + 1
                            );
                        }
                        break;
                    }
                }
            }
        }

        // If use_best_model is enabled and eval set was tracked, trim trees to best_iteration + 1
        if has_eval && self.config.use_best_model && !trees.is_empty() {
            let keep_count = (best_iteration + 1).min(trees.len());
            trees.truncate(keep_count);
        }

        TrainedEnsemble {
            trees,
            learning_rate: self.config.learning_rate,
            base_score,
            best_iteration,
            best_score: if has_eval { Some(best_score) } else { None },
            eval_history,
        }
    }
}

use rayon::prelude::*;
use rand::{seq::SliceRandom, SeedableRng};
use crate::traits::{LossFunction, ObliviousTree};

// ============================================================================
// Combinatorial Helpers for Exact TreeSHAP Weights
// ============================================================================

fn factorial(n: usize) -> f64 {
    let mut res = 1.0f64;
    for i in 2..=n {
        res *= i as f64;
    }
    res
}

fn n_choose_k(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k = k.min(n - k);
    let mut res = 1.0f64;
    for i in 1..=k {
        res = res * (n - k + i) as f64 / i as f64;
    }
    res
}

/// Precomputes exact Shapley kernel weights W(m, D) for oblivious trees.
///
/// W(m, D) = sum_{s=0..m} binom(m, s) * (s! * (D - 1 - s)! / D!) * (1 / 2^(D - s))
fn compute_shap_weight(m: usize, d: usize) -> f32 {
    if d == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let d_fact = factorial(d);

    for s in 0..=m {
        let comb = n_choose_k(m, s);
        let weight = (factorial(s) * factorial(d - 1 - s)) / d_fact;
        let p2 = (2.0f64).powi((d - s) as i32);
        sum += comb * weight / p2;
    }
    sum as f32
}

/// Precomputed weights table for depths up to 16.
/// Table layout: `SHAP_WEIGHTS[depth][m]` where depth in 1..=16, m in 0..depth.
pub struct ShapWeightTable {
    weights: Vec<Vec<f32>>,
}

impl ShapWeightTable {
    pub fn new(max_depth: usize) -> Self {
        let mut weights = Vec::with_capacity(max_depth + 1);
        weights.push(Vec::new()); // depth 0 unused

        for d in 1..=max_depth {
            let mut row = Vec::with_capacity(d);
            for m in 0..d {
                row.push(compute_shap_weight(m, d));
            }
            weights.push(row);
        }
        Self { weights }
    }

    #[inline]
    pub fn get(&self, depth: usize, m: usize) -> f32 {
        if depth == 0 || depth >= self.weights.len() || m >= self.weights[depth].len() {
            0.0
        } else {
            self.weights[depth][m]
        }
    }
}

// Global cached table for depths up to 16
static LAZY_WEIGHTS: std::sync::OnceLock<ShapWeightTable> = std::sync::OnceLock::new();

pub fn get_shap_table() -> &'static ShapWeightTable {
    LAZY_WEIGHTS.get_or_init(|| ShapWeightTable::new(16))
}

// ============================================================================
// Fast Oblivious Tree SHAP: Exact TreeSHAP in O(D * 2^D) per sample
// ============================================================================

/// Result of Tree SHAP calculation for a sample.
#[derive(Debug, Clone)]
pub struct ShapValues {
    /// Expected value of the model (base prediction / bias).
    pub base_value: f32,
    /// Exact Shapley attribution per feature.
    pub values: Vec<f32>,
}

/// Computes exact Shapley values for a single oblivious tree in O(D * 2^D) time.
///
/// Guaranteed to satisfy the Shapley Efficiency Axiom:
/// `base_value + sum(shap_values) == tree.predict(sample)`
pub fn tree_shap_single(tree: &ObliviousTree, sample: &[f32], num_features: usize) -> ShapValues {
    let depth = tree.depth;
    let mut values = vec![0.0f32; num_features];

    if depth == 0 {
        return ShapValues {
            base_value: tree.leaf_values.first().copied().unwrap_or(0.0),
            values,
        };
    }

    let num_leaves = 1 << depth;
    // Base value is the unconditioned expectation of the tree
    let base_value: f32 = tree.leaf_values.iter().sum::<f32>() / (num_leaves as f32);

    // Sample bitmask: which side does sample take at each split level
    let mut sample_mask = 0usize;
    for (d, split) in tree.splits.iter().enumerate() {
        if sample[split.feature_idx] > split.continuous_threshold {
            sample_mask |= 1 << d;
        }
    }

    let table = get_shap_table();

    // Iterate over all 2^D leaves
    for leaf in 0..num_leaves {
        let v_l = tree.leaf_values[leaf];
        if v_l.abs() < 1e-12 {
            continue;
        }

        // bitmask diff: 0 where leaf matches sample, 1 where they differ
        let diff = leaf ^ sample_mask;
        // Total matching split levels across the whole tree
        let total_matching = depth - diff.count_ones() as usize;

        for d in 0..depth {
            let split = &tree.splits[d];
            let bit_matches = ((diff >> d) & 1) == 0;
            // Number of OTHER splits where leaf matches sample
            let m = if bit_matches {
                total_matching - 1
            } else {
                total_matching
            };

            let w = table.get(depth, m);
            let contrib = if bit_matches { w * v_l } else { -w * v_l };

            if split.feature_idx < num_features {
                values[split.feature_idx] += contrib;
            }
        }
    }

    ShapValues { base_value, values }
}

/// Computes exact Shapley values for an ensemble of oblivious trees for a single sample.
pub fn ensemble_tree_shap(
    trees: &[ObliviousTree],
    base_score: f32,
    sample: &[f32],
    num_features: usize,
) -> ShapValues {
    let mut total_base = base_score;
    let mut total_values = vec![0.0f32; num_features];

    for tree in trees {
        let res = tree_shap_single(tree, sample, num_features);
        total_base += res.base_value;
        for (dst, src) in total_values.iter_mut().zip(res.values) {
            *dst += src;
        }
    }

    ShapValues {
        base_value: total_base,
        values: total_values,
    }
}

/// Computes exact Shapley values for an ensemble across multiple samples in parallel.
///
/// Output: `Vec<Vec<f32>>` of shape `[num_samples, num_features]`, along with overall base value.
pub fn ensemble_tree_shap_batch(
    trees: &[ObliviousTree],
    base_score: f32,
    samples: &[f32],
    num_samples: usize,
    num_features: usize,
) -> (f32, Vec<Vec<f32>>) {
    assert_eq!(samples.len(), num_samples * num_features);

    let total_base: f32 = base_score
        + trees
            .iter()
            .map(|t| {
                let n_leaves = 1 << t.depth;
                t.leaf_values.iter().sum::<f32>() / n_leaves as f32
            })
            .sum::<f32>();

    let shap_matrix: Vec<Vec<f32>> = (0..num_samples)
        .into_par_iter()
        .map(|i| {
            let sample = &samples[i * num_features..(i + 1) * num_features];
            let mut vals = vec![0.0f32; num_features];

            for tree in trees {
                let res = tree_shap_single(tree, sample, num_features);
                for (dst, src) in vals.iter_mut().zip(res.values) {
                    *dst += src;
                }
            }
            vals
        })
        .collect();

    (total_base, shap_matrix)
}

// ============================================================================
// Feature Importance Algorithms
// ============================================================================

/// Method used to evaluate feature importance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureImportanceType {
    /// Evaluates how much feature splits change predictions across leaves.
    PredictionValuesChange,
    /// Permutation feature importance measuring change in loss metric.
    LossFunctionChange,
}

/// Computes PredictionValuesChange feature importances.
///
/// For each split level in every tree, calculates the leaf value differences between
/// corresponding subtrees differing only in that split, normalized to sum to 100.0.
pub fn feature_importance_prediction_values_change(
    trees: &[ObliviousTree],
    num_features: usize,
) -> Vec<f32> {
    let mut importances = vec![0.0f32; num_features];

    for tree in trees {
        let depth = tree.depth;
        if depth == 0 {
            continue;
        }

        let num_leaves = 1 << depth;
        for (d, split) in tree.splits.iter().enumerate() {
            let f = split.feature_idx;
            if f >= num_features {
                continue;
            }

            // At depth d, leaf pairs differ by bit d: (leaf, leaf | (1 << d))
            let bit = 1 << d;
            let mut diff_sum = 0.0f32;
            let mut pairs = 0usize;

            for leaf in 0..num_leaves {
                if (leaf & bit) == 0 {
                    let other_leaf = leaf | bit;
                    let diff = tree.leaf_values[other_leaf] - tree.leaf_values[leaf];
                    diff_sum += diff * diff;
                    pairs += 1;
                }
            }

            if pairs > 0 {
                importances[f] += (diff_sum / pairs as f32).sqrt();
            }
        }
    }

    // Normalize to sum to 100.0
    let total: f32 = importances.iter().sum();
    if total > 1e-12 {
        for val in &mut importances {
            *val = (*val / total) * 100.0;
        }
    }

    importances
}

/// Computes LossFunctionChange (Permutation Importance) on a dataset.
pub fn feature_importance_loss_function_change(
    trees: &[ObliviousTree],
    base_score: f32,
    data: &[f32],
    targets: &[f32],
    num_samples: usize,
    num_features: usize,
    loss: &dyn LossFunction,
    seed: u64,
) -> Vec<f32> {
    assert_eq!(data.len(), num_samples * num_features);
    assert_eq!(targets.len(), num_samples);

    if num_samples == 0 || num_features == 0 {
        return vec![0.0; num_features];
    }

    // Baseline predictions
    let mut base_preds = vec![base_score; num_samples];
    for tree in trees {
        for i in 0..num_samples {
            let sample = &data[i * num_features..(i + 1) * num_features];
            base_preds[i] += tree.predict_continuous(sample);
        }
    }

    let baseline_loss = loss.evaluate_metric(targets, &base_preds);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut importances = vec![0.0f32; num_features];
    let lower_better = loss.lower_is_better();

    for f in 0..num_features {
        // Extract column f and permute
        let mut col_vals: Vec<f32> = (0..num_samples)
            .map(|i| data[i * num_features + f])
            .collect();
        col_vals.shuffle(&mut rng);

        // Compute loss with permuted feature f
        let mut permuted_preds = vec![base_score; num_samples];
        for tree in trees {
            for i in 0..num_samples {
                let orig_sample = &data[i * num_features..(i + 1) * num_features];
                // Temporarily build permuted sample
                let mut sample_buf = orig_sample.to_vec();
                sample_buf[f] = col_vals[i];
                permuted_preds[i] += tree.predict_continuous(&sample_buf);
            }
        }

        let permuted_loss = loss.evaluate_metric(targets, &permuted_preds);
        let change = if lower_better {
            permuted_loss - baseline_loss
        } else {
            baseline_loss - permuted_loss
        };

        importances[f] = change.max(0.0);
    }

    // Normalize to 100.0
    let total: f32 = importances.iter().sum();
    if total > 1e-12 {
        for val in &mut importances {
            *val = (*val / total) * 100.0;
        }
    }

    importances
}

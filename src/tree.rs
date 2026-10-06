pub use crate::traits::{ObliviousTree, SplitCandidate, SplitCondition, SplitType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Computes the oblivious tree leaf index branchlessly using bitmask manipulation.
///
/// For depth D, the leaf index is in 0..2^D:
///   leaf = sum_{d=0}^{D-1} ((binned_sample[split_d.feature] > split_d.bin_threshold) as usize) << d
#[inline]
pub fn compute_leaf_index_binned(splits: &[SplitCondition], binned_sample: &[u8]) -> usize {
    let mut leaf = 0usize;
    for (d, split) in splits.iter().enumerate() {
        let bit = (binned_sample[split.feature_idx] > split.bin_threshold) as usize;
        leaf |= bit << d;
    }
    leaf
}

/// Computes the oblivious tree leaf index from continuous features.
#[inline]
pub fn compute_leaf_index_continuous(splits: &[SplitCondition], continuous_sample: &[f32]) -> usize {
    let mut leaf = 0usize;
    for (d, split) in splits.iter().enumerate() {
        let bit = (continuous_sample[split.feature_idx] > split.continuous_threshold) as usize;
        leaf |= bit << d;
    }
    leaf
}

/// In-place partition leaves update on CPU:
/// For depth `d`, updates: `leaf_indices[i] |= ((binned_data[i, f*] > b*) as u32) << d`.
/// Vectorized and parallelized using `rayon`.
pub fn partition_leaves_cpu(
    binned_data: &[u8],
    leaf_indices: &mut [u32],
    feature_idx: usize,
    bin_threshold: u8,
    depth: usize,
    num_samples: usize,
    num_features: usize,
) {
    assert_eq!(binned_data.len(), num_samples * num_features);
    assert_eq!(leaf_indices.len(), num_samples);

    let bit_shift = depth as u32;

    leaf_indices
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, leaf_idx)| {
            let bin = binned_data[i * num_features + feature_idx];
            let bit = (bin > bin_threshold) as u32;
            *leaf_idx |= bit << bit_shift;
        });
}

/// Evaluates optimal leaf values for an oblivious tree given sample gradients, hessians,
/// current leaf assignments, and L2 leaf regularization lambda.
///
/// leaf_val_l = - (sum_{i in leaf_l} g_i) / (sum_{i in leaf_l} h_i + lambda)
pub fn calculate_leaf_values(
    leaf_indices: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    num_leaves: usize,
    l2_reg: f32,
) -> Vec<f32> {
    assert_eq!(leaf_indices.len(), gradients.len());
    assert_eq!(gradients.len(), hessians.len());

    let mut sum_g = vec![0.0f32; num_leaves];
    let mut sum_h = vec![0.0f32; num_leaves];

    for i in 0..leaf_indices.len() {
        let leaf = leaf_indices[i] as usize;
        if leaf < num_leaves {
            sum_g[leaf] += gradients[i];
            sum_h[leaf] += hessians[i];
        }
    }

    let mut leaf_values = vec![0.0f32; num_leaves];
    for l in 0..num_leaves {
        let denom = sum_h[l] + l2_reg;
        if denom > 1e-12 {
            leaf_values[l] = -sum_g[l] / denom;
        } else {
            leaf_values[l] = 0.0;
        }
    }

    leaf_values
}

/// Calculates the split gain of partitioning leaves at current depth `d`.
///
/// Gain = 0.5 * sum_l [ (sum_g_l)^2 / (sum_h_l + lambda) ]
pub fn calculate_split_gain(
    current_leaf_indices: &[u32],
    binned_data: &[u8],
    gradients: &[f32],
    hessians: &[f32],
    feature_idx: usize,
    bin_threshold: u8,
    current_depth: usize,
    num_samples: usize,
    num_features: usize,
    l2_reg: f32,
) -> f32 {
    let current_leaves = 1 << current_depth;
    let next_leaves = 1 << (current_depth + 1);

    let mut next_sum_g = vec![0.0f32; next_leaves];
    let mut next_sum_h = vec![0.0f32; next_leaves];

    let mut curr_sum_g = vec![0.0f32; current_leaves];
    let mut curr_sum_h = vec![0.0f32; current_leaves];

    let bit_shift = current_depth as u32;

    for i in 0..num_samples {
        let curr_leaf = current_leaf_indices[i] as usize;
        let bin = binned_data[i * num_features + feature_idx];
        let bit = (bin > bin_threshold) as u32;
        let next_leaf = curr_leaf | ((bit << bit_shift) as usize);

        let g = gradients[i];
        let h = hessians[i];

        curr_sum_g[curr_leaf] += g;
        curr_sum_h[curr_leaf] += h;

        next_sum_g[next_leaf] += g;
        next_sum_h[next_leaf] += h;
    }

    let curr_score: f32 = curr_sum_g
        .iter()
        .zip(curr_sum_h.iter())
        .map(|(&g, &h)| {
            let denom = h + l2_reg;
            if denom > 1e-12 {
                (g * g) / denom
            } else {
                0.0
            }
        })
        .sum();

    let next_score: f32 = next_sum_g
        .iter()
        .zip(next_sum_h.iter())
        .map(|(&g, &h)| {
            let denom = h + l2_reg;
            if denom > 1e-12 {
                (g * g) / denom
            } else {
                0.0
            }
        })
        .sum();

    0.5 * (next_score - curr_score)
}

/// Finds the best split candidate over candidate features and border thresholds.
pub fn find_best_split(
    current_leaf_indices: &[u32],
    binned_data: &[u8],
    gradients: &[f32],
    hessians: &[f32],
    candidate_features: &[usize],
    borders_per_feature: &[Vec<f32>],
    current_depth: usize,
    num_samples: usize,
    num_features: usize,
    l2_reg: f32,
) -> Option<SplitCandidate> {
    let mut best_candidate: Option<SplitCandidate> = None;

    for &f in candidate_features {
        let num_borders = borders_per_feature[f].len();
        for b in 0..num_borders {
            let gain = calculate_split_gain(
                current_leaf_indices,
                binned_data,
                gradients,
                hessians,
                f,
                b as u8,
                current_depth,
                num_samples,
                num_features,
                l2_reg,
            );

            if let Some(ref best) = best_candidate {
                if gain > best.gain {
                    best_candidate = Some(SplitCandidate {
                        feature_idx: f,
                        bin_threshold: b as u8,
                        gain,
                    });
                }
            } else if gain > 0.0 {
                best_candidate = Some(SplitCandidate {
                    feature_idx: f,
                    bin_threshold: b as u8,
                    gain,
                });
            }
        }
    }

    best_candidate
}

/// Helper methods for ObliviousTree.
pub trait ObliviousTreeExt {
    /// Batch predictions on row-major binned data.
    fn predict_binned_batch(
        &self,
        binned_data: &[u8],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<f32>;

    /// Batch predictions on row-major continuous data.
    fn predict_continuous_batch(
        &self,
        raw_data: &[f32],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<f32>;

    /// Predicts leaf indices [0..2^depth) for a batch of binned samples.
    fn predict_leaf_indices_binned(
        &self,
        binned_data: &[u8],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<u32>;

    /// Computes exact Tree SHAP values for a single binned sample.
    ///
    /// SHAP efficiency axiom: sum_{f} shap[f] == tree.predict(x) - E[tree]
    fn compute_shap_binned(&self, binned_sample: &[u8], num_features: usize) -> Vec<f32>;
}

impl ObliviousTreeExt for ObliviousTree {
    fn predict_binned_batch(
        &self,
        binned_data: &[u8],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<f32> {
        assert_eq!(binned_data.len(), num_samples * num_features);

        let mut predictions = vec![0.0f32; num_samples];
        predictions
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, pred)| {
                let sample_slice = &binned_data[i * num_features..(i + 1) * num_features];
                *pred = self.predict_binned(sample_slice);
            });

        predictions
    }

    fn predict_continuous_batch(
        &self,
        raw_data: &[f32],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<f32> {
        assert_eq!(raw_data.len(), num_samples * num_features);

        let mut predictions = vec![0.0f32; num_samples];
        predictions
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, pred)| {
                let sample_slice = &raw_data[i * num_features..(i + 1) * num_features];
                *pred = self.predict_continuous(sample_slice);
            });

        predictions
    }

    fn predict_leaf_indices_binned(
        &self,
        binned_data: &[u8],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<u32> {
        assert_eq!(binned_data.len(), num_samples * num_features);

        let mut leaf_indices = vec![0u32; num_samples];
        leaf_indices
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, leaf)| {
                let sample_slice = &binned_data[i * num_features..(i + 1) * num_features];
                *leaf = self.predict_leaf_binned(sample_slice) as u32;
            });

        leaf_indices
    }

    fn compute_shap_binned(&self, binned_sample: &[u8], num_features: usize) -> Vec<f32> {
        let mut shap_values = vec![0.0f32; num_features];
        let num_leaves = 1 << self.depth;
        if num_leaves == 0 {
            return shap_values;
        }

        // Expected value over uniform distribution of leaves: E[tree]
        let mean_leaf: f32 = self.leaf_values.iter().sum::<f32>() / num_leaves as f32;

        // For an oblivious tree, each level d splits on a single feature split_d.
        // Let sample's bit at level d be b_d = (binned_sample[feat_d] > thresh_d) as usize.
        // The marginal contribution of level d is:
        //   E[tree | bit_d = b_d] - E[tree]
        for (d, split) in self.splits.iter().enumerate() {
            let sample_bit = (binned_sample[split.feature_idx] > split.bin_threshold) as usize;

            // Compute conditional average over leaves where bit at level d matches sample_bit
            let mut sum_cond = 0.0f32;
            let mut count_cond = 0;

            for (leaf, &val) in self.leaf_values.iter().enumerate() {
                let leaf_bit = (leaf >> d) & 1;
                if leaf_bit == sample_bit {
                    sum_cond += val;
                    count_cond += 1;
                }
            }

            let cond_mean = if count_cond > 0 {
                sum_cond / count_cond as f32
            } else {
                mean_leaf
            };

            let level_shap = cond_mean - mean_leaf;
            shap_values[split.feature_idx] += level_shap;
        }

        shap_values
    }
}

/// An ensemble of Oblivious Decision Trees (Gradient Boosted Forest).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeEnsemble {
    pub trees: Vec<ObliviousTree>,
    pub learning_rate: f32,
    pub base_score: f32,
}

impl TreeEnsemble {
    pub fn new(learning_rate: f32, base_score: f32) -> Self {
        Self {
            trees: Vec::new(),
            learning_rate,
            base_score,
        }
    }

    /// Adds a trained oblivious tree to the ensemble.
    pub fn add_tree(&mut self, tree: ObliviousTree) {
        self.trees.push(tree);
    }

    /// Number of trees in the ensemble.
    #[inline]
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Predicts raw continuous output for a single binned sample.
    pub fn predict_binned(&self, binned_sample: &[u8]) -> f32 {
        let mut pred = self.base_score;
        for tree in &self.trees {
            pred += self.learning_rate * tree.predict_binned(binned_sample);
        }
        pred
    }

    /// Predicts raw continuous output for a single continuous sample.
    pub fn predict_continuous(&self, continuous_sample: &[f32]) -> f32 {
        let mut pred = self.base_score;
        for tree in &self.trees {
            pred += self.learning_rate * tree.predict_continuous(continuous_sample);
        }
        pred
    }

    /// Batch predictions on row-major binned data.
    pub fn predict_binned_batch(
        &self,
        binned_data: &[u8],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<f32> {
        let mut predictions = vec![self.base_score; num_samples];
        for tree in &self.trees {
            let tree_preds = tree.predict_binned_batch(binned_data, num_samples, num_features);
            for i in 0..num_samples {
                predictions[i] += self.learning_rate * tree_preds[i];
            }
        }
        predictions
    }

    /// Batch predictions on row-major continuous data.
    pub fn predict_continuous_batch(
        &self,
        raw_data: &[f32],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<f32> {
        let mut predictions = vec![self.base_score; num_samples];
        for tree in &self.trees {
            let tree_preds = tree.predict_continuous_batch(raw_data, num_samples, num_features);
            for i in 0..num_samples {
                predictions[i] += self.learning_rate * tree_preds[i];
            }
        }
        predictions
    }

    /// Staged predictions after each boosting iteration (useful for validation curves / early stopping).
    pub fn staged_predict_binned(
        &self,
        binned_data: &[u8],
        num_samples: usize,
        num_features: usize,
    ) -> Vec<Vec<f32>> {
        let mut current_preds = vec![self.base_score; num_samples];
        let mut staged = Vec::with_capacity(self.trees.len());

        for tree in &self.trees {
            let tree_preds = tree.predict_binned_batch(binned_data, num_samples, num_features);
            for i in 0..num_samples {
                current_preds[i] += self.learning_rate * tree_preds[i];
            }
            staged.push(current_preds.clone());
        }

        staged
    }

    /// Computes feature importances based on split frequency across all trees.
    pub fn feature_importances(&self, num_features: usize) -> Vec<f32> {
        let mut importances = vec![0.0f32; num_features];
        let mut total_splits = 0.0f32;

        for tree in &self.trees {
            for split in &tree.splits {
                if split.feature_idx < num_features {
                    importances[split.feature_idx] += 1.0;
                    total_splits += 1.0;
                }
            }
        }

        if total_splits > 0.0 {
            for imp in &mut importances {
                *imp /= total_splits;
            }
        }

        importances
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oblivious_tree_bitmask_prediction() {
        // Construct a depth-2 oblivious tree:
        // Level 0: feature 0 > bin 2 (or continuous > 1.5)
        // Level 1: feature 1 > bin 5 (or continuous > 10.0)
        let splits = vec![
            SplitCondition {
                feature_idx: 0,
                bin_threshold: 2,
                continuous_threshold: 1.5,
                split_type: SplitType::Numerical,
            },
            SplitCondition {
                feature_idx: 1,
                bin_threshold: 5,
                continuous_threshold: 10.0,
                split_type: SplitType::Numerical,
            },
        ];

        // 4 leaves:
        // leaf 0 (00): f0 <= 2, f1 <= 5 -> 10.0
        // leaf 1 (01): f0 > 2,  f1 <= 5 -> 20.0
        // leaf 2 (10): f0 <= 2, f1 > 5  -> 30.0
        // leaf 3 (11): f0 > 2,  f1 > 5  -> 40.0
        let leaf_values = vec![10.0, 20.0, 30.0, 40.0];
        let tree = ObliviousTree::new(2, splits, leaf_values);

        // Test sample A: f0=1, f1=4 -> leaf 0
        assert_eq!(tree.predict_leaf_binned(&[1, 4]), 0);
        assert_eq!(tree.predict_binned(&[1, 4]), 10.0);
        assert_eq!(tree.predict_leaf_continuous(&[1.0, 4.0]), 0);
        assert_eq!(tree.predict_continuous(&[1.0, 4.0]), 10.0);

        // Test sample B: f0=3, f1=4 -> leaf 1 (bit 0 = 1, bit 1 = 0)
        assert_eq!(tree.predict_leaf_binned(&[3, 4]), 1);
        assert_eq!(tree.predict_binned(&[3, 4]), 20.0);
        assert_eq!(tree.predict_leaf_continuous(&[2.0, 4.0]), 1);
        assert_eq!(tree.predict_continuous(&[2.0, 4.0]), 20.0);

        // Test sample C: f0=1, f1=7 -> leaf 2 (bit 0 = 0, bit 1 = 1)
        assert_eq!(tree.predict_leaf_binned(&[1, 7]), 2);
        assert_eq!(tree.predict_binned(&[1, 7]), 30.0);

        // Test sample D: f0=4, f1=8 -> leaf 3 (bit 0 = 1, bit 1 = 1)
        assert_eq!(tree.predict_leaf_binned(&[4, 8]), 3);
        assert_eq!(tree.predict_binned(&[4, 8]), 40.0);
    }

    #[test]
    fn test_batch_prediction_and_branchless_partition() {
        let splits = vec![
            SplitCondition {
                feature_idx: 0,
                bin_threshold: 1,
                continuous_threshold: 1.0,
                split_type: SplitType::Numerical,
            },
        ];
        let tree = ObliviousTree::new(1, splits, vec![-1.0, 1.0]);

        // 3 samples, 2 features
        let binned_data = vec![
            0, 0, // sample 0: f0=0 <= 1 -> leaf 0
            2, 0, // sample 1: f0=2 > 1  -> leaf 1
            1, 0, // sample 2: f0=1 <= 1 -> leaf 0
        ];

        let preds = tree.predict_binned_batch(&binned_data, 3, 2);
        assert_eq!(preds, vec![-1.0, 1.0, -1.0]);

        // Test partition_leaves_cpu
        let mut leaf_indices = vec![0u32; 3];
        partition_leaves_cpu(&binned_data, &mut leaf_indices, 0, 1, 0, 3, 2);
        assert_eq!(leaf_indices, vec![0, 1, 0]);
    }

    #[test]
    fn test_leaf_value_calculation_with_l2_reg() {
        let leaf_indices = vec![0, 0, 1, 1];
        let gradients = vec![-1.0, -2.0, 3.0, 1.0];
        let hessians = vec![1.0, 1.0, 1.0, 1.0];
        let l2_reg = 1.0;

        // Leaf 0: sum_g = -3.0, sum_h = 2.0 -> -(-3.0)/(2.0 + 1.0) = 3.0 / 3.0 = 1.0
        // Leaf 1: sum_g = 4.0,  sum_h = 2.0 -> -(4.0)/(2.0 + 1.0)  = -4.0 / 3.0
        let leaves = calculate_leaf_values(&leaf_indices, &gradients, &hessians, 2, l2_reg);
        assert!((leaves[0] - 1.0).abs() < 1e-6);
        assert!((leaves[1] - (-4.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_tree_ensemble() {
        let splits1 = vec![SplitCondition {
            feature_idx: 0,
            bin_threshold: 0,
            continuous_threshold: 0.0,
            split_type: SplitType::Numerical,
        }];
        let tree1 = ObliviousTree::new(1, splits1, vec![1.0, 2.0]);

        let splits2 = vec![SplitCondition {
            feature_idx: 0,
            bin_threshold: 0,
            continuous_threshold: 0.0,
            split_type: SplitType::Numerical,
        }];
        let tree2 = ObliviousTree::new(1, splits2, vec![0.5, 1.5]);

        let mut ensemble = TreeEnsemble::new(0.1, 0.0);
        ensemble.add_tree(tree1);
        ensemble.add_tree(tree2);

        // Sample f0 = 0 (leaf 0 in both):
        // tree 1: 1.0, tree 2: 0.5
        // ensemble = 0.0 + 0.1 * (1.0 + 0.5) = 0.15
        let pred = ensemble.predict_binned(&[0]);
        assert!((pred - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_oblivious_tree_shap_efficiency() {
        let splits = vec![
            SplitCondition {
                feature_idx: 0,
                bin_threshold: 1,
                continuous_threshold: 1.0,
                split_type: SplitType::Numerical,
            },
            SplitCondition {
                feature_idx: 1,
                bin_threshold: 2,
                continuous_threshold: 2.0,
                split_type: SplitType::Numerical,
            },
        ];
        let leaves = vec![1.0, 3.0, 5.0, 7.0];
        let tree = ObliviousTree::new(2, splits, leaves.clone());

        let sample = vec![2u8, 3u8]; // f0=2 > 1, f1=3 > 2 -> leaf 3, pred = 7.0
        let shap = tree.compute_shap_binned(&sample, 2);

        let pred = tree.predict_binned(&sample);
        let mean_leaf = leaves.iter().sum::<f32>() / 4.0; // (1+3+5+7)/4 = 4.0

        let shap_sum: f32 = shap.iter().sum();
        // SHAP efficiency: sum(shap) == pred - mean
        assert!(
            (shap_sum - (pred - mean_leaf)).abs() < 1e-5,
            "SHAP sum {} != pred {} - mean {}",
            shap_sum, pred, mean_leaf
        );
    }
}

use serde::{Deserialize, Serialize};

/// Represents a single binary split test in an oblivious decision tree level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitCondition {
    /// Zero-based index of the feature being tested.
    pub feature_idx: usize,
    /// Binned border index threshold (for binned integer data).
    pub bin_threshold: u8,
    /// Continuous threshold value (for raw floating point feature).
    pub continuous_threshold: f32,
    /// Type of feature split.
    pub split_type: SplitType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitType {
    Numerical,
    OneHot,
    Ctr,
}

/// An Oblivious Decision Tree where all nodes at level `k` share the same split test.
/// A tree of depth `D` has `D` splits and exactly `2^D` leaf values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObliviousTree {
    pub depth: usize,
    pub splits: Vec<SplitCondition>,
    pub leaf_values: Vec<f32>,
}

impl ObliviousTree {
    pub fn new(depth: usize, splits: Vec<SplitCondition>, leaf_values: Vec<f32>) -> Self {
        assert_eq!(splits.len(), depth, "Splits length must match depth");
        assert_eq!(leaf_values.len(), 1 << depth, "Leaf values length must be 2^depth");
        Self { depth, splits, leaf_values }
    }

    /// Evaluates the leaf index for a sample given continuous feature values.
    #[inline]
    pub fn predict_leaf_continuous(&self, sample: &[f32]) -> usize {
        let mut leaf = 0usize;
        for (d, split) in self.splits.iter().enumerate() {
            if sample[split.feature_idx] > split.continuous_threshold {
                leaf |= 1 << d;
            }
        }
        leaf
    }

    /// Evaluates the prediction for a sample given continuous feature values.
    #[inline]
    pub fn predict_continuous(&self, sample: &[f32]) -> f32 {
        let leaf = self.predict_leaf_continuous(sample);
        self.leaf_values[leaf]
    }

    /// Evaluates the leaf index for a sample given pre-binned u8 features.
    #[inline]
    pub fn predict_leaf_binned(&self, binned_sample: &[u8]) -> usize {
        let mut leaf = 0usize;
        for (d, split) in self.splits.iter().enumerate() {
            if binned_sample[split.feature_idx] > split.bin_threshold {
                leaf |= 1 << d;
            }
        }
        leaf
    }

    /// Evaluates the prediction for a sample given pre-binned u8 features.
    #[inline]
    pub fn predict_binned(&self, binned_sample: &[u8]) -> f32 {
        let leaf = self.predict_leaf_binned(binned_sample);
        self.leaf_values[leaf]
    }
}

/// Candidate split score and specifications.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitCandidate {
    pub feature_idx: usize,
    pub bin_threshold: u8,
    pub gain: f32,
}

/// Abstract backend interface for gradient boosting compute acceleration.
/// Implemented by both WebGPU engine and Rayon CPU engine.
pub trait ComputeBackend: Send + Sync {
    /// Returns the descriptive name of the backend (e.g. "WebGPU: AMD Radeon Graphics" or "CPU (Rayon)").
    fn name(&self) -> &str;

    /// Is this backend WebGPU-accelerated?
    fn is_gpu(&self) -> bool;

    /// Accumulates 2D gradient and hessian histograms for all leaves and feature bins.
    /// Output buffer layout: [num_features * num_leaves * max_bins * 2] where last dimension is [sum_g, sum_h].
    fn compute_histograms(
        &self,
        binned_data: &[u8],      // Row-major [num_samples, num_features]
        leaf_indices: &[u32],    // [num_samples] current leaf indices (0 .. 2^depth - 1)
        gradients: &[f32],       // [num_samples]
        hessians: &[f32],        // [num_samples]
        num_samples: usize,
        num_features: usize,
        num_leaves: usize,
        max_bins: usize,
    ) -> Vec<f32>;

    /// Partitions sample leaf indices on device/CPU when winning split is applied.
    /// For depth `d`, updates: leaf_indices[i] |= ((binned_data[i, f*] > b*) as u32) << d.
    fn partition_leaves(
        &self,
        binned_data: &[u8],
        leaf_indices: &mut [u32],
        feature_idx: usize,
        bin_threshold: u8,
        depth: usize,
        num_samples: usize,
        num_features: usize,
    );

    /// Vectorized in-place update of raw predictions:
    /// predictions[i] += learning_rate * leaf_values[leaf_indices[i]].
    fn update_predictions(
        &self,
        predictions: &mut [f32],
        leaf_indices: &[u32],
        leaf_values: &[f32],
        learning_rate: f32,
    );
}

/// Loss function interface for computing first and second order derivatives and metrics.
pub trait LossFunction: Send + Sync {
    fn name(&self) -> &str;

    /// Computes (gradient, hessian) for a single sample.
    /// Convention: negative gradient of loss with respect to raw prediction y_hat.
    /// g = dL/dy_hat, h = d^2L/dy_hat^2.
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32);

    /// Batch gradient and hessian calculation.
    fn compute_gradients_hessians(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        assert_eq!(y_true.len(), y_pred.len());
        for i in 0..y_true.len() {
            let (g, h) = self.gradient_hessian(y_true[i], y_pred[i]);
            gradients[i] = g;
            hessians[i] = h;
        }
    }

    /// Evaluates aggregate metric on dataset (e.g. RMSE, Logloss, etc.).
    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32;

    /// Whether a lower metric value is better (e.g. true for RMSE, false for Accuracy/AUC).
    fn lower_is_better(&self) -> bool {
        true
    }
}

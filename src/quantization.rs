use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Supported quantization algorithms for continuous feature binning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMethod {
    /// Divides the feature range [min, max] into equal-length segments.
    Uniform,
    /// Partitions samples into bins with approximately equal sample counts (sample quantiles).
    Median,
    /// Combines Uniform and Median quantization borders.
    UniformAndQuantiles,
    /// CatBoost's default: Greedily maximizes the sum of logarithms of bin weights.
    GreedyLogSum,
}

/// Policy for handling NaN (Not a Number) values during quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NanMode {
    /// NaN values are assigned to bin 0 (treated as smaller than all values).
    Min,
    /// NaN values are assigned to the maximum bin (treated as larger than all values).
    Max,
    /// NaN values cause an error / panic.
    Forbidden,
}

impl Default for NanMode {
    fn default() -> Self {
        NanMode::Min
    }
}

/// Maximum number of borders allowed per continuous feature (maps to at most 255 bins: 0..254).
pub const MAX_BORDERS_COUNT: usize = 254;

/// Fits quantization borders for a single continuous feature vector.
///
/// Returns a sorted `Vec<f32>` of strictly increasing threshold borders (length <= min(max_borders, 254)).
/// If there are K borders [b_0 < b_1 < ... < b_{K-1}], a value x is assigned to:
/// - bin 0 if x <= b_0
/// - bin i if b_{i-1} < x <= b_i
/// - bin K if x > b_{K-1}
pub fn fit_borders(
    values: &[f32],
    max_borders: usize,
    method: QuantizationMethod,
    nan_mode: NanMode,
) -> Vec<f32> {
    let max_borders = max_borders.min(MAX_BORDERS_COUNT);
    if max_borders == 0 || values.is_empty() {
        return Vec::new();
    }

    // Filter finite values
    let mut finite_values: Vec<f32> = values
        .iter()
        .copied()
        .filter(|v| {
            if v.is_nan() {
                assert!(
                    nan_mode != NanMode::Forbidden,
                    "NaN encountered with NanMode::Forbidden"
                );
                false
            } else {
                v.is_finite()
            }
        })
        .collect();

    if finite_values.len() < 2 {
        return Vec::new();
    }

    // Sort finite values
    finite_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Extract sorted unique values and their frequency counts
    let mut unique_values: Vec<f32> = Vec::new();
    let mut unique_counts: Vec<usize> = Vec::new();

    for &v in &finite_values {
        if unique_values.is_empty() {
            unique_values.push(v);
            unique_counts.push(1);
        } else {
            let last_idx = unique_values.len() - 1;
            // Float equality with small epsilon to group nearly identical points
            if (v - unique_values[last_idx]).abs() <= f32::EPSILON {
                unique_counts[last_idx] += 1;
            } else {
                unique_values.push(v);
                unique_counts.push(1);
            }
        }
    }

    let num_unique = unique_values.len();
    if num_unique <= 1 {
        return Vec::new();
    }

    // If the number of candidate gaps is <= max_borders, every gap between distinct
    // values is a border.
    if num_unique - 1 <= max_borders {
        return (0..num_unique - 1)
            .map(|i| (unique_values[i] + unique_values[i + 1]) * 0.5)
            .collect();
    }

    match method {
        QuantizationMethod::Uniform => fit_borders_uniform(&unique_values, max_borders),
        QuantizationMethod::Median => {
            fit_borders_median(&unique_values, &unique_counts, max_borders)
        }
        QuantizationMethod::UniformAndQuantiles => {
            fit_borders_uniform_and_quantiles(&unique_values, &unique_counts, max_borders)
        }
        QuantizationMethod::GreedyLogSum => {
            fit_borders_greedy_log_sum(&unique_values, &unique_counts, max_borders)
        }
    }
}

/// Uniform quantization: divides [min, max] into equal-length segments and
/// picks the closest midpoints between distinct observed values.
fn fit_borders_uniform(unique_values: &[f32], max_borders: usize) -> Vec<f32> {
    let min_val = unique_values[0];
    let max_val = unique_values[unique_values.len() - 1];
    let range = max_val - min_val;
    if range <= 0.0 {
        return Vec::new();
    }

    let step = range / (max_borders + 1) as f32;
    let mut candidate_borders = Vec::with_capacity(max_borders);

    for k in 1..=max_borders {
        let target = min_val + k as f32 * step;
        // Find adjacent unique values u_i <= target < u_{i+1}
        let pos = match unique_values.binary_search_by(|v| v.partial_cmp(&target).unwrap_or(Ordering::Equal)) {
            Ok(idx) => {
                if idx < unique_values.len() - 1 {
                    idx
                } else {
                    idx.saturating_sub(1)
                }
            }
            Err(idx) => {
                if idx == 0 {
                    0
                } else if idx >= unique_values.len() {
                    unique_values.len() - 2
                } else {
                    idx - 1
                }
            }
        };

        let border = (unique_values[pos] + unique_values[pos + 1]) * 0.5;
        candidate_borders.push(border);
    }

    dedup_and_sort_borders(candidate_borders, max_borders)
}

/// Median / Quantile quantization: places borders to equalize sample counts in each bin.
fn fit_borders_median(
    unique_values: &[f32],
    unique_counts: &[usize],
    max_borders: usize,
) -> Vec<f32> {
    let total_samples: usize = unique_counts.iter().sum();
    let num_unique = unique_values.len();
    if total_samples == 0 || num_unique < 2 {
        return Vec::new();
    }

    // Compute prefix sum of counts
    let mut prefix_counts = Vec::with_capacity(num_unique);
    let mut run = 0;
    for &c in unique_counts {
        run += c;
        prefix_counts.push(run);
    }

    let mut candidate_borders = Vec::with_capacity(max_borders);
    for k in 1..=max_borders {
        let target_count = (k * total_samples) / (max_borders + 1);
        let pos = match prefix_counts.binary_search(&target_count) {
            Ok(idx) => {
                if idx < num_unique - 1 {
                    idx
                } else {
                    num_unique - 2
                }
            }
            Err(idx) => {
                if idx == 0 {
                    0
                } else if idx >= num_unique {
                    num_unique - 2
                } else {
                    idx - 1
                }
            }
        };

        let border = (unique_values[pos] + unique_values[pos + 1]) * 0.5;
        candidate_borders.push(border);
    }

    dedup_and_sort_borders(candidate_borders, max_borders)
}

/// UniformAndQuantiles quantization: combines Uniform borders and Median borders.
fn fit_borders_uniform_and_quantiles(
    unique_values: &[f32],
    unique_counts: &[usize],
    max_borders: usize,
) -> Vec<f32> {
    let half_borders = (max_borders / 2).max(1);
    let rem_borders = max_borders - half_borders;

    let mut median_borders = fit_borders_median(unique_values, unique_counts, half_borders);
    let mut uniform_borders = fit_borders_uniform(unique_values, rem_borders);

    median_borders.append(&mut uniform_borders);
    dedup_and_sort_borders(median_borders, max_borders)
}

/// Structure representing a candidate interval in GreedyLogSum quantization.
#[derive(Debug, Clone)]
struct IntervalCandidate {
    left: usize,
    right: usize,
    best_split: usize,
    gain: f64,
}

impl PartialEq for IntervalCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain
    }
}

impl Eq for IntervalCandidate {}

impl PartialOrd for IntervalCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IntervalCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain.partial_cmp(&other.gain).unwrap_or(Ordering::Equal)
    }
}

/// GreedyLogSum quantization:
/// Greedily partitions intervals of distinct values using a priority queue to maximize
/// the sum of logarithms of bin weights: sum_b ln(weight_b).
/// The gain of splitting interval [l, r] at m into [l, m] and [m+1, r] is:
///   gain = ln(W(l, m)) + ln(W(m+1, r)) - ln(W(l, r))
fn fit_borders_greedy_log_sum(
    unique_values: &[f32],
    unique_counts: &[usize],
    max_borders: usize,
) -> Vec<f32> {
    let num_unique = unique_values.len();
    if num_unique < 2 {
        return Vec::new();
    }

    // Prefix sum of weights (using f64 to avoid overflow and preserve precision)
    let mut prefix_w = Vec::with_capacity(num_unique + 1);
    prefix_w.push(0.0f64);
    for &c in unique_counts {
        let last = *prefix_w.last().unwrap();
        prefix_w.push(last + c as f64);
    }

    let interval_weight = |l: usize, r: usize| -> f64 {
        prefix_w[r + 1] - prefix_w[l]
    };

    let find_best_split = |l: usize, r: usize| -> Option<IntervalCandidate> {
        if r <= l {
            return None;
        }
        let total_w = interval_weight(l, r);
        if total_w <= 0.0 {
            return None;
        }
        let total_log = total_w.ln();

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_m = l;

        // Scan candidate split points m in [l, r - 1]
        for m in l..r {
            let w_left = interval_weight(l, m);
            let w_right = interval_weight(m + 1, r);
            if w_left > 0.0 && w_right > 0.0 {
                let gain = w_left.ln() + w_right.ln() - total_log;
                if gain > best_gain {
                    best_gain = gain;
                    best_m = m;
                }
            }
        }

        if best_gain.is_finite() {
            Some(IntervalCandidate {
                left: l,
                right: r,
                best_split: best_m,
                gain: best_gain,
            })
        } else {
            None
        }
    };

    let mut heap = BinaryHeap::new();
    if let Some(initial) = find_best_split(0, num_unique - 1) {
        heap.push(initial);
    }

    let mut selected_splits = Vec::with_capacity(max_borders);

    while let Some(candidate) = heap.pop() {
        selected_splits.push(candidate.best_split);
        if selected_splits.len() >= max_borders {
            break;
        }

        // Left child [left, best_split]
        if candidate.best_split > candidate.left {
            if let Some(left_cand) = find_best_split(candidate.left, candidate.best_split) {
                heap.push(left_cand);
            }
        }

        // Right child [best_split + 1, right]
        if candidate.right > candidate.best_split + 1 {
            if let Some(right_cand) = find_best_split(candidate.best_split + 1, candidate.right) {
                heap.push(right_cand);
            }
        }
    }

    selected_splits.sort_unstable();
    selected_splits.dedup();

    let borders: Vec<f32> = selected_splits
        .into_iter()
        .map(|m| (unique_values[m] + unique_values[m + 1]) * 0.5)
        .collect();

    dedup_and_sort_borders(borders, max_borders)
}

/// Sorts, deduplicates, and limits borders to `max_borders`.
fn dedup_and_sort_borders(mut borders: Vec<f32>, max_borders: usize) -> Vec<f32> {
    borders.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    borders.retain(|v| v.is_finite());

    let mut result = Vec::with_capacity(borders.len().min(max_borders));
    for b in borders {
        if result.is_empty() {
            result.push(b);
        } else {
            let last = *result.last().unwrap();
            // Distinct threshold check
            if (b - last).abs() > f32::EPSILON && b > last {
                result.push(b);
            }
        }
        if result.len() >= max_borders {
            break;
        }
    }

    result
}

/// Quantizes a single continuous f32 value into a u8 bin index.
///
/// If `borders` has length K:
/// - Returns 0 if val <= borders[0]
/// - Returns i if borders[i-1] < val <= borders[i]
/// - Returns K if val > borders[K-1]
#[inline]
pub fn quantize_value(val: f32, borders: &[f32], nan_mode: NanMode) -> u8 {
    if val.is_nan() {
        return match nan_mode {
            NanMode::Min => 0,
            NanMode::Max => borders.len() as u8,
            NanMode::Forbidden => panic!("Encountered NaN during quantization with NanMode::Forbidden"),
        };
    }

    if borders.is_empty() {
        return 0;
    }

    // Branchless / binary search partition point
    // borders.partition_point(|&b| val > b) returns the number of borders strictly smaller than val
    let bin = borders.partition_point(|&b| val > b);
    bin as u8
}

/// Quantizes an entire column / slice of continuous f32 values into u8 bins.
pub fn quantize_column(values: &[f32], borders: &[f32], nan_mode: NanMode) -> Vec<u8> {
    values
        .iter()
        .map(|&v| quantize_value(v, borders, nan_mode))
        .collect()
}

/// Quantizes a 2D matrix stored in Row-Major layout [num_samples, num_features] into u8 bins.
///
/// Output is a flat `Vec<u8>` of size `num_samples * num_features`.
/// Parallelized over sample chunks using `rayon`.
pub fn quantize_matrix_row_major(
    raw_data: &[f32],
    num_samples: usize,
    borders_per_feature: &[Vec<f32>],
    nan_mode: NanMode,
) -> Vec<u8> {
    let num_features = borders_per_feature.len();
    assert_eq!(
        raw_data.len(),
        num_samples * num_features,
        "Matrix dimensions mismatch raw_data.len() != num_samples * num_features"
    );

    let mut binned_data = vec![0u8; num_samples * num_features];

    // Parallelize by rows/samples
    binned_data
        .par_chunks_exact_mut(num_features)
        .zip(raw_data.par_chunks_exact(num_features))
        .for_each(|(out_row, in_row)| {
            for f in 0..num_features {
                out_row[f] = quantize_value(in_row[f], &borders_per_feature[f], nan_mode);
            }
        });

    binned_data
}

/// Quantizes a 2D matrix stored in Column-Major layout [num_features, num_samples] into u8 bins.
///
/// Output is a flat `Vec<u8>` of size `num_features * num_samples`.
/// Parallelized over features using `rayon`.
pub fn quantize_matrix_col_major(
    raw_data: &[f32],
    num_samples: usize,
    borders_per_feature: &[Vec<f32>],
    nan_mode: NanMode,
) -> Vec<u8> {
    let num_features = borders_per_feature.len();
    assert_eq!(
        raw_data.len(),
        num_features * num_samples,
        "Matrix dimensions mismatch raw_data.len() != num_features * num_samples"
    );

    let mut binned_data = vec![0u8; num_features * num_samples];

    // Parallelize by columns/features
    binned_data
        .par_chunks_exact_mut(num_samples)
        .zip(raw_data.par_chunks_exact(num_samples))
        .zip(borders_per_feature.par_iter())
        .for_each(|((out_col, in_col), borders)| {
            for i in 0..num_samples {
                out_col[i] = quantize_value(in_col[i], borders, nan_mode);
            }
        });

    binned_data
}

/// Quantizer model for a single numerical feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureQuantizer {
    pub borders: Vec<f32>,
    pub method: QuantizationMethod,
    pub nan_mode: NanMode,
}

impl FeatureQuantizer {
    /// Fits a quantizer for a feature.
    pub fn fit(
        values: &[f32],
        max_borders: usize,
        method: QuantizationMethod,
        nan_mode: NanMode,
    ) -> Self {
        let borders = fit_borders(values, max_borders, method, nan_mode);
        Self {
            borders,
            method,
            nan_mode,
        }
    }

    /// Number of borders in this quantizer.
    #[inline]
    pub fn num_borders(&self) -> usize {
        self.borders.len()
    }

    /// Number of discrete bins (borders.len() + 1).
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.borders.len() + 1
    }

    /// Transforms a single value into a u8 bin.
    #[inline]
    pub fn transform_value(&self, val: f32) -> u8 {
        quantize_value(val, &self.borders, self.nan_mode)
    }

    /// Transforms a slice of values into u8 bins.
    pub fn transform(&self, values: &[f32]) -> Vec<u8> {
        quantize_column(values, &self.borders, self.nan_mode)
    }
}

/// Multi-feature quantization grid storing quantizers for each feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationGrid {
    pub quantizers: Vec<FeatureQuantizer>,
}

impl QuantizationGrid {
    /// Fits a quantization grid from row-major continuous data [num_samples, num_features].
    pub fn fit_row_major(
        raw_data: &[f32],
        num_samples: usize,
        num_features: usize,
        max_borders: usize,
        method: QuantizationMethod,
        nan_mode: NanMode,
    ) -> Self {
        assert_eq!(raw_data.len(), num_samples * num_features);

        let quantizers: Vec<FeatureQuantizer> = (0..num_features)
            .into_par_iter()
            .map(|f| {
                // Collect column data
                let col: Vec<f32> = (0..num_samples)
                    .map(|i| raw_data[i * num_features + f])
                    .collect();
                FeatureQuantizer::fit(&col, max_borders, method, nan_mode)
            })
            .collect();

        Self { quantizers }
    }

    /// Fits a quantization grid from column-major continuous data [num_features, num_samples].
    pub fn fit_col_major(
        raw_data: &[f32],
        num_samples: usize,
        num_features: usize,
        max_borders: usize,
        method: QuantizationMethod,
        nan_mode: NanMode,
    ) -> Self {
        assert_eq!(raw_data.len(), num_features * num_samples);

        let quantizers: Vec<FeatureQuantizer> = (0..num_features)
            .into_par_iter()
            .map(|f| {
                let start = f * num_samples;
                let end = start + num_samples;
                FeatureQuantizer::fit(&raw_data[start..end], max_borders, method, nan_mode)
            })
            .collect();

        Self { quantizers }
    }

    /// Extracts all border arrays.
    pub fn borders(&self) -> Vec<Vec<f32>> {
        self.quantizers.iter().map(|q| q.borders.clone()).collect()
    }

    /// Transforms row-major continuous data to row-major binned u8.
    pub fn transform_row_major(&self, raw_data: &[f32], num_samples: usize) -> Vec<u8> {
        let borders = self.borders();
        let nan_mode = self.quantizers.first().map(|q| q.nan_mode).unwrap_or_default();
        quantize_matrix_row_major(raw_data, num_samples, &borders, nan_mode)
    }

    /// Transforms column-major continuous data to column-major binned u8.
    pub fn transform_col_major(&self, raw_data: &[f32], num_samples: usize) -> Vec<u8> {
        let borders = self.borders();
        let nan_mode = self.quantizers.first().map(|q| q.nan_mode).unwrap_or_default();
        quantize_matrix_col_major(raw_data, num_samples, &borders, nan_mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_constant_feature() {
        let values = vec![3.14; 100];
        let borders = fit_borders(&values, 32, QuantizationMethod::Uniform, NanMode::Min);
        assert!(borders.is_empty(), "Constant feature should have 0 borders");

        let binned = quantize_column(&values, &borders, NanMode::Min);
        assert!(binned.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_quantize_two_distinct_values() {
        let values = vec![1.0, 2.0, 1.0, 2.0, 1.0];
        let borders = fit_borders(&values, 32, QuantizationMethod::Uniform, NanMode::Min);
        assert_eq!(borders.len(), 1);
        assert!((borders[0] - 1.5).abs() < 1e-5);

        assert_eq!(quantize_value(1.0, &borders, NanMode::Min), 0);
        assert_eq!(quantize_value(1.5, &borders, NanMode::Min), 0); // <= 1.5 goes to bin 0
        assert_eq!(quantize_value(1.51, &borders, NanMode::Min), 1);
        assert_eq!(quantize_value(2.0, &borders, NanMode::Min), 1);
    }

    #[test]
    fn test_quantize_uniform() {
        // Values from 0 to 10
        let values: Vec<f32> = (0..=100).map(|i| i as f32).collect();
        let borders = fit_borders(&values, 4, QuantizationMethod::Uniform, NanMode::Min);
        assert!(borders.len() <= 4);
        assert!(borders.windows(2).all(|w| w[0] < w[1]));

        // Test bin monotonic increase
        let binned = quantize_column(&values, &borders, NanMode::Min);
        assert!(binned.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(*binned.first().unwrap(), 0);
        assert_eq!(*binned.last().unwrap(), borders.len() as u8);
    }

    #[test]
    fn test_quantize_median_quantiles() {
        // Skewed distribution: mostly 0s, few large values
        let mut values = vec![0.0; 80];
        values.extend((1..=20).map(|i| i as f32 * 10.0));
        let borders = fit_borders(&values, 5, QuantizationMethod::Median, NanMode::Min);
        assert!(!borders.is_empty());
        assert!(borders.windows(2).all(|w| w[0] < w[1]));

        let binned = quantize_column(&values, &borders, NanMode::Min);
        assert!(binned.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_quantize_greedy_log_sum() {
        let values: Vec<f32> = (0..200).map(|i| (i as f32).sin() * 100.0).collect();
        let borders = fit_borders(&values, 32, QuantizationMethod::GreedyLogSum, NanMode::Min);
        assert!(!borders.is_empty());
        assert!(borders.len() <= 32);
        assert!(borders.windows(2).all(|w| w[0] < w[1]));

        let binned = quantize_column(&values, &borders, NanMode::Min);
        for &b in &binned {
            assert!((b as usize) <= borders.len());
        }
    }

    #[test]
    fn test_nan_handling() {
        let borders = vec![1.0, 2.0, 3.0];
        assert_eq!(quantize_value(f32::NAN, &borders, NanMode::Min), 0);
        assert_eq!(quantize_value(f32::NAN, &borders, NanMode::Max), 3);
    }

    #[test]
    fn test_row_and_col_major_equivalence() {
        let num_samples = 20;
        let num_features = 3;
        let mut raw_data = Vec::with_capacity(num_samples * num_features);
        for i in 0..num_samples {
            for f in 0..num_features {
                raw_data.push((i * 10 + f) as f32);
            }
        }

        let grid = QuantizationGrid::fit_row_major(
            &raw_data,
            num_samples,
            num_features,
            8,
            QuantizationMethod::Uniform,
            NanMode::Min,
        );
        let binned_row = grid.transform_row_major(&raw_data, num_samples);

        // Convert raw_data to column-major
        let mut raw_col = vec![0.0f32; num_samples * num_features];
        for i in 0..num_samples {
            for f in 0..num_features {
                raw_col[f * num_samples + i] = raw_data[i * num_features + f];
            }
        }

        let binned_col = grid.transform_col_major(&raw_col, num_samples);

        // Verify element-wise match
        for i in 0..num_samples {
            for f in 0..num_features {
                let row_val = binned_row[i * num_features + f];
                let col_val = binned_col[f * num_samples + i];
                assert_eq!(row_val, col_val);
            }
        }
    }

    #[test]
    fn test_consistency_with_continuous_splits() {
        // Crucial test: continuous sample[f] > continuous_threshold
        // must EXACTLY equal binned_sample[f] > bin_threshold
        let values: Vec<f32> = vec![0.1, 1.5, 2.3, 4.8, 5.0, 9.9, 12.0];
        let borders = fit_borders(&values, 4, QuantizationMethod::GreedyLogSum, NanMode::Min);

        for (split_bin_idx, &continuous_threshold) in borders.iter().enumerate() {
            let split_bin_u8 = split_bin_idx as u8;
            for &val in &[0.0, 0.1, 1.0, 1.5, 2.0, 2.3, 3.0, 4.8, 5.0, 6.0, 9.9, 10.0, 12.0, 15.0] {
                let continuous_branch = val > continuous_threshold;
                let binned = quantize_value(val, &borders, NanMode::Min);
                let binned_branch = binned > split_bin_u8;
                assert_eq!(
                    continuous_branch, binned_branch,
                    "Mismatch for val={}, threshold={}, bin_thresh={}, binned={}",
                    val, continuous_threshold, split_bin_u8, binned
                );
            }
        }
    }
}

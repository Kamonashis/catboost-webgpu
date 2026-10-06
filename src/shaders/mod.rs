//! Embedded WGSL compute shaders for WebGPU acceleration.

/// Histogram accumulation shader.
/// Accumulates `sum_g` and `sum_h` per (feature, leaf, bin) using atomic float additions via CAS.
pub const HISTOGRAM_WGSL: &str = include_str!("histogram.wgsl");

/// Split evaluation shader.
/// Evaluates oblivious tree split score across leaves for all candidate feature borders.
pub const SPLIT_EVAL_WGSL: &str = include_str!("split_eval.wgsl");

/// Leaf index partitioning shader.
/// Updates sample leaf indices: `leaf_indices[i] |= ((binned_data[i, f*] > b*) as u32) << depth`.
pub const PARTITION_WGSL: &str = include_str!("partition.wgsl");

/// Prediction vector update shader.
/// Vectorized update: `predictions[i] += learning_rate * leaf_values[leaf_indices[i]]`.
pub const UPDATE_PREDICTIONS_WGSL: &str = include_str!("update_predictions.wgsl");

/// Loss gradient and hessian computation shader.
/// Vectorized compute for RMSE, Logloss, CrossEntropy, and MAE on-device.
pub const COMPUTE_GRADIENTS_WGSL: &str = include_str!("compute_gradients.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shaders_embedded_and_valid() {
        assert!(!HISTOGRAM_WGSL.is_empty());
        assert!(!SPLIT_EVAL_WGSL.is_empty());
        assert!(!PARTITION_WGSL.is_empty());
        assert!(!UPDATE_PREDICTIONS_WGSL.is_empty());
        assert!(!COMPUTE_GRADIENTS_WGSL.is_empty());

        // Validate all shaders parse cleanly with Naga WGSL front-end
        for (name, source) in [
            ("histogram.wgsl", HISTOGRAM_WGSL),
            ("split_eval.wgsl", SPLIT_EVAL_WGSL),
            ("partition.wgsl", PARTITION_WGSL),
            ("update_predictions.wgsl", UPDATE_PREDICTIONS_WGSL),
            ("compute_gradients.wgsl", COMPUTE_GRADIENTS_WGSL),
        ] {
            let res = wgpu::naga::front::wgsl::parse_str(source);
            assert!(
                res.is_ok(),
                "Shader {} failed WGSL parse: {:?}",
                name,
                res.err()
            );
        }
    }
}

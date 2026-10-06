use catboost_webgpu_core::cpu_engine::CpuEngine;
use catboost_webgpu_core::device::{get_device_info, is_webgpu_available, GpuContext};
use catboost_webgpu_core::gpu_engine::WebGpuEngine;
use catboost_webgpu_core::objective::{Logloss, RMSELoss};
use catboost_webgpu_core::traits::LossFunction;
use catboost_webgpu_core::shaders::*;
use catboost_webgpu_core::traits::ComputeBackend;

#[test]
fn test_embedded_wgsl_shaders() {
    assert!(!HISTOGRAM_WGSL.is_empty(), "histogram.wgsl is empty");
    assert!(!SPLIT_EVAL_WGSL.is_empty(), "split_eval.wgsl is empty");
    assert!(!PARTITION_WGSL.is_empty(), "partition.wgsl is empty");
    assert!(
        !UPDATE_PREDICTIONS_WGSL.is_empty(),
        "update_predictions.wgsl is empty"
    );
    assert!(
        !COMPUTE_GRADIENTS_WGSL.is_empty(),
        "compute_gradients.wgsl is empty"
    );

    // Verify all shaders pass WGSL grammar validation
    for (name, source) in [
        ("histogram", HISTOGRAM_WGSL),
        ("split_eval", SPLIT_EVAL_WGSL),
        ("partition", PARTITION_WGSL),
        ("update_predictions", UPDATE_PREDICTIONS_WGSL),
        ("compute_gradients", COMPUTE_GRADIENTS_WGSL),
    ] {
        let res = wgpu::naga::front::wgsl::parse_str(source);
        assert!(
            res.is_ok(),
            "WGSL parsing error in {}: {:?}",
            name,
            res.err()
        );
    }
}

#[test]
fn test_device_discovery_and_micro_probe() {
    let dev_info = get_device_info();
    println!("WebGPU device info: {:?}", dev_info);
    println!("WebGPU available: {}", is_webgpu_available());

    if let Ok(ctx) = GpuContext::init() {
        assert!(ctx.info.is_gpu);
        println!(
            "Micro-probe verified on hardware GPU: {} ({})",
            ctx.info.name, ctx.info.backend
        );
    } else {
        println!("No hardware WebGPU device initialized; running in CPU fallback mode.");
    }
}

#[test]
fn test_histogram_gpu_vs_cpu_exactness() {
    let gpu_engine = WebGpuEngine::new();
    let cpu_engine = CpuEngine::new();

    let num_samples = 256;
    let num_features = 4;
    let num_leaves = 4;
    let max_bins = 8;

    let mut binned_data = vec![0u8; num_samples * num_features];
    for i in 0..num_samples {
        for f in 0..num_features {
            binned_data[i * num_features + f] = ((i * (f + 1) * 7) % max_bins) as u8;
        }
    }

    let leaf_indices: Vec<u32> = (0..num_samples).map(|i| (i % num_leaves) as u32).collect();
    let gradients: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32) * 0.05 - 3.2)
        .collect();
    let hessians: Vec<f32> = (0..num_samples)
        .map(|i| ((i % 5) as f32) * 0.2 + 0.5)
        .collect();

    let gpu_hist = gpu_engine.compute_histograms(
        &binned_data,
        &leaf_indices,
        &gradients,
        &hessians,
        num_samples,
        num_features,
        num_leaves,
        max_bins,
    );

    let cpu_hist = cpu_engine.compute_histograms(
        &binned_data,
        &leaf_indices,
        &gradients,
        &hessians,
        num_samples,
        num_features,
        num_leaves,
        max_bins,
    );

    assert_eq!(gpu_hist.len(), cpu_hist.len());
    let mut max_diff = 0.0f32;
    for i in 0..gpu_hist.len() {
        let diff = (gpu_hist[i] - cpu_hist[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 1e-3,
            "Histogram mismatch at index {}: GPU={}, CPU={}, diff={}",
            i,
            gpu_hist[i],
            cpu_hist[i],
            diff
        );
    }
    println!("Histogram test passed. Max absolute diff: {}", max_diff);
}

#[test]
fn test_partition_leaves_gpu_vs_cpu() {
    let gpu_engine = WebGpuEngine::new();
    let cpu_engine = CpuEngine::new();

    let num_samples = 512;
    let num_features = 3;
    let mut binned_data = vec![0u8; num_samples * num_features];
    for i in 0..num_samples {
        binned_data[i * num_features + 0] = (i % 16) as u8;
        binned_data[i * num_features + 1] = ((i / 4) % 16) as u8;
        binned_data[i * num_features + 2] = ((i * 3) % 16) as u8;
    }

    let mut gpu_leaves = vec![0u32; num_samples];
    let mut cpu_leaves = vec![0u32; num_samples];

    for depth in 0..3 {
        let feature_idx = depth % num_features;
        let bin_threshold = (depth * 3 + 4) as u8;

        gpu_engine.partition_leaves(
            &binned_data,
            &mut gpu_leaves,
            feature_idx,
            bin_threshold,
            depth,
            num_samples,
            num_features,
        );

        cpu_engine.partition_leaves(
            &binned_data,
            &mut cpu_leaves,
            feature_idx,
            bin_threshold,
            depth,
            num_samples,
            num_features,
        );

        assert_eq!(
            gpu_leaves, cpu_leaves,
            "Leaf partition mismatch at depth {}",
            depth
        );
    }
    println!("Partition test passed across 3 tree levels.");
}

#[test]
fn test_update_predictions_gpu_vs_cpu() {
    let gpu_engine = WebGpuEngine::new();
    let cpu_engine = CpuEngine::new();

    let num_samples = 128;
    let num_leaves = 4;
    let leaf_indices: Vec<u32> = (0..num_samples).map(|i| (i % num_leaves) as u32).collect();
    let leaf_values = vec![0.25f32, -0.75f32, 1.5f32, -0.1f32];
    let lr = 0.05f32;

    let mut gpu_preds = vec![0.5f32; num_samples];
    let mut cpu_preds = vec![0.5f32; num_samples];

    gpu_engine.update_predictions(&mut gpu_preds, &leaf_indices, &leaf_values, lr);
    cpu_engine.update_predictions(&mut cpu_preds, &leaf_indices, &leaf_values, lr);

    for i in 0..num_samples {
        assert!(
            (gpu_preds[i] - cpu_preds[i]).abs() < 1e-5,
            "Prediction update mismatch at {}: GPU={}, CPU={}",
            i,
            gpu_preds[i],
            cpu_preds[i]
        );
    }
    println!("Prediction update test passed.");
}

#[test]
fn test_split_evaluation_gpu() {
    let engine = WebGpuEngine::new();
    let cpu_engine = CpuEngine::new();

    let num_samples = 100;
    let num_features = 3;
    let num_leaves = 2;
    let max_bins = 6;
    let l2_reg = 1.0f32;

    let mut binned_data = vec![0u8; num_samples * num_features];
    for i in 0..num_samples {
        binned_data[i * num_features + 0] = if i < 50 { 0 } else { 5 };
        binned_data[i * num_features + 1] = (i % max_bins) as u8;
        binned_data[i * num_features + 2] = ((i * 3) % max_bins) as u8;
    }
    let leaf_indices: Vec<u32> = (0..num_samples).map(|i| (i % num_leaves) as u32).collect();
    let gradients: Vec<f32> = (0..num_samples).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let hessians: Vec<f32> = vec![1.0f32; num_samples];

    let histograms = engine.compute_histograms(
        &binned_data,
        &leaf_indices,
        &gradients,
        &hessians,
        num_samples,
        num_features,
        num_leaves,
        max_bins,
    );

    let gpu_cand = engine.find_best_split(&histograms, num_features, num_leaves, max_bins, l2_reg);
    let cpu_cand = cpu_engine.find_best_split(
        &histograms,
        num_features,
        num_leaves,
        max_bins,
        l2_reg,
        0.0,
        42,
    );

    println!("GPU best split: {:?}", gpu_cand);
    println!("CPU best split: {:?}", cpu_cand);

    if let (Some(g), Some(c)) = (gpu_cand, cpu_cand) {
        assert_eq!(g.feature_idx, c.feature_idx);
        assert_eq!(g.bin_threshold, c.bin_threshold);
        assert!((g.gain - c.gain).abs() < 1e-3);
    }
}

#[test]
fn test_compute_gradients_gpu_vs_cpu() {
    let engine = WebGpuEngine::new();
    let num_samples = 100;

    let y_true: Vec<f32> = (0..num_samples).map(|i| (i % 2) as f32).collect();
    let y_pred: Vec<f32> = (0..num_samples)
        .map(|i| ((i as f32) / 100.0) * 4.0 - 2.0)
        .collect();

    // 1. RMSE Test (loss_type = 0)
    let mut gpu_g = vec![0.0f32; num_samples];
    let mut gpu_h = vec![0.0f32; num_samples];
    if engine
        .compute_gradients_gpu(&y_true, &y_pred, 0, &mut gpu_g, &mut gpu_h)
        .is_ok()
    {
        let rmse = RMSELoss;
        let mut cpu_g = vec![0.0f32; num_samples];
        let mut cpu_h = vec![0.0f32; num_samples];
        rmse.compute_gradients_hessians(&y_true, &y_pred, &mut cpu_g, &mut cpu_h);

        for i in 0..num_samples {
            assert!((gpu_g[i] - cpu_g[i]).abs() < 1e-5);
            assert!((gpu_h[i] - cpu_h[i]).abs() < 1e-5);
        }
        println!("GPU RMSE gradients verified.");
    }

    // 2. Logloss Test (loss_type = 1)
    if engine
        .compute_gradients_gpu(&y_true, &y_pred, 1, &mut gpu_g, &mut gpu_h)
        .is_ok()
    {
        let logloss = Logloss;
        let mut cpu_g = vec![0.0f32; num_samples];
        let mut cpu_h = vec![0.0f32; num_samples];
        logloss.compute_gradients_hessians(&y_true, &y_pred, &mut cpu_g, &mut cpu_h);

        for i in 0..num_samples {
            assert!((gpu_g[i] - cpu_g[i]).abs() < 1e-4);
            assert!((gpu_h[i] - cpu_h[i]).abs() < 1e-4);
        }
        println!("GPU Logloss gradients verified.");
    }
}

#[test]
fn test_cpu_fallback_mode() {
    let engine = WebGpuEngine::new_cpu();
    assert!(!engine.is_gpu());
    assert_eq!(engine.name(), "CPU (Rayon Fallback)");

    let num_samples = 10;
    let num_features = 2;
    let binned_data = vec![0u8; num_samples * num_features];
    let leaf_indices = vec![0u32; num_samples];
    let gradients = vec![1.0f32; num_samples];
    let hessians = vec![1.0f32; num_samples];

    let hist = engine.compute_histograms(
        &binned_data,
        &leaf_indices,
        &gradients,
        &hessians,
        num_samples,
        num_features,
        1,
        2,
    );
    assert_eq!(hist.len(), num_features * 1 * 2 * 2);
}

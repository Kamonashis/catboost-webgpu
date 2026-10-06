use catboost_webgpu_core::boosting::{
    BaggingType, BoostingConfig, BoostingEngine, BoostingType,
};
use catboost_webgpu_core::cpu_engine::CpuEngine;
use catboost_webgpu_core::importance::{
    ensemble_tree_shap_batch, feature_importance_loss_function_change,
    feature_importance_prediction_values_change, tree_shap_single,
};
use catboost_webgpu_core::model::CatBoostModel;
use catboost_webgpu_core::objective::*;
use catboost_webgpu_core::traits::{ComputeBackend, LossFunction, ObliviousTree, SplitCondition, SplitType};

// ============================================================================
// 1. Objectives Tests
// ============================================================================

#[test]
fn test_rmse_objective() {
    let loss = RMSELoss;
    let (g, h) = loss.gradient_hessian(3.0, 5.0);
    assert_eq!(g, 2.0); // y_pred - y_true
    assert_eq!(h, 1.0);

    let y_true = vec![1.0, 2.0, 3.0];
    let y_pred = vec![1.0, 2.0, 5.0];
    let metric = loss.evaluate_metric(&y_true, &y_pred);
    // diffs: 0, 0, 2 -> sqrt(4 / 3) = 1.1547
    assert!((metric - (4.0f32 / 3.0).sqrt()).abs() < 1e-5);
    assert!(loss.lower_is_better());
}

#[test]
fn test_mae_objective() {
    let loss = MAELoss;
    let (g1, h1) = loss.gradient_hessian(2.0, 5.0);
    assert_eq!(g1, 1.0);
    assert_eq!(h1, 1.0);

    let (g2, _) = loss.gradient_hessian(5.0, 2.0);
    assert_eq!(g2, -1.0);

    let y_true = vec![1.0, 2.0, 3.0];
    let y_pred = vec![2.0, 1.0, 5.0];
    let metric = loss.evaluate_metric(&y_true, &y_pred);
    // |1| + |1| + |2| = 4 / 3
    assert!((metric - 4.0 / 3.0).abs() < 1e-5);
}

#[test]
fn test_mape_objective() {
    let loss = MAPELoss;
    let y_true = vec![100.0, 200.0];
    let y_pred = vec![110.0, 180.0];
    // error: 10/100 = 10%, 20/200 = 10% -> 10%
    let metric = loss.evaluate_metric(&y_true, &y_pred);
    assert!((metric - 10.0).abs() < 1e-4);
}

#[test]
fn test_huber_objective() {
    let loss = HuberLoss::new(2.0);
    // Within delta=2.0
    let (g1, h1) = loss.gradient_hessian(10.0, 11.0);
    assert_eq!(g1, 1.0);
    assert_eq!(h1, 1.0);

    // Outside delta=2.0 (diff = 5.0 > 2.0)
    let (g2, h2) = loss.gradient_hessian(10.0, 15.0);
    assert_eq!(g2, 2.0); // delta * signum
    assert_eq!(h2, 1.0);
}

#[test]
fn test_quantile_objective() {
    let loss_median = QuantileLoss::new(0.5);
    let (g_over, _) = loss_median.gradient_hessian(10.0, 12.0);
    assert_eq!(g_over, 0.5); // 1 - 0.5
    let (g_under, _) = loss_median.gradient_hessian(10.0, 8.0);
    assert_eq!(g_under, -0.5); // -alpha

    let loss_90 = QuantileLoss::new(0.9);
    let (g_under_90, _) = loss_90.gradient_hessian(10.0, 8.0);
    assert_eq!(g_under_90, -0.9);
}

#[test]
fn test_poisson_objective() {
    let loss = PoissonLoss;
    let y_true = 2.0;
    let y_pred = 0.0; // lambda = 1.0
    let (g, h) = loss.gradient_hessian(y_true, y_pred);
    assert_eq!(g, -1.0); // 1.0 - 2.0
    assert_eq!(h, 1.0);
}

#[test]
fn test_logloss_objective() {
    let loss = Logloss;
    let (g0, _) = loss.gradient_hessian(1.0, 0.0);
    // sigmoid(0) = 0.5 -> g = 0.5 - 1.0 = -0.5
    assert!((g0 - (-0.5)).abs() < 1e-5);

    let y_true = vec![1.0, 0.0];
    let y_pred = vec![0.0, 0.0];
    let metric = loss.evaluate_metric(&y_true, &y_pred);
    // -ln(0.5) = 0.693147
    assert!((metric - 0.693147).abs() < 1e-4);
}

#[test]
fn test_multiclass_objective() {
    let loss = MultiClassLoss::new(3);
    let y_true = vec![0.0, 1.0];
    // Sample 0: logits [2.0, 0.0, 0.0], true = 0
    // Sample 1: logits [0.0, 2.0, 0.0], true = 1
    let y_pred = vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
    let mut grads = vec![0.0; 6];
    let mut hess = vec![0.0; 6];
    loss.compute_gradients_hessians(&y_true, &y_pred, &mut grads, &mut hess);

    // Class 0 for sample 0 should have negative gradient (predicted high and true=0)
    assert!(grads[0] < 0.0);
    // Class 1 for sample 0 should have positive gradient
    assert!(grads[1] > 0.0);

    let metric = loss.evaluate_metric(&y_true, &y_pred);
    assert!(metric > 0.0 && metric < 1.0);
}

#[test]
fn test_ranking_pairlogit_and_queryrmse() {
    let pair_loss = PairLogit::new(vec![3]);
    let y_true = vec![3.0, 2.0, 1.0]; // Perfect ranking order
    let y_pred_bad = vec![1.0, 2.0, 3.0]; // Inverted predictions
    let mut grads = vec![0.0; 3];
    let mut hess = vec![0.0; 3];
    pair_loss.compute_gradients_hessians(&y_true, &y_pred_bad, &mut grads, &mut hess);
    // Top sample (y=3) predicted lower than others -> should have strong negative gradient to increase
    assert!(grads[0] < 0.0);
    // Bottom sample (y=1) predicted higher than others -> should have positive gradient to decrease
    assert!(grads[2] > 0.0);

    let query_rmse = QueryRMSE::new(vec![3]);
    let mut q_grads = vec![0.0; 3];
    let mut q_hess = vec![0.0; 3];
    query_rmse.compute_gradients_hessians(&y_true, &y_pred_bad, &mut q_grads, &mut q_hess);
    assert_eq!(q_grads.len(), 3);
}

#[test]
fn test_custom_objective() {
    let custom = CustomObjective::new(
        "my_abs",
        |yt, yp, g, h| {
            for i in 0..yt.len() {
                g[i] = (yp[i] - yt[i]).signum();
                h[i] = 1.0;
            }
        },
        |yt, yp| {
            yt.iter().zip(yp.iter()).map(|(&t, &p)| (p - t).abs()).sum::<f32>() / yt.len() as f32
        },
        true,
    );

    assert_eq!(custom.name(), "my_abs");
    let (g, h) = custom.gradient_hessian(10.0, 15.0);
    assert_eq!(g, 1.0);
    assert_eq!(h, 1.0);
}

// ============================================================================
// 2. CPU Engine Tests
// ============================================================================

#[test]
fn test_cpu_engine_histograms_match_sequential() {
    let engine = CpuEngine::new();
    let num_samples = 1000;
    let num_features = 8;
    let num_leaves = 4;
    let max_bins = 16;

    // Generate deterministic binned data
    let mut binned_data = vec![0u8; num_samples * num_features];
    let mut leaf_indices = vec![0u32; num_samples];
    let mut grads = vec![0.0f32; num_samples];
    let mut hess = vec![0.0f32; num_samples];

    for i in 0..num_samples {
        leaf_indices[i] = (i % num_leaves) as u32;
        grads[i] = (i as f32 * 0.01).sin();
        hess[i] = 1.0 + (i as f32 * 0.02).cos().abs();
        for f in 0..num_features {
            binned_data[i * num_features + f] = ((i * 7 + f * 13) % max_bins) as u8;
        }
    }

    // Parallel histograms via CpuEngine
    let par_hists = engine.compute_histograms(
        &binned_data,
        &leaf_indices,
        &grads,
        &hess,
        num_samples,
        num_features,
        num_leaves,
        max_bins,
    );

    // Compute reference sequential histograms
    let mut ref_hists = vec![0.0f32; num_features * num_leaves * max_bins * 2];
    for i in 0..num_samples {
        let leaf = leaf_indices[i] as usize;
        let g = grads[i];
        let h = hess[i];
        for f in 0..num_features {
            let bin = binned_data[i * num_features + f] as usize;
            let idx = (f * num_leaves + leaf) * max_bins * 2 + bin * 2;
            ref_hists[idx] += g;
            ref_hists[idx + 1] += h;
        }
    }

    assert_eq!(par_hists.len(), ref_hists.len());
    for (idx, (&p, &r)) in par_hists.iter().zip(ref_hists.iter()).enumerate() {
        assert!(
            (p - r).abs() < 1e-4,
            "Histogram mismatch at index {}: par={}, ref={}",
            idx,
            p,
            r
        );
    }
}

#[test]
fn test_cpu_engine_partition_and_update() {
    let engine = CpuEngine::new();
    let num_samples = 100;
    let num_features = 3;

    let mut binned_data = vec![0u8; num_samples * num_features];
    for i in 0..num_samples {
        binned_data[i * num_features + 1] = if i >= 50 { 10 } else { 2 };
    }

    let mut leaf_indices = vec![0u32; num_samples];
    // Split feature 1 at threshold 5, depth 0
    engine.partition_leaves(&binned_data, &mut leaf_indices, 1, 5, 0, num_samples, num_features);

    for i in 0..num_samples {
        let expected = if i >= 50 { 1 } else { 0 };
        assert_eq!(leaf_indices[i], expected);
    }

    // Test update_predictions
    let mut predictions = vec![0.0f32; num_samples];
    let leaf_values = vec![1.5f32, -2.5f32];
    engine.update_predictions(&mut predictions, &leaf_indices, &leaf_values, 0.1);

    for i in 0..num_samples {
        let expected = if i >= 50 { 0.1 * -2.5 } else { 0.1 * 1.5 };
        assert!((predictions[i] - expected).abs() < 1e-6);
    }
}

// ============================================================================
// 3. Boosting Engine Tests (Plain & Ordered, Bagging, Early Stopping)
// ============================================================================

#[test]
fn test_plain_boosting_regression_convergence() {
    // Generate synthetic regression: y = 2 * (x0 > 50) - 3 * (x1 > 100)
    let num_samples = 400;
    let num_features = 4;
    let mut binned_data = vec![0u8; num_samples * num_features];
    let mut targets = vec![0.0f32; num_samples];

    for i in 0..num_samples {
        let x0 = (i % 100) as u8;
        let x1 = ((i * 3) % 200) as u8;
        let x2 = ((i * 7) % 250) as u8;
        let x3 = ((i * 11) % 250) as u8;
        binned_data[i * num_features + 0] = x0;
        binned_data[i * num_features + 1] = x1;
        binned_data[i * num_features + 2] = x2;
        binned_data[i * num_features + 3] = x3;

        let y = if x0 > 50 { 2.0 } else { -1.0 } + if x1 > 100 { -3.0 } else { 1.5 };
        targets[i] = y;
    }

    let config = BoostingConfig {
        iterations: 30,
        learning_rate: 0.2,
        depth: 3,
        l2_leaf_reg: 1.0,
        random_strength: 0.0,
        bagging_type: BaggingType::None,
        boosting_type: BoostingType::Plain,
        max_bins: 254,
        early_stopping_rounds: None,
        use_best_model: true,
        seed: 42,
        verbose: 0,
    };

    let engine = BoostingEngine::new(config);
    let loss = RMSELoss;
    let ensemble = engine.fit(
        &binned_data,
        &targets,
        num_samples,
        num_features,
        None,
        None,
        None,
        None,
        &loss,
    );

    assert_eq!(ensemble.trees.len(), 30);

    // Compute predictions with model
    let model = CatBoostModel::new(
        ensemble.trees,
        ensemble.learning_rate,
        ensemble.base_score,
        vec!["f0".into(), "f1".into(), "f2".into(), "f3".into()],
        vec![vec![]; 4],
        "RMSE".into(),
    );

    let preds = model.predict_binned_batch(&binned_data, num_samples);
    let final_rmse = loss.evaluate_metric(&targets, &preds);
    let initial_rmse = loss.evaluate_metric(&targets, &vec![0.0; num_samples]);

    // Ensure significant error reduction
    assert!(
        final_rmse < initial_rmse * 0.4,
        "Final RMSE ({}) should be much lower than initial ({})",
        final_rmse,
        initial_rmse
    );
}

#[test]
fn test_ordered_boosting_convergence() {
    let num_samples = 300;
    let num_features = 3;
    let mut binned_data = vec![0u8; num_samples * num_features];
    let mut targets = vec![0.0f32; num_samples];

    for i in 0..num_samples {
        let x0 = (i % 120) as u8;
        let x1 = ((i * 5) % 180) as u8;
        binned_data[i * num_features + 0] = x0;
        binned_data[i * num_features + 1] = x1;
        binned_data[i * num_features + 2] = (i % 50) as u8;

        targets[i] = if x0 > 60 { 4.0 } else { -2.0 };
    }

    let config = BoostingConfig {
        iterations: 25,
        learning_rate: 0.15,
        depth: 2,
        l2_leaf_reg: 2.0,
        random_strength: 0.1,
        bagging_type: BaggingType::Bayesian { bagging_temperature: 0.5 },
        boosting_type: BoostingType::Ordered { num_permutations: 2 },
        max_bins: 254,
        early_stopping_rounds: None,
        use_best_model: true,
        seed: 123,
        verbose: 0,
    };

    let engine = BoostingEngine::new(config);
    let loss = RMSELoss;
    let ensemble = engine.fit(
        &binned_data,
        &targets,
        num_samples,
        num_features,
        None,
        None,
        None,
        None,
        &loss,
    );

    assert_eq!(ensemble.trees.len(), 25);
    let model = CatBoostModel::new(
        ensemble.trees,
        ensemble.learning_rate,
        ensemble.base_score,
        vec!["f0".into(), "f1".into(), "f2".into()],
        vec![vec![]; 3],
        "RMSE".into(),
    );
    let preds = model.predict_binned_batch(&binned_data, num_samples);
    let rmse = loss.evaluate_metric(&targets, &preds);
    assert!(rmse < 2.0, "Ordered boosting RMSE = {} should converge", rmse);
}

#[test]
fn test_bagging_modes_and_early_stopping() {
    let num_samples = 200;
    let num_features = 2;
    let binned_data = vec![50u8; num_samples * num_features];
    let targets = vec![1.0f32; num_samples];

    // Eval dataset with diverging targets so eval error degrades and early stops
    let eval_binned = vec![50u8; 50 * num_features];
    let eval_targets = vec![-10.0f32; 50];

    // Test with early stopping after 3 rounds
    let config = BoostingConfig {
        iterations: 50,
        learning_rate: 0.1,
        depth: 2,
        l2_leaf_reg: 1.0,
        random_strength: 0.0,
        bagging_type: BaggingType::MVS { subsample: 0.8 },
        boosting_type: BoostingType::Plain,
        max_bins: 254,
        early_stopping_rounds: Some(3),
        use_best_model: true,
        seed: 777,
        verbose: 0,
    };

    let engine = BoostingEngine::new(config);
    let loss = RMSELoss;
    let ensemble = engine.fit(
        &binned_data,
        &targets,
        num_samples,
        num_features,
        None,
        Some(&eval_binned),
        Some(&eval_targets),
        Some(50),
        &loss,
    );

    // Early stopping should have triggered well before 50 iterations
    assert!(
        ensemble.trees.len() < 50,
        "Early stopping should terminate iterations earlier, got {}",
        ensemble.trees.len()
    );
    assert!(ensemble.best_score.is_some());
}

// ============================================================================
// 4. Feature Importance & Fast Oblivious Tree SHAP Tests
// ============================================================================

#[test]
fn test_tree_shap_efficiency_and_dummy_axiom() {
    // Construct an oblivious tree of depth 3 testing features 0 and 1 only
    let splits = vec![
        SplitCondition {
            feature_idx: 0,
            bin_threshold: 5,
            continuous_threshold: 0.5,
            split_type: SplitType::Numerical,
        },
        SplitCondition {
            feature_idx: 1,
            bin_threshold: 10,
            continuous_threshold: 2.5,
            split_type: SplitType::Numerical,
        },
        SplitCondition {
            feature_idx: 0,
            bin_threshold: 15,
            continuous_threshold: 1.2,
            split_type: SplitType::Numerical,
        },
    ];

    let leaf_values = vec![1.0, 3.0, 5.0, 7.0, -2.0, 4.0, -6.0, 8.0];
    let tree = ObliviousTree::new(3, splits, leaf_values);

    let num_features = 4; // Feature 2 and 3 are dummy features!
    let sample = vec![0.8, 3.0, 99.0, -50.0];

    let tree_pred = tree.predict_continuous(&sample);
    let shap = tree_shap_single(&tree, &sample, num_features);

    // 1. Efficiency axiom: base_value + sum(shap) == tree_pred
    let sum_shap: f32 = shap.values.iter().sum();
    let reconstructed = shap.base_value + sum_shap;
    assert!(
        (reconstructed - tree_pred).abs() < 1e-5,
        "Efficiency axiom failed: pred={}, reconstructed={} (base={}, sum={})",
        tree_pred,
        reconstructed,
        shap.base_value,
        sum_shap
    );

    // 2. Dummy feature axiom: features 2 and 3 are not in any split -> must be exactly 0.0
    assert_eq!(
        shap.values[2], 0.0,
        "Dummy feature 2 must have 0 SHAP, got {}",
        shap.values[2]
    );
    assert_eq!(
        shap.values[3], 0.0,
        "Dummy feature 3 must have 0 SHAP, got {}",
        shap.values[3]
    );
}

#[test]
fn test_ensemble_tree_shap_batch_efficiency() {
    let t1 = ObliviousTree::new(
        2,
        vec![
            SplitCondition {
                feature_idx: 0,
                bin_threshold: 1,
                continuous_threshold: 10.0,
                split_type: SplitType::Numerical,
            },
            SplitCondition {
                feature_idx: 1,
                bin_threshold: 2,
                continuous_threshold: 20.0,
                split_type: SplitType::Numerical,
            },
        ],
        vec![0.5, 1.5, -0.5, 2.5],
    );

    let t2 = ObliviousTree::new(
        1,
        vec![SplitCondition {
            feature_idx: 0,
            bin_threshold: 3,
            continuous_threshold: 5.0,
            split_type: SplitType::Numerical,
        }],
        vec![-1.0, 1.0],
    );

    let trees = vec![t1, t2];
    let base_score = 0.2;
    let num_features = 3;
    let samples = vec![
        12.0, 25.0, 0.0,  // sample 0
        2.0, 10.0, 5.0,   // sample 1
        8.0, 30.0, 100.0, // sample 2
    ];

    let (base_val, shap_matrix) = ensemble_tree_shap_batch(&trees, base_score, &samples, 3, num_features);

    for i in 0..3 {
        let sample = &samples[i * num_features..(i + 1) * num_features];
        let true_pred: f32 = base_score + trees.iter().map(|t| t.predict_continuous(sample)).sum::<f32>();
        let shap_sum: f32 = shap_matrix[i].iter().sum();
        let reconstructed = base_val + shap_sum;
        assert!(
            (reconstructed - true_pred).abs() < 1e-5,
            "Batch efficiency failed for sample {}: pred={}, reconstructed={}",
            i,
            true_pred,
            reconstructed
        );
    }
}

#[test]
fn test_feature_importance_methods() {
    let splits = vec![
        SplitCondition {
            feature_idx: 0,
            bin_threshold: 1,
            continuous_threshold: 0.0,
            split_type: SplitType::Numerical,
        },
        SplitCondition {
            feature_idx: 1,
            bin_threshold: 1,
            continuous_threshold: 0.0,
            split_type: SplitType::Numerical,
        },
    ];
    let tree = ObliviousTree::new(2, splits, vec![10.0, 20.0, 10.0, 20.0]);
    let trees = vec![tree];

    let imp_pred = feature_importance_prediction_values_change(&trees, 3);
    assert_eq!(imp_pred.len(), 3);
    let total: f32 = imp_pred.iter().sum();
    assert!((total - 100.0).abs() < 1e-4);

    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let targets = vec![1.0, 2.0, 3.0];
    let imp_loss = feature_importance_loss_function_change(
        &trees,
        0.0,
        &data,
        &targets,
        3,
        3,
        &RMSELoss,
        42,
    );
    assert_eq!(imp_loss.len(), 3);
}

// ============================================================================
// 5. Model Serialization & Standalone Python Code Generator Tests
// ============================================================================

#[test]
fn test_model_json_and_binary_persistence() {
    let t1 = ObliviousTree::new(
        1,
        vec![SplitCondition {
            feature_idx: 0,
            bin_threshold: 5,
            continuous_threshold: 2.5,
            split_type: SplitType::Numerical,
        }],
        vec![1.2, -3.4],
    );

    let model = CatBoostModel::new(
        vec![t1],
        0.1,
        0.5,
        vec!["feature_a".into(), "feature_b".into()],
        vec![vec![1.0, 2.5, 5.0], vec![0.5, 1.5]],
        "RMSE".into(),
    );

    let json_path = "/tmp/test_catboost_model.json";
    let cbm_path = "/tmp/test_catboost_model.cbm";

    // 1. JSON roundtrip
    model.save_model(json_path, "json").unwrap();
    let loaded_json = CatBoostModel::load_model(json_path, "json").unwrap();

    let sample = vec![3.0, 1.0];
    assert_eq!(model.predict(&sample), loaded_json.predict(&sample));
    assert_eq!(model.feature_names, loaded_json.feature_names);

    // 2. Binary CBM roundtrip
    model.save_model(cbm_path, "cbm").unwrap();
    let loaded_cbm = CatBoostModel::load_model(cbm_path, "cbm").unwrap();

    assert_eq!(model.predict(&sample), loaded_cbm.predict(&sample));
    assert_eq!(model.feature_names, loaded_cbm.feature_names);
    assert_eq!(model.trees.len(), loaded_cbm.trees.len());
    assert_eq!(model.trees[0].leaf_values, loaded_cbm.trees[0].leaf_values);

    // Cleanup
    let _ = std::fs::remove_file(json_path);
    let _ = std::fs::remove_file(cbm_path);
}

#[test]
fn test_python_code_generator_execution() {
    let t1 = ObliviousTree::new(
        2,
        vec![
            SplitCondition {
                feature_idx: 0,
                bin_threshold: 2,
                continuous_threshold: 1.5,
                split_type: SplitType::Numerical,
            },
            SplitCondition {
                feature_idx: 1,
                bin_threshold: 4,
                continuous_threshold: 3.5,
                split_type: SplitType::Numerical,
            },
        ],
        vec![10.0, 20.0, 30.0, 40.0],
    );

    let model = CatBoostModel::new(
        vec![t1],
        0.5,
        1.0,
        vec!["col_x".into(), "col_y".into()],
        vec![vec![1.5], vec![3.5]],
        "RMSE".into(),
    );

    let py_path = "/tmp/test_exported_predictor.py";
    model.export_python(py_path).unwrap();

    // Verify Python file exists
    assert!(std::path::Path::new(py_path).exists());

    // Execute Python script to verify syntax and prediction matches Rust
    let test_sample = vec![2.0, 5.0]; // sample[0] > 1.5 (bit 0 = 1), sample[1] > 3.5 (bit 1 = 1) -> leaf 3 (value 40.0)
    let rust_pred = model.predict(&test_sample); // 1.0 + 0.5 * 40.0 = 21.0

    let py_eval_script = format!(
        "import sys\nsys.path.append('/tmp')\nfrom test_exported_predictor import CatBoostPredictor\npred = CatBoostPredictor.predict([2.0, 5.0])\nassert abs(pred - {:.6}) < 1e-4, f'Mismatch: {{pred}} vs {:.6}'\nprint('SUCCESS')",
        rust_pred, rust_pred
    );

    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&py_eval_script)
        .output()
        .expect("Failed to execute python3");

    assert!(
        output.status.success(),
        "Python script failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let _ = std::fs::remove_file(py_path);
}

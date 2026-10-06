use catboost_webgpu_core::categorical::{
    CtrConfig, CtrEncoder, FeaturePair, PairCtrEncoder,
};
use catboost_webgpu_core::dataset::DatasetBuilder;
use catboost_webgpu_core::quantization::{
    fit_borders, quantize_column, NanMode, QuantizationGrid, QuantizationMethod,
    MAX_BORDERS_COUNT,
};
use catboost_webgpu_core::traits::{ObliviousTree, SplitCondition, SplitType};
use catboost_webgpu_core::tree::{
    compute_leaf_index_binned, compute_leaf_index_continuous, find_best_split,
};

// ============================================================================
// 1. Quantization Algorithm Tests
// ============================================================================

#[test]
fn test_all_quantization_methods_produce_strictly_monotonic_borders() {
    let mut values = Vec::new();
    // Non-linear distribution with clusters and repeats
    for i in 0..500 {
        let x = (i as f32 * 0.05).sin() * 50.0 + (i as f32 * 0.1);
        values.push(x);
    }

    let methods = [
        QuantizationMethod::Uniform,
        QuantizationMethod::Median,
        QuantizationMethod::UniformAndQuantiles,
        QuantizationMethod::GreedyLogSum,
    ];

    for &method in &methods {
        let borders = fit_borders(&values, 64, method, NanMode::Min);
        assert!(
            !borders.is_empty(),
            "Method {:?} produced empty borders",
            method
        );
        assert!(
            borders.len() <= 64,
            "Method {:?} produced {} borders > 64",
            method,
            borders.len()
        );
        // Verify strictly increasing
        for w in borders.windows(2) {
            assert!(
                w[0] < w[1],
                "Method {:?} produced non-strictly increasing borders: {} >= {}",
                method,
                w[0],
                w[1]
            );
        }

        // Verify all binned values fall in 0..=borders.len()
        let binned = quantize_column(&values, &borders, NanMode::Min);
        for &bin in &binned {
            assert!(
                (bin as usize) <= borders.len(),
                "Bin {} exceeds borders count {}",
                bin,
                borders.len()
            );
        }
    }
}

#[test]
fn test_quantization_up_to_254_borders() {
    // 1000 unique values
    let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.25).collect();
    let borders = fit_borders(&values, MAX_BORDERS_COUNT, QuantizationMethod::GreedyLogSum, NanMode::Min);

    assert_eq!(borders.len(), MAX_BORDERS_COUNT);
    assert_eq!(borders.len(), 254);

    let binned = quantize_column(&values, &borders, NanMode::Min);
    // Bins range from 0 to 254 (255 distinct bins)
    let min_bin = *binned.iter().min().unwrap();
    let max_bin = *binned.iter().max().unwrap();
    assert_eq!(min_bin, 0);
    assert_eq!(max_bin, 254);
}

#[test]
fn test_quantization_row_col_major_equivalence_with_random_data() {
    let num_samples = 50;
    let num_features = 8;
    let mut raw_row_major = Vec::with_capacity(num_samples * num_features);
    for i in 0..num_samples {
        for f in 0..num_features {
            let val = ((i * 13 + f * 7) % 97) as f32 * 1.5;
            raw_row_major.push(val);
        }
    }

    let mut raw_col_major = vec![0.0f32; num_samples * num_features];
    for i in 0..num_samples {
        for f in 0..num_features {
            raw_col_major[f * num_samples + i] = raw_row_major[i * num_features + f];
        }
    }

    let grid = QuantizationGrid::fit_row_major(
        &raw_row_major,
        num_samples,
        num_features,
        16,
        QuantizationMethod::UniformAndQuantiles,
        NanMode::Min,
    );

    let binned_row = grid.transform_row_major(&raw_row_major, num_samples);
    let binned_col = grid.transform_col_major(&raw_col_major, num_samples);

    for i in 0..num_samples {
        for f in 0..num_features {
            let val_row = binned_row[i * num_features + f];
            let val_col = binned_col[f * num_samples + i];
            assert_eq!(val_row, val_col, "Mismatch at sample {}, feature {}", i, f);
        }
    }
}

// ============================================================================
// 2. Categorical Processing & Leakage Prevention Tests
// ============================================================================

#[test]
fn test_ctr_mathematical_formula_and_permutation_leakage_prevention() {
    // 5 samples with categories and targets
    let categories = vec![10u32, 20, 10, 10, 20];
    let targets = vec![1.0f32, 0.0, 0.0, 1.0, 1.0];
    // Permutation order: [2, 0, 4, 1, 3]
    // sample 2 is visited 1st (perm pos 0)
    // sample 0 is visited 2nd (perm pos 1)
    // sample 4 is visited 3rd (perm pos 2)
    // sample 1 is visited 4th (perm pos 3)
    // sample 3 is visited 5th (perm pos 4)
    let permutation = vec![2, 0, 4, 1, 3];
    let prior = 0.4f32;
    let a = 2.0f32; // smoothing factor

    let config = CtrConfig {
        prior: Some(prior),
        prior_weight: a,
        max_borders: 32,
        quantization_method: QuantizationMethod::Uniform,
    };

    let (ctr_cont, _ctr_bin, _encoder) =
        CtrEncoder::fit_ordered(0, &categories, &targets, &permutation, &config);

    // Permutation pos 0: sample 2 (cat 10)
    // Seen so far for cat 10: 0 samples.
    // CTR = (0 + 0.4*2) / (0 + 2) = 0.8 / 2 = 0.4
    assert!((ctr_cont[2] - 0.4).abs() < 1e-6);

    // Permutation pos 1: sample 0 (cat 10)
    // Seen so far for cat 10: sample 2 (y=0.0). sum=0, count=1
    // CTR = (0 + 0.4*2) / (1 + 2) = 0.8 / 3 ≈ 0.2666667
    assert!((ctr_cont[0] - (0.8 / 3.0)).abs() < 1e-6);

    // Permutation pos 2: sample 4 (cat 20)
    // Seen so far for cat 20: 0 samples.
    // CTR = (0 + 0.4*2) / (0 + 2) = 0.4
    assert!((ctr_cont[4] - 0.4).abs() < 1e-6);

    // Permutation pos 3: sample 1 (cat 20)
    // Seen so far for cat 20: sample 4 (y=1.0). sum=1, count=1
    // CTR = (1.0 + 0.4*2) / (1 + 2) = 1.8 / 3 = 0.6
    assert!((ctr_cont[1] - 0.6).abs() < 1e-6);

    // Permutation pos 4: sample 3 (cat 10)
    // Seen so far for cat 10: sample 2 (y=0.0) and sample 0 (y=1.0). sum=1, count=2
    // CTR = (1.0 + 0.4*2) / (2 + 2) = 1.8 / 4 = 0.45
    assert!((ctr_cont[3] - 0.45).abs() < 1e-6);

    // Rigorous zero-leakage check: mutating sample 3's target MUST NOT change ctr_cont[3]
    let mut modified_targets = targets.clone();
    modified_targets[3] = 9999.0;
    let (ctr_mod, _, _) =
        CtrEncoder::fit_ordered(0, &categories, &modified_targets, &permutation, &config);
    assert!(
        (ctr_mod[3] - ctr_cont[3]).abs() < 1e-6,
        "Sample 3 leaked its own target!"
    );
}

#[test]
fn test_categorical_feature_pairs_and_quantization() {
    let cat_a = vec![1u32, 1, 2, 2, 1];
    let cat_b = vec![10u32, 20, 10, 20, 10];
    let targets = vec![1.0f32, 0.0, 1.0, 0.0, 1.0];
    let perm = vec![0, 1, 2, 3, 4];
    let pair = FeaturePair::new(0, 1);
    let config = CtrConfig {
        prior: Some(0.5),
        prior_weight: 1.0,
        max_borders: 8,
        quantization_method: QuantizationMethod::GreedyLogSum,
    };

    let (cont, binned, encoder) =
        PairCtrEncoder::fit_ordered(pair, &cat_a, &cat_b, &targets, &perm, &config);

    assert_eq!(cont.len(), 5);
    assert_eq!(binned.len(), 5);
    // Binned values are valid u8
    for &b in &binned {
        assert!((b as usize) <= encoder.borders.len());
    }

    // Check inference consistency
    let test_binned = encoder.transform_binned(&[1, 2], &[10, 20]);
    assert_eq!(test_binned.len(), 2);
}

// ============================================================================
// 3. Dataset Representation & Validation Tests
// ============================================================================

#[test]
fn test_dataset_ranking_groups_and_weights() {
    let binned = vec![
        1, 2, // doc 0
        2, 3, // doc 1
        1, 4, // doc 2
        0, 1, // doc 3
    ];
    let targets = vec![3.0, 1.0, 0.0, 2.0];
    let weights = vec![1.5, 0.5, 1.0, 2.0];
    let groups = vec![100, 100, 200, 200]; // 2 queries / groups

    let ds = DatasetBuilder::new(4, 2)
        .binned_features(binned)
        .targets(targets)
        .weights(weights.clone())
        .group_ids(groups.clone())
        .build()
        .unwrap();

    assert_eq!(ds.num_samples(), 4);
    assert_eq!(ds.num_features(), 2);
    assert_eq!(ds.weight(0), 1.5);
    assert_eq!(ds.weight(1), 0.5);
    assert_eq!(ds.group_id(0), Some(100));
    assert_eq!(ds.group_id(2), Some(200));

    // Slice group 100 (docs 0..2)
    let q100 = ds.slice(0..2).unwrap();
    assert_eq!(q100.num_samples(), 2);
    assert_eq!(q100.group_id(0), Some(100));
    assert_eq!(q100.targets(), &[3.0, 1.0]);
}

// ============================================================================
// 4. Oblivious Decision Tree Tests
// ============================================================================

#[test]
fn test_oblivious_tree_branchless_and_continuous_identity() {
    // 3 splits
    let splits = vec![
        SplitCondition {
            feature_idx: 0,
            bin_threshold: 3,
            continuous_threshold: 3.5,
            split_type: SplitType::Numerical,
        },
        SplitCondition {
            feature_idx: 1,
            bin_threshold: 1,
            continuous_threshold: 10.0,
            split_type: SplitType::OneHot,
        },
        SplitCondition {
            feature_idx: 2,
            bin_threshold: 5,
            continuous_threshold: 0.5,
            split_type: SplitType::Ctr,
        },
    ];

    let leaf_values: Vec<f32> = (0..8).map(|i| i as f32 * 10.0).collect();
    let tree = ObliviousTree::new(3, splits.clone(), leaf_values.clone());

    // Test all 8 binary combinations
    for b0 in 0..=1 {
        for b1 in 0..=1 {
            for b2 in 0..=1 {
                let bin_f0 = if b0 == 1 { 4 } else { 2 };
                let bin_f1 = if b1 == 1 { 2 } else { 0 };
                let bin_f2 = if b2 == 1 { 6 } else { 4 };

                let cont_f0 = if b0 == 1 { 4.0 } else { 3.0 };
                let cont_f1 = if b1 == 1 { 15.0 } else { 5.0 };
                let cont_f2 = if b2 == 1 { 0.8 } else { 0.2 };

                let binned_sample = [bin_f0, bin_f1, bin_f2];
                let continuous_sample = [cont_f0, cont_f1, cont_f2];

                let expected_leaf = b0 | (b1 << 1) | (b2 << 2);

                let leaf_bin = compute_leaf_index_binned(&splits, &binned_sample);
                let leaf_cont = compute_leaf_index_continuous(&splits, &continuous_sample);

                assert_eq!(leaf_bin, expected_leaf);
                assert_eq!(leaf_cont, expected_leaf);
                assert_eq!(tree.predict_binned(&binned_sample), leaf_values[expected_leaf]);
                assert_eq!(
                    tree.predict_continuous(&continuous_sample),
                    leaf_values[expected_leaf]
                );
            }
        }
    }
}

#[test]
fn test_find_best_split_and_split_gain() {
    // 4 samples: 2 in class 0, 2 in class 1
    // Feature 0 perfectly separates them:
    // samples 0, 1: feat0 = 0 -> grad = -1.0
    // samples 2, 3: feat0 = 2 -> grad = 1.0
    let binned_data = vec![
        0, 5, // sample 0
        0, 6, // sample 1
        2, 5, // sample 2
        2, 6, // sample 3
    ];
    let gradients = vec![-2.0, -2.0, 2.0, 2.0];
    let hessians = vec![1.0, 1.0, 1.0, 1.0];
    let leaf_indices = vec![0u32; 4];
    let borders = vec![vec![1.0], vec![5.5]];

    let best = find_best_split(
        &leaf_indices,
        &binned_data,
        &gradients,
        &hessians,
        &[0, 1],
        &borders,
        0,
        4,
        2,
        0.0,
    );

    assert!(best.is_some());
    let cand = best.unwrap();
    // Feature 0 threshold 0 (bin > 0) separates samples 0,1 from 2,3
    assert_eq!(cand.feature_idx, 0);
    assert_eq!(cand.bin_threshold, 0);
    assert!(cand.gain > 0.0);
}

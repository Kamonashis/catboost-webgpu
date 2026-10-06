use crate::quantization::{fit_borders, quantize_column, quantize_value, NanMode, QuantizationMethod};
use crate::traits::SplitType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Ordered Target Statistics (CTR) computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtrConfig {
    /// Global target prior. If None, automatically computed as the mean of training targets.
    pub prior: Option<f32>,
    /// Weight of the prior (smoothing parameter `a`, default: 1.0).
    pub prior_weight: f32,
    /// Maximum number of quantization borders for the CTR feature (default: 254).
    pub max_borders: usize,
    /// Quantization method for binning CTR continuous values.
    pub quantization_method: QuantizationMethod,
}

impl Default for CtrConfig {
    fn default() -> Self {
        Self {
            prior: None,
            prior_weight: 1.0,
            max_borders: 254,
            quantization_method: QuantizationMethod::GreedyLogSum,
        }
    }
}

/// One-hot encoder for low-cardinality categorical features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneHotEncoder {
    /// Feature index in original feature set.
    pub feature_idx: usize,
    /// Sorted unique categories mapped to binary indicator columns.
    pub categories: Vec<u32>,
    /// Maximum allowed cardinality for one-hot encoding.
    pub one_hot_max_size: usize,
}

impl OneHotEncoder {
    /// Attempts to fit a one-hot encoder on the categorical values.
    /// Returns `None` if the cardinality exceeds `one_hot_max_size`.
    pub fn try_fit(feature_idx: usize, values: &[u32], one_hot_max_size: usize) -> Option<Self> {
        let mut unique: Vec<u32> = values.to_vec();
        unique.sort_unstable();
        unique.dedup();

        if unique.len() <= one_hot_max_size && !unique.is_empty() {
            Some(Self {
                feature_idx,
                categories: unique,
                one_hot_max_size,
            })
        } else {
            None
        }
    }

    /// Number of resulting binary features generated.
    #[inline]
    pub fn num_features(&self) -> usize {
        self.categories.len()
    }

    /// Transforms a single sample category into binary u8 indicators (0 or 1).
    pub fn transform_sample(&self, cat_val: u32) -> Vec<u8> {
        self.categories
            .iter()
            .map(|&c| if c == cat_val { 1u8 } else { 0u8 })
            .collect()
    }

    /// Transforms a column of categorical values into a row-major binned u8 matrix
    /// of shape [num_samples, num_categories].
    pub fn transform_column(&self, values: &[u32]) -> Vec<u8> {
        let num_samples = values.len();
        let num_categories = self.categories.len();
        let mut out = vec![0u8; num_samples * num_categories];

        for (i, &val) in values.iter().enumerate() {
            for (k, &c) in self.categories.iter().enumerate() {
                if val == c {
                    out[i * num_categories + k] = 1u8;
                }
            }
        }

        out
    }

    /// Returns the border thresholds for one-hot features (always [0.5] per binary feature).
    pub fn borders_per_feature(&self) -> Vec<Vec<f32>> {
        vec![vec![0.5f32]; self.categories.len()]
    }
}

/// Ordered Target Statistics (CTR) encoder for a single categorical feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtrEncoder {
    pub feature_idx: usize,
    /// Global prior value.
    pub prior: f32,
    /// Smoothing weight `a`.
    pub prior_weight: f32,
    /// Cumulative target sum for each category from full training data (for inference).
    pub total_target_sums: HashMap<u32, f32>,
    /// Cumulative occurrence counts for each category from full training data (for inference).
    pub total_counts: HashMap<u32, usize>,
    /// Quantization borders for mapping continuous CTR values to u8 bins.
    pub borders: Vec<f32>,
}

impl CtrEncoder {
    /// Computes Ordered Target Statistics over a dataset permutation with zero target leakage.
    ///
    /// For permutation σ:
    /// At permutation index i (sample idx = σ[i]):
    ///   CTR_{σ[i]} = (sum_{j < i, cat_{σ[j]} == cat_{σ[i]}} y_{σ[j]} + prior * a)
    ///                / (sum_{j < i, cat_{σ[j]} == cat_{σ[i]}} 1 + a)
    ///
    /// Returns: (continuous_ctr_values, binned_ctr_values, encoder)
    pub fn fit_ordered(
        feature_idx: usize,
        categories: &[u32],
        targets: &[f32],
        permutation: &[usize],
        config: &CtrConfig,
    ) -> (Vec<f32>, Vec<u8>, Self) {
        let num_samples = categories.len();
        assert_eq!(targets.len(), num_samples, "Targets length mismatch");
        assert_eq!(permutation.len(), num_samples, "Permutation length mismatch");

        // Determine prior: user specified or target mean
        let prior = config.prior.unwrap_or_else(|| {
            if targets.is_empty() {
                0.0
            } else {
                let sum: f32 = targets.iter().sum();
                sum / targets.len() as f32
            }
        });
        let a = config.prior_weight;

        let mut ctr_continuous = vec![0.0f32; num_samples];

        // Running statistics during permutation traversal
        let mut running_sums: HashMap<u32, f32> = HashMap::new();
        let mut running_counts: HashMap<u32, usize> = HashMap::new();

        // Zero target leakage loop: sample idx only sees samples j < i
        for &idx in permutation {
            let cat = categories[idx];
            let y = targets[idx];

            let sum = running_sums.get(&cat).copied().unwrap_or(0.0);
            let count = running_counts.get(&cat).copied().unwrap_or(0);

            // Compute CTR BEFORE updating with current sample's target
            let ctr_val = (sum + prior * a) / (count as f32 + a);
            ctr_continuous[idx] = ctr_val;

            // Update running history
            running_sums.insert(cat, sum + y);
            running_counts.insert(cat, count + 1);
        }

        // Fit quantization borders on the computed continuous CTR values
        let borders = fit_borders(
            &ctr_continuous,
            config.max_borders,
            config.quantization_method,
            NanMode::Min,
        );

        // Quantize training CTR values
        let ctr_binned = quantize_column(&ctr_continuous, &borders, NanMode::Min);

        let encoder = Self {
            feature_idx,
            prior,
            prior_weight: a,
            total_target_sums: running_sums,
            total_counts: running_counts,
            borders,
        };

        (ctr_continuous, ctr_binned, encoder)
    }

    /// Evaluates continuous CTR for test samples using cumulative training statistics.
    /// Unseen categories evaluate directly to the prior.
    pub fn transform_continuous(&self, categories: &[u32]) -> Vec<f32> {
        let a = self.prior_weight;
        let prior = self.prior;

        categories
            .iter()
            .map(|&cat| {
                if let (Some(&sum), Some(&count)) = (
                    self.total_target_sums.get(&cat),
                    self.total_counts.get(&cat),
                ) {
                    (sum + prior * a) / (count as f32 + a)
                } else {
                    prior
                }
            })
            .collect()
    }

    /// Evaluates continuous CTR for a single category value.
    #[inline]
    pub fn transform_single_continuous(&self, cat: u32) -> f32 {
        let a = self.prior_weight;
        let prior = self.prior;

        if let (Some(&sum), Some(&count)) = (
            self.total_target_sums.get(&cat),
            self.total_counts.get(&cat),
        ) {
            (sum + prior * a) / (count as f32 + a)
        } else {
            prior
        }
    }

    /// Evaluates binned u8 CTR for test samples.
    pub fn transform_binned(&self, categories: &[u32]) -> Vec<u8> {
        let continuous = self.transform_continuous(categories);
        quantize_column(&continuous, &self.borders, NanMode::Min)
    }

    /// Evaluates binned u8 CTR for a single category.
    #[inline]
    pub fn transform_single_binned(&self, cat: u32) -> u8 {
        let val = self.transform_single_continuous(cat);
        quantize_value(val, &self.borders, NanMode::Min)
    }
}

/// Dynamic interaction pair between two categorical features (Feature A, Feature B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FeaturePair {
    pub feature_idx_1: usize,
    pub feature_idx_2: usize,
}

impl FeaturePair {
    pub fn new(f1: usize, f2: usize) -> Self {
        assert_ne!(f1, f2, "Feature pair cannot contain the same feature");
        // Canonical ordering (f1 < f2)
        if f1 < f2 {
            Self {
                feature_idx_1: f1,
                feature_idx_2: f2,
            }
        } else {
            Self {
                feature_idx_1: f2,
                feature_idx_2: f1,
            }
        }
    }

    /// Combines two categorical values into a single 64-bit composite category key.
    #[inline]
    pub fn combine_values(cat1: u32, cat2: u32) -> u64 {
        ((cat1 as u64) << 32) | (cat2 as u64)
    }
}

/// Ordered Target Statistics (CTR) encoder for dynamic feature interaction pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairCtrEncoder {
    pub pair: FeaturePair,
    pub prior: f32,
    pub prior_weight: f32,
    pub total_target_sums: HashMap<u64, f32>,
    pub total_counts: HashMap<u64, usize>,
    pub borders: Vec<f32>,
}

impl PairCtrEncoder {
    /// Computes Ordered Target Statistics for dynamic feature pairs over a permutation.
    pub fn fit_ordered(
        pair: FeaturePair,
        categories_1: &[u32],
        categories_2: &[u32],
        targets: &[f32],
        permutation: &[usize],
        config: &CtrConfig,
    ) -> (Vec<f32>, Vec<u8>, Self) {
        let num_samples = categories_1.len();
        assert_eq!(categories_2.len(), num_samples);
        assert_eq!(targets.len(), num_samples);
        assert_eq!(permutation.len(), num_samples);

        let prior = config.prior.unwrap_or_else(|| {
            if targets.is_empty() {
                0.0
            } else {
                let sum: f32 = targets.iter().sum();
                sum / targets.len() as f32
            }
        });
        let a = config.prior_weight;

        let mut ctr_continuous = vec![0.0f32; num_samples];
        let mut running_sums: HashMap<u64, f32> = HashMap::new();
        let mut running_counts: HashMap<u64, usize> = HashMap::new();

        for &idx in permutation {
            let pair_key = FeaturePair::combine_values(categories_1[idx], categories_2[idx]);
            let y = targets[idx];

            let sum = running_sums.get(&pair_key).copied().unwrap_or(0.0);
            let count = running_counts.get(&pair_key).copied().unwrap_or(0);

            let ctr_val = (sum + prior * a) / (count as f32 + a);
            ctr_continuous[idx] = ctr_val;

            running_sums.insert(pair_key, sum + y);
            running_counts.insert(pair_key, count + 1);
        }

        let borders = fit_borders(
            &ctr_continuous,
            config.max_borders,
            config.quantization_method,
            NanMode::Min,
        );
        let ctr_binned = quantize_column(&ctr_continuous, &borders, NanMode::Min);

        let encoder = Self {
            pair,
            prior,
            prior_weight: a,
            total_target_sums: running_sums,
            total_counts: running_counts,
            borders,
        };

        (ctr_continuous, ctr_binned, encoder)
    }

    /// Evaluates continuous pair CTR for test samples.
    pub fn transform_continuous(
        &self,
        categories_1: &[u32],
        categories_2: &[u32],
    ) -> Vec<f32> {
        let a = self.prior_weight;
        let prior = self.prior;

        categories_1
            .iter()
            .zip(categories_2.iter())
            .map(|(&c1, &c2)| {
                let pair_key = FeaturePair::combine_values(c1, c2);
                if let (Some(&sum), Some(&count)) = (
                    self.total_target_sums.get(&pair_key),
                    self.total_counts.get(&pair_key),
                ) {
                    (sum + prior * a) / (count as f32 + a)
                } else {
                    prior
                }
            })
            .collect()
    }

    /// Evaluates binned u8 pair CTR for test samples.
    pub fn transform_binned(
        &self,
        categories_1: &[u32],
        categories_2: &[u32],
    ) -> Vec<u8> {
        let continuous = self.transform_continuous(categories_1, categories_2);
        quantize_column(&continuous, &self.borders, NanMode::Min)
    }
}

/// Description of an encoded feature generated by categorical preprocessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedFeatureMeta {
    pub name: String,
    pub split_type: SplitType,
    pub source_feature_idx: usize,
    pub pair_source_feature_idx: Option<usize>,
    pub borders: Vec<f32>,
}

/// Comprehensive preprocessor managing categorical features (One-Hot, CTR, and Pair Interactions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalProcessor {
    pub one_hot_max_size: usize,
    pub one_hot_encoders: Vec<OneHotEncoder>,
    pub ctr_encoders: Vec<CtrEncoder>,
    pub pair_ctr_encoders: Vec<PairCtrEncoder>,
    pub feature_metadata: Vec<EncodedFeatureMeta>,
}

impl CategoricalProcessor {
    pub fn new(one_hot_max_size: usize) -> Self {
        Self {
            one_hot_max_size,
            one_hot_encoders: Vec::new(),
            ctr_encoders: Vec::new(),
            pair_ctr_encoders: Vec::new(),
            feature_metadata: Vec::new(),
        }
    }

    /// Fits categorical preprocessor on training data and returns binned u8 matrix
    /// for all generated categorical features [num_samples, total_cat_features].
    pub fn fit_transform(
        &mut self,
        categorical_columns: &[Vec<u32>],
        cat_feature_indices: &[usize],
        targets: &[f32],
        permutation: &[usize],
        pairs: &[FeaturePair],
        ctr_config: &CtrConfig,
    ) -> (Vec<u8>, usize) {
        let num_samples = targets.len();
        let mut binned_features_cols: Vec<Vec<u8>> = Vec::new();
        self.feature_metadata.clear();

        // 1. Process individual categorical features
        for (col_idx, &feat_idx) in cat_feature_indices.iter().enumerate() {
            let col = &categorical_columns[col_idx];

            if let Some(one_hot) = OneHotEncoder::try_fit(feat_idx, col, self.one_hot_max_size) {
                // One-Hot Encoding
                let one_hot_data = one_hot.transform_column(col);
                let num_feats = one_hot.num_features();

                for (sub_idx, &cat_val) in one_hot.categories.iter().enumerate() {
                    let mut col_bin = Vec::with_capacity(num_samples);
                    for i in 0..num_samples {
                        col_bin.push(one_hot_data[i * num_feats + sub_idx]);
                    }
                    binned_features_cols.push(col_bin);

                    self.feature_metadata.push(EncodedFeatureMeta {
                        name: format!("cat_{}_onehot_{}", feat_idx, cat_val),
                        split_type: SplitType::OneHot,
                        source_feature_idx: feat_idx,
                        pair_source_feature_idx: None,
                        borders: vec![0.5],
                    });
                }
                self.one_hot_encoders.push(one_hot);
            } else {
                // Ordered CTR Encoding
                let (_ctr_cont, ctr_bin, encoder) = CtrEncoder::fit_ordered(
                    feat_idx,
                    col,
                    targets,
                    permutation,
                    ctr_config,
                );
                let borders = encoder.borders.clone();
                binned_features_cols.push(ctr_bin);

                self.feature_metadata.push(EncodedFeatureMeta {
                    name: format!("cat_{}_ctr", feat_idx),
                    split_type: SplitType::Ctr,
                    source_feature_idx: feat_idx,
                    pair_source_feature_idx: None,
                    borders,
                });
                self.ctr_encoders.push(encoder);
            }
        }

        // 2. Process dynamic feature pairs
        for &pair in pairs {
            let pos1 = cat_feature_indices.iter().position(|&f| f == pair.feature_idx_1);
            let pos2 = cat_feature_indices.iter().position(|&f| f == pair.feature_idx_2);

            if let (Some(idx1), Some(idx2)) = (pos1, pos2) {
                let col1 = &categorical_columns[idx1];
                let col2 = &categorical_columns[idx2];

                let (_pair_cont, pair_bin, pair_encoder) = PairCtrEncoder::fit_ordered(
                    pair,
                    col1,
                    col2,
                    targets,
                    permutation,
                    ctr_config,
                );
                let borders = pair_encoder.borders.clone();
                binned_features_cols.push(pair_bin);

                self.feature_metadata.push(EncodedFeatureMeta {
                    name: format!("pair_{}_{}_ctr", pair.feature_idx_1, pair.feature_idx_2),
                    split_type: SplitType::Ctr,
                    source_feature_idx: pair.feature_idx_1,
                    pair_source_feature_idx: Some(pair.feature_idx_2),
                    borders,
                });
                self.pair_ctr_encoders.push(pair_encoder);
            }
        }

        let total_cat_features = binned_features_cols.len();
        let mut row_major = vec![0u8; num_samples * total_cat_features];

        // Transpose column-wise vectors into row-major matrix
        for (f, col) in binned_features_cols.iter().enumerate() {
            for i in 0..num_samples {
                row_major[i * total_cat_features + f] = col[i];
            }
        }

        (row_major, total_cat_features)
    }

    /// Transforms test/inference categorical columns into binned u8 row-major matrix.
    pub fn transform(
        &self,
        categorical_columns: &[Vec<u32>],
        cat_feature_indices: &[usize],
    ) -> (Vec<u8>, usize) {
        let num_samples = categorical_columns.first().map(|c| c.len()).unwrap_or(0);
        let total_cat_features = self.feature_metadata.len();
        let mut binned_features_cols: Vec<Vec<u8>> = Vec::with_capacity(total_cat_features);

        for one_hot in &self.one_hot_encoders {
            let pos = cat_feature_indices
                .iter()
                .position(|&f| f == one_hot.feature_idx)
                .expect("Feature index not found in categorical_columns");
            let col = &categorical_columns[pos];
            let one_hot_data = one_hot.transform_column(col);
            let num_feats = one_hot.num_features();

            for sub_idx in 0..num_feats {
                let mut col_bin = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    col_bin.push(one_hot_data[i * num_feats + sub_idx]);
                }
                binned_features_cols.push(col_bin);
            }
        }

        for ctr in &self.ctr_encoders {
            let pos = cat_feature_indices
                .iter()
                .position(|&f| f == ctr.feature_idx)
                .expect("Feature index not found in categorical_columns");
            let col = &categorical_columns[pos];
            binned_features_cols.push(ctr.transform_binned(col));
        }

        for pair_ctr in &self.pair_ctr_encoders {
            let pos1 = cat_feature_indices
                .iter()
                .position(|&f| f == pair_ctr.pair.feature_idx_1)
                .expect("Pair feature 1 index not found");
            let pos2 = cat_feature_indices
                .iter()
                .position(|&f| f == pair_ctr.pair.feature_idx_2)
                .expect("Pair feature 2 index not found");
            let col1 = &categorical_columns[pos1];
            let col2 = &categorical_columns[pos2];
            binned_features_cols.push(pair_ctr.transform_binned(col1, col2));
        }

        let mut row_major = vec![0u8; num_samples * total_cat_features];
        for (f, col) in binned_features_cols.iter().enumerate() {
            for i in 0..num_samples {
                row_major[i * total_cat_features + f] = col[i];
            }
        }

        (row_major, total_cat_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_encoding() {
        let values = vec![10u32, 20, 10, 30, 20];
        let encoder = OneHotEncoder::try_fit(0, &values, 3).expect("Should fit one hot");
        assert_eq!(encoder.num_features(), 3);
        assert_eq!(encoder.categories, vec![10, 20, 30]);

        let binned = encoder.transform_column(&values);
        // sample 0: category 10 -> [1, 0, 0]
        assert_eq!(&binned[0..3], &[1, 0, 0]);
        // sample 1: category 20 -> [0, 1, 0]
        assert_eq!(&binned[3..6], &[0, 1, 0]);
        // sample 3: category 30 -> [0, 0, 1]
        assert_eq!(&binned[9..12], &[0, 0, 1]);

        // Unseen category 99 -> [0, 0, 0]
        let unseen = encoder.transform_sample(99);
        assert_eq!(unseen, vec![0, 0, 0]);
    }

    #[test]
    fn test_one_hot_cardinality_threshold() {
        let values = vec![1, 2, 3, 4, 5];
        // max size 3 < 5 unique values -> returns None
        assert!(OneHotEncoder::try_fit(0, &values, 3).is_none());
    }

    #[test]
    fn test_ordered_ctr_zero_target_leakage() {
        // Essential test: ensure sample i NEVER sees y_i or any j > i
        let categories = vec![1u32, 1, 1, 1];
        let targets = vec![1.0f32, 0.0, 1.0, 0.0];
        let permutation = vec![0, 1, 2, 3]; // identity permutation
        let config = CtrConfig {
            prior: Some(0.5),
            prior_weight: 1.0, // a = 1.0
            max_borders: 32,
            quantization_method: QuantizationMethod::Uniform,
        };

        let (ctr_cont, _ctr_bin, _encoder) =
            CtrEncoder::fit_ordered(0, &categories, &targets, &permutation, &config);

        // Position 0 (sample 0): sees 0 samples. CTR = (0 + 0.5 * 1) / (0 + 1) = 0.5
        assert!((ctr_cont[0] - 0.5).abs() < 1e-6);

        // Position 1 (sample 1): sees sample 0 (y=1.0). CTR = (1.0 + 0.5 * 1) / (1 + 1) = 1.5 / 2 = 0.75
        assert!((ctr_cont[1] - 0.75).abs() < 1e-6);

        // Position 2 (sample 2): sees sample 0 (y=1.0) and sample 1 (y=0.0).
        // CTR = (1.0 + 0.0 + 0.5 * 1) / (2 + 1) = 1.5 / 3 = 0.5
        assert!((ctr_cont[2] - 0.5).abs() < 1e-6);

        // Position 3 (sample 3): sees sample 0, 1, 2 (sum=2.0, count=3).
        // CTR = (2.0 + 0.5 * 1) / (3 + 1) = 2.5 / 4 = 0.625
        assert!((ctr_cont[3] - 0.625).abs() < 1e-6);

        // Zero Leakage Verification:
        // If we change target[0] from 1.0 to 1000.0, ctr_cont[0] MUST NOT CHANGE!
        let mut modified_targets = targets.clone();
        modified_targets[0] = 1000.0;
        let (ctr_mod, _, _) =
            CtrEncoder::fit_ordered(0, &categories, &modified_targets, &permutation, &config);
        assert_eq!(
            ctr_mod[0], ctr_cont[0],
            "Zero leakage violation: changing target of sample 0 modified its own CTR!"
        );
    }

    #[test]
    fn test_ordered_ctr_inference() {
        let categories = vec![10u32, 10, 20, 20];
        let targets = vec![1.0f32, 1.0, 0.0, 0.0];
        let permutation = vec![0, 1, 2, 3];
        let config = CtrConfig {
            prior: Some(0.5),
            prior_weight: 1.0,
            max_borders: 32,
            quantization_method: QuantizationMethod::Uniform,
        };

        let (_ctr_cont, _ctr_bin, encoder) =
            CtrEncoder::fit_ordered(0, &categories, &targets, &permutation, &config);

        // Test inference for seen and unseen categories
        // Category 10: sum = 2.0, count = 2 -> (2.0 + 0.5*1)/(2+1) = 2.5/3 ≈ 0.8333
        let cat10_ctr = encoder.transform_single_continuous(10);
        assert!((cat10_ctr - (2.5 / 3.0)).abs() < 1e-5);

        // Category 20: sum = 0.0, count = 2 -> (0.0 + 0.5*1)/(2+1) = 0.5/3 ≈ 0.1667
        let cat20_ctr = encoder.transform_single_continuous(20);
        assert!((cat20_ctr - (0.5 / 3.0)).abs() < 1e-5);

        // Unseen category 99 -> prior = 0.5
        let unseen_ctr = encoder.transform_single_continuous(99);
        assert!((unseen_ctr - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_dynamic_feature_pairs() {
        let cat1 = vec![1u32, 1, 2, 2];
        let cat2 = vec![10u32, 20, 10, 20];
        let targets = vec![1.0f32, 0.0, 0.0, 1.0];
        let permutation = vec![0, 1, 2, 3];
        let pair = FeaturePair::new(0, 1);
        let config = CtrConfig {
            prior: Some(0.5),
            prior_weight: 1.0,
            max_borders: 16,
            quantization_method: QuantizationMethod::Uniform,
        };

        let (pair_cont, pair_bin, pair_encoder) =
            PairCtrEncoder::fit_ordered(pair, &cat1, &cat2, &targets, &permutation, &config);

        assert_eq!(pair_cont.len(), 4);
        assert_eq!(pair_bin.len(), 4);

        // All 4 pairs are distinct ((1,10), (1,20), (2,10), (2,20))
        // So in training permutation, each is seen for the first time -> all evaluate to prior 0.5!
        for &val in &pair_cont {
            assert!((val - 0.5).abs() < 1e-6);
        }

        // Test inference
        let test_cont = pair_encoder.transform_continuous(&[1, 99], &[10, 99]);
        // (1, 10): sum=1.0, count=1 -> (1.0 + 0.5) / 2 = 0.75
        assert!((test_cont[0] - 0.75).abs() < 1e-5);
        // Unseen pair (99, 99) -> prior 0.5
        assert!((test_cont[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_categorical_processor_integration() {
        let cat1 = vec![0u32, 1, 0, 1]; // Low cardinality <= 2 -> One-Hot
        let cat2 = vec![100u32, 200, 300, 400]; // High cardinality > 2 -> CTR
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        let permutation = vec![0, 1, 2, 3];
        let pairs = vec![FeaturePair::new(0, 1)];

        let mut processor = CategoricalProcessor::new(2);
        let (train_binned, num_feats) = processor.fit_transform(
            &[cat1.clone(), cat2.clone()],
            &[0, 1],
            &targets,
            &permutation,
            &pairs,
            &CtrConfig::default(),
        );

        // 2 one-hot features from cat1 + 1 CTR from cat2 + 1 pair CTR = 4 features
        assert_eq!(num_feats, 4);
        assert_eq!(train_binned.len(), 4 * 4);

        let (test_binned, test_num_feats) = processor.transform(
            &[cat1, cat2],
            &[0, 1],
        );
        assert_eq!(test_num_feats, 4);
        assert_eq!(test_binned.len(), 4 * 4);
    }
}

use crate::categorical::{CategoricalProcessor, CtrConfig, FeaturePair};
use crate::quantization::{fit_borders, quantize_matrix_row_major, NanMode, QuantizationMethod};
use crate::traits::SplitType;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DatasetError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Empty dataset: {0}")]
    EmptyDataset(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// In-memory dataset holding binned feature matrices, targets, weights, group IDs, and permutations.
/// Optimized for branchless oblivious tree histogram accumulation on WebGPU and multi-threaded CPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Number of samples in the dataset.
    pub num_samples: usize,
    /// Number of binned features.
    pub num_features: usize,
    /// Row-major binned feature matrix [num_samples, num_features] of compact u8 bins (0..254).
    pub binned_features: Vec<u8>,
    /// Optional raw continuous features [num_samples, num_raw_features] (retained for inference / metrics).
    pub raw_features: Option<Vec<f32>>,
    /// Target label array [num_samples].
    pub targets: Vec<f32>,
    /// Optional sample weights [num_samples]. Defaults to 1.0 per sample if None.
    pub weights: Option<Vec<f32>>,
    /// Optional group / query IDs for ranking objectives [num_samples].
    pub group_ids: Option<Vec<u32>>,
    /// Random permutations of 0..num_samples for ordered boosting and online CTR.
    pub permutations: Vec<Vec<usize>>,
    /// Names of features.
    pub feature_names: Vec<String>,
    /// Quantization borders for each feature (length == num_features).
    pub feature_borders: Vec<Vec<f32>>,
    /// Split type for each feature (Numerical, OneHot, Ctr).
    pub feature_types: Vec<SplitType>,
}

impl Dataset {
    /// Returns the number of samples.
    #[inline]
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Returns the number of binned features.
    #[inline]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Returns the raw binned buffer [num_samples * num_features].
    #[inline]
    pub fn binned_features(&self) -> &[u8] {
        &self.binned_features
    }

    /// Returns the binned feature slice for sample `sample_idx`.
    #[inline]
    pub fn sample_binned(&self, sample_idx: usize) -> &[u8] {
        let start = sample_idx * self.num_features;
        &self.binned_features[start..start + self.num_features]
    }

    /// Returns the binned u8 value for sample `sample_idx` and feature `feature_idx`.
    #[inline]
    pub fn get_bin(&self, sample_idx: usize, feature_idx: usize) -> u8 {
        self.binned_features[sample_idx * self.num_features + feature_idx]
    }

    /// Returns target slice.
    #[inline]
    pub fn targets(&self) -> &[f32] {
        &self.targets
    }

    /// Returns target for sample `sample_idx`.
    #[inline]
    pub fn target(&self, sample_idx: usize) -> f32 {
        self.targets[sample_idx]
    }

    /// Returns optional weights slice.
    #[inline]
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Returns weight for sample `sample_idx`, or 1.0 if no weights are defined.
    #[inline]
    pub fn weight(&self, sample_idx: usize) -> f32 {
        match &self.weights {
            Some(w) => w[sample_idx],
            None => 1.0,
        }
    }

    /// Returns optional group IDs slice.
    #[inline]
    pub fn group_ids(&self) -> Option<&[u32]> {
        self.group_ids.as_deref()
    }

    /// Returns group ID for sample `sample_idx` if present.
    #[inline]
    pub fn group_id(&self, sample_idx: usize) -> Option<u32> {
        self.group_ids.as_ref().map(|g| g[sample_idx])
    }

    /// Returns permutations slice.
    #[inline]
    pub fn permutations(&self) -> &[Vec<usize>] {
        &self.permutations
    }

    /// Returns a specific permutation if available.
    #[inline]
    pub fn permutation(&self, perm_idx: usize) -> Option<&[usize]> {
        self.permutations.get(perm_idx).map(|v| v.as_slice())
    }

    /// Returns feature borders slice.
    #[inline]
    pub fn feature_borders(&self) -> &[Vec<f32>] {
        &self.feature_borders
    }

    /// Returns borders for a specific feature.
    #[inline]
    pub fn borders_for_feature(&self, feature_idx: usize) -> &[f32] {
        &self.feature_borders[feature_idx]
    }

    /// Returns feature types.
    #[inline]
    pub fn feature_types(&self) -> &[SplitType] {
        &self.feature_types
    }

    /// Returns feature names.
    #[inline]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Generates `num_permutations` independent uniform random permutations of sample indices.
    pub fn generate_permutations(&mut self, num_permutations: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        self.permutations.clear();
        self.permutations.reserve(num_permutations);

        for _ in 0..num_permutations {
            let mut perm: Vec<usize> = (0..self.num_samples).collect();
            perm.shuffle(&mut rng);
            self.permutations.push(perm);
        }
    }

    /// Transposes row-major binned features into column-major layout [num_features, num_samples].
    pub fn to_column_major(&self) -> Vec<u8> {
        let mut col_major = vec![0u8; self.num_samples * self.num_features];
        for i in 0..self.num_samples {
            for f in 0..self.num_features {
                col_major[f * self.num_samples + i] = self.binned_features[i * self.num_features + f];
            }
        }
        col_major
    }

    /// Slices the dataset along the sample dimension [range.start..range.end].
    pub fn slice(&self, range: Range<usize>) -> Result<Self, DatasetError> {
        if range.start >= self.num_samples || range.end > self.num_samples || range.start >= range.end {
            return Err(DatasetError::InvalidConfig(format!(
                "Invalid slice range {:?} for dataset with {} samples",
                range, self.num_samples
            )));
        }

        let slice_len = range.end - range.start;
        let binned_start = range.start * self.num_features;
        let binned_end = range.end * self.num_features;
        let sliced_binned = self.binned_features[binned_start..binned_end].to_vec();

        let sliced_raw = self.raw_features.as_ref().map(|raw| {
            let num_raw_feats = raw.len() / self.num_samples;
            let start = range.start * num_raw_feats;
            let end = range.end * num_raw_feats;
            raw[start..end].to_vec()
        });

        let sliced_targets = self.targets[range.start..range.end].to_vec();
        let sliced_weights = self
            .weights
            .as_ref()
            .map(|w| w[range.start..range.end].to_vec());
        let sliced_groups = self
            .group_ids
            .as_ref()
            .map(|g| g[range.start..range.end].to_vec());

        // Re-generate identity permutation for slice
        let sliced_perms = vec![(0..slice_len).collect()];

        Ok(Self {
            num_samples: slice_len,
            num_features: self.num_features,
            binned_features: sliced_binned,
            raw_features: sliced_raw,
            targets: sliced_targets,
            weights: sliced_weights,
            group_ids: sliced_groups,
            permutations: sliced_perms,
            feature_names: self.feature_names.clone(),
            feature_borders: self.feature_borders.clone(),
            feature_types: self.feature_types.clone(),
        })
    }

    /// Creates a dataset directly from raw continuous features, automatically fitting borders
    /// and generating random permutations.
    pub fn from_raw_continuous(
        raw_data: &[f32],
        num_samples: usize,
        num_features: usize,
        targets: Vec<f32>,
        max_borders: usize,
        method: QuantizationMethod,
        seed: u64,
    ) -> Result<Self, DatasetError> {
        if num_samples == 0 || num_features == 0 {
            return Err(DatasetError::EmptyDataset(
                "num_samples and num_features must be greater than zero".into(),
            ));
        }
        if raw_data.len() != num_samples * num_features {
            return Err(DatasetError::DimensionMismatch(format!(
                "raw_data length {} does not match num_samples * num_features ({} * {})",
                raw_data.len(),
                num_samples,
                num_features
            )));
        }
        if targets.len() != num_samples {
            return Err(DatasetError::DimensionMismatch(format!(
                "targets length {} does not match num_samples {}",
                targets.len(),
                num_samples
            )));
        }

        // Fit quantization borders for each continuous feature
        let mut feature_borders: Vec<Vec<f32>> = Vec::with_capacity(num_features);
        for f in 0..num_features {
            let col: Vec<f32> = (0..num_samples)
                .map(|i| raw_data[i * num_features + f])
                .collect();
            let borders = fit_borders(&col, max_borders, method, NanMode::Min);
            feature_borders.push(borders);
        }

        let binned_features =
            quantize_matrix_row_major(raw_data, num_samples, &feature_borders, NanMode::Min);

        let feature_names: Vec<String> = (0..num_features)
            .map(|f| format!("feature_{}", f))
            .collect();
        let feature_types = vec![SplitType::Numerical; num_features];

        let mut dataset = Self {
            num_samples,
            num_features,
            binned_features,
            raw_features: Some(raw_data.to_vec()),
            targets,
            weights: None,
            group_ids: None,
            permutations: Vec::new(),
            feature_names,
            feature_borders,
            feature_types,
        };

        dataset.generate_permutations(4, seed);
        Ok(dataset)
    }

    /// Creates a dataset with both numerical features and categorical features.
    pub fn from_mixed(
        numerical_raw: &[f32],
        num_numerical_features: usize,
        categorical_columns: &[Vec<u32>],
        targets: Vec<f32>,
        one_hot_max_size: usize,
        pairs: &[FeaturePair],
        max_borders: usize,
        method: QuantizationMethod,
        seed: u64,
    ) -> Result<Self, DatasetError> {
        let num_samples = targets.len();
        if num_samples == 0 {
            return Err(DatasetError::EmptyDataset("Dataset is empty".into()));
        }

        // Generate base permutation for categorical Ordered CTR
        let mut rng = StdRng::seed_from_u64(seed);
        let mut base_perm: Vec<usize> = (0..num_samples).collect();
        base_perm.shuffle(&mut rng);

        // 1. Process continuous numerical features
        let mut all_binned_cols: Vec<Vec<u8>> = Vec::new();
        let mut feature_names: Vec<String> = Vec::new();
        let mut feature_borders: Vec<Vec<f32>> = Vec::new();
        let mut feature_types: Vec<SplitType> = Vec::new();

        if num_numerical_features > 0 {
            if numerical_raw.len() != num_samples * num_numerical_features {
                return Err(DatasetError::DimensionMismatch(
                    "numerical_raw length does not match num_samples * num_numerical_features".into(),
                ));
            }

            for f in 0..num_numerical_features {
                let col: Vec<f32> = (0..num_samples)
                    .map(|i| numerical_raw[i * num_numerical_features + f])
                    .collect();
                let borders = fit_borders(&col, max_borders, method, NanMode::Min);
                let binned = crate::quantization::quantize_column(&col, &borders, NanMode::Min);

                all_binned_cols.push(binned);
                feature_names.push(format!("num_{}", f));
                feature_borders.push(borders);
                feature_types.push(SplitType::Numerical);
            }
        }

        // 2. Process categorical features
        if !categorical_columns.is_empty() {
            let cat_indices: Vec<usize> = (0..categorical_columns.len()).collect();
            let mut cat_processor = CategoricalProcessor::new(one_hot_max_size);
            let ctr_config = CtrConfig {
                max_borders,
                quantization_method: method,
                ..Default::default()
            };

            let (cat_binned_row_major, total_cat_feats) = cat_processor.fit_transform(
                categorical_columns,
                &cat_indices,
                &targets,
                &base_perm,
                pairs,
                &ctr_config,
            );

            // Extract each categorical column
            for f in 0..total_cat_feats {
                let mut col = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    col.push(cat_binned_row_major[i * total_cat_feats + f]);
                }
                all_binned_cols.push(col);

                let meta = &cat_processor.feature_metadata[f];
                feature_names.push(meta.name.clone());
                feature_borders.push(meta.borders.clone());
                feature_types.push(meta.split_type);
            }
        }

        let total_features = all_binned_cols.len();
        if total_features == 0 {
            return Err(DatasetError::EmptyDataset(
                "No features (numerical or categorical) provided".into(),
            ));
        }

        // Transpose all columns into flat row-major matrix [num_samples, total_features]
        let mut binned_features = vec![0u8; num_samples * total_features];
        for (f, col) in all_binned_cols.iter().enumerate() {
            for i in 0..num_samples {
                binned_features[i * total_features + f] = col[i];
            }
        }

        let mut full_raw = vec![0.0f32; num_samples * total_features];
        for i in 0..num_samples {
            for f in 0..num_numerical_features {
                full_raw[i * total_features + f] = numerical_raw[i * num_numerical_features + f];
            }
            for f in num_numerical_features..total_features {
                full_raw[i * total_features + f] = binned_features[i * total_features + f] as f32;
            }
        }
        let raw_features = Some(full_raw);

        let mut dataset = Self {
            num_samples,
            num_features: total_features,
            binned_features,
            raw_features,
            targets,
            weights: None,
            group_ids: None,
            permutations: vec![base_perm],
            feature_names,
            feature_borders,
            feature_types,
        };

        dataset.generate_permutations(4, seed);
        Ok(dataset)
    }

    /// Validates internal consistency of the dataset buffers.
    pub fn validate(&self) -> Result<(), DatasetError> {
        if self.num_samples == 0 {
            return Err(DatasetError::EmptyDataset("Dataset has 0 samples".into()));
        }
        if self.num_features == 0 {
            return Err(DatasetError::EmptyDataset("Dataset has 0 features".into()));
        }
        if self.binned_features.len() != self.num_samples * self.num_features {
            return Err(DatasetError::DimensionMismatch(format!(
                "binned_features length {} != num_samples * num_features ({} * {})",
                self.binned_features.len(),
                self.num_samples,
                self.num_features
            )));
        }
        if self.targets.len() != self.num_samples {
            return Err(DatasetError::DimensionMismatch(format!(
                "targets length {} != num_samples {}",
                self.targets.len(),
                self.num_samples
            )));
        }
        if let Some(w) = &self.weights {
            if w.len() != self.num_samples {
                return Err(DatasetError::DimensionMismatch(format!(
                    "weights length {} != num_samples {}",
                    w.len(),
                    self.num_samples
                )));
            }
        }
        if let Some(g) = &self.group_ids {
            if g.len() != self.num_samples {
                return Err(DatasetError::DimensionMismatch(format!(
                    "group_ids length {} != num_samples {}",
                    g.len(),
                    self.num_samples
                )));
            }
        }
        if self.feature_borders.len() != self.num_features {
            return Err(DatasetError::DimensionMismatch(format!(
                "feature_borders length {} != num_features {}",
                self.feature_borders.len(),
                self.num_features
            )));
        }
        if self.feature_types.len() != self.num_features {
            return Err(DatasetError::DimensionMismatch(format!(
                "feature_types length {} != num_features {}",
                self.feature_types.len(),
                self.num_features
            )));
        }
        for (p_idx, perm) in self.permutations.iter().enumerate() {
            if perm.len() != self.num_samples {
                return Err(DatasetError::DimensionMismatch(format!(
                    "permutation {} length {} != num_samples {}",
                    p_idx,
                    perm.len(),
                    self.num_samples
                )));
            }
        }

        Ok(())
    }
}

/// Fluent builder for constructing a Dataset with optional fields and validation.
#[derive(Debug, Default)]
pub struct DatasetBuilder {
    num_samples: usize,
    num_features: usize,
    binned_features: Option<Vec<u8>>,
    raw_features: Option<Vec<f32>>,
    targets: Option<Vec<f32>>,
    weights: Option<Vec<f32>>,
    group_ids: Option<Vec<u32>>,
    permutations: Vec<Vec<usize>>,
    feature_names: Option<Vec<String>>,
    feature_borders: Option<Vec<Vec<f32>>>,
    feature_types: Option<Vec<SplitType>>,
}

impl DatasetBuilder {
    pub fn new(num_samples: usize, num_features: usize) -> Self {
        Self {
            num_samples,
            num_features,
            ..Default::default()
        }
    }

    pub fn binned_features(mut self, binned: Vec<u8>) -> Self {
        self.binned_features = Some(binned);
        self
    }

    pub fn raw_features(mut self, raw: Vec<f32>) -> Self {
        self.raw_features = Some(raw);
        self
    }

    pub fn targets(mut self, targets: Vec<f32>) -> Self {
        self.targets = Some(targets);
        self
    }

    pub fn weights(mut self, weights: Vec<f32>) -> Self {
        self.weights = Some(weights);
        self
    }

    pub fn group_ids(mut self, group_ids: Vec<u32>) -> Self {
        self.group_ids = Some(group_ids);
        self
    }

    pub fn permutations(mut self, perms: Vec<Vec<usize>>) -> Self {
        self.permutations = perms;
        self
    }

    pub fn feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    pub fn feature_borders(mut self, borders: Vec<Vec<f32>>) -> Self {
        self.feature_borders = Some(borders);
        self
    }

    pub fn feature_types(mut self, types: Vec<SplitType>) -> Self {
        self.feature_types = Some(types);
        self
    }

    pub fn build(self) -> Result<Dataset, DatasetError> {
        let binned = self
            .binned_features
            .ok_or_else(|| DatasetError::EmptyDataset("binned_features is required".into()))?;
        let targets = self
            .targets
            .ok_or_else(|| DatasetError::EmptyDataset("targets is required".into()))?;

        let feature_names = self.feature_names.unwrap_or_else(|| {
            (0..self.num_features)
                .map(|f| format!("feature_{}", f))
                .collect()
        });

        let feature_borders = self
            .feature_borders
            .unwrap_or_else(|| vec![Vec::new(); self.num_features]);

        let feature_types = self
            .feature_types
            .unwrap_or_else(|| vec![SplitType::Numerical; self.num_features]);

        let mut dataset = Dataset {
            num_samples: self.num_samples,
            num_features: self.num_features,
            binned_features: binned,
            raw_features: self.raw_features,
            targets,
            weights: self.weights,
            group_ids: self.group_ids,
            permutations: self.permutations,
            feature_names,
            feature_borders,
            feature_types,
        };

        if dataset.permutations.is_empty() {
            dataset.generate_permutations(1, 42);
        }

        dataset.validate()?;
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_from_raw_continuous() {
        let num_samples = 10;
        let num_features = 2;
        let raw = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
            5.0, 50.0,
            6.0, 60.0,
            7.0, 70.0,
            8.0, 80.0,
            9.0, 90.0,
            10.0, 100.0,
        ];
        let targets = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let ds = Dataset::from_raw_continuous(
            &raw,
            num_samples,
            num_features,
            targets,
            4,
            QuantizationMethod::Uniform,
            123,
        )
        .expect("Dataset creation failed");

        assert_eq!(ds.num_samples(), 10);
        assert_eq!(ds.num_features(), 2);
        assert_eq!(ds.permutations().len(), 4);
        assert_eq!(ds.feature_borders().len(), 2);

        // Verify binned access
        assert_eq!(ds.sample_binned(0).len(), 2);
        assert_eq!(ds.get_bin(0, 0), 0);
        assert_eq!(ds.get_bin(9, 0), ds.borders_for_feature(0).len() as u8);
        assert_eq!(ds.weight(0), 1.0);
    }

    #[test]
    fn test_dataset_slice() {
        let raw = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ds = Dataset::from_raw_continuous(
            &raw,
            6,
            1,
            targets,
            4,
            QuantizationMethod::Uniform,
            42,
        )
        .unwrap();

        let slice = ds.slice(2..5).unwrap();
        assert_eq!(slice.num_samples(), 3);
        assert_eq!(slice.targets(), &[2.0, 3.0, 4.0]);
        assert_eq!(slice.get_bin(0, 0), ds.get_bin(2, 0));
        assert_eq!(slice.get_bin(1, 0), ds.get_bin(3, 0));
        assert_eq!(slice.get_bin(2, 0), ds.get_bin(4, 0));
    }

    #[test]
    fn test_dataset_column_major_transpose() {
        let binned = vec![
            10, 20, 30, // sample 0
            11, 21, 31, // sample 1
        ];
        let targets = vec![0.0, 1.0];

        let ds = DatasetBuilder::new(2, 3)
            .binned_features(binned)
            .targets(targets)
            .build()
            .unwrap();

        let col_major = ds.to_column_major();
        // feat 0: [10, 11]
        // feat 1: [20, 21]
        // feat 2: [30, 31]
        assert_eq!(col_major, vec![10, 11, 20, 21, 30, 31]);
    }

    #[test]
    fn test_dataset_mixed_with_categoricals() {
        let numerical = vec![1.5f32, 2.5, 3.5, 4.5];
        let cat_col = vec![10u32, 20, 10, 20];
        let targets = vec![0.0f32, 1.0, 0.0, 1.0];

        let ds = Dataset::from_mixed(
            &numerical,
            1,
            &[cat_col],
            targets,
            2, // one_hot_max_size 2 -> will produce 2 one-hot features
            &[],
            8,
            QuantizationMethod::Uniform,
            42,
        )
        .unwrap();

        // 1 numerical + 2 one-hot = 3 features
        assert_eq!(ds.num_samples(), 4);
        assert_eq!(ds.num_features(), 3);
        assert_eq!(ds.feature_types()[0], SplitType::Numerical);
        assert_eq!(ds.feature_types()[1], SplitType::OneHot);
        assert_eq!(ds.feature_types()[2], SplitType::OneHot);
    }
}

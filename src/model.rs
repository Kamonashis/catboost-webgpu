use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::importance::{
    ensemble_tree_shap, ensemble_tree_shap_batch, feature_importance_prediction_values_change,
    FeatureImportanceType, ShapValues,
};
use crate::objective::sigmoid;
use crate::traits::{ObliviousTree, SplitCondition, SplitType};

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Unsupported format: {0}. Supported: 'json', 'cbm', 'python'")]
    UnsupportedFormat(String),
    #[error("Corrupt binary format: {0}")]
    CorruptBinary(String),
}

/// Serialized CatBoost model holding trained oblivious trees and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatBoostModel {
    pub trees: Vec<ObliviousTree>,
    pub learning_rate: f32,
    pub base_score: f32,
    pub target_scale: f32,
    pub target_offset: f32,
    pub feature_names: Vec<String>,
    pub feature_borders: Vec<Vec<f32>>,
    pub loss_name: String,
    pub metadata: HashMap<String, String>,
}

impl CatBoostModel {
    pub fn new(
        trees: Vec<ObliviousTree>,
        learning_rate: f32,
        base_score: f32,
        feature_names: Vec<String>,
        feature_borders: Vec<Vec<f32>>,
        loss_name: String,
    ) -> Self {
        Self {
            trees,
            learning_rate,
            base_score,
            target_scale: 1.0,
            target_offset: 0.0,
            feature_names,
            feature_borders,
            loss_name,
            metadata: HashMap::new(),
        }
    }

    /// Evaluates raw ensemble prediction for a single continuous sample.
    #[inline]
    pub fn predict_raw(&self, sample: &[f32]) -> f32 {
        let mut pred = self.base_score;
        for tree in &self.trees {
            pred += self.learning_rate * tree.predict_continuous(sample);
        }
        pred
    }

    /// Evaluates scaled final prediction for a single continuous sample.
    #[inline]
    pub fn predict(&self, sample: &[f32]) -> f32 {
        let raw = self.predict_raw(sample);
        raw * self.target_scale + self.target_offset
    }

    /// Evaluates scaled final predictions for a batch of samples in parallel using Rayon.
    pub fn predict_batch(&self, samples: &[f32], num_samples: usize) -> Vec<f32> {
        let num_features = if num_samples > 0 {
            samples.len() / num_samples
        } else {
            0
        };
        assert_eq!(samples.len(), num_samples * num_features);

        (0..num_samples)
            .into_par_iter()
            .map(|i| {
                let sample = &samples[i * num_features..(i + 1) * num_features];
                self.predict(sample)
            })
            .collect()
    }

    /// Evaluates raw ensemble prediction for pre-binned features.
    #[inline]
    pub fn predict_binned(&self, binned_sample: &[u8]) -> f32 {
        let mut pred = self.base_score;
        for tree in &self.trees {
            pred += self.learning_rate * tree.predict_binned(binned_sample);
        }
        pred * self.target_scale + self.target_offset
    }

    /// Evaluates predictions for binned batch data in parallel.
    pub fn predict_binned_batch(&self, binned_data: &[u8], num_samples: usize) -> Vec<f32> {
        let num_features = if num_samples > 0 {
            binned_data.len() / num_samples
        } else {
            0
        };
        assert_eq!(binned_data.len(), num_samples * num_features);

        (0..num_samples)
            .into_par_iter()
            .map(|i| {
                let sample = &binned_data[i * num_features..(i + 1) * num_features];
                self.predict_binned(sample)
            })
            .collect()
    }

    /// Predicts probabilities for classification [P(0), P(1)].
    #[inline]
    pub fn predict_proba(&self, sample: &[f32]) -> Vec<f32> {
        let raw = self.predict_raw(sample);
        let p1 = sigmoid(raw);
        vec![1.0 - p1, p1]
    }

    /// Predicts probabilities for a batch of samples in parallel.
    pub fn predict_proba_batch(&self, samples: &[f32], num_samples: usize) -> Vec<Vec<f32>> {
        let num_features = if num_samples > 0 {
            samples.len() / num_samples
        } else {
            0
        };
        assert_eq!(samples.len(), num_samples * num_features);

        (0..num_samples)
            .into_par_iter()
            .map(|i| {
                let sample = &samples[i * num_features..(i + 1) * num_features];
                self.predict_proba(sample)
            })
            .collect()
    }

    /// Returns leaf indices for each tree in the ensemble for a given sample.
    pub fn predict_leaf_indices(&self, sample: &[f32]) -> Vec<usize> {
        self.trees
            .iter()
            .map(|t| t.predict_leaf_continuous(sample))
            .collect()
    }

    /// Computes feature importances across all trees.
    pub fn feature_importance(&self, importance_type: FeatureImportanceType) -> Vec<f32> {
        let num_features = self.feature_names.len().max(
            self.trees
                .iter()
                .flat_map(|t| t.splits.iter().map(|s| s.feature_idx + 1))
                .max()
                .unwrap_or(0),
        );

        match importance_type {
            FeatureImportanceType::PredictionValuesChange => {
                feature_importance_prediction_values_change(&self.trees, num_features)
            }
            FeatureImportanceType::LossFunctionChange => {
                // Return uniform or empty if no data provided
                feature_importance_prediction_values_change(&self.trees, num_features)
            }
        }
    }

    /// Computes exact Oblivious Tree SHAP values for a sample in O(D * 2^D) per tree.
    pub fn tree_shap(&self, sample: &[f32]) -> ShapValues {
        let num_features = self.feature_names.len().max(sample.len());
        let scaled_trees: Vec<ObliviousTree> = self
            .trees
            .iter()
            .map(|t| {
                let scaled_leaves: Vec<f32> = t
                    .leaf_values
                    .iter()
                    .map(|&v| v * self.learning_rate * self.target_scale)
                    .collect();
                ObliviousTree::new(t.depth, t.splits.clone(), scaled_leaves)
            })
            .collect();
        let scaled_base = self.base_score * self.target_scale + self.target_offset;
        ensemble_tree_shap(&scaled_trees, scaled_base, sample, num_features)
    }

    /// Computes exact Tree SHAP values for a batch of samples in parallel.
    pub fn tree_shap_batch(
        &self,
        samples: &[f32],
        num_samples: usize,
    ) -> (f32, Vec<Vec<f32>>) {
        let num_features = if num_samples > 0 {
            samples.len() / num_samples
        } else {
            self.feature_names.len()
        };
        let scaled_trees: Vec<ObliviousTree> = self
            .trees
            .iter()
            .map(|t| {
                let scaled_leaves: Vec<f32> = t
                    .leaf_values
                    .iter()
                    .map(|&v| v * self.learning_rate * self.target_scale)
                    .collect();
                ObliviousTree::new(t.depth, t.splits.clone(), scaled_leaves)
            })
            .collect();
        let scaled_base = self.base_score * self.target_scale + self.target_offset;
        ensemble_tree_shap_batch(
            &scaled_trees,
            scaled_base,
            samples,
            num_samples,
            num_features,
        )
    }

    // ========================================================================
    // Model Persistence: JSON, CBM (Binary), and Standalone Python Code Gen
    // ========================================================================

    /// Saves model to disk in the specified format: "json", "cbm", or "python".
    pub fn save_model(&self, path: &str, format: &str) -> Result<(), ModelError> {
        match format.to_lowercase().as_str() {
            "json" => {
                let file = File::create(path)?;
                let writer = BufWriter::new(file);
                serde_json::to_writer_pretty(writer, self)?;
                Ok(())
            }
            "cbm" | "binary" => {
                let file = File::create(path)?;
                let mut writer = BufWriter::new(file);
                self.write_cbm(&mut writer)?;
                Ok(())
            }
            "python" | "py" => {
                self.export_python(path)?;
                Ok(())
            }
            _ => Err(ModelError::UnsupportedFormat(format.to_string())),
        }
    }

    /// Loads model from disk in "json" or "cbm" format.
    pub fn load_model(path: &str, format: &str) -> Result<Self, ModelError> {
        match format.to_lowercase().as_str() {
            "json" => {
                let file = File::open(path)?;
                let reader = BufReader::new(file);
                let model: CatBoostModel = serde_json::from_reader(reader)?;
                Ok(model)
            }
            "cbm" | "binary" => {
                let file = File::open(path)?;
                let mut reader = BufReader::new(file);
                Self::read_cbm(&mut reader)
            }
            _ => Err(ModelError::UnsupportedFormat(format.to_string())),
        }
    }

    /// Binary format writer (CBM: CatBoost Model Binary)
    fn write_cbm<W: Write>(&self, w: &mut W) -> Result<(), ModelError> {
        // Magic header: "CBMB" (4 bytes) + Version: u32 = 1
        w.write_all(b"CBMB")?;
        w.write_all(&1u32.to_le_bytes())?;

        // Scalar parameters
        w.write_all(&self.learning_rate.to_le_bytes())?;
        w.write_all(&self.base_score.to_le_bytes())?;
        w.write_all(&self.target_scale.to_le_bytes())?;
        w.write_all(&self.target_offset.to_le_bytes())?;

        // Loss name
        let loss_bytes = self.loss_name.as_bytes();
        w.write_all(&(loss_bytes.len() as u32).to_le_bytes())?;
        w.write_all(loss_bytes)?;

        // Feature names
        w.write_all(&(self.feature_names.len() as u32).to_le_bytes())?;
        for name in &self.feature_names {
            let nb = name.as_bytes();
            w.write_all(&(nb.len() as u32).to_le_bytes())?;
            w.write_all(nb)?;
        }

        // Feature borders
        w.write_all(&(self.feature_borders.len() as u32).to_le_bytes())?;
        for borders in &self.feature_borders {
            w.write_all(&(borders.len() as u32).to_le_bytes())?;
            for &b in borders {
                w.write_all(&b.to_le_bytes())?;
            }
        }

        // Trees
        w.write_all(&(self.trees.len() as u32).to_le_bytes())?;
        for tree in &self.trees {
            w.write_all(&(tree.depth as u32).to_le_bytes())?;
            for split in &tree.splits {
                w.write_all(&(split.feature_idx as u32).to_le_bytes())?;
                w.write_all(&[split.bin_threshold])?;
                w.write_all(&split.continuous_threshold.to_le_bytes())?;
                let st_byte = match split.split_type {
                    SplitType::Numerical => 0u8,
                    SplitType::OneHot => 1u8,
                    SplitType::Ctr => 2u8,
                };
                w.write_all(&[st_byte])?;
            }
            w.write_all(&(tree.leaf_values.len() as u32).to_le_bytes())?;
            for &lv in &tree.leaf_values {
                w.write_all(&lv.to_le_bytes())?;
            }
        }

        // Metadata map
        w.write_all(&(self.metadata.len() as u32).to_le_bytes())?;
        for (k, v) in &self.metadata {
            let kb = k.as_bytes();
            w.write_all(&(kb.len() as u32).to_le_bytes())?;
            w.write_all(kb)?;
            let vb = v.as_bytes();
            w.write_all(&(vb.len() as u32).to_le_bytes())?;
            w.write_all(vb)?;
        }

        w.flush()?;
        Ok(())
    }

    /// Binary format reader (CBM)
    fn read_cbm<R: Read>(r: &mut R) -> Result<Self, ModelError> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"CBMB" {
            return Err(ModelError::CorruptBinary("Invalid magic bytes".into()));
        }

        let mut v_buf = [0u8; 4];
        r.read_exact(&mut v_buf)?;
        let version = u32::from_le_bytes(v_buf);
        if version != 1 {
            return Err(ModelError::CorruptBinary(format!(
                "Unsupported CBM version: {}",
                version
            )));
        }

        let mut f4 = [0u8; 4];
        r.read_exact(&mut f4)?;
        let learning_rate = f32::from_le_bytes(f4);

        r.read_exact(&mut f4)?;
        let base_score = f32::from_le_bytes(f4);

        r.read_exact(&mut f4)?;
        let target_scale = f32::from_le_bytes(f4);

        r.read_exact(&mut f4)?;
        let target_offset = f32::from_le_bytes(f4);

        // Loss name
        r.read_exact(&mut f4)?;
        let loss_len = u32::from_le_bytes(f4) as usize;
        let mut loss_bytes = vec![0u8; loss_len];
        r.read_exact(&mut loss_bytes)?;
        let loss_name = String::from_utf8(loss_bytes)
            .map_err(|e| ModelError::CorruptBinary(e.to_string()))?;

        // Feature names
        r.read_exact(&mut f4)?;
        let num_feat_names = u32::from_le_bytes(f4) as usize;
        let mut feature_names = Vec::with_capacity(num_feat_names);
        for _ in 0..num_feat_names {
            r.read_exact(&mut f4)?;
            let slen = u32::from_le_bytes(f4) as usize;
            let mut sbytes = vec![0u8; slen];
            r.read_exact(&mut sbytes)?;
            let name = String::from_utf8(sbytes)
                .map_err(|e| ModelError::CorruptBinary(e.to_string()))?;
            feature_names.push(name);
        }

        // Feature borders
        r.read_exact(&mut f4)?;
        let num_border_feats = u32::from_le_bytes(f4) as usize;
        let mut feature_borders = Vec::with_capacity(num_border_feats);
        for _ in 0..num_border_feats {
            r.read_exact(&mut f4)?;
            let n_borders = u32::from_le_bytes(f4) as usize;
            let mut borders = Vec::with_capacity(n_borders);
            for _ in 0..n_borders {
                r.read_exact(&mut f4)?;
                borders.push(f32::from_le_bytes(f4));
            }
            feature_borders.push(borders);
        }

        // Trees
        r.read_exact(&mut f4)?;
        let num_trees = u32::from_le_bytes(f4) as usize;
        let mut trees = Vec::with_capacity(num_trees);

        for _ in 0..num_trees {
            r.read_exact(&mut f4)?;
            let depth = u32::from_le_bytes(f4) as usize;
            let mut splits = Vec::with_capacity(depth);

            for _ in 0..depth {
                r.read_exact(&mut f4)?;
                let f_idx = u32::from_le_bytes(f4) as usize;
                let mut b_thresh = [0u8; 1];
                r.read_exact(&mut b_thresh)?;
                r.read_exact(&mut f4)?;
                let c_thresh = f32::from_le_bytes(f4);
                let mut st = [0u8; 1];
                r.read_exact(&mut st)?;
                let split_type = match st[0] {
                    0 => SplitType::Numerical,
                    1 => SplitType::OneHot,
                    2 => SplitType::Ctr,
                    _ => SplitType::Numerical,
                };
                splits.push(SplitCondition {
                    feature_idx: f_idx,
                    bin_threshold: b_thresh[0],
                    continuous_threshold: c_thresh,
                    split_type,
                });
            }

            r.read_exact(&mut f4)?;
            let num_leaves = u32::from_le_bytes(f4) as usize;
            let mut leaf_values = Vec::with_capacity(num_leaves);
            for _ in 0..num_leaves {
                r.read_exact(&mut f4)?;
                leaf_values.push(f32::from_le_bytes(f4));
            }

            trees.push(ObliviousTree::new(depth, splits, leaf_values));
        }

        // Metadata
        r.read_exact(&mut f4)?;
        let num_meta = u32::from_le_bytes(f4) as usize;
        let mut metadata = HashMap::with_capacity(num_meta);
        for _ in 0..num_meta {
            r.read_exact(&mut f4)?;
            let klen = u32::from_le_bytes(f4) as usize;
            let mut kbytes = vec![0u8; klen];
            r.read_exact(&mut kbytes)?;
            let k = String::from_utf8(kbytes)
                .map_err(|e| ModelError::CorruptBinary(e.to_string()))?;

            r.read_exact(&mut f4)?;
            let vlen = u32::from_le_bytes(f4) as usize;
            let mut vbytes = vec![0u8; vlen];
            r.read_exact(&mut vbytes)?;
            let v = String::from_utf8(vbytes)
                .map_err(|e| ModelError::CorruptBinary(e.to_string()))?;

            metadata.insert(k, v);
        }

        Ok(Self {
            trees,
            learning_rate,
            base_score,
            target_scale,
            target_offset,
            feature_names,
            feature_borders,
            loss_name,
            metadata,
        })
    }

    /// Exports model as a standalone pure Python script with zero external dependencies.
    pub fn export_python(&self, path: &str) -> Result<(), ModelError> {
        let mut f = File::create(path)?;

        writeln!(f, "# Auto-generated by catboost-webgpu")?;
        writeln!(f, "# Standalone Python predictor with ZERO external dependencies\n")?;
        writeln!(f, "import math\n")?;

        writeln!(f, "class CatBoostPredictor:")?;
        writeln!(f, "    \"\"\"High-performance Oblivious Tree Predictor generated by catboost-webgpu.\"\"\"")?;
        writeln!(f, "    BASE_SCORE = {:.8}", self.base_score)?;
        writeln!(f, "    LEARNING_RATE = {:.8}", self.learning_rate)?;
        writeln!(f, "    TARGET_SCALE = {:.8}", self.target_scale)?;
        writeln!(f, "    TARGET_OFFSET = {:.8}", self.target_offset)?;
        writeln!(f, "    LOSS_NAME = \"{}\"", self.loss_name)?;
        writeln!(f, "    NUM_TREES = {}", self.trees.len())?;

        // Feature names
        writeln!(f, "    FEATURE_NAMES = [")?;
        for name in &self.feature_names {
            writeln!(f, "        \"{}\",", name)?;
        }
        writeln!(f, "    ]\n")?;

        // Splits table: list of [(feature_idx, continuous_threshold)] per tree
        writeln!(f, "    TREES_SPLITS = [")?;
        for tree in &self.trees {
            write!(f, "        [")?;
            for split in &tree.splits {
                write!(f, "({}, {:.8}), ", split.feature_idx, split.continuous_threshold)?;
            }
            writeln!(f, "],")?;
        }
        writeln!(f, "    ]\n")?;

        // Leaf values table: list of leaf value lists per tree
        writeln!(f, "    TREES_LEAF_VALUES = [")?;
        for tree in &self.trees {
            write!(f, "        [")?;
            for val in &tree.leaf_values {
                write!(f, "{:.8}, ", val)?;
            }
            writeln!(f, "],")?;
        }
        writeln!(f, "    ]\n")?;

        // Predictor methods
        writeln!(f, "    @staticmethod")?;
        writeln!(f, "    def _sigmoid(z):")?;
        writeln!(f, "        if z >= 0:")?;
        writeln!(f, "            return 1.0 / (1.0 + math.exp(-z))")?;
        writeln!(f, "        else:")?;
        writeln!(f, "            ez = math.exp(z)")?;
        writeln!(f, "            return ez / (1.0 + ez)\n")?;

        writeln!(f, "    @classmethod")?;
        writeln!(f, "    def predict_raw(cls, sample):")?;
        writeln!(f, "        pred = cls.BASE_SCORE")?;
        writeln!(f, "        for splits, leaf_vals in zip(cls.TREES_SPLITS, cls.TREES_LEAF_VALUES):")?;
        writeln!(f, "            leaf = 0")?;
        writeln!(f, "            for d, (feat_idx, thresh) in enumerate(splits):")?;
        writeln!(f, "                if sample[feat_idx] > thresh:")?;
        writeln!(f, "                    leaf |= (1 << d)")?;
        writeln!(f, "            pred += cls.LEARNING_RATE * leaf_vals[leaf]")?;
        writeln!(f, "        return pred\n")?;

        writeln!(f, "    @classmethod")?;
        writeln!(f, "    def predict(cls, sample):")?;
        writeln!(f, "        raw = cls.predict_raw(sample)")?;
        writeln!(f, "        return raw * cls.TARGET_SCALE + cls.TARGET_OFFSET\n")?;

        writeln!(f, "    @classmethod")?;
        writeln!(f, "    def predict_proba(cls, sample):")?;
        writeln!(f, "        raw = cls.predict_raw(sample)")?;
        writeln!(f, "        p = cls._sigmoid(raw)")?;
        writeln!(f, "        return [1.0 - p, p]\n")?;

        writeln!(f, "    @classmethod")?;
        writeln!(f, "    def predict_batch(cls, samples):")?;
        writeln!(f, "        return [cls.predict(s) for s in samples]\n")?;

        writeln!(f, "    @classmethod")?;
        writeln!(f, "    def predict_proba_batch(cls, samples):")?;
        writeln!(f, "        return [cls.predict_proba(s) for s in samples]\n")?;

        // Verification block
        writeln!(f, "if __name__ == '__main__':")?;
        writeln!(f, "    print(f'CatBoostPredictor loaded successfully with {{CatBoostPredictor.NUM_TREES}} oblivious trees.')")?;

        f.flush()?;
        Ok(())
    }
}

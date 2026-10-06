use rayon::prelude::*;
use crate::traits::{ComputeBackend, SplitCandidate};
use rand::Rng;
use rand_distr::StandardNormal;

/// High-performance multi-threaded CPU compute engine using Rayon.
#[derive(Debug, Clone, Default)]
pub struct CpuEngine;

impl CpuEngine {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates all candidate splits across features and bins in parallel to find the best oblivious split.
    ///
    /// For depth `d`, there are `num_leaves = 2^d` leaves.
    /// Returns the split candidate with the highest regularization-adjusted gain.
    pub fn find_best_split(
        &self,
        histograms: &[f32],
        num_features: usize,
        num_leaves: usize,
        max_bins: usize,
        l2_leaf_reg: f32,
        random_strength: f32,
        seed: u64,
    ) -> Option<SplitCandidate> {
        if num_features == 0 || num_leaves == 0 || max_bins <= 1 {
            return None;
        }

        let feat_stride = num_leaves * max_bins * 2;
        assert_eq!(
            histograms.len(),
            num_features * feat_stride,
            "Histograms buffer size mismatch"
        );

        // Precompute total gradient and hessian per leaf for base score calculation
        // Total G and H across all bins for each leaf are identical across all features.
        // We compute it once using feature 0 (or sum up).
        let mut base_g = vec![0.0f32; num_leaves];
        let mut base_h = vec![0.0f32; num_leaves];
        for l in 0..num_leaves {
            let mut sum_g = 0.0f32;
            let mut sum_h = 0.0f32;
            for b in 0..max_bins {
                let idx = (0 * num_leaves + l) * max_bins * 2 + b * 2;
                sum_g += histograms[idx];
                sum_h += histograms[idx + 1];
            }
            base_g[l] = sum_g;
            base_h[l] = sum_h;
        }

        let mut base_score = 0.0f32;
        for l in 0..num_leaves {
            let denom = base_h[l] + l2_leaf_reg;
            if denom > 1e-12 {
                base_score += (base_g[l] * base_g[l]) / denom;
            }
        }

        // Parallel search across features
        let best_candidate = (0..num_features)
            .into_par_iter()
            .filter_map(|f| {
                let mut local_rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed.wrapping_add(f as u64 * 10007 + 1));
                let f_offset = f * feat_stride;
                let mut best_f_cand: Option<SplitCandidate> = None;

                // For each leaf, we maintain cumulative sums as we sweep through bins
                // cum_g[l], cum_h[l]
                let mut cum_g = vec![0.0f32; num_leaves];
                let mut cum_h = vec![0.0f32; num_leaves];

                // For each bin threshold b in 0..(max_bins - 1)
                for b in 0..(max_bins - 1) {
                    for l in 0..num_leaves {
                        let idx = f_offset + l * max_bins * 2 + b * 2;
                        cum_g[l] += histograms[idx];
                        cum_h[l] += histograms[idx + 1];
                    }

                    let mut split_score = 0.0f32;
                    let mut valid = false;

                    for l in 0..num_leaves {
                        let g_left = cum_g[l];
                        let h_left = cum_h[l];
                        let g_right = base_g[l] - g_left;
                        let h_right = base_h[l] - h_left;

                        let denom_l = h_left + l2_leaf_reg;
                        let denom_r = h_right + l2_leaf_reg;

                        if denom_l > 1e-12 && denom_r > 1e-12 {
                            split_score += (g_left * g_left) / denom_l + (g_right * g_right) / denom_r;
                            valid = true;
                        }
                    }

                    if valid {
                        let raw_gain = split_score - base_score;
                        let noise: f32 = if random_strength > 0.0 {
                            let n: f32 = local_rng.sample(StandardNormal);
                            n * random_strength * raw_gain.abs().max(1e-4)
                        } else {
                            0.0
                        };
                        let total_gain = raw_gain + noise;

                        if best_f_cand.is_none() || total_gain > best_f_cand.unwrap().gain {
                            best_f_cand = Some(SplitCandidate {
                                feature_idx: f,
                                bin_threshold: b as u8,
                                gain: total_gain,
                            });
                        }
                    }
                }

                best_f_cand
            })
            .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap_or(std::cmp::Ordering::Equal));

        best_candidate
    }

    /// Computes optimal leaf values given final leaf partitioning:
    /// leaf_value[l] = - sum_g[l] / (sum_h[l] + l2_leaf_reg)
    pub fn compute_leaf_values(
        &self,
        leaf_indices: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        num_leaves: usize,
        l2_leaf_reg: f32,
    ) -> Vec<f32> {
        assert_eq!(leaf_indices.len(), gradients.len());
        assert_eq!(gradients.len(), hessians.len());

        let num_samples = leaf_indices.len();
        if num_samples == 0 || num_leaves == 0 {
            return vec![0.0; num_leaves];
        }

        // Parallel chunked accumulation of sum_g and sum_h per leaf
        let num_threads = rayon::current_num_threads();
        let chunk_size = (num_samples + num_threads - 1) / num_threads;

        let (total_g, total_h) = leaf_indices
            .par_chunks(chunk_size)
            .zip(gradients.par_chunks(chunk_size))
            .zip(hessians.par_chunks(chunk_size))
            .fold(
                || (vec![0.0f32; num_leaves], vec![0.0f32; num_leaves]),
                |(mut local_g, mut local_h), ((l_chunk, g_chunk), h_chunk)| {
                    for i in 0..l_chunk.len() {
                        let leaf = l_chunk[i] as usize;
                        if leaf < num_leaves {
                            local_g[leaf] += g_chunk[i];
                            local_h[leaf] += h_chunk[i];
                        }
                    }
                    (local_g, local_h)
                },
            )
            .reduce(
                || (vec![0.0f32; num_leaves], vec![0.0f32; num_leaves]),
                |(mut g1, mut h1), (g2, h2)| {
                    for l in 0..num_leaves {
                        g1[l] += g2[l];
                        h1[l] += h2[l];
                    }
                    (g1, h1)
                },
            );

        let mut leaf_values = vec![0.0f32; num_leaves];
        for l in 0..num_leaves {
            let denom = total_h[l] + l2_leaf_reg;
            if denom > 1e-12 {
                leaf_values[l] = -total_g[l] / denom;
            } else {
                leaf_values[l] = 0.0;
            }
        }

        leaf_values
    }
}

impl ComputeBackend for CpuEngine {
    fn name(&self) -> &str {
        "CPU (Rayon)"
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn compute_histograms(
        &self,
        binned_data: &[u8],
        leaf_indices: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        num_samples: usize,
        num_features: usize,
        num_leaves: usize,
        max_bins: usize,
    ) -> Vec<f32> {
        let total_size = num_features * num_leaves * max_bins * 2;
        let mut histograms = vec![0.0f32; total_size];

        if num_samples == 0 || num_features == 0 || num_leaves == 0 || max_bins == 0 {
            return histograms;
        }

        assert_eq!(binned_data.len(), num_samples * num_features);
        assert_eq!(leaf_indices.len(), num_samples);
        assert_eq!(gradients.len(), num_samples);
        assert_eq!(hessians.len(), num_samples);

        let feat_stride = num_leaves * max_bins * 2;
        let num_threads = rayon::current_num_threads();

        // If we have enough features relative to threads, parallelize by features
        if num_features >= num_threads {
            histograms
                .par_chunks_exact_mut(feat_stride)
                .enumerate()
                .for_each(|(f, feat_hist)| {
                    for i in 0..num_samples {
                        let leaf = leaf_indices[i] as usize;
                        let bin = binned_data[i * num_features + f] as usize;
                        if leaf < num_leaves && bin < max_bins {
                            let idx = (leaf * max_bins + bin) * 2;
                            feat_hist[idx] += gradients[i];
                            feat_hist[idx + 1] += hessians[i];
                        }
                    }
                });
        } else {
            // Parallelize across sample chunks and reduce
            let chunk_size = ((num_samples + num_threads - 1) / num_threads).max(256);

            let chunk_hists: Vec<Vec<f32>> = (0..num_samples)
                .into_par_iter()
                .chunks(chunk_size)
                .map(|sample_indices| {
                    let mut local_hist = vec![0.0f32; total_size];
                    for i in sample_indices {
                        let leaf = leaf_indices[i] as usize;
                        let g = gradients[i];
                        let h = hessians[i];
                        let row_offset = i * num_features;

                        if leaf < num_leaves {
                            for f in 0..num_features {
                                let bin = binned_data[row_offset + f] as usize;
                                if bin < max_bins {
                                    let idx = (f * num_leaves + leaf) * max_bins * 2 + bin * 2;
                                    local_hist[idx] += g;
                                    local_hist[idx + 1] += h;
                                }
                            }
                        }
                    }
                    local_hist
                })
                .collect();

            // Parallel sum of chunk histograms
            histograms
                .par_chunks_mut(feat_stride)
                .enumerate()
                .for_each(|(f, feat_hist)| {
                    let start = f * feat_stride;
                    let end = start + feat_stride;
                    for ch in &chunk_hists {
                        for (dst, &src) in feat_hist.iter_mut().zip(&ch[start..end]) {
                            *dst += src;
                        }
                    }
                });
        }

        histograms
    }

    fn partition_leaves(
        &self,
        binned_data: &[u8],
        leaf_indices: &mut [u32],
        feature_idx: usize,
        bin_threshold: u8,
        depth: usize,
        num_samples: usize,
        num_features: usize,
    ) {
        assert_eq!(leaf_indices.len(), num_samples);
        assert_eq!(binned_data.len(), num_samples * num_features);
        assert!(feature_idx < num_features);

        let bit_shift = depth as u32;
        const CHUNK_SIZE: usize = 2048;

        leaf_indices
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let offset = chunk_idx * CHUNK_SIZE;
                for (j, leaf) in chunk.iter_mut().enumerate() {
                    let sample_idx = offset + j;
                    let bin = binned_data[sample_idx * num_features + feature_idx];
                    if bin > bin_threshold {
                        *leaf |= 1u32 << bit_shift;
                    }
                }
            });
    }

    fn update_predictions(
        &self,
        predictions: &mut [f32],
        leaf_indices: &[u32],
        leaf_values: &[f32],
        learning_rate: f32,
    ) {
        assert_eq!(predictions.len(), leaf_indices.len());
        const CHUNK_SIZE: usize = 2048;

        predictions
            .par_chunks_mut(CHUNK_SIZE)
            .zip(leaf_indices.par_chunks(CHUNK_SIZE))
            .for_each(|(p_chunk, l_chunk)| {
                for (p, &leaf) in p_chunk.iter_mut().zip(l_chunk) {
                    *p += learning_rate * leaf_values[leaf as usize];
                }
            });
    }
}

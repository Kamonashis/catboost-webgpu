use std::sync::Arc;
use crate::traits::LossFunction;

/// Helper: numerically stable sigmoid
#[inline]
pub fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

// ============================================================================
// Regression Objectives
// ============================================================================

/// Root Mean Squared Error (RMSE) / Squared Loss: L = 0.5 * (y_pred - y_true)^2
#[derive(Debug, Clone, Copy, Default)]
pub struct RMSELoss;

impl LossFunction for RMSELoss {
    fn name(&self) -> &str {
        "RMSE"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        (y_pred - y_true, 1.0)
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        if y_true.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| {
                let diff = yp - yt;
                diff * diff
            })
            .sum();
        (sum_sq / y_true.len() as f32).sqrt()
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Mean Absolute Error (MAE): L = |y_pred - y_true|
#[derive(Debug, Clone, Copy, Default)]
pub struct MAELoss;

impl LossFunction for MAELoss {
    fn name(&self) -> &str {
        "MAE"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        let diff = y_pred - y_true;
        let g = if diff > 0.0 {
            1.0
        } else if diff < 0.0 {
            -1.0
        } else {
            0.0
        };
        (g, 1.0)
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        if y_true.is_empty() {
            return 0.0;
        }
        let sum_abs: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| (yp - yt).abs())
            .sum();
        sum_abs / y_true.len() as f32
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Mean Absolute Percentage Error (MAPE): L = |(y_true - y_pred) / y_true|
#[derive(Debug, Clone, Copy, Default)]
pub struct MAPELoss;

impl LossFunction for MAPELoss {
    fn name(&self) -> &str {
        "MAPE"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        let denom = y_true.abs().max(1e-6);
        let diff = y_pred - y_true;
        let sign = if diff > 0.0 {
            1.0
        } else if diff < 0.0 {
            -1.0
        } else {
            0.0
        };
        (sign / denom, 1.0 / denom)
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        if y_true.is_empty() {
            return 0.0;
        }
        let sum_pct: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| ((yp - yt) / yt.abs().max(1e-6)).abs())
            .sum();
        100.0 * sum_pct / y_true.len() as f32
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Huber Loss with transition threshold delta
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    pub delta: f32,
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl HuberLoss {
    pub fn new(delta: f32) -> Self {
        assert!(delta > 0.0, "Huber delta must be positive");
        Self { delta }
    }
}

impl LossFunction for HuberLoss {
    fn name(&self) -> &str {
        "Huber"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        let diff = y_pred - y_true;
        let abs_diff = diff.abs();
        if abs_diff <= self.delta {
            (diff, 1.0)
        } else {
            (self.delta * diff.signum(), 1.0)
        }
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        if y_true.is_empty() {
            return 0.0;
        }
        let sum_loss: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| {
                let diff = (yp - yt).abs();
                if diff <= self.delta {
                    0.5 * diff * diff
                } else {
                    self.delta * (diff - 0.5 * self.delta)
                }
            })
            .sum();
        sum_loss / y_true.len() as f32
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Quantile (Pinball) Loss for quantile regression at alpha in (0, 1)
#[derive(Debug, Clone, Copy)]
pub struct QuantileLoss {
    pub alpha: f32,
}

impl Default for QuantileLoss {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

impl QuantileLoss {
    pub fn new(alpha: f32) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "Quantile alpha must be in (0, 1)"
        );
        Self { alpha }
    }
}

impl LossFunction for QuantileLoss {
    fn name(&self) -> &str {
        "Quantile"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        let diff = y_pred - y_true;
        let g = if diff >= 0.0 {
            1.0 - self.alpha
        } else {
            -self.alpha
        };
        (g, 1.0)
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        if y_true.is_empty() {
            return 0.0;
        }
        let sum_loss: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| {
                let diff = yt - yp;
                if diff >= 0.0 {
                    self.alpha * diff
                } else {
                    (self.alpha - 1.0) * diff
                }
            })
            .sum();
        sum_loss / y_true.len() as f32
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Poisson Regression Loss: L = exp(y_pred) - y_true * y_pred
#[derive(Debug, Clone, Copy, Default)]
pub struct PoissonLoss;

impl LossFunction for PoissonLoss {
    fn name(&self) -> &str {
        "Poisson"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        let lambda = y_pred.clamp(-30.0, 30.0).exp();
        let g = lambda - y_true.max(0.0);
        let h = lambda.clamp(1e-16, 1e16);
        (g, h)
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        if y_true.is_empty() {
            return 0.0;
        }
        // Poisson deviance: 2 * sum(y * ln(y / lambda) - (y - lambda))
        let sum_deviance: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| {
                let y = yt.max(0.0);
                let lambda = yp.clamp(-30.0, 30.0).exp();
                if y > 1e-12 {
                    2.0 * (y * (y / lambda.max(1e-12)).ln() - (y - lambda))
                } else {
                    2.0 * lambda
                }
            })
            .sum();
        sum_deviance / y_true.len() as f32
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

// ============================================================================
// Classification Objectives
// ============================================================================

/// Binary Logloss (Logistic loss): L = - [y * ln(p) + (1-y) * ln(1-p)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Logloss;

impl LossFunction for Logloss {
    fn name(&self) -> &str {
        "Logloss"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        let p = sigmoid(y_pred);
        let g = p - y_true;
        let h = (p * (1.0 - p)).max(1e-16);
        (g, h)
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        if y_true.is_empty() {
            return 0.0;
        }
        let sum_ll: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| {
                let p = sigmoid(yp).clamp(1e-15, 1.0 - 1e-15);
                -(yt * p.ln() + (1.0 - yt) * (1.0 - p).ln())
            })
            .sum();
        sum_ll / y_true.len() as f32
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Binary CrossEntropy for soft probability labels y in [0.0, 1.0]
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossEntropy;

impl LossFunction for CrossEntropy {
    fn name(&self) -> &str {
        "CrossEntropy"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        let p = sigmoid(y_pred);
        let g = p - y_true.clamp(0.0, 1.0);
        let h = (p * (1.0 - p)).max(1e-16);
        (g, h)
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        if y_true.is_empty() {
            return 0.0;
        }
        let sum_ce: f32 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| {
                let p = sigmoid(yp).clamp(1e-15, 1.0 - 1e-15);
                -(yt * p.ln() + (1.0 - yt) * (1.0 - p).ln())
            })
            .sum();
        sum_ce / y_true.len() as f32
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// MultiClass Softmax Loss for K classes
#[derive(Debug, Clone)]
pub struct MultiClassLoss {
    pub num_classes: usize,
}

impl MultiClassLoss {
    pub fn new(num_classes: usize) -> Self {
        assert!(num_classes >= 2, "MultiClass requires num_classes >= 2");
        Self { num_classes }
    }
}

impl LossFunction for MultiClassLoss {
    fn name(&self) -> &str {
        "MultiClass"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        // Fallback for single class scalar: treat as binary logit
        let p = sigmoid(y_pred);
        (p - y_true, (p * (1.0 - p)).max(1e-16))
    }

    fn compute_gradients_hessians(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        let k = self.num_classes;
        let n = y_true.len();

        if y_pred.len() == n * k {
            assert_eq!(gradients.len(), n * k);
            assert_eq!(hessians.len(), n * k);

            for i in 0..n {
                let row_start = i * k;
                let logits = &y_pred[row_start..row_start + k];
                let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let mut sum_exp = 0.0f32;
                for j in 0..k {
                    sum_exp += (logits[j] - max_logit).exp();
                }

                let target_class = y_true[i] as usize;
                for j in 0..k {
                    let p = (logits[j] - max_logit).exp() / sum_exp.max(1e-16);
                    let target = if j == target_class { 1.0 } else { 0.0 };
                    let idx = row_start + j;
                    gradients[idx] = p - target;
                    hessians[idx] = (p * (1.0 - p)).max(1e-16);
                }
            }
        } else {
            // Standard 1D matching
            assert_eq!(y_pred.len(), n);
            for i in 0..n {
                let (g, h) = self.gradient_hessian(y_true[i], y_pred[i]);
                gradients[i] = g;
                hessians[i] = h;
            }
        }
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        let k = self.num_classes;
        let n = y_true.len();
        if n == 0 {
            return 0.0;
        }

        if y_pred.len() == n * k {
            let mut total_loss = 0.0f32;
            for i in 0..n {
                let row_start = i * k;
                let logits = &y_pred[row_start..row_start + k];
                let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let mut sum_exp = 0.0f32;
                for j in 0..k {
                    sum_exp += (logits[j] - max_logit).exp();
                }

                let target_class = y_true[i] as usize;
                let target_logit = if target_class < k {
                    logits[target_class]
                } else {
                    0.0
                };
                let p = ((target_logit - max_logit).exp() / sum_exp.max(1e-16)).clamp(1e-15, 1.0);
                total_loss -= p.ln();
            }
            total_loss / n as f32
        } else {
            // Binary fallback
            let sum_ll: f32 = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(&yt, &yp)| {
                    let p = sigmoid(yp).clamp(1e-15, 1.0 - 1e-15);
                    -(yt * p.ln() + (1.0 - yt) * (1.0 - p).ln())
                })
                .sum();
            sum_ll / n as f32
        }
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

// ============================================================================
// Ranking Objectives
// ============================================================================

/// PairLogit: Pairwise ranking loss using logistic loss on paired preferences.
///
/// Within each query group, for all pairs (i, j) with y_i > y_j:
/// Loss = log(1 + exp(-(y_pred_i - y_pred_j))).
#[derive(Debug, Clone, Default)]
pub struct PairLogit {
    /// Sizes of consecutive groups/queries in the dataset.
    pub group_sizes: Vec<usize>,
}

impl PairLogit {
    pub fn new(group_sizes: Vec<usize>) -> Self {
        Self { group_sizes }
    }

    /// Helper to partition slices by group sizes, or treat as one group if empty.
    fn get_groups(&self, total_samples: usize) -> Vec<(usize, usize)> {
        if self.group_sizes.is_empty() {
            vec![(0, total_samples)]
        } else {
            let mut groups = Vec::with_capacity(self.group_sizes.len());
            let mut offset = 0;
            for &size in &self.group_sizes {
                let end = (offset + size).min(total_samples);
                if end > offset {
                    groups.push((offset, end));
                }
                offset = end;
                if offset >= total_samples {
                    break;
                }
            }
            if offset < total_samples {
                groups.push((offset, total_samples));
            }
            groups
        }
    }
}

impl LossFunction for PairLogit {
    fn name(&self) -> &str {
        "PairLogit"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        // Fallback for single sample
        (y_pred - y_true, 1.0)
    }

    fn compute_gradients_hessians(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        assert_eq!(y_true.len(), y_pred.len());
        let n = y_true.len();
        gradients.fill(0.0);
        hessians.fill(0.0);

        let groups = self.get_groups(n);
        for (start, end) in groups {
            let group_len = end - start;
            if group_len <= 1 {
                for i in start..end {
                    hessians[i] = 1.0;
                }
                continue;
            }

            for i in start..end {
                for j in (i + 1)..end {
                    let diff_y = y_true[i] - y_true[j];
                    if diff_y.abs() < 1e-7 {
                        continue;
                    }

                    // (pos_idx, neg_idx)
                    let (pos, neg) = if diff_y > 0.0 { (i, j) } else { (j, i) };
                    let diff_pred = y_pred[pos] - y_pred[neg];
                    let p = sigmoid(diff_pred); // p that pos > neg
                    // Loss = -log(p) = log(1 + exp(-diff_pred))
                    // dL/d(diff_pred) = -(1 - p)
                    let grad = -(1.0 - p);
                    let hess = (p * (1.0 - p)).max(1e-16);

                    // dL/d(pos) = grad, dL/d(neg) = -grad
                    gradients[pos] += grad;
                    gradients[neg] -= grad;
                    hessians[pos] += hess;
                    hessians[neg] += hess;
                }
            }

            // Ensure minimal hessian for numerical stability
            for i in start..end {
                if hessians[i] < 1e-8 {
                    hessians[i] = 1.0;
                }
            }
        }
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        let n = y_true.len();
        if n == 0 {
            return 0.0;
        }

        let groups = self.get_groups(n);
        let mut total_loss = 0.0f32;
        let mut total_pairs = 0usize;

        for (start, end) in groups {
            for i in start..end {
                for j in (i + 1)..end {
                    let diff_y = y_true[i] - y_true[j];
                    if diff_y.abs() < 1e-7 {
                        continue;
                    }
                    let (pos, neg) = if diff_y > 0.0 { (i, j) } else { (j, i) };
                    let diff_pred = y_pred[pos] - y_pred[neg];
                    let p = sigmoid(diff_pred).clamp(1e-15, 1.0);
                    total_loss -= p.ln();
                    total_pairs += 1;
                }
            }
        }

        if total_pairs > 0 {
            total_loss / total_pairs as f32
        } else {
            0.0
        }
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

/// QueryRMSE: Group-mean centered RMSE for ranking.
///
/// Within each query group q:
/// shift = mean_{i in q}(y_true[i] - y_pred[i])
/// Sample error = y_true[i] - (y_pred[i] + shift).
#[derive(Debug, Clone, Default)]
pub struct QueryRMSE {
    pub group_sizes: Vec<usize>,
}

impl QueryRMSE {
    pub fn new(group_sizes: Vec<usize>) -> Self {
        Self { group_sizes }
    }

    fn get_groups(&self, total_samples: usize) -> Vec<(usize, usize)> {
        if self.group_sizes.is_empty() {
            vec![(0, total_samples)]
        } else {
            let mut groups = Vec::with_capacity(self.group_sizes.len());
            let mut offset = 0;
            for &size in &self.group_sizes {
                let end = (offset + size).min(total_samples);
                if end > offset {
                    groups.push((offset, end));
                }
                offset = end;
                if offset >= total_samples {
                    break;
                }
            }
            if offset < total_samples {
                groups.push((offset, total_samples));
            }
            groups
        }
    }
}

impl LossFunction for QueryRMSE {
    fn name(&self) -> &str {
        "QueryRMSE"
    }

    #[inline]
    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        (y_pred - y_true, 1.0)
    }

    fn compute_gradients_hessians(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        assert_eq!(y_true.len(), y_pred.len());
        let n = y_true.len();
        let groups = self.get_groups(n);

        for (start, end) in groups {
            let group_len = end - start;
            if group_len == 0 {
                continue;
            }

            let mut sum_diff = 0.0f32;
            for i in start..end {
                sum_diff += y_true[i] - y_pred[i];
            }
            let shift = sum_diff / group_len as f32;

            for i in start..end {
                // error = (y_pred[i] + shift) - y_true[i]
                gradients[i] = y_pred[i] + shift - y_true[i];
                hessians[i] = 1.0;
            }
        }
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        assert_eq!(y_true.len(), y_pred.len());
        let n = y_true.len();
        if n == 0 {
            return 0.0;
        }

        let groups = self.get_groups(n);
        let mut total_sq_err = 0.0f32;

        for (start, end) in groups {
            let group_len = end - start;
            if group_len == 0 {
                continue;
            }

            let mut sum_diff = 0.0f32;
            for i in start..end {
                sum_diff += y_true[i] - y_pred[i];
            }
            let shift = sum_diff / group_len as f32;

            for i in start..end {
                let err = y_true[i] - (y_pred[i] + shift);
                total_sq_err += err * err;
            }
        }

        (total_sq_err / n as f32).sqrt()
    }

    fn lower_is_better(&self) -> bool {
        true
    }
}

// ============================================================================
// Custom Callback Objective Adapter
// ============================================================================

pub type CustomLossFn = Arc<dyn Fn(&[f32], &[f32], &mut [f32], &mut [f32]) + Send + Sync>;
pub type CustomMetricFn = Arc<dyn Fn(&[f32], &[f32]) -> f32 + Send + Sync>;

/// Flexible adapter wrapping user-provided callbacks for gradients, hessians, and metrics.
#[derive(Clone)]
pub struct CustomObjective {
    pub name_str: String,
    pub compute_fn: CustomLossFn,
    pub metric_fn: CustomMetricFn,
    pub is_lower_better: bool,
}

impl CustomObjective {
    pub fn new<F, M>(name: &str, compute_fn: F, metric_fn: M, lower_is_better: bool) -> Self
    where
        F: Fn(&[f32], &[f32], &mut [f32], &mut [f32]) + Send + Sync + 'static,
        M: Fn(&[f32], &[f32]) -> f32 + Send + Sync + 'static,
    {
        Self {
            name_str: name.to_string(),
            compute_fn: Arc::new(compute_fn),
            metric_fn: Arc::new(metric_fn),
            is_lower_better: lower_is_better,
        }
    }
}

impl LossFunction for CustomObjective {
    fn name(&self) -> &str {
        &self.name_str
    }

    fn gradient_hessian(&self, y_true: f32, y_pred: f32) -> (f32, f32) {
        let mut g = [0.0f32];
        let mut h = [1.0f32];
        (self.compute_fn)(&[y_true], &[y_pred], &mut g, &mut h);
        (g[0], h[0])
    }

    fn compute_gradients_hessians(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        (self.compute_fn)(y_true, y_pred, gradients, hessians);
    }

    fn evaluate_metric(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        (self.metric_fn)(y_true, y_pred)
    }

    fn lower_is_better(&self) -> bool {
        self.is_lower_better
    }
}

/// Factory function creating a standard loss function by name.
pub fn create_loss_function(name: &str) -> Result<Box<dyn LossFunction>, String> {
    match name.to_lowercase().as_str() {
        "rmse" | "squared_error" | "regression" => Ok(Box::new(RMSELoss)),
        "mae" | "l1" => Ok(Box::new(MAELoss)),
        "mape" => Ok(Box::new(MAPELoss)),
        "huber" => Ok(Box::new(HuberLoss::default())),
        "quantile" => Ok(Box::new(QuantileLoss::default())),
        "poisson" => Ok(Box::new(PoissonLoss)),
        "logloss" | "binary" => Ok(Box::new(Logloss)),
        "crossentropy" => Ok(Box::new(CrossEntropy)),
        "pairlogit" => Ok(Box::new(PairLogit::default())),
        "queryrmse" => Ok(Box::new(QueryRMSE::default())),
        _ => Err(format!("Unknown loss function: {}", name)),
    }
}

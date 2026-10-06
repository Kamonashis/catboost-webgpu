struct SplitEvalParams {
    num_features: u32,
    num_leaves: u32,
    max_bins: u32,
    l2_leaf_reg: f32,
};

@group(0) @binding(0) var<uniform> params: SplitEvalParams;
@group(0) @binding(1) var<storage, read> histograms: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cand_idx = global_id.x;
    let total_candidates = params.num_features * params.max_bins;

    if (cand_idx >= total_candidates) {
        return;
    }

    let f = cand_idx / params.max_bins;
    let b = cand_idx % params.max_bins;

    // A border threshold b on the last bin is not a valid split (all samples <= b)
    if (b >= params.max_bins - 1u) {
        scores[cand_idx] = -1.0e30;
        return;
    }

    let feat_offset = f * (params.num_leaves * params.max_bins * 2u);
    var base_score: f32 = 0.0;
    var split_score: f32 = 0.0;
    var valid: bool = false;

    // Evaluate split score across all oblivious tree leaves
    for (var l: u32 = 0u; l < params.num_leaves; l = l + 1u) {
        let leaf_offset = feat_offset + l * (params.max_bins * 2u);
        var g_total: f32 = 0.0;
        var h_total: f32 = 0.0;
        var g_left: f32 = 0.0;
        var h_left: f32 = 0.0;

        for (var bin: u32 = 0u; bin < params.max_bins; bin = bin + 1u) {
            let idx = leaf_offset + bin * 2u;
            let g = histograms[idx];
            let h = histograms[idx + 1u];
            g_total = g_total + g;
            h_total = h_total + h;
            if (bin <= b) {
                g_left = g_left + g;
                h_left = h_left + h;
            }
        }

        let denom_base = h_total + params.l2_leaf_reg;
        if (denom_base > 1.0e-12) {
            base_score = base_score + (g_total * g_total) / denom_base;
        }

        let g_right = g_total - g_left;
        let h_right = h_total - h_left;

        let denom_l = h_left + params.l2_leaf_reg;
        let denom_r = h_right + params.l2_leaf_reg;

        if (denom_l > 1.0e-12 && denom_r > 1.0e-12) {
            split_score = split_score + (g_left * g_left) / denom_l + (g_right * g_right) / denom_r;
            valid = true;
        }
    }

    if (valid) {
        scores[cand_idx] = split_score - base_score;
    } else {
        scores[cand_idx] = -1.0e30;
    }
}

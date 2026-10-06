struct HistogramParams {
    num_samples: u32,
    num_features: u32,
    num_leaves: u32,
    max_bins: u32,
};

@group(0) @binding(0) var<uniform> params: HistogramParams;
@group(0) @binding(1) var<storage, read> binned_data: array<u32>;
@group(0) @binding(2) var<storage, read> leaf_indices: array<u32>;
@group(0) @binding(3) var<storage, read> gh_pairs: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> histograms: array<atomic<u32>>;

// Atomically adds a float value to a storage buffer element using bitcast CAS loop
fn atomic_add_f32(index: u32, val: f32) {
    if (val == 0.0) {
        return;
    }
    var current_u32 = atomicLoad(&histograms[index]);
    loop {
        let current_f32 = bitcast<f32>(current_u32);
        let new_f32 = current_f32 + val;
        let new_u32 = bitcast<u32>(new_f32);
        let res = atomicCompareExchangeWeak(&histograms[index], current_u32, new_u32);
        if (res.exchanged) {
            break;
        }
        current_u32 = res.old_value;
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let sample_idx = global_id.x;
    let feat_idx = global_id.y;

    if (sample_idx >= params.num_samples || feat_idx >= params.num_features) {
        return;
    }

    let leaf = leaf_indices[sample_idx];
    if (leaf >= params.num_leaves) {
        return;
    }

    // Extract u8 bin from packed u32 binned_data
    let byte_idx = sample_idx * params.num_features + feat_idx;
    let word = binned_data[byte_idx / 4u];
    let bin = (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;

    if (bin >= params.max_bins) {
        return;
    }

    let gh = gh_pairs[sample_idx];
    let g = gh.x;
    let h = gh.y;

    // Output index: ((feature * num_leaves + leaf) * max_bins + bin) * 2
    let base_idx = ((feat_idx * params.num_leaves + leaf) * params.max_bins + bin) * 2u;
    atomic_add_f32(base_idx, g);
    atomic_add_f32(base_idx + 1u, h);
}

@compute @workgroup_size(256)
fn clear_histograms(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_elements = params.num_features * params.num_leaves * params.max_bins * 2u;
    let idx = global_id.x;
    if (idx < total_elements) {
        atomicStore(&histograms[idx], 0u);
    }
}

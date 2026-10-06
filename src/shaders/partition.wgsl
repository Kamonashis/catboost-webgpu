struct PartitionParams {
    num_samples: u32,
    num_features: u32,
    feature_idx: u32,
    bin_threshold: u32,
    depth: u32,
};

@group(0) @binding(0) var<uniform> params: PartitionParams;
@group(0) @binding(1) var<storage, read> binned_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> leaf_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_samples) {
        return;
    }

    // Extract feature bin threshold comparison
    let byte_idx = idx * params.num_features + params.feature_idx;
    let word = binned_data[byte_idx / 4u];
    let bin_val = (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;

    // Oblivious decision tree index update: leaf |= (bin > threshold) << depth
    let bit = select(0u, 1u, bin_val > params.bin_threshold) << params.depth;
    leaf_indices[idx] = leaf_indices[idx] | bit;
}

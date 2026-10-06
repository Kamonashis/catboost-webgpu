struct UpdateParams {
    num_samples: u32,
    learning_rate: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: UpdateParams;
@group(0) @binding(1) var<storage, read_write> predictions: array<f32>;
@group(0) @binding(2) var<storage, read> leaf_indices: array<u32>;
@group(0) @binding(3) var<storage, read> leaf_values: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_samples) {
        return;
    }

    let leaf = leaf_indices[idx];
    predictions[idx] = predictions[idx] + params.learning_rate * leaf_values[leaf];
}

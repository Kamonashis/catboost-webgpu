struct GradientParams {
    num_samples: u32,
    loss_type: u32,
    param1: f32,
    param2: f32,
};

@group(0) @binding(0) var<uniform> params: GradientParams;
@group(0) @binding(1) var<storage, read> y_true: array<f32>;
@group(0) @binding(2) var<storage, read> y_pred: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradients: array<f32>;
@group(0) @binding(4) var<storage, read_write> hessians: array<f32>;

// Numerically stable sigmoid function
fn sigmoid(z: f32) -> f32 {
    if (z >= 0.0) {
        return 1.0 / (1.0 + exp(-z));
    } else {
        let ez = exp(z);
        return ez / (1.0 + ez);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_samples) {
        return;
    }

    let yt = y_true[idx];
    let yp = y_pred[idx];
    var g: f32 = 0.0;
    var h: f32 = 1.0;

    switch (params.loss_type) {
        case 0u: { // RMSE: L = 0.5 * (yp - yt)^2 -> g = yp - yt, h = 1.0
            g = yp - yt;
            h = 1.0;
        }
        case 1u, 3u: { // Logloss / CrossEntropy
            let p = sigmoid(yp);
            g = p - yt;
            h = max(p * (1.0 - p), 1.0e-16);
        }
        case 2u: { // MAE: L = |yp - yt|
            let diff = yp - yt;
            if (diff > 0.0) {
                g = 1.0;
            } else if (diff < 0.0) {
                g = -1.0;
            } else {
                g = 0.0;
            }
            h = 1.0;
        }
        default: {
            g = yp - yt;
            h = 1.0;
        }
    }

    gradients[idx] = g;
    hessians[idx] = h;
}

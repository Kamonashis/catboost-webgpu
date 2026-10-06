//! WebGPU compute engine implementing `ComputeBackend` for GPU acceleration.

use std::sync::Arc;
use crate::cpu_engine::CpuEngine;
use crate::device::{get_or_init_gpu_context, DeviceError, DeviceInfo, GpuContext};
use crate::shaders::{
    COMPUTE_GRADIENTS_WGSL, HISTOGRAM_WGSL, PARTITION_WGSL, SPLIT_EVAL_WGSL,
    UPDATE_PREDICTIONS_WGSL,
};
use crate::traits::{ComputeBackend, SplitCandidate};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HistogramUniforms {
    num_samples: u32,
    num_features: u32,
    num_leaves: u32,
    max_bins: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitEvalUniforms {
    num_features: u32,
    num_leaves: u32,
    max_bins: u32,
    l2_leaf_reg: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PartitionUniforms {
    num_samples: u32,
    num_features: u32,
    feature_idx: u32,
    bin_threshold: u32,
    depth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateUniforms {
    num_samples: u32,
    learning_rate: f32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GradientUniforms {
    num_samples: u32,
    loss_type: u32,
    param1: f32,
    param2: f32,
}

struct GpuPipelines {
    context: Arc<GpuContext>,
    histogram_pipeline: wgpu::ComputePipeline,
    histogram_bgl: wgpu::BindGroupLayout,
    split_eval_pipeline: wgpu::ComputePipeline,
    split_eval_bgl: wgpu::BindGroupLayout,
    partition_pipeline: wgpu::ComputePipeline,
    partition_bgl: wgpu::BindGroupLayout,
    update_predictions_pipeline: wgpu::ComputePipeline,
    update_predictions_bgl: wgpu::BindGroupLayout,
    compute_gradients_pipeline: wgpu::ComputePipeline,
    compute_gradients_bgl: wgpu::BindGroupLayout,
}

impl GpuPipelines {
    fn new(context: Arc<GpuContext>) -> Result<Self, DeviceError> {
        let device = &context.device;

        let hist_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hist_shader"),
            source: wgpu::ShaderSource::Wgsl(HISTOGRAM_WGSL.into()),
        });
        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("histogram_pipeline"),
            layout: None,
            module: &hist_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let histogram_bgl = histogram_pipeline.get_bind_group_layout(0);

        let split_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("split_eval_shader"),
            source: wgpu::ShaderSource::Wgsl(SPLIT_EVAL_WGSL.into()),
        });
        let split_eval_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("split_eval_pipeline"),
            layout: None,
            module: &split_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let split_eval_bgl = split_eval_pipeline.get_bind_group_layout(0);

        let partition_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("partition_shader"),
            source: wgpu::ShaderSource::Wgsl(PARTITION_WGSL.into()),
        });
        let partition_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("partition_pipeline"),
            layout: None,
            module: &partition_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let partition_bgl = partition_pipeline.get_bind_group_layout(0);

        let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("update_predictions_shader"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_PREDICTIONS_WGSL.into()),
        });
        let update_predictions_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_predictions_pipeline"),
            layout: None,
            module: &update_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let update_predictions_bgl = update_predictions_pipeline.get_bind_group_layout(0);

        let grad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_gradients_shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_GRADIENTS_WGSL.into()),
        });
        let compute_gradients_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_gradients_pipeline"),
            layout: None,
            module: &grad_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let compute_gradients_bgl = compute_gradients_pipeline.get_bind_group_layout(0);

        Ok(Self {
            context,
            histogram_pipeline,
            histogram_bgl,
            split_eval_pipeline,
            split_eval_bgl,
            partition_pipeline,
            partition_bgl,
            update_predictions_pipeline,
            update_predictions_bgl,
            compute_gradients_pipeline,
            compute_gradients_bgl,
        })
    }
}

/// Helper function to map a staging buffer, copy into slice, poll indefinitely, and unmap.
fn map_and_read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    out: &mut [T],
) -> Result<(), DeviceError> {
    let slice = staging_buffer.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| DeviceError::MicroProbeFailed(format!("Device poll failed: {:?}", e)))?;

    futures::executor::block_on(rx)
        .map_err(|e| DeviceError::MicroProbeFailed(format!("Channel error: {:?}", e)))?
        .map_err(|e| DeviceError::BufferMapFailed(format!("MapAsync error: {:?}", e)))?;

    let data = slice.get_mapped_range();
    let cast: &[T] = bytemuck::cast_slice(&data);
    out.copy_from_slice(cast);
    drop(data);
    staging_buffer.unmap();
    Ok(())
}

fn map_and_read_to_vec<T: bytemuck::Pod + Default>(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<T>, DeviceError> {
    let mut vec = vec![T::default(); count];
    map_and_read_buffer(device, staging_buffer, &mut vec)?;
    Ok(vec)
}

/// High-performance WebGPU compute engine with seamless CPU fallback.
pub struct WebGpuEngine {
    gpu_pipelines: Option<GpuPipelines>,
    cpu_fallback: CpuEngine,
    name: String,
}

impl WebGpuEngine {
    /// Initializes WebGpuEngine, auto-detecting GPU and falling back gracefully to multi-threaded CPU.
    pub fn new() -> Self {
        if let Some(ctx) = get_or_init_gpu_context() {
            let name = format!("WebGPU: {} ({})", ctx.info.name, ctx.info.backend);
            match GpuPipelines::new(ctx) {
                Ok(pipelines) => {
                    return Self {
                        gpu_pipelines: Some(pipelines),
                        cpu_fallback: CpuEngine::new(),
                        name,
                    };
                }
                Err(_err) => {}
            }
        }

        Self::new_cpu()
    }

    /// Initializes WebGpuEngine explicitly targeting GPU, returning error if unavailable.
    pub fn try_new() -> Result<Self, DeviceError> {
        let ctx = get_or_init_gpu_context().ok_or(DeviceError::NoAdapterFound)?;
        let name = format!("WebGPU: {} ({})", ctx.info.name, ctx.info.backend);
        let pipelines = GpuPipelines::new(ctx)?;
        Ok(Self {
            gpu_pipelines: Some(pipelines),
            cpu_fallback: CpuEngine::new(),
            name,
        })
    }

    /// Creates an explicit CPU fallback engine.
    pub fn new_cpu() -> Self {
        Self {
            gpu_pipelines: None,
            cpu_fallback: CpuEngine::new(),
            name: "CPU (Rayon Fallback)".to_string(),
        }
    }

    /// Returns device information of the active compute backend.
    pub fn device_info(&self) -> DeviceInfo {
        if let Some(ref pipelines) = self.gpu_pipelines {
            pipelines.context.info.clone()
        } else {
            DeviceInfo::cpu_fallback()
        }
    }

    /// Dispatches split evaluation kernel on GPU to compute candidate split gains.
    pub fn evaluate_splits_gpu(
        &self,
        histograms: &[f32],
        num_features: usize,
        num_leaves: usize,
        max_bins: usize,
        l2_leaf_reg: f32,
    ) -> Result<Vec<f32>, DeviceError> {
        let pipelines = self.gpu_pipelines.as_ref().ok_or(DeviceError::NoAdapterFound)?;
        let device = &pipelines.context.device;
        let queue = &pipelines.context.queue;

        let total_candidates = num_features * max_bins;
        if total_candidates == 0 || histograms.is_empty() {
            return Ok(vec![-1e30; total_candidates]);
        }

        let uniforms = SplitEvalUniforms {
            num_features: num_features as u32,
            num_leaves: num_leaves as u32,
            max_bins: max_bins as u32,
            l2_leaf_reg,
        };

        let ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_eval_ubo"),
            size: std::mem::size_of::<SplitEvalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ubo, 0, bytemuck::bytes_of(&uniforms));

        let hist_bytes = (histograms.len() * 4) as u64;
        let hist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_hist_buf"),
            size: hist_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&hist_buf, 0, bytemuck::cast_slice(histograms));

        let scores_bytes = (total_candidates * 4) as u64;
        let scores_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_scores_buf"),
            size: scores_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_scores_staging"),
            size: scores_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_eval_bg"),
            layout: &pipelines.split_eval_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scores_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("split_eval_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("split_eval_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.split_eval_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (total_candidates as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&scores_buf, 0, &staging_buf, 0, scores_bytes);
        queue.submit(Some(encoder.finish()));

        map_and_read_to_vec::<f32>(device, &staging_buf, total_candidates)
    }

    /// Evaluates all candidate splits and returns the optimal split candidate.
    pub fn find_best_split(
        &self,
        histograms: &[f32],
        num_features: usize,
        num_leaves: usize,
        max_bins: usize,
        l2_leaf_reg: f32,
    ) -> Option<SplitCandidate> {
        if self.is_gpu() {
            if let Ok(scores) = self.evaluate_splits_gpu(
                histograms,
                num_features,
                num_leaves,
                max_bins,
                l2_leaf_reg,
            ) {
                let mut best_cand: Option<SplitCandidate> = None;
                for f in 0..num_features {
                    for b in 0..(max_bins.saturating_sub(1)) {
                        let score = scores[f * max_bins + b];
                        if score > 0.0 {
                            if best_cand.is_none() || score > best_cand.unwrap().gain {
                                best_cand = Some(SplitCandidate {
                                    feature_idx: f,
                                    bin_threshold: b as u8,
                                    gain: score,
                                });
                            }
                        }
                    }
                }
                return best_cand;
            }
        }

        // CPU Fallback
        self.cpu_fallback.find_best_split(
            histograms,
            num_features,
            num_leaves,
            max_bins,
            l2_leaf_reg,
            0.0,
            42,
        )
    }

    /// Dispatches on-device loss gradient and hessian computation.
    pub fn compute_gradients_gpu(
        &self,
        y_true: &[f32],
        y_pred: &[f32],
        loss_type: u32,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<(), DeviceError> {
        let pipelines = self.gpu_pipelines.as_ref().ok_or(DeviceError::NoAdapterFound)?;
        let device = &pipelines.context.device;
        let queue = &pipelines.context.queue;

        let n = y_true.len();
        if n == 0 {
            return Ok(());
        }

        assert_eq!(y_pred.len(), n);
        assert_eq!(gradients.len(), n);
        assert_eq!(hessians.len(), n);

        let uniforms = GradientUniforms {
            num_samples: n as u32,
            loss_type,
            param1: 0.0,
            param2: 0.0,
        };

        let ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_ubo"),
            size: std::mem::size_of::<GradientUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ubo, 0, bytemuck::bytes_of(&uniforms));

        let buf_size = (n * 4) as u64;

        let yt_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yt_buf"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&yt_buf, 0, bytemuck::cast_slice(y_true));

        let yp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yp_buf"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&yp_buf, 0, bytemuck::cast_slice(y_pred));

        let g_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("g_buf"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let h_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("h_buf"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_g = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_g"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_h = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_h"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grad_bg"),
            layout: &pipelines.compute_gradients_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: yt_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: yp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: h_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("grad_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grad_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.compute_gradients_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&g_buf, 0, &staging_g, 0, buf_size);
        encoder.copy_buffer_to_buffer(&h_buf, 0, &staging_h, 0, buf_size);
        queue.submit(Some(encoder.finish()));

        map_and_read_buffer::<f32>(device, &staging_g, gradients)?;
        map_and_read_buffer::<f32>(device, &staging_h, hessians)?;

        Ok(())
    }
}

impl Default for WebGpuEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for WebGpuEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_gpu(&self) -> bool {
        self.gpu_pipelines.is_some()
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
        let total_floats = num_features * num_leaves * max_bins * 2;
        if num_samples == 0 || num_features == 0 || num_leaves == 0 || max_bins == 0 {
            return vec![0.0; total_floats];
        }

        // Check if GPU pipeline is available
        if let Some(ref pipelines) = self.gpu_pipelines {
            let device = &pipelines.context.device;
            let queue = &pipelines.context.queue;

            let uniforms = HistogramUniforms {
                num_samples: num_samples as u32,
                num_features: num_features as u32,
                num_leaves: num_leaves as u32,
                max_bins: max_bins as u32,
            };

            let ubo = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hist_ubo"),
                size: std::mem::size_of::<HistogramUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ubo, 0, bytemuck::bytes_of(&uniforms));

            // Pad binned_data slice to multiple of 4 bytes for WebGPU buffer binding
            let padded_len = (binned_data.len() + 3) & !3;
            let mut padded_bytes = binned_data.to_vec();
            padded_bytes.resize(padded_len.max(4), 0u8);

            let data_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hist_data_buf"),
                size: padded_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&data_buf, 0, &padded_bytes);

            let leaf_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hist_leaf_buf"),
                size: ((leaf_indices.len() * 4).max(4)) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&leaf_buf, 0, bytemuck::cast_slice(leaf_indices));

            let mut gh_pairs = Vec::with_capacity(num_samples * 2);
            for i in 0..num_samples {
                gh_pairs.push(gradients[i]);
                gh_pairs.push(hessians[i]);
            }

            let gh_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hist_gh_buf"),
                size: ((gh_pairs.len() * 4).max(4)) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&gh_buf, 0, bytemuck::cast_slice(&gh_pairs));

            let hist_bytes = ((total_floats * 4).max(4)) as u64;
            let hist_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hist_output_buf"),
                size: hist_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Initialize output histogram buffer with zero bits
            let zeros = vec![0u8; hist_bytes as usize];
            queue.write_buffer(&hist_buf, 0, &zeros);

            let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hist_staging_buf"),
                size: hist_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hist_bg"),
                layout: &pipelines.histogram_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: data_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: leaf_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gh_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: hist_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hist_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hist_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.histogram_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroups_x = (num_samples as u32 + 255) / 256;
                let workgroups_y = num_features as u32;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            encoder.copy_buffer_to_buffer(&hist_buf, 0, &staging_buf, 0, hist_bytes);
            queue.submit(Some(encoder.finish()));

            if let Ok(res) = map_and_read_to_vec::<f32>(device, &staging_buf, total_floats) {
                return res;
            }
        }

        // Graceful fallback to CPU
        self.cpu_fallback.compute_histograms(
            binned_data,
            leaf_indices,
            gradients,
            hessians,
            num_samples,
            num_features,
            num_leaves,
            max_bins,
        )
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
        if num_samples == 0 {
            return;
        }

        if let Some(ref pipelines) = self.gpu_pipelines {
            let device = &pipelines.context.device;
            let queue = &pipelines.context.queue;

            let uniforms = PartitionUniforms {
                num_samples: num_samples as u32,
                num_features: num_features as u32,
                feature_idx: feature_idx as u32,
                bin_threshold: bin_threshold as u32,
                depth: depth as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };

            let ubo = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("part_ubo"),
                size: std::mem::size_of::<PartitionUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ubo, 0, bytemuck::bytes_of(&uniforms));

            let padded_len = (binned_data.len() + 3) & !3;
            let mut padded_bytes = binned_data.to_vec();
            padded_bytes.resize(padded_len.max(4), 0u8);

            let data_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("part_data_buf"),
                size: padded_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&data_buf, 0, &padded_bytes);

            let leaf_bytes = ((leaf_indices.len() * 4).max(4)) as u64;
            let leaf_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("part_leaf_buf"),
                size: leaf_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&leaf_buf, 0, bytemuck::cast_slice(leaf_indices));

            let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("part_staging_buf"),
                size: leaf_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("part_bg"),
                layout: &pipelines.partition_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: data_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: leaf_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("part_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("part_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.partition_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = (num_samples as u32 + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&leaf_buf, 0, &staging_buf, 0, leaf_bytes);
            queue.submit(Some(encoder.finish()));

            if map_and_read_buffer::<u32>(device, &staging_buf, leaf_indices).is_ok() {
                return;
            }
        }

        // Graceful fallback to CPU
        self.cpu_fallback.partition_leaves(
            binned_data,
            leaf_indices,
            feature_idx,
            bin_threshold,
            depth,
            num_samples,
            num_features,
        );
    }

    fn update_predictions(
        &self,
        predictions: &mut [f32],
        leaf_indices: &[u32],
        leaf_values: &[f32],
        learning_rate: f32,
    ) {
        let n = predictions.len();
        if n == 0 {
            return;
        }

        if let Some(ref pipelines) = self.gpu_pipelines {
            let device = &pipelines.context.device;
            let queue = &pipelines.context.queue;

            let uniforms = UpdateUniforms {
                num_samples: n as u32,
                learning_rate,
                _pad0: 0,
                _pad1: 0,
            };

            let ubo = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upd_ubo"),
                size: std::mem::size_of::<UpdateUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ubo, 0, bytemuck::bytes_of(&uniforms));

            let pred_bytes = ((n * 4).max(4)) as u64;
            let pred_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upd_pred_buf"),
                size: pred_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&pred_buf, 0, bytemuck::cast_slice(predictions));

            let leaf_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upd_leaf_buf"),
                size: ((leaf_indices.len() * 4).max(4)) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&leaf_buf, 0, bytemuck::cast_slice(leaf_indices));

            let val_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upd_val_buf"),
                size: ((leaf_values.len() * 4).max(4)) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&val_buf, 0, bytemuck::cast_slice(leaf_values));

            let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upd_staging_buf"),
                size: pred_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("upd_bg"),
                layout: &pipelines.update_predictions_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pred_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: leaf_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: val_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("upd_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("upd_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.update_predictions_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = (n as u32 + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&pred_buf, 0, &staging_buf, 0, pred_bytes);
            queue.submit(Some(encoder.finish()));

            if map_and_read_buffer::<f32>(device, &staging_buf, predictions).is_ok() {
                return;
            }
        }

        // Graceful fallback to CPU
        self.cpu_fallback.update_predictions(
            predictions,
            leaf_indices,
            leaf_values,
            learning_rate,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = WebGpuEngine::new();
        println!("Engine initialized: {} (is_gpu={})", engine.name(), engine.is_gpu());
        assert!(!engine.name().is_empty());
    }

    #[test]
    fn test_compute_histograms_consistency() {
        let engine = WebGpuEngine::new();
        let cpu_engine = CpuEngine::new();

        let num_samples = 32;
        let num_features = 2;
        let num_leaves = 2;
        let max_bins = 4;

        // Create sample binned data
        let mut binned_data = vec![0u8; num_samples * num_features];
        for i in 0..num_samples {
            binned_data[i * num_features + 0] = (i % max_bins) as u8;
            binned_data[i * num_features + 1] = ((i / 2) % max_bins) as u8;
        }

        let leaf_indices: Vec<u32> = (0..num_samples).map(|i| (i % 2) as u32).collect();
        let gradients: Vec<f32> = (0..num_samples).map(|i| (i as f32) * 0.1).collect();
        let hessians: Vec<f32> = (0..num_samples).map(|_| 1.0f32).collect();

        let hist_result = engine.compute_histograms(
            &binned_data,
            &leaf_indices,
            &gradients,
            &hessians,
            num_samples,
            num_features,
            num_leaves,
            max_bins,
        );

        let cpu_result = cpu_engine.compute_histograms(
            &binned_data,
            &leaf_indices,
            &gradients,
            &hessians,
            num_samples,
            num_features,
            num_leaves,
            max_bins,
        );

        assert_eq!(hist_result.len(), cpu_result.len());
        for i in 0..hist_result.len() {
            assert!(
                (hist_result[i] - cpu_result[i]).abs() < 1e-4,
                "Mismatch at {}: gpu={} cpu={}",
                i,
                hist_result[i],
                cpu_result[i]
            );
        }
    }

    #[test]
    fn test_partition_leaves_consistency() {
        let engine = WebGpuEngine::new();
        let cpu_engine = CpuEngine::new();

        let num_samples = 16;
        let num_features = 2;
        let mut binned_data = vec![0u8; num_samples * num_features];
        for i in 0..num_samples {
            binned_data[i * num_features + 0] = i as u8;
            binned_data[i * num_features + 1] = (i * 2) as u8;
        }

        let mut gpu_leaves = vec![0u32; num_samples];
        let mut cpu_leaves = vec![0u32; num_samples];

        engine.partition_leaves(&binned_data, &mut gpu_leaves, 0, 7, 0, num_samples, num_features);
        cpu_engine.partition_leaves(&binned_data, &mut cpu_leaves, 0, 7, 0, num_samples, num_features);

        assert_eq!(gpu_leaves, cpu_leaves);
    }

    #[test]
    fn test_update_predictions_consistency() {
        let engine = WebGpuEngine::new();
        let cpu_engine = CpuEngine::new();

        let mut gpu_preds = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut cpu_preds = vec![1.0f32, 2.0, 3.0, 4.0];
        let leaf_indices = vec![0u32, 1, 0, 1];
        let leaf_values = vec![0.5f32, -0.5f32];
        let lr = 0.1f32;

        engine.update_predictions(&mut gpu_preds, &leaf_indices, &leaf_values, lr);
        cpu_engine.update_predictions(&mut cpu_preds, &leaf_indices, &leaf_values, lr);

        for (gp, cp) in gpu_preds.iter().zip(cpu_preds.iter()) {
            assert!((gp - cp).abs() < 1e-5);
        }
    }
}

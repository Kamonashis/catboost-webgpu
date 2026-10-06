//! WebGPU device manager, hardware discovery, and micro-probe verification.

use std::sync::{Arc, OnceLock};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("No compatible WebGPU adapter found on system")]
    NoAdapterFound,
    #[error("Failed to request WebGPU device: {0}")]
    RequestDeviceFailed(String),
    #[error("Micro-probe compute shader execution failed: {0}")]
    MicroProbeFailed(String),
    #[error("Buffer mapping failed: {0}")]
    BufferMapFailed(String),
}

/// Metadata and performance limits of the active compute hardware.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub is_gpu: bool,
    pub max_buffer_size: u64,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_storage_buffer_binding_size: u64,
}

impl DeviceInfo {
    /// Returns default device info representing CPU fallback mode.
    pub fn cpu_fallback() -> Self {
        Self {
            name: "CPU (Rayon Fallback)".to_string(),
            backend: "CPU".to_string(),
            device_type: "Cpu".to_string(),
            is_gpu: false,
            max_buffer_size: u64::MAX,
            max_compute_workgroup_size_x: 1024,
            max_compute_invocations_per_workgroup: 1024,
            max_storage_buffer_binding_size: u64::MAX,
        }
    }
}

/// Encapsulates the initialized WebGPU device, execution queue, and hardware metadata.
#[derive(Debug, Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
    pub info: DeviceInfo,
}

impl GpuContext {
    /// Discovers an available WebGPU adapter (prioritizing Vulkan, Metal, DX12),
    /// initializes a device with downlevel limits, and runs a compute micro-probe.
    pub fn init() -> Result<Self, DeviceError> {
        let instance = wgpu::Instance::default();

        let adapter_opt = futures::executor::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).ok().or_else(|| {
            // Secondary fallback: attempt any backend (e.g. GL or CPU / llvmpipe)
            let fallback_instance = wgpu::Instance::default();
            futures::executor::block_on(fallback_instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })).ok()
        });

        let adapter = adapter_opt.ok_or(DeviceError::NoAdapterFound)?;
        let adapter_info = adapter.get_info();

        let mut desc = wgpu::DeviceDescriptor::default();
        desc.label = Some("catboost-webgpu-device");
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = adapter
            .limits()
            .max_storage_buffers_per_shader_stage
            .max(limits.max_storage_buffers_per_shader_stage);
        desc.required_limits = limits;

        let (device, queue) = futures::executor::block_on(adapter.request_device(&desc))
            .map_err(|e| DeviceError::RequestDeviceFailed(e.to_string()))?;

        // Run compute micro-probe to verify device compute pipeline execution
        run_micro_probe(&device, &queue)?;

        let limits = adapter.limits();
        let is_gpu = match adapter_info.device_type {
            wgpu::DeviceType::DiscreteGpu
            | wgpu::DeviceType::IntegratedGpu
            | wgpu::DeviceType::VirtualGpu => true,
            _ => false,
        };

        let info = DeviceInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
            is_gpu,
            max_buffer_size: limits.max_buffer_size,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
        };

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            info,
        })
    }
}

/// Executes a micro-probe compute shader to verify GPU execution and polling.
pub fn run_micro_probe(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), DeviceError> {
    let probe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe_compute_shader"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var<storage, read_write> test_buf: array<u32>;

            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                test_buf[id.x] = test_buf[id.x] * 2u;
            }
            "#.into(),
        ),
    });

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_storage_buf"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_staging_buf"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let input_val: [u32; 1] = [42];
    queue.write_buffer(&storage_buffer, 0, bytemuck::cast_slice(&input_val));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("probe_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe_bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("probe_pl"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe_pipeline"),
        layout: Some(&pipeline_layout),
        module: &probe_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("probe_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, 4);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| DeviceError::MicroProbeFailed(format!("Device poll failed: {:?}", e)))?;

    let map_res = futures::executor::block_on(receiver)
        .map_err(|e| DeviceError::MicroProbeFailed(format!("Channel error: {:?}", e)))?
        .map_err(|e| DeviceError::BufferMapFailed(format!("MapAsync error: {:?}", e)))?;

    let _ = map_res;
    let data = buffer_slice.get_mapped_range();
    let result: &[u32] = bytemuck::cast_slice(&data);
    let output = result[0];
    drop(data);
    staging_buffer.unmap();

    if output != 84 {
        return Err(DeviceError::MicroProbeFailed(format!(
            "Micro-probe assertion failed: expected 84, got {}",
            output
        )));
    }

    Ok(())
}

static GPU_CONTEXT: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();

/// Returns the shared GPU context, initialized on first invocation.
/// Returns None if no compatible device is found or probe fails.
pub fn get_or_init_gpu_context() -> Option<Arc<GpuContext>> {
    GPU_CONTEXT
        .get_or_init(|| {
            match GpuContext::init() {
                Ok(ctx) => Some(Arc::new(ctx)),
                Err(_err) => {
                    None
                }
            }
        })
        .clone()
}

/// Returns true if a functional WebGPU device is available and verified.
pub fn is_webgpu_available() -> bool {
    get_or_init_gpu_context().is_some()
}

/// Returns the active compute device information, falling back gracefully to CPU metadata.
pub fn get_device_info() -> DeviceInfo {
    if let Some(ctx) = get_or_init_gpu_context() {
        ctx.info.clone()
    } else {
        DeviceInfo::cpu_fallback()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_discovery_and_info() {
        let info = get_device_info();
        println!("Detected device: {} (is_gpu={}, backend={})", info.name, info.is_gpu, info.backend);
        assert!(!info.name.is_empty());
        assert!(!info.backend.is_empty());
    }

    #[test]
    fn test_micro_probe_standalone() {
        if let Ok(ctx) = GpuContext::init() {
            assert!(ctx.info.is_gpu);
            assert_eq!(ctx.info.backend, "Vulkan");
        }
    }

    #[test]
    fn test_cpu_fallback_info() {
        let fallback = DeviceInfo::cpu_fallback();
        assert!(!fallback.is_gpu);
        assert_eq!(fallback.backend, "CPU");
        assert_eq!(fallback.name, "CPU (Rayon Fallback)");
    }
}

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};

#[cfg(feature = "cuda")]
use crate::attention::cublas::CublasContext;
#[cfg(feature = "cuda")]
use crate::error::ZealotError;
#[cfg(feature = "cuda")]
use crate::model_loader::ModelWeights;

/// GPU execution context: owns the CUDA device, cuBLAS handle, and streams.
#[cfg(feature = "cuda")]
pub struct GpuContext {
    pub device: Arc<CudaDevice>,
    pub cublas: CublasContext,
    pub compute_stream: CudaStream,
    pub d2h_stream: CudaStream,
}

// Safety: CUDA streams are thread-safe when used with proper synchronization.
// The engine owns a single GpuContext and drives it from a single thread.
#[cfg(feature = "cuda")]
unsafe impl Send for GpuContext {}

#[cfg(feature = "cuda")]
impl GpuContext {
    pub fn new(device_id: usize) -> Result<Self, ZealotError> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| ZealotError::Internal(format!("CudaDevice {device_id}: {e}")))?;
        let cublas = CublasContext::new()?;
        // fork_default_stream creates a non-blocking stream that synchronizes
        // with the default stream in code order (as documented by cudarc).
        let compute_stream = device
            .fork_default_stream()
            .map_err(|e| ZealotError::Internal(format!("compute stream: {e}")))?;
        let d2h_stream = device
            .fork_default_stream()
            .map_err(|e| ZealotError::Internal(format!("d2h stream: {e}")))?;
        // Bind cuBLAS to the compute stream via the raw CUstream handle.
        cublas
            .set_stream(compute_stream.stream)
            .map_err(|e| ZealotError::Internal(format!("cublas set_stream: {e}")))?;
        Ok(Self {
            device,
            cublas,
            compute_stream,
            d2h_stream,
        })
    }

    /// Allocate a zero-filled `CudaSlice<f32>` of `count` elements on the device.
    pub fn alloc_zeros_f32(&self, count: usize) -> Result<CudaSlice<f32>, ZealotError> {
        self.device
            .alloc_zeros::<f32>(count)
            .map_err(|e| ZealotError::Internal(format!("alloc_zeros({count}): {e}")))
    }
}

/// GPU-side mirror of `LayerWeights`.  All layer weights reside in device memory.
#[cfg(feature = "cuda")]
pub struct GpuLayerWeights {
    pub q_proj: CudaSlice<f32>,
    pub k_proj: CudaSlice<f32>,
    pub v_proj: CudaSlice<f32>,
    pub attn_proj: CudaSlice<f32>,
    pub rms_attn_weight: CudaSlice<f32>,
    pub rms_ffn_weight: CudaSlice<f32>,
    pub gate_proj: CudaSlice<f32>,
    pub up_proj: CudaSlice<f32>,
    pub down_proj: CudaSlice<f32>,
}

/// GPU-side mirror of `ModelWeights`.  All weights reside in device memory
/// as `CudaSlice<f32>`.
#[cfg(feature = "cuda")]
pub struct GpuWeights {
    pub embedding: CudaSlice<f32>,
    pub q_proj: CudaSlice<f32>,
    pub k_proj: CudaSlice<f32>,
    pub v_proj: CudaSlice<f32>,
    pub attn_proj: CudaSlice<f32>,
    pub lm_head: CudaSlice<f32>,
    pub rms_attn_weight: CudaSlice<f32>,
    pub rms_ffn_weight: CudaSlice<f32>,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub intermediate_dim: usize,
    /// Multi-layer GPU weights.
    pub layers: Vec<GpuLayerWeights>,
    pub rms_final_weight: CudaSlice<f32>,
    pub num_kv_heads: usize,
}

#[cfg(feature = "cuda")]
impl GpuWeights {
    /// Upload CPU-side `ModelWeights` to the GPU via the given context.
    pub fn upload(cpu: &ModelWeights, ctx: &GpuContext) -> Result<Self, ZealotError> {
        let dev = &ctx.device;
        let copy_in = |src: &[f32], label: &str| -> Result<CudaSlice<f32>, ZealotError> {
            dev.htod_sync_copy(src)
                .map_err(|e| ZealotError::Internal(format!("{label}: htod_sync_copy: {e}")))
        };

        let mut gpu_layers = Vec::with_capacity(cpu.layers.len());
        for (i, layer) in cpu.layers.iter().enumerate() {
            gpu_layers.push(GpuLayerWeights {
                q_proj: copy_in(&layer.q_proj, &format!("layer[{i}].q_proj"))?,
                k_proj: copy_in(&layer.k_proj, &format!("layer[{i}].k_proj"))?,
                v_proj: copy_in(&layer.v_proj, &format!("layer[{i}].v_proj"))?,
                attn_proj: copy_in(&layer.attn_proj, &format!("layer[{i}].attn_proj"))?,
                rms_attn_weight: copy_in(
                    &layer.rms_attn_weight,
                    &format!("layer[{i}].rms_attn_weight"),
                )?,
                rms_ffn_weight: copy_in(
                    &layer.rms_ffn_weight,
                    &format!("layer[{i}].rms_ffn_weight"),
                )?,
                gate_proj: copy_in(&layer.gate_proj, &format!("layer[{i}].gate_proj"))?,
                up_proj: copy_in(&layer.up_proj, &format!("layer[{i}].up_proj"))?,
                down_proj: copy_in(&layer.down_proj, &format!("layer[{i}].down_proj"))?,
            });
        }

        Ok(Self {
            embedding: copy_in(&cpu.embedding, "embedding")?,
            q_proj: copy_in(&cpu.q_proj, "q_proj")?,
            k_proj: copy_in(&cpu.k_proj, "k_proj")?,
            v_proj: copy_in(&cpu.v_proj, "v_proj")?,
            attn_proj: copy_in(&cpu.attn_proj, "attn_proj")?,
            lm_head: copy_in(&cpu.lm_head, "lm_head")?,
            rms_attn_weight: copy_in(&cpu.rms_attn_weight, "rms_attn_weight")?,
            rms_ffn_weight: copy_in(&cpu.rms_ffn_weight, "rms_ffn_weight")?,
            hidden_dim: cpu.hidden_dim,
            num_heads: cpu.num_heads,
            head_dim: cpu.head_dim,
            num_layers: cpu.num_layers,
            vocab_size: cpu.vocab_size,
            intermediate_dim: cpu.intermediate_dim,
            layers: gpu_layers,
            rms_final_weight: copy_in(&cpu.rms_final_weight, "rms_final_weight")?,
            num_kv_heads: cpu.num_kv_heads,
        })
    }
}

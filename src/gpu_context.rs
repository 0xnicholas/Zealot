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
        })
    }
}

#[cfg(feature = "cuda")]
use crate::attention::{AttentionBackend, AttentionBatch, CpuAttention};
#[cfg(feature = "cuda")]
use crate::error::ZealotError;

#[cfg(feature = "cuda")]
pub struct CudaAttention {
    #[allow(dead_code)]
    dev: cudarc::driver::CudaDevice,
    #[allow(dead_code)]
    ptx_module: Option<cudarc::driver::CudaModule>,
}

#[cfg(feature = "cuda")]
impl CudaAttention {
    pub fn new(device_id: usize) -> Result<Self, ZealotError> {
        let dev = cudarc::driver::CudaDevice::new(device_id)
            .map_err(|e| ZealotError::Internal(format!("cuda device {}: {}", device_id, e)))?;
        Ok(Self { dev, ptx_module: None })
    }
}

#[cfg(feature = "cuda")]
impl AttentionBackend for CudaAttention {
    fn forward(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: &AttentionBatch,
    ) -> Result<Vec<f32>, ZealotError> {
        tracing::warn!("CudaAttention stub active, falling back to CPU");
        CpuAttention.forward(q, k, v, batch)
    }
}

// zealot/src/attention/kernels/mod.rs

#[cfg(feature = "cuda")]
use cudarc::driver::CudaFunction;
#[cfg(feature = "cuda")]
use crate::error::ZealotError;

#[cfg(feature = "cuda")]
static ATTENTION_PTX: &str = include!(concat!(env!("OUT_DIR"), "/attention.ptx"));
#[cfg(feature = "cuda")]
static RMS_NORM_PTX: &str = include!(concat!(env!("OUT_DIR"), "/rms_norm.ptx"));
#[cfg(feature = "cuda")]
static EMBEDDING_PTX: &str = include!(concat!(env!("OUT_DIR"), "/embedding.ptx"));

#[cfg(feature = "cuda")]
pub struct KernelModule {
    pub attention: CudaFunction,
    pub rms_norm: CudaFunction,
    pub embedding: CudaFunction,
}

#[cfg(feature = "cuda")]
impl KernelModule {
    pub fn load(device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<Self, ZealotError> {
        let attn_mod = device
            .load_ptx(ATTENTION_PTX.into(), "attention_ptx", &[])
            .map_err(|e| ZealotError::Internal(format!("load attention PTX: {e}")))?;
        let rms_mod = device
            .load_ptx(RMS_NORM_PTX.into(), "rms_norm_ptx", &[])
            .map_err(|e| ZealotError::Internal(format!("load rms_norm PTX: {e}")))?;
        let emb_mod = device
            .load_ptx(EMBEDDING_PTX.into(), "embedding_ptx", &[])
            .map_err(|e| ZealotError::Internal(format!("load embedding PTX: {e}")))?;

        Ok(Self {
            attention: attn_mod
                .get_fn("tiled_attention_kernel")
                .map_err(|e| ZealotError::Internal(format!("get attention fn: {e}")))?,
            rms_norm: rms_mod
                .get_fn("rms_norm_kernel")
                .map_err(|e| ZealotError::Internal(format!("get rms_norm fn: {e}")))?,
            embedding: emb_mod
                .get_fn("embedding_gather_kernel")
                .map_err(|e| ZealotError::Internal(format!("get embedding fn: {e}")))?,
        })
    }
}

/// Helper: get device pointer from CudaSlice
#[cfg(feature = "cuda")]
pub fn dev_ptr<T>(slice: &cudarc::driver::CudaSlice<T>) -> *const T {
    **slice as *const T
}

#[cfg(feature = "cuda")]
pub fn dev_ptr_mut<T>(slice: &mut cudarc::driver::CudaSlice<T>) -> *mut T {
    **slice
}

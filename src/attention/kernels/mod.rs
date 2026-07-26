// zealot/src/attention/kernels/mod.rs

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaFunction, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::ZealotError;

#[cfg(feature = "cuda")]
static ATTENTION_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/attention.ptx"));
#[cfg(feature = "cuda")]
static RMS_NORM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/rms_norm.ptx"));
#[cfg(feature = "cuda")]
static EMBEDDING_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/embedding.ptx"));
#[cfg(feature = "cuda")]
static TRANSPOSE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/transpose.ptx"));
#[cfg(feature = "cuda")]
static PAGED_ATTENTION_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/paged_attention.ptx"));

#[cfg(feature = "cuda")]
pub struct KernelModule {
    pub attention: CudaFunction,
    pub rms_norm: CudaFunction,
    pub embedding: CudaFunction,
    pub transpose: CudaFunction,
    pub paged_attention: CudaFunction,
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
        let trans_mod = device
            .load_ptx(TRANSPOSE_PTX.into(), "transpose_ptx", &[])
            .map_err(|e| ZealotError::Internal(format!("load transpose PTX: {e}")))?;
        let pa_mod = device
            .load_ptx(PAGED_ATTENTION_PTX.into(), "paged_attention_ptx", &[])
            .map_err(|e| ZealotError::Internal(format!("load paged_attention PTX: {e}")))?;

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
            transpose: trans_mod
                .get_fn("qkv_transpose_kernel")
                .map_err(|e| ZealotError::Internal(format!("get transpose fn: {e}")))?,
            paged_attention: pa_mod
                .get_fn("paged_attention_kernel")
                .map_err(|e| ZealotError::Internal(format!("get paged_attention fn: {e}")))?,
        })
    }
}

/// Helper: get device pointer from CudaSlice (typed).
///
/// Uses the [`DevicePtr`] trait to extract the raw `CUdeviceptr`
/// and cast it to a typed `*const T`.
#[cfg(feature = "cuda")]
pub fn dev_ptr<T>(slice: &cudarc::driver::CudaSlice<T>) -> *const T {
    *slice.device_ptr() as *const T
}

#[cfg(feature = "cuda")]
pub fn dev_ptr_mut<T>(slice: &mut cudarc::driver::CudaSlice<T>) -> *mut T {
    *slice.device_ptr_mut() as *mut T
}

// ── Kernel launch helpers (raw-pointer variants) ───────────────────────────

/// Launch the QKV transpose kernel with raw device pointers.
///
/// `direction`: 0 = interleaved → head-major, 1 = head-major → interleaved.
///
/// # Safety
/// The caller must ensure all pointers are valid device allocations for the
/// duration of the launch.
#[cfg(feature = "cuda")]
pub fn launch_qkv_transpose(
    kernel: &CudaFunction,
    src: *const f32,
    dst: *mut f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    direction: i32,
) -> Result<(), ZealotError> {
    let total = (seq_len * num_heads * head_dim) as u32;
    let cfg = LaunchConfig {
        grid_dim: ((total + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    use std::ffi::c_void;
    let s = seq_len as i32;
    let nh = num_heads as i32;
    let hd = head_dim as i32;
    let mut params: Vec<*mut c_void> = vec![
        std::ptr::addr_of!(src) as *mut c_void,
        std::ptr::addr_of!(dst) as *mut c_void,
        std::ptr::addr_of!(s) as *mut c_void,
        std::ptr::addr_of!(nh) as *mut c_void,
        std::ptr::addr_of!(hd) as *mut c_void,
        std::ptr::addr_of!(direction) as *mut c_void,
    ];

    unsafe {
        kernel
            .clone()
            .launch(cfg, &mut params)
            .map_err(|e| ZealotError::Internal(format!("qkv_transpose kernel: {e}")))
    }
}

/// Launch the RMS norm kernel with raw device pointers.
///
/// # Safety
/// The caller must ensure all pointers are valid device allocations for the
/// duration of the launch.
#[cfg(feature = "cuda")]
pub fn launch_rms_norm_raw(
    kernel: &CudaFunction,
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(), ZealotError> {
    let r = rows as u32;
    let d = dim as u32;

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    use std::ffi::c_void;
    let mut params: Vec<*mut c_void> = vec![
        std::ptr::addr_of!(x) as *mut c_void,
        std::ptr::addr_of!(weight) as *mut c_void,
        std::ptr::addr_of!(out) as *mut c_void,
        std::ptr::addr_of!(r) as *mut c_void,
        std::ptr::addr_of!(d) as *mut c_void,
        std::ptr::addr_of!(eps) as *mut c_void,
    ];

    unsafe {
        kernel
            .clone()
            .launch(cfg, &mut params)
            .map_err(|e| ZealotError::Internal(format!("rms_norm kernel: {e}")))
    }
}

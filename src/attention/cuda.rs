#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda")]
use crate::error::ZealotError;

/// GPU-native flash-attention operator.
///
/// Owns a pre-compiled CUDA kernel (`CudaFunction`) and a handle to the
/// CUDA device it was loaded on.  The kernel is launched via cudarc's
/// `LaunchAsync` trait; all compute happens on the device stream.
#[cfg(feature = "cuda")]
pub struct CudaAttention {
    kernel: CudaFunction,
    #[allow(dead_code)]
    dev: Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl CudaAttention {
    /// Create a `CudaAttention` that will launch `kernel` on `device`.
    pub fn new(kernel: CudaFunction, device: Arc<CudaDevice>) -> Self {
        Self { kernel, dev: device }
    }

    /// Launch the flash-attention kernel on the device's default stream.
    ///
    /// All input pointers must reference device memory allocated on the
    /// same GPU.  The layout is `[num_heads, seq_len, head_dim]` in
    /// row-major order (head → position → dimension).
    ///
    /// # Safety
    ///
    /// The caller is responsible for ensuring that `q`, `k`, `v`, and `o`
    /// are valid device pointers for the duration of the kernel launch.
    pub fn forward_gpu(
        &self,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *mut f32,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(), ZealotError> {
        // ── Compute launch geometry ──────────────────────────────────
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        // Block size heuristic: use 64 unless head_dim is very large.
        let br = if head_dim > 128 { 32 } else { 64 };
        let bc = br;
        let grid_y = seq_len.div_ceil(br) as u32;
        let shared_mem_bytes: u32 =
            ((br * head_dim + bc * head_dim * 2 + br * bc) * 4) as u32;

        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, grid_y, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        };

        // ── Pack kernel parameters ───────────────────────────────────
        // CUDA expects each param slot to be a void* pointing to the
        // actual argument value.  We use cudarc's raw-pointer launch
        // variant.
        use std::ffi::c_void;

        let causal_u32: u32 = if causal { 1 } else { 0 };

        let mut params: Vec<*mut c_void> = vec![
            std::ptr::addr_of!(q) as *mut c_void,
            std::ptr::addr_of!(k) as *mut c_void,
            std::ptr::addr_of!(v) as *mut c_void,
            std::ptr::addr_of!(o) as *mut c_void,
            std::ptr::addr_of!(seq_len) as *mut c_void,
            std::ptr::addr_of!(num_heads) as *mut c_void,
            std::ptr::addr_of!(head_dim) as *mut c_void,
            std::ptr::addr_of!(scale) as *mut c_void,
            std::ptr::addr_of!(causal_u32) as *mut c_void,
        ];

        // Safety: all param pointers reference stack variables whose
        // values are copied into the launch descriptor by the CUDA
        // driver before `launch` returns.
        unsafe {
            self.kernel
                .launch(cfg, &mut params)
                .map_err(|e| ZealotError::Internal(format!("attention kernel launch: {e}")))
        }
    }
}

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda")]
use std::ffi::c_void;

#[cfg(feature = "cuda")]
use crate::error::ZealotError;

/// GPU-native flash-attention operator.
///
/// Owns pre-compiled CUDA kernels (`CudaFunction`) for both tiled
/// flash-attention and PagedAttention, plus a handle to the CUDA device
/// they were loaded on.  Kernels are launched via cudarc's `LaunchAsync`
/// trait; all compute happens on the device stream.
#[cfg(feature = "cuda")]
pub struct CudaAttention {
    kernel: CudaFunction,
    paged_kernel: CudaFunction,
    #[allow(dead_code)]
    dev: Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl CudaAttention {
    /// Create a `CudaAttention` that will launch `kernel` (tiled attention)
    /// and `paged_kernel` (PagedAttention) on `device`.
    pub fn new(kernel: CudaFunction, paged_kernel: CudaFunction, device: Arc<CudaDevice>) -> Self {
        Self { kernel, paged_kernel, dev: device }
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
                .clone()
                .launch(cfg, &mut params)
                .map_err(|e| ZealotError::Internal(format!("attention kernel launch: {e}")))
        }
    }

    /// Launch the PagedAttention kernel for batched decode (or chunked prefill
    /// continuation) over the KV cache.
    ///
    /// `q` is the query vectors `[num_seqs, num_heads, head_dim]` (interleaved).
    /// `k_pool` / `v_pool` are the flat per-layer KV cache pools.
    /// `block_tables` is `[num_seqs * max_blocks_per_seq]` of i32 block IDs.
    /// `seq_lens` is `[num_seqs]` of i32 total token counts per sequence.
    /// `o` is the output `[num_seqs, num_heads, head_dim]` (interleaved).
    ///
    /// # Safety
    ///
    /// The caller is responsible for ensuring that all pointers are valid
    /// device allocations for the duration of the kernel launch.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_forward(
        &self,
        q: *const f32,
        k_pool: *const f32,
        v_pool: *const f32,
        block_tables: *const i32,
        seq_lens: *const i32,
        o: *mut f32,
        num_seqs: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<(), ZealotError> {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let shared_mem =
            (block_size * head_dim * 2 * std::mem::size_of::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_seqs as u32, num_heads as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        let ns = num_seqs as i32;
        let nh = num_heads as i32;
        let nkh = num_kv_heads as i32;
        let hd = head_dim as i32;
        let bs = block_size as i32;
        let mb = max_blocks_per_seq as i32;

        let mut params: Vec<*mut c_void> = vec![
            std::ptr::addr_of!(q) as *mut c_void,
            std::ptr::addr_of!(k_pool) as *mut c_void,
            std::ptr::addr_of!(v_pool) as *mut c_void,
            std::ptr::addr_of!(o) as *mut c_void,
            std::ptr::addr_of!(block_tables) as *mut c_void,
            std::ptr::addr_of!(seq_lens) as *mut c_void,
            std::ptr::addr_of!(ns) as *mut c_void,
            std::ptr::addr_of!(nh) as *mut c_void,
            std::ptr::addr_of!(nkh) as *mut c_void,
            std::ptr::addr_of!(hd) as *mut c_void,
            std::ptr::addr_of!(mb) as *mut c_void,
            std::ptr::addr_of!(bs) as *mut c_void,
            std::ptr::addr_of!(scale) as *mut c_void,
        ];

        unsafe {
            self.paged_kernel
                .clone()
                .launch(cfg, &mut params)
                .map_err(|e| ZealotError::Internal(format!("paged_attention: {e}")))?;
        }
        Ok(())
    }
}

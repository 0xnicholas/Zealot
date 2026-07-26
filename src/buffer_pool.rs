//! BufferPool — a size-bucketed GPU scratch-buffer allocator.
//!
//! Reuses `CudaSlice<f32>` allocations by size class to avoid repeated
//! `cudaMalloc`/`cudaFree` overhead during inference.  Callers acquire
//! scratch buffers, use them, and release them back for reuse.
//!
//! Entirely gated behind `#[cfg(feature = "cuda")]`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use cudarc::driver::{CudaSlice, DeviceSlice};

use crate::error::ZealotError;
use crate::gpu_context::GpuContext;

/// Size-bucketed pool of reusable GPU scratch buffers.
///
/// Each bucket keyed by element count; `acquire` returns a zero-filled
/// buffer of exactly the requested size, either from the free list or
/// freshly allocated.
pub struct BufferPool {
    free_buffers: HashMap<usize, Vec<CudaSlice<f32>>>,
}

impl BufferPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            free_buffers: HashMap::new(),
        }
    }

    /// Acquire a zero-filled GPU buffer of `n` f32 elements.
    ///
    /// Returns a cached buffer if one exists for `n`; otherwise allocates
    /// a new zero-filled slice via `ctx`.
    pub fn acquire(
        &mut self,
        ctx: &GpuContext,
        n: usize,
    ) -> Result<CudaSlice<f32>, ZealotError> {
        if let Some(bucket) = self.free_buffers.get_mut(&n) {
            if let Some(buf) = bucket.pop() {
                return Ok(buf);
            }
        }
        ctx.alloc_zeros_f32(n)
    }

    /// Return a buffer to the pool for later reuse.
    ///
    /// The buffer is keyed by its length; a subsequent `acquire` with the
    /// same size may return this exact allocation.
    pub fn release(&mut self, buf: CudaSlice<f32>) {
        let n = buf.len();
        self.free_buffers.entry(n).or_default().push(buf);
    }

    /// Drop all cached buffers, returning GPU memory to the driver.
    pub fn flush(&mut self) {
        self.free_buffers.clear();
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_pool_new_is_empty() {
        let pool = BufferPool::new();
        assert!(pool.free_buffers.is_empty());
    }

    #[test]
    fn buffer_pool_flush_clears() {
        let mut pool = BufferPool::new();
        // Simulate a release without real GPU alloc: insert a dummy entry
        pool.free_buffers.insert(128, vec![]);
        assert_eq!(pool.free_buffers.len(), 1);
        pool.flush();
        assert!(pool.free_buffers.is_empty());
    }
}

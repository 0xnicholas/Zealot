//! KVCache — GPU memory pool for key/value caches across layers.
//!
//! Manages per-sequence block tables and stores K/V activations in
//! paged GPU memory.  Integrates with [`BlockManager`] for physical
//! block allocation.  The PagedAttention CUDA kernel reads directly
//! from the flat K/V pools using block tables.
//!
//! Entirely gated behind `#[cfg(feature = "cuda")]`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

use crate::block_manager::BlockManager;
use crate::error::ZealotError;

// ── SeqKvState ──────────────────────────────────────────────────────────────

/// Per-sequence KV cache metadata.  Owns the GPU-side `block_table`
/// and tracks which physical blocks are in use.
pub struct SeqKvState {
    /// GPU-side block table: `[max_blocks]` of i32, filled with -1 sentinels.
    pub block_table: CudaSlice<i32>,
    /// CPU-side copy of the allocated block IDs (same order as block_table).
    pub block_ids: Vec<usize>,
    /// Total tokens stored so far (K/V length).
    pub cached_tokens: usize,
    /// Write offset within the current block (0..block_size).
    pub block_offset: usize,
    /// Maximum number of blocks this sequence may hold.
    pub max_blocks: usize,
}

// ── KVCache ─────────────────────────────────────────────────────────────────

/// Multi-layer paged KV cache.
///
/// Memory layout for each pool `k_pools[layer]` / `v_pools[layer]`:
///
/// ```text
/// [block_id * block_size * num_kv_heads * head_dim
///   + kv_head * block_size * head_dim
///   + token_in_block * head_dim
///   + dim]
/// ```
///
/// Total bytes per layer: `num_blocks × block_size × num_kv_heads × head_dim × 4`.
pub struct KVCache {
    pub k_pools: Vec<CudaSlice<f32>>,
    pub v_pools: Vec<CudaSlice<f32>>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,

    device: Arc<CudaDevice>,
    seqs: HashMap<String, SeqKvState>,
}

impl KVCache {
    /// Allocate flat K/V pools for every layer.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self, ZealotError> {
        let per_layer = num_blocks * block_size * num_kv_heads * head_dim;

        let mut k_pools = Vec::with_capacity(num_layers);
        let mut v_pools = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_pools.push(
                device
                    .alloc_zeros::<f32>(per_layer)
                    .map_err(|e| ZealotError::Internal(format!("KVCache k_pool alloc: {e}")))?,
            );
            v_pools.push(
                device
                    .alloc_zeros::<f32>(per_layer)
                    .map_err(|e| ZealotError::Internal(format!("KVCache v_pool alloc: {e}")))?,
            );
        }

        Ok(Self {
            k_pools,
            v_pools,
            num_blocks,
            block_size,
            num_layers,
            num_kv_heads,
            head_dim,
            device,
            seqs: HashMap::new(),
        })
    }

    // ── Sequence lifecycle ──────────────────────────────────────────────

    /// Register a new sequence, allocating its GPU block table.
    pub fn register_seq(&mut self, id: &str, max_blocks: usize) -> Result<(), ZealotError> {
        let table: Vec<i32> = vec![-1; max_blocks];
        let block_table = self
            .device
            .htod_sync_copy(&table)
            .map_err(|e| ZealotError::Internal(format!("KVCache block_table: {e}")))?;
        self.seqs.insert(
            id.to_string(),
            SeqKvState {
                block_table,
                block_ids: Vec::with_capacity(max_blocks),
                cached_tokens: 0,
                block_offset: 0,
                max_blocks,
            },
        );
        Ok(())
    }

    /// Allocate a new physical block for `id` via the [`BlockManager`].
    pub fn alloc_block(&mut self, id: &str, bm: &BlockManager) -> Result<usize, ZealotError> {
        let blk = bm.try_allocate()?;
        let block_id = blk.block_id;
        let val: Vec<i32> = vec![block_id as i32];

        // Scope 1: get idx from state
        let idx = {
            let state = self.seq_state(id)?;
            state.block_ids.len()
        };

        // Scope 2: upload block id to GPU
        let tmp = self
            .device
            .htod_sync_copy(&val)
            .map_err(|e| ZealotError::Internal(format!("KVCache alloc_block: {e}")))?;
        {
            let state = self.seq_state_mut(id)?;
            let mut dst = state
                .block_table
                .try_slice_mut(idx..idx + 1)
                .ok_or_else(|| ZealotError::Internal("block_table slice oob".into()))?;
            self.device
                .dtod_copy(&tmp, &mut dst)
                .map_err(|e| ZealotError::Internal(format!("KVCache alloc_block copy: {e}")))?;
        }

        // Scope 3: update CPU-side tracking
        {
            let state = self.seq_state_mut(id)?;
            state.block_ids.push(block_id);
        }

        Ok(block_id)
    }

    /// Release a sequence (drop its GPU block_table and metadata).
    pub fn release_seq(&mut self, id: &str) {
        self.seqs.remove(id);
    }

    // ── Queries ─────────────────────────────────────────────────────────

    /// Immutable reference to a sequence's KV state.
    pub fn seq_state(&self, id: &str) -> Result<&SeqKvState, ZealotError> {
        self.seqs
            .get(id)
            .ok_or_else(|| ZealotError::Internal(format!("KVCache: unknown seq {id}")))
    }

    fn seq_state_mut(&mut self, id: &str) -> Result<&mut SeqKvState, ZealotError> {
        self.seqs
            .get_mut(id)
            .ok_or_else(|| ZealotError::Internal(format!("KVCache: unknown seq {id}")))
    }

    /// Maximum blocks this sequence can hold.
    pub fn max_blocks(&self, id: &str) -> Result<usize, ZealotError> {
        Ok(self.seq_state(id)?.max_blocks)
    }

    // ── Raw pointer access (for CUDA kernel launches) ───────────────────

    /// GPU pointer to the block table for `id`.
    pub fn block_table_ptr(&self, id: &str) -> Result<*const i32, ZealotError> {
        let state = self.seq_state(id)?;
        Ok(*state.block_table.device_ptr() as *const i32)
    }

    /// GPU pointer to the K pool for `layer`.
    pub fn k_pool_ptr(&self, layer: usize) -> *const f32 {
        *self.k_pools[layer].device_ptr() as *const f32
    }

    /// GPU pointer to the V pool for `layer`.
    pub fn v_pool_ptr(&self, layer: usize) -> *const f32 {
        *self.v_pools[layer].device_ptr() as *const f32
    }

    // ── Store / append K/V ──────────────────────────────────────────────

    /// Store `n_tokens` worth of K/V activations starting at `start_token`.
    ///
    /// On entry `k` and `v` are GPU slices of shape `[n_tokens, num_kv_heads, head_dim]`
    /// (flat, row-major: token → kv_head → dim).
    pub fn store_kv(
        &mut self,
        layer: usize,
        id: &str,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        start_token: usize,
        n_tokens: usize,
    ) -> Result<(), ZealotError> {
        // Extract needed values before any mutable borrows of pools.
        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let block_size = self.block_size;
        let kv_data = num_kv_heads * head_dim;
        let kv_block_stride = block_size * kv_data;
        let block_ids = { self.seq_state(id)?.block_ids.clone() };

        for t in 0..n_tokens {
            let global_token = start_token + t;
            let block_idx = global_token / block_size;
            let token_in_block = global_token % block_size;

            if block_idx >= block_ids.len() {
                return Err(ZealotError::Internal(format!(
                    "KVCache store_kv: block_idx {block_idx} OOB (have {} blocks)",
                    block_ids.len()
                )));
            }

            let block_id = block_ids[block_idx];
            let src_base = t * kv_data;

            for h in 0..num_kv_heads {
                let pool_off = block_id * kv_block_stride
                    + h * block_size * head_dim
                    + token_in_block * head_dim;
                let src_off = src_base + h * head_dim;

                let src_view = k
                    .try_slice(src_off..src_off + head_dim)
                    .ok_or_else(|| ZealotError::Internal("KVCache store_kv: k slice OOB".into()))?;
                let mut dst_view = self.k_pools[layer]
                    .try_slice_mut(pool_off..pool_off + head_dim)
                    .ok_or_else(|| {
                        ZealotError::Internal("KVCache store_kv: k_pool slice OOB".into())
                    })?;
                self.device
                    .dtod_copy(&src_view, &mut dst_view)
                    .map_err(|e| ZealotError::Internal(format!("KVCache store_kv k copy: {e}")))?;

                let src_view = v
                    .try_slice(src_off..src_off + head_dim)
                    .ok_or_else(|| ZealotError::Internal("KVCache store_kv: v slice OOB".into()))?;
                let mut dst_view = self.v_pools[layer]
                    .try_slice_mut(pool_off..pool_off + head_dim)
                    .ok_or_else(|| {
                        ZealotError::Internal("KVCache store_kv: v_pool slice OOB".into())
                    })?;
                self.device
                    .dtod_copy(&src_view, &mut dst_view)
                    .map_err(|e| ZealotError::Internal(format!("KVCache store_kv v copy: {e}")))?;
            }
        }
        Ok(())
    }

    /// Append a single token's K/V activations and advance the write pointer.
    ///
    /// `k` / `v` are GPU slices of shape `[num_kv_heads, head_dim]` (one token).
    pub fn append_kv(
        &mut self,
        layer: usize,
        id: &str,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
    ) -> Result<(), ZealotError> {
        // Collect scalar values and block_ids before mutating pools.
        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let block_size = self.block_size;
        let kv_data = num_kv_heads * head_dim;
        let kv_block_stride = block_size * kv_data;

        // Extract seq state data, releasing the mutable borrow.
        let (block_ids, token, block_offset) = {
            let state = self.seq_state_mut(id)?;
            let ids = state.block_ids.clone();
            let tok = state.cached_tokens;
            let off = state.block_offset;
            (ids, tok, off)
        };

        let block_idx = token / block_size;
        let token_in_block = block_offset;

        if block_idx >= block_ids.len() {
            return Err(ZealotError::Internal(format!(
                "KVCache append_kv: block_idx {block_idx} OOB (have {} blocks)",
                block_ids.len()
            )));
        }

        let block_id = block_ids[block_idx];

        for h in 0..num_kv_heads {
            let pool_off =
                block_id * kv_block_stride + h * block_size * head_dim + token_in_block * head_dim;
            let src_off = h * head_dim;

            let src_view = k
                .try_slice(src_off..src_off + head_dim)
                .ok_or_else(|| ZealotError::Internal("KVCache append_kv: k slice OOB".into()))?;
            let mut dst_view = self.k_pools[layer]
                .try_slice_mut(pool_off..pool_off + head_dim)
                .ok_or_else(|| {
                    ZealotError::Internal("KVCache append_kv: k_pool slice OOB".into())
                })?;
            self.device
                .dtod_copy(&src_view, &mut dst_view)
                .map_err(|e| ZealotError::Internal(format!("KVCache append_kv k copy: {e}")))?;

            let src_view = v
                .try_slice(src_off..src_off + head_dim)
                .ok_or_else(|| ZealotError::Internal("KVCache append_kv: v slice OOB".into()))?;
            let mut dst_view = self.v_pools[layer]
                .try_slice_mut(pool_off..pool_off + head_dim)
                .ok_or_else(|| {
                    ZealotError::Internal("KVCache append_kv: v_pool slice OOB".into())
                })?;
            self.device
                .dtod_copy(&src_view, &mut dst_view)
                .map_err(|e| ZealotError::Internal(format!("KVCache append_kv v copy: {e}")))?;
        }

        // Update sequence state.
        {
            let state = self.seq_state_mut(id)?;
            state.cached_tokens += 1;
            state.block_offset += 1;
            if state.block_offset >= block_size {
                state.block_offset = 0;
            }
        }
        Ok(())
    }
}

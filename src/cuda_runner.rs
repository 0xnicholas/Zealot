//! CudaModelRunner — GPU inference runner using CUDA kernels + cuBLAS.
//!
//! Implements the `ModelRunner` trait entirely on GPU.  All compute
//! happens via custom CUDA kernels (embedding gather, RMS norm, tiled
//! flash-attention) plus cuBLAS for linear projections.
//!
//! Entirely gated behind `#[cfg(feature = "cuda")]` — no CUDA symbols
//! leak into the default build.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::attention::cublas::CublasOp;
use crate::attention::cuda::CudaAttention;
use crate::attention::kernels::{self, dev_ptr, dev_ptr_mut, KernelModule};
use crate::buffer_pool::BufferPool;
use crate::engine::{ModelRunner, StepOut};
use crate::error::ZealotError;
use crate::gpu_context::{GpuContext, GpuWeights};
use crate::kv_cache::KVCache;
use crate::model_loader;
use crate::scheduler::{SchedulerConfig, Sequence};

// ── CudaModelRunner ──────────────────────────────────────────────────────────

/// GPU-native model runner.  Owns the CUDA context, uploaded weights,
/// pre-compiled PTX kernels, the flash-attention operator, paged KV cache,
/// and a scratch-buffer pool.
pub struct CudaModelRunner {
    ctx: GpuContext,
    weights: GpuWeights,
    kernels: KernelModule,
    attn: CudaAttention,
    kv_cache: KVCache,
    pool: BufferPool,
    tokenizer_py: Option<Py<PyAny>>,
    eos_token_id: Option<i64>,
}

impl CudaModelRunner {
    // ── Load ────────────────────────────────────────────────────────────

    /// Load model weights from HuggingFace `model_id`, upload them to GPU
    /// `device_id`, compile PTX kernels, and create the attention operator.
    ///
    /// Weights are first loaded onto CPU via PyO3 (same `model_loader`
    /// path used by `CpuModelRunner`), then transferred to GPU with
    /// `htod_sync_copy`.
    pub fn load(model_id: &str, device_id: usize) -> Result<Self, ZealotError> {
        // 1. CPU weights via PyO3
        let cpu_weights = model_loader::load_weights(model_id)?;

        // 2. GPU context + weight upload
        let ctx = GpuContext::new(device_id)?;
        let weights = GpuWeights::upload(&cpu_weights, &ctx)?;

        // 3. PTX kernels (attention, rms_norm, embedding)
        let kernels = KernelModule::load(&ctx.device)?;

        // 4. Attention operator (tiled + paged + GQA)
        let attn = CudaAttention::new(
            kernels.attention.clone(),
            kernels.paged_attention.clone(),
            kernels.paged_attention_gqa.clone(),
            ctx.device.clone(),
        );

        // 5. KV cache (read block_size/num_gpu_blocks from SchedulerConfig)
        let cfg = SchedulerConfig::default();
        let kv_cache = KVCache::new(
            cfg.num_gpu_blocks,
            cfg.block_size,
            weights.num_layers,
            weights.num_kv_heads,
            weights.head_dim,
            ctx.device.clone(),
        )?;

        // 6. Tokenizer + eos_token_id (PyO3, same pattern as CpuModelRunner)
        let tokenizer_py = Python::with_gil(
            |py| -> Result<Option<Py<PyAny>>, ZealotError> {
                if let Ok(sp) = std::env::var("ZEALOT_SITE_PACKAGES") {
                    let sys = py
                        .import("sys")
                        .map_err(model_loader::py_err("import sys"))?;
                    sys.getattr("path")
                        .map_err(model_loader::py_err("sys.path"))?
                        .call_method1("insert", (0, sp))
                        .map_err(model_loader::py_err("sys.path.insert"))?;
                }

                let _torch = py
                    .import("torch")
                    .map_err(model_loader::py_err(
                        "import torch (venv? set ZEALOT_SITE_PACKAGES)",
                    ))?;
                let transformers = py
                    .import("transformers")
                    .map_err(model_loader::py_err("import transformers"))?;

                let tokenizer = transformers
                    .getattr("AutoTokenizer")
                    .map_err(model_loader::py_err("AutoTokenizer"))?
                    .call_method1("from_pretrained", (model_id,))
                    .map_err(model_loader::py_err("tokenizer from_pretrained"))?;
                Ok(Some(tokenizer.unbind()))
            },
        )?;

        let eos_token_id = extract_eos(tokenizer_py.as_ref());

        // 7. Warmup: trigger CUDA lazy init
        let runner = Self {
            ctx,
            weights,
            kernels,
            attn,
            kv_cache,
            pool: BufferPool::new(),
            tokenizer_py,
            eos_token_id,
        };
        runner.warmup()?;
        Ok(runner)
    }

    // ── warmup ──────────────────────────────────────────────────────────

    /// Run a trivial cuBLAS matmul to force lazy CUDA context / library
    /// initialisation before the first real inference step.
    pub fn warmup(&self) -> Result<(), ZealotError> {
        let dummy = self.ctx.alloc_zeros_f32(1)?;
        let mut out = self.ctx.alloc_zeros_f32(1)?;
        let dummy_ptr = dev_ptr(&dummy);
        let out_ptr = dev_ptr_mut(&mut out);
        self.ctx.cublas.sgemm(
            1, 1, 1, dummy_ptr, 1, CublasOp::N, dummy_ptr, 1, CublasOp::N, out_ptr, 1,
        )?;
        Ok(())
    }

    // ── bench_attention (stub) ──────────────────────────────────────────

    /// Benchmark the attention kernel.  Deferred to Sprint B.
    #[allow(dead_code)]
    fn bench_attention(&self) -> Result<(), ZealotError> {
        Ok(())
    }

    // ── launch helpers ──────────────────────────────────────────────────

    /// Launch the embedding gather kernel.
    ///
    /// # Safety (kernel FFI)
    ///
    /// The caller must ensure `ids`, `embedding`, and `out` are valid
    /// device allocations for the duration of the launch.
    unsafe fn launch_embedding_gather_impl(
        &self,
        ids: &CudaSlice<i64>,
        embedding: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<(), ZealotError> {
        let s = seq_len as u32;
        let h = hidden_dim as u32;
        let ids_ptr = dev_ptr(ids);
        let emb_ptr = dev_ptr(embedding);
        let out_ptr = dev_ptr_mut(out);

        let cfg = LaunchConfig {
            grid_dim: (((seq_len + 255) / 256) as u32, ((hidden_dim + 31) / 32) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        use std::ffi::c_void;
        let mut params: Vec<*mut c_void> = vec![
            std::ptr::addr_of!(ids_ptr) as *mut c_void,
            std::ptr::addr_of!(emb_ptr) as *mut c_void,
            std::ptr::addr_of!(out_ptr) as *mut c_void,
            std::ptr::addr_of!(s) as *mut c_void,
            std::ptr::addr_of!(h) as *mut c_void,
        ];

        self.kernels
            .embedding
            .clone()
            .launch(cfg, &mut params)
            .map_err(|e| ZealotError::Internal(format!("embedding gather kernel: {e}")))
    }

    fn launch_embedding_gather(
        &self,
        ids: &CudaSlice<i64>,
        embedding: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<(), ZealotError> {
        unsafe {
            self.launch_embedding_gather_impl(ids, embedding, out, seq_len, hidden_dim)
        }
    }

}

// ── ModelRunner trait impl ──────────────────────────────────────────────────

impl ModelRunner for CudaModelRunner {
    fn tokenize_chat(&self, messages: &[(String, String)]) -> Result<Vec<i64>, ZealotError> {
        // Duplicated from CpuModelRunner — acceptable for Sprint A.
        Python::with_gil(
            |py| -> Result<Vec<i64>, ZealotError> {
                let tokenizer = self
                    .tokenizer_py
                    .as_ref()
                    .ok_or_else(|| ZealotError::Internal("tokenizer not loaded".into()))?
                    .bind(py);
                let list = PyList::empty(py);
                for (role, content) in messages {
                    let d = PyDict::new(py);
                    d.set_item("role", role)
                        .map_err(model_loader::py_err("msg role"))?;
                    d.set_item("content", content)
                        .map_err(model_loader::py_err("msg content"))?;
                    list.append(d)
                        .map_err(model_loader::py_err("msg append"))?;
                }
                let kwargs = PyDict::new(py);
                kwargs
                    .set_item("tokenize", true)
                    .map_err(model_loader::py_err("kw"))?;
                kwargs
                    .set_item("add_generation_prompt", true)
                    .map_err(model_loader::py_err("kw"))?;
                match tokenizer.call_method("apply_chat_template", (list,), Some(&kwargs)) {
                    Ok(ids) => ids
                        .extract::<Vec<i64>>()
                        .map_err(model_loader::py_err("chat_template ids")),
                    Err(_) => {
                        let text = messages
                            .iter()
                            .map(|(r, c)| format!("{r}: {c}"))
                            .collect::<Vec<_>>()
                            .join("\n")
                            + "\n";
                        tokenizer
                            .call_method1("encode", (text,))
                            .map_err(model_loader::py_err("encode"))?
                            .extract::<Vec<i64>>()
                            .map_err(model_loader::py_err("encode ids"))
                    }
                }
            },
        )
    }

    fn eos_token_id(&self) -> Option<i64> {
        self.eos_token_id
    }

    fn step(&mut self, batch: &mut [&mut Sequence]) -> Result<Vec<StepOut>, ZealotError> {
        let mut outs = Vec::with_capacity(batch.len());

        for seq in batch.iter() {
            let token_ids = seq.step_input();
            let is_prefill = seq.is_prefill();
            let seq_len = token_ids.len();
            let is_first_chunk = is_prefill && seq.prefill_pos == 0;

            let nh = self.weights.num_heads;
            let nkh = self.weights.num_kv_heads;
            let hd = self.weights.head_dim;
            let hid = self.weights.hidden_dim;
            let proj_dim = nh * hd;
            let kv_dim = nkh * hd;

            // ── 1. Embedding ──────────────────────────────────────
            let ids_gpu = self
                .ctx
                .device
                .htod_sync_copy(&token_ids)
                .map_err(|e| ZealotError::Internal(format!("htod ids: {e}")))?;
            let mut hidden = self.pool.acquire(&self.ctx, seq_len * hid)?;
            self.launch_embedding_gather(
                &ids_gpu,
                &self.weights.embedding,
                &mut hidden,
                seq_len,
                hid,
            )?;
            drop(ids_gpu);

            // ── 2. Layer loop ─────────────────────────────────────
            let inter_dim = self.weights.intermediate_dim;

            for layer_idx in 0..self.weights.num_layers {
                let lw = &self.weights.layers[layer_idx];

                // ── Attention sub-block ────────────────────────────
                // hidden is the residual for the attention sub-block.
                // It stays alive through RMS norm, QKV, attention, output_proj,
                // then participates in the first add.

                // RMS Norm (attn weight)
                let mut normed = self.pool.acquire(&self.ctx, seq_len * hid)?;
                kernels::launch_rms_norm_raw(
                    &self.kernels.rms_norm,
                    dev_ptr(&hidden),
                    dev_ptr(&lw.rms_attn_weight),
                    dev_ptr_mut(&mut normed),
                    seq_len,
                    hid,
                    1e-6,
                )?;

                // QKV projection
                let mut q = self.pool.acquire(&self.ctx, seq_len * proj_dim)?;
                let mut k = self.pool.acquire(&self.ctx, seq_len * kv_dim)?;
                let mut v = self.pool.acquire(&self.ctx, seq_len * kv_dim)?;
                // cuBLAS: normed[seq,hid] @ q_proj[nh*hd,hid]^T = q[seq,nh*hd]
                self.ctx.cublas.sgemm(
                    seq_len as i32,
                    proj_dim as i32,
                    hid as i32,
                    dev_ptr(&normed),
                    hid as i32,
                    CublasOp::N,
                    dev_ptr(&lw.q_proj),
                    proj_dim as i32,
                    CublasOp::T,
                    dev_ptr_mut(&mut q),
                    proj_dim as i32,
                )?;
                self.ctx.cublas.sgemm(
                    seq_len as i32,
                    kv_dim as i32,
                    hid as i32,
                    dev_ptr(&normed),
                    hid as i32,
                    CublasOp::N,
                    dev_ptr(&lw.k_proj),
                    kv_dim as i32,
                    CublasOp::T,
                    dev_ptr_mut(&mut k),
                    kv_dim as i32,
                )?;
                self.ctx.cublas.sgemm(
                    seq_len as i32,
                    kv_dim as i32,
                    hid as i32,
                    dev_ptr(&normed),
                    hid as i32,
                    CublasOp::N,
                    dev_ptr(&lw.v_proj),
                    kv_dim as i32,
                    CublasOp::T,
                    dev_ptr_mut(&mut v),
                    kv_dim as i32,
                )?;
                self.pool.release(normed);

                // GPU transpose K,V (interleaved → head-major)
                let mut k_hm = self.pool.acquire(&self.ctx, seq_len * kv_dim)?;
                let mut v_hm = self.pool.acquire(&self.ctx, seq_len * kv_dim)?;
                kernels::launch_qkv_transpose(
                    &self.kernels.transpose,
                    dev_ptr(&k),
                    dev_ptr_mut(&mut k_hm),
                    seq_len,
                    nkh,
                    hd,
                    0,
                )?;
                kernels::launch_qkv_transpose(
                    &self.kernels.transpose,
                    dev_ptr(&v),
                    dev_ptr_mut(&mut v_hm),
                    seq_len,
                    nkh,
                    hd,
                    0,
                )?;

                // Store K/V to cache
                if is_prefill {
                    self.kv_cache.store_kv(
                        layer_idx,
                        &seq.request_id,
                        &k_hm,
                        &v_hm,
                        seq.prefill_pos,
                        seq_len,
                    )?;
                } else {
                    self.kv_cache.append_kv(
                        layer_idx,
                        &seq.request_id,
                        &k_hm,
                        &v_hm,
                    )?;
                }
                self.pool.release(k_hm);
                self.pool.release(v_hm);

                // Attention
                let mut attn_hm = self.pool.acquire(&self.ctx, seq_len * proj_dim)?;
                if is_first_chunk {
                    // Tiled self-attention within this chunk
                    let mut q_hm = self.pool.acquire(&self.ctx, seq_len * proj_dim)?;
                    kernels::launch_qkv_transpose(
                        &self.kernels.transpose,
                        dev_ptr(&q),
                        dev_ptr_mut(&mut q_hm),
                        seq_len,
                        nh,
                        hd,
                        0,
                    )?;
                    self.attn.forward_gpu(
                        dev_ptr(&q_hm),
                        dev_ptr(&k),
                        dev_ptr(&v),
                        dev_ptr_mut(&mut attn_hm),
                        seq_len,
                        nh,
                        hd,
                        true,
                    )?;
                    self.pool.release(q_hm);
                } else {
                    // PagedAttention
                    let total = if is_prefill {
                        seq.prefill_pos + seq_len
                    } else {
                        self.kv_cache.seq_state(&seq.request_id)?.cached_tokens
                    };
                    let seq_len_gpu = self
                        .ctx
                        .device
                        .htod_sync_copy(&[total as i32])
                        .map_err(|e| {
                            ZealotError::Internal(format!("htod seq_len: {e}"))
                        })?;
                    self.attn.paged_forward(
                        dev_ptr(&q),
                        self.kv_cache.k_pool_ptr(layer_idx),
                        self.kv_cache.v_pool_ptr(layer_idx),
                        self.kv_cache.block_table_ptr(&seq.request_id)?,
                        dev_ptr(&seq_len_gpu),
                        dev_ptr_mut(&mut attn_hm),
                        1,
                        nh,
                        nkh,
                        hd,
                        self.kv_cache.block_size,
                        self.kv_cache.max_blocks(&seq.request_id)?,
                    )?;
                    drop(seq_len_gpu);
                }
                self.pool.release(q);
                self.pool.release(k);
                self.pool.release(v);

                // Reverse transpose attn (head-major → interleaved)
                let mut attn_inter = self.pool.acquire(&self.ctx, seq_len * proj_dim)?;
                kernels::launch_qkv_transpose(
                    &self.kernels.transpose,
                    dev_ptr(&attn_hm),
                    dev_ptr_mut(&mut attn_inter),
                    nh,
                    seq_len,
                    hd,
                    1,
                )?;
                self.pool.release(attn_hm);

                // Output projection: [seq_len, hid]
                let mut hidden_proj = self.pool.acquire(&self.ctx, seq_len * hid)?;
                self.ctx.cublas.sgemm(
                    seq_len as i32,
                    hid as i32,
                    proj_dim as i32,
                    dev_ptr(&attn_inter),
                    proj_dim as i32,
                    CublasOp::N,
                    dev_ptr(&lw.attn_proj),
                    hid as i32,
                    CublasOp::T,
                    dev_ptr_mut(&mut hidden_proj),
                    hid as i32,
                )?;
                self.pool.release(attn_inter);

                // ── First residual: hidden = hidden_proj + residual ──
                {
                    // hidden is the residual (from embedding or previous layer)
                    let mut hidden_out = self.pool.acquire(&self.ctx, seq_len * hid)?;
                    kernels::launch_add(
                        &self.kernels.add,
                        dev_ptr(&hidden_proj),
                        dev_ptr(&hidden),
                        dev_ptr_mut(&mut hidden_out),
                        seq_len * hid,
                    )?;
                    self.pool.release(hidden_proj);
                    self.pool.release(hidden);
                    hidden = hidden_out;
                }

                // ── FFN sub-block: SiLU gate-mul + down ────────────
                // hidden is the residual for the FFN sub-block.

                // RMS Norm (ffn weight)
                let mut ffn_normed = self.pool.acquire(&self.ctx, seq_len * hid)?;
                kernels::launch_rms_norm_raw(
                    &self.kernels.rms_norm,
                    dev_ptr(&hidden),
                    dev_ptr(&lw.rms_ffn_weight),
                    dev_ptr_mut(&mut ffn_normed),
                    seq_len,
                    hid,
                    1e-6,
                )?;

                // Gate projection: ffn_normed [seq,hid] @ gate_proj^T [inter,hid] → gate [seq,inter]
                let mut gate = self.pool.acquire(&self.ctx, seq_len * inter_dim)?;
                self.ctx.cublas.sgemm(
                    seq_len as i32,
                    inter_dim as i32,
                    hid as i32,
                    dev_ptr(&ffn_normed),
                    hid as i32,
                    CublasOp::N,
                    dev_ptr(&lw.gate_proj),
                    inter_dim as i32,
                    CublasOp::T,
                    dev_ptr_mut(&mut gate),
                    inter_dim as i32,
                )?;

                // Up projection: ffn_normed [seq,hid] @ up_proj^T [inter,hid] → up [seq,inter]
                let mut up = self.pool.acquire(&self.ctx, seq_len * inter_dim)?;
                self.ctx.cublas.sgemm(
                    seq_len as i32,
                    inter_dim as i32,
                    hid as i32,
                    dev_ptr(&ffn_normed),
                    hid as i32,
                    CublasOp::N,
                    dev_ptr(&lw.up_proj),
                    inter_dim as i32,
                    CublasOp::T,
                    dev_ptr_mut(&mut up),
                    inter_dim as i32,
                )?;
                self.pool.release(ffn_normed);

                // SiLU gate-mul: gated = silu(gate) * up  [same shape]
                let mut gated = self.pool.acquire(&self.ctx, seq_len * inter_dim)?;
                kernels::launch_silu_gate_mul(
                    &self.kernels.silu_gate_mul,
                    dev_ptr(&gate),
                    dev_ptr(&up),
                    dev_ptr_mut(&mut gated),
                    seq_len,
                    inter_dim,
                )?;
                self.pool.release(gate);
                self.pool.release(up);

                // Down projection: gated [seq,inter] @ down_proj^T [hid,inter] → hidden_ffn [seq,hid]
                let mut hidden_ffn = self.pool.acquire(&self.ctx, seq_len * hid)?;
                self.ctx.cublas.sgemm(
                    seq_len as i32,
                    hid as i32,
                    inter_dim as i32,
                    dev_ptr(&gated),
                    inter_dim as i32,
                    CublasOp::N,
                    dev_ptr(&lw.down_proj),
                    hid as i32,
                    CublasOp::T,
                    dev_ptr_mut(&mut hidden_ffn),
                    hid as i32,
                )?;
                self.pool.release(gated);

                // ── Second residual: hidden = hidden_ffn + residual ──
                {
                    // hidden is the residual (first residual output)
                    let mut hidden_out = self.pool.acquire(&self.ctx, seq_len * hid)?;
                    kernels::launch_add(
                        &self.kernels.add,
                        dev_ptr(&hidden_ffn),
                        dev_ptr(&hidden),
                        dev_ptr_mut(&mut hidden_out),
                        seq_len * hid,
                    )?;
                    self.pool.release(hidden_ffn);
                    self.pool.release(hidden);
                    hidden = hidden_out;
                }
            }

            // ── 3. Final RMS Norm (last token only) ─────────────────
            let mut final_h = self.pool.acquire(&self.ctx, hid)?;
            let last_offset = (seq_len - 1) * hid;
            unsafe {
                kernels::launch_rms_norm_raw(
                    &self.kernels.rms_norm,
                    dev_ptr(&hidden).add(last_offset),
                    dev_ptr(&self.weights.rms_final_weight),
                    dev_ptr_mut(&mut final_h),
                    1,
                    hid,
                    1e-6,
                )?;
            }
            self.pool.release(hidden);

            // ── 4. LM Head ─────────────────────────────────────────
            let mut logits_gpu =
                self.pool.acquire(&self.ctx, self.weights.vocab_size)?;
            self.ctx.cublas.sgemm(
                1,
                self.weights.vocab_size as i32,
                hid as i32,
                dev_ptr(&final_h),
                hid as i32,
                CublasOp::N,
                dev_ptr(&self.weights.lm_head),
                self.weights.vocab_size as i32,
                CublasOp::T,
                dev_ptr_mut(&mut logits_gpu),
                self.weights.vocab_size as i32,
            )?;
            self.pool.release(final_h);

            let mut logits_cpu = vec![0.0f32; self.weights.vocab_size];
            self.ctx
                .device
                .dtoh_sync_copy_into(&logits_gpu, &mut logits_cpu)
                .map_err(|e| {
                    ZealotError::Internal(format!("logits dtoh: {e}"))
                })?;
            self.pool.release(logits_gpu);

            self.pool.flush();
            outs.push(StepOut {
                request_id: seq.request_id.clone(),
                token: None,
                logits: Some(logits_cpu),
                text: None,
            });
        }
        Ok(outs)
    }
}

/// Extract `eos_token_id` from a PyO3 tokenizer, returning `None` on
/// any failure.
fn extract_eos(tokenizer_py: Option<&Py<PyAny>>) -> Option<i64> {
    tokenizer_py.and_then(|t| {
        Python::with_gil(|py| t.bind(py).getattr("eos_token_id").and_then(|v| v.extract()).ok())
    })
}

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
use crate::attention::kernels::{dev_ptr, dev_ptr_mut, KernelModule};
use crate::engine::{ModelRunner, StepOut};
use crate::error::ZealotError;
use crate::gpu_context::{GpuContext, GpuWeights};
use crate::model_loader;
use crate::scheduler::Sequence;

// ── CudaModelRunner ──────────────────────────────────────────────────────────

/// GPU-native model runner.  Owns the CUDA context, uploaded weights,
/// pre-compiled PTX kernels, and the flash-attention operator.
pub struct CudaModelRunner {
    ctx: GpuContext,
    weights: GpuWeights,
    kernels: KernelModule,
    attn: CudaAttention,
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

        // 4. Attention operator
        let attn = CudaAttention::new(kernels.attention.clone(), ctx.device.clone());

        // 5. Tokenizer + eos_token_id (PyO3, same pattern as CpuModelRunner)
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

        // 6. Warmup: trigger CUDA lazy init
        let runner = Self {
            ctx,
            weights,
            kernels,
            attn,
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

    /// Launch the RMS norm kernel.
    ///
    /// # Safety (kernel FFI)
    ///
    /// The caller must ensure `x`, `weight`, and `out` are valid device
    /// allocations for the duration of the launch.
    unsafe fn launch_rms_norm_impl(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(), ZealotError> {
        let r = rows as u32;
        let d = dim as u32;
        let x_ptr = dev_ptr(x);
        let w_ptr = dev_ptr(weight);
        let out_ptr = dev_ptr_mut(out);

        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        use std::ffi::c_void;
        let mut params: Vec<*mut c_void> = vec![
            std::ptr::addr_of!(x_ptr) as *mut c_void,
            std::ptr::addr_of!(w_ptr) as *mut c_void,
            std::ptr::addr_of!(out_ptr) as *mut c_void,
            std::ptr::addr_of!(r) as *mut c_void,
            std::ptr::addr_of!(d) as *mut c_void,
            std::ptr::addr_of!(eps) as *mut c_void,
        ];

        self.kernels
            .rms_norm
            .launch(cfg, &mut params)
            .map_err(|e| ZealotError::Internal(format!("rms_norm kernel: {e}")))
    }

    fn launch_rms_norm(
        &self,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        dim: usize,
    ) -> Result<(), ZealotError> {
        let eps: f32 = 1e-6;
        unsafe { self.launch_rms_norm_impl(x, weight, out, rows, dim, eps) }
    }

    // ── step internals (per-sequence) ───────────────────────────────────

    /// Execute one model forward pass on a single sequence.
    ///
    /// Pipeline (all GPU, except Q/K/V reshape which is CPU for Sprint A):
    ///   token_ids → GPU → embedding → RMS norm → Q/K/V cuBLAS → reshape(CPU)
    ///   → attention → output proj → LM head → logits → CPU
    fn step_sequence(&self, seq: &Sequence) -> Result<StepOut, ZealotError> {
        let token_ids = seq.step_input();
        let seq_len = token_ids.len();
        let hidden_dim = self.weights.hidden_dim;
        let num_heads = self.weights.num_heads;
        let head_dim = self.weights.head_dim;
        let proj_dim = num_heads * head_dim;
        let vocab_size = self.weights.vocab_size;

        // ── 1. Upload token ids → GPU ──────────────────────────────────
        let ids_gpu = self.ctx.device.htod_sync_copy(&token_ids).map_err(|e| {
            ZealotError::Internal(format!("htod ids (len={seq_len}): {e}"))
        })?;

        // ── 2. Embedding gather ────────────────────────────────────────
        let mut embed_gpu = self.ctx.alloc_zeros_f32(seq_len * hidden_dim)?;
        self.launch_embedding_gather(
            &ids_gpu,
            &self.weights.embedding,
            &mut embed_gpu,
            seq_len,
            hidden_dim,
        )?;
        drop(ids_gpu);

        // ── 3. RMS Norm ───────────────────────────────────────────────
        let mut normed_gpu = self.ctx.alloc_zeros_f32(seq_len * hidden_dim)?;
        self.launch_rms_norm(
            &embed_gpu,
            &self.weights.rms_attn_weight,
            &mut normed_gpu,
            seq_len,
            hidden_dim,
        )?;
        drop(embed_gpu);

        // ── 4. cuBLAS Q/K/V projections ───────────────────────────────
        // normed [seq_len, hidden_dim] @ weight [proj_dim, hidden_dim]^T
        //   → q/k/v [seq_len, proj_dim]
        let normed_ptr = dev_ptr(&normed_gpu);

        let mut q_gpu = self.ctx.alloc_zeros_f32(seq_len * proj_dim)?;
        let mut k_gpu = self.ctx.alloc_zeros_f32(seq_len * proj_dim)?;
        let mut v_gpu = self.ctx.alloc_zeros_f32(seq_len * proj_dim)?;

        let sgemm_qkv = |weight: &CudaSlice<f32>, out: &mut CudaSlice<f32>| {
            self.ctx.cublas.sgemm(
                seq_len as i32,
                proj_dim as i32,
                hidden_dim as i32,
                normed_ptr,
                hidden_dim as i32,
                CublasOp::N,
                dev_ptr(weight),
                hidden_dim as i32,
                CublasOp::T,
                dev_ptr_mut(out),
                proj_dim as i32,
            )
        };

        sgemm_qkv(&self.weights.q_proj, &mut q_gpu)?;
        sgemm_qkv(&self.weights.k_proj, &mut k_gpu)?;
        sgemm_qkv(&self.weights.v_proj, &mut v_gpu)?;
        drop(normed_gpu);

        // ── 5. Reshape Q/K/V: [seq, proj] → [heads, seq, head] (CPU) ─
        // TODO(Sprint B): replace with GPU transpose / fused QKV kernel
        let q_cpu = self.gpu_to_cpu_f32(&q_gpu, "q dtoh")?;
        let k_cpu = self.gpu_to_cpu_f32(&k_gpu, "k dtoh")?;
        let v_cpu = self.gpu_to_cpu_f32(&v_gpu, "v dtoh")?;
        drop(q_gpu);
        drop(k_gpu);
        drop(v_gpu);

        let q_attn = reshape_qkv(&q_cpu, seq_len, num_heads, head_dim);
        let k_attn = reshape_qkv(&k_cpu, seq_len, num_heads, head_dim);
        let v_attn = reshape_qkv(&v_cpu, seq_len, num_heads, head_dim);

        let q_gpu2 = self.cpu_to_gpu_f32(&q_attn, "q htod")?;
        let k_gpu2 = self.cpu_to_gpu_f32(&k_attn, "k htod")?;
        let v_gpu2 = self.cpu_to_gpu_f32(&v_attn, "v htod")?;

        // ── 6. Flash attention (GPU kernel) ───────────────────────────
        let mut attn_out_gpu = self.ctx.alloc_zeros_f32(
            num_heads * seq_len * head_dim,
        )?;
        self.attn.forward_gpu(
            dev_ptr(&q_gpu2),
            dev_ptr(&k_gpu2),
            dev_ptr(&v_gpu2),
            dev_ptr_mut(&mut attn_out_gpu),
            seq_len,
            num_heads,
            head_dim,
            true, // causal
        )?;
        drop(q_gpu2);
        drop(k_gpu2);
        drop(v_gpu2);

        // ── 7. Extract last row → hidden [proj_dim] ───────────────────
        // attn_out layout: [num_heads, seq_len, head_dim]
        // last row = collect dim 0..head_dim for each head at position (seq_len-1)
        let attn_out_cpu = self.gpu_to_cpu_f32(&attn_out_gpu, "attn_out dtoh")?;
        drop(attn_out_gpu);

        let last_row = gather_last_row(&attn_out_cpu, seq_len, num_heads, head_dim);
        let mut hidden_gpu = self.cpu_to_gpu_f32(&last_row, "last_row htod")?;

        // ── 8. Output projection ──────────────────────────────────────
        // last_row [1, proj_dim] @ attn_proj [hidden_dim, proj_dim]^T
        //   → hidden [1, hidden_dim]
        let mut hidden_out_gpu = self.ctx.alloc_zeros_f32(hidden_dim)?;
        self.ctx.cublas.sgemm(
            1,
            hidden_dim as i32,
            proj_dim as i32,
            dev_ptr(&hidden_gpu),
            proj_dim as i32,
            CublasOp::N,
            dev_ptr(&self.weights.attn_proj),
            proj_dim as i32,
            CublasOp::T,
            dev_ptr_mut(&mut hidden_out_gpu),
            hidden_dim as i32,
        )?;
        drop(hidden_gpu);

        // ── 9. LM head ────────────────────────────────────────────────
        // hidden [1, hidden_dim] @ lm_head [vocab_size, hidden_dim]^T
        //   → logits [1, vocab_size]
        let mut logits_gpu = self.ctx.alloc_zeros_f32(vocab_size)?;
        self.ctx.cublas.sgemm(
            1,
            vocab_size as i32,
            hidden_dim as i32,
            dev_ptr(&hidden_out_gpu),
            hidden_dim as i32,
            CublasOp::N,
            dev_ptr(&self.weights.lm_head),
            hidden_dim as i32,
            CublasOp::T,
            dev_ptr_mut(&mut logits_gpu),
            vocab_size as i32,
        )?;
        drop(hidden_out_gpu);

        // ── 10. Download logits → CPU ─────────────────────────────────
        let logits_cpu = self.gpu_to_cpu_f32(&logits_gpu, "logits dtoh")?;
        drop(logits_gpu);

        Ok(StepOut {
            request_id: seq.request_id.clone(),
            token: None,
            logits: Some(logits_cpu),
            text: None,
        })
    }

    // ── GPU ↔ CPU helpers ──────────────────────────────────────────────

    fn gpu_to_cpu_f32(
        &self,
        slice: &CudaSlice<f32>,
        label: &str,
    ) -> Result<Vec<f32>, ZealotError> {
        self.ctx
            .device
            .dtoh_sync_copy(slice)
            .map_err(|e| ZealotError::Internal(format!("{label}: dtoh_sync_copy: {e}")))
    }

    fn cpu_to_gpu_f32(
        &self,
        src: &[f32],
        label: &str,
    ) -> Result<CudaSlice<f32>, ZealotError> {
        self.ctx
            .device
            .htod_sync_copy(src)
            .map_err(|e| ZealotError::Internal(format!("{label}: htod_sync_copy: {e}")))
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
        // Per-sequence (not batched across sequences) — acceptable for Sprint A.
        for seq in batch.iter() {
            outs.push(self.step_sequence(seq)?);
        }
        Ok(outs)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Reshape Q/K/V from cuBLAS output [seq_len, proj_dim]
/// into attention-kernel layout [num_heads, seq_len, head_dim].
///
/// cuBLAS output: element [pos, head * head_dim + dim]
/// Attention layout: element [head, pos, dim]
fn reshape_qkv(
    src: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let proj_dim = num_heads * head_dim;
    let mut out = vec![0.0_f32; num_heads * seq_len * head_dim];
    for head in 0..num_heads {
        for pos in 0..seq_len {
            for dim in 0..head_dim {
                out[head * seq_len * head_dim + pos * head_dim + dim] =
                    src[pos * proj_dim + head * head_dim + dim];
            }
        }
    }
    out
}

/// Extract the last token's hidden representation from the attention
/// output [num_heads, seq_len, head_dim].
///
/// Returns a [proj_dim] vector concatenating all heads' last position
/// vectors: [head0_dim0, ..., head0_dim(d-1), head1_dim0, ...].
fn gather_last_row(
    attn_out: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let last = seq_len.saturating_sub(1);
    let mut out = vec![0.0_f32; num_heads * head_dim];
    for head in 0..num_heads {
        let base = head * seq_len * head_dim + last * head_dim;
        for dim in 0..head_dim {
            out[head * head_dim + dim] = attn_out[base + dim];
        }
    }
    out
}

/// Extract `eos_token_id` from a PyO3 tokenizer, returning `None` on
/// any failure.
fn extract_eos(tokenizer_py: Option<&Py<PyAny>>) -> Option<i64> {
    tokenizer_py.and_then(|t| {
        Python::with_gil(|py| t.bind(py).getattr("eos_token_id").and_then(|v| v.extract()).ok())
    })
}

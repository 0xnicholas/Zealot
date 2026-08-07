use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::attention::{AttentionBackend, AttentionBatch, CpuAttention};
use crate::engine::{ModelRunner, StepOut};
use crate::error::ZealotError;
use crate::model_loader;
use crate::scheduler::Sequence;

#[allow(dead_code)]
struct KvCache {
    key: Vec<f32>,
    value: Vec<f32>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
pub struct CpuModelRunner {
    attn: Box<dyn AttentionBackend>,
    #[allow(dead_code)]
    states: HashMap<String, KvCache>,
    tokenizer_py: Option<Py<PyAny>>,
    embedding: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    attn_proj: Vec<f32>,
    lm_head: Vec<f32>,
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    eos_token_id: Option<i64>,
}

impl CpuModelRunner {
    pub fn new_cpu() -> Self {
        Self {
            attn: Box::new(CpuAttention),
            states: HashMap::new(),
            tokenizer_py: None,
            embedding: Vec::new(),
            q_proj: Vec::new(),
            k_proj: Vec::new(),
            v_proj: Vec::new(),
            attn_proj: Vec::new(),
            lm_head: Vec::new(),
            hidden_dim: 0,
            num_heads: 0,
            num_kv_heads: 1,
            head_dim: 0,
            num_layers: 0,
            vocab_size: 0,
            eos_token_id: None,
        }
    }

    pub fn load(model_id: &str) -> Result<Self, ZealotError> {
        let weights = model_loader::load_weights(model_id)?;
        let tokenizer_py = Python::with_gil(|py| -> Option<Py<PyAny>> {
            if let Ok(sp) = std::env::var("ZEALOT_SITE_PACKAGES") {
                let sys = py.import("sys").ok()?;
                sys.getattr("path")
                    .ok()?
                    .call_method1("insert", (0, sp))
                    .ok()?;
            }
            let transformers = py.import("transformers").ok()?;
            let tokenizer = transformers
                .getattr("AutoTokenizer")
                .ok()?
                .call_method1("from_pretrained", (model_id,))
                .ok()?;
            Some(tokenizer.unbind())
        });

        let eos_token_id: Option<i64> = tokenizer_py.as_ref().and_then(|t| {
            Python::with_gil(|py| {
                t.bind(py)
                    .getattr("eos_token_id")
                    .and_then(|v| v.extract())
                    .ok()
            })
        });

        Ok(Self {
            attn: Box::new(CpuAttention),
            states: HashMap::new(),
            tokenizer_py,
            embedding: weights.embedding,
            q_proj: weights.q_proj,
            k_proj: weights.k_proj,
            v_proj: weights.v_proj,
            attn_proj: weights.attn_proj,
            lm_head: weights.lm_head,
            hidden_dim: weights.hidden_dim,
            num_heads: weights.num_heads,
            num_kv_heads: weights.num_kv_heads.max(1),
            head_dim: weights.head_dim,
            num_layers: weights.num_layers,
            vocab_size: weights.vocab_size,
            eos_token_id,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn load_cuda(model_id: &str, _device_id: usize) -> Result<Self, ZealotError> {
        // CudaAttention is now GPU-native (does not impl AttentionBackend).
        // The full GPU model runner will be wired up in a later milestone.
        let _ = model_id;
        Err(ZealotError::Internal(
            "CudaModelRunner not yet wired — use CpuModelRunner or PyModelRunner".into(),
        ))
    }
}

impl ModelRunner for CpuModelRunner {
    fn tokenize_chat(&self, messages: &[(String, String)]) -> Result<Vec<i64>, ZealotError> {
        Python::with_gil(|py| -> Result<Vec<i64>, ZealotError> {
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
                list.append(d).map_err(model_loader::py_err("msg append"))?;
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
        })
    }

    fn eos_token_id(&self) -> Option<i64> {
        self.eos_token_id
    }

    fn step(&mut self, batch: &mut [&mut Sequence]) -> Result<Vec<StepOut>, ZealotError> {
        let mut outs = Vec::with_capacity(batch.len());

        for seq in batch.iter() {
            let token_ids = seq.step_input();
            let seq_len = token_ids.len();

            let mut embed_out = vec![0.0_f32; seq_len * self.hidden_dim];
            for (i, &id) in token_ids.iter().enumerate() {
                let start = id as usize * self.hidden_dim;
                for d in 0..self.hidden_dim {
                    embed_out[i * self.hidden_dim + d] =
                        *self.embedding.get(start + d).unwrap_or(&0.0);
                }
            }

            let proj_dim = self.num_heads * self.head_dim;
            // M5-⑫（issue #26）：GQA —— k/v 按 kv 头宽投影（num_kv_heads < num_heads
            // 时全宽投影会越界 panic），再 repeat-interleave 展开到 q 头数。
            let kv_heads = self.num_kv_heads.max(1);
            let kv_proj_dim = kv_heads * self.head_dim;
            let q_projected = crate::attention::matmul(
                &embed_out,
                &self.q_proj,
                seq_len,
                self.hidden_dim,
                proj_dim,
            );
            let k_projected = crate::attention::matmul(
                &embed_out,
                &self.k_proj,
                seq_len,
                self.hidden_dim,
                kv_proj_dim,
            );
            let v_projected = crate::attention::matmul(
                &embed_out,
                &self.v_proj,
                seq_len,
                self.hidden_dim,
                kv_proj_dim,
            );

            let h = self.num_heads;
            let d = self.head_dim;
            let reshape = |src: &[f32]| {
                let mut out = vec![0.0_f32; h * seq_len * d];
                for head in 0..h {
                    for pos in 0..seq_len {
                        for dim in 0..d {
                            out[head * seq_len * d + pos * d + dim] =
                                src[pos * proj_dim + head * d + dim];
                        }
                    }
                }
                out
            };
            let q = reshape(&q_projected);
            let k = repeat_interleave_kv(&k_projected, kv_heads, h, seq_len, d);
            let v = repeat_interleave_kv(&v_projected, kv_heads, h, seq_len, d);

            let attn_batch = AttentionBatch {
                num_seqs: 1,
                num_heads: h,
                head_dim: d,
                max_seq_len: seq_len,
            };
            let attn_out = self.attn.forward(&q, &k, &v, &attn_batch)?;

            let last = seq_len.saturating_sub(1);
            let mut last_hidden = vec![0.0_f32; self.hidden_dim];
            for head in 0..h {
                let base = head * seq_len * d + last * d;
                for dim in 0..d {
                    if base + dim < attn_out.len() {
                        last_hidden[head * d + dim] = attn_out[base + dim];
                    }
                }
            }

            let attn_hidden = crate::attention::matmul(
                &last_hidden,
                &self.attn_proj,
                1,
                self.hidden_dim,
                self.hidden_dim,
            );
            let logits = crate::attention::matmul(
                &attn_hidden,
                &self.lm_head,
                1,
                self.hidden_dim,
                self.vocab_size,
            );

            outs.push(StepOut {
                request_id: seq.request_id.clone(),
                token: None,
                logits: Some(logits),
                text: None,
            });
        }

        Ok(outs)
    }

    fn drop_state(&mut self, request_id: &str) {
        self.states.remove(request_id);
    }
}

/// GQA kv 头展开（repeat-interleave）：
/// 把 [seq_len × kv_heads×d] 的扁平投影展为 [num_heads × seq_len × d]，
/// 第 head 个 q 头共享第 head % kv_heads 个 kv 头（MHA 时恒等）。
fn repeat_interleave_kv(
    src: &[f32],
    kv_heads: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; num_heads * seq_len * head_dim];
    for head in 0..num_heads {
        let kh = head % kv_heads;
        for pos in 0..seq_len {
            for dim in 0..head_dim {
                out[head * seq_len * head_dim + pos * head_dim + dim] =
                    src[pos * kv_heads * head_dim + kh * head_dim + dim];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cpu_has_cpu_attention() {
        let runner = CpuModelRunner::new_cpu();
        assert_eq!(runner.states.len(), 0);
        assert_eq!(runner.hidden_dim, 0);
    }
}

#[cfg(test)]
mod gqa_tests {
    use super::*;

    #[test]
    fn repeat_interleave_expands_kv_heads() {
        // 2 kv 头 → 4 q 头：head % kv_heads 映射
        let kv_heads = 2;
        let num_heads = 4;
        let seq_len = 3;
        let d = 2;
        // src[pos, kh, dim] = pos*100 + kh*10 + dim
        let src: Vec<f32> = (0..seq_len * kv_heads * d).map(|i| i as f32).collect();
        let out = repeat_interleave_kv(&src, kv_heads, num_heads, seq_len, d);
        assert_eq!(out.len(), num_heads * seq_len * d);
        for head in 0..num_heads {
            let kh = head % kv_heads;
            for pos in 0..seq_len {
                for dim in 0..d {
                    assert_eq!(
                        out[head * seq_len * d + pos * d + dim],
                        src[pos * kv_heads * d + kh * d + dim]
                    );
                }
            }
        }
    }

    #[test]
    fn repeat_interleave_mha_matches_old_reshape() {
        // MHA（kv == heads）：输出与旧全宽 reshape 的 [heads×seq×d] 布局一致
        let seq_len = 2;
        let heads = 2;
        let d = 3;
        let src: Vec<f32> = (0..seq_len * heads * d).map(|i| i as f32).collect();
        let out = repeat_interleave_kv(&src, heads, heads, seq_len, d);
        let mut expect = vec![0.0_f32; heads * seq_len * d];
        for head in 0..heads {
            for pos in 0..seq_len {
                for dim in 0..d {
                    expect[head * seq_len * d + pos * d + dim] =
                        src[pos * heads * d + head * d + dim];
                }
            }
        }
        assert_eq!(out, expect);
    }

    #[test]
    fn gqa_step_no_panic() {
        // 合成 GQA 权重（kv=2 < heads=4）跑一步：k/v 越界回归（issue #26）。
        let mut runner = CpuModelRunner::new_cpu();
        runner.hidden_dim = 8;
        runner.num_heads = 4;
        runner.num_kv_heads = 2;
        runner.head_dim = 2;
        runner.num_layers = 1;
        runner.vocab_size = 4;
        let rand = |len: usize| (0..len).map(|i| (i % 7) as f32 / 7.0).collect::<Vec<_>>();
        runner.embedding = rand(4 * 8); // [vocab, hidden]
        runner.q_proj = rand(8 * 8); // [hidden, heads*d]
        runner.k_proj = rand(8 * 4); // [hidden, kv_heads*d] ← GQA 窄投影
        runner.v_proj = rand(8 * 4);
        runner.attn_proj = rand(8 * 8);
        runner.lm_head = rand(8 * 4); // [hidden, vocab]

        let mut sched =
            crate::scheduler::Scheduler::new(crate::scheduler::SchedulerConfig::default()).unwrap();
        let mut seq = sched
            .make_sequence(
                "gqa".into(),
                vec![1, 2, 3],
                16,
                crate::scheduler::Priority::Medium,
                Some(0),
                crate::sampling::SamplingParams::default(),
            )
            .unwrap();
        let mut batch = [&mut seq];
        let outs = runner.step(&mut batch).expect("GQA step 不应越界 panic");
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].logits.as_ref().expect("logits").len(), 4);
    }
}

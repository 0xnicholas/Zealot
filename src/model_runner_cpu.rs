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
            head_dim: 0,
            num_layers: 0,
            vocab_size: 0,
            eos_token_id: None,
        }
    }

    pub fn load(model_id: &str) -> Result<Self, ZealotError> {
        let weights = model_loader::load_weights(model_id)?;
        let tokenizer_py = Python::with_gil(|py| -> Result<Option<Py<PyAny>>, ZealotError> {
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
        })?;

        let eos_token_id: Option<i64> = tokenizer_py
            .as_ref()
            .and_then(|t| {
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
                proj_dim,
            );
            let v_projected = crate::attention::matmul(
                &embed_out,
                &self.v_proj,
                seq_len,
                self.hidden_dim,
                proj_dim,
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
            let k = reshape(&k_projected);
            let v = reshape(&v_projected);

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

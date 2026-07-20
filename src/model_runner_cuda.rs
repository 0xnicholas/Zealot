use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::attention::{AttentionBackend, AttentionBatch, CpuAttention};
use crate::engine::{ModelRunner, StepOut};
use crate::error::ZealotError;
use crate::scheduler::Sequence;

fn py_err(ctx: &str) -> impl Fn(PyErr) -> ZealotError + '_ {
    move |e: PyErr| ZealotError::Internal(format!("{ctx}: {e}"))
}

#[allow(dead_code)]
struct KvCache {
    key: Vec<f32>,
    value: Vec<f32>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

pub struct CudaModelRunner {
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

impl CudaModelRunner {
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
        Python::with_gil(|py| -> Result<Self, ZealotError> {
            if let Ok(sp) = std::env::var("ZEALOT_SITE_PACKAGES") {
                let sys = py.import("sys").map_err(py_err("import sys"))?;
                sys.getattr("path")
                    .map_err(py_err("sys.path"))?
                    .call_method1("insert", (0, sp))
                    .map_err(py_err("sys.path.insert"))?;
            }

            let _torch = py.import("torch")
                .map_err(py_err("import torch (venv? set ZEALOT_SITE_PACKAGES)"))?;
            let transformers = py.import("transformers")
                .map_err(py_err("import transformers"))?;

            let tokenizer = transformers
                .getattr("AutoTokenizer")
                .map_err(py_err("AutoTokenizer"))?
                .call_method1("from_pretrained", (model_id,))
                .map_err(py_err("tokenizer from_pretrained"))?;
            let model = transformers
                .getattr("AutoModelForCausalLM")
                .map_err(py_err("AutoModelForCausalLM"))?
                .call_method1("from_pretrained", (model_id,))
                .map_err(py_err("model from_pretrained"))?;

            let eos_token_id: Option<i64> =
                tokenizer.getattr("eos_token_id").and_then(|v| v.extract()).ok();

            let config = model.getattr("config").map_err(py_err("model.config"))?;
            let hidden_dim: usize = config.getattr("hidden_size").map_err(py_err("hidden_size"))?.extract().map_err(py_err("hidden_size"))?;
            let num_heads: usize = config.getattr("num_attention_heads").map_err(py_err("num_attention_heads"))?.extract().map_err(py_err("num_attention_heads"))?;
            let head_dim: usize = hidden_dim / num_heads;
            let num_layers: usize = config.getattr("num_hidden_layers").map_err(py_err("num_hidden_layers"))?.extract().map_err(py_err("num_hidden_layers"))?;

            let wte = model
                .getattr("transformer")
                .and_then(|t| t.getattr("wte"))
                .or_else(|_| model.getattr("model").and_then(|m| m.getattr("embed_tokens")))
                .map_err(py_err("embed_tokens"))?
                .getattr("weight")
                .map_err(py_err("weight"))?;
            let wte_data: Vec<f32> = to_vec_f32(&wte)?;
            let vocab_size = wte_data.len() / hidden_dim;

            let extract_layer = |suffix: &str| -> Result<Vec<f32>, ZealotError> {
                let layer = model
                    .getattr("transformer")
                    .and_then(|t| t.getattr("h"))
                    .or_else(|_| model.getattr("model").and_then(|m| m.getattr("layers")))
                    .map_err(py_err("layers"))?;
                let l0 = layer.get_item(0).map_err(py_err("layer[0]"))?;
                let attn = l0.getattr("attn")
                    .or_else(|_| l0.getattr("self_attn").or_else(|_| l0.getattr("attention")))
                    .map_err(py_err("attn"))?;
                let w = attn.getattr(suffix).map_err(py_err("qkv suffix"))?.getattr("weight").map_err(py_err("weight"))?;
                to_vec_f32(&w)
            };

            let q_proj = extract_layer("q_proj")?;
            let k_proj = extract_layer("k_proj")?;
            let v_proj = extract_layer("v_proj")?;
            let attn_proj = extract_layer("o_proj")?;

            let lm_head_data = model
                .getattr("lm_head")
                .ok()
                .and_then(|h| h.getattr("weight").ok())
                .and_then(|w| to_vec_f32(&w).ok())
                .unwrap_or_else(|| wte_data.clone());

            Ok(Self {
                attn: Box::new(CpuAttention),
                states: HashMap::new(),
                tokenizer_py: Some(tokenizer.unbind()),
                embedding: wte_data,
                q_proj,
                k_proj,
                v_proj,
                attn_proj,
                lm_head: lm_head_data,
                hidden_dim,
                num_heads,
                head_dim,
                num_layers,
                vocab_size,
                eos_token_id,
            })
        })
    }

    #[cfg(feature = "cuda")]
    pub fn load_cuda(model_id: &str, device_id: usize) -> Result<Self, ZealotError> {
        use crate::attention::cuda::CudaAttention;
        let mut s = Self::load(model_id)?;
        s.attn = Box::new(CudaAttention::new(device_id)?);
        Ok(s)
    }
}

fn to_vec_f32(tensor: &Bound<'_, PyAny>) -> Result<Vec<f32>, ZealotError> {
    let cpu = tensor.call_method0("cpu").or_else(|_| -> Result<Bound<'_, PyAny>, PyErr> { Ok(tensor.to_owned()) })
        .map_err(|e| ZealotError::Internal(format!("cpu: {e}")))?;
    let flat = cpu
        .call_method1("reshape", (-1,))
        .map_err(|e| ZealotError::Internal(format!("reshape: {e}")))?;
    let list = flat
        .call_method0("tolist")
        .map_err(|e| ZealotError::Internal(format!("tolist: {e}")))?;
    list.extract::<Vec<f32>>()
        .map_err(|e| ZealotError::Internal(format!("extract: {e}")))
}

impl ModelRunner for CudaModelRunner {
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
                d.set_item("role", role).map_err(py_err("msg role"))?;
                d.set_item("content", content).map_err(py_err("msg content"))?;
                list.append(d).map_err(py_err("msg append"))?;
            }
            let kwargs = PyDict::new(py);
            kwargs.set_item("tokenize", true).map_err(py_err("kw"))?;
            kwargs.set_item("add_generation_prompt", true).map_err(py_err("kw"))?;
            match tokenizer.call_method("apply_chat_template", (list,), Some(&kwargs)) {
                Ok(ids) => ids.extract::<Vec<i64>>().map_err(py_err("chat_template ids")),
                Err(_) => {
                    let text = messages
                        .iter()
                        .map(|(r, c)| format!("{r}: {c}"))
                        .collect::<Vec<_>>()
                        .join("\n") + "\n";
                    tokenizer
                        .call_method1("encode", (text,))
                        .map_err(py_err("encode"))?
                        .extract::<Vec<i64>>()
                        .map_err(py_err("encode ids"))
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
                    embed_out[i * self.hidden_dim + d] = *self.embedding.get(start + d).unwrap_or(&0.0);
                }
            }

            let proj_dim = self.num_heads * self.head_dim;
            let q_projected = crate::attention::matmul(&embed_out, &self.q_proj, seq_len, self.hidden_dim, proj_dim);
            let k_projected = crate::attention::matmul(&embed_out, &self.k_proj, seq_len, self.hidden_dim, proj_dim);
            let v_projected = crate::attention::matmul(&embed_out, &self.v_proj, seq_len, self.hidden_dim, proj_dim);

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
                &last_hidden, &self.attn_proj, 1, self.hidden_dim, self.hidden_dim,
            );
            let logits = crate::attention::matmul(
                &attn_hidden, &self.lm_head, 1, self.hidden_dim, self.vocab_size,
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
        let runner = CudaModelRunner::new_cpu();
        assert_eq!(runner.states.len(), 0);
        assert_eq!(runner.hidden_dim, 0);
    }
}

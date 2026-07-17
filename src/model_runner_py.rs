//! PyModelRunner — dev-mode ModelRunner：嵌入式 CPython + PyTorch CPU。
//!
//! ⚠️ DEV-MODE ONLY：前向计算在 Python（PyTorch）里，每个 decode step 都穿越
//! Rust→Python 边界并持有 GIL。目标架构（ADR-009）里 Python 仅在启动时加载
//! 权重、不进 decode loop；本 runner 是 GPU/CUDA kernel 到位前的验证替身，
//! 让 Zealot 独立进程可以先跑通真实端到端推理。
//!
//! 环境：嵌入式解释器是构建期 PYO3_PYTHON 指定的 python3.12；torch /
//! transformers 装在独立 venv，运行时用 ZEALOT_SITE_PACKAGES 指向其
//! site-packages（不污染系统 Python）。

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::engine::{ModelRunner, StepOut};
use crate::error::ZealotError;
use crate::scheduler::Sequence;

fn py_err(ctx: &str) -> impl Fn(PyErr) -> ZealotError + '_ {
    move |e: PyErr| ZealotError::Internal(format!("{ctx}: {e}"))
}

pub struct PyModelRunner {
    torch: Py<PyAny>,
    tokenizer: Py<PyAny>,
    model: Py<PyAny>,
    /// request_id → past_key_values（PyTorch KV cache，dev-mode 放在 Python 侧；
    /// BlockManager 的 block 簿记与之并行，仅验证分配逻辑）
    states: HashMap<String, Py<PyAny>>,
    eos_token_id: Option<i64>,
}

impl PyModelRunner {
    /// 加载 HF 模型（CPU）。耗时操作（下载/读盘），调用方应放在阻塞线程。
    pub fn load(model_id: &str) -> Result<Self, ZealotError> {
        Python::with_gil(|py| -> Result<Self, ZealotError> {
            // venv 依赖注入（torch/transformers 不在系统 site-packages）
            if let Ok(sp) = std::env::var("ZEALOT_SITE_PACKAGES") {
                let sys = py.import("sys").map_err(py_err("import sys"))?;
                sys.getattr("path")
                    .map_err(py_err("sys.path"))?
                    .call_method1("insert", (0, sp))
                    .map_err(py_err("sys.path.insert"))?;
            }

            let torch = py.import("torch").map_err(py_err(
                "import torch (venv? set ZEALOT_SITE_PACKAGES)",
            ))?;
            let transformers = py
                .import("transformers")
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
            model.call_method0("eval").map_err(py_err("model.eval"))?;
            // dev-mode 不做梯度
            torch
                .call_method1("set_grad_enabled", (false,))
                .map_err(py_err("set_grad_enabled"))?;

            let eos_token_id: Option<i64> = tokenizer
                .getattr("eos_token_id")
                .and_then(|v| v.extract()).ok();

            Ok(Self {
                torch: torch.unbind().into(),
                tokenizer: tokenizer.unbind(),
                model: model.unbind(),
                states: HashMap::new(),
                eos_token_id,
            })
        })
    }

    pub fn eos_token_id(&self) -> Option<i64> {
        self.eos_token_id
    }

    /// chat 消息 → token ids。优先 apply_chat_template；模型无模板时
    /// 退化为 "role: content" 纯文本拼接（如 tiny-random-gpt2）。
    pub fn tokenize_chat(&self, messages: &[(String, String)]) -> Result<Vec<i64>, ZealotError> {
        Python::with_gil(|py| -> Result<Vec<i64>, ZealotError> {
            let tokenizer = self.tokenizer.bind(py);
            let list = PyList::empty(py);
            for (role, content) in messages {
                let d = PyDict::new(py);
                d.set_item("role", role).map_err(py_err("msg role"))?;
                d.set_item("content", content)
                    .map_err(py_err("msg content"))?;
                list.append(d).map_err(py_err("msg append"))?;
            }
            let kwargs = PyDict::new(py);
            kwargs.set_item("tokenize", true).map_err(py_err("kw"))?;
            kwargs
                .set_item("add_generation_prompt", true)
                .map_err(py_err("kw"))?;
            match tokenizer.call_method("apply_chat_template", (list,), Some(&kwargs)) {
                Ok(ids) => ids.extract::<Vec<i64>>().map_err(py_err("chat_template ids")),
                Err(_) => {
                    // 无 chat template 的模型：纯文本拼接
                    let text = messages
                        .iter()
                        .map(|(r, c)| format!("{r}: {c}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                        + "\n";
                    tokenizer
                        .call_method1("encode", (text,))
                        .map_err(py_err("encode"))?
                        .extract::<Vec<i64>>()
                        .map_err(py_err("encode ids"))
                }
            }
        })
    }

    /// 增量 detokenize。将用于未来的 Rust tokenizer 集成。
    #[allow(dead_code)]
    fn decode_token(&self, py: Python<'_>, token: i64) -> Option<String> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("skip_special_tokens", true).ok()?;
        let ids = PyList::new(py, [token]).ok()?;
        self.tokenizer
            .bind(py)
            .call_method("decode", (ids,), Some(&kwargs))
            .and_then(|s| s.extract::<String>())
            .ok()
    }
}

impl ModelRunner for PyModelRunner {
    fn step(&mut self, batch: &mut [&mut Sequence]) -> Result<Vec<StepOut>, ZealotError> {
        Python::with_gil(|py| -> Result<Vec<StepOut>, ZealotError> {
            let torch = self.torch.bind(py);
            let mut outs = Vec::with_capacity(batch.len());
            for seq in batch.iter() {
                let ids = seq.step_input();
                let input = PyList::new(py, [ids]).map_err(py_err("input ids"))?;
                let tensor = torch
                    .call_method1("tensor", (input,))
                    .map_err(py_err("torch.tensor"))?;

                let kwargs = PyDict::new(py);
                kwargs.set_item("use_cache", true).map_err(py_err("kw"))?;
                if let Some(past) = self.states.get(&seq.request_id) {
                    kwargs
                        .set_item("past_key_values", past.bind(py))
                        .map_err(py_err("kw past"))?;
                }
                let out = self
                    .model
                    .call(py, (tensor,), Some(&kwargs))
                    .map_err(py_err("model forward"))?;
                let out = out.bind(py);

                // Extract raw logits for the last token position.
                // Engine will run the Rust Sampler on these.
                let logits = out.getattr("logits").map_err(py_err("logits"))?;
                let last = logits.get_item((0, -1)).map_err(py_err("logits last"))?;
                // Convert to Python list then to Rust Vec<f32>
                let logits_list: Vec<f32> = last
                    .call_method0("tolist")
                    .and_then(|l| l.extract())
                    .map_err(py_err("logits tolist"))?;

                let past = out
                    .getattr("past_key_values")
                    .map_err(py_err("past_key_values"))?;
                self.states.insert(seq.request_id.clone(), past.unbind());

                outs.push(StepOut {
                    request_id: seq.request_id.clone(),
                    token: None,     // Let Engine's Sampler decide
                    logits: Some(logits_list),
                    text: None,      // Detokenization happens post-sampling (future: Rust tokenizer)
                });
            }
            Ok(outs)
        })
    }

    fn drop_state(&mut self, request_id: &str) {
        self.states.remove(request_id);
    }
}

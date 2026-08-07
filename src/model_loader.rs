use pyo3::prelude::*;

use crate::error::ZealotError;

/// Per-layer weight bundle extracted from a HuggingFace model via PyO3.
pub struct LayerWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub rms_attn_weight: Vec<f32>,
    pub rms_ffn_weight: Vec<f32>,
    pub gate_proj: Vec<f32>, // [hidden_dim, intermediate_dim]
    pub up_proj: Vec<f32>,   // [hidden_dim, intermediate_dim]
    pub down_proj: Vec<f32>, // [intermediate_dim, hidden_dim]
}

/// Weight and metadata extracted from a HuggingFace model via PyO3.
pub struct ModelWeights {
    pub embedding: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub rms_attn_weight: Vec<f32>,
    pub rms_ffn_weight: Vec<f32>,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub intermediate_dim: usize,
    /// Multi-layer weights (for GPU runner).  Coexists with the scalar
    /// fields above (which CpuModelRunner still uses).
    pub layers: Vec<LayerWeights>,
    pub num_kv_heads: usize,
    pub rms_final_weight: Vec<f32>,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub fn py_err(ctx: &str) -> impl Fn(PyErr) -> ZealotError + '_ {
    move |e: PyErr| ZealotError::Internal(format!("{ctx}: {e}"))
}

pub fn to_vec_f32(tensor: &Bound<'_, PyAny>) -> Result<Vec<f32>, ZealotError> {
    let cpu = tensor
        .call_method0("cpu")
        .or_else(|_| -> Result<Bound<'_, PyAny>, PyErr> { Ok(tensor.to_owned()) })
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

/// Probe a chain of attributes, trying each chain in `chains` until one succeeds.
/// Each chain is a slice of attribute names to follow from `obj`.
pub fn get_attr_chain<'py>(
    obj: &Bound<'py, PyAny>,
    chains: &[&[&str]],
    label: &str,
) -> Result<Bound<'py, PyAny>, ZealotError> {
    let mut last_err: Option<PyErr> = None;
    for chain in chains {
        let mut current = obj.to_owned();
        let mut ok = true;
        for attr in chain.iter() {
            match current.getattr(*attr) {
                Ok(val) => current = val,
                Err(e) => {
                    last_err = Some(e);
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Ok(current);
        }
    }
    let detail = last_err
        .map(|e| e.to_string())
        .unwrap_or_else(|| "unknown".into());
    Err(ZealotError::Internal(format!(
        "{label}: no attribute chain worked ({detail})"
    )))
}

/// Get the first layer (index 0) from the model's transformer block.
pub fn get_first_layer<'py>(model: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>, ZealotError> {
    let container = get_attr_chain(
        model,
        &[&["transformer", "h"], &["model", "layers"]],
        "layers",
    )?;
    container.get_item(0).map_err(py_err("layer[0]"))
}

/// Extract a weight vector from a parent object by probing attribute chains then
/// accessing the named suffix's `.weight` attribute.
pub fn extract_weight<'py>(
    parent: &Bound<'py, PyAny>,
    attr_chains: &[&[&str]],
    suffix: &str,
    label: &str,
) -> Result<Vec<f32>, ZealotError> {
    let container = get_attr_chain(parent, attr_chains, label)?;
    let sub = container.getattr(suffix).map_err(py_err(label))?;
    let w = sub.getattr("weight").map_err(py_err("weight"))?;
    to_vec_f32(&w)
}

/// Extract a weight vector from a parent by probing attribute chains, then
/// accessing `.weight` directly (for norms / embeddings).
pub fn extract_direct_weight<'py>(
    parent: &Bound<'py, PyAny>,
    attr_chains: &[&[&str]],
    label: &str,
) -> Result<Vec<f32>, ZealotError> {
    let container = get_attr_chain(parent, attr_chains, label)?;
    let w = container.getattr("weight").map_err(py_err("weight"))?;
    to_vec_f32(&w)
}

// ── 架构探测与命名适配（M5-⑫，issue #26）─────────────────────────────────

/// 模型架构风格：决定权重命名适配（不再写死 LLaMA 命名）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArchStyle {
    /// LLaMA 系（llama/mistral）：self_attn.q/k/v/o_proj + mlp.gate/up/down_proj
    Llama,
    /// GPT-2（gpt2）：attn.c_attn（3×embed_dim Conv1D，行切分 q/k/v）+ attn.c_proj；
    /// mlp.c_fc（充当 gate/up）+ mlp.c_proj（充当 down）
    Gpt2,
}

/// 探测架构（config.model_type）；不在支持列表 → 显式报错（不再 AttributeError 汤）。
pub fn probe_arch(config: &Bound<'_, PyAny>) -> Result<ArchStyle, ZealotError> {
    let model_type: String = config
        .getattr("model_type")
        .map_err(py_err("config.model_type"))?
        .extract()
        .map_err(py_err("config.model_type"))?;
    match model_type.as_str() {
        "llama" | "mistral" => Ok(ArchStyle::Llama),
        "gpt2" => Ok(ArchStyle::Gpt2),
        other => Err(ZealotError::Internal(format!(
            "不支持的模型架构 {other:?}（支持: llama/mistral/gpt2）"
        ))),
    }
}

/// q/k/v/o 投影权重（架构适配：GPT-2 c_attn 行切分 q/k/v）。
/// 返回 (q, k, v, o)，均为 [hidden, proj] 扁平布局。
pub type AttnWeights = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

pub fn extract_attn_weights(
    parent: &Bound<'_, PyAny>,
    attn_chains: &[&[&str]],
    style: ArchStyle,
    label: &str,
) -> Result<AttnWeights, ZealotError> {
    match style {
        ArchStyle::Llama => {
            let q = extract_weight(parent, attn_chains, "q_proj", &format!("{label}.q_proj"))?;
            let k = extract_weight(parent, attn_chains, "k_proj", &format!("{label}.k_proj"))?;
            let v = extract_weight(parent, attn_chains, "v_proj", &format!("{label}.v_proj"))?;
            let o = extract_weight(parent, attn_chains, "o_proj", &format!("{label}.o_proj"))?;
            Ok((q, k, v, o))
        }
        ArchStyle::Gpt2 => {
            // Conv1D 权重形状 (out, in)：out = 3×embed_dim，行序 [q | k | v]
            let c_attn = extract_weight(parent, attn_chains, "c_attn", &format!("{label}.c_attn"))?;
            let per = c_attn.len() / 3;
            if per == 0 || c_attn.len() % 3 != 0 {
                return Err(ZealotError::Internal(format!(
                    "{label}.c_attn 形状异常（len {} 非 3 的倍数）",
                    c_attn.len()
                )));
            }
            let q = c_attn[..per].to_vec();
            let k = c_attn[per..2 * per].to_vec();
            let v = c_attn[2 * per..].to_vec();
            let o = extract_weight(parent, attn_chains, "c_proj", &format!("{label}.c_proj"))?;
            Ok((q, k, v, o))
        }
    }
}

/// gate/up/down 权重（架构适配：GPT-2 c_fc/c_proj）。
/// GPT-2 MLP 为 act(c_fc(x)) → c_proj；LLaMA 为 gate(x)·up(x) → down。
/// 适配：gate := c_fc、up := c_fc（无独立第二支，CPU runner 未使用 FFN 权重，
/// 语义等价性留给未来 runner 前向时处理）。
pub type MlpWeights = (Vec<f32>, Vec<f32>, Vec<f32>);

pub fn extract_mlp_weights(
    parent: &Bound<'_, PyAny>,
    style: ArchStyle,
    label: &str,
) -> Result<MlpWeights, ZealotError> {
    let mlp_chains: &[&[&str]] = &[&["mlp"]];
    match style {
        ArchStyle::Llama => {
            let gate = extract_weight(
                parent,
                mlp_chains,
                "gate_proj",
                &format!("{label}.gate_proj"),
            )?;
            let up = extract_weight(parent, mlp_chains, "up_proj", &format!("{label}.up_proj"))?;
            let down = extract_weight(
                parent,
                mlp_chains,
                "down_proj",
                &format!("{label}.down_proj"),
            )?;
            Ok((gate, up, down))
        }
        ArchStyle::Gpt2 => {
            let gate = extract_weight(parent, mlp_chains, "c_fc", &format!("{label}.c_fc"))?;
            let up = gate.clone();
            let down = extract_weight(parent, mlp_chains, "c_proj", &format!("{label}.c_proj"))?;
            Ok((gate, up, down))
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Load model weights and config from a HuggingFace model ID via PyO3.
///
/// This extracts embedding, Q/K/V/O projection weights, RMS norm weights
/// (attention and FFN), and the LM head.  Tokenizer loading is handled
/// separately by the caller (if needed).
pub fn load_weights(model_id: &str) -> Result<ModelWeights, ZealotError> {
    Python::with_gil(|py| -> Result<ModelWeights, ZealotError> {
        // ── Setup site-packages if ZEALOT_SITE_PACKAGES is set ──────────
        if let Ok(sp) = std::env::var("ZEALOT_SITE_PACKAGES") {
            let sys = py.import("sys").map_err(py_err("import sys"))?;
            sys.getattr("path")
                .map_err(py_err("sys.path"))?
                .call_method1("insert", (0, sp))
                .map_err(py_err("sys.path.insert"))?;
        }

        let _torch = py
            .import("torch")
            .map_err(py_err("import torch (venv? set ZEALOT_SITE_PACKAGES)"))?;
        let transformers = py
            .import("transformers")
            .map_err(py_err("import transformers"))?;

        let model = transformers
            .getattr("AutoModelForCausalLM")
            .map_err(py_err("AutoModelForCausalLM"))?
            .call_method1("from_pretrained", (model_id,))
            .map_err(py_err("model from_pretrained"))?;

        // ── Config ────────────────────────────────────────────────────
        let config = model.getattr("config").map_err(py_err("model.config"))?;
        let hidden_dim: usize = config
            .getattr("hidden_size")
            .map_err(py_err("hidden_size"))?
            .extract()
            .map_err(py_err("hidden_size"))?;
        let num_heads: usize = config
            .getattr("num_attention_heads")
            .map_err(py_err("num_attention_heads"))?
            .extract()
            .map_err(py_err("num_attention_heads"))?;
        let head_dim: usize = hidden_dim / num_heads;
        let num_layers: usize = config
            .getattr("num_hidden_layers")
            .map_err(py_err("num_hidden_layers"))?
            .extract()
            .map_err(py_err("num_hidden_layers"))?;
        let intermediate_dim: usize = config
            .getattr("intermediate_size")
            .map_err(py_err("intermediate_size"))?
            .extract()
            .map_err(py_err("intermediate_size"))?;

        // ── Embedding ─────────────────────────────────────────────────
        let embedding = extract_direct_weight(
            &model,
            &[&["transformer", "wte"], &["model", "embed_tokens"]],
            "embed_tokens",
        )?;
        let vocab_size = embedding.len() / hidden_dim;

        // ── 架构探测（M5-⑫）：LLaMA 系 / GPT-2 命名适配 ─────────────────
        let arch = probe_arch(&config)?;

        // ── Attention weights (Q/K/V/O projection) ────────────────────
        let first_layer = get_first_layer(&model)?;
        let attn_chains: &[&[&str]] = &[&["attn"], &["self_attn"], &["attention"]];

        let (q_proj, k_proj, v_proj, attn_proj) =
            extract_attn_weights(&first_layer, attn_chains, arch, "layer[0].attn")?;

        // ── FFN weights ────────────────────────────────────────────
        let (gate_proj, up_proj, down_proj) =
            extract_mlp_weights(&first_layer, arch, "layer[0].mlp")?;

        // ── RMS norm weights ──────────────────────────────────────────
        let rms_attn_chains: &[&[&str]] = &[&["input_layernorm"], &["attention_norm"], &["ln_1"]];
        let rms_ffn_chains: &[&[&str]] = &[&["post_attention_layernorm"], &["ffn_norm"], &["ln_2"]];

        let rms_attn_weight = extract_direct_weight(&first_layer, rms_attn_chains, "rms_attn")?;
        let rms_ffn_weight = extract_direct_weight(&first_layer, rms_ffn_chains, "rms_ffn")
            .unwrap_or_else(|_| rms_attn_weight.clone());

        // ── LM head ───────────────────────────────────────────────────
        let lm_head = model
            .getattr("lm_head")
            .ok()
            .and_then(|h| h.getattr("weight").ok())
            .and_then(|w| to_vec_f32(&w).ok())
            .unwrap_or_else(|| embedding.clone());

        // ── num_kv_heads ───────────────────────────────────────────
        let num_kv_heads: usize = config
            .getattr("num_key_value_heads")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(num_heads);

        // ── rms_final_weight ───────────────────────────────────────
        let rms_final_weight = vec![1.0_f32; hidden_dim]; // placeholder

        // ── layers: collect first layer into Vec<LayerWeights> ─────
        let first_layer_weights = LayerWeights {
            q_proj: q_proj.clone(),
            k_proj: k_proj.clone(),
            v_proj: v_proj.clone(),
            attn_proj: attn_proj.clone(),
            rms_attn_weight: rms_attn_weight.clone(),
            rms_ffn_weight: rms_ffn_weight.clone(),
            gate_proj: gate_proj.clone(),
            up_proj: up_proj.clone(),
            down_proj: down_proj.clone(),
        };

        Ok(ModelWeights {
            embedding,
            q_proj,
            k_proj,
            v_proj,
            attn_proj,
            lm_head,
            rms_attn_weight,
            rms_ffn_weight,
            hidden_dim,
            num_heads,
            head_dim,
            num_layers,
            vocab_size,
            intermediate_dim,
            layers: vec![first_layer_weights],
            num_kv_heads,
            rms_final_weight,
        })
    })
}

/// Load all layer weights separately (for multi-layer GPU runner).
///
/// Unlike `load_weights` which extracts scalar fields from the first layer,
/// this function iterates every transformer layer and returns per-layer
/// weight bundles in `ModelWeights::layers`.  All other fields (embedding,
/// lm_head, config metadata) are extracted identically.
pub fn load_weights_all_layers(model_id: &str) -> Result<ModelWeights, ZealotError> {
    Python::with_gil(|py| -> Result<ModelWeights, ZealotError> {
        // ── Setup site-packages if ZEALOT_SITE_PACKAGES is set ──────────
        if let Ok(sp) = std::env::var("ZEALOT_SITE_PACKAGES") {
            let sys = py.import("sys").map_err(py_err("import sys"))?;
            sys.getattr("path")
                .map_err(py_err("sys.path"))?
                .call_method1("insert", (0, sp))
                .map_err(py_err("sys.path.insert"))?;
        }

        let _torch = py
            .import("torch")
            .map_err(py_err("import torch (venv? set ZEALOT_SITE_PACKAGES)"))?;
        let transformers = py
            .import("transformers")
            .map_err(py_err("import transformers"))?;

        let model = transformers
            .getattr("AutoModelForCausalLM")
            .map_err(py_err("AutoModelForCausalLM"))?
            .call_method1("from_pretrained", (model_id,))
            .map_err(py_err("model from_pretrained"))?;

        // ── Config ────────────────────────────────────────────────────
        let config = model.getattr("config").map_err(py_err("model.config"))?;
        let hidden_dim: usize = config
            .getattr("hidden_size")
            .map_err(py_err("hidden_size"))?
            .extract()
            .map_err(py_err("hidden_size"))?;
        let num_heads: usize = config
            .getattr("num_attention_heads")
            .map_err(py_err("num_attention_heads"))?
            .extract()
            .map_err(py_err("num_attention_heads"))?;
        let head_dim: usize = hidden_dim / num_heads;
        let num_layers: usize = config
            .getattr("num_hidden_layers")
            .map_err(py_err("num_hidden_layers"))?
            .extract()
            .map_err(py_err("num_hidden_layers"))?;
        let num_kv_heads: usize = config
            .getattr("num_key_value_heads")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(num_heads);
        let intermediate_dim: usize = config
            .getattr("intermediate_size")
            .map_err(py_err("intermediate_size"))?
            .extract()
            .map_err(py_err("intermediate_size"))?;

        // ── Embedding ─────────────────────────────────────────────────
        let embedding = extract_direct_weight(
            &model,
            &[&["transformer", "wte"], &["model", "embed_tokens"]],
            "embed_tokens",
        )?;
        let vocab_size = embedding.len() / hidden_dim;

        // ── All layers ────────────────────────────────────────────────
        let container = get_attr_chain(
            &model,
            &[&["transformer", "h"], &["model", "layers"]],
            "layers",
        )?;

        let attn_chains: &[&[&str]] = &[&["attn"], &["self_attn"], &["attention"]];
        let rms_attn_chains: &[&[&str]] = &[&["input_layernorm"], &["attention_norm"], &["ln_1"]];
        let rms_ffn_chains: &[&[&str]] = &[&["post_attention_layernorm"], &["ffn_norm"], &["ln_2"]];

        // 架构探测（M5-⑫）：LLaMA 系 / GPT-2 命名适配
        let arch = probe_arch(&config)?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = container
                .get_item(i)
                .map_err(py_err(&format!("layers[{i}]")))?;

            let (q_proj, k_proj, v_proj, attn_proj) =
                extract_attn_weights(&layer, attn_chains, arch, &format!("layer[{i}].attn"))?;
            let rms_attn_weight =
                extract_direct_weight(&layer, rms_attn_chains, &format!("layer[{i}].rms_attn"))?;
            let rms_ffn_weight =
                extract_direct_weight(&layer, rms_ffn_chains, &format!("layer[{i}].rms_ffn"))
                    .unwrap_or_else(|_| rms_attn_weight.clone());
            let (gate_proj, up_proj, down_proj) =
                extract_mlp_weights(&layer, arch, &format!("layer[{i}].mlp"))?;

            layers.push(LayerWeights {
                q_proj,
                k_proj,
                v_proj,
                attn_proj,
                rms_attn_weight,
                rms_ffn_weight,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        // ── rms_final_weight ───────────────────────────────────────────
        let rms_final_weight = extract_direct_weight(
            &model,
            &[&["norm"], &["model", "norm"], &["transformer", "ln_f"]],
            "rms_final",
        )
        .unwrap_or_else(|_| vec![1.0_f32; hidden_dim]);

        // ── LM head ───────────────────────────────────────────────────
        let lm_head = model
            .getattr("lm_head")
            .ok()
            .and_then(|h| h.getattr("weight").ok())
            .and_then(|w| to_vec_f32(&w).ok())
            .unwrap_or_else(|| embedding.clone());

        // ── Scalar fields for backward compat (use first layer) ───────
        let first = &layers[0];

        Ok(ModelWeights {
            embedding,
            q_proj: first.q_proj.clone(),
            k_proj: first.k_proj.clone(),
            v_proj: first.v_proj.clone(),
            attn_proj: first.attn_proj.clone(),
            lm_head,
            rms_attn_weight: first.rms_attn_weight.clone(),
            rms_ffn_weight: first.rms_ffn_weight.clone(),
            hidden_dim,
            num_heads,
            head_dim,
            num_layers,
            vocab_size,
            intermediate_dim,
            layers,
            num_kv_heads,
            rms_final_weight,
        })
    })
}

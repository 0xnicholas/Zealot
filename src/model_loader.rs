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
pub fn get_first_layer<'py>(
    model: &Bound<'py, PyAny>,
) -> Result<Bound<'py, PyAny>, ZealotError> {
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

        // ── Embedding ─────────────────────────────────────────────────
        let embedding = extract_direct_weight(
            &model,
            &[&["transformer", "wte"], &["model", "embed_tokens"]],
            "embed_tokens",
        )?;
        let vocab_size = embedding.len() / hidden_dim;

        // ── Attention weights (Q/K/V/O projection) ────────────────────
        let first_layer = get_first_layer(&model)?;
        let attn_chains: &[&[&str]] = &[&["attn"], &["self_attn"], &["attention"]];

        let q_proj = extract_weight(&first_layer, attn_chains, "q_proj", "q_proj")?;
        let k_proj = extract_weight(&first_layer, attn_chains, "k_proj", "k_proj")?;
        let v_proj = extract_weight(&first_layer, attn_chains, "v_proj", "v_proj")?;
        let attn_proj = extract_weight(&first_layer, attn_chains, "o_proj", "o_proj")?;

        // ── RMS norm weights ──────────────────────────────────────────
        let rms_attn_chains: &[&[&str]] =
            &[&["input_layernorm"], &["attention_norm"], &["ln_1"]];
        let rms_ffn_chains: &[&[&str]] =
            &[&["post_attention_layernorm"], &["ffn_norm"], &["ln_2"]];

        let rms_attn_weight =
            extract_direct_weight(&first_layer, rms_attn_chains, "rms_attn")?;
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
        let rms_attn_chains: &[&[&str]] =
            &[&["input_layernorm"], &["attention_norm"], &["ln_1"]];
        let rms_ffn_chains: &[&[&str]] =
            &[&["post_attention_layernorm"], &["ffn_norm"], &["ln_2"]];

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = container
                .get_item(i)
                .map_err(py_err(&format!("layers[{i}]")))?;

            let q_proj = extract_weight(&layer, attn_chains, "q_proj", &format!("layer[{i}].q_proj"))?;
            let k_proj = extract_weight(&layer, attn_chains, "k_proj", &format!("layer[{i}].k_proj"))?;
            let v_proj = extract_weight(&layer, attn_chains, "v_proj", &format!("layer[{i}].v_proj"))?;
            let attn_proj =
                extract_weight(&layer, attn_chains, "o_proj", &format!("layer[{i}].o_proj"))?;
            let rms_attn_weight = extract_direct_weight(
                &layer,
                rms_attn_chains,
                &format!("layer[{i}].rms_attn"),
            )?;
            let rms_ffn_weight = extract_direct_weight(&layer, rms_ffn_chains, &format!("layer[{i}].rms_ffn"))
                .unwrap_or_else(|_| rms_attn_weight.clone());

            layers.push(LayerWeights {
                q_proj,
                k_proj,
                v_proj,
                attn_proj,
                rms_attn_weight,
                rms_ffn_weight,
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
            layers,
            num_kv_heads,
            rms_final_weight,
        })
    })
}

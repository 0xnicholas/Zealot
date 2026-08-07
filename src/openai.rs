//! OpenAI-compatible request/response types for the HTTP inference endpoint.
//! Covers the ChatGPT API surface used by `/v1/chat/completions`.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::constrained_decode::matcher::ConstrainedGrammar;
use crate::constrained_decode::schema::JsonSchemaCompiler;

// ── Incoming request ────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// `response_format.type`（M5-④）。
#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatType {
    Text,
    JsonObject,
    JsonSchema,
}

/// `response_format`（M5-④）：json_schema 经 CD 编译链路接入 mask 钩子约束解码。
#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    pub r#type: ResponseFormatType,
    #[serde(default)]
    pub json_schema: Option<JsonSchemaResponseFormat>,
}

/// `response_format.json_schema`（OpenAI JSON Schema 响应格式）。
#[derive(Debug, Deserialize)]
pub struct JsonSchemaResponseFormat {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// 未提供时由调用方报 400（OpenAI 语义：json_schema 必须携带 schema）。
    pub schema: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

/// `tools[].function`（M5-④）。
#[derive(Debug, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// `tools[]`（M5-④）。
#[derive(Debug, Deserialize)]
pub struct Tool {
    /// 固定 "function"（OpenAI 语义）。
    pub r#type: String,
    pub function: ToolFunction,
}

/// `tool_choice`："auto"/"none"/"required" 或 {type:function, function:{name}}。
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Plain(String),
    Function {
        r#type: Option<String>,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    // ── M5-④：结构化输出 ──
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
}

fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}

// ── M5-④：strict 校验与约束 grammar 解析 ─────────────

/// OpenAI strict 模式检查（M5-④）：
/// 1. 根必须是 object；
/// 2. 每个 object 的所有属性必须列入 required；
/// 3. 每个 object 必须显式 `additionalProperties: false`。
///
/// 递归覆盖嵌套 object（properties / array items / anyOf·oneOf 分支）；
/// 失败返回带路径的可读错误（调用方转 400）。
pub fn validate_strict_schema(schema: &serde_json::Value) -> Result<(), String> {
    fn check_object(
        obj: &serde_json::Map<String, serde_json::Value>,
        path: &str,
    ) -> Result<(), String> {
        let required: Vec<&str> = obj
            .get("required")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        if obj.get("additionalProperties").and_then(|v| v.as_bool()) != Some(false) {
            return Err(format!(
                "{path}: strict mode requires \"additionalProperties\": false"
            ));
        }
        if let Some(props) = obj.get("properties").and_then(|v| v.as_object()) {
            for name in props.keys() {
                if !required.contains(&name.as_str()) {
                    return Err(format!(
                        "{path}: strict mode requires property \"{name}\" to be in \"required\""
                    ));
                }
            }
        }
        Ok(())
    }

    fn check(v: &serde_json::Value, path: &str) -> Result<(), String> {
        let obj = match v.as_object() {
            Some(o) => o,
            // 非 object 节点（string/number/... 字面量 schema）无需额外规则
            None => return Ok(()),
        };
        let is_object = obj.get("type").and_then(|t| t.as_str()) == Some("object")
            || obj.contains_key("properties");
        if is_object {
            check_object(obj, path)?;
        }
        // 递归：properties / items / anyOf·oneOf 分支
        if let Some(props) = obj.get("properties").and_then(|v| v.as_object()) {
            for (name, sub) in props {
                check(sub, &format!("{path}.\"{name}\""))?;
            }
        }
        if let Some(items) = obj.get("items") {
            check(items, &format!("{path}[*]"))?;
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(branches) = obj.get(key).and_then(|v| v.as_array()) {
                for (i, branch) in branches.iter().enumerate() {
                    check(branch, &format!("{path}.{key}[{i}]"))?;
                }
            }
        }
        Ok(())
    }

    // 根必须是 object（OpenAI strict 语义）
    let root = schema.as_object().ok_or_else(|| {
        "strict mode requires the root schema to be an object with \"type\": \"object\"".to_string()
    })?;
    let root_is_object = root.get("type").and_then(|t| t.as_str()) == Some("object")
        || root.contains_key("properties");
    if !root_is_object {
        return Err(
            "strict mode requires the root schema to be an object with \"type\": \"object\""
                .to_string(),
        );
    }
    check(schema, "root")
}

/// 工具调用信封（M5-④）：`{"name": <工具名>, "arguments": <参数 schema>}`。
fn tool_envelope(tool: &Tool) -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "name": { "enum": [tool.function.name.clone()] },
            "arguments": tool
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| json!({ "type": "object" })),
        },
        "required": ["name", "arguments"],
        "additionalProperties": false,
    })
}

/// 根据 `response_format` / `tools` / `tool_choice` 解析约束 grammar。
///
/// 语义（M5-④）：
/// - `response_format: json_schema` → schema 经 CD 编译链路（strict 时先校验）；
/// - `json_object` → 不约束（additionalProperties 通配键留 Phase 3，见 matcher.rs 注释）；
/// - 强制 `tool_choice`（具名函数或 "required"）→ 工具信封 grammar（参数 schema 同样 strict 校验）；
/// - `auto`/`none`/未提供 → 模型自由决定，不约束；
/// - 强制工具 + json_schema 同时出现 → 错误（OpenAI 不允许组合）。
///
/// 编译/校验失败返回 `Err(可读消息)`，调用方转 400。
pub fn resolve_grammar(
    compiler: &JsonSchemaCompiler,
    response_format: Option<&ResponseFormat>,
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
) -> Result<Option<ConstrainedGrammar>, String> {
    // json_schema 与强制工具调用互斥（OpenAI 语义）——先于分支处理
    let forced_tool = matches!(tool_choice, Some(ToolChoice::Function { .. }))
        || matches!(tool_choice, Some(ToolChoice::Plain(s)) if s == "required");
    if json_schema_format_present(response_format) && forced_tool {
        return Err("cannot combine response_format with forced tool_choice".to_string());
    }

    // json_schema：strict 先校验再编译
    if let Some(rf) = response_format {
        match rf.r#type {
            ResponseFormatType::JsonSchema => {
                let spec = rf.json_schema.as_ref().ok_or_else(|| {
                    "response_format.json_schema is required when type is \"json_schema\""
                        .to_string()
                })?;
                let schema = spec
                    .schema
                    .as_ref()
                    .ok_or_else(|| "response_format.json_schema.schema is required".to_string())?;
                if spec.strict.unwrap_or(false) {
                    validate_strict_schema(schema)
                        .map_err(|e| format!("strict schema check failed: {e}"))?;
                }
                return compiler
                    .compile_value(schema)
                    .map(Some)
                    .map_err(|e| format!("schema compile failed: {e}"));
            }
            ResponseFormatType::JsonObject | ResponseFormatType::Text => {}
        }
    }

    // tools / tool_choice
    let Some(tools) = tools else {
        return Ok(None);
    };
    if tools.is_empty() {
        return Ok(None);
    }
    match tool_choice {
        None => Ok(None),
        Some(ToolChoice::Plain(s)) if s == "auto" || s == "none" => Ok(None),
        Some(ToolChoice::Plain(s)) if s == "required" => {
            if tools.len() == 1 {
                let envelope = tool_envelope(&tools[0]);
                strict_check_tool(&tools[0])?;
                return compiler
                    .compile_value(&envelope)
                    .map(Some)
                    .map_err(|e| format!("tool schema compile failed: {e}"));
            }
            // 多工具：anyOf 信封并集
            let mut branches: Vec<serde_json::Value> = Vec::with_capacity(tools.len());
            for t in tools {
                strict_check_tool(t)?;
                branches.push(tool_envelope(t));
            }
            let union = json!({ "anyOf": branches });
            compiler
                .compile_value(&union)
                .map(Some)
                .map_err(|e| format!("tool schema compile failed: {e}"))
        }
        Some(ToolChoice::Plain(other)) => Err(format!(
            "invalid tool_choice \"{other}\" (expected \"auto\", \"none\" or \"required\")"
        )),
        Some(ToolChoice::Function { function, .. }) => {
            let tool = tools
                .iter()
                .find(|t| t.function.name == function.name)
                .ok_or_else(|| format!("tool_choice names unknown tool \"{}\"", function.name))?;
            strict_check_tool(tool)?;
            let envelope = tool_envelope(tool);
            compiler
                .compile_value(&envelope)
                .map(Some)
                .map_err(|e| format!("tool schema compile failed: {e}"))
        }
    }
}

/// 强制调用工具时其参数 schema 必须满足 strict 规则（OpenAI 语义）。
fn strict_check_tool(tool: &Tool) -> Result<(), String> {
    match tool.function.parameters.as_ref() {
        Some(p) => validate_strict_schema(p)
            .map_err(|e| format!("tool \"{}\" parameters: {e}", tool.function.name)),
        None => Err(format!(
            "tool \"{}\" requires parameters for forced calls",
            tool.function.name
        )),
    }
}

/// response_format 是否请求了 json_schema（与强制工具调用互斥）。
fn json_schema_format_present(response_format: Option<&ResponseFormat>) -> bool {
    response_format
        .map(|r| r.r#type == ResponseFormatType::JsonSchema)
        .unwrap_or(false)
}

// ── SSE streaming chunks ───────────────────────────

#[derive(Debug, Serialize)]
pub struct SseChunk {
    pub id: String,
    #[serde(rename = "object")]
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<SseChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct SseChoice {
    pub index: u32,
    pub delta: SseDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SseDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Non-streaming response ─────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    #[serde(rename = "object")]
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ResponseChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ResponseChoice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

// ── SSE helpers ────────────────────────────────────

pub fn sse_role(request_id: &str, model: &str, ts: u64) -> SseChunk {
    SseChunk {
        id: request_id.into(),
        object: "chat.completion.chunk".into(),
        created: ts,
        model: model.into(),
        choices: vec![SseChoice {
            index: 0,
            delta: SseDelta {
                role: Some("assistant".into()),
                content: None,
            },
            finish_reason: None,
        }],
        usage: None,
    }
}

pub fn sse_content(request_id: &str, model: &str, ts: u64, text: &str) -> SseChunk {
    SseChunk {
        id: request_id.into(),
        object: "chat.completion.chunk".into(),
        created: ts,
        model: model.into(),
        choices: vec![SseChoice {
            index: 0,
            delta: SseDelta {
                role: None,
                content: Some(text.into()),
            },
            finish_reason: None,
        }],
        usage: None,
    }
}

pub fn sse_finish(
    request_id: &str,
    model: &str,
    ts: u64,
    reason: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> SseChunk {
    SseChunk {
        id: request_id.into(),
        object: "chat.completion.chunk".into(),
        created: ts,
        model: model.into(),
        choices: vec![SseChoice {
            index: 0,
            delta: SseDelta {
                role: None,
                content: None,
            },
            finish_reason: Some(reason.into()),
        }],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }),
    }
}

pub fn sse_done() -> &'static str {
    "data: [DONE]\n\n"
}

pub fn serialize_sse_chunk(chunk: &SseChunk) -> String {
    let json = serde_json::to_string(chunk).unwrap_or_else(|_| "{}".into());
    format!("data: {json}\n\n")
}

pub fn non_streaming_response(
    request_id: &str,
    model: &str,
    ts: u64,
    content: String,
    finish_reason: String,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> String {
    let resp = ChatCompletionResponse {
        id: request_id.into(),
        object: "chat.completion".into(),
        created: ts,
        model: model.into(),
        choices: vec![ResponseChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };
    serde_json::to_string(&resp).unwrap_or_else(|_| "{}".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrained_decode::schema::JsonSchemaCompiler;

    fn compiler() -> JsonSchemaCompiler {
        JsonSchemaCompiler::new(16, 100_000, 1_000)
    }

    fn j(v: &str) -> serde_json::Value {
        serde_json::from_str(v).unwrap()
    }

    // ── strict 校验 ──

    #[test]
    fn strict_accepts_conforming_schema() {
        let schema = j(
            r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"object","properties":{"c":{"type":"string"}},"required":["c"],"additionalProperties":false}},"required":["a","b"],"additionalProperties":false}"#,
        );
        assert!(validate_strict_schema(&schema).is_ok());
    }

    #[test]
    fn strict_rejects_non_object_root() {
        let schema = j(r#"{"type":"string"}"#);
        let err = validate_strict_schema(&schema).unwrap_err();
        assert!(err.contains("root"), "{err}");
    }

    #[test]
    fn strict_rejects_missing_required() {
        let schema = j(
            r#"{"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":false}"#,
        );
        let err = validate_strict_schema(&schema).unwrap_err();
        assert!(err.contains("required") && err.contains("a"), "{err}");
    }

    #[test]
    fn strict_rejects_permissive_additional_properties() {
        let schema =
            j(r#"{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"]}"#);
        let err = validate_strict_schema(&schema).unwrap_err();
        assert!(err.contains("additionalProperties"), "{err}");
    }

    #[test]
    fn strict_rejects_nested_violation() {
        let schema = j(
            r#"{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"boolean"},"c":{"type":"boolean"}},"required":["b"],"additionalProperties":false}},"required":["a"],"additionalProperties":false}"#,
        );
        let err = validate_strict_schema(&schema).unwrap_err();
        assert!(err.contains("root.\"a\""), "路径应指向嵌套属性: {err}");
    }

    #[test]
    fn strict_rejects_array_items_violation() {
        let schema = j(
            r#"{"type":"object","properties":{"arr":{"type":"array","items":{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"]}}},"required":["arr"],"additionalProperties":false}"#,
        );
        let err = validate_strict_schema(&schema).unwrap_err();
        assert!(err.contains("additionalProperties"), "{err}");
    }

    // ── grammar 解析 ──

    fn req(text: &str) -> ChatCompletionRequest {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn parse_response_format_json_schema() {
        let r = req(
            r#"{"model":"m","messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object"}}}}"#,
        );
        let rf = r.response_format.unwrap();
        assert_eq!(rf.r#type, ResponseFormatType::JsonSchema);
        assert!(rf.json_schema.unwrap().strict.unwrap());
    }

    #[test]
    fn parse_tool_choice_function() {
        let r = req(
            r#"{"model":"m","messages":[],"tool_choice":{"type":"function","function":{"name":"f1"}}}"#,
        );
        match r.tool_choice.unwrap() {
            ToolChoice::Function { function, .. } => assert_eq!(function.name, "f1"),
            _ => panic!("expected function tool_choice"),
        }
    }

    #[test]
    fn resolve_json_schema_compiles_grammar() {
        let r = req(
            r#"{"model":"m","messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"],"additionalProperties":false}}}}"#,
        );
        let g = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect("resolve")
        .expect("grammar");
        assert!(g.state_count() > 0);
    }

    #[test]
    fn resolve_strict_failure_propagates() {
        let r = req(
            r#"{"model":"m","messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object","properties":{"a":{"type":"boolean"}}}}}}"#,
        );
        let err = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect_err("strict violation must fail");
        assert!(err.contains("strict"), "{err}");
    }

    #[test]
    fn resolve_json_schema_missing_spec_errors() {
        let r = req(r#"{"model":"m","messages":[],"response_format":{"type":"json_schema"}}"#);
        let err = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect_err("missing json_schema must fail");
        assert!(err.contains("json_schema"), "{err}");
    }

    #[test]
    fn resolve_tool_choice_function_envelope() {
        let r = req(
            r#"{"model":"m","messages":[],"tools":[{"type":"function","function":{"name":"f1","parameters":{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"],"additionalProperties":false}}}],"tool_choice":{"type":"function","function":{"name":"f1"}}}"#,
        );
        let g = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect("resolve")
        .expect("grammar");
        assert!(g.state_count() > 0);
    }

    #[test]
    fn resolve_tool_choice_required_multi_tool_union() {
        let r = req(
            r#"{"model":"m","messages":[],"tools":[{"type":"function","function":{"name":"f1","parameters":{"type":"object","additionalProperties":false}}},{"type":"function","function":{"name":"f2","parameters":{"type":"object","additionalProperties":false}}}],"tool_choice":"required"}"#,
        );
        let g = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect("resolve")
        .expect("grammar");
        assert!(g.state_count() > 0);
    }

    #[test]
    fn resolve_tool_choice_auto_is_unconstrained() {
        let r = req(
            r#"{"model":"m","messages":[],"tools":[{"type":"function","function":{"name":"f1","parameters":{"type":"object","additionalProperties":false}}}],"tool_choice":"auto"}"#,
        );
        let g = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect("resolve");
        assert!(g.is_none(), "auto 不约束");
    }

    #[test]
    fn resolve_tool_choice_unknown_tool_errors() {
        let r = req(
            r#"{"model":"m","messages":[],"tools":[{"type":"function","function":{"name":"f1","parameters":{"type":"object","additionalProperties":false}}}],"tool_choice":{"type":"function","function":{"name":"ghost"}}}"#,
        );
        let err = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect_err("unknown tool must fail");
        assert!(err.contains("unknown tool"), "{err}");
    }

    #[test]
    fn resolve_tool_requires_strict_params() {
        let r = req(
            r#"{"model":"m","messages":[],"tools":[{"type":"function","function":{"name":"f1","parameters":{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"]}}}],"tool_choice":{"type":"function","function":{"name":"f1"}}}"#,
        );
        let err = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect_err("permissive params must fail");
        assert!(err.contains("parameters"), "{err}");
    }

    #[test]
    fn resolve_json_schema_plus_forced_tool_conflict() {
        let r = req(
            r#"{"model":"m","messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}},"tools":[{"type":"function","function":{"name":"f1","parameters":{"type":"object","additionalProperties":false}}}],"tool_choice":{"type":"function","function":{"name":"f1"}}}"#,
        );
        let err = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect_err("json_schema + forced tool must conflict");
        assert!(err.contains("cannot combine"), "{err}");
    }

    #[test]
    fn resolve_json_object_is_unconstrained() {
        let r = req(r#"{"model":"m","messages":[],"response_format":{"type":"json_object"}}"#);
        let g = resolve_grammar(
            &compiler(),
            r.response_format.as_ref(),
            r.tools.as_deref(),
            r.tool_choice.as_ref(),
        )
        .expect("resolve");
        assert!(g.is_none(), "json_object 不约束（通配键 Phase 3）");
    }
}

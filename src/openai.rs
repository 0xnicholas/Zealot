//! OpenAI-compatible request/response types for the HTTP inference endpoint.
//! Covers the ChatGPT API surface used by `/v1/chat/completions`.

use serde::{Deserialize, Serialize};

// ── Incoming request ────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
}

fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
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

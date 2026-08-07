//! End-to-end: zealot-backend 的 gRPC plumbing（不含真实模型加载——
//! 真实推理见 cpu_infer_e2e.rs）。

mod common;

use common::runtime_v1::{
    inference_runtime_client::InferenceRuntimeClient, BatchRequest, HealthCheckRequest,
    ListModelsRequest, LoadModelRequest, Quantization, Status as HealthStatus, StreamControl,
};
use common::{spawn_backend, stream_start};

#[tokio::test]
async fn health_and_error_paths_without_model() {
    let (_guard, addr) = spawn_backend().await;
    let mut client = InferenceRuntimeClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");

    // 空载：HEALTHY
    let health = client
        .health_check(HealthCheckRequest {})
        .await
        .expect("health_check")
        .into_inner();
    assert_eq!(health.status(), HealthStatus::Healthy);

    // 无模型时 Infer → FAILED_PRECONDITION
    let err = client
        .infer(tokio_stream::iter(vec![stream_start("req-1", "ghost")]))
        .await
        .expect_err("infer without model must fail");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);

    // 首帧非 STREAM_START → INVALID_ARGUMENT
    let mut bad = stream_start("req-2", "ghost");
    bad.control = StreamControl::Cancel as i32;
    let err = client
        .infer(tokio_stream::iter(vec![bad]))
        .await
        .expect_err("non-STREAM_START first frame must fail");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    // 模型列表为空
    let list = client
        .list_models(ListModelsRequest {})
        .await
        .expect("list_models")
        .into_inner();
    assert!(list.models.is_empty());

    // Batch 未实现
    let err = client
        .submit_batch(BatchRequest {
            model_id: "m".into(),
            items: vec![],
        })
        .await
        .expect_err("SubmitBatch is unimplemented");
    assert_eq!(err.code(), tonic::Code::Unimplemented);
}

#[tokio::test]
async fn load_failure_resets_slot() {
    let (_guard, addr) = spawn_backend().await;
    let mut client = InferenceRuntimeClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");

    // 不存在的本地模型路径 → INTERNAL（transformers 按本地目录处理，快速失败，不走网络）
    let err = client
        .load_model(LoadModelRequest {
            model_id: "/nonexistent/zealot-model-xyz".into(),
            quantization: Quantization::Unspecified as i32,
            gpu_count: 0,
            gpu_type: String::new(),
            labels: Default::default(),
        })
        .await
        .expect_err("loading a nonexistent model must fail");
    assert_eq!(err.code(), tonic::Code::Internal);

    // slot 复位：健康检查恢复 HEALTHY，列表仍为空，且可以再次 LoadModel
    let health = client
        .health_check(HealthCheckRequest {})
        .await
        .expect("health_check")
        .into_inner();
    assert_eq!(health.status(), HealthStatus::Healthy);
    let list = client
        .list_models(ListModelsRequest {})
        .await
        .expect("list_models")
        .into_inner();
    assert!(list.models.is_empty());
}

// ── M5-④：HTTP 结构化输出端到端（dev mock runner，CI 可跑）──────────────

use std::process::Command;

/// curl POST /v1/chat/completions，返回 (HTTP 状态码, 响应体)。
fn http_chat(http_addr: &str, body: &str) -> (String, String) {
    let output = Command::new("curl")
        .args([
            "-s",
            "-w",
            "\n%{http_code}",
            "-X",
            "POST",
            &format!("http://{http_addr}/v1/chat/completions"),
            "-H",
            "Content-Type: application/json",
            "-d",
            body,
        ])
        .output()
        .expect("curl failed");
    assert!(output.status.success(), "curl exit: {}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let (resp, code) = stdout
        .rsplit_once('\n')
        .map(|(r, c)| (r.to_string(), c.trim().to_string()))
        .unwrap_or((stdout.to_string(), "000".into()));
    (code, resp)
}

/// 加载 mock:tiny 并返回 (guard, http_addr)。
async fn load_mock() -> (common::ChildGuard, String) {
    let (guard, addr, http_addr) = common::spawn_backend_full().await;
    let mut client = InferenceRuntimeClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");
    let resp = client
        .load_model(LoadModelRequest {
            model_id: "mock:tiny".into(),
            quantization: Quantization::Unspecified as i32,
            gpu_count: 0,
            gpu_type: String::new(),
            labels: Default::default(),
        })
        .await
        .expect("load mock:tiny");
    assert!(resp.into_inner().ready);
    (guard, http_addr)
}

#[tokio::test]
async fn http_response_format_json_schema_constrained_output() {
    let (_guard, http_addr) = load_mock().await;
    let (code, body) = http_chat(
        &http_addr,
        r#"{
            "model": "mock:tiny",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 32,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "out",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "boolean"}},
                        "required": ["name"],
                        "additionalProperties": false
                    }
                }
            }
        }"#,
    );
    assert_eq!(code, "200", "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("parse response");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    // mock logits 偏好最小合法 id → 确定性输出；严格 schema 约束 + 收尾 eos 强制放行
    assert_eq!(content, r#"{"name":true}"#);
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn http_response_format_strict_violation_returns_400() {
    let (_guard, http_addr) = load_mock().await;
    // required 缺失 + additionalProperties 缺失 → strict 检查失败 → 400
    let (code, body) = http_chat(
        &http_addr,
        r#"{
            "model": "mock:tiny",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "bad",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "boolean"}}
                    }
                }
            }
        }"#,
    );
    assert_eq!(code, "400");
    let v: serde_json::Value = serde_json::from_str(&body).expect("parse error body");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("strict"));
}

#[tokio::test]
async fn http_response_format_invalid_schema_returns_400() {
    let (_guard, http_addr) = load_mock().await;
    // json_schema 缺 schema 字段 → 400
    let (code, body) = http_chat(
        &http_addr,
        r#"{
            "model": "mock:tiny",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "response_format": {"type": "json_schema", "json_schema": {"name": "nope"}}
        }"#,
    );
    assert_eq!(code, "400", "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("parse error body");
    assert!(v["error"]["message"].as_str().unwrap().contains("schema"));
}

#[tokio::test]
async fn http_tools_forced_call_constrained_output() {
    let (_guard, http_addr) = load_mock().await;
    let (code, body) = http_chat(
        &http_addr,
        r#"{
            "model": "mock:tiny",
            "messages": [{"role": "user", "content": "weather?"}],
            "max_tokens": 64,
            "temperature": 0,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "boolean"}},
                        "required": ["name"],
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }"#,
    );
    assert_eq!(code, "200", "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("parse response");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    // 信封 grammar：{"name":"get_weather","arguments":{...}}
    let parsed: serde_json::Value = serde_json::from_str(content).expect("content is JSON");
    assert_eq!(parsed["name"], "get_weather");
    assert!(parsed["arguments"].is_object());
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn http_tools_constrained_integer_param() {
    // M5-⑤：工具参数 integer 范围约束真实执行（mock 恒定 argmax：
    // 最小 id 胜出 → age 取 1，"0"(7) 与 "1"(6) 中 1 更小）。
    let (_guard, http_addr) = load_mock().await;
    let (code, body) = http_chat(
        &http_addr,
        r#"{
            "model": "mock:tiny",
            "messages": [{"role": "user", "content": "age?"}],
            "max_tokens": 64,
            "temperature": 0,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 1}},
                        "required": ["age"],
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }"#,
    );
    assert_eq!(code, "200", "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("parse response");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(content).expect("content is JSON");
    assert_eq!(parsed["name"], "get_weather");
    // 区间 [0,1]：mock 恒选最小 id → "1"(id 6) < "0"(id 7)
    assert_eq!(parsed["arguments"]["age"], 1);
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn http_tools_conflicting_range_returns_400() {
    // M5-⑤：工具参数空区间 → 400（不产出死 grammar）
    let (_guard, http_addr) = load_mock().await;
    let (code, body) = http_chat(
        &http_addr,
        r#"{
            "model": "mock:tiny",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 64,
            "temperature": 0,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer", "minimum": 10, "maximum": 5}},
                        "required": ["x"],
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": {"type": "function", "function": {"name": "f"}}
        }"#,
    );
    assert_eq!(code, "400", "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("parse response");
    assert!(v["error"]["message"].as_str().unwrap().contains("空区间"));
}

#[tokio::test]
async fn http_tool_choice_unknown_tool_returns_400() {
    let (_guard, http_addr) = load_mock().await;
    let (code, body) = http_chat(
        &http_addr,
        r#"{
            "model": "mock:tiny",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "tools": [{
                "type": "function",
                "function": {"name": "a", "parameters": {"type": "object", "additionalProperties": false}}
            }],
            "tool_choice": {"type": "function", "function": {"name": "ghost"}}
        }"#,
    );
    assert_eq!(code, "400");
    let v: serde_json::Value = serde_json::from_str(&body).expect("parse error body");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unknown tool"));
}

#[tokio::test]
async fn http_json_object_without_schema_is_unconstrained() {
    let (_guard, http_addr) = load_mock().await;
    // json_object（无 schema）→ 不约束（通配键 Phase 3）→ mock 恒定 argmax `get_weather`
    let (code, body) = http_chat(
        &http_addr,
        r#"{
            "model": "mock:tiny",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "temperature": 0,
            "response_format": {"type": "json_object"}
        }"#,
    );
    assert_eq!(code, "200", "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("parse response");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content, "get_weatherget_weatherget_weatherget_weather");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

//! HTTP endpoint e2e test: verifies /v1/chat/completions works via HTTP.
//!
//! Relies on curl being available on PATH. Runs against an externally
//! started zealot-backend (see README for setup steps).
//!
//! Usage:
//!   ZEALOT_HTTP_ADDR=http://localhost:8081 cargo test --test http_e2e -- --ignored

use std::process::Command;

#[test]
#[ignore]
fn http_chat_non_streaming() {
    let addr = std::env::var("ZEALOT_HTTP_ADDR").unwrap_or_else(|_| "http://localhost:8081".into());

    let output = Command::new("curl")
        .args([
            "-s", "-X", "POST",
            &format!("{addr}/v1/chat/completions"),
            "-H", "Content-Type: application/json",
            "-d", r#"{"model":"/tmp/tiny-llama","messages":[{"role":"user","content":"Hello"}],"max_tokens":4,"stream":false}"#,
        ])
        .output()
        .expect("curl failed");

    assert!(output.status.success(), "curl exit: {}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let body: serde_json::Value = serde_json::from_str(&stdout).expect("parse JSON");
    assert_eq!(body["object"], "chat.completion");
    assert!(!body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .is_empty());
    eprintln!("non-streaming OK: {body}");
}

#[test]
#[ignore]
fn http_chat_streaming() {
    let addr = std::env::var("ZEALOT_HTTP_ADDR").unwrap_or_else(|_| "http://localhost:8081".into());

    let output = Command::new("curl")
        .args([
            "-s", "-X", "POST",
            &format!("{addr}/v1/chat/completions"),
            "-H", "Content-Type: application/json",
            "-d", r#"{"model":"/tmp/tiny-llama","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true}"#,
        ])
        .output()
        .expect("curl failed");

    assert!(output.status.success(), "curl exit: {}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("data: [DONE]"), "missing [DONE] terminator");
    assert!(
        stdout.contains("chat.completion.chunk"),
        "missing SSE chunks"
    );
    assert!(stdout.contains("finish_reason"), "missing finish_reason");
    eprintln!("streaming OK, {} bytes", stdout.len());
}

//! Shared helpers for zealot-backend integration tests.

#![allow(dead_code)]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

// 同 src/bin：clippy 1.98 起 result_large_err 命中 tonic 生成代码
#[allow(clippy::result_large_err)]
pub mod runtime_v1 {
    tonic::include_proto!("runtime.v1");
}

use runtime_v1::{ChatMessage, InferParams, InferRequest, StreamControl};

pub struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// 以端口 0 拉起 zealot-backend，返回 (guard, gRPC 地址, HTTP 地址)。
pub async fn spawn_backend_full() -> (ChildGuard, String, String) {
    let exe = env!("CARGO_BIN_EXE_zealot-backend");
    let mut child = Command::new(exe)
        .env("ZEALOT_GRPC_PORT", "0")
        .env("ZEALOT_HTTP_PORT", "0")
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn zealot-backend");
    let stdout = child.stdout.take().expect("stdout piped");
    let line = tokio::task::spawn_blocking(move || {
        BufReader::new(stdout)
            .lines()
            .next()
            .expect("backend prints its listen address")
            .expect("read listen address")
    })
    .await
    .expect("join stdout reader");
    // Format: "zealot-backend listening on <gRPC-addr> (gRPC) <HTTP-addr> (HTTP)"
    let rest = line
        .strip_prefix("zealot-backend listening on ")
        .expect("listen address line");
    let mut parts = rest.splitn(2, " (gRPC) ");
    let grpc_addr = parts.next().unwrap_or(rest).replace("0.0.0.0", "127.0.0.1");
    let http_addr = parts
        .next()
        .unwrap_or("")
        .strip_suffix(" (HTTP)")
        .unwrap_or("")
        .replace("0.0.0.0", "127.0.0.1");
    (ChildGuard(child), grpc_addr, http_addr)
}

/// 仅 gRPC 地址的旧版入口（大部分 backend_e2e 用）。
pub async fn spawn_backend() -> (ChildGuard, String) {
    let (guard, grpc, _http) = spawn_backend_full().await;
    (guard, grpc)
}

pub fn stream_start(request_id: &str, model: &str) -> InferRequest {
    InferRequest {
        request_id: request_id.into(),
        model: model.into(),
        messages: vec![],
        params: None,
        control: StreamControl::StreamStart as i32,
        scheduling_hint: None,
    }
}

pub fn stream_start_chat(
    request_id: &str,
    model: &str,
    content: &str,
    max_tokens: u32,
) -> InferRequest {
    InferRequest {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: content.into(),
        }],
        params: Some(InferParams {
            max_tokens,
            ..Default::default()
        }),
        ..stream_start(request_id, model)
    }
}

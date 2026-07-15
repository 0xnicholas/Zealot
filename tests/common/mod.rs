//! Shared helpers for zealot-backend integration tests.

#![allow(dead_code)]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

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

/// 以端口 0 拉起 zealot-backend，返回实际监听地址。
/// 子进程继承父进程环境（含 ZEALOT_SITE_PACKAGES）。
pub async fn spawn_backend() -> (ChildGuard, String) {
    let exe = env!("CARGO_BIN_EXE_zealot-backend");
    let mut child = Command::new(exe)
        .env("ZEALOT_GRPC_PORT", "0")
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
    let addr = line
        .strip_prefix("zealot-backend listening on ")
        .expect("listen address line")
        .replace("0.0.0.0", "127.0.0.1");
    (ChildGuard(child), addr)
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

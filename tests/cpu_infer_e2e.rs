//! CPU 端到端：真实模型推理（dev-mode PyModelRunner，PyTorch CPU）。
//!
//! 运行条件：
//!   ZEALOT_E2E_MODEL=<hf repo id>        如 hf-internal-testing/tiny-random-gpt2
//!   ZEALOT_SITE_PACKAGES=<venv>/lib/python3.12/site-packages
//! 未设置 ZEALOT_E2E_MODEL 时跳过（CI 无模型环境）。

mod common;

use std::time::Duration;

use common::runtime_v1::{
    inference_runtime_client::InferenceRuntimeClient, infer_response::Payload,
    ListModelsRequest, LoadModelRequest,
};
use common::{spawn_backend, stream_start_chat};
use tokio_stream::StreamExt;

#[tokio::test]
async fn cpu_end_to_end_infer() {
    let Ok(model) = std::env::var("ZEALOT_E2E_MODEL") else {
        eprintln!("skip: ZEALOT_E2E_MODEL not set");
        return;
    };

    let (_guard, addr) = spawn_backend().await;
    let mut client = InferenceRuntimeClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");

    // LoadModel：首次 HF 下载可能数分钟
    let load = tokio::time::timeout(
        Duration::from_secs(600),
        client.load_model(LoadModelRequest {
            model_id: model.clone(),
            ..Default::default()
        }),
    )
    .await
    .expect("load_model timed out")
    .expect("load_model")
    .into_inner();
    assert!(load.ready);

    let list = client
        .list_models(ListModelsRequest {})
        .await
        .expect("list_models")
        .into_inner();
    assert_eq!(list.models.len(), 1);
    assert_eq!(list.models[0].model_id, model);

    // Infer：真实生成（CPU prefill/decode 较慢，逐帧放宽超时）
    let req = stream_start_chat("req-cpu-1", &model, "Hello, who are you?", 16);
    let mut stream = client
        .infer(tokio_stream::iter(vec![req]))
        .await
        .expect("infer")
        .into_inner();

    let mut deltas = 0usize;
    let mut text = String::new();
    let mut usage = None;
    let mut finish_reason = String::new();
    while let Some(frame) = tokio::time::timeout(Duration::from_secs(180), stream.next())
        .await
        .expect("stream frame timed out")
        .transpose()
        .expect("stream error")
    {
        match frame.payload.expect("payload set") {
            Payload::Delta(d) => {
                deltas += 1;
                text.push_str(&d.text);
            }
            Payload::Final(f) => {
                usage = f.usage;
                finish_reason = f.finish_reason;
            }
        }
    }

    let usage = usage.expect("final usage frame");
    assert!(deltas >= 1, "should generate at least one token");
    assert_eq!(usage.completion_tokens as usize, deltas);
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens
    );
    assert!(usage.prompt_tokens > 0);
    assert!(matches!(finish_reason.as_str(), "stop" | "length"));
    eprintln!("generated {deltas} tokens, finish={finish_reason}, text={text:?}");
}

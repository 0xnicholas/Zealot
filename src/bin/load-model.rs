//! One-shot helper: connects to zealot-backend gRPC and loads a model.
//! Usage: ZEALOT_SITE_PACKAGES=... MODEL=hf-internal-testing/tiny-random-gpt2 cargo run --bin load-model

use std::time::Duration;

mod runtime_v1 {
    tonic::include_proto!("runtime.v1");
}

use runtime_v1::inference_runtime_client::InferenceRuntimeClient;
use runtime_v1::LoadModelRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("GRPC_ADDR").unwrap_or_else(|_| "http://127.0.0.1:9091".into());
    let model = std::env::var("MODEL").unwrap_or_else(|_| "hf-internal-testing/tiny-random-gpt2".into());

    let mut client = InferenceRuntimeClient::connect(addr).await?;
    println!("Loading model: {model}...");

    let resp = tokio::time::timeout(
        Duration::from_secs(600),
        client.load_model(LoadModelRequest {
            model_id: model.clone(),
            ..Default::default()
        }),
    )
    .await??;

    let inner = resp.into_inner();
    println!("LoadModel response: ready={}, message={}", inner.ready, inner.message);
    Ok(())
}

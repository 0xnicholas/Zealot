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

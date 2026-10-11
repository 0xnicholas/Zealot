//! zealot-backend — Zealot 独立进程的 gRPC 入口（ADR-010 Runtime Interface）。
//!
//! 当前形态（Phase 2 M4-M5，dev-mode）：
//! - 单模型：LoadModel 加载 HF 模型到 CPU（PyModelRunner，PyTorch CPU 前向）
//! - Engine actor：独立线程驱动 Scheduler + ModelRunner 步进循环，
//!   gRPC 侧经 channel 提交/取消请求，token 流经 per-request channel 回推
//! - Infer 为真实推理（greedy sampling）；Batch 仍 UNIMPLEMENTED（Scheduler
//!   的批处理语义后续里程碑接入）；GPU 指标为 0（无 GPU）
//!
//! 目标形态：Python 仅在启动时加载权重，decode loop 全 Rust + CUDA（ADR-009）。

use std::collections::HashMap;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::post,
    Json, Router,
};
use futures::StreamExt;
use tokio_stream::{wrappers::ReceiverStream, wrappers::TcpListenerStream, Stream};
use tonic::{transport::Server, Request, Status, Streaming};

use zealot_engine::constrained_decode::matcher::ConstrainedGrammar;
use zealot_engine::constrained_decode::schema::JsonSchemaCompiler;
#[cfg(feature = "cuda")]
use zealot_engine::cuda_runner::CudaModelRunner;
use zealot_engine::engine::{Engine, ModelRunner};
use zealot_engine::error::ZealotError;
use zealot_engine::mock_runner::MockModelRunner;
use zealot_engine::model_runner_cpu::CpuModelRunner;
use zealot_engine::openai::{self, ChatCompletionRequest};
use zealot_engine::sampling::SamplingParams;
use zealot_engine::scheduler::{Priority, Scheduler, SchedulerConfig};
use zealot_engine::tokenizer::Tokenizer;

// tonic 生成代码不参与本仓 clippy 硬闸：CI stable 工具链（@stable 浮动）
// 持续新增 lint（1.98 result_large_err、1.99 double_must_use 均命中
// tonic::Status/生成项），逐条豁免追不完，对生成模块整体豁免 clippy
#[allow(clippy::all)]
mod runtime_v1 {
    tonic::include_proto!("runtime.v1");
}

use runtime_v1::{
    infer_response::Payload,
    inference_runtime_server::{InferenceRuntime, InferenceRuntimeServer},
    BatchRequest, BatchResponse, BatchStatusRequest, BatchStatusResponse, FinalResponse,
    HealthCheckRequest, HealthCheckResponse, InferRequest, InferResponse, ListModelsRequest,
    ListModelsResponse, LoadModelRequest, LoadModelResponse, LoadedModel, MetricsSnapshot,
    Status as HealthStatus, StreamControl, StreamDelta, UnloadModelRequest, UnloadModelResponse,
    Usage,
};

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;
type Tx = tokio::sync::mpsc::Sender<Result<InferResponse, Status>>;

// ── Engine actor ──────────────────────────────────────

enum EngineCmd {
    Infer {
        request_id: String,
        messages: Vec<(String, String)>,
        max_tokens: usize,
        priority: Priority,
        sampling_params: SamplingParams,
        /// M5-④：HTTP 链路解析出的约束 grammar（gRPC 链路为 None）。
        /// Box：自动机体积大，避免 Infer 变体膨胀触发 large_enum_variant。
        grammar: Option<Box<ConstrainedGrammar>>,
        tx: Tx,
    },
    Cancel {
        request_id: String,
    },
    Shutdown,
}

struct SlotHandle {
    cmd_tx: std::sync::mpsc::Sender<EngineCmd>,
    model: LoadedModel,
}

/// 引擎槽 + HTTP 链路共享的 schema 编译器（LRU 缓存跨请求生效）。
#[derive(Clone)]
struct Runtime {
    slot: Arc<Mutex<ModelSlot>>,
    schema_compiler: Arc<JsonSchemaCompiler>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            slot: Arc::new(Mutex::new(ModelSlot::Empty)),
            // max_states 与自动机上限（2^18）一致；编译超 1s 拒绝（防对抗 schema）
            schema_compiler: Arc::new(JsonSchemaCompiler::new(16, 262_144, 1_000)),
        }
    }
}

#[derive(Default)]
enum ModelSlot {
    #[default]
    Empty,
    Loading,
    Ready(SlotHandle),
}

fn usage_frame(request_id: &str, prompt: usize, completion: usize, reason: &str) -> InferResponse {
    InferResponse {
        request_id: request_id.into(),
        payload: Some(Payload::Final(FinalResponse {
            usage: Some(Usage {
                prompt_tokens: prompt as u32,
                completion_tokens: completion as u32,
                total_tokens: (prompt + completion) as u32,
            }),
            finish_reason: reason.into(),
        })),
    }
}

/// 返回 false = 收到 Shutdown，线程退出。
fn handle_cmd(
    engine: &mut Engine,
    eos: Option<i64>,
    cmd: EngineCmd,
    inflight: &mut HashMap<String, Tx>,
) -> bool {
    match cmd {
        EngineCmd::Infer {
            request_id,
            messages,
            max_tokens,
            priority,
            sampling_params,
            grammar,
            tx,
        } => {
            let made = engine.runner().tokenize_chat(&messages).and_then(|ids| {
                engine.scheduler_mut().make_sequence(
                    request_id.clone(),
                    ids,
                    max_tokens,
                    priority,
                    eos,
                    sampling_params,
                )
            });
            match made {
                Ok(seq) => {
                    // M5-④：HTTP 链路解析出的约束 grammar 绑定到 sequence
                    let seq = match grammar {
                        Some(g) => seq.with_grammar(*g),
                        None => seq,
                    };
                    engine.scheduler_mut().add(seq);
                    inflight.insert(request_id, tx);
                }
                Err(e) => {
                    let _ = tx.blocking_send(Err(Status::invalid_argument(e.to_string())));
                }
            }
        }
        EngineCmd::Cancel { request_id } => {
            // ADR-010：取消也要发 final 帧（已生成 token 照常计费）
            if let Some((p, c)) = engine.cancel(&request_id) {
                if let Some(tx) = inflight.remove(&request_id) {
                    let _ = tx.blocking_send(Ok(usage_frame(&request_id, p, c, "cancel")));
                }
            }
        }
        EngineCmd::Shutdown => return false,
    }
    true
}

fn run_engine(mut engine: Engine, eos: Option<i64>, cmd_rx: std::sync::mpsc::Receiver<EngineCmd>) {
    let mut inflight: HashMap<String, Tx> = HashMap::new();
    loop {
        // 先 drain 掉积压命令（取消/新请求在步间得到处理）
        while let Ok(cmd) = cmd_rx.try_recv() {
            if !handle_cmd(&mut engine, eos, cmd, &mut inflight) {
                return;
            }
        }
        if engine.is_idle() {
            if inflight.is_empty() {
                // 完全空闲：阻塞等下一条命令
                match cmd_rx.recv() {
                    Ok(cmd) => {
                        if !handle_cmd(&mut engine, eos, cmd, &mut inflight) {
                            return;
                        }
                    }
                    Err(_) => return, // 所有 sender 断开
                }
                continue;
            }
            // inflight 非空但 scheduler 空闲：不应发生；防死等
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        match engine.step() {
            Ok(res) => {
                for t in res.tokens {
                    let Some(tx) = inflight.get(&t.request_id) else {
                        continue;
                    };
                    let frame = InferResponse {
                        request_id: t.request_id.clone(),
                        payload: Some(Payload::Delta(StreamDelta {
                            text: t.text.unwrap_or_else(|| format!("<tk:{}>", t.token)),
                        })),
                    };
                    if tx.blocking_send(Ok(frame)).is_err() {
                        // 客户端断开：取消并回收资源
                        engine.cancel(&t.request_id);
                        inflight.remove(&t.request_id);
                    }
                }
                for f in res.finished {
                    if let Some(tx) = inflight.remove(&f.request_id) {
                        let _ = tx.blocking_send(Ok(usage_frame(
                            &f.request_id,
                            f.prompt_tokens,
                            f.completion_tokens,
                            f.reason.as_str(),
                        )));
                    }
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for (_, tx) in inflight.drain() {
                    let _ = tx.blocking_send(Err(Status::internal(msg.clone())));
                }
            }
        }
    }
}

// ── gRPC service ─────────────────────────────────────

impl Runtime {
    // tonic::Status 含动态 details 字段，Err 变体 >128B；poison 分支极少触发，
    // 保持 Result<_, Status> 签名以便 10 处 `?` 直传，scoped allow 明确记录此权衡。
    #[allow(clippy::result_large_err)]
    fn lock_slot(&self) -> Result<std::sync::MutexGuard<'_, ModelSlot>, Status> {
        self.slot
            .lock()
            .map_err(|_| Status::internal("engine slot lock poisoned"))
    }
}

fn map_priority(p: i32) -> Priority {
    use runtime_v1::Priority as P;
    match P::try_from(p).unwrap_or(P::Medium) {
        P::Lowest => Priority::Lowest,
        P::Medium => Priority::Medium,
        P::High => Priority::High,
        P::Highest => Priority::Highest,
    }
}
impl Runtime {
    // ── 模型加载公共路径（M5-⑪ Bench，issue #25）────────────────────
    /// 加载模型公共路径（gRPC LoadModel 与 HTTP /admin/load 共用）。
    async fn load_model_common(
        &self,
        model_id: String,
        quantization: i32,
    ) -> Result<LoadedModel, ZealotError> {
        {
            let slot = self
                .lock_slot()
                .map_err(|st| ZealotError::Internal(st.to_string()))?;
            match &*slot {
                ModelSlot::Empty => {}
                ModelSlot::Loading => {
                    return Err(ZealotError::FailedPrecondition(
                        "a model load is already in progress".into(),
                    ))
                }
                ModelSlot::Ready(h) => {
                    return Err(ZealotError::AlreadyExists(format!(
                        "model '{}' already loaded; UnloadModel first",
                        h.model.model_id
                    )))
                }
            }
        }
        *self
            .lock_slot()
            .map_err(|st| ZealotError::Internal(st.to_string()))? = ModelSlot::Loading;

        // 加载是重阻塞操作（HF 下载 + CPU 读盘），移出 tokio 工作线程
        // （M5-④：mock: 前缀走确定性 dev runner，附内置 byte 词表 tokenizer）
        let model_id_clone = model_id.clone();
        let (runner, mock_tok): (Box<dyn ModelRunner>, Option<Tokenizer>) =
            match tokio::task::spawn_blocking(
                move || -> Result<(Box<dyn ModelRunner>, Option<Tokenizer>), ZealotError> {
                    if model_id_clone.starts_with("mock:") {
                        let runner = MockModelRunner::load(&model_id_clone)?;
                        let tok = runner.tokenizer();
                        return Ok((Box::new(runner), Some(tok)));
                    }
                    #[cfg(feature = "cuda")]
                    {
                        match CudaModelRunner::load(&model_id_clone, 0) {
                            Ok(r) => return Ok((Box::new(r), None)),
                            Err(e) => {
                                tracing::warn!("GPU runner unavailable, falling back to CPU: {e}");
                            }
                        }
                    }
                    Ok((Box::new(CpuModelRunner::load(&model_id_clone)?), None))
                },
            )
            .await
            {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    *self
                        .lock_slot()
                        .map_err(|st| ZealotError::Internal(st.to_string()))? = ModelSlot::Empty;
                    return Err(e);
                }
                Err(e) => {
                    *self
                        .lock_slot()
                        .map_err(|st| ZealotError::Internal(st.to_string()))? = ModelSlot::Empty;
                    return Err(ZealotError::Internal(format!("load task failed: {e}")));
                }
            };

        let eos = runner.eos_token_id();
        let sched = Scheduler::new(SchedulerConfig::default())
            .map_err(|e| ZealotError::Internal(e.to_string()))?;
        let mut engine = Engine::new(sched, runner);
        // Try to load tokenizer.json from model cache for Rust-side decoding
        if let Ok(tok) = Tokenizer::load(&model_id) {
            engine = engine.with_tokenizer(tok);
        } else if let Some(tok) = mock_tok {
            // dev mock：附加内置 byte 词表（vocab-mode 真实 token 校验）
            engine = engine.with_tokenizer(tok);
        }
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || run_engine(engine, eos, cmd_rx));

        let model = LoadedModel {
            model_id: model_id.clone(),
            quantization,
            gpu_count: 0, // CPU dev-mode
            ready: true,
        };
        *self
            .lock_slot()
            .map_err(|st| ZealotError::Internal(st.to_string()))? = ModelSlot::Ready(SlotHandle {
            cmd_tx,
            model: model.clone(),
        });
        Ok(model)
    }
}

fn infer_params_to_sampling(params: &runtime_v1::InferParams) -> SamplingParams {
    SamplingParams {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        repetition_penalty: if params.repetition_penalty == 0.0 {
            1.0
        } else {
            params.repetition_penalty
        },
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        seed: if params.seed == 0 {
            None
        } else {
            Some(params.seed)
        },
    }
}

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

#[tonic::async_trait]
impl InferenceRuntime for Runtime {
    // ── 模型生命周期 ──────────────────────────────────────

    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<tonic::Response<LoadModelResponse>, Status> {
        let req = request.into_inner();
        self.load_model_common(req.model_id.clone(), req.quantization)
            .await
            .map_err(|e| match &e {
                ZealotError::FailedPrecondition(m) => Status::failed_precondition(m.clone()),
                ZealotError::AlreadyExists(m) => Status::already_exists(m.clone()),
                other => Status::internal(other.to_string()),
            })?;
        Ok(tonic::Response::new(LoadModelResponse {
            model_id: req.model_id,
            ready: true,
            message: "loaded on CPU (dev-mode PyTorch runner)".into(),
        }))
    }

    async fn unload_model(
        &self,
        request: Request<UnloadModelRequest>,
    ) -> Result<tonic::Response<UnloadModelResponse>, Status> {
        let req = request.into_inner();
        let taken = std::mem::replace(&mut *self.lock_slot()?, ModelSlot::Empty);
        match taken {
            ModelSlot::Empty => Ok(tonic::Response::new(UnloadModelResponse {
                model_id: req.model_id,
                unloaded: false,
                message: "no model loaded".into(),
            })),
            ModelSlot::Loading => {
                *self.lock_slot()? = ModelSlot::Loading;
                Err(Status::failed_precondition("load in progress"))
            }
            ModelSlot::Ready(h) => {
                let _ = h.cmd_tx.send(EngineCmd::Shutdown);
                Ok(tonic::Response::new(UnloadModelResponse {
                    model_id: req.model_id,
                    unloaded: true,
                    message: String::new(),
                }))
            }
        }
    }

    async fn list_models(
        &self,
        _request: Request<ListModelsRequest>,
    ) -> Result<tonic::Response<ListModelsResponse>, Status> {
        let models = match &*self.lock_slot()? {
            ModelSlot::Ready(h) => vec![h.model.clone()],
            _ => vec![],
        };
        Ok(tonic::Response::new(ListModelsResponse { models }))
    }

    // ── 推理执行（双向流）────────────────────────────────

    type InferStream = ResponseStream<InferResponse>;

    async fn infer(
        &self,
        request: Request<Streaming<InferRequest>>,
    ) -> Result<tonic::Response<Self::InferStream>, Status> {
        let mut inbound = request.into_inner();
        let first = inbound
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("empty Infer stream"))?;
        if first.control() != StreamControl::StreamStart {
            return Err(Status::invalid_argument(
                "first InferRequest must be STREAM_START",
            ));
        }

        let cmd_tx = {
            let slot = self.lock_slot()?;
            match &*slot {
                ModelSlot::Ready(h) if h.model.model_id == first.model => h.cmd_tx.clone(),
                ModelSlot::Ready(h) => {
                    return Err(Status::not_found(format!(
                        "model '{}' not loaded (loaded: '{}')",
                        first.model, h.model.model_id
                    )))
                }
                _ => return Err(Status::failed_precondition("no model loaded")),
            }
        };

        let request_id = if first.request_id.is_empty() {
            format!("req-{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed))
        } else {
            first.request_id
        };
        let messages: Vec<(String, String)> = first
            .messages
            .into_iter()
            .map(|m| (m.role, m.content))
            .collect();
        if messages.is_empty() {
            return Err(Status::invalid_argument("messages must not be empty"));
        }
        // dev cap：无 GPU 时限制生成长度，默认 32
        let max_tokens = match first.params.as_ref().map(|p| p.max_tokens) {
            None | Some(0) => 32,
            Some(n) => (n as usize).min(512),
        };
        let priority = first
            .scheduling_hint
            .map(|h| map_priority(h.priority))
            .unwrap_or_default();

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let sampling = first
            .params
            .as_ref()
            .map(infer_params_to_sampling)
            .unwrap_or_default();
        cmd_tx
            .send(EngineCmd::Infer {
                request_id: request_id.clone(),
                messages,
                max_tokens,
                priority,
                sampling_params: sampling,
                // gRPC 链路暂不携带 grammar（后续里程碑）
                grammar: None,
                tx,
            })
            .map_err(|_| Status::internal("engine thread gone"))?;

        // 转发后续帧的 CANCEL。inbound EOF 不取消——gRPC 双向流客户端
        // 可以 half-close 发送侧同时继续读响应。
        let rid = request_id;
        tokio::spawn(async move {
            loop {
                match inbound.message().await {
                    Ok(Some(frame)) if frame.control() == StreamControl::Cancel => {
                        let _ = cmd_tx.send(EngineCmd::Cancel {
                            request_id: rid.clone(),
                        });
                    }
                    Ok(Some(_)) => {
                        // MESSAGE_APPEND 未实现（多轮对话后续里程碑）
                    }
                    Ok(None) | Err(_) => break,
                }
            }
        });

        Ok(tonic::Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    // ── 批处理（后续里程碑）──────────────────────────────

    async fn submit_batch(
        &self,
        _request: Request<BatchRequest>,
    ) -> Result<tonic::Response<BatchResponse>, Status> {
        Err(Status::unimplemented(
            "SubmitBatch lands with the Rust Scheduler (Phase 2 M5+)",
        ))
    }

    async fn get_batch_status(
        &self,
        _request: Request<BatchStatusRequest>,
    ) -> Result<tonic::Response<BatchStatusResponse>, Status> {
        Err(Status::unimplemented(
            "GetBatchStatus lands with the Rust Scheduler (Phase 2 M5+)",
        ))
    }

    // ── 健康检查与指标 ───────────────────────────────────

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<tonic::Response<HealthCheckResponse>, Status> {
        // GPU 指标在 DCGM 接入前为 0。
        let status = match &*self.lock_slot()? {
            ModelSlot::Ready(_) => HealthStatus::Healthy,
            ModelSlot::Loading => HealthStatus::Degraded,
            ModelSlot::Empty => HealthStatus::Healthy,
        };
        Ok(tonic::Response::new(HealthCheckResponse {
            status: status as i32,
            queue_depth: 0,
            gpu_util_pct: 0.0,
            memory_used_gb: 0.0,
            active_requests: 0,
        }))
    }

    type StreamMetricsStream = ResponseStream<MetricsSnapshot>;

    async fn stream_metrics(
        &self,
        _request: Request<()>,
    ) -> Result<tonic::Response<Self::StreamMetricsStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let snapshot = MetricsSnapshot {
                    timestamp_unix_ms: now_ms,
                    gpu_util_pct: 0.0,
                    memory_used_gb: 0.0,
                    active_requests: 0,
                    queue_depth: 0,
                    tokens_per_second: 0.0,
                };
                if tx.send(Ok(snapshot)).await.is_err() {
                    break; // client disconnected
                }
            }
        });
        Ok(tonic::Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

// ── OpenAI HTTP handler ─────────────────────────────

async fn health_check(State(backend): State<Runtime>) -> Json<serde_json::Value> {
    let status = match backend.slot.lock() {
        Ok(slot) => match &*slot {
            ModelSlot::Ready(h) => serde_json::json!({
                "status": "healthy",
                "model": h.model.model_id,
                "ready": true,
            }),
            ModelSlot::Loading => serde_json::json!({
                "status": "degraded",
                "model": null,
                "ready": false,
            }),
            ModelSlot::Empty => serde_json::json!({
                "status": "healthy",
                "model": null,
                "ready": false,
            }),
        },
        Err(_) => serde_json::json!({"status": "unhealthy", "ready": false}),
    };
    Json(status)
}

/// bench/内部用 dev 加载口：`POST /admin/load` {"model": "mock:tiny"}（M5-⑪ Bench，issue #25）。
/// 与 gRPC LoadModel 走同一 load_model_common 路径。
#[derive(serde::Deserialize)]
struct AdminLoadRequest {
    model: String,
}

async fn admin_load(
    State(backend): State<Runtime>,
    Json(body): Json<AdminLoadRequest>,
) -> axum::response::Response {
    match backend.load_model_common(body.model.clone(), 0).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({"model": body.model, "ready": true})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn handle_openai_chat(
    State(backend): State<Runtime>,
    Json(body): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let (model_id, cmd_tx) = {
        let slot = match backend.slot.lock() {
            Ok(s) => s,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "engine slot lock poisoned"})),
                )
                    .into_response()
            }
        };
        match &*slot {
            ModelSlot::Ready(h) => (h.model.model_id.clone(), h.cmd_tx.clone()),
            _ => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error": "no model loaded"})),
                )
                    .into_response()
            }
        }
    };

    let request_id = format!(
        "chatcmpl-{}",
        REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let messages: Vec<(String, String)> = body
        .messages
        .into_iter()
        .map(|m| (m.role, m.content))
        .collect();
    let max_tokens = body.max_tokens.unwrap_or(32).min(512) as usize;

    let sampling_params = zealot_engine::sampling::SamplingParams {
        temperature: body.temperature,
        top_k: 0,
        top_p: body.top_p,
        repetition_penalty: 1.0,
        frequency_penalty: body.frequency_penalty,
        presence_penalty: body.presence_penalty,
        seed: body.seed,
    };

    // M5-④：response_format / tools / tool_choice → 约束 grammar（编译失败 400）
    let grammar = match openai::resolve_grammar(
        &backend.schema_compiler,
        body.response_format.as_ref(),
        body.tools.as_deref(),
        body.tool_choice.as_ref(),
    ) {
        Ok(g) => g.map(Box::new),
        Err(msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": msg,
                        "type": "invalid_request_error",
                        "param": null,
                        "code": null,
                    }
                })),
            )
                .into_response()
        }
    };

    let (tx, rx) = tokio::sync::mpsc::channel(64);
    if cmd_tx
        .send(EngineCmd::Infer {
            request_id: request_id.clone(),
            messages,
            max_tokens,
            priority: Priority::Medium,
            sampling_params,
            grammar,
            tx,
        })
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "engine thread gone"})),
        )
            .into_response();
    }

    if body.stream {
        streaming_response(rx, &request_id, &model_id, ts).into_response()
    } else {
        non_streaming_response(rx, &request_id, &model_id, ts)
            .await
            .into_response()
    }
}

fn streaming_response(
    rx: tokio::sync::mpsc::Receiver<Result<InferResponse, tonic::Status>>,
    request_id: &str,
    model: &str,
    ts: u64,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let rid = request_id.to_owned();
    let mdl = model.to_owned();
    let stream = ReceiverStream::new(rx).flat_map(move |frame| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        if let Ok(resp) = frame {
            match resp.payload {
                Some(Payload::Delta(d)) => {
                    let json = serde_json::to_string(&openai::sse_content(&rid, &mdl, ts, &d.text))
                        .unwrap_or_default();
                    events.push(Ok(Event::default().data(json)));
                }
                Some(Payload::Final(f)) => {
                    let pt = f.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0);
                    let ct = f.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0);
                    let json = serde_json::to_string(&openai::sse_finish(
                        &rid,
                        &mdl,
                        ts,
                        &f.finish_reason,
                        pt,
                        ct,
                    ))
                    .unwrap_or_default();
                    events.push(Ok(Event::default().data(json)));
                    events.push(Ok(Event::default().data("[DONE]")));
                }
                None => {}
            }
        }
        futures::stream::iter(events)
    });
    Sse::new(stream)
}

async fn non_streaming_response(
    rx: tokio::sync::mpsc::Receiver<Result<InferResponse, tonic::Status>>,
    request_id: &str,
    model: &str,
    ts: u64,
) -> Json<serde_json::Value> {
    let mut content = String::new();
    let mut finish_reason = String::from("stop");
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;

    let mut stream = ReceiverStream::new(rx);
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(resp) => match resp.payload {
                Some(Payload::Delta(d)) => content.push_str(&d.text),
                Some(Payload::Final(f)) => {
                    finish_reason = f.finish_reason;
                    if let Some(u) = f.usage {
                        prompt_tokens = u.prompt_tokens;
                        completion_tokens = u.completion_tokens;
                    }
                }
                None => {}
            },
            Err(_) => break,
        }
    }

    let json = openai::non_streaming_response(
        request_id,
        model,
        ts,
        content,
        finish_reason,
        prompt_tokens,
        completion_tokens,
    );
    let value: serde_json::Value = serde_json::from_str(&json).unwrap_or_default();
    Json(value)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    pyo3::prepare_freethreaded_python();

    let grpc_port: u16 = std::env::var("ZEALOT_GRPC_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9091);
    let http_port: u16 = std::env::var("ZEALOT_HTTP_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8081);

    let backend = Runtime::default();

    // gRPC server
    let grpc_listener = tokio::net::TcpListener::bind(("0.0.0.0", grpc_port)).await?;
    let grpc_addr = grpc_listener.local_addr()?;
    let grpc_backend = backend.clone();
    tokio::spawn(async move {
        Server::builder()
            .add_service(InferenceRuntimeServer::new(grpc_backend))
            .serve_with_incoming(TcpListenerStream::new(grpc_listener))
            .await
            .expect("gRPC server failed");
    });

    // HTTP server
    let http_listener = tokio::net::TcpListener::bind(("0.0.0.0", http_port)).await?;
    let http_addr = http_listener.local_addr()?;
    // First stdout line is parsed by tests (common/mod.rs). Must keep format.
    println!("zealot-backend listening on {grpc_addr} (gRPC) {http_addr} (HTTP)");

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_openai_chat))
        .route("/admin/load", post(admin_load))
        .route("/health", axum::routing::get(health_check))
        .route("/ready", axum::routing::get(health_check))
        .with_state(backend);

    axum::serve(http_listener, app).await?;
    Ok(())
}

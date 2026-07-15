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
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio_stream::{wrappers::ReceiverStream, wrappers::TcpListenerStream, Stream};
use tonic::{transport::Server, Request, Response, Status, Streaming};

use zealot_engine::engine::Engine;
use zealot_engine::model_runner_py::PyModelRunner;
use zealot_engine::scheduler::{Priority, Scheduler, SchedulerConfig};

mod runtime_v1 {
    tonic::include_proto!("runtime.v1");
}

use runtime_v1::{
    inference_runtime_server::{InferenceRuntime, InferenceRuntimeServer},
    infer_response::Payload, BatchRequest, BatchResponse, BatchStatusRequest,
    BatchStatusResponse, FinalResponse, HealthCheckRequest, HealthCheckResponse,
    InferRequest, InferResponse, ListModelsRequest, ListModelsResponse,
    LoadModelRequest, LoadModelResponse, LoadedModel, MetricsSnapshot,
    StreamControl, StreamDelta, UnloadModelRequest, UnloadModelResponse, Usage,
    Status as HealthStatus,
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
        tx: Tx,
    },
    Cancel {
        request_id: String,
    },
    Shutdown,
}

struct EngineHandle {
    cmd_tx: std::sync::mpsc::Sender<EngineCmd>,
    model: LoadedModel,
}

enum EngineSlot {
    Empty,
    Loading,
    Ready(EngineHandle),
}

impl Default for EngineSlot {
    fn default() -> Self {
        EngineSlot::Empty
    }
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
    engine: &mut Engine<PyModelRunner>,
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
            tx,
        } => {
            let made = engine
                .runner()
                .tokenize_chat(&messages)
                .and_then(|ids| {
                    engine.scheduler_mut().make_sequence(
                        request_id.clone(),
                        ids,
                        max_tokens,
                        priority,
                        eos,
                    )
                });
            match made {
                Ok(seq) => {
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

fn run_engine(
    mut engine: Engine<PyModelRunner>,
    eos: Option<i64>,
    cmd_rx: std::sync::mpsc::Receiver<EngineCmd>,
) {
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
                            text: t.text.unwrap_or_default(),
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

#[derive(Default)]
struct ZealotBackend {
    slot: Mutex<EngineSlot>,
}

impl ZealotBackend {
    fn lock_slot(&self) -> Result<std::sync::MutexGuard<'_, EngineSlot>, Status> {
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

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

#[tonic::async_trait]
impl InferenceRuntime for ZealotBackend {
    // ── 模型生命周期 ──────────────────────────────────────

    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();
        {
            let slot = self.lock_slot()?;
            match &*slot {
                EngineSlot::Empty => {}
                EngineSlot::Loading => {
                    return Err(Status::failed_precondition(
                        "a model load is already in progress",
                    ))
                }
                EngineSlot::Ready(h) => {
                    return Err(Status::already_exists(format!(
                        "model '{}' already loaded; UnloadModel first",
                        h.model.model_id
                    )))
                }
            }
        }
        *self.lock_slot()? = EngineSlot::Loading;

        let model_id = req.model_id.clone();
        // 加载是重阻塞操作（HF 下载 + CPU 读盘），移出 tokio 工作线程
        let runner = match tokio::task::spawn_blocking(move || PyModelRunner::load(&model_id)).await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                *self.lock_slot()? = EngineSlot::Empty;
                return Err(Status::internal(format!("model load failed: {e}")));
            }
            Err(e) => {
                *self.lock_slot()? = EngineSlot::Empty;
                return Err(Status::internal(format!("load task failed: {e}")));
            }
        };

        let eos = runner.eos_token_id();
        let sched = Scheduler::new(SchedulerConfig::default())
            .map_err(|e| Status::internal(e.to_string()))?;
        let engine = Engine::new(sched, runner);
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || run_engine(engine, eos, cmd_rx));

        let model = LoadedModel {
            model_id: req.model_id.clone(),
            quantization: req.quantization,
            gpu_count: 0, // CPU dev-mode
            ready: true,
        };
        *self.lock_slot()? = EngineSlot::Ready(EngineHandle { cmd_tx, model });
        Ok(Response::new(LoadModelResponse {
            model_id: req.model_id,
            ready: true,
            message: "loaded on CPU (dev-mode PyTorch runner)".into(),
        }))
    }

    async fn unload_model(
        &self,
        request: Request<UnloadModelRequest>,
    ) -> Result<Response<UnloadModelResponse>, Status> {
        let req = request.into_inner();
        let taken = std::mem::replace(&mut *self.lock_slot()?, EngineSlot::Empty);
        match taken {
            EngineSlot::Empty => Ok(Response::new(UnloadModelResponse {
                model_id: req.model_id,
                unloaded: false,
                message: "no model loaded".into(),
            })),
            EngineSlot::Loading => {
                *self.lock_slot()? = EngineSlot::Loading;
                Err(Status::failed_precondition("load in progress"))
            }
            EngineSlot::Ready(h) => {
                let _ = h.cmd_tx.send(EngineCmd::Shutdown);
                Ok(Response::new(UnloadModelResponse {
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
    ) -> Result<Response<ListModelsResponse>, Status> {
        let models = match &*self.lock_slot()? {
            EngineSlot::Ready(h) => vec![h.model.clone()],
            _ => vec![],
        };
        Ok(Response::new(ListModelsResponse { models }))
    }

    // ── 推理执行（双向流）────────────────────────────────

    type InferStream = ResponseStream<InferResponse>;

    async fn infer(
        &self,
        request: Request<Streaming<InferRequest>>,
    ) -> Result<Response<Self::InferStream>, Status> {
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
                EngineSlot::Ready(h) if h.model.model_id == first.model => h.cmd_tx.clone(),
                EngineSlot::Ready(h) => {
                    return Err(Status::not_found(format!(
                        "model '{}' not loaded (loaded: '{}')",
                        first.model, h.model.model_id
                    )))
                }
                _ => return Err(Status::failed_precondition("no model loaded")),
            }
        };

        let request_id = if first.request_id.is_empty() {
            format!(
                "req-{}",
                REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
            )
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
        cmd_tx
            .send(EngineCmd::Infer {
                request_id: request_id.clone(),
                messages,
                max_tokens,
                priority,
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

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    // ── 批处理（后续里程碑）──────────────────────────────

    async fn submit_batch(
        &self,
        _request: Request<BatchRequest>,
    ) -> Result<Response<BatchResponse>, Status> {
        Err(Status::unimplemented(
            "SubmitBatch lands with the Rust Scheduler (Phase 2 M5+)",
        ))
    }

    async fn get_batch_status(
        &self,
        _request: Request<BatchStatusRequest>,
    ) -> Result<Response<BatchStatusResponse>, Status> {
        Err(Status::unimplemented(
            "GetBatchStatus lands with the Rust Scheduler (Phase 2 M5+)",
        ))
    }

    // ── 健康检查与指标 ───────────────────────────────────

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        // GPU 指标在 DCGM 接入前为 0。
        let status = match &*self.lock_slot()? {
            EngineSlot::Ready(_) => HealthStatus::Healthy,
            EngineSlot::Loading => HealthStatus::Degraded,
            EngineSlot::Empty => HealthStatus::Healthy,
        };
        Ok(Response::new(HealthCheckResponse {
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
    ) -> Result<Response<Self::StreamMetricsStream>, Status> {
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
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 嵌入式 CPython：Model Loader / dev-mode runner 用（ADR-009）
    pyo3::prepare_freethreaded_python();

    let port: u16 = std::env::var("ZEALOT_GRPC_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9091);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    let local = listener.local_addr()?;
    // 集成测试解析这一行获取实际端口（端口 0 时）。
    println!("zealot-backend listening on {local}");

    Server::builder()
        .add_service(InferenceRuntimeServer::new(ZealotBackend::default()))
        .serve_with_incoming(TcpListenerStream::new(listener))
        .await?;
    Ok(())
}

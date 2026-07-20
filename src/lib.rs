use pyo3::prelude::*;

pub mod attention;
pub mod block_manager;
mod constrained_decode;
pub mod engine;
pub mod error;
pub mod model_runner_cuda;
pub mod model_runner_py;
pub mod sampling;
pub mod scheduler;
pub mod tokenizer;

/// Zealot Inference Engine — standalone engine (no vLLM fork), ADR-009.
///
/// Architecture:
///   Rust   ──→ main process: gRPC server (tonic) + Scheduler
///              + Block Manager + Constrained Decode
///   Python ──→ Model Loader only (HuggingFace, embedded via PyO3 at startup)
///   CUDA   ──→ Attention Kernel + Quantization (modified, not rewritten)
///
/// M4 state: components ship as a cdylib Python extension (`zealot_engine`)
/// for GPU-free development and testing. The tonic main process with
/// embedded CPython is the integration target (docs/architecture.md §5).

#[pymodule]
#[pyo3(name = "zealot_engine")]
fn zealot_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    block_manager::register(m)?;
    constrained_decode::register(m)?;
    Ok(())
}

use pyo3::prelude::*;

mod block_manager;
mod constrained_decode;
mod error;

/// Zealot Inference Engine — Rust-accelerated vLLM components.
///
/// Architecture (ADR-009):
///   Python ──→ API Server + Model Loader (retained from vLLM)
///   Rust   ──→ Block Manager + Constrained Decode + Scheduler (component-level replacement)
///   CUDA   ──→ Attention Kernel + Quantization (modified, not rewritten)
///
/// Language boundary: PyO3 (Rust → Python .so), zero-copy where possible.
/// The Rust components are injected into vLLM's Python codebase via the
/// same FFI mechanism vLLM already uses for its C++ extensions (_C.abi3.so).

#[pymodule]
#[pyo3(name = "zealot_engine")]
fn zealot_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    block_manager::register(m)?;
    constrained_decode::register(m)?;
    Ok(())
}

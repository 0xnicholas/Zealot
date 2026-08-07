// zealot/src/attention/cublas.rs
#![allow(non_camel_case_types, dead_code)]

#[cfg(feature = "cuda")]
use crate::error::ZealotError;
#[cfg(feature = "cuda")]
use cudarc::driver::sys::CUstream;

#[cfg(feature = "cuda")]
type cudaStream_t = CUstream;

#[cfg(feature = "cuda")]
type CublasHandle = *mut std::ffi::c_void;

#[cfg(feature = "cuda")]
#[repr(i32)]
#[derive(Clone, Copy)]
pub enum CublasOp {
    N = 0,
    T = 1,
}

#[cfg(feature = "cuda")]
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum CublasStatus {
    Success = 0,
}

#[cfg(feature = "cuda")]
#[link(name = "cublas")]
extern "C" {
    fn cublasCreate_v2(handle: *mut CublasHandle) -> CublasStatus;
    fn cublasDestroy_v2(handle: CublasHandle) -> CublasStatus;
    fn cublasSgemm_v2(
        handle: CublasHandle,
        transa: CublasOp,
        transb: CublasOp,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> CublasStatus;
    fn cublasSetStream_v2(handle: CublasHandle, stream: cudaStream_t) -> CublasStatus;
    fn cublasGetErrorString(status: CublasStatus) -> *const std::ffi::c_char;
}

#[cfg(feature = "cuda")]
pub struct CublasContext {
    handle: CublasHandle,
}

#[cfg(feature = "cuda")]
impl CublasContext {
    pub fn new() -> Result<Self, ZealotError> {
        let mut handle: CublasHandle = std::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut handle) };
        if status != CublasStatus::Success || handle.is_null() {
            return Err(ZealotError::Internal("cublasCreate_v2 failed".into()));
        }
        Ok(Self { handle })
    }

    /// C = op(A) × op(B). alpha=1, beta=0 for standard matmul.
    pub fn sgemm(
        &self,
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        lda: i32,
        trans_a: CublasOp,
        b: *const f32,
        ldb: i32,
        trans_b: CublasOp,
        c: *mut f32,
        ldc: i32,
    ) -> Result<(), ZealotError> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = unsafe {
            cublasSgemm_v2(
                self.handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                &alpha,
                a,
                lda,
                b,
                ldb,
                &beta,
                c,
                ldc,
            )
        };
        if status != CublasStatus::Success {
            let msg = unsafe {
                let ptr = cublasGetErrorString(status);
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            };
            return Err(ZealotError::Internal(format!("cublasSgemm: {msg}")));
        }
        Ok(())
    }

    /// Safety: CUstream and cudaStream_t are the same pointer type on all
    /// CUDA platforms (driver API ↔ runtime API type-punning).
    /// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
    pub fn set_stream(&self, stream: CUstream) -> Result<(), ZealotError> {
        let status = unsafe { cublasSetStream_v2(self.handle, stream) };
        if status != CublasStatus::Success {
            return Err(ZealotError::Internal("cublasSetStream_v2 failed".into()));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for CublasContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cublasDestroy_v2(self.handle) };
        }
    }
}

// Safety: cuBLAS handle is thread-safe (cuBLAS docs guarantee this)
#[cfg(feature = "cuda")]
unsafe impl Send for CublasContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CublasContext {}

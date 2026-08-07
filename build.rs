// zealot/build.rs
use std::path::Path;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Loom 并发模拟（M4 §12-2，issue #7）──
    // loom model checker 要求本 crate 用 `loom::sync` 类型编译；loom 0.7
    // 的同步原语本身就是模型实现，`cfg(loom)` 只决定用户 crate 选哪种类型。
    // 通过环境变量触发并用 build script 的 rustc-cfg 只对本 crate 生效，
    // 避免 RUSTFLAGS="--cfg loom" 全局生效时破坏 tokio 等依赖的编译。
    // 运行方式：ZEALOT_LOOM=1 cargo test --lib loom -- --test-threads=1
    println!("cargo:rerun-if-env-changed=ZEALOT_LOOM");
    if std::env::var_os("ZEALOT_LOOM").is_some() {
        println!("cargo:rustc-cfg=loom");
        println!("cargo::rustc-check-cfg=cfg(loom)");
    } else {
        println!("cargo::rustc-check-cfg=cfg(loom)");
    }

    // ── Existing proto compilation ──
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/runtime/v1/runtime.proto"], &["proto"])?;

    // ── CUDA kernel compilation ──
    compile_cuda_kernels();

    Ok(())
}

fn compile_cuda_kernels() {
    // Only compile kernels when --features cuda is active
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let kernels_dir = Path::new("src/attention/kernels");
    let out_dir_val = std::env::var("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir_val);
    let cu_files = [
        "attention.cu",
        "rms_norm.cu",
        "embedding.cu",
        "transpose.cu",
        "paged_attention.cu",
        "activation.cu",
    ];

    // Find nvcc
    let nvcc = match which::which("nvcc") {
        Ok(path) => path,
        Err(_) => {
            println!("cargo:warning=nvcc not found — generating PTX stubs (no GPU compute)");
            for cu_file in &cu_files {
                let ptx_name = format!("{}.ptx", cu_file.trim_end_matches(".cu"));
                let ptx_path = out_dir.join(&ptx_name);
                std::fs::write(&ptx_path, "\"\"").unwrap();
            }
            return;
        }
    };

    // Detect CUDA library path for cuBLAS linking
    if let Some(lib_dir) = detect_cuda_lib(&nvcc) {
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=cublas");
    }

    for cu_file in &cu_files {
        let ptx_name = format!("{}.ptx", cu_file.trim_end_matches(".cu"));
        let ptx_path = out_dir.join(&ptx_name);
        let src_path = kernels_dir.join(cu_file);

        let status = Command::new(&nvcc)
            .args(["-ptx", "-o"])
            .arg(&ptx_path)
            .arg(&src_path)
            .args(["--gpu-architecture=sm_80,sm_89,sm_90"])
            .status();

        match status {
            Ok(s) if s.success() => println!("cargo:rerun-if-changed={}", src_path.display()),
            Ok(s) => println!("cargo:warning=nvcc failed for {cu_file} (exit {})", s),
            Err(e) => println!("cargo:warning=cannot run nvcc for {cu_file}: {e}"),
        }
    }
}

fn detect_cuda_lib(nvcc: &Path) -> Option<String> {
    // Try common paths
    let candidates = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/opt/cuda/lib64",
    ];
    for dir in &candidates {
        if Path::new(dir).join("libcublas.so").exists() {
            return Some(dir.to_string());
        }
    }
    // Fallback: try to infer from nvcc path
    if let Some(base) = nvcc.parent().and_then(|p| p.parent()) {
        let lib = base.join("lib64");
        if lib.join("libcublas.so").exists() {
            return Some(lib.to_string_lossy().to_string());
        }
    }
    None
}

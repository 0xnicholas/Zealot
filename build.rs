// Build script: compile the Runtime Interface proto (ADR-010) for the
// zealot-backend gRPC server. protoc is vendored so no system install is needed.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &["../proto/runtime/v1/runtime.proto"],
            &["../proto"],
        )?;
    Ok(())
}

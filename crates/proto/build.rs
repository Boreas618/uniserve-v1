//! Compiles the cross-build interfaces' Protobuf schemas into Rust, and
//! names the build this crate is compiled in (`UNISERVE_SOURCE_DIGEST`).
//!
//! The schemas are compiled with `protox`, a Protobuf compiler written in
//! Rust, so a build needs no `protoc`.

include!("source_digest.rs");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let schemas = [manifest.join("process.proto")];
    for schema in &schemas {
        println!("cargo:rerun-if-changed={}", schema.display());
    }
    let descriptors = protox::compile(&schemas, [&manifest])?;
    prost_build::Config::new().compile_fds(descriptors)?;

    emit_source_digest()?;
    Ok(())
}

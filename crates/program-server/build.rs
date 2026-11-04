//! Compiles the Program Server's Protobuf schema into Rust.
//!
//! The schema is compiled with `protox`, a Protobuf compiler written in Rust,
//! so a build needs no `protoc`. It lies under `proto/`, and the build reruns
//! when it changes. The protocol is append-only (design section 6.3): fields
//! are added to its messages and never renumbered or removed, so a client of
//! an earlier build keeps talking to a server of a later one.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let schema = manifest.join("proto/program_server.proto");
    println!("cargo:rerun-if-changed={}", schema.display());
    let descriptor = protox::compile([&schema], [manifest.join("proto")])?;
    prost_build::Config::new().compile_fds(descriptor)?;
    Ok(())
}

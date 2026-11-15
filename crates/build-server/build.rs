//! Compiles the Build Server's Protobuf schema into Rust.
//!
//! The schema is compiled with `protox`, a Protobuf compiler written in Rust,
//! so a build needs no `protoc`. It lies under `proto/`, and the build reruns
//! when it changes. The protocol is one release's (design section 6.3): the
//! Build Server and its clients of one release speak it.
//!
//! The script also records the Rust target triple the crate is compiled for,
//! cargo's `TARGET`, as `BUILD_SERVER_HOST_TARGET`: the target of a module that
//! runs on this host's CPU, the engine's executable (`Target::host`).

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let schema = manifest.join("proto/build_server.proto");
    println!("cargo:rerun-if-changed={}", schema.display());
    let descriptor = protox::compile([&schema], [manifest.join("proto")])?;
    prost_build::Config::new().compile_fds(descriptor)?;
    println!(
        "cargo:rustc-env=BUILD_SERVER_HOST_TARGET={}",
        std::env::var("TARGET")?
    );
    Ok(())
}

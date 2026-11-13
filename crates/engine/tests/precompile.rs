//! `uniserve-engine precompile` is how a scheduling policy package becomes the artifact the Build
//! Server stores: the package's `[toolchain]` declares the version the command reports, and the
//! engine loads what it wrote with the same wasmtime configuration
//! (`engine::scheduler::policy::load`).
//!
//! These cases state the interface from the outside — the command line the Build Server runs, and
//! the artifact the engine then loads — because the two processes share files, not memory.

#![cfg(test)]

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use engine::scheduler::policy;

/// The engine executable, built by this test's `cargo`.
const ENGINE: &str = env!("CARGO_BIN_EXE_uniserve-engine");

/// The eight bytes of an empty core module: the magic and the version.
const EMPTY_MODULE: &[u8] = b"\0asm\x01\0\0\0";

/// The `precompile` command with `arguments`.
fn precompile(arguments: &[&std::ffi::OsStr]) -> Output {
    Command::new(ENGINE)
        .arg("precompile")
        .args(arguments)
        .output()
        .expect("the engine executable runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A directory of this test's own, so two runs of the cases do not collide.
fn directory(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("engine-precompile-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).expect("a temporary directory");
    directory
}

/// The version a package's `[toolchain]` must declare is the last whitespace token of
/// `precompile --version`: that is what the Build Server reads to compare a package against the
/// toolchain it builds it with.
#[test]
fn the_version_is_the_wasmtime_a_package_must_declare() {
    let output = precompile(&["--version".as_ref()]);
    assert!(output.status.success(), "{}", stderr(&output));

    let stdout = String::from_utf8(output.stdout).expect("utf-8");
    let version = stdout.split_whitespace().last();
    assert_eq!(version, Some(policy::WASMTIME_VERSION));
}

/// A command line the command cannot serve is a usage failure, not a failed build: exit 2, with
/// the missing or unknown flag named.
#[test]
fn a_command_line_it_cannot_serve_is_a_usage_failure() {
    let output = precompile(&["--wasm".as_ref(), "policy.wasm".as_ref()]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(stderr(&output).contains("--out"), "{}", stderr(&output));

    let output = precompile(&["--nonsense".as_ref()]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("--nonsense"),
        "{}",
        stderr(&output)
    );
}

/// What the command wrote is what the engine loads: the artifact is a precompiled module, not one
/// compiled again from source, and it lands in a directory that did not exist.
#[test]
fn the_artifact_it_wrote_is_what_the_engine_loads() {
    let directory = directory("round-trip");
    let wasm = directory.join("policy.wasm");
    let artifact = directory.join("nested/modules/policy.module.wasm");
    fs::write(&wasm, EMPTY_MODULE).expect("the raw module is written");

    let output = precompile(&[
        "--wasm".as_ref(),
        wasm.as_os_str(),
        "--out".as_ref(),
        artifact.as_os_str(),
    ]);
    assert!(output.status.success(), "{}", stderr(&output));

    let bytes = fs::read(&artifact).expect("the artifact is written");
    assert_eq!(
        wasmtime::Engine::detect_precompiled(&bytes),
        Some(wasmtime::Precompiled::Module)
    );
    let module = policy::load(&bytes).expect("the engine loads the artifact");
    assert_eq!(
        module.exports().count(),
        0,
        "the artifact is the module it was given, and an empty module exports nothing"
    );
    let _ = fs::remove_dir_all(&directory);
}

/// A file the command cannot read, and one that is not a module, each name what failed: a
/// package's build log has to say which of its inputs was wrong.
#[test]
fn an_input_it_cannot_use_names_what_failed() {
    let directory = directory("failures");
    let out = directory.join("out.wasm");

    let missing = directory.join("absent.wasm");
    let output = precompile(&[
        "--wasm".as_ref(),
        missing.as_os_str(),
        "--out".as_ref(),
        out.as_os_str(),
    ]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("cannot read"),
        "{}",
        stderr(&output)
    );
    assert!(
        stderr(&output).contains("absent.wasm"),
        "{}",
        stderr(&output)
    );

    let garbage = directory.join("garbage.wasm");
    fs::write(&garbage, b"not a module at all").expect("the file is written");
    let output = precompile(&[
        "--wasm".as_ref(),
        garbage.as_os_str(),
        "--out".as_ref(),
        out.as_os_str(),
    ]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("cannot precompile"),
        "{}",
        stderr(&output)
    );
    let _ = fs::remove_dir_all(&directory);
}

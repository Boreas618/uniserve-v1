//! `scheduling-precompile` is how a policy package becomes the artifact the
//! Program Server stores (design §5.2, §5.3): the package's `[toolchain]`
//! declares the version this binary reports, and the main process loads what it
//! wrote with an engine built the same way.
//!
//! These cases state the contract from the outside — the command line a package
//! is built with, and the artifact the main process then loads — because the
//! two processes never share memory, only files.
#![cfg(test)]

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

/// The tool the Program Server runs, built by this test's `cargo`.
const TOOL: &str = env!("CARGO_BIN_EXE_scheduling-precompile");

/// The eight bytes of an empty core module: the magic and the version.
const EMPTY_MODULE: &[u8] = b"\0asm\x01\0\0\0";

fn tool() -> Command {
    Command::new(TOOL)
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A directory of this test's own, so two runs of the cases do not collide.
fn workspace(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "scheduling-precompile-{name}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a temporary directory");
    dir
}

/// The version a package's `[toolchain]` must declare is the last whitespace
/// token of `--version`: that is what the Program Server reads to compare a
/// package against the toolchain it will build it with.
#[test]
fn version_reports_the_wasmtime_the_package_must_declare() {
    let output = tool().arg("--version").output().expect("the tool runs");
    assert!(output.status.success(), "{}", stderr(&output));
    let stdout = String::from_utf8(output.stdout).expect("utf-8");
    let version = stdout
        .split_whitespace()
        .last()
        .expect("a version token at the end of the line");
    assert_eq!(version, scheduling::WASMTIME_VERSION);
}

/// A command line the tool does not understand is a usage failure, not a failed
/// build: exit 2, and the missing flag named.
#[test]
fn a_command_line_it_cannot_serve_is_a_usage_failure() {
    let output = tool()
        .args(["--wasm", "policy.wasm"])
        .output()
        .expect("the tool runs");
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("--out is required"),
        "{}",
        stderr(&output)
    );

    let output = tool().arg("--nonsense").output().expect("the tool runs");
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("--nonsense"),
        "{}",
        stderr(&output)
    );
}

/// What the tool wrote is what the main process loads: the artifact is
/// detected as a precompiled module, not recompiled from source, and it comes
/// out of a directory that did not exist.
#[test]
fn the_artifact_the_tool_wrote_is_what_the_main_process_loads() {
    let dir = workspace("round-trip");
    let wasm = dir.join("policy.wasm");
    let artifact = dir.join("nested/modules/policy.module.wasm");
    fs::write(&wasm, EMPTY_MODULE).expect("the raw module is written");

    let output = tool()
        .args(["--wasm"])
        .arg(&wasm)
        .args(["--out"])
        .arg(&artifact)
        .output()
        .expect("the tool runs");
    assert!(output.status.success(), "{}", stderr(&output));

    let bytes = fs::read(&artifact).expect("the artifact is written");
    let engine = scheduling::Engine::new().expect("this host has a policy engine");
    let module = engine
        .module(&bytes)
        .expect("the main process loads the artifact");
    assert!(
        module
            .instantiate(
                &scheduling::guest::Snapshot::default(),
                &Default::default(),
                &Default::default()
            )
            .is_err(),
        "the artifact carries the policy's exports, and an empty module has none"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A file the tool cannot read, and one that is not a module, each name what
/// failed: a package's build log has to say which of its inputs was wrong.
#[test]
fn an_input_it_cannot_use_names_what_failed() {
    let dir = workspace("failures");
    let missing = dir.join("absent.wasm");
    let output = tool()
        .args(["--wasm"])
        .arg(&missing)
        .args(["--out"])
        .arg(dir.join("out.wasm"))
        .output()
        .expect("the tool runs");
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

    let garbage = dir.join("garbage.wasm");
    fs::write(&garbage, b"not a module at all").expect("the file is written");
    let output = tool()
        .args(["--wasm"])
        .arg(&garbage)
        .args(["--out"])
        .arg(dir.join("out.wasm"))
        .output()
        .expect("the tool runs");
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("cannot precompile"),
        "{}",
        stderr(&output)
    );
    let _ = fs::remove_dir_all(&dir);
}

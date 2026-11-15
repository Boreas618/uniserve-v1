//! The policy a test engine's schedulers instantiate.
//!
//! A deployed engine takes its policy from the Build Server; a test engine has none, and a
//! scheduler with no policy decides nothing (no output, no admission, no batch), so an engine that
//! serves a request must be given a module. [`packaged_policy`] is that module: the real `arrival`
//! package this tree builds, with the arguments its revision runs with — the package's declared
//! defaults ([`packaged_parameters`]), replaced by limits that fit the ceilings of the groups the
//! engine's schedulers serve (`engine::scheduler::policy::ceilings`). Every scheduler instantiates
//! it as it instantiates a deployment's, so a caller's fixtures exercise the mechanism rather than
//! a stand-in for it, and there is no second policy in the tree to keep honest.

use std::path::Path;
use std::sync::Arc;

use ::engine::scheduler::policy;
use anyhow::Result;
use scheduling::{Limits, Parameters};

/// The scheduling policy package this tree builds, relative to the workspace.
const PACKAGE: &str = "crates/scheduling/policies/arrival";

/// The arguments a revision of the tree's policy package runs with when its deployment states
/// none: the defaults its manifest declares, as the Build Server fills them in.
///
/// # Panics
///
/// When the package's manifest cannot be read or declares parameters the policy interface does
/// not name.
pub fn packaged_parameters() -> Parameters {
    let root = Path::new(proto::WORKSPACE).join(PACKAGE);
    let manifest = build_server::manifest::Manifest::load(&root)
        .unwrap_or_else(|error| panic!("reading the policy package {}: {error}", root.display()));
    toml::Value::Table(manifest.parameters.into_iter().collect())
        .try_into()
        .unwrap_or_else(|error| {
            panic!(
                "the policy package {} declares parameters the policy interface does not name: \
                 {error}",
                root.display()
            )
        })
}

/// The arguments the packaged policy runs under in an engine whose groups' captured buckets bound
/// it by `ceilings` (`engine::scheduler::policy::ceilings`): every limit states its ceiling, so the
/// limits the module declares are the ones a scheduler checks a policy against and cannot be
/// refused, and every other parameter is the package's default.
///
/// # Panics
///
/// As [`packaged_parameters`].
pub fn packaged_arguments(ceilings: &Limits) -> Parameters {
    Parameters {
        chunked_prefill_size: Some(ceilings.chunked_prefill_size),
        max_queued_requests: ceilings.max_queued_requests,
        max_batch: Some(ceilings.max_batch),
        max_num_batched_tokens: Some(ceilings.max_num_batched_tokens),
        prefetch_min_tokens: Some(ceilings.prefetch_min_tokens),
        timeout: Some(ceilings.timeout),
        ..packaged_parameters()
    }
}

/// [`packaged_module`] loaded for a scheduler's policy engine, with [`packaged_arguments`] of
/// `ceilings`: the policy `engine::Engine::new` instantiates on every scheduler and, with several,
/// the route.
///
/// # Errors
///
/// When this host has no policy engine, or the module does not load.
///
/// # Panics
///
/// As [`packaged_module`] and [`packaged_parameters`].
pub fn packaged_policy(ceilings: &Limits) -> Result<(Arc<wasmtime::Module>, Parameters)> {
    let module = policy::load(&packaged_module())
        .map_err(|error| anyhow::anyhow!("loading the arrival policy module: {error}"))?;
    Ok((Arc::new(module), packaged_arguments(ceilings)))
}

/// The `arrival` module this tree builds for `wasm32-unknown-unknown`, as a deployment's store
/// serves a policy, made once per process and shared by every fixture that installs a policy.
///
/// `SCHED_POLICY_WASM` names a module to serve with instead, which a caller testing a build of its
/// own sets. Unset, the module is built from this tree's `scheduling-policy-arrival` on first use,
/// into a target directory of its own (`target/it-policy`), so a test run needs no setup step and
/// the build does not contend for the lock of the tree the running tests were built in.
///
/// # Panics
///
/// When `SCHED_POLICY_WASM` names an unreadable file, or the build fails; the message names the
/// command that builds the module.
pub fn packaged_module() -> Vec<u8> {
    static MODULE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    MODULE
        .get_or_init(|| match std::env::var("SCHED_POLICY_WASM") {
            Ok(path) => std::fs::read(&path).unwrap_or_else(|error| {
                panic!("SCHED_POLICY_WASM names `{path}`, which cannot be read: {error}")
            }),
            Err(_) => build_packaged_module(),
        })
        .clone()
}

/// Builds this tree's `arrival` policy package and returns the module bytes.
///
/// The build runs the same `cargo` that started the tests (`CARGO`), so the toolchain and the
/// target the caller selected are the ones the module is built with. A failure states the status
/// and the end of the build's stderr, and names the export that skips the build.
fn build_packaged_module() -> Vec<u8> {
    let workspace = std::path::PathBuf::from(proto::WORKSPACE);
    // The build's own target directory when the caller set one, else Cargo's default one.
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| workspace.join("target"), std::path::PathBuf::from)
        .join("it-policy");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = std::process::Command::new(&cargo)
        .current_dir(&workspace)
        .arg("build")
        .args(["-q", "-p", "scheduling-policy-arrival"])
        .args(["--target", "wasm32-unknown-unknown", "--release"])
        .arg("--target-dir")
        .arg(&target_dir)
        .output()
        .unwrap_or_else(|error| {
            panic!("running {cargo:?} to build the arrival policy module: {error}")
        });

    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail: Vec<&str> = stderr.lines().rev().take(20).collect();
    let tail: Vec<&str> = tail.into_iter().rev().collect();
    assert!(
        output.status.success(),
        "building this tree's arrival policy module failed ({}). Set SCHED_POLICY_WASM to a \
         module built with `cargo build -p scheduling-policy-arrival --target \
         wasm32-unknown-unknown --release` to run without the build. The build's stderr \
         ended:\n{}",
        output.status,
        tail.join("\n"),
    );

    let path = target_dir.join("wasm32-unknown-unknown/release/scheduling_policy_arrival.wasm");
    std::fs::read(&path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ceilings of groups whose buckets hold `calls` calls and prefills of `tokens` tokens.
    fn ceilings(calls: usize, tokens: u32) -> Limits {
        Limits {
            chunked_prefill_size: tokens,
            max_batch: calls,
            max_num_batched_tokens: tokens,
            ..policy::ceilings([])
        }
    }

    /// The arguments state every limit at its ceiling, so the limits the module declares are the
    /// ones the mechanism checks a policy against, and every other parameter at the package's
    /// default.
    #[test]
    fn the_packaged_arguments_state_the_ceilings_and_the_package_defaults() {
        let ceilings = ceilings(8, 1024);
        let arguments = packaged_arguments(&ceilings);
        assert_eq!(arguments.max_batch, Some(ceilings.max_batch));
        assert_eq!(
            arguments.max_num_batched_tokens,
            Some(ceilings.max_num_batched_tokens)
        );
        assert_eq!(
            arguments.chunked_prefill_size,
            Some(ceilings.chunked_prefill_size)
        );
        assert_eq!(
            arguments.prefetch_min_tokens,
            Some(ceilings.prefetch_min_tokens)
        );
        assert_eq!(arguments.timeout, Some(ceilings.timeout));

        let defaults = packaged_parameters();
        assert_eq!(arguments.order, defaults.order);
        assert_eq!(arguments.prefill_window, defaults.prefill_window);
        assert!(defaults.order.is_some() && defaults.prefill_window.is_some());
    }

    /// An engine whose buckets hold two calls is given arguments of two calls, not the package's
    /// default of more, which its ceilings would refuse.
    #[test]
    fn an_engine_of_fewer_calls_states_fewer() {
        let small = packaged_arguments(&ceilings(2, 64));
        assert_eq!(small.max_batch, Some(2));
        assert_eq!(small.max_num_batched_tokens, Some(64));
        assert!(packaged_parameters().max_batch > Some(2));
    }
}

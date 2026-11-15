//! The policy a test engine's schedulers instantiate.
//!
//! A deployed engine takes its policy from the Build Server; a test engine has none, and a
//! scheduler with no policy decides nothing (no output, no admission, no batch), so an engine that
//! serves a request must be given a module. [`packaged_policy`] is that module: the real `fcfs`
//! package this tree builds, with the arguments that make its declared limits fit the ceilings of
//! the groups the engine's schedulers serve (`engine::scheduler::policy::ceilings`). Every scheduler
//! instantiates it as it instantiates a deployment's, so a caller's fixtures exercise the
//! mechanism rather than a stand-in for it, and there is no second policy in the tree to keep
//! honest.

use std::sync::Arc;

use ::engine::scheduler::policy;
use anyhow::Result;
use scheduling::{Limits, Parameters};

/// The arguments that state `limits` as the policy's parameters: every limit a deployment can
/// state.
fn parameters_of(limits: &Limits) -> Parameters {
    Parameters {
        chunked_prefill_size: Some(limits.chunked_prefill_size),
        max_queued_requests: limits.max_queued_requests,
        max_batch: Some(limits.max_batch),
        max_num_batched_tokens: Some(limits.max_num_batched_tokens),
        prefetch_min_tokens: Some(limits.prefetch_min_tokens),
        timeout: Some(limits.timeout),
        ..Parameters::default()
    }
}

/// The arguments `fcfs` runs under in an engine whose groups' captured buckets bound it by
/// `ceilings` (`engine::scheduler::policy::ceilings`): they state the ceilings themselves, so the limits
/// the module declares are the ones a scheduler checks a policy against and cannot be refused.
pub fn packaged_arguments(ceilings: &Limits) -> Parameters {
    parameters_of(ceilings)
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
/// As [`packaged_module`].
pub fn packaged_policy(ceilings: &Limits) -> Result<(Arc<wasmtime::Module>, Parameters)> {
    let module = policy::load(&packaged_module())
        .map_err(|error| anyhow::anyhow!("loading the fcfs policy module: {error}"))?;
    Ok((Arc::new(module), packaged_arguments(ceilings)))
}

/// The `fcfs` module this tree builds for `wasm32-unknown-unknown`, as a deployment's store serves
/// a policy, made once per process and shared by every fixture that installs a policy.
///
/// `SCHED_POLICY_WASM` names a module to serve with instead, which a caller testing a build of its
/// own sets. Unset, the module is built from this tree's `scheduling-policy-fcfs` on first use,
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

/// Builds this tree's `fcfs` policy package and returns the module bytes.
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
        .args(["-q", "-p", "scheduling-policy-fcfs"])
        .args(["--target", "wasm32-unknown-unknown", "--release"])
        .arg("--target-dir")
        .arg(&target_dir)
        .output()
        .unwrap_or_else(|error| {
            panic!("running {cargo:?} to build the fcfs policy module: {error}")
        });

    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail: Vec<&str> = stderr.lines().rev().take(20).collect();
    let tail: Vec<&str> = tail.into_iter().rev().collect();
    assert!(
        output.status.success(),
        "building this tree's fcfs policy module failed ({}). Set SCHED_POLICY_WASM to a module \
         built with `cargo build -p scheduling-policy-fcfs --target wasm32-unknown-unknown \
         --release` to run without the build. The build's stderr ended:\n{}",
        output.status,
        tail.join("\n"),
    );

    let path = target_dir.join("wasm32-unknown-unknown/release/scheduling_policy_fcfs.wasm");
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
            ..Limits::default()
        }
    }

    #[test]
    fn the_packaged_arguments_state_limits_the_ceilings_allow() {
        let ceilings = ceilings(8, 1024);

        // The arguments state the ceilings themselves, so the limits the module declares are the
        // ones the mechanism checks a policy against and cannot be refused for an engine of these
        // ceilings.
        let limits = Limits::from_parameters(&packaged_arguments(&ceilings));
        assert_eq!(limits.max_batch, ceilings.max_batch);
        assert_eq!(
            limits.max_num_batched_tokens,
            ceilings.max_num_batched_tokens
        );
        assert_eq!(limits.chunked_prefill_size, ceilings.chunked_prefill_size);
        assert_eq!(limits.prefetch_min_tokens, ceilings.prefetch_min_tokens);
        assert_eq!(limits.timeout, ceilings.timeout);
    }

    #[test]
    fn an_engine_of_fewer_calls_states_fewer() {
        // `fcfs`'s own default declares 128 calls, which an engine whose buckets hold two would
        // be refused for: the arguments are what make the declaration fit.
        let small = packaged_arguments(&ceilings(2, 64));
        assert_eq!(small.max_batch, Some(2));
        assert_eq!(small.max_num_batched_tokens, Some(64));
        assert_eq!(
            Limits::from_parameters(&small).max_batch,
            2,
            "the module declares what the arguments state, not the package default"
        );
    }
}

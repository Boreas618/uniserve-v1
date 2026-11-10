//! The policy source a test epoch installs.
//!
//! A deployed epoch takes its policy from the Program Server; a test epoch has none, and a
//! scheduler with no policy decides nothing — no plan, no admission, no batch — so an epoch that
//! submits a request must be given a module. [`packaged_policy`] is that source: the real `fcfs`
//! program this tree builds, with the arguments that make its declared limits fit the capacity the
//! caller's epoch was built for. It installs through the same precopy and switchover a deployment
//! uses, so a caller's fixtures exercise the mechanism rather than a stand-in for it, and there is
//! no second policy in the tree to keep honest.

use scheduler::{BatchCapacity, Limits, Parameters, PolicyProgram, ceilings_of};

/// The arguments that state `limits` as a program's parameters: every limit a deployment can
/// state.
fn parameters_of(limits: Limits) -> Parameters {
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

/// The `fcfs` policy module this tree builds, and the program that names it: the source a caller
/// installs through the real precopy and switchover.
///
/// The program is [`packaged_program`]`(capacity)`, so the limits it declares fit any fixture's
/// build config and the mechanism's check of a policy's limits against the ceilings cannot refuse
/// it. The module is [`packaged_module`].
pub fn packaged_policy(capacity: BatchCapacity) -> (PolicyProgram, Vec<u8>) {
    (packaged_program(capacity), packaged_module())
}

/// The program `fcfs` revision one runs under for an epoch of `capacity`: arguments that state
/// [`ceilings_of`]`(capacity)`, the ceilings the mechanism checks a policy's declaration against.
pub fn packaged_program(capacity: BatchCapacity) -> PolicyProgram {
    PolicyProgram {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: parameters_of(ceilings_of(capacity)),
        stated: std::collections::BTreeMap::new(),
    }
}

/// The `fcfs` module this tree builds for `wasm32-unknown-unknown`, read once per process and
/// shared by every fixture that installs a policy.
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

    /// The capacity of a pool the tests build their scheduler for.
    fn capacity() -> BatchCapacity {
        BatchCapacity {
            max_batch: 8,
            max_num_batched_tokens: 1024,
        }
    }

    #[test]
    fn the_packaged_program_states_limits_the_build_config_allows() {
        let program = packaged_program(capacity());
        assert_eq!(program.backend, "fcfs", "the module is this tree's fcfs");
        assert_eq!(program.revision, 1);
        assert!(
            program.stated.is_empty(),
            "no argument is stated, so a Program Server resolves none for it"
        );

        // The arguments state the ceilings themselves, so the limits the module declares are the
        // ones the mechanism checks a policy against and cannot be refused for an epoch of this
        // capacity.
        let limits = Limits::from_parameters(&program.arguments);
        let ceilings = ceilings_of(capacity());
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
    fn an_epoch_built_for_fewer_calls_states_fewer() {
        // `fcfs`'s own default declares 128 calls, which a test epoch built for two would be
        // refused for: the program's arguments are what make the declaration fit.
        let small = packaged_program(BatchCapacity {
            max_batch: 2,
            max_num_batched_tokens: 64,
        });
        assert_eq!(small.arguments.max_batch, Some(2));
        assert_eq!(small.arguments.max_num_batched_tokens, Some(64));
        assert_eq!(
            Limits::from_parameters(&small.arguments).max_batch,
            2,
            "the module declares what this program states, not the package default"
        );
    }
}

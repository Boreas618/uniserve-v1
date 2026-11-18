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
//!
//! The policy packages a test runs are the workspace's own ([`package`]), built by the cargo that
//! built the tests ([`module`]): the engine's policy host tests the packaged policy, the engine's
//! start test installs it into a Build Server, and the fake engine serves it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

use ::engine::scheduler::policy;
use anyhow::Result;
use scheduling::{Limits, Parameters};

/// The arguments a revision of the tree's policy package runs with when its deployment states
/// none: the defaults its manifest declares, as the Build Server fills them in.
///
/// # Panics
///
/// When the package's manifest cannot be read or declares parameters the policy interface does
/// not name.
pub fn packaged_parameters() -> Parameters {
    let root = package(proto::config::DEFAULT_SCHEDULING);
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

/// The arguments the packaged policy runs under in an engine whose policy budget bounds its limits
/// by `ceilings` (`engine::scheduler::policy::ceilings`): every limit states its bound, so the
/// limits the module declares are the ones a scheduler checks a policy against and cannot be
/// refused, and every other parameter is the package's default.
///
/// # Panics
///
/// As [`packaged_parameters`].
pub fn packaged_arguments(ceilings: &Limits) -> Parameters {
    Parameters {
        prefetch_min_tokens: Some(ceilings.prefetch_min_tokens),
        timeout: Some(ceilings.timeout),
        ..packaged_parameters()
    }
}

/// [`packaged_module`] loaded for a scheduler's policy engine, with [`packaged_arguments`] of
/// `ceilings`: the policy `engine::Engine::new` instantiates on every scheduler.
///
/// The module is compiled once per test process and shared: compiling it is the costly step (a
/// debug build takes seconds of a core), and an engine instantiates the module it is given on
/// the module's own engine, so every fake engine of a suite can share one compilation, as every
/// scheduler of a deployment shares the module its store served.
///
/// # Errors
///
/// When this host has no policy engine, or the module does not load.
///
/// # Panics
///
/// As [`packaged_module`] and [`packaged_parameters`].
pub fn packaged_policy(ceilings: &Limits) -> Result<(Arc<wasmtime::Module>, Parameters)> {
    static MODULE: OnceLock<std::result::Result<Arc<wasmtime::Module>, String>> = OnceLock::new();

    let module = MODULE
        .get_or_init(|| {
            policy::load(&packaged_module())
                .map(Arc::new)
                .map_err(|error| format!("loading the arrival policy module: {error}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)?;
    Ok((module, packaged_arguments(ceilings)))
}

/// The module of the workspace's default policy package ([`module`]), as a deployment's store
/// serves a policy.
///
/// # Panics
///
/// As [`module`].
pub fn packaged_module() -> Vec<u8> {
    module(proto::config::DEFAULT_SCHEDULING)
}

/// The directory of the workspace's scheduling policy package whose backend is `backend`: the
/// workspace member whose `package.toml` states the scheduling operator and that backend, as the
/// Build Server reads a package from its directory. A test runs in its package's directory, from
/// which cargo finds the workspace.
///
/// # Panics
///
/// When `cargo metadata` fails, or no member is that package.
pub fn package(backend: &str) -> PathBuf {
    member(backend).0
}

/// The wasm32 binary of the workspace's policy package `backend` ([`package`]), built once per
/// test process by the cargo that built the tests, into a target directory of its own beside the
/// workspace's (`it-policy`, so the build does not wait for the lock of the build that runs the
/// tests), and read from the artifact cargo's JSON messages name.
///
/// # Panics
///
/// When the package is not found, or its build fails or reports no wasm artifact.
#[expect(clippy::unwrap_used, reason = "test support fails the test that asks")]
pub fn module(backend: &str) -> Vec<u8> {
    static MODULES: Mutex<BTreeMap<String, Vec<u8>>> = Mutex::new(BTreeMap::new());

    // The lock is held through a build, so one test process builds each package once.
    let mut modules = MODULES.lock().unwrap();
    if let Some(module) = modules.get(backend) {
        return module.clone();
    }

    let (_, id, target) = member(backend);
    let build = Command::new(env!("CARGO"))
        .args(["build", "--release", "--locked", "--offline", "--lib"])
        .args(["--message-format=json-render-diagnostics", "--target"])
        .arg(build_server::target::WASM)
        .arg("--package")
        .arg(&id)
        .arg("--target-dir")
        .arg(target.join("it-policy"))
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "building the policy package {backend}: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let artifact = build
        .stdout
        .split(|byte| *byte == b'\n')
        .filter_map(|line| serde_json::from_slice::<serde_json::Value>(line).ok())
        .filter(|message| message["reason"] == "compiler-artifact" && message["package_id"] == id)
        .flat_map(|message| message["filenames"].as_array().cloned().unwrap_or_default())
        .filter_map(|file| file.as_str().map(PathBuf::from))
        .find(|file| {
            file.extension()
                .is_some_and(|extension| extension == "wasm")
        })
        .unwrap_or_else(|| panic!("cargo reported no wasm artifact of {backend}"));
    let module = std::fs::read(artifact).unwrap();
    modules.insert(backend.to_owned(), module.clone());
    module
}

/// The directory, the cargo package id and the workspace's target directory of the policy
/// package `backend` ([`package`]).
#[expect(clippy::unwrap_used, reason = "test support fails the test that asks")]
fn member(backend: &str) -> (PathBuf, String, PathBuf) {
    let metadata = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
        ])
        .output()
        .unwrap();
    assert!(
        metadata.status.success(),
        "cargo metadata: {}",
        String::from_utf8_lossy(&metadata.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&metadata.stdout).unwrap();

    let (directory, id) = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|package| {
            let directory = Path::new(package["manifest_path"].as_str()?).parent()?;
            let manifest = build_server::manifest::Manifest::load(directory).ok()?;
            if manifest.op != proto::module::PROGRAM || manifest.backend != backend {
                return None;
            }
            Some((directory.to_path_buf(), package["id"].as_str()?.to_owned()))
        })
        .unwrap_or_else(|| panic!("no workspace member is the policy package {backend}"));
    let target = PathBuf::from(metadata["target_directory"].as_str().unwrap());
    (directory, id, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arguments state every limit at its bound, so the limits the module declares are the
    /// ones the mechanism checks a policy against, and every other parameter at the package's
    /// default.
    #[test]
    fn the_packaged_arguments_state_the_ceilings_and_the_package_defaults() {
        let ceilings = policy::ceilings([], &proto::config::PolicyBudget::default()).limits;
        let arguments = packaged_arguments(&ceilings);
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
}

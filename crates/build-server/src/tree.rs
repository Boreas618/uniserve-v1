//! A UniServe source tree, as a Build Server started with `--source` reads it.
//!
//! The tree holds two things a Build Server serves a deployment with:
//!
//! - the Build Server's Python environment, the locked project
//!   [`ENVIRONMENT_PROJECT`], which uv creates when the server starts
//!   ([`crate::toolchain::Environment`]);
//! - the serving packages of the tree's release ([`packages`]): every kernel
//!   package of the operator tree (`crates/ops/kernels/<op>/<backend>`), the
//!   scheduling policy and the engine. The upstream server installs them when
//!   it starts ([`crate::server::BuildServer::install_tree`]), so a deployment
//!   selects from the tree it was started with.

use std::path::{Path, PathBuf};

use crate::Manifest;
use crate::Result;

/// The Build Server's Python project, relative to the tree's root.
pub const ENVIRONMENT_PROJECT: &str = "crates/build-server/python";

/// The operator tree: one directory per operator, one package directory per
/// backend.
const KERNEL_TREE: &str = "crates/ops/kernels";

/// The scheduling policy package.
const POLICY_PACKAGE: &str = "crates/scheduling/policies/arrival";

/// The engine package.
const ENGINE_PACKAGE: &str = "crates/engine";

/// The serving packages of the tree at `source`, in a stable order: the kernel
/// packages by operator and backend, then the scheduling policy and the engine.
///
/// # Errors
///
/// Fails when the operator tree cannot be read.
pub fn packages(source: &Path) -> Result<Vec<PathBuf>> {
    let mut kernels = Vec::new();
    for op in std::fs::read_dir(source.join(KERNEL_TREE))? {
        let op = op?.path();
        if !op.is_dir() {
            continue;
        }
        for backend in std::fs::read_dir(&op)? {
            let backend = backend?.path();
            if backend.join(Manifest::FILE).is_file() {
                kernels.push(backend);
            }
        }
    }
    kernels.sort();

    kernels.push(source.join(POLICY_PACKAGE));
    kernels.push(source.join(ENGINE_PACKAGE));
    Ok(kernels)
}

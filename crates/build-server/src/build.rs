//! Building a module store ahead of serving: what `uniserve build-server
//! build` does.
//!
//! A store is built from two kinds of input. A key list names one
//! `operator<TAB>descriptor key` per line, the kernel keys a deployment
//! launches: a serving deployment's kernel inventory, the `op` and
//! `descriptor` labels of its `uniserve_kernel_launches_total` rows in the
//! observation store (`/admin/query`), states them. For every
//! operator the lists name, each package under `<packages>/<operator>/` is
//! installed into the store's package root, every key a package's manifest
//! covers is prepared for one target, and each key is resolved to the module
//! it names. A key is built when one installed backend resolves it; a second
//! backend that cannot is not a failure, because the operator is served. A
//! package that fills a whole interface with one module, a scheduling policy
//! or the engine, is named by its directory: it is installed and its one
//! module is built for its own target.
//!
//! The store is a directory with the two roots a Build Server serves:
//!
//! ```text
//! <store>/packages   the installed package revisions (`--package-root`)
//! <store>/modules    the committed modules (`--store-root`)
//! ```
//!
//! Such a directory is what a release ships: a Build Server given it as a
//! read-only root serves its packages and their modules with no toolchain, so
//! a first start builds nothing.
//!
//! Building into an existing store extends it. A package whose build inputs
//! (its files, and the toolchain with its compile driver and pins) are
//! unchanged since its latest installed revision keeps that revision
//! ([`BuildServer::install_if_changed`]), so modules built earlier stay the ones the
//! store serves, and only a package of changed inputs is installed as a new
//! revision, whose modules are built again.
//!
//! The builds are real: nvcc, Triton, CuTe DSL and cargo run with the host's
//! toolchains, so a build of a deployment takes minutes. The build runs a
//! Build Server in process, which holds the store's package root for the
//! build's length: a second build of the same store, or a server that installs
//! into it, is refused before it installs anything.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::manifest::SINGLE_KEY;
use crate::protocol::{Descriptors, PackageRequest, PrepareRequest, Selector};
use crate::server::{BuildServer, ServerConfig};
use crate::target::Target;
use crate::{Error, Manifest, Result};
use proto::module::ModuleSlot;

/// What a build is told of each key once its package is done: the operator,
/// the key, and the backends that built it with the number of files each
/// module holds. No backend means the key did not build.
pub type Report<'a> = dyn FnMut(&str, &str, &[(String, usize)]) + 'a;

/// The directory of a store that holds its installed package revisions.
pub const PACKAGES: &str = "packages";

/// The directory of a store that holds its committed modules.
pub const MODULES: &str = "modules";

/// Reads key lists: `operator<TAB>key` per line, blank lines and `#` comments
/// skipped. Returns each line's operator and key.
///
/// # Errors
///
/// Fails when a file cannot be read or a line is not `operator<TAB>key`. A
/// third field is refused rather than read as part of the key, which would key
/// a module by a name no launch asks for.
pub fn read_keys(paths: &[PathBuf]) -> Result<Vec<(String, String)>> {
    let invalid = |path: &Path, message: String| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: {message}", path.display()),
        ))
    };

    let mut keys = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(path).map_err(|error| {
            Error::Io(std::io::Error::new(
                error.kind(),
                format!("reading {}: {error}", path.display()),
            ))
        })?;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((op, key)) = line.split_once('\t') else {
                return Err(invalid(
                    path,
                    format!("a key line is `operator<TAB>key`, not {line:?}"),
                ));
            };
            if key.contains('\t') {
                return Err(invalid(
                    path,
                    format!("{op:?} states a third field; a key line is `operator<TAB>key`"),
                ));
            }
            keys.push((op.trim().to_owned(), key.trim().to_owned()));
        }
    }
    Ok(keys)
}

/// Builds a store in the roots `config` names: `keys`, each an operator and
/// one of its kernel keys, for `target` from the packages under `packages`,
/// and the one module of each package directory in `single`. Each key is
/// reported to `built` once its package is done, as the operator, the key and
/// the backends that built it with the number of files each module holds; no
/// backend means the key did not build. Returns one line per failure: a
/// failed prepare or resolve, and each key no backend built.
///
/// `config` is a Build Server's configuration whose package root and first
/// store root are the store's, and whose targets include `target`. `packages`
/// holds one directory per operator, with one package directory per backend.
/// Every path must be absolute: a build runs in the installed copy of its
/// package.
///
/// # Errors
///
/// Fails when the server cannot open, or a package tree cannot be read or
/// installed. A key that does not build is not an error here: it is reported
/// with no backend and among the failures.
pub fn build(
    config: &ServerConfig,
    packages: &Path,
    keys: &[(String, String)],
    target: Option<&Target>,
    single: &[PathBuf],
    built: &mut Report<'_>,
) -> Result<Vec<String>> {
    let server = BuildServer::open(config)?;
    let mut failures = build_kernels(&server, packages, keys, target, built)?;
    failures.extend(build_single(&server, single, built)?);
    Ok(failures)
}

/// Installs, prepares and resolves every kernel key on a running server.
fn build_kernels(
    server: &BuildServer,
    packages: &Path,
    keys: &[(String, String)],
    target: Option<&Target>,
    built: &mut Report<'_>,
) -> Result<Vec<String>> {
    let mut failures = Vec::new();
    let Some(target) = target else {
        // Kernel modules are built for a device's architecture, which a build
        // of single-module packages alone does not name.
        failures.extend(
            keys.iter()
                .map(|(op, key)| format!("{op} {key}: the build names no target for kernel keys")),
        );
        return Ok(failures);
    };

    let mut ops: Vec<&str> = keys.iter().map(|(op, _)| op.as_str()).collect();
    ops.sort_unstable();
    ops.dedup();

    for op in ops {
        let backends = install(server, &packages.join(op))?;
        let op_keys: Vec<&str> = keys
            .iter()
            .filter(|(key_op, _)| key_op == op)
            .map(|(_, key)| key.as_str())
            .collect();

        // Each key's backends, with the number of files each one's module
        // holds.
        let mut resolved: BTreeMap<String, Vec<(String, usize)>> = BTreeMap::new();
        for (backend, manifest) in &backends {
            // A backend is asked only for the keys its own manifest covers: an
            // operator with several backends serves a key through whichever
            // one declares it, and a backend that never claimed a key is not a
            // failure when it cannot build it.
            let covered: Vec<String> = op_keys
                .iter()
                .filter(|key| manifest.covers(key))
                .map(|key| (*key).to_owned())
                .collect();
            if covered.is_empty() {
                continue;
            }
            for (key, files) in build_package(server, op, backend, &covered, target, &mut failures)
            {
                resolved
                    .entry(key)
                    .or_default()
                    .push((backend.clone(), files));
            }
        }

        for key in op_keys {
            let backends_built = resolved.remove(key).unwrap_or_default();
            if backends_built.is_empty() {
                let why = if backends.is_empty() {
                    "no package fills the operator"
                } else {
                    "no backend built it"
                };
                failures.push(format!("{op} {key}: {why}"));
            }
            built(op, key, &backends_built);
        }
    }
    Ok(failures)
}

/// Installs each package directory of `single` and builds its one module.
fn build_single(
    server: &BuildServer,
    single: &[PathBuf],
    built: &mut Report<'_>,
) -> Result<Vec<String>> {
    let mut failures = Vec::new();
    for directory in single {
        let (backend, manifest) = install_package(server, directory)?;
        let modules = build_package(
            server,
            &manifest.op,
            &backend,
            &[SINGLE_KEY.to_owned()],
            &manifest.single_target(),
            &mut failures,
        );
        let backends_built: Vec<(String, usize)> = modules
            .into_iter()
            .map(|(_, files)| (backend.clone(), files))
            .collect();
        if backends_built.is_empty() {
            failures.push(format!(
                "{} {SINGLE_KEY}: its package did not build",
                manifest.op
            ));
        }
        built(&manifest.op, SINGLE_KEY, &backends_built);
    }
    Ok(failures)
}

/// Prepares `keys` of the latest revision of `op`'s `backend` for `target` and
/// resolves each one: returns the keys that resolved, each with the number of
/// files its module holds, and adds a line to `failures` for what did not.
fn build_package(
    server: &BuildServer,
    op: &str,
    backend: &str,
    keys: &[String],
    target: &Target,
    failures: &mut Vec<String>,
) -> Vec<(String, usize)> {
    let request = PrepareRequest {
        targets: vec![target.clone()],
        packages: vec![PackageRequest {
            slot: ModuleSlot::all(op),
            selector: Selector::Latest {
                backend: backend.to_owned(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            slot: ModuleSlot::all(op),
            keys: keys.to_vec(),
        }],
    };
    // A prepare that fails names its first failed build; the backend's keys
    // are then reported unbuilt rather than built again one by one by
    // `resolve`.
    let revision = match server.prepare(&request, &mut |_| {}) {
        Ok(prepared) => prepared[0].1.clone(),
        Err(error) => {
            failures.push(format!("{op} {backend}: prepare: {error}"));
            return Vec::new();
        }
    };

    let mut resolved = Vec::new();
    for key in keys {
        match server.resolve(op, &revision, target, key, &mut |_| {}) {
            Ok(module) => resolved.push((key.clone(), module.files.len())),
            Err(error) => failures.push(format!("{op} {backend} {key}: resolve: {error}")),
        }
    }
    resolved
}

/// Makes every package of one operator installed, and returns each one's
/// backend and manifest.
///
/// A package whose files are those of its latest installed revision keeps that
/// revision; any other is installed as a new one. The package tree states the
/// operators it has packages for as its directories: an operator it has no
/// directory for has no package, and its keys are reported unbuilt.
///
/// # Errors
///
/// Fails when the directory cannot be read or a package cannot be installed:
/// the tree, not one key, is then at fault.
fn install(server: &BuildServer, directory: &Path) -> Result<Vec<(String, Manifest)>> {
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut packages: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() && path.join(Manifest::FILE).is_file() {
            packages.push(path);
        }
    }
    // Backends in name order, so the report reads the same on every host.
    packages.sort();

    packages
        .iter()
        .map(|package| install_package(server, package))
        .collect()
}

/// Makes the package in `directory` installed, at its latest revision when
/// that one was installed from the same build inputs, and returns its backend
/// and manifest.
fn install_package(server: &BuildServer, directory: &Path) -> Result<(String, Manifest)> {
    let installed = server.install_if_changed(directory)?;
    let manifest = Manifest::load(directory).map_err(|source| Error::Manifest {
        package: directory.to_path_buf(),
        source: Box::new(source),
    })?;
    Ok((installed.backend, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key list states one `operator<TAB>key` per line, in the order written; blank lines and
    /// `#` comments are skipped, surrounding whitespace is trimmed, and several lists read as
    /// their lines in turn.
    #[test]
    fn a_key_list_reads_each_line_as_an_operator_and_key() {
        let directory = tempfile::tempdir().expect("a scratch directory");
        let first = directory.path().join("first.txt");
        let second = directory.path().join("second.txt");
        std::fs::write(
            &first,
            "# kernel inventory\n\
             rms_norm\tdtype=bf16;hidden=4096\n\
             \n\
             attention\thead_dim=128  \n",
        )
        .expect("the first list is written");
        std::fs::write(&second, "rms_norm\tdtype=bf16;hidden=2048\n")
            .expect("the second list is written");

        let read = read_keys(&[first, second]).expect("the key lists read");
        assert_eq!(
            read,
            [
                ("rms_norm", "dtype=bf16;hidden=4096"),
                ("attention", "head_dim=128"),
                ("rms_norm", "dtype=bf16;hidden=2048"),
            ]
            .map(|(op, key)| (op.to_owned(), key.to_owned()))
        );
    }

    /// A line that is not one operator and one key - no tab, or a third field - is refused
    /// rather than read as a key no launch asks for.
    #[test]
    fn a_line_of_no_operator_and_key_is_refused() {
        let directory = tempfile::tempdir().expect("a scratch directory");
        for line in ["rms_norm dtype=bf16", "rms_norm\tdtype=bf16\t96"] {
            let path = directory.path().join("keys.txt");
            std::fs::write(&path, format!("{line}\n")).expect("the key list is written");
            read_keys(&[path]).expect_err("the line is refused");
        }
    }
}

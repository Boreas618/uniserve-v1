//! Building a module store from descriptor key lists: what
//! `uniserve program-server build` does.
//!
//! A key list names one `operator<TAB>descriptor key` per line, the
//! descriptors a deployment launches; `uniserve programs` writes a
//! deployment's ([`key_list`]). For every operator the lists name, each
//! package under `<packages>/<operator>/` is installed into the store's
//! package root, every key a package's manifest covers is prepared for one
//! target, and each key is resolved to the module it names. A key is built
//! when one installed backend resolves it; a second backend that cannot is
//! not a failure, because the operator is served.
//!
//! The store is a directory with the two roots a Program Server serves:
//!
//! ```text
//! <store>/packages   the installed package revisions (`--package-root`)
//! <store>/modules    the committed modules (`--store-root`)
//! ```
//!
//! Building into an existing store extends it. A package whose files are
//! unchanged since its latest installed revision keeps that revision
//! ([`ProgramServer::installed`]), so modules built earlier stay under the
//! revision the store serves, and only a changed package is installed as a
//! new revision.
//!
//! The builds are real: nvcc, Triton, CuTe DSL and wasm run with the host's
//! toolchains, so a build of a deployment takes minutes. The server is driven
//! as its own process would be, by a serving thread on the socket its
//! configuration names, and a build whose socket another server owns is
//! refused before it installs anything.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::program::ProgramSlot;
use crate::protocol::{Descriptors, PrepareRequest, ProgramRequest, Selector};
use crate::server::{ProgramServer, ServerConfig};
use crate::target::Target;
use crate::{Error, Manifest, Result};

/// One line of a key list: an operator and one of its descriptor keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key {
    pub op: String,
    pub key: String,
}

/// What became of one key: the backends that built it, with the number of
/// files each module holds. No backend means the key did not build.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub key: Key,
    pub backends: Vec<(String, usize)>,
}

/// What a build did: every key's outcome, and every failure behind the keys
/// that did not build, in the order they happened.
#[derive(Debug, Default)]
pub struct Report {
    pub outcomes: Vec<Outcome>,
    /// One line per failed install, prepare or resolve, and per key no
    /// backend built. A key can fail in more than one backend.
    pub failures: Vec<String>,
}

impl Report {
    /// The keys no backend built.
    pub fn unbuilt(&self) -> Vec<&Key> {
        self.outcomes
            .iter()
            .filter(|outcome| outcome.backends.is_empty())
            .map(|outcome| &outcome.key)
            .collect()
    }
}

/// Reads key lists: `operator<TAB>key` per line, blank lines and `#` comments
/// skipped.
///
/// # Errors
///
/// Fails when a file cannot be read or a line is not `operator<TAB>key`. A
/// third field is refused rather than read as part of the key, which would key
/// a module by a name no launch asks for.
pub fn read_keys(paths: &[PathBuf]) -> Result<Vec<Key>> {
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
            keys.push(Key {
                op: op.trim().to_owned(),
                key: key.trim().to_owned(),
            });
        }
    }
    Ok(keys)
}

/// Writes `keys`, descriptor keys by operator name, as a key list [`read_keys`] reads: one
/// `operator<TAB>key` line per key, sorted by operator and then by key, each once.
///
/// # Errors
///
/// Fails when an operator or a key is empty, holds a tab or a line break, or has surrounding
/// whitespace: no line states it so that it reads back as written.
pub fn key_list(keys: &BTreeMap<String, BTreeSet<String>>) -> Result<String> {
    let unwritable = |field: &str| {
        field.is_empty() || field.contains(['\t', '\n', '\r']) || field.trim() != field
    };
    let mut list = String::new();
    for (op, keys) in keys {
        for key in keys {
            if unwritable(op) || unwritable(key) {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{op:?} {key:?} cannot be written as an `operator<TAB>key` line"),
                )));
            }
            // Writing to a `String` cannot fail.
            let _ = writeln!(list, "{op}\t{key}");
        }
    }
    Ok(list)
}

/// Builds `keys` for `target` from the packages under `packages` into the
/// roots `config` names, reporting each key's outcome to `report` once its
/// operator is done.
///
/// `config` is a Program Server's configuration whose package root and first
/// store root are the store's, and whose devices include `target`; its socket
/// is where the build's own server answers while it runs. `packages` holds one
/// directory per operator, with one package directory per backend. Every path
/// must be absolute: a build runs in the installed copy of its package.
///
/// # Errors
///
/// Fails when the server cannot open or serve, or the package tree cannot be
/// read. A key that does not build is not an error here: it is an outcome with
/// no backend, and [`Report::unbuilt`] lists it.
pub fn build(
    config: &ServerConfig,
    packages: &Path,
    keys: &[Key],
    target: &Target,
    report: &mut dyn FnMut(&Outcome),
) -> Result<Report> {
    // The socket is bound before anything is installed, so a second build or
    // a server already answering for this store stops this one before it
    // writes a revision.
    let Some((listener, lock)) = service::bind(&config.socket)? else {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::AddrInUse,
            format!(
                "a program server answers on {}: the store is in use",
                config.socket.display()
            ),
        )));
    };
    let server = Arc::new(ProgramServer::open(config)?);

    // The compile queue is driven while the server answers, so the server runs
    // as its own process would: one thread serving, this one asking it for
    // modules.
    let stop = Arc::new(AtomicBool::new(false));
    let serving = std::thread::spawn({
        let server = Arc::clone(&server);
        let stop = Arc::clone(&stop);
        move || server.serve_on(listener, lock, stop)
    });

    let built = build_keys(&server, packages, keys, target, report);

    stop.store(true, Ordering::SeqCst);
    let served = serving
        .join()
        .unwrap_or_else(|_| Err(Error::Protocol("the build's server panicked".into())));
    let built = built?;
    served?;
    Ok(built)
}

/// Installs, prepares and resolves every key on a running server.
fn build_keys(
    server: &ProgramServer,
    packages: &Path,
    keys: &[Key],
    target: &Target,
    report: &mut dyn FnMut(&Outcome),
) -> Result<Report> {
    let mut built = Report::default();
    let mut ops: Vec<&str> = keys.iter().map(|key| key.op.as_str()).collect();
    ops.sort_unstable();
    ops.dedup();

    for op in ops {
        let backends = install(server, &packages.join(op))?;
        let op_keys: Vec<&Key> = keys.iter().filter(|key| key.op == op).collect();

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
                .filter(|key| manifest.covers(&key.key))
                .map(|key| key.key.clone())
                .collect();
            if covered.is_empty() {
                continue;
            }

            let request = PrepareRequest {
                targets: vec![target.clone()],
                programs: vec![ProgramRequest {
                    slot: ProgramSlot::all(op),
                    selector: Selector::Latest {
                        backend: backend.clone(),
                        arguments: toml::Table::new(),
                    },
                }],
                descriptors: vec![Descriptors {
                    op: op.to_owned(),
                    keys: covered.clone(),
                }],
            };
            // A prepare that fails names its first failed build; the backend's
            // keys are then reported unbuilt rather than built again one by
            // one by `resolve`.
            let program = match server.prepare(&request, &mut |_| {}) {
                Ok(prepared) => prepared.programs[0].program.clone(),
                Err(error) => {
                    built
                        .failures
                        .push(format!("{op} {backend}: prepare: {error}"));
                    continue;
                }
            };

            for key in &covered {
                match server.resolve(op, &program, target, key, &mut |_| {}) {
                    Ok(module) => resolved
                        .entry(key.clone())
                        .or_default()
                        .push((backend.clone(), module.files.len())),
                    Err(error) => built
                        .failures
                        .push(format!("{op} {backend} {key}: resolve: {error}")),
                }
            }
        }

        for key in op_keys {
            let backends_built = resolved.remove(key.key.as_str()).unwrap_or_default();
            if backends_built.is_empty() {
                let why = if backends.is_empty() {
                    "no package fills the operator"
                } else {
                    "no backend built it"
                };
                built.failures.push(format!("{op} {}: {why}", key.key));
            }
            let outcome = Outcome {
                key: key.clone(),
                backends: backends_built,
            };
            report(&outcome);
            built.outcomes.push(outcome);
        }
    }
    Ok(built)
}

/// Makes every package of one operator installed, and answers each one's
/// backend and manifest.
///
/// A package whose files are those of its latest installed revision keeps that
/// revision; any other is installed as a new one. An operator with no
/// directory has no package, and its keys are reported unbuilt.
///
/// # Errors
///
/// Fails when the directory cannot be read or a package cannot be installed:
/// the tree, not one key, is then at fault.
fn install(server: &ProgramServer, directory: &Path) -> Result<Vec<(String, Manifest)>> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(Error::Io(std::io::Error::new(
                error.kind(),
                format!("reading {}: {error}", directory.display()),
            )));
        }
    };
    let mut packages: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() && path.join("package.toml").is_file() {
            packages.push(path);
        }
    }
    // Backends in name order, so the report reads the same on every host.
    packages.sort();

    let mut backends = Vec::with_capacity(packages.len());
    for package in packages {
        let installed = match server.installed(&package)? {
            Some(installed) => installed,
            None => server.install(&package)?,
        };
        let manifest = Manifest::load(&package).map_err(|source| Error::Manifest {
            package: package.clone(),
            source: Box::new(source),
        })?;
        backends.push((installed.package.backend, manifest));
    }
    Ok(backends)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(entries: &[(&str, &[&str])]) -> BTreeMap<String, BTreeSet<String>> {
        entries
            .iter()
            .map(|(op, keys)| {
                (
                    (*op).to_owned(),
                    keys.iter().map(|key| (*key).to_owned()).collect(),
                )
            })
            .collect()
    }

    /// A written key list states every key once, sorted by operator and key, and reads back as
    /// the keys it was written from.
    #[test]
    fn a_key_list_reads_back_as_written() {
        let written = key_list(&keys(&[
            (
                "rms_norm",
                &["dtype=bf16;hidden=4096", "dtype=bf16;hidden=2048"],
            ),
            ("attention", &["head_dim=128"]),
        ]))
        .expect("the keys are writable");
        assert_eq!(
            written,
            "attention\thead_dim=128\n\
             rms_norm\tdtype=bf16;hidden=2048\n\
             rms_norm\tdtype=bf16;hidden=4096\n"
        );

        let directory = tempfile::tempdir().expect("a scratch directory");
        let path = directory.path().join("keys.txt");
        std::fs::write(&path, &written).expect("the key list is written");
        let read: Vec<(String, String)> = read_keys(&[path])
            .expect("the key list reads")
            .into_iter()
            .map(|key| (key.op, key.key))
            .collect();
        assert_eq!(
            read,
            [
                ("attention", "head_dim=128"),
                ("rms_norm", "dtype=bf16;hidden=2048"),
                ("rms_norm", "dtype=bf16;hidden=4096"),
            ]
            .map(|(op, key)| (op.to_owned(), key.to_owned()))
        );
    }

    /// A field no line states as it is - a tab, a line break, surrounding whitespace or nothing -
    /// is refused rather than written as another key.
    #[test]
    fn a_key_no_line_states_is_refused() {
        for (op, key) in [
            ("rms_norm", "a\tb"),
            ("rms_norm", "a\nb"),
            ("rms_norm", " a"),
            ("rms_norm", ""),
            ("rms\tnorm", "a"),
        ] {
            key_list(&keys(&[(op, &[key])])).expect_err("the key is refused");
        }
    }
}

//! Fixing a policy package's programs at the deployment's Build Server: what an apply does for
//! `[programs.<name>]` before the rest of the module set (`specs/configuration.md` §3, §4).
//!
//! A crate of the package (`programs/<dir>`) is installed as the program package
//! `programs.<dir>`, a revision of its own whenever its files change; an installed package
//! (`uniserve:<package>`) is fixed to its latest revision. Either revision's module is the wasm32
//! module the Build Server builds and resolves, which states its kind (`program::kind`). The
//! fixed revisions and module digests go into the revision's `lock.toml` with the module set, so
//! a restart of the revision runs the same programs and never resolves `latest` again.
//!
//! A package runs any number of control programs and at most one scheduling, one routing and one
//! KV program; a control or KV program states its `period`. A processing program is no package's:
//! an epoch's module set fixes it beside its engine (`ModuleSet::processing`).

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use program::Kind;
use proto::module::PackageRevision;
use sha2::{Digest as _, Sha256};

use super::Loaded;
use crate::config::program::Source;
use crate::policy::Package;

/// The package's programs as an apply fixed them: each as the module set records it, and each
/// with its module.
#[derive(Debug, Clone, Default)]
pub(crate) struct Fixed {
    pub(crate) programs: BTreeMap<String, proto::module::Program>,
    pub(crate) loaded: Vec<Loaded>,
}

/// Fixes the programs `package` lists at the Build Server at `socket` and loads their modules.
/// A program `locked` records under its name with the same source runs the revision recorded
/// there: the one its revision's apply fixed.
///
/// Blocks on the Build Server, which may build the modules first.
///
/// # Errors
///
/// Fails when the Build Server cannot be reached, refuses to install a crate or holds no such
/// package, a module does not build or states no kind of program, or the programs break a
/// package's rules: more than one scheduling, routing or KV program, or a control or KV program
/// without a `period`.
pub(crate) fn fix(
    socket: &Path,
    package: &Package,
    locked: &BTreeMap<String, proto::module::Program>,
) -> anyhow::Result<Fixed> {
    let mut fixed = Fixed::default();
    if package.config.programs.is_empty() {
        return Ok(fixed);
    }
    let mut client = build_server::Client::connect(socket)
        .with_context(|| format!("connecting to the Build Server at {}", socket.display()))?;

    for (name, entry) in &package.config.programs {
        let source = entry.source.to_string();
        let (backend, revision) = match (locked.get(name), &entry.source) {
            (Some(locked), _) if locked.source == source => {
                (locked.backend.clone(), locked.revision)
            }
            (_, Source::Package(directory)) => {
                let installed = client
                    .install_program(directory, &package.crate_files(directory))
                    .with_context(|| format!("installing `programs.{name}` ({source})"))?;
                (installed.backend, installed.revision)
            }
            (_, Source::Installed(backend)) => {
                let mut latest = PackageRevision {
                    backend: backend.clone(),
                    revision: 0,
                    arguments: toml::Table::new(),
                };
                crate::engine::control::fix_revision(
                    &mut client,
                    proto::module::PROGRAM,
                    &mut latest,
                )
                .with_context(|| format!("fixing `programs.{name}` ({source})"))?;
                (latest.backend, latest.revision)
            }
        };

        let bytes = module(&mut client, &backend, revision)
            .with_context(|| format!("resolving `programs.{name}` ({source})"))?;
        let (kind, version) = program::kind(&bytes)
            .map_err(|error| anyhow::anyhow!("`programs.{name}` ({source}): {error}"))?;
        ensure!(
            version == kind.interface_version(),
            "`programs.{name}` ({source}) is a {} program of interface version {version}, and \
             this release runs version {}",
            kind.name(),
            kind.interface_version()
        );
        let digest = format!("{:x}", Sha256::digest(&bytes));
        fixed.programs.insert(
            name.clone(),
            proto::module::Program {
                source,
                backend,
                revision,
                kind,
                digest: digest.clone(),
            },
        );
        fixed.loaded.push(Loaded {
            name: name.clone(),
            entry: entry.clone(),
            kind,
            digest,
            bytes: Arc::from(bytes),
        });
    }

    check(&fixed.loaded)?;
    Ok(fixed)
}

/// Checks a package's rules on its programs' kinds: at most one scheduling, one routing and one
/// KV program, and a period for every program its host steps periodically.
fn check(loaded: &[Loaded]) -> anyhow::Result<()> {
    for kind in [Kind::Scheduling, Kind::Routing, Kind::Kv] {
        let named: Vec<&str> = loaded
            .iter()
            .filter(|program| program.kind == kind)
            .map(|program| program.name.as_str())
            .collect();
        if named.len() > 1 {
            bail!(
                "a package runs at most one {} program, and `programs` lists {}",
                kind.name(),
                named.join(", ")
            );
        }
    }
    for program in loaded {
        // A processing program is the request side of the engine's tree, which an epoch's
        // module set takes beside its engine (`worker_config.processing`), not a package's.
        ensure!(
            program.kind != Kind::Processing,
            "`programs.{}` is a processing program, which an epoch takes from its engine's tree \
             (`worker_config.processing`): remove it from `programs`",
            program.name
        );
        if matches!(program.kind, Kind::Control | Kind::Kv) {
            ensure!(
                program.entry.period.is_some(),
                "`programs.{}` is a {} program: state its `period`, how often it steps",
                program.name,
                program.kind.name()
            );
        }
    }
    Ok(())
}

/// The wasm32 module of revision `revision` of the program package `backend`, which the Build
/// Server builds first when its store does not hold it.
fn module(
    client: &mut build_server::Client,
    backend: &str,
    revision: u32,
) -> anyhow::Result<Vec<u8>> {
    let package = PackageRevision {
        backend: backend.to_owned(),
        revision,
        arguments: toml::Table::new(),
    };
    let module = client.resolve(
        proto::module::PROGRAM,
        &package,
        &build_server::Target::wasm(),
        build_server::manifest::SINGLE_KEY,
        &mut |_| {},
    )?;
    let file = module
        .files
        .into_iter()
        .find(|file| file.name == build_server::source::MODULE_WASM)
        .with_context(|| {
            format!(
                "the module carries no {}",
                build_server::source::MODULE_WASM
            )
        })?;
    let mut bytes = Vec::with_capacity(usize::try_from(file.size).unwrap_or(0));
    (&file.file)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {}", file.name))?;
    Ok(bytes)
}

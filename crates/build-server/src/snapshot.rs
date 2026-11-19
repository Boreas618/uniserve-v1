//! The snapshot an installation takes of a package: the files a revision
//! holds, and the digest that names them among a module's inputs.
//!
//! A package is its directory's files. A package cargo builds (the engine, a
//! scheduling policy, the processing program) is a crate of a workspace, and
//! its revision is the **whole workspace**: `package.toml` at the revision's
//! root and, under [`WORKSPACE`], every file cargo packages of every workspace
//! member (`cargo package --list`, which honours each crate's `include` and
//! `exclude` and the repository's ignore rules) with the workspace's manifest,
//! lockfile, toolchain file and cargo configuration, exactly as they are. The
//! revision is then the workspace cargo builds offline with its own lockfile,
//! wherever the tree it was installed from is, and any change to the workspace
//! is a change of the package's digest. Nothing is rewritten: no generated
//! manifest, no reduced lockfile.
//!
//! The registry and git sources the lockfile names are vendored once per
//! lockfile into the Build Server's cache ([`vendored`]), so a build reads no
//! crate from a mutable registry cache and fetches nothing.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::inputs::{BYTECODE_CACHE, ORIGIN, RECORD};
use crate::manifest::Manifest;
use crate::source::Cache;
use crate::toolchain::CARGO;
use crate::{Error, Result};

/// The directory of an installed cargo package's revision that holds the
/// snapshot of its workspace.
pub const WORKSPACE: &str = "workspace";

/// The files of a workspace root a snapshot holds, beside its members', when the workspace has
/// them: its manifest, lockfile and toolchain file, and the cargo configuration every build of it
/// reads (a target's flags).
const WORKSPACE_FILES: [&str; 4] = [
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
];

/// The directory of a lockfile's vendored sources inside its provisioning.
const VENDOR_SOURCES: &str = "sources";

/// The cargo configuration of a lockfile's provisioning, which names its
/// vendored sources.
const VENDOR_CONFIG: &str = "config.toml";

/// Named files: each by its name inside what holds it (an installed revision,
/// a module's directory) and where it is now.
pub type Files = Vec<(String, PathBuf)>;

/// What an installation of the package in `directory`, whose manifest is
/// `manifest`, holds: each file as its name inside the installed revision and
/// where it is now, in name order.
///
/// # Errors
///
/// Fails when the directory cannot be read, or a cargo package's workspace
/// cannot be listed by cargo.
pub fn package_files(directory: &Path, manifest: &Manifest) -> Result<Files> {
    let mut files = Vec::new();
    if !manifest.source.kind.builds_with_cargo() {
        tree_files(directory, Path::new(""), &mut files)?;
        return Ok(files);
    }

    files.push((Manifest::FILE.to_owned(), directory.join(Manifest::FILE)));
    files.extend(workspace_files(directory)?);
    Ok(files)
}

/// The directory an installation of the package in `directory` takes its
/// files from: the root of a cargo package's workspace, and the package's own
/// directory otherwise. Its git state is the revision's [`crate::inputs::Origin`].
///
/// # Errors
///
/// Fails when cargo cannot report a cargo package's workspace.
pub fn source_root(directory: &Path, manifest: &Manifest) -> Result<PathBuf> {
    if manifest.source.kind.builds_with_cargo() {
        return Ok(metadata(directory)?.workspace_root);
    }
    Ok(directory.to_path_buf())
}

/// The snapshot of the workspace the crate in `directory` belongs to, each file
/// named under [`WORKSPACE`]: the workspace's manifest, lockfile and toolchain
/// file, and every file cargo packages of every member, in name order.
///
/// # Errors
///
/// Fails when cargo cannot list the workspace.
pub fn workspace_files(directory: &Path) -> Result<Files> {
    let workspace = metadata(directory)?;
    let mut listed: BTreeMap<String, PathBuf> = BTreeMap::new();
    for name in WORKSPACE_FILES {
        let path = workspace.workspace_root.join(name);
        if path.is_file() {
            listed.insert(format!("{WORKSPACE}/{name}"), path);
        }
    }
    for member in &workspace.packages {
        let crate_root = member.directory()?;
        let relative = crate_root
            .strip_prefix(&workspace.workspace_root)
            .map_err(|_| invalid(format!("{} is outside its workspace", crate_root.display())))?;
        for file in package_list(&workspace.workspace_root, &member.name)? {
            // `cargo package` lists what it generates for a package (its VCS
            // record, a normalized manifest's original, a lockfile of a crate
            // with binaries) beside the crate's own files; only a file the crate
            // holds is a source.
            let path = crate_root.join(&file);
            if path.is_file() {
                let name = Path::new(WORKSPACE).join(relative).join(&file);
                listed.insert(name.to_string_lossy().into_owned(), path);
            }
        }
    }
    Ok(listed.into_iter().collect())
}

/// The SHA-256 of a snapshot, in hex: each file framed by its name and its
/// length, in the order given, so no two snapshots hash one byte stream.
///
/// # Errors
///
/// Fails when a file cannot be read.
pub fn files_digest(files: &[(String, PathBuf)]) -> Result<String> {
    let mut hasher = Sha256::new();
    for (name, path) in files {
        let bytes = fs::read(path)?;
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Writes a snapshot into `staged`, the directory that becomes the revision:
/// each file under its name with its mode, fsynced.
///
/// # Errors
///
/// Fails when a file cannot be copied.
pub fn write_files(files: &[(String, PathBuf)], staged: &Path) -> Result<()> {
    for (name, source) in files {
        let destination = staged.join(name);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source, &destination)?;
        fs::File::open(&destination)?.sync_all()?;
    }
    Ok(())
}

/// Adds every file below `directory`, named by `prefix` and its path below the
/// directory, depth first in name order. Python bytecode and a revision's own
/// record are no inputs and are left out. A symbolic link counts as what it
/// names, as an installation copies it.
fn tree_files(directory: &Path, prefix: &Path, files: &mut Files) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<io::Result<_>>()?;
    entries.sort();

    for name in entries {
        if name == BYTECODE_CACHE
            || (prefix.as_os_str().is_empty() && (name == RECORD || name == ORIGIN))
        {
            continue;
        }
        let path = directory.join(&name);
        let relative = prefix.join(&name);
        if fs::metadata(&path)?.is_dir() {
            tree_files(&path, &relative, files)?;
        } else {
            files.push((relative.to_string_lossy().into_owned(), path));
        }
    }
    Ok(())
}

/// The workspace cargo reports for a directory (`cargo metadata --no-deps`).
#[derive(Debug, Deserialize)]
struct Workspace {
    workspace_root: PathBuf,
    packages: Vec<Member>,
}

/// One member of a workspace, as cargo reports it.
#[derive(Clone, Debug, Deserialize)]
pub struct Member {
    /// cargo's package identifier, which its build messages name.
    pub id: String,
    pub name: String,
    manifest_path: PathBuf,
}

impl Member {
    /// The crate's directory.
    fn directory(&self) -> Result<&Path> {
        self.manifest_path
            .parent()
            .ok_or_else(|| invalid(format!("{} has no directory", self.manifest_path.display())))
    }
}

/// Runs cargo in `directory` and returns its standard output.
fn cargo(directory: &Path, arguments: &[&str]) -> Result<Vec<u8>> {
    let output = crate::toolchain::cargo(directory)
        .args(arguments)
        .output()
        .map_err(|error| {
            invalid(format!(
                "{CARGO} cannot be run in {}: {error}",
                directory.display()
            ))
        })?;
    if !output.status.success() {
        return Err(invalid(format!(
            "{CARGO} {} failed in {}: {}",
            arguments.join(" "),
            directory.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output.stdout)
}

/// The workspace the crate in `directory` belongs to, with its members.
fn metadata(directory: &Path) -> Result<Workspace> {
    let stdout = cargo(
        directory,
        &[
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
            "--locked",
        ],
    )?;
    serde_json::from_slice(&stdout).map_err(|error| invalid(format!("cargo metadata: {error}")))
}

/// The files cargo packages of the workspace member `name`, relative to its
/// directory.
fn package_list(workspace_root: &Path, name: &str) -> Result<Vec<String>> {
    let stdout = cargo(
        workspace_root,
        &[
            "package",
            "--list",
            "--allow-dirty",
            "--offline",
            "--quiet",
            "--package",
            name,
        ],
    )?;
    Ok(String::from_utf8_lossy(&stdout)
        .lines()
        .map(str::to_owned)
        .collect())
}

/// The member of the installed cargo package's revision at `revision` that is
/// the package: the workspace crate whose `package.toml` is the revision's own,
/// the file the installation copied from that crate.
///
/// # Errors
///
/// Fails when cargo cannot list the snapshot's workspace, or no member, or more
/// than one, holds the revision's manifest.
pub fn member(revision: &Path) -> Result<Member> {
    let installed = fs::read(revision.join(Manifest::FILE))?;
    let workspace = revision.join(WORKSPACE);
    let mut found = Vec::new();
    for member in metadata(&workspace)?.packages {
        let stated = member.directory()?.join(Manifest::FILE);
        if stated.is_file() && fs::read(&stated)? == installed {
            found.push(member);
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        count => Err(invalid(format!(
            "{count} members of the workspace at {} hold the package's {}",
            workspace.display(),
            Manifest::FILE
        ))),
    }
}

/// The cargo configuration that replaces every source the lockfile of the
/// workspace snapshot at `workspace` names with its vendored copy, provisioned
/// once per set of external packages the lockfile locks in the Build Server's
/// cache ([`Cache::vendor`], [`external_digest`]): two lockfiles that differ
/// only in their workspace members, such as a workspace and the same
/// workspace with a policy package's program crate added as a member, share
/// one provisioning.
///
/// The provisioning is `cargo vendor --locked --offline` into a staging
/// directory beside the destination ([`crate::store::Staging`]), which holds
/// the sources and the configuration cargo prints, rewritten to name the
/// sources' final location, and is renamed into place whole: builds that
/// provision one lockfile at once never read a partial copy, and a provisioning
/// that died leaves a staging directory the next one removes. A lockfile that
/// names no registry or git source has nothing to vendor, and its configuration
/// replaces nothing.
///
/// # Errors
///
/// Fails when the lockfile cannot be read or cargo cannot vendor its sources.
pub fn vendored(workspace: &Path, cache: &Cache) -> Result<PathBuf> {
    let lock = external_digest(&fs::read_to_string(workspace.join("Cargo.lock"))?)?;
    let destination = cache.vendor(&lock);
    let config = destination.join(VENDOR_CONFIG);
    if config.is_file() {
        return Ok(config);
    }
    let parent = destination
        .parent()
        .ok_or_else(|| invalid(format!("{} has no directory", destination.display())))?;
    let staging = crate::store::Staging::new(parent)?;
    let printed = cargo(
        workspace,
        &[
            "vendor",
            "--locked",
            "--offline",
            "--versioned-dirs",
            &staging.path().join(VENDOR_SOURCES).to_string_lossy(),
        ],
    )?;
    let mut replacement: toml::Table = toml::from_str(&String::from_utf8_lossy(&printed))
        .map_err(|error| invalid(format!("cargo vendor printed no configuration: {error}")))?;
    if let Some(directory) = replacement
        .get_mut("source")
        .and_then(|source| source.get_mut("vendored-sources"))
        .and_then(toml::Value::as_table_mut)
    {
        directory.insert(
            "directory".to_owned(),
            toml::Value::String(
                destination
                    .join(VENDOR_SOURCES)
                    .to_string_lossy()
                    .into_owned(),
            ),
        );
    }
    let text = toml::to_string(&replacement).map_err(|error| invalid(error.to_string()))?;
    fs::write(staging.path().join(VENDOR_CONFIG), text)?;
    match staging.publish(&destination) {
        Ok(()) => Ok(config),
        // Another build provisioned the same lockfile first: its copy is the
        // same sources.
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::AlreadyExists | io::ErrorKind::DirectoryNotEmpty
            ) && config.is_file() =>
        {
            Ok(config)
        }
        Err(error) => Err(Error::Io(error)),
    }
}

/// The SHA-256, in hex, of the external packages the lockfile `text` locks:
/// each package with a `source` (a registry or a git repository), by its name,
/// version, source and checksum, in the lockfile's order. Workspace members and
/// path dependencies have no source and are no part of it, since nothing is
/// vendored for them.
///
/// # Errors
///
/// Fails when the text is no lockfile.
pub fn external_digest(text: &str) -> Result<String> {
    let lock: toml::Table = toml::from_str(text)
        .map_err(|error| invalid(format!("the lockfile does not read: {error}")))?;
    let mut hasher = Sha256::new();
    let packages = lock
        .get("package")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for package in packages {
        let field = |key: &str| package.get(key).and_then(toml::Value::as_str);
        let Some(source) = field("source") else {
            continue;
        };
        for part in [
            field("name").unwrap_or_default(),
            field("version").unwrap_or_default(),
            source,
            field("checksum").unwrap_or_default(),
        ] {
            hasher.update(part.as_bytes());
            hasher.update([0]);
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn invalid(message: String) -> Error {
    Error::Io(io::Error::new(io::ErrorKind::InvalidData, message))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(root: &Path, files: &[(&str, &str)]) {
        for (name, text) in files {
            let path = root.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
    }

    fn digest(root: &Path) -> String {
        let mut files = Vec::new();
        tree_files(root, Path::new(""), &mut files).unwrap();
        files_digest(&files).unwrap()
    }

    /// Two lockfiles that differ only in their workspace members lock one set of external
    /// packages, which one vendored copy serves; another version of an external package is
    /// another set.
    #[test]
    fn the_vendored_sources_follow_the_external_packages_alone() {
        let lock = |members: &str, version: &str| {
            format!(
                "version = 4\n\n{members}[[package]]\nname = \"serde\"\nversion = \"{version}\"\n\
                 source = \"registry+https://github.com/rust-lang/crates.io-index\"\n\
                 checksum = \"ab\"\n"
            )
        };
        let sdk = lock(
            "[[package]]\nname = \"scheduling\"\nversion = \"2.0.0\"\n\n",
            "1.0.1",
        );
        let program = lock(
            "[[package]]\nname = \"program-first\"\nversion = \"0.1.0\"\n\
             dependencies = [\"serde\"]\n\n[[package]]\nname = \"scheduling\"\n\
             version = \"2.0.0\"\n\n",
            "1.0.1",
        );
        let other = lock("", "1.0.2");
        assert_eq!(
            external_digest(&sdk).unwrap(),
            external_digest(&program).unwrap()
        );
        assert_ne!(
            external_digest(&sdk).unwrap(),
            external_digest(&other).unwrap()
        );
    }

    /// A changed byte, a renamed file and an added file each change a package's digest; bytecode
    /// and the revision's records do not.
    #[test]
    fn a_package_digest_follows_the_files_a_build_reads() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        tree(
            root,
            &[(Manifest::FILE, "op = 1\n"), ("src/kernel.py", "x = 1\n")],
        );
        let base = digest(root);

        tree(
            root,
            &[
                ("src/__pycache__/kernel.pyc", "bytecode"),
                (RECORD, "record"),
                (ORIGIN, "origin"),
            ],
        );
        assert_eq!(digest(root), base);

        tree(root, &[("src/kernel.py", "x = 2\n")]);
        let edited = digest(root);
        assert_ne!(edited, base);

        fs::rename(root.join("src/kernel.py"), root.join("src/other.py")).unwrap();
        assert_ne!(digest(root), edited);
    }
}

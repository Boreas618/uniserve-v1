//! The snapshot an installation takes of a package: the files a revision
//! holds, and the digest that names them among a module's inputs.
//!
//! A package is its directory's files. A package cargo builds is more than its
//! directory: its crate inherits from a workspace and depends on crates of it,
//! so its revision holds `package.toml` and, under [`WORKSPACE`], the crate
//! with every workspace crate it depends on, the workspace's lockfile and
//! toolchain file, and a workspace manifest naming exactly those crates as
//! members. A crate the workspace depends on by a path outside itself is
//! copied beside the workspace, under [`VENDORED`], and the manifest names it
//! there. The revision is then a workspace cargo builds offline with the
//! lockfile, wherever the tree it was installed from is, and a change to any
//! crate the package depends on is a change of the package's digest.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::inputs::{BYTECODE_CACHE, RECORD};
use crate::{Error, Result};

/// The directory of an installed cargo package's revision that holds the
/// snapshot of its workspace.
pub const WORKSPACE: &str = "workspace";

/// The directory of an installed cargo package's revision that holds the
/// crates its workspace depends on by a path outside itself, one directory per
/// dependency name.
pub const VENDORED: &str = "vendored";

/// The name of a cargo snapshot's workspace manifest inside the revision.
const WORKSPACE_MANIFEST: &str = "workspace/Cargo.toml";

/// The source kinds whose packages cargo builds from a workspace snapshot, as
/// a manifest's `[source] kind` spells them.
const CARGO_KINDS: [&str; 2] = ["cargo", "wasm"];

/// Files of a snapshot: each by its name inside the installed revision and
/// where it is now.
pub type Files = Vec<(String, PathBuf)>;

/// What an installation of the package in `directory` holds: each file as its
/// name inside the installed revision and where it is now, and, for a package
/// cargo builds, the text of the workspace manifest the revision is built
/// under ([`workspace_manifest`]).
///
/// # Errors
///
/// Fails when the directory cannot be read, or a cargo package's workspace
/// does not read: no workspace manifest at or above it, a manifest that does
/// not parse, or a dependency outside the workspace.
pub fn package_files(directory: &Path) -> Result<(Files, Option<String>)> {
    let mut files = Vec::new();
    if !builds_with_cargo(directory) {
        tree_files(directory, Path::new(""), &|_, _| true, &mut files)?;
        return Ok((files, None));
    }

    files.push(("package.toml".to_owned(), directory.join("package.toml")));
    let root = workspace_root(directory)?;
    let package = relative_to(&root, directory)?;
    let (members, vendored) = workspace_members(&root, &package)?;
    for name in ["Cargo.lock", "rust-toolchain.toml"] {
        if root.join(name).is_file() {
            files.push((format!("{WORKSPACE}/{name}"), root.join(name)));
        }
    }
    for member in &members {
        crate_files(
            &root.join(member),
            &Path::new(WORKSPACE).join(member),
            &mut files,
        )?;
    }
    for (name, directory) in &vendored {
        crate_files(directory, &Path::new(VENDORED).join(name), &mut files)?;
    }
    let manifest = workspace_manifest(&root, &members, &vendored, &package)?;
    Ok((files, Some(manifest)))
}

/// Adds the files of the crate at `crate_root`, named below `prefix`.
///
/// A build writes its output under `target`, a dot entry is a tool's own
/// state, and a directory holding its own `package.toml` is another package's
/// source: none of them is the crate's.
fn crate_files(crate_root: &Path, prefix: &Path, files: &mut Vec<(String, PathBuf)>) -> Result<()> {
    let part_of_crate = |path: &Path, name: &OsStr| {
        let name = name.to_string_lossy();
        !(name == "target"
            || name.starts_with('.')
            || (path != crate_root && path.join("package.toml").is_file()))
    };
    tree_files(crate_root, prefix, &part_of_crate, files)
}

/// The SHA-256 of a snapshot, in hex: each file framed by its name and its
/// length, in the order given, and the generated workspace manifest of a cargo
/// package last, so no two snapshots hash one byte stream.
///
/// # Errors
///
/// Fails when a file cannot be read.
pub fn files_digest(files: &[(String, PathBuf)], generated: Option<&str>) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut frame = |name: &str, bytes: &[u8]| {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    for (name, path) in files {
        frame(name, &fs::read(path)?);
    }
    if let Some(manifest) = generated {
        frame(WORKSPACE_MANIFEST, manifest.as_bytes());
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// The SHA-256 of a directory tree's files, in hex.
///
/// Every file is hashed with its path relative to `root`, in path order, so a
/// renamed or moved file changes the digest as a changed byte does. A symbolic
/// link counts as what it names, as an install copies it. Python bytecode
/// ([`BYTECODE_CACHE`]) and a revision's own record ([`RECORD`]) are not
/// inputs and are skipped.
///
/// # Errors
///
/// Fails when the tree cannot be read.
pub fn tree_digest(root: &Path) -> Result<String> {
    let mut files = Vec::new();
    tree_files(root, Path::new(""), &|_, _| true, &mut files)?;
    files_digest(&files, None)
}

/// Writes a snapshot into `staged`, the directory that becomes the revision:
/// each file under its name with its mode, fsynced, and the generated
/// workspace manifest of a cargo package.
///
/// # Errors
///
/// Fails when a file cannot be copied or written.
pub fn write_files(
    files: &[(String, PathBuf)],
    generated: Option<&str>,
    staged: &Path,
) -> Result<()> {
    for (name, source) in files {
        let destination = staged.join(name);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source, &destination)?;
        fs::File::open(&destination)?.sync_all()?;
    }

    if let Some(manifest) = generated {
        let path = staged.join(WORKSPACE_MANIFEST);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, manifest)?;
        fs::File::open(&path)?.sync_all()?;
    }
    Ok(())
}

/// Whether the package in `directory` states a source kind cargo builds.
fn builds_with_cargo(directory: &Path) -> bool {
    fs::read_to_string(directory.join("package.toml"))
        .ok()
        .and_then(|text| toml::from_str::<toml::Table>(&text).ok())
        .and_then(|manifest| {
            let kind = manifest.get("source")?.get("kind")?.as_str()?;
            Some(CARGO_KINDS.contains(&kind))
        })
        .unwrap_or(false)
}

/// Adds every file below `directory`, named by `prefix` and its path below the
/// directory, depth first in name order. An entry `keep` refuses, given its
/// path and name, is left out with everything below it; Python bytecode and a
/// revision's own record always are.
fn tree_files(
    directory: &Path,
    prefix: &Path,
    keep: &dyn Fn(&Path, &OsStr) -> bool,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<io::Result<_>>()?;
    entries.sort();

    for name in entries {
        if name == BYTECODE_CACHE || (prefix.as_os_str().is_empty() && name == RECORD) {
            continue;
        }
        let path = directory.join(&name);
        if !keep(&path, &name) {
            continue;
        }
        let relative = prefix.join(&name);
        // A symbolic link counts as what it names.
        if fs::metadata(&path)?.is_dir() {
            tree_files(&path, &relative, keep, files)?;
        } else {
            files.push((relative.to_string_lossy().into_owned(), path));
        }
    }
    Ok(())
}

/// The root of the cargo workspace the crate in `directory` belongs to: the
/// nearest directory at or above it whose `Cargo.toml` has a `[workspace]`.
fn workspace_root(directory: &Path) -> Result<PathBuf> {
    let directory = fs::canonicalize(directory)?;
    directory
        .ancestors()
        .find(|candidate| {
            cargo_manifest(candidate).is_ok_and(|manifest| manifest.contains_key("workspace"))
        })
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            invalid(format!(
                "no cargo workspace holds {}: no Cargo.toml at or above it has a [workspace]",
                directory.display()
            ))
        })
}

/// The `Cargo.toml` of `directory`, parsed.
fn cargo_manifest(directory: &Path) -> Result<toml::Table> {
    let path = directory.join("Cargo.toml");
    let text = fs::read_to_string(&path)?;
    toml::from_str(&text).map_err(|error| invalid(format!("{}: {error}", path.display())))
}

/// `path` relative to the workspace root `root`, which must hold it.
fn relative_to(root: &Path, path: &Path) -> Result<PathBuf> {
    let canonical = fs::canonicalize(path)?;
    canonical
        .strip_prefix(root)
        .map(Path::to_path_buf)
        .map_err(|_| {
            invalid(format!(
                "{} is outside the workspace at {}",
                canonical.display(),
                root.display()
            ))
        })
}

/// The crates that building and resolving the crate at `package` of the
/// workspace at `root` needs: the workspace's own, relative to the root and in
/// path order, and those the workspace depends on by a path outside itself,
/// each by its dependency name and its directory.
///
/// The workspace's own are the crate itself and, transitively, every path
/// dependency of a crate already in the set, whether stated by `path` or
/// inherited from the workspace. Development dependencies count: cargo
/// resolves them for every member of a workspace, so a snapshot without them
/// does not load.
///
/// # Errors
///
/// Fails when a manifest does not read, or a crate states a path outside the
/// workspace itself: only a dependency the workspace states can be vendored,
/// because only the workspace manifest is written for the snapshot.
fn workspace_members(root: &Path, package: &Path) -> Result<(Vec<PathBuf>, Files)> {
    let workspace = cargo_manifest(root)?;
    let inherited = workspace
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(toml::Value::as_table);

    let mut members = BTreeSet::new();
    let mut vendored = std::collections::BTreeMap::new();
    let mut pending = vec![package.to_path_buf()];
    while let Some(member) = pending.pop() {
        if !members.insert(member.clone()) {
            continue;
        }
        let manifest = cargo_manifest(&root.join(&member))?;

        // The dependency tables of a manifest: its own three, and the three of
        // each `[target.<cfg>]`.
        let mut tables = vec![&manifest];
        if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
            tables.extend(targets.values().filter_map(toml::Value::as_table));
        }

        for table in tables {
            for kind in ["dependencies", "dev-dependencies", "build-dependencies"] {
                let Some(dependencies) = table.get(kind).and_then(toml::Value::as_table) else {
                    continue;
                };
                for (name, dependency) in dependencies {
                    let inherits =
                        dependency.get("workspace").and_then(toml::Value::as_bool) == Some(true);
                    match dependency.get("path").and_then(toml::Value::as_str) {
                        Some(path) => {
                            pending.push(relative_to(root, &root.join(&member).join(path))?);
                        }
                        None if inherits => {
                            let Some(path) = inherited
                                .and_then(|inherited| inherited.get(name)?.get("path")?.as_str())
                            else {
                                continue;
                            };
                            // `join` keeps an absolute path as it is.
                            let directory = fs::canonicalize(root.join(path))?;
                            match directory.strip_prefix(root) {
                                Ok(inside) => pending.push(inside.to_path_buf()),
                                Err(_) => {
                                    vendored.insert(name.clone(), directory);
                                }
                            }
                        }
                        None => {}
                    }
                }
            }
        }
    }
    Ok((
        members.into_iter().collect(),
        vendored.into_iter().collect(),
    ))
}

/// The workspace manifest a cargo package's revision is built under: the
/// manifest of the workspace at `root` with `members` as its only members,
/// the crate at `package` as the one cargo builds by default, and no root
/// package.
///
/// The workspace's shared package fields, dependency versions, lints and
/// profiles are kept, so every member compiles as it does in its tree. A
/// workspace dependency on a crate of `vendored` names the copy beside the
/// workspace, and one on a crate the snapshot does not hold is dropped,
/// because no member inherits it.
fn workspace_manifest(
    root: &Path,
    members: &[PathBuf],
    vendored: &[(String, PathBuf)],
    package: &Path,
) -> Result<String> {
    let original = cargo_manifest(root)?;
    let mut workspace = original
        .get("workspace")
        .and_then(toml::Value::as_table)
        .cloned()
        .unwrap_or_default();

    let named = |path: &Path| toml::Value::String(path.to_string_lossy().into_owned());
    workspace.insert(
        "members".to_owned(),
        toml::Value::Array(members.iter().map(|member| named(member)).collect()),
    );
    workspace.insert(
        "default-members".to_owned(),
        toml::Value::Array(vec![named(package)]),
    );
    workspace.remove("exclude");

    if let Some(dependencies) = workspace
        .get_mut("dependencies")
        .and_then(toml::Value::as_table_mut)
    {
        dependencies.retain(|name, dependency| {
            let Some(table) = dependency.as_table_mut() else {
                return true;
            };
            let Some(path) = table.get("path").and_then(toml::Value::as_str) else {
                return true;
            };
            if vendored.iter().any(|(vendored, _)| vendored == name) {
                table.insert(
                    "path".to_owned(),
                    toml::Value::String(format!("../{VENDORED}/{name}")),
                );
                return true;
            }
            members.iter().any(|member| member == Path::new(path))
        });
    }

    let mut manifest = toml::Table::new();
    manifest.insert("workspace".to_owned(), toml::Value::Table(workspace));
    for section in ["profile", "patch"] {
        if let Some(value) = original.get(section) {
            manifest.insert(section.to_owned(), value.clone());
        }
    }
    toml::to_string(&manifest).map_err(|error| invalid(error.to_string()))
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

    /// A changed byte, a renamed file and an added file each change a tree's digest; bytecode and
    /// the revision's record do not.
    #[test]
    fn a_tree_digest_follows_the_files_a_build_reads() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        tree(
            root,
            &[("package.toml", "op = 1\n"), ("src/kernel.py", "x = 1\n")],
        );
        let base = tree_digest(root).unwrap();

        tree(
            root,
            &[
                ("src/__pycache__/kernel.pyc", "bytecode"),
                (RECORD, "record"),
            ],
        );
        assert_eq!(tree_digest(root).unwrap(), base);

        tree(root, &[("src/kernel.py", "x = 2\n")]);
        let edited = tree_digest(root).unwrap();
        assert_ne!(edited, base);

        fs::rename(root.join("src/kernel.py"), root.join("src/other.py")).unwrap();
        assert_ne!(tree_digest(root).unwrap(), edited);

        tree(root, &[("src/kernel.py", "x = 2\n")]);
        assert_ne!(tree_digest(root).unwrap(), edited);
    }

    /// A workspace at `root` of an application crate that depends on `core` by inheritance, on
    /// `tools` only for its tests and on `outer`, a crate beside the workspace, through the
    /// workspace; and a crate nothing of it depends on.
    fn workspace(root: &Path) {
        tree(
            root,
            &[
                ("../outer/Cargo.toml", "[package]\nname = \"outer\"\n"),
                ("../outer/src/lib.rs", "\n"),
                (
                    "Cargo.toml",
                    "[workspace]\nresolver = \"3\"\nmembers = [\"crates/app\", \"crates/core\", \
                     \"crates/tools\", \"crates/other\"]\n\n[workspace.package]\nversion = \
                     \"1.0.0\"\n\n[workspace.dependencies]\ncore = { path = \"crates/core\" }\n\
                     other = { path = \"crates/other\" }\nlibc = \"0.2\"\nouter = { path = \
                     \"../outer\" }\n\n[profile.release]\n\
                     lto = \"thin\"\n\n[package]\nname = \"root\"\n",
                ),
                ("Cargo.lock", "version = 4\n"),
                ("rust-toolchain.toml", "[toolchain]\nchannel = \"stable\"\n"),
                (
                    "crates/app/Cargo.toml",
                    "[package]\nname = \"app\"\n\n[dependencies]\ncore.workspace = true\n\
                     outer.workspace = true\n\n\
                     [dev-dependencies]\ntools = { path = \"../tools\" }\n",
                ),
                ("crates/app/package.toml", "[source]\nkind = \"cargo\"\n"),
                ("crates/app/src/main.rs", "fn main() {}\n"),
                ("crates/app/target/release/app", "built"),
                ("crates/app/.cache/state", "state"),
                ("crates/core/Cargo.toml", "[package]\nname = \"core\"\n"),
                ("crates/core/src/lib.rs", "\n"),
                ("crates/core/kernels/package.toml", "another package\n"),
                ("crates/core/kernels/kernel.py", "\n"),
                ("crates/tools/Cargo.toml", "[package]\nname = \"tools\"\n"),
                ("crates/tools/src/lib.rs", "\n"),
                ("crates/other/Cargo.toml", "[package]\nname = \"other\"\n"),
                ("crates/other/src/lib.rs", "\n"),
            ],
        );
    }

    /// A cargo package's snapshot is its crate and the workspace crates it depends on, tests
    /// included, under a workspace manifest naming exactly them: not a crate it does not depend
    /// on, a build's output, a tool's state or another package's source.
    #[test]
    fn a_cargo_package_is_its_crate_and_the_workspace_crates_it_depends_on() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("tree");
        workspace(&root);

        let (files, manifest) = package_files(&root.join("crates/app")).unwrap();
        let names: Vec<&str> = files.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "package.toml",
                "workspace/Cargo.lock",
                "workspace/rust-toolchain.toml",
                "workspace/crates/app/Cargo.toml",
                "workspace/crates/app/package.toml",
                "workspace/crates/app/src/main.rs",
                "workspace/crates/core/Cargo.toml",
                "workspace/crates/core/src/lib.rs",
                "workspace/crates/tools/Cargo.toml",
                "workspace/crates/tools/src/lib.rs",
                "vendored/outer/Cargo.toml",
                "vendored/outer/src/lib.rs",
            ]
        );

        let manifest: toml::Table = toml::from_str(&manifest.unwrap()).unwrap();
        let workspace = manifest["workspace"].as_table().unwrap();
        let listed = |key: &str| -> Vec<&str> {
            workspace[key]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(toml::Value::as_str)
                .collect()
        };
        assert_eq!(
            listed("members"),
            ["crates/app", "crates/core", "crates/tools"]
        );
        assert_eq!(listed("default-members"), ["crates/app"]);
        let dependencies = workspace["dependencies"].as_table().unwrap();
        assert!(dependencies.contains_key("core") && dependencies.contains_key("libc"));
        assert!(!dependencies.contains_key("other"));
        assert_eq!(
            dependencies["outer"]["path"].as_str(),
            Some("../vendored/outer"),
            "a crate outside the workspace is named where the snapshot holds it"
        );
        assert_eq!(workspace["package"]["version"].as_str(), Some("1.0.0"));
        assert_eq!(manifest["profile"]["release"]["lto"].as_str(), Some("thin"));
        assert!(!manifest.contains_key("package"));
    }

    /// A cargo package's digest follows every crate it depends on and nothing else of its tree.
    #[test]
    fn a_cargo_package_digest_follows_the_crates_it_depends_on() {
        let directory = tempfile::tempdir().unwrap();
        let root = &directory.path().join("tree");
        workspace(root);
        let digest = || {
            let (files, manifest) = package_files(&root.join("crates/app")).unwrap();
            files_digest(&files, manifest.as_deref()).unwrap()
        };
        let base = digest();

        tree(root, &[("crates/other/src/lib.rs", "// edited\n")]);
        tree(root, &[("crates/core/kernels/kernel.py", "# edited\n")]);
        assert_eq!(digest(), base);

        tree(root, &[("crates/core/src/lib.rs", "// edited\n")]);
        let edited = digest();
        assert_ne!(edited, base);

        tree(root, &[("Cargo.lock", "version = 4\n# edited\n")]);
        let relocked = digest();
        assert_ne!(relocked, edited);

        tree(root, &[("../outer/src/lib.rs", "// edited\n")]);
        assert_ne!(digest(), relocked);
    }
}

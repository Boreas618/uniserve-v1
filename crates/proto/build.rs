//! Compiles the cross-build interfaces' Protobuf schemas and the operator
//! vocabulary into Rust, and names the build this crate is compiled in
//! (`UNISERVE_SOURCE_DIGEST`).
//!
//! The schemas are compiled with `protox`, a Protobuf compiler written in
//! Rust, so a build needs no `protoc`. They and the vocabulary lie under
//! `src/`, so they count towards the build identity like every other source.
//!
//! The build identity is a digest of the workspace sources every artifact of
//! a build is compiled from. The engine process, a node's daemon and the rank
//! runtime library are installed as separate artifacts and exchange the engine
//! interface's messages in a positional encoding, so each registration
//! states the build it runs and the engine refuses any other; the package
//! version cannot tell two builds of one version apart, the digest does.
//!
//! The digest covers what Cargo compiles those artifacts from: the
//! workspace's manifest, lockfile and toolchain file, and of each member
//! crate under `crates/` its manifest, its build script
//! and every file under its `src/`; the workspace's root package is the
//! `uniserve` executable, whose manifest is the workspace manifest and whose
//! sources are the root `src/`. A member's tests, benches and examples
//! are part of no artifact and are left out. Each file enters with its path
//! relative to the workspace root and its length, so the digest depends on
//! neither where the tree is checked out nor how files split their bytes.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

/// Files at the workspace root that configure every build of it.
const ROOT_FILES: [&str; 3] = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"];

/// Hex digits of the digest the build version carries: its leading 64 bits,
/// which tell builds apart as a short commit hash does.
const DIGEST_HEX_DIGITS: usize = 16;

/// The operator vocabulary, relative to this crate: the one statement of the
/// operators' names, which [`emit_operators`] compiles into
/// `proto::module::operator` and the engine package names as the vocabulary
/// its revisions record at the Build Server (`operators` in
/// `crates/engine/package.toml`).
const OPERATORS: &str = "src/module/operators.toml";

/// The workspace root, as Cargo itself locates it: the directory of the
/// workspace manifest. The Cargo that runs this script names itself in `CARGO`,
/// and a build script runs in its package's directory.
fn workspace_root() -> io::Result<PathBuf> {
    let cargo =
        std::env::var_os("CARGO").ok_or_else(|| io::Error::other("Cargo did not set CARGO"))?;
    let located = Command::new(cargo)
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .output()?;
    if !located.status.success() {
        return Err(io::Error::other(format!(
            "cargo locate-project failed: {}",
            String::from_utf8_lossy(&located.stderr)
        )));
    }
    let manifest = PathBuf::from(
        String::from_utf8(located.stdout)
            .map_err(io::Error::other)?
            .trim(),
    );
    manifest
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| io::Error::other("the workspace manifest has no directory"))
}

/// Emits `UNISERVE_SOURCE_DIGEST`, the digest of the workspace sources every
/// artifact of the build is compiled from, and `UNISERVE_WORKSPACE`, the
/// workspace root it was computed over.
fn emit_source_digest() -> io::Result<()> {
    let root = workspace_root()?;
    println!("cargo:rustc-env=UNISERVE_WORKSPACE={}", root.display());

    // Every input as a path relative to the root, and the paths whose change
    // reruns this script: each existing input file and each member's `src/`
    // directory, which Cargo scans recursively, so an added source file
    // counts too.
    let mut inputs: Vec<PathBuf> = ROOT_FILES.iter().map(PathBuf::from).collect();
    let mut members = Vec::new();
    find_members(&root, Path::new("crates"), &mut members)?;
    let mut watched = Vec::new();
    for member in &members {
        inputs.push(member.join("Cargo.toml"));
        inputs.push(member.join("build.rs"));
        let sources = member.join("src");
        watched.push(sources.clone());
        list_files(&root, &sources, &mut inputs)?;
    }
    // The workspace's root package is the `uniserve` executable: its manifest
    // is the workspace manifest (`ROOT_FILES`) and its sources are the root
    // `src/`. The snapshot an engine package is built from is a workspace
    // without a root package, so it has none.
    let root_sources = PathBuf::from("src");
    if root.join(&root_sources).is_dir() {
        watched.push(root_sources.clone());
        list_files(&root, &root_sources, &mut inputs)?;
    }
    inputs.sort();
    inputs.dedup();

    let mut digest = Sha256::new();
    for input in &inputs {
        // An optional input that is absent (a member without a build
        // script) is skipped; a path Cargo is told to watch must exist, or
        // Cargo reruns the script on every build.
        let contents = match std::fs::read(root.join(input)) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        watched.push(input.clone());
        digest.update(input.as_os_str().as_encoded_bytes());
        digest.update([0]);
        digest.update((contents.len() as u64).to_le_bytes());
        digest.update(&contents);
    }
    for path in &watched {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    println!(
        "cargo:rustc-env=UNISERVE_SOURCE_DIGEST={}",
        &hex[..DIGEST_HEX_DIGITS]
    );
    Ok(())
}

/// Writes `operators.rs` into `OUT_DIR` from the vocabulary [`OPERATORS`] of
/// the crate at `manifest`, which `proto::module::operator` includes: one
/// constant per operator, named by the operator's name in upper case, `ALL`,
/// every constant in the order the vocabulary lists them, and `VERSIONS`, each
/// operator past its first interface version with the version it launches.
///
/// The vocabulary is a TOML table of two keys: `operators`, an array of names,
/// and `interface_versions`, a table of the version of each operator's
/// interface past its first, by name, which is stated even when it names
/// none. A name is a lowercase ASCII letter followed by lowercase letters,
/// digits and `_`, so it is one path segment of the module store and its upper
/// case one Rust identifier, and each name is listed once; a version names a
/// listed operator and is a positive integer. Anything else fails the build,
/// naming the file. The Build Server reads the same file of an engine revision
/// by the same rules (`build_server::manifest::read_operators`), and also an
/// installed revision's vocabulary of the previous format, without
/// `interface_versions`, which this build never compiles
/// (`build_server::manifest::VOCABULARY_FORMATS`).
fn emit_operators(manifest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let path = manifest.join(OPERATORS);
    println!("cargo:rerun-if-changed={}", path.display());
    let table: toml::Table = std::fs::read_to_string(&path)?.parse()?;
    let invalid = |reason: String| format!("{}: {reason}", path.display());
    if let Some(key) = table
        .keys()
        .find(|key| !["operators", "interface_versions"].contains(&key.as_str()))
    {
        return Err(invalid(format!("`{key}` is no key of the vocabulary")).into());
    }
    let names = table
        .get("operators")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| invalid("`operators` is no array of names".to_owned()))?;
    let versions = table
        .get("interface_versions")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| {
            invalid(
                "`interface_versions` is no table of the operators' interface versions past \
                 their first"
                    .to_owned(),
            )
        })?;

    let mut listed = std::collections::BTreeSet::new();
    let mut constants = String::new();
    let mut all = String::new();
    for name in names {
        let name = name
            .as_str()
            .ok_or_else(|| invalid(format!("{name} is no name")))?;
        let spelled = name.starts_with(|first: char| first.is_ascii_lowercase())
            && name
                .chars()
                .all(|char| char.is_ascii_lowercase() || char.is_ascii_digit() || char == '_');
        if !spelled {
            return Err(invalid(format!(
                "`{name}` is no operator name: a lowercase letter, then lowercase letters, digits \
                 and `_`"
            ))
            .into());
        }
        if !listed.insert(name) {
            return Err(invalid(format!("`{name}` is listed twice")).into());
        }

        let constant = name.to_ascii_uppercase();
        constants.push_str(&format!(
            "/// The operator `{name}`.\npub const {constant}: &str = \"{name}\";\n"
        ));
        all.push_str(&format!("    {constant},\n"));
    }

    // Each stated version, as the constant of its operator and a positive integer.
    let mut stated = String::new();
    for (name, version) in versions {
        if !listed.contains(name.as_str()) {
            return Err(invalid(format!(
                "`interface_versions` states `{name}`, which the vocabulary does not list"
            ))
            .into());
        }
        let version = version
            .as_integer()
            .filter(|version| (1..=i64::from(u32::MAX)).contains(version))
            .ok_or_else(|| {
                invalid(format!(
                    "`{name}`'s interface version is {version}: a positive integer"
                ))
            })?;
        stated.push_str(&format!(
            "    ({}, {version}),\n",
            name.to_ascii_uppercase()
        ));
    }

    let generated = format!(
        "{constants}\n/// Every operator of the vocabulary, in the order it lists them.\n\
         pub const ALL: &[&str] = &[\n{all}];\n\n\
         /// The operators whose interface is past its first version, each with the version this\n\
         /// build launches, as the vocabulary's `interface_versions` states them.\n\
         const VERSIONS: &[(&str, u32)] = &[\n{stated}];\n"
    );
    let out = PathBuf::from(std::env::var("OUT_DIR")?);
    std::fs::write(out.join("operators.rs"), generated)?;
    Ok(())
}

/// Adds every member crate below `directory` (relative to `root`): each
/// directory holding a `Cargo.toml`, without descending into it.
fn find_members(root: &Path, directory: &Path, members: &mut Vec<PathBuf>) -> io::Result<()> {
    if root.join(directory).join("Cargo.toml").is_file() {
        members.push(directory.to_path_buf());
        return Ok(());
    }
    for entry in std::fs::read_dir(root.join(directory))? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            find_members(root, &directory.join(entry.file_name()), members)?;
        }
    }
    Ok(())
}

/// Adds every file below `directory` (relative to `root`), recursively.
fn list_files(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(root.join(directory))? {
        let entry = entry?;
        let path = directory.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            list_files(root, &path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let schemas = [
        manifest.join("src/handoff.proto"),
        manifest.join("src/process.proto"),
        manifest.join("src/request.proto"),
    ];
    for schema in &schemas {
        println!("cargo:rerun-if-changed={}", schema.display());
    }
    let descriptors = protox::compile(&schemas, [manifest.join("src")])?;
    // An engine process's launch file is JSON (`config::EngineArgs`), and the handover messages it
    // carries read and write there as well as on the wire.
    let mut config = prost_build::Config::new();
    for message in [
        "Predecessor",
        "CacheSegments",
        "CacheSegment",
        "ExpertOrder",
    ] {
        config.type_attribute(
            format!(".uniserve.request.{message}"),
            "#[derive(serde::Serialize, serde::Deserialize)]",
        );
    }
    config.compile_fds(descriptors)?;

    emit_operators(&manifest)?;
    emit_source_digest()?;
    Ok(())
}

//! Names the build this crate is compiled in: a digest of the workspace
//! sources every artifact of the build is compiled from.
//!
//! The head (`uniserve`), a host's launcher (`uniserve-launcher`) and the rank
//! runtime library are installed as separate artifacts and exchange this
//! crate's messages in a positional encoding, so each registration states
//! the build it runs and the head refuses any other (`registration::VERSION`).
//! The package version cannot tell two builds of one version apart; the
//! digest emitted here as `UNISERVE_SOURCE_DIGEST` does.
//!
//! The digest covers what Cargo compiles those artifacts from: the
//! workspace's manifest, lockfile, toolchain file and Cargo configuration,
//! and of each member crate under `crates/` its manifest, its build script
//! and every file under its `src/`. A member's tests, benches and examples
//! are part of no artifact and are left out. Each file enters with its path
//! relative to the workspace root and its length, so the digest depends on
//! neither where the tree is checked out nor how files split their bytes.

use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Files at the workspace root that configure every build of it.
const ROOT_FILES: [&str; 4] = [
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
];

/// Hex digits of the digest the build version carries: its leading 64 bits,
/// which tell builds apart as a short commit hash does.
const DIGEST_HEX_DIGITS: usize = 16;

fn main() -> io::Result<()> {
    // This crate is `crates/ipc` of the workspace.
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("Cargo did not set CARGO_MANIFEST_DIR"))?;
    let root = manifest.join("../..");

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

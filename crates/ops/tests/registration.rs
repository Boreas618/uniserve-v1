//! Every operator this crate declares can be installed by name.
//!
//! A program package is installed by `(operator, descriptor_key)`, and `install` answers `Ok(None)`
//! for a name it does not know — so an operator that is declared but missing from the table is a
//! module that never loads, silently, at precopy time. That has happened three times on this branch,
//! each time for a new operator nobody registered. The check is therefore structural and cheap: the
//! table's entries and the crate's `impl Op for …` declarations must name the same set of types.
//!
//! The scan reads the crate's own sources at test time. It does not parse Rust; it looks for two
//! shapes, one per side of the comparison, and reports both sets when they differ. A file counts
//! only once the module tree declares it, so a module being written beside the tree (declared in no
//! `mod` line yet) is nobody's problem until it joins the crate — and the moment it does, this test
//! asks for its name.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Every `.rs` file under `dir`, recursively.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("{}: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            sources(&path, out)?;
        } else if path.extension().is_some_and(|suffix| suffix == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Every `.md` and `.txt` file under `dir`, recursively: the notes the programs swarm reads.
fn documents(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("{}: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            documents(&path, out)?;
        } else if path
            .extension()
            .is_some_and(|suffix| suffix == "md" || suffix == "txt")
        {
            out.push(path);
        }
    }
    Ok(())
}

/// The module names the source declares with a `mod` line.
fn module_name(line: &str) -> Option<String> {
    let rest = line
        .strip_prefix("pub mod ")
        .or_else(|| line.strip_prefix("mod "))?;
    let name: String = rest
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// `impl Op for Type` at the start of a source line, answering `Type`.
fn declared(line: &str) -> Option<String> {
    let rest = line.strip_prefix("impl Op for ")?;
    let name: String = rest
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// `<crate::module::Type as Op>::NAME,` at the start of a source line, answering `Type`.
fn registered(line: &str) -> Option<String> {
    let rest = line.strip_prefix("<crate::")?;
    let path = rest.split(" as Op>").next()?;
    let name = path.rsplit("::").next()?;
    (!name.is_empty()).then(|| name.to_owned())
}

#[test]
fn every_declared_operator_is_registered_by_name() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&root, &mut files)?;

    let mut modules = BTreeSet::new();
    let mut texts = Vec::new();
    for file in files {
        let text =
            fs::read_to_string(&file).map_err(|error| format!("{}: {error}", file.display()))?;
        for line in text.lines() {
            if let Some(name) = module_name(line.trim()) {
                modules.insert(name);
            }
        }
        let stem = file
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_owned();
        texts.push((stem, text));
    }

    let mut declared_types = BTreeSet::new();
    let mut registered_types = BTreeSet::new();
    for (stem, text) in texts {
        // The crate root and a directory's `mod.rs` are always part of the tree; any other file
        // must be a module some line declares.
        if stem != "lib" && stem != "mod" && !modules.contains(&stem) {
            continue;
        }
        for line in text.lines() {
            let line = line.trim();
            if let Some(name) = declared(line) {
                declared_types.insert(name);
            }
            if let Some(name) = registered(line) {
                registered_types.insert(name);
            }
        }
    }

    let unregistered: Vec<&String> = declared_types.difference(&registered_types).collect();
    let unknown: Vec<&String> = registered_types.difference(&declared_types).collect();
    assert!(
        unregistered.is_empty() && unknown.is_empty(),
        "the install table and the crate's operators disagree:\n  \
         declared but not registered: {unregistered:?}\n  \
         registered but not declared: {unknown:?}"
    );
    Ok(())
}

#[test]
fn every_installed_operator_is_documented_for_the_programs() -> Result<(), String> {
    // `notes/programs/*.md` is what the `programs` swarm builds a package from: an operator with no
    // row there is a signature nobody can compile against. The check reads the repository's notes,
    // and a build of this crate outside the repository has none to compare with.
    let notes = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../notes/programs");
    if !notes.is_dir() {
        return Ok(());
    }
    let mut files = Vec::new();
    documents(&notes, &mut files)?;
    let mut text = String::new();
    for file in &files {
        text.push_str(
            &fs::read_to_string(file).map_err(|error| format!("{}: {error}", file.display()))?,
        );
        text.push('\n');
    }

    let undocumented: Vec<&str> = ops::operators()
        .iter()
        .copied()
        .filter(|name| !text.contains(&format!("`{name}`")))
        .collect();
    assert!(
        undocumented.is_empty(),
        "installed but documented in none of notes/programs/*.md: {undocumented:?}"
    );
    Ok(())
}

#[test]
fn the_names_this_build_installs_are_distinct() {
    // Two operators answering one name would make the second unloadable and the first ambiguous.
    let names = ops::operators();
    let distinct: BTreeSet<&&str> = names.iter().collect();
    assert_eq!(
        names.len(),
        distinct.len(),
        "{} names, {} distinct",
        names.len(),
        distinct.len()
    );
}

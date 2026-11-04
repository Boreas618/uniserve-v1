//! The module store (design section 5.3).
//!
//! A module is stored under a path made only of names: the operator, the
//! backend, the revision, the target and the descriptor key,
//!
//! ```text
//! <op>/<backend>/<revision>/<target>/<descriptor key>/
//! ```
//!
//! and a revision already fixes the sources, tuning data and toolchain the
//! module was built from (design sections 5.1 and 5.2). The same path is
//! therefore always the same module, wherever it was built, and nothing needs
//! a content digest.
//!
//! Roots: the first root is writable and holds the modules this host built;
//! the others are shared, read-only roots that other hosts mount, so a module
//! one host built is found by all of them.
//!
//! Publication: a build writes into a workspace directory and the store moves
//! that directory into place: every file is fsynced, the directory is renamed
//! atomically, and a `.committed` mark is written into it. Only a directory
//! with the mark is a module; a reader that sees a half-written directory
//! without one ignores it, and a build interrupted before the rename leaves
//! nothing a reader can find.

use std::io;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// The mark that makes a module directory a committed module.
pub const COMMITTED: &str = ".committed";

/// The path of one module in the store: pure names, no content.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModulePath {
    pub op: String,
    pub backend: String,
    pub revision: u32,
    pub target: String,
    pub descriptor: String,
}

impl ModulePath {
    /// A module path, checked: every name but the revision is one path
    /// component, so no name can point outside its root.
    pub fn new(
        op: impl Into<String>,
        backend: impl Into<String>,
        revision: u32,
        target: impl Into<String>,
        descriptor: impl Into<String>,
    ) -> Result<Self> {
        let path = ModulePath {
            op: op.into(),
            backend: backend.into(),
            revision,
            target: target.into(),
            descriptor: descriptor.into(),
        };
        for (name, value) in [
            ("operator", &path.op),
            ("backend", &path.backend),
            ("target", &path.target),
            ("descriptor", &path.descriptor),
        ] {
            if !is_name(value) {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{name} {value:?} is not one path component"),
                )));
            }
        }
        Ok(path)
    }

    /// The path relative to a store root: `<op>/<backend>/<revision>/<target>/<descriptor key>`.
    pub fn relative(&self) -> PathBuf {
        PathBuf::from(&self.op)
            .join(&self.backend)
            .join(self.revision.to_string())
            .join(&self.target)
            .join(&self.descriptor)
    }
}

impl std::fmt::Display for ModulePath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}/{}/{}/{}/{}",
            self.op, self.backend, self.revision, self.target, self.descriptor
        )
    }
}

/// One root of the store: a directory of modules, writable or read-only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Root {
    pub path: PathBuf,
    /// Whether this root takes the modules built on this host. Only the first
    /// root is writable; the shared ones are mounted read-only.
    pub writable: bool,
}

/// The roots a module is looked up in and published to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Store {
    roots: Vec<Root>,
}

impl Store {
    /// Opens a store over its roots, the first of which is writable.
    pub fn open(roots: Vec<Root>) -> Result<Store> {
        if roots.is_empty() {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a module store needs at least one root",
            )));
        }
        let writable = roots.iter().filter(|root| root.writable).count();
        if writable != 1 || !roots[0].writable {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a module store has exactly one writable root, and it comes first",
            )));
        }
        for root in &roots {
            std::fs::create_dir_all(&root.path)?;
        }
        Ok(Store { roots })
    }

    /// The roots, writable first.
    pub fn roots(&self) -> &[Root] {
        &self.roots
    }

    /// The directory a module is published to: the writable root's.
    pub fn directory(&self, path: &ModulePath) -> PathBuf {
        self.roots[0].path.join(path.relative())
    }

    /// Whether a committed module exists in any root.
    pub fn contains(&self, path: &ModulePath) -> bool {
        self.find(path).is_some()
    }

    /// The directory of a committed module, from the first root that has one.
    pub fn find(&self, path: &ModulePath) -> Option<PathBuf> {
        let relative = path.relative();
        self.roots.iter().find_map(|root| {
            let directory = root.path.join(&relative);
            directory.join(COMMITTED).is_file().then_some(directory)
        })
    }

    /// The files of a committed module, in name order, from the first root
    /// that has one. A file's name is its path inside the module's directory,
    /// so tuning data in a subdirectory keeps its place.
    pub fn files(&self, path: &ModulePath) -> Result<Option<Vec<StoreFile>>> {
        let Some(directory) = self.find(path) else {
            return Ok(None);
        };
        let mut files = Vec::new();
        collect(&directory, Path::new(""), &mut files)?;
        files.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(Some(files))
    }

    /// Publishes the files a build wrote as the module at `path`.
    ///
    /// `workspace` is the build's directory; `files` names the files in it
    /// that belong to the module, in the order they are stored. Every file is
    /// fsynced, the directory is renamed into place atomically, and the
    /// `.committed` mark is written last, so a reader never finds a module
    /// that is partly written.
    pub fn publish(&self, path: &ModulePath, workspace: &Path, files: &[String]) -> Result<()> {
        for name in files {
            if !is_relative_name(name) {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{name:?} is not a path inside a module directory"),
                )));
            }
        }
        let destination = self.roots[0].path.join(path.relative());
        let parent = destination.parent().ok_or_else(|| {
            Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{path} has no parent directory"),
            ))
        })?;
        std::fs::create_dir_all(parent)?;
        let staged = parent.join(format!(
            ".{}.staged-{}",
            path.descriptor,
            std::process::id()
        ));
        if staged.exists() {
            std::fs::remove_dir_all(&staged)?;
        }
        std::fs::create_dir_all(&staged)?;
        for name in files {
            let source = workspace.join(name);
            let destination = staged.join(name);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&source, &destination)?;
            std::fs::File::open(&destination)?.sync_all()?;
        }
        sync_directory(&staged)?;
        if destination.exists() {
            if destination.join(COMMITTED).is_file() {
                // A revision fixes a module's content, so a committed module
                // is never replaced (design section 5.3).
                std::fs::remove_dir_all(&staged)?;
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("{path} is already committed"),
                )));
            }
            // A build that died between the rename and the mark left a
            // directory no reader treats as a module; this publish replaces
            // it.
            std::fs::remove_dir_all(&destination)?;
        }
        std::fs::rename(&staged, &destination)?;
        sync_directory(parent)?;
        // The mark makes the directory a module. It is written after the
        // rename and fsynced with its directory, so a reader that finds it
        // finds every file.
        std::fs::File::create(destination.join(COMMITTED))?.sync_all()?;
        sync_directory(&destination)?;
        sync_directory(parent)?;
        Ok(())
    }

    /// Every committed module under the roots, from the first root that has
    /// it, in path order.
    pub fn list(&self) -> Result<Vec<ModulePath>> {
        let mut modules: Vec<ModulePath> = Vec::new();
        for root in &self.roots {
            for op in directories(&root.path)? {
                for backend in directories(&op)? {
                    for revision in directories(&backend)? {
                        let Some(revision) = revision
                            .file_name()
                            .and_then(|name| name.to_str())
                            .and_then(|name| name.parse::<u32>().ok())
                        else {
                            continue;
                        };
                        for target in directories(&revision_path(revision, &backend))? {
                            for descriptor in directories(&target)? {
                                if !descriptor.join(COMMITTED).is_file() {
                                    continue;
                                }
                                modules.push(ModulePath {
                                    op: file_name(&op),
                                    backend: file_name(&backend),
                                    revision,
                                    target: file_name(&target),
                                    descriptor: file_name(&descriptor),
                                });
                            }
                        }
                    }
                }
            }
        }
        // A module may be in several roots; the first root that has it is the
        // one a lookup finds, so a later duplicate is dropped.
        let mut seen = std::collections::HashSet::new();
        modules.retain(|module| seen.insert(module.clone()));
        modules.sort();
        Ok(modules)
    }
}

/// One file of a module: its name inside the module's directory, and where it
/// is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreFile {
    /// The file's path inside the module's directory, such as `module.cubin`
    /// or `tuning/tactics.json`.
    pub name: String,
    pub path: PathBuf,
}

/// Whether a build's file name is a path inside the module directory:
/// relative, and every component a name.
fn is_relative_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && name.split('/').all(|component| is_name(component))
}

/// Whether a store name is one path component: non-empty, no separator, and
/// neither `.` nor `..`.
fn is_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains('/') && !name.contains('\0')
}

/// Adds every file below a module's directory, with its name relative to it.
fn collect(directory: &Path, prefix: &Path, files: &mut Vec<StoreFile>) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        if prefix.as_os_str().is_empty() && name == COMMITTED {
            continue;
        }
        let path = entry.path();
        let relative = prefix.join(&name);
        if entry.file_type()?.is_dir() {
            collect(&path, &relative, files)?;
            continue;
        }
        files.push(StoreFile {
            name: relative.to_string_lossy().into_owned(),
            path,
        });
    }
    Ok(())
}

/// A path's last name, as a `String`. Every name the store makes is UTF-8.
fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn revision_path(revision: u32, backend: &Path) -> PathBuf {
    backend.join(revision.to_string())
}

/// The subdirectories of a directory, in name order.
fn directories(path: &Path) -> Result<Vec<PathBuf>> {
    let mut directories = Vec::new();
    if !path.is_dir() {
        return Ok(directories);
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        // A name starting with `.` is not a name the store makes: it is a
        // staging directory or a lock.
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        directories.push(entry.path());
    }
    directories.sort();
    Ok(directories)
}

/// Flushes a directory's entries, so a rename into it survives a crash.
fn sync_directory(path: &Path) -> io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

//! The module store (design section 5.3).
//!
//! A module is stored under a path made of names and one digest: the
//! operator, the backend, the revision, the digest of its build inputs, the
//! target and the descriptor key,
//!
//! ```text
//! <op>/<backend>/<revision>/<inputs>/<target>/<descriptor key>/
//! ```
//!
//! The inputs digest ([`crate::inputs::Inputs::digest`]) names everything that
//! determines the module's bytes beside the target and the key: the package's
//! files, the toolchain's version and the shared compile driver with its pins.
//! A revision number alone does not: two package roots number their own
//! installations, and a driver, a pin or a toolchain changes without one. The
//! same path is therefore always the same module, wherever it was built, and a
//! module built from other inputs is never found, so it is built again.
//!
//! Roots: the first root is writable and holds the modules this host built;
//! the others are shared, read-only roots that other hosts mount, so a module
//! one host built is found by all of them.
//!
//! Publication: a build writes into a workspace directory and the store copies
//! its files into a staging directory beside the module's place, writes the
//! `.committed` mark there, fsyncs every file and renames the directory into
//! place atomically. Only a directory with the mark is a module, and the mark
//! arrives with every file, so a reader never finds a partial module and a
//! publish interrupted before the rename leaves nothing a reader can find.
//!
//! The mark is the module's record: it states the version of the interface
//! the module's package implements, which `resolve` hands to the loader with
//! the files so that a loader built for another version refuses the module.
//!
//! A staging directory is owned by its writer through a lock it holds for the
//! staging's life ([`Staging`]): a staging directory whose lock no live
//! process holds is a crashed writer's, and the next publish beside it removes
//! it.

use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::snapshot::Files;
use crate::{Error, Result};

/// The mark that makes a module directory a committed module, and the
/// module's record: a TOML file stating [`INTERFACE_VERSION`].
pub const COMMITTED: &str = ".committed";

/// The key of the record that states the interface version a module was built
/// for.
pub const INTERFACE_VERSION: &str = "interface_version";

/// The key of a module's record that states the oldest version of its interface the module
/// implements, beside [`INTERFACE_VERSION`], its newest; a record without it states its newest
/// alone.
pub const INTERFACE_OLDEST: &str = "interface_oldest";

/// The prefix of a staging directory's name, and of its lock's.
const STAGING: &str = ".staged-";

/// The suffix of a staging directory's lock beside it.
const STAGING_LOCK: &str = ".lock";

/// The longest a module path's last segment may be, in bytes.
///
/// A descriptor key is the last segment of a module's path (design section 5.3: every module sits
/// under `<operator>/<backend>/<revision>/<inputs>/<target>/<descriptor key>`), so a key must be
/// one file name, and this is the filesystem's own cap rather than a policy: a longer name has no
/// directory to live in, whatever the store would like. Publishing stages under a name of its own,
/// short and independent of the key, so the key's length is the only thing measured here.
///
/// A key that reaches this bound is a design smell even when it fits: it usually names values that
/// are launch arguments rather than properties of the problem — a stride the caller varies, a
/// capacity it chooses — and the fix belongs in the descriptor, not in the store, which is why this
/// bound refuses only what cannot be stored.
pub const DESCRIPTOR_NAME_LIMIT: usize = 255;

/// The path of one module in the store: pure names, no content.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModulePath {
    pub op: String,
    pub backend: String,
    pub revision: u32,
    /// The digest of the revision's build inputs ([`crate::inputs`]), in hex.
    pub inputs: String,
    pub target: String,
    pub descriptor: String,
}

impl ModulePath {
    /// A module path, checked: every name but the revision is one path
    /// segment, so no name can point outside its root.
    pub fn new(
        op: impl Into<String>,
        backend: impl Into<String>,
        revision: u32,
        inputs: impl Into<String>,
        target: impl Into<String>,
        descriptor: impl Into<String>,
    ) -> Result<Self> {
        let path = ModulePath {
            op: op.into(),
            backend: backend.into(),
            revision,
            inputs: inputs.into(),
            target: target.into(),
            descriptor: descriptor.into(),
        };
        for (name, value) in [
            ("operator", &path.op),
            ("backend", &path.backend),
            ("inputs digest", &path.inputs),
            ("target", &path.target),
            ("descriptor", &path.descriptor),
        ] {
            if !is_name(value) {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{name} {value:?} is not one path segment"),
                )));
            }
        }
        if path.descriptor.len() > DESCRIPTOR_NAME_LIMIT {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "the descriptor key is {} bytes and a module path segment holds at most \
                     {DESCRIPTOR_NAME_LIMIT}: {}",
                    path.descriptor.len(),
                    path.descriptor
                ),
            )));
        }
        Ok(path)
    }

    /// The path relative to a store root:
    /// `<op>/<backend>/<revision>/<inputs>/<target>/<descriptor key>`.
    pub fn relative(&self) -> PathBuf {
        PathBuf::from(&self.op)
            .join(&self.backend)
            .join(self.revision.to_string())
            .join(&self.inputs)
            .join(&self.target)
            .join(&self.descriptor)
    }
}

impl std::fmt::Display for ModulePath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}/{}/{}/{}/{}/{}",
            self.op, self.backend, self.revision, self.inputs, self.target, self.descriptor
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
            recorded_version(&directory).map(|_| directory)
        })
    }

    /// The interface versions the committed module at `path` was built for, as
    /// its record states them, from the first root that has the module.
    pub fn interface(&self, path: &ModulePath) -> Option<service::Versions> {
        let relative = path.relative();
        self.roots
            .iter()
            .find_map(|root| recorded_version(&root.path.join(&relative)))
    }

    /// The files of a committed module, from the first root that has one:
    /// each file's name and where it is, in name order. A file's name is its
    /// path inside the module's directory, such as `module.cubin` or
    /// `tuning/tactics.json`, so tuning data in a subdirectory keeps its place.
    pub fn files(&self, path: &ModulePath) -> Result<Option<Files>> {
        let Some(directory) = self.find(path) else {
            return Ok(None);
        };
        let mut files = Files::new();
        collect(&directory, Path::new(""), &mut files)?;
        files.sort();
        Ok(Some(files))
    }

    /// Publishes the files a build wrote as the module at `path`, built for
    /// the versions `interface` of its package's interface.
    ///
    /// `workspace` is the build's directory; `files` names the files in it
    /// that belong to the module, in the order they are stored. The files and
    /// the `.committed` mark, which records the interface version, are fsynced
    /// in a staging directory that is renamed into place atomically, so a
    /// reader never finds a module that is partly written.
    pub fn publish(
        &self,
        path: &ModulePath,
        workspace: &Path,
        files: &[String],
        interface: service::Versions,
    ) -> Result<()> {
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
        // The staging name carries no part of the key: a key may be as long as a
        // file name allows, and a staging name that embedded it would quietly
        // lower that bound by the length of a suffix.
        let staging = Staging::new(parent)?;
        let staged = staging.path();
        for name in files {
            let source = workspace.join(name);
            let destination = staged.join(name);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&source, &destination)?;
            File::open(&destination)?.sync_all()?;
        }
        let mark = staged.join(COMMITTED);
        std::fs::write(
            &mark,
            format!(
                "{INTERFACE_VERSION} = {}\n{INTERFACE_OLDEST} = {}\n",
                interface.newest(),
                interface.oldest()
            ),
        )?;
        File::open(&mark)?.sync_all()?;
        sync_directory(staged)?;
        // A path fixes a module's content, so a committed module is never
        // replaced (design section 5.3): the rename refuses a destination that
        // holds one.
        staging
            .publish(&destination)
            .map_err(|error| match error.kind() {
                io::ErrorKind::AlreadyExists | io::ErrorKind::DirectoryNotEmpty => {
                    let held = if recorded_version(&destination).is_some() {
                        "is already committed"
                    } else {
                        "holds a directory that is no committed module (no readable mark)"
                    };
                    Error::Io(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        format!("{path} {held}"),
                    ))
                }
                _ => Error::Io(error),
            })
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
                        for inputs in directories(&revision_path(revision, &backend))? {
                            for target in directories(&inputs)? {
                                for descriptor in directories(&target)? {
                                    if recorded_version(&descriptor).is_none() {
                                        continue;
                                    }
                                    modules.push(ModulePath {
                                        op: file_name(&op),
                                        backend: file_name(&backend),
                                        revision,
                                        inputs: file_name(&inputs),
                                        target: file_name(&target),
                                        descriptor: file_name(&descriptor),
                                    });
                                }
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

/// The interface versions the mark of the module directory `directory` records,
/// or `None` when the directory is no committed module. A mark that states no
/// oldest version records its newest alone.
fn recorded_version(directory: &Path) -> Option<service::Versions> {
    let record = std::fs::read_to_string(directory.join(COMMITTED)).ok()?;
    let record: toml::Table = toml::from_str(&record).ok()?;
    let version = |key: &str| u32::try_from(record.get(key)?.as_integer()?).ok();
    let newest = version(INTERFACE_VERSION)?;
    service::Versions::stated(version(INTERFACE_OLDEST).unwrap_or(newest), newest)
}

/// Whether a build's file name is a path inside the module directory:
/// relative, and every segment a name.
fn is_relative_name(name: &str) -> bool {
    !name.is_empty() && !name.starts_with('/') && name.split('/').all(is_name)
}

/// Whether a store name is one path segment: non-empty, no separator, and
/// neither `.` nor `..`.
fn is_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains('/') && !name.contains('\0')
}

/// Adds every file below a module's directory, with its name relative to it.
fn collect(directory: &Path, prefix: &Path, files: &mut Files) -> Result<()> {
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
        files.push((relative.to_string_lossy().into_owned(), path));
    }
    Ok(())
}

/// A path's last name, as a `String`. Every name the Build Server makes is
/// UTF-8.
pub(crate) fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn revision_path(revision: u32, backend: &Path) -> PathBuf {
    backend.join(revision.to_string())
}

/// The subdirectories of a directory, in name order, without the ones whose
/// name starts with `.`: those are a writer's staging directories, not names
/// the Build Server publishes.
pub(crate) fn directories(path: &Path) -> Result<Vec<PathBuf>> {
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

/// A directory a writer fills beside the place it publishes it to, and the
/// lock that makes the writer its owner.
///
/// The lock is a file beside the directory (`.staged-<name>.lock`), created and
/// locked (`flock`) before the directory exists and held until the staging is
/// published or dropped, so a staging directory whose lock is free is a crashed
/// writer's. Every new staging sweeps those beside it ([`Staging::sweep`]); a
/// live writer's staging is never touched, whatever process or host made it.
pub(crate) struct Staging {
    directory: PathBuf,
    lock: PathBuf,
    /// The locked file; closing it releases the lock.
    _held: File,
    published: bool,
}

impl Staging {
    /// A new staging directory in `parent`, after the crashed writers'
    /// leftovers there are removed.
    ///
    /// # Errors
    ///
    /// Fails when the directory, its lock or a sweep cannot be made.
    pub(crate) fn new(parent: &Path) -> Result<Staging> {
        std::fs::create_dir_all(parent)?;
        Self::sweep(parent)?;
        loop {
            let named = tempfile::Builder::new()
                .prefix(STAGING)
                .suffix(STAGING_LOCK)
                .tempfile_in(parent)?;
            let (held, lock) = named.keep().map_err(|error| Error::Io(error.error))?;
            flock(&held, libc::LOCK_EX)?;
            // A sweeper may have taken and removed the lock between its creation
            // and this lock: the file this process holds is then no longer the one
            // at the path, and another is made.
            if !same_file(&held, &lock) {
                continue;
            }
            let name = file_name(&lock);
            let directory = parent.join(name.trim_end_matches(STAGING_LOCK));
            std::fs::create_dir(&directory)?;
            return Ok(Staging {
                directory,
                lock,
                _held: held,
                published: false,
            });
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.directory
    }

    /// Renames the staging directory into place at `destination`, which must
    /// not hold a non-empty directory, and fsyncs its parent.
    ///
    /// # Errors
    ///
    /// The rename's error: `AlreadyExists` or `DirectoryNotEmpty` when the
    /// destination is taken.
    pub(crate) fn publish(mut self, destination: &Path) -> io::Result<()> {
        std::fs::rename(&self.directory, destination)?;
        self.published = true;
        if let Some(parent) = destination.parent() {
            sync_directory(parent)?;
        }
        Ok(())
    }

    /// Removes the staging directories in `parent` whose lock no writer
    /// holds, with their locks.
    fn sweep(parent: &Path) -> Result<()> {
        for entry in std::fs::read_dir(parent)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            let Some(staged) = name
                .strip_prefix(STAGING)
                .and_then(|rest| rest.strip_suffix(STAGING_LOCK))
            else {
                continue;
            };
            let lock = parent.join(&name);
            let Ok(file) = File::open(&lock) else {
                continue;
            };
            // A writer holds its lock for the staging's whole life.
            if flock(&file, libc::LOCK_EX | libc::LOCK_NB).is_err() {
                continue;
            }
            let directory = parent.join(format!("{STAGING}{staged}"));
            if directory.is_dir() {
                std::fs::remove_dir_all(&directory)?;
            }
            // Another publish beside this one may have swept the same leftover.
            match std::fs::remove_file(&lock) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(Error::Io(error)),
            }
        }
        Ok(())
    }
}

impl Drop for Staging {
    /// An unpublished staging directory is removed with its lock; a published
    /// one leaves only its lock to remove.
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
        let _ = std::fs::remove_file(&self.lock);
    }
}

/// Holds the lock file `.lock` of the directory `root` for the life of the
/// returned file: the one writer of what the directory holds, such as an
/// upstream Build Server's package root.
///
/// # Errors
///
/// `WouldBlock` when another live process holds the lock, or the lock's
/// creation error.
pub(crate) fn hold(root: &Path) -> Result<File> {
    std::fs::create_dir_all(root)?;
    let path = root.join(".lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)?;
    flock(&file, libc::LOCK_EX | libc::LOCK_NB).map_err(|error| {
        Error::Io(io::Error::new(
            error.kind(),
            format!("{} is held by another writer: {error}", path.display()),
        ))
    })?;
    Ok(file)
}

/// Takes `operation` (`LOCK_EX`, with `LOCK_NB` for a try) on `file`.
fn flock(file: &File, operation: libc::c_int) -> io::Result<()> {
    // SAFETY: flock on a descriptor `file` owns for the call.
    if unsafe { libc::flock(file.as_raw_fd(), operation) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Whether the open `file` is the file at `path` now.
fn same_file(file: &File, path: &Path) -> bool {
    match (file.metadata(), std::fs::metadata(path)) {
        (Ok(held), Ok(named)) => held.dev() == named.dev() && held.ino() == named.ino(),
        _ => false,
    }
}

/// Flushes a directory's entries, so a rename into it survives a crash.
fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

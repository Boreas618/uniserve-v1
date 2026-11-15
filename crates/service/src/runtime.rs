//! Where a process keeps its runtime files: the user's runtime directory and the instance
//! directories in it.
//!
//! Every runtime file of UniServe lives under `$XDG_RUNTIME_DIR/uniserve` ([`RuntimeDirectory`]),
//! a directory only its user enters (mode 0700). The XDG Base Directory specification makes the
//! runtime directory the place for a user's sockets and other non-essential runtime files, owned
//! by the user, mode 0700, and removed at the user's logout; a process run without one has no
//! place for them and is refused rather than given a shared fallback.
//!
//! The host services keep their files beside their socket in that directory
//! ([`crate::host`]). Every other server owns an instance directory there ([`Instance`]): it
//! creates the directory under a name no other instance holds, every socket and file of the
//! instance lives inside it, and its owner removes it when the instance ends. A caller never
//! derives an instance's paths: the instance reports them in its readiness record
//! ([`crate::Readiness`]).

use std::io;
use std::os::unix::fs::{DirBuilderExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};

/// The environment variable that names the user's runtime directory.
const XDG_RUNTIME_DIR: &str = "XDG_RUNTIME_DIR";

/// The name of UniServe's directory inside the user's runtime directory.
const UNISERVE: &str = "uniserve";

/// The mode of every directory UniServe creates for its runtime files: its owner alone.
const PRIVATE: u32 = 0o700;

/// UniServe's directory inside the user's runtime directory: `$XDG_RUNTIME_DIR/uniserve`, mode
/// 0700.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDirectory(PathBuf);

impl RuntimeDirectory {
    /// The calling user's: `uniserve` inside the directory `XDG_RUNTIME_DIR` names.
    ///
    /// # Errors
    ///
    /// `NotFound` when `XDG_RUNTIME_DIR` is unset, empty or not absolute, and what
    /// [`RuntimeDirectory::within`] fails with.
    pub fn of_user() -> io::Result<Self> {
        let runtime = std::env::var_os(XDG_RUNTIME_DIR)
            .map(PathBuf::from)
            .filter(|runtime| runtime.is_absolute())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "{XDG_RUNTIME_DIR} names no absolute directory: UniServe keeps its \
                         sockets in the user's runtime directory, which the login session \
                         provides (or which the caller creates, mode 0700, and names there)"
                    ),
                )
            })?;
        Self::within(&runtime)
    }

    /// UniServe's directory inside the runtime directory `runtime`, created with mode 0700 when
    /// absent.
    ///
    /// # Errors
    ///
    /// `NotFound` when `runtime` is not an existing directory: the runtime directory is the
    /// session's, and UniServe never creates it. Otherwise what creating or restricting the
    /// directory fails with.
    pub fn within(runtime: &Path) -> io::Result<Self> {
        if !runtime.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("the runtime directory {} does not exist", runtime.display()),
            ));
        }
        let directory = runtime.join(UNISERVE);
        create_private(&directory)?;
        Ok(Self(directory))
    }

    /// The directory's path.
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// The socket of the host service `service` in this directory: `<service>.sock`.
    pub fn socket(&self, service: &str) -> PathBuf {
        self.0.join(format!("{service}.sock"))
    }
}

/// A directory a process owns for one instance of a server, inside the runtime directory, with
/// every socket and file of the instance in it. The owner removes it with everything in it when
/// the instance ends, unless it handed the instance to a successor ([`Instance::hand_over`]).
#[derive(Debug)]
pub struct Instance {
    path: PathBuf,
    /// Whether this process removes the directory when it drops the instance.
    owned: bool,
}

impl Instance {
    /// Creates a new instance directory of `kind` in `runtime`, mode 0700, named `<kind>.<six
    /// random characters>`: the name is chosen by creating the directory exclusively, so no
    /// other instance of any process holds it.
    ///
    /// # Errors
    ///
    /// What creating the directory fails with.
    pub fn create(runtime: &RuntimeDirectory, kind: &str) -> io::Result<Self> {
        let directory = tempfile::Builder::new()
            .prefix(&format!("{kind}."))
            .permissions(std::fs::Permissions::from_mode(PRIVATE))
            .tempdir_in(runtime.path())?;
        Ok(Self {
            path: directory.keep(),
            owned: true,
        })
    }

    /// Goes on with the instance at `path` a predecessor handed over, which this process now
    /// removes when it ends.
    pub fn resume(path: PathBuf) -> Self {
        Self { path, owned: true }
    }

    /// The directory's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The path of the file or socket `name` in the instance.
    pub fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    /// The directory's path, for a successor to go on with ([`Instance::resume`]); this process
    /// no longer removes it.
    pub fn hand_over(mut self) -> PathBuf {
        self.owned = false;
        self.path.clone()
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if self.owned {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// Creates `directory` with mode 0700 when absent, and makes an existing one 0700: a socket in it
/// is then reachable by its owner alone, whatever the socket file's own mode.
fn create_private(directory: &Path) -> io::Result<()> {
    match std::fs::DirBuilder::new().mode(PRIVATE).create(directory) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            std::fs::set_permissions(directory, std::fs::Permissions::from_mode(PRIVATE))
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::MetadataExt as _;

    use super::*;

    /// A runtime directory that does not exist is refused, and no directory is created for it.
    #[test]
    fn a_missing_runtime_directory_is_refused() {
        let parent = tempfile::tempdir().unwrap();
        let missing = parent.path().join("missing");

        let refused = RuntimeDirectory::within(&missing).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::NotFound);
        assert!(!missing.exists());
    }

    /// Instances of one kind get directories of their own, private to their owner, which their
    /// owner removes when they end unless a successor took them over.
    #[test]
    fn an_instance_owns_a_private_directory_of_its_own() {
        let session = tempfile::tempdir().unwrap();
        let runtime = RuntimeDirectory::within(session.path()).unwrap();
        assert_eq!(
            std::fs::metadata(runtime.path()).unwrap().mode() & 0o777,
            PRIVATE
        );

        let first = Instance::create(&runtime, "daemon").unwrap();
        let second = Instance::create(&runtime, "daemon").unwrap();
        assert_ne!(first.path(), second.path());
        assert_eq!(
            std::fs::metadata(first.path()).unwrap().mode() & 0o777,
            PRIVATE
        );
        std::fs::write(first.file("admin.sock"), b"").unwrap();

        let ended = first.path().to_owned();
        drop(first);
        assert!(!ended.exists());

        let handed = second.hand_over();
        assert!(handed.exists());
        drop(Instance::resume(handed.clone()));
        assert!(!handed.exists());
    }
}

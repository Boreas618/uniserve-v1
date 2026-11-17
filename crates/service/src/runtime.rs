//! Where a process keeps its runtime files: the user's runtime directory and the instance
//! directories in it.
//!
//! Every runtime file of UniServe lives under `$XDG_RUNTIME_DIR/uniserve` ([`RuntimeDirectory`]),
//! a directory only its user enters (mode 0700). The XDG Base Directory specification makes the
//! runtime directory the place for a user's sockets and other non-essential runtime files, owned
//! by the user, mode 0700, and removed at the user's logout. A process run without one, such as
//! one started outside a login session, falls back, as the specification directs, to a directory
//! of the same capabilities with a warning: the session directory `/run/user/<uid>` when the
//! system has one for the user, else `uniserve-<uid>` in the system's temporary directory, which
//! the user alone owns and enters ([`fallback`]). Every process of the user without the variable
//! finds the same directory, so its services and their clients meet there.
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

/// Where the login sessions of a system keep each user's runtime directory, `<uid>` inside it.
const SESSION_RUNTIME: &str = "/run/user";

/// The name of UniServe's directory inside the user's runtime directory.
const UNISERVE: &str = "uniserve";

/// The mode of every directory UniServe creates for its runtime files: its owner alone.
const PRIVATE: u32 = 0o700;

/// UniServe's directory inside the user's runtime directory: `$XDG_RUNTIME_DIR/uniserve`, mode
/// 0700.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDirectory(PathBuf);

impl RuntimeDirectory {
    /// The calling user's: `uniserve` inside the directory `XDG_RUNTIME_DIR` names, or, when it
    /// names no absolute directory, inside the user's fallback runtime directory ([`fallback`]),
    /// with a warning.
    ///
    /// # Errors
    ///
    /// What [`fallback`] and [`RuntimeDirectory::within`] fail with.
    pub fn of_user() -> io::Result<Self> {
        let named = std::env::var_os(XDG_RUNTIME_DIR)
            .map(PathBuf::from)
            .filter(|runtime| runtime.is_absolute());
        let runtime = match named {
            Some(runtime) => runtime,
            None => {
                let uid = rustix::process::getuid().as_raw();
                let runtime = fallback(Path::new(SESSION_RUNTIME), &std::env::temp_dir(), uid)?;
                // Once per process: a process resolves its runtime directory for every socket.
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        runtime = %runtime.display(),
                        "{XDG_RUNTIME_DIR} names no absolute directory, so UniServe keeps its \
                         sockets in this runtime directory of the user's instead"
                    );
                });
                runtime
            }
        };
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

/// The runtime directory of the user `uid` when no `XDG_RUNTIME_DIR` names one: `<uid>` in the
/// session runtime directories `sessions` (`/run/user`) when it is a directory the user owns and
/// alone enters, else `uniserve-<uid>` in the temporary directory `temporary`, created with mode
/// 0700 when absent. A directory shared by the system's users may hold an entry of that name
/// that another user made, so an existing one is used only when it is a directory, not a symbolic
/// link, that the user owns and alone enters.
///
/// # Errors
///
/// `PermissionDenied` naming the directory when the one in `temporary` exists and is not such a
/// directory; otherwise what creating it fails with.
fn fallback(sessions: &Path, temporary: &Path, uid: u32) -> io::Result<PathBuf> {
    let session = sessions.join(uid.to_string());
    if private_to(&session, uid)? {
        return Ok(session);
    }

    let directory = temporary.join(format!("{UNISERVE}-{uid}"));
    match std::fs::DirBuilder::new().mode(PRIVATE).create(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    if private_to(&directory, uid)? {
        Ok(directory)
    } else {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "{} is not a directory that user {uid} owns and alone enters, so it cannot be \
                 that user's runtime directory: remove it, or name a runtime directory in \
                 {XDG_RUNTIME_DIR}",
                directory.display()
            ),
        ))
    }
}

/// Whether `path` is a directory, not a symbolic link, that the user `uid` owns and no other user
/// may enter or read; `false` when nothing is there.
fn private_to(path: &Path, uid: u32) -> io::Result<bool> {
    use std::os::unix::fs::MetadataExt as _;

    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            Ok(metadata.is_dir() && metadata.uid() == uid && metadata.mode() & 0o077 == 0)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
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

    /// Without `XDG_RUNTIME_DIR`, the user's session directory is the runtime directory when the
    /// system has one that is the user's alone; else a directory of the user's own in the
    /// temporary directory is, created private and found again by every later process.
    #[test]
    fn a_user_without_a_runtime_directory_falls_back_to_a_private_one() {
        let uid = rustix::process::getuid().as_raw();
        let root = tempfile::tempdir().unwrap();
        let (sessions, temporary) = (root.path().join("run-user"), root.path().join("tmp"));
        std::fs::create_dir_all(&temporary).unwrap();

        let created = fallback(&sessions, &temporary, uid).unwrap();
        assert_eq!(created, temporary.join(format!("uniserve-{uid}")));
        assert_eq!(std::fs::metadata(&created).unwrap().mode() & 0o777, PRIVATE);
        assert_eq!(fallback(&sessions, &temporary, uid).unwrap(), created);

        let session = sessions.join(uid.to_string());
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(PRIVATE)
            .create(&session)
            .unwrap();
        assert_eq!(fallback(&sessions, &temporary, uid).unwrap(), session);
    }

    /// A fallback directory in the shared temporary directory that others may enter, or that is a
    /// symbolic link, is refused, and nothing is changed: another user may have put it there.
    #[test]
    fn a_fallback_directory_others_may_enter_is_refused() {
        let uid = rustix::process::getuid().as_raw();
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("run-user");

        let open = root.path().join("open");
        std::fs::create_dir_all(open.join(format!("uniserve-{uid}"))).unwrap();
        std::fs::set_permissions(
            open.join(format!("uniserve-{uid}")),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        let refused = fallback(&sessions, &open, uid).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(
            std::fs::metadata(open.join(format!("uniserve-{uid}")))
                .unwrap()
                .mode()
                & 0o777,
            0o755
        );

        let linked = root.path().join("linked");
        let target = root.path().join("elsewhere");
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(PRIVATE)
            .create(&target)
            .unwrap();
        std::fs::create_dir_all(&linked).unwrap();
        std::os::unix::fs::symlink(&target, linked.join(format!("uniserve-{uid}"))).unwrap();
        let refused = fallback(&sessions, &linked, uid).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::PermissionDenied);
    }

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

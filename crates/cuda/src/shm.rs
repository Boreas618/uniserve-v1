//! The shared memory the CUDA driver and NCCL leave in `/dev/shm` when a
//! process ends without destroying its contexts and communicators, and its
//! removal.
//!
//! A process that is killed or crashes leaves the driver's interprocess-event
//! memory, named for its process id ([`remove_event_memory`]), and its NCCL
//! communicators' proxy memory, under random names its mappings show
//! ([`NcclSegments`]). The process that reaps it removes both while its
//! zombie still holds the process id, and a worker that ends without its
//! orderly close removes its own as the last thing it does. Neither needs
//! CUDA: they encode the libraries' naming.

/// Where POSIX shared-memory objects live on Linux.
const SHARED_MEMORY: &str = "/dev/shm";

/// Removes the shared memory the CUDA driver created for the interprocess
/// events of process `pid`, run by user `uid`, and returns the paths it
/// removed. It needs no CUDA: it encodes the driver's naming of that memory,
/// observed with driver 580.82.07 by tracing `shm_open` and `shm_unlink`
/// across processes, users, threads and contexts.
///
/// The driver backs each context's interprocess events with one POSIX
/// shared-memory object, which it creates exclusively (`O_CREAT | O_EXCL`,
/// mode 0600) with the context's first such event, and names
/// `cuda.shm.<uid>.<pid>.<n>`: the creating process's user id and process id
/// in lowercase hexadecimal (user 65534 is `fffe`), and the context's
/// ordinal in that process, from 1, in decimal. The process id is the
/// process's (`getpid`), whichever of its threads created the event, and a
/// process that opens another's event handle opens the other's object by
/// that name and creates none of its own. The driver unlinks the object only
/// when the context is destroyed (`PrimaryContext::reset_all`),
/// which a process that is killed or crashes never does, so the parent that
/// reaps such a process removes it instead.
///
/// The caller calls this after `pid` exited and before reaping it: until the
/// process is reaped its zombie holds the process id, so no live process can
/// own a name with that id. A process may also remove its own names as the
/// last thing before it ends without destroying its contexts; removing a
/// name leaves every mapping of the object valid until its holder ends. Exactly the names of that form with this user and
/// process id are removed. Every matching object is attempted; the first
/// failure other than an object already gone is returned.
pub fn remove_event_memory(uid: u32, pid: u32) -> std::io::Result<Vec<std::path::PathBuf>> {
    let owned = |name: &str| {
        let mut fields = name.split('.');
        let (Some("cuda"), Some("shm"), Some(user), Some(process), Some(ordinal), None) = (
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        ) else {
            return false;
        };
        user == format!("{uid:x}")
            && process == format!("{pid:x}")
            && !ordinal.is_empty()
            && ordinal.bytes().all(|byte| byte.is_ascii_digit())
    };

    let mut removed = Vec::new();
    let mut failure = None;
    for entry in std::fs::read_dir(SHARED_MEMORY)? {
        let entry = entry?;
        if !entry.file_name().to_str().is_some_and(owned) {
            continue;
        }
        let path = entry.path();
        match std::fs::remove_file(&path) {
            Ok(()) => removed.push(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                failure.get_or_insert(error);
            }
        }
    }
    failure.map_or(Ok(removed), Err)
}

/// Where NCCL creates its shared-memory objects: `/dev/shm/nccl-<random>`.
const NCCL_PREFIX: &str = "/dev/shm/nccl-";

/// NCCL's named shared-memory objects that a process maps, each by path and
/// inode.
///
/// This encodes NCCL's naming, observed with NCCL 2.30.4 by tracing
/// `mkstemp`, `open` and `unlink` across the ranks of a group. NCCL creates
/// its shared memory under random names (`/dev/shm/nccl-XXXXXX`). A
/// connection buffer is created by one rank and unlinked by its peer once
/// the peer has attached; a pool per communicator is created by the
/// communicator's proxy thread, mapped by its own thread too, and unlinked
/// only when the communicator is destroyed or aborted. A process that ends
/// without either therefore leaves its pools' names, which carry no process
/// id: the names still linked among a process's mappings
/// (`/proc/<pid>/maps`, where an unlinked one reads `(deleted)`) are its own.
/// Every one a group's ranks created was created before they reported
/// ready, none while they served. The inode tells an object from a later
/// one of the same name, so a removal never touches another's.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NcclSegments(Vec<(std::path::PathBuf, u64)>);

impl NcclSegments {
    /// The NCCL objects process `pid` (this process for `None`) maps under
    /// names still linked.
    pub fn mapped_by(pid: Option<u32>) -> std::io::Result<Self> {
        let maps = match pid {
            Some(pid) => std::fs::read_to_string(format!("/proc/{pid}/maps"))?,
            None => std::fs::read_to_string("/proc/self/maps")?,
        };
        let mut segments = Self::default();
        for line in maps.lines() {
            // address, permissions, offset, device, inode, path and, for an
            // unlinked object, "(deleted)".
            let fields: Vec<&str> = line.split_whitespace().collect();
            let (Some(inode), Some(path), None) = (fields.get(4), fields.get(5), fields.get(6))
            else {
                continue;
            };
            if !path.starts_with(NCCL_PREFIX) {
                continue;
            }
            if let Ok(inode) = inode.parse() {
                segments.add(std::path::PathBuf::from(path), inode);
            }
        }
        Ok(segments)
    }

    /// Adds the objects of `other` that are not held yet.
    pub fn extend(&mut self, other: Self) {
        for (path, inode) in other.0 {
            self.add(path, inode);
        }
    }

    fn add(&mut self, path: std::path::PathBuf, inode: u64) {
        if !self.0.contains(&(path.clone(), inode)) {
            self.0.push((path, inode));
        }
    }

    /// Unlinks every object whose name still names it, by inode, and
    /// returns the paths removed. Unlinking removes the name only: every
    /// mapping of the object stays valid until its holder ends. Every object
    /// is attempted; the first failure other than one already gone is
    /// returned.
    pub fn remove(&self) -> std::io::Result<Vec<std::path::PathBuf>> {
        use std::os::unix::fs::MetadataExt;

        let mut removed = Vec::new();
        let mut failure = None;
        for (path, inode) in &self.0 {
            let current = match std::fs::metadata(path) {
                Ok(metadata) => metadata.ino(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    failure.get_or_insert(error);
                    continue;
                }
            };
            if current != *inode {
                continue;
            }
            match std::fs::remove_file(path) {
                Ok(()) => removed.push(path.clone()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        failure.map_or(Ok(removed), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shared-memory object created under NCCL's prefix and mapped by
    /// this process; unmapped when dropped.
    #[cfg(target_os = "linux")]
    struct Mapped {
        path: std::path::PathBuf,
        address: *mut libc::c_void,
    }

    #[cfg(target_os = "linux")]
    impl Mapped {
        fn new(name: &str) -> Self {
            use std::os::fd::AsRawFd;

            let path = std::path::PathBuf::from(format!("{NCCL_PREFIX}{name}"));
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            file.set_len(4096).unwrap();
            // SAFETY: maps one page of the file just sized, shared.
            let address = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    4096,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(address, libc::MAP_FAILED);
            Self { path, address }
        }
    }

    #[cfg(target_os = "linux")]
    impl Drop for Mapped {
        fn drop(&mut self) {
            // SAFETY: unmaps the page `new` mapped.
            unsafe { libc::munmap(self.address, 4096) };
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// Of the NCCL objects this process maps, those still linked are held
    /// and removed by name, an unlinked one is not held, and an object that
    /// replaced a held one under its name is left.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_process_loses_exactly_the_nccl_names_it_maps() {
        let tag = std::process::id();
        let linked = Mapped::new(&format!("test-{tag}-linked"));
        let unlinked = Mapped::new(&format!("test-{tag}-unlinked"));
        std::fs::remove_file(&unlinked.path).unwrap();
        let replaced = Mapped::new(&format!("test-{tag}-replaced"));

        let held = NcclSegments::mapped_by(None).unwrap();
        // The replacement keeps the name under a new inode.
        std::fs::remove_file(&replaced.path).unwrap();
        std::fs::write(&replaced.path, b"").unwrap();

        let paths: Vec<&std::path::PathBuf> = held.0.iter().map(|(path, _)| path).collect();
        assert!(paths.contains(&&linked.path));
        assert!(!paths.contains(&&unlinked.path));
        let removed = held.remove().unwrap();
        assert!(removed.contains(&linked.path));
        assert!(!linked.path.exists());
        assert!(replaced.path.exists());
    }

    /// Of the objects named like the driver's, only those of the exited
    /// process's user and id are removed: not a live process's, another
    /// user's, ones with the ids in decimal, or a name of another form.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_dead_process_loses_exactly_its_event_memory() {
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = child.id();
        // Its exit is observed without reaping it, so its id stays taken.
        // SAFETY: an all-zero `siginfo_t` is valid; the call fills it.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        // SAFETY: waits for this test's own child without reaping it.
        let waited =
            unsafe { libc::waitid(libc::P_PID, pid, &mut info, libc::WEXITED | libc::WNOWAIT) };
        assert_eq!(waited, 0);

        // SAFETY: getuid has no preconditions.
        let uid = unsafe { libc::getuid() };
        let path = |name: String| std::path::Path::new(SHARED_MEMORY).join(name);
        let dead = [
            path(format!("cuda.shm.{uid:x}.{pid:x}.1")),
            path(format!("cuda.shm.{uid:x}.{pid:x}.2")),
        ];
        let kept = [
            path(format!("cuda.shm.{uid:x}.{:x}.1", std::process::id())),
            path(format!("cuda.shm.{:x}.{pid:x}.1", uid + 1)),
            path(format!("cuda.shm.{uid}.{pid}.1")),
            path(format!("cuda.shm.{uid:x}.{pid:x}")),
            path(format!("cuda.shm.{uid:x}.{pid:x}.1.part")),
        ];
        // Created exclusively, so that an object of a live process's is
        // never touched.
        for file in dead.iter().chain(&kept) {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(file)
                .unwrap();
        }

        let removed = remove_event_memory(uid, pid);
        child.wait().unwrap();
        let survived: Vec<bool> = kept.iter().map(|file| file.exists()).collect();
        for file in dead.iter().chain(&kept) {
            let _ = std::fs::remove_file(file);
        }

        let mut removed = removed.unwrap();
        removed.sort();
        assert_eq!(removed, dead);
        assert_eq!(survived, [true; 5]);
    }
}

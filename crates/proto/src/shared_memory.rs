//! Physically backed shared-memory allocations: anonymous memory files ([`anonymous`]), which
//! cross processes only by descriptor and which the kernel frees with the last descriptor or
//! mapping, and named POSIX objects ([`SharedMemory`]), which a process on the same host opens by
//! name.

use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::atomic::{AtomicU64, Ordering};

/// Owns a shared-memory name. Open descriptors and mappings retain the kernel
/// allocation independently; dropping this owner prevents subsequent opens.
#[derive(Debug)]
pub struct SharedMemory {
    name: String,
    size: usize,
}

impl SharedMemory {
    /// Open an existing segment without taking responsibility for unlinking.
    pub fn open(name: &str, writable: bool) -> io::Result<File> {
        let name = CString::new(format!("/{}", name.trim_start_matches('/')))?;
        let access = if writable {
            libc::O_RDWR
        } else {
            libc::O_RDONLY
        };

        // SAFETY: name is NUL-terminated; a successful call returns an owned fd.
        let descriptor = unsafe { libc::shm_open(name.as_ptr(), access | libc::O_CLOEXEC, 0) };
        if descriptor < 0 {
            return Err(io::Error::last_os_error());
        }

        // SAFETY: shm_open succeeded, so `descriptor` is a new open descriptor that
        // nothing else owns; the returned `File` becomes its only owner.
        Ok(unsafe { File::from_raw_fd(descriptor) })
    }

    /// Allocate physical storage under a fresh name and return its writable descriptor. The
    /// caller may close the descriptor as soon as writing or mapping ends.
    pub fn create(size: usize) -> io::Result<(Self, File)> {
        static NEXT: AtomicU64 = AtomicU64::new(0);

        if size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "shared memory must not be empty",
            ));
        }

        let length = libc::off_t::try_from(size).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "shared memory is too large")
        })?;
        let name = format!(
            "uniserve-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let path = CString::new(format!("/{name}"))?;

        // SAFETY: path is a NUL-terminated name; the returned fd is owned here.
        let descriptor = unsafe {
            libc::shm_open(
                path.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR | libc::O_CLOEXEC,
                0o600,
            )
        };
        if descriptor < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: shm_open returned one new descriptor owned by this call.
        let file = unsafe { File::from_raw_fd(descriptor) };
        let storage = Self { name, size };
        reserve(&file, length)?;
        Ok((storage, file))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Transfer unlink responsibility to a receiving process.
    pub fn into_name(mut self) -> String {
        std::mem::take(&mut self.name)
    }
}

/// Allocates `size` bytes of physically reserved anonymous shared memory (`memfd_create`) and
/// returns its writable descriptor.
///
/// The memory has no name: it reaches another process only as a descriptor passed beside a
/// message (`SCM_RIGHTS`), and the kernel frees it once the last descriptor and the last mapping
/// of it are gone, so nothing outlives a process that crashes.
///
/// # Errors
///
/// Fails when `size` is zero or does not fit a file offset, or when the kernel refuses the file or
/// its storage.
pub fn anonymous(size: usize) -> io::Result<File> {
    if size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "shared memory must not be empty",
        ));
    }
    let length = libc::off_t::try_from(size)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "shared memory is too large"))?;

    // SAFETY: the label is a NUL-terminated literal; a successful call returns an owned
    // descriptor.
    let descriptor = unsafe { libc::memfd_create(c"uniserve-media".as_ptr(), libc::MFD_CLOEXEC) };
    if descriptor < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: memfd_create returned one new descriptor that nothing else owns.
    let file = unsafe { File::from_raw_fd(descriptor) };
    reserve(&file, length)?;
    Ok(file)
}

/// Reserves `length` bytes of pages for `file` before anyone maps it writable: a sparse
/// allocation can otherwise raise SIGBUS during a later mapped write.
fn reserve(file: &File, length: libc::off_t) -> io::Result<()> {
    // posix_fallocate returns its error number directly, without errno.
    // SAFETY: `file` keeps the descriptor open for the call, which takes no pointers.
    let error = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, length) };
    if error != 0 {
        return Err(io::Error::from_raw_os_error(error));
    }
    Ok(())
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        if !self.name.is_empty()
            && let Ok(path) = CString::new(format!("/{}", self.name))
        {
            // SAFETY: this owner created the NUL-terminated POSIX name.
            unsafe { libc::shm_unlink(path.as_ptr()) };
        }
    }
}

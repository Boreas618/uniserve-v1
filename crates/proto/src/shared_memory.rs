//! Physically backed anonymous memory files ([`anonymous`]): they have no name, cross processes
//! only by descriptor, and the kernel frees them with their last descriptor or mapping. A file
//! whose contents are final is sealed ([`seal`]), so no holder of a descriptor can change them.

use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};

/// The seals of a file whose contents are final: no write, no growth, no shrinking, and no seal
/// removed or added after them.
const FINAL_SEALS: libc::c_int =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

/// Allocates `size` bytes of physically reserved anonymous shared memory (`memfd_create`) and
/// returns its writable descriptor.
///
/// The memory has no name: it reaches another process only as a descriptor passed beside a
/// message (`SCM_RIGHTS`), and the kernel frees it once the last descriptor and the last mapping
/// of it are gone, so nothing outlives a process that crashes. The file admits seals ([`seal`]).
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
    let descriptor = unsafe {
        libc::memfd_create(
            c"uniserve-media".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    if descriptor < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: memfd_create returned one new descriptor that nothing else owns.
    let file = unsafe { File::from_raw_fd(descriptor) };
    reserve(&file, length)?;
    Ok(file)
}

/// Seals `file`, an [`anonymous`] file whose contents are final, against writes, size changes and
/// further seals. A writable shared mapping of it must not exist.
///
/// # Errors
///
/// When the kernel refuses a seal: the file admits none, or a writable mapping of it exists.
pub fn seal(file: &File) -> io::Result<()> {
    // SAFETY: fcntl takes the file's open descriptor and integers only.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, FINAL_SEALS) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Whether the file `descriptor` names is sealed against writes and size changes, as [`seal`]
/// leaves it.
///
/// # Errors
///
/// When the kernel cannot read the file's seals.
pub fn sealed(descriptor: &impl AsRawFd) -> io::Result<bool> {
    // SAFETY: fcntl takes an open descriptor and integers only.
    let seals = unsafe { libc::fcntl(descriptor.as_raw_fd(), libc::F_GET_SEALS) };
    if seals < 0 {
        return Err(io::Error::last_os_error());
    }
    let required = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK;
    Ok(seals & required == required)
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

//! Media bytes between the processes of a node: published as named POSIX
//! shared-memory objects ([`MediaSource`]), and passed by descriptor and
//! mapped read-only ([`MappedMedia`]).
//!
//! A video request's condition media travel from the daemon to the workers
//! by name: the daemon publishes each and keeps it until the request ends,
//! and a worker's media reader opens it by its [`MediaLocator`]. A worker's
//! generated media travel to the engine the same way, and an artifact from
//! the engine to the daemon by descriptor beside its event.

use std::io::{self, Write as _};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};

use serde::{Deserialize, Serialize};

use crate::request::ArtifactMedia;
use crate::shared_memory::SharedMemory;

/// Where published media lies: a POSIX shared-memory object, by its name
/// without the leading `/`, and the byte count published in it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaLocator {
    /// The object's name without its leading `/`.
    pub name: String,
    /// The published bytes.
    pub bytes: u64,
}

/// Media bytes this process published, under a fresh name, as a POSIX
/// shared-memory object for other processes on this host to read.
///
/// The daemon publishes each fetched condition this way; the worker's media
/// reader opens the object by name, reads it and never unlinks it. The
/// publisher keeps the name: dropping the value unlinks it, and the kernel
/// frees the storage once no reader still maps it. The daemon holds a
/// request's exports until the request's events end.
#[derive(Debug)]
pub struct MediaSource {
    storage: SharedMemory,
}

impl MediaSource {
    /// Copies `bytes` into a new shared-memory object.
    ///
    /// The object is created exclusively, readable and writable by its owner
    /// only, and physically reserved to exactly `bytes.len()` bytes.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when `bytes` is empty or storage allocation or
    /// writing fails; no object is left behind.
    pub fn publish(bytes: &[u8]) -> io::Result<Self> {
        if bytes.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "published media must not be empty",
            ));
        }
        let (storage, mut file) = SharedMemory::create(bytes.len())?;
        file.write_all(bytes)?;
        Ok(Self { storage })
    }

    /// The object's name without its leading `/` and its byte count.
    pub fn locator(&self) -> MediaLocator {
        MediaLocator {
            name: self.storage.name().to_owned(),
            bytes: self.storage.size() as u64,
        }
    }

    /// Hand the named object to its receiving process. The receiver must
    /// claim it with `SharedMedia::open`, which unlinks it on receipt.
    /// Dropping this publisher afterwards leaves the name available.
    pub fn into_locator(self) -> MediaLocator {
        MediaLocator {
            bytes: self.storage.size() as u64,
            name: self.storage.into_name(),
        }
    }
}

impl PartialEq for MediaSource {
    /// Two exports are equal when they name the same object.
    fn eq(&self, other: &Self) -> bool {
        self.storage.name() == other.storage.name()
    }
}

impl Eq for MediaSource {}

/// The first `bytes` bytes of a shared-memory object another process
/// published and passed by descriptor, mapped read-only for as long as this
/// value lives.
#[derive(Debug)]
pub struct MappedMedia {
    address: *mut libc::c_void,
    bytes: usize,
    descriptor: OwnedFd,
}

// SAFETY: the pointer refers to a `PROT_READ` mapping of an object its
// publisher no longer writes (the request interface's contract for an
// artifact) and that stays mapped until this value drops, so moving it across
// threads cannot race.
unsafe impl Send for MappedMedia {}
// SAFETY: as for Send above; shared references only read the immutable
// mapping.
unsafe impl Sync for MappedMedia {}

impl MappedMedia {
    /// Maps the first `bytes` bytes of the object `descriptor` names.
    ///
    /// # Errors
    ///
    /// Fails when `bytes` is zero, the object is shorter, or the mapping
    /// fails.
    pub fn map(descriptor: OwnedFd, bytes: u64) -> std::io::Result<Self> {
        let invalid = |message: &str| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, message.to_owned())
        };
        let bytes = usize::try_from(bytes).map_err(|_| invalid("the artifact is too large"))?;
        if bytes == 0 {
            return Err(invalid("an artifact is never empty"));
        }
        // SAFETY: an all-zero `stat` is a valid value for the call to fill.
        let mut status: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: the descriptor is open and `status` a valid out-pointer.
        if unsafe { libc::fstat(descriptor.as_raw_fd(), &mut status) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        if (status.st_size as u64) < bytes as u64 {
            return Err(invalid("the artifact's object is shorter than its bytes"));
        }
        // SAFETY: the descriptor names a readable object of at least `bytes`
        // bytes; the mapping is read-only and private to this value.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ,
                libc::MAP_SHARED,
                descriptor.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            address,
            bytes,
            descriptor,
        })
    }
}

impl ArtifactMedia for MappedMedia {
    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping covers `bytes` readable bytes until drop, and
        // nobody writes them.
        unsafe { std::slice::from_raw_parts(self.address.cast::<u8>(), self.bytes) }
    }

    fn descriptor(&self) -> Option<BorrowedFd<'_>> {
        Some(self.descriptor.as_fd())
    }
}

impl Drop for MappedMedia {
    fn drop(&mut self) {
        // SAFETY: the address and length are this value's own mapping, which
        // nothing borrows past its lifetime.
        unsafe {
            libc::munmap(self.address, self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::*;

    /// An export is readable by name while it lives and unlinked once
    /// it drops.
    #[test]
    fn a_published_source_is_readable_until_dropped() {
        let source = MediaSource::publish(b"condition bytes").unwrap();
        let locator = source.locator();
        assert_eq!(locator.bytes, 15);
        let path = CString::new(format!("/{}", locator.name)).unwrap();

        // SAFETY: path is a valid NUL-terminated POSIX shm name.
        let descriptor = unsafe { libc::shm_open(path.as_ptr(), libc::O_RDONLY, 0) };
        assert!(descriptor >= 0);
        let mut read = vec![0_u8; 15];
        // SAFETY: descriptor is open and `read` holds 15 writable bytes.
        let count = unsafe { libc::read(descriptor, read.as_mut_ptr().cast(), 15) };
        // SAFETY: descriptor is open.
        unsafe { libc::close(descriptor) };
        assert_eq!(count, 15);
        assert_eq!(read, b"condition bytes");

        drop(source);
        // SAFETY: path is a valid NUL-terminated POSIX shm name.
        let reopened = unsafe { libc::shm_open(path.as_ptr(), libc::O_RDONLY, 0) };
        assert!(reopened < 0, "the name outlived its publisher");
    }

    #[test]
    fn empty_media_is_refused() {
        assert!(MediaSource::publish(b"").is_err());
    }
}

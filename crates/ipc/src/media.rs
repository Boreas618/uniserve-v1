//! Media bytes handed between processes through POSIX shared storage.
//!
//! Generated artifacts travel from a worker to the engine: a model worker
//! publishes finished media bytes through [`MediaSource`] and reports its
//! name as an `ArtifactHandle::PosixShm` in the call result. The engine claims
//! it with [`SharedMedia::open`] when it receives the batch result
//! (`WorkerResult::receive`), and the resulting `Arc<SharedMedia>` reaches the
//! server through the engine's output events. Opening unlinks the name, so a
//! later open of the same name fails and the kernel frees the storage when the
//! last reference to it goes away.
//!
//! Condition media travel the other way: the server publishes each fetched
//! condition as a [`MediaSource`], which keeps the name until it drops, and
//! the worker's media reader opens it by name.

use std::ffi::CString;
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::sync::Arc;

use crate::shared_memory::SharedMemory;

/// Why a shared-storage media export could not be claimed.
///
/// Operating-system failures carry their `io::Error` as the source, shared so
/// that the error can be cloned with the call result that holds it.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SharedMediaError {
    /// The published length does not fit this host's address space.
    #[error("generated media of {0} bytes is too large for this host")]
    TooLarge(u64),
    /// The locator is empty, has a `/` or a NUL in its name, or publishes
    /// no bytes.
    #[error("generated media has an invalid shared-storage locator")]
    Locator,
    /// The named object could not be opened.
    #[error("failed to open generated media shared storage")]
    Open(#[source] Arc<io::Error>),
    /// The name could not be unlinked.
    #[error("failed to claim generated media shared storage")]
    Claim(#[source] Arc<io::Error>),
    /// The object's size could not be read.
    #[error("failed to inspect generated media shared storage")]
    Inspect(#[source] Arc<io::Error>),
    /// The object holds fewer bytes than its locator publishes.
    #[error("generated media shared storage holds {extent} bytes, fewer than its {bytes}")]
    Short {
        /// Bytes the object holds.
        extent: u64,
        /// Bytes the locator publishes.
        bytes: u64,
    },
    /// The published extent could not be mapped.
    #[error("failed to map generated media shared storage")]
    Map(#[source] Arc<io::Error>),
}

/// Read-only mapping of one claimed shared-memory media export.
///
/// The mapping covers exactly the published byte count and is unmapped on
/// drop. Share it with `Arc` rather than reopening: the name no longer exists
/// once it has been claimed.
#[derive(Debug)]
pub struct SharedMedia {
    address: *mut libc::c_void,
    bytes: usize,
}

// SAFETY: the pointer refers to a `PROT_READ` mapping that no process writes
// after export (the `open` contract) and that stays mapped until this value
// drops, so moving it across threads cannot race.
unsafe impl Send for SharedMedia {}
// SAFETY: as for Send above; shared references only read the immutable
// mapping, so access from several threads at once cannot race.
unsafe impl Sync for SharedMedia {}

impl SharedMedia {
    /// Claims an immutable POSIX shared-storage object and maps its published extent.
    ///
    /// `name` is the object name without its leading `/`, and `num_bytes` is
    /// the published length; the object may be longer, and only the first
    /// `num_bytes` bytes are mapped.
    ///
    /// Opening transfers ownership: the name is unlinked immediately, and bytes
    /// remain readable until this mapping is dropped. Once the name has been
    /// unlinked, a later failure (inspection, a short object, or the mapping)
    /// still leaves the storage released. Failures before that point (a zero
    /// or oversized length, an empty name or one containing `/` or NUL, a
    /// failed open, or a failed unlink) leave any named object in place.
    ///
    /// # Errors
    ///
    /// Returns the [`SharedMediaError`] of the first failed step.
    ///
    /// # Safety
    ///
    /// The publisher must relinquish all writes and resizing before this call,
    /// and no process may modify the object while the mapping remains live.
    pub unsafe fn open(name: &str, num_bytes: u64) -> Result<Self, SharedMediaError> {
        let bytes =
            usize::try_from(num_bytes).map_err(|_| SharedMediaError::TooLarge(num_bytes))?;

        // Reject a malformed locator before touching any object: the name must
        // be one non-empty component (the leading `/` is added below), and a
        // export is never empty.
        if bytes == 0 || name.is_empty() || name.contains('/') {
            return Err(SharedMediaError::Locator);
        }
        let canonical_name =
            CString::new(format!("/{name}")).map_err(|_| SharedMediaError::Locator)?;
        let file = SharedMemory::open(name, false)
            .map_err(|error| SharedMediaError::Open(Arc::new(error)))?;

        // Claim the name immediately; the descriptor retains the bytes even
        // if subsequent inspection or mapping fails.
        // SAFETY: `canonical_name` is a NUL-terminated string that outlives the call.
        if unsafe { libc::shm_unlink(canonical_name.as_ptr()) } != 0 {
            return Err(SharedMediaError::Claim(
                Arc::new(io::Error::last_os_error()),
            ));
        }

        let extent = file
            .metadata()
            .map_err(|error| SharedMediaError::Inspect(Arc::new(error)))?
            .len();
        if extent < num_bytes {
            return Err(SharedMediaError::Short {
                extent,
                bytes: num_bytes,
            });
        }

        // SAFETY: descriptor names a readable shared-storage object of at least `bytes` bytes.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(SharedMediaError::Map(Arc::new(io::Error::last_os_error())));
        }
        Ok(Self { address, bytes })
    }

    /// Number of published bytes owned by this mapping.
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// Whether the published extent is empty; successful mappings are nonempty.
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Borrows the immutable published extent for validation or response output.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the mapping remains live for this borrow and is immutable after export.
        unsafe { std::slice::from_raw_parts(self.address.cast(), self.bytes) }
    }
}

impl AsRef<[u8]> for SharedMedia {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl Drop for SharedMedia {
    /// Unmaps the published extent. The name was unlinked at open, so the
    /// kernel frees the storage once no process maps it or holds it open.
    fn drop(&mut self) {
        // SAFETY: address is the live mapping created in `open` with exactly this extent.
        unsafe { libc::munmap(self.address, self.bytes) };
    }
}

/// Media bytes this process published, under a fresh name, as a POSIX
/// shared-memory object for other processes on this host to read.
///
/// The server publishes each fetched condition this way; the worker's media
/// reader opens the object by name, reads it and never unlinks it. The
/// publisher keeps the name: dropping the value unlinks it, and the kernel
/// frees the storage once no reader still maps it. The engine holds a
/// request's exports until the request retires, after its last call.
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
    pub fn locator(&self) -> models::video::MediaLocator {
        models::video::MediaLocator {
            name: self.storage.name().to_owned(),
            bytes: self.storage.size() as u64,
        }
    }

    /// Hand the named object to its receiving process. The receiver must
    /// claim it with `SharedMedia::open`, which unlinks it on receipt.
    /// Dropping this publisher afterwards leaves the name available.
    pub fn into_locator(self) -> models::video::MediaLocator {
        models::video::MediaLocator {
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

#[cfg(test)]
mod tests {
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

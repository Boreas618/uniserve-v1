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
use std::io;
use std::os::fd::AsRawFd;
use std::sync::Arc;

use proto::SharedMemory;

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
    /// The object's descriptor, which another process maps the same bytes
    /// through (`ArtifactMedia::descriptor`).
    file: std::fs::File,
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
        Ok(Self {
            address,
            bytes,
            file,
        })
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

impl proto::request::ArtifactMedia for SharedMedia {
    fn bytes(&self) -> &[u8] {
        self.as_bytes()
    }

    fn descriptor(&self) -> Option<std::os::fd::BorrowedFd<'_>> {
        use std::os::fd::AsFd as _;

        Some(self.file.as_fd())
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

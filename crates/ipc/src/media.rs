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

/// The media an image decoding publishes: the image's base64 PNG, and, when
/// the decoding writes its request's feedback image product
/// (`Call::image_output`), that product's values after it (see
/// [`append_feedback`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageMedia<'a> {
    /// The base64 PNG the client receives.
    pub png_b64: &'a [u8],
    /// The feedback product's values, when the decoding wrote it.
    pub feedback: Option<proto::request::ImagePixels>,
}

/// The byte naming a feedback product's value range in an image decoding's
/// media.
const fn range_byte(range: proto::request::PixelRange) -> u8 {
    match range {
        proto::request::PixelRange::Unit => 0,
        proto::request::PixelRange::SignedUnit => 1,
    }
}

/// Appends a feedback image product's `values` (BF16 `[3, height, width]`,
/// row-major, little-endian) and its value `range` to the base64 PNG `media`
/// of the image decoding that wrote it: the values, then one byte naming
/// the range (0 for `[0, 1]`, 1 for `[-1, 1]`).
pub fn append_feedback(media: &mut Vec<u8>, values: &[u8], range: proto::request::PixelRange) {
    media.extend_from_slice(values);
    media.push(range_byte(range));
}

/// Reads the media of an image decoding of a `height` x `width` image,
/// which carries the feedback product's values after its PNG when
/// `feedback` (the call wrote `Call::image_output`).
///
/// Returns `None` when the media is shorter than the values it must carry
/// or names an unknown range.
pub fn parse_image_media(
    bytes: &[u8],
    height: u32,
    width: u32,
    feedback: bool,
) -> Option<ImageMedia<'_>> {
    if !feedback {
        return Some(ImageMedia {
            png_b64: bytes,
            feedback: None,
        });
    }
    let values = usize::try_from(proto::request::ImagePixels::value_bytes(height, width)).ok()?;
    let (range, rest) = bytes.split_last()?;
    let png_end = rest.len().checked_sub(values)?;
    let range = match *range {
        0 => proto::request::PixelRange::Unit,
        1 => proto::request::PixelRange::SignedUnit,
        _ => return None,
    };
    Some(ImageMedia {
        png_b64: &rest[..png_end],
        feedback: Some(proto::request::ImagePixels {
            height,
            width,
            range,
            values: rest[png_end..].to_vec(),
        }),
    })
}

#[cfg(test)]
mod image_media_tests {
    use super::*;
    use proto::request::PixelRange;

    /// An image decoding's media round-trips its PNG and its feedback
    /// product's values and range; media without feedback is the PNG alone,
    /// and media too short for the values it must carry is refused.
    #[test]
    fn image_media_carries_the_feedback_product_after_the_png() {
        let png = b"iVBORw0KGgo=".to_vec();
        let values: Vec<u8> = (0..2 * 3 * 4 * 2).map(|value| value as u8).collect();
        let mut media = png.clone();
        append_feedback(&mut media, &values, PixelRange::SignedUnit);
        let parsed = parse_image_media(&media, 2, 4, true).unwrap();
        assert_eq!(parsed.png_b64, png.as_slice());
        let feedback = parsed.feedback.unwrap();
        assert_eq!(feedback.values, values);
        assert_eq!(feedback.range, PixelRange::SignedUnit);
        assert!(feedback.is_complete());

        let plain = parse_image_media(&png, 2, 4, false).unwrap();
        assert_eq!(plain.png_b64, png.as_slice());
        assert!(plain.feedback.is_none());
        assert!(parse_image_media(&png, 2, 4, true).is_none());
    }
}

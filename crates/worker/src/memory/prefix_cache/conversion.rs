//! The import conversion program: one source span decoded and encoded into a
//! destination page's units.
//!
//! An import reads another rank's cache export. When the source stores this
//! rank's representation, the transport writes the live pages directly. When
//! it does not (a different element type, FP8 with other scale groups, or a
//! page shape this rank does not store), the span is read into the
//! workspace's scratch and the *conversion program* decodes it and encodes it
//! into the destination units, growing the destination's FP8 scales and
//! rescaling resident values as the destination encoding requires.
//!
//! The program is a kernel program of the numerical library: it is bound once
//! per import workspace and launched per span on that workspace's stream, so
//! nothing on the import thread allocates during the copy. This module holds
//! the request the worker hands the program and the trait the program
//! implements; the program itself belongs to the library, which owns element
//! types, scale groups and rounding.

use std::sync::Arc;

use platform::Stream;

use crate::Result;
use crate::memory::allocation::Storage;

/// One source span to convert, and the destination units to write it into.
pub struct ConversionRequest {
    /// The destination group's index in the pool's plan.
    pub(crate) group: usize,
}

/// A bound import conversion program.
///
/// One program serves one import workspace: it is built once with the
/// workspace's scratch, its stream and the pool's fields, and launched per
/// span afterwards. A launch enqueues its kernels on the stream it is given;
/// the caller has already ordered the span's reads before it.
pub trait PageConversion: Send + Sync {
    /// Decode `request`'s span and encode it into the destination units.
    fn convert(&self, stream: Option<&Stream>, request: &ConversionRequest) -> Result<()>;
}

/// The program of a rank that imports no cache: every conversion is refused
/// by name rather than performed.
///
/// A rank whose deployment never takes another rank's export still builds its
/// memory the same way, so it needs a program; this one states that the path is
/// unreachable, and a request that does reach it says why instead of silently
/// writing nothing.
pub struct Unimported;

impl PageConversion for Unimported {
    fn convert(&self, _stream: Option<&Stream>, request: &ConversionRequest) -> Result<()> {
        Err(crate::Error::Unsupported(format!(
            "this rank imports no cache, so group {} has nothing to convert",
            request.group
        )))
    }
}

/// The scratch a workspace converts through, allocated once.
pub(crate) struct ConversionScratch {
    storage: Arc<Storage>,
}

impl ConversionScratch {
    /// Scratch bytes of one workspace: two raw source pages of up to 8-byte
    /// elements, one FP32 conversion page and the FP32 source scales:
    /// `20 E + 8 S` for `E` page elements and `S` page scale entries.
    pub(crate) fn bytes(elements: u64, scales: u64) -> u64 {
        20 * elements + 8 * scales
    }

    /// Allocate the scratch of `bytes` on `device` (`None` for host storage).
    pub(crate) fn allocate(device: Option<platform::DeviceId>, bytes: u64) -> Result<Self> {
        let bytes = bytes.max(1);
        let size = usize::try_from(bytes)
            .map_err(|_| crate::Error::Invalid("the conversion scratch is too large".into()))?;
        let storage = match device {
            Some(device) => Storage::device(device, size)?,
            None => Storage::host(size),
        };
        Ok(Self { storage, })
    }

    /// The scratch's storage.
    pub(crate) fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

}

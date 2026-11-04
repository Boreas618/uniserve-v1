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
//! the request the executor hands the program and the trait the program
//! implements; the program itself belongs to the library, which owns element
//! types, scale groups and rounding.

use std::sync::Arc;

use platform::Stream;

use crate::Result;
use crate::memory::allocation::ElementType;
use crate::memory::allocation::{Storage, StorageView};

/// One source span to convert, and the destination units to write it into.
pub(crate) struct ConversionRequest<'a> {
    /// The workspace's scratch, `[20 E + 8 S)` bytes: the raw keys at
    /// `[0, 8 E)`, the raw values at `[8 E, 16 E)`, the FP32 conversion
    /// values at `[16 E, 20 E)` and the FP32 source scales at `[20 E, 20 E + 8 S)`,
    /// where `E` is `elements` and `S` is `scales`.
    pub(crate) scratch: StorageView,
    /// `E` above: elements of the largest group page.
    pub(crate) elements: u64,
    /// `S` above: scale entries of the largest group page.
    pub(crate) scales: u64,
    /// The destination group's index in the pool's plan.
    pub(crate) group: usize,
    /// The source's stored element type.
    pub(crate) source_dtype: ElementType,
    /// The source's logical element type, through which decoded values round
    /// before the destination encoding.
    pub(crate) compute_dtype: ElementType,
    /// Tokens of the span: `[layers, heads, head_dim]` each.
    pub(crate) count: u64,
    /// First destination token within the page.
    pub(crate) offset: u64,
    /// Token offset of the span's first token within its first source page.
    pub(crate) source_offset: u64,
    /// Tokens per page of the source, which select its scale rows.
    pub(crate) source_block_size: u32,
    /// Heads of one source scale group; 0 for a dense (unscaled) source.
    pub(crate) scale_heads: u32,
    /// Source pages the span covers.
    pub(crate) pages: u64,
    /// The destination page's units, one per row of `units_per_page`.
    pub(crate) units: &'a [u32],
    /// The pool's unit fields, keyed as the library reads them
    /// (`prefix_cache.key.values`, `prefix_cache.key.scale`, ...).
    pub(crate) fields: &'a [(String, StorageView)],
}

/// A bound import conversion program.
///
/// One program serves one import workspace: it is built once with the
/// workspace's scratch, its stream and the pool's fields, and launched per
/// span afterwards. A launch enqueues its kernels on the stream it is given;
/// the caller has already ordered the span's reads before it.
pub(crate) trait PageConversion: Send + Sync {
    /// Decode `request`'s span and encode it into the destination units.
    fn convert(&self, stream: Option<&Stream>, request: &ConversionRequest<'_>) -> Result<()>;
}

/// The scratch a workspace converts through, allocated once.
pub(crate) struct ConversionScratch {
    storage: Arc<Storage>,
    bytes: u64,
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
        Ok(Self { storage, bytes })
    }

    /// The scratch's storage.
    pub(crate) fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

    /// The whole scratch as one `u8` view.
    pub(crate) fn view(&self) -> Result<StorageView> {
        StorageView::contiguous(
            Arc::clone(&self.storage) as Arc<dyn crate::memory::allocation::Backing>,
            0,
            ElementType::U8,
            &[self.bytes as i64],
        )
    }
}

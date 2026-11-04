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

use platform::{CopyBatch, SourceOrder, Stream};

use crate::Result;
use crate::memory::allocation::ElementType;
use crate::memory::allocation::{Backing, Storage, StorageView};

/// One source span to convert, and the destination units to write it into.
pub struct ConversionRequest<'a> {
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
pub trait PageConversion: Send + Sync {
    /// Decode `request`'s span and encode it into the destination units.
    fn convert(&self, stream: Option<&Stream>, request: &ConversionRequest<'_>) -> Result<()>;
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
    fn convert(&self, _stream: Option<&Stream>, request: &ConversionRequest<'_>) -> Result<()> {
        Err(crate::Error::Unsupported(format!(
            "this rank imports no cache, so group {} has nothing to convert",
            request.group
        )))
    }
}

/// The generation the import path resolves its kernels on.
///
/// The program-layer hot update swaps the generation under the executor, so the
/// import path asks for the one serving now rather than holding a kernel: a
/// conversion launched after a switchover runs the new program.
///
/// The contract is "serving until compared", not "newest": a generation answers
/// only once its paired comparison has passed - the shadow's agreement on the
/// last operator, the import conversion programs included - and until then this
/// answers the generation that already served. Without that ordering the first
/// thing a wrong conversion corrupts is the data no request checks: the pages
/// it imports.
pub(crate) trait GenerationSource: Send + Sync {
    /// The generation that may serve: the latest one whose paired comparison
    /// has passed, and before any has, the one already serving.
    fn current(&self) -> Arc<ops::Generation>;
}

/// One group's two descriptors: how its fields decode, and how they encode.
pub(crate) struct GroupDescriptors {
    pub(crate) conversion: ops::prefix_cache::ConversionDescriptor,
    pub(crate) copy: ops::prefix_cache::CopyRegionDescriptor,
}

/// The import conversion program, bound to a pool's groups and to the
/// generation that serves.
///
/// It holds one staging buffer of the destination page's units, because a
/// request carries them as host page indices while the copy kernel reads a
/// device view; the copy of those few words runs per span, on the span's
/// stream, before the launch that reads them.
pub(crate) struct ImportConversion {
    generations: Arc<dyn GenerationSource>,
    groups: Vec<GroupDescriptors>,
    units: Arc<Storage>,
    units_per_page: u32,
}

impl ImportConversion {
    /// Bind the program to `groups`' descriptors and to the serving generation,
    /// staging up to `units_per_page` destination units on `device`.
    pub(crate) fn new(
        generations: Arc<dyn GenerationSource>,
        groups: Vec<GroupDescriptors>,
        units_per_page: u32,
        device: Option<platform::DeviceId>,
    ) -> Result<Self> {
        if groups.is_empty() || units_per_page == 0 {
            return Err(crate::Error::Invalid(
                "an import conversion program needs a group and a unit".into(),
            ));
        }
        let bytes = usize::try_from(u64::from(units_per_page) * 4)
            .map_err(|_| crate::Error::Invariant("a page's units overflow memory".into()))?;
        let units = match device {
            Some(device) => Storage::device(device, bytes)?,
            None => Storage::host(bytes),
        };
        Ok(Self {
            generations,
            groups,
            units,
            units_per_page,
        })
    }

    /// The device view of a field, with the element type the kernel operand
    /// describes; a storage element no operand describes is refused by name.
    fn tensor(view: &StorageView) -> Result<ops::TensorView> {
        let dtype = view.dtype().to_ops().ok_or_else(|| {
            crate::Error::Invalid(format!(
                "an import field holds {:?}, which no kernel operand type describes",
                view.dtype()
            ))
        })?;
        let shape: Vec<u64> = view.shape().iter().map(|&extent| extent as u64).collect();
        Ok(ops::TensorView::new(
            ops::DevicePtr::new(view.address()),
            dtype,
            ops::Shape::new(&shape),
        ))
    }
}

impl PageConversion for ImportConversion {
    fn convert(&self, stream: Option<&Stream>, request: &ConversionRequest<'_>) -> Result<()> {
        let stream = stream.ok_or_else(|| {
            crate::Error::Invalid("a conversion program launches on a stream".into())
        })?;
        let group = self.groups.get(request.group).ok_or_else(|| {
            crate::Error::Invalid(format!(
                "the conversion program holds no descriptor for group {}",
                request.group
            ))
        })?;
        if request.units.len() > self.units_per_page as usize {
            return Err(crate::Error::Invalid(format!(
                "a page of {} units does not fit the program's {}",
                request.units.len(),
                self.units_per_page
            )));
        }

        // The destination units are host page indices in the request and a
        // device view in the copy's arguments, so they are staged here.
        let staged = self.units.address();
        let units = StorageView::whole(
            Arc::clone(&self.units) as Arc<dyn Backing>,
            ElementType::I32,
        )?;
        // SAFETY: the staging buffer spans `units_per_page` words, the source
        // slice outlives the call, and the launch that reads the staged units
        // is enqueued after this copy on the same stream.
        unsafe {
            let mut batch = CopyBatch::new(SourceOrder::DuringCall);
            batch.push(
                staged,
                request.units.as_ptr() as u64,
                request.units.len() * 4,
            );
            batch.submit(stream)?;
        }

        let generation = self.generations.current();
        let decode = generation.op::<ops::prefix_cache::PageConversion>(&group.conversion)?;
        <ops::prefix_cache::PageConversion as ops::Op>::launch(
            decode,
            &group.conversion,
            ops::prefix_cache::ConversionArgs {
                scratch: Self::tensor(&request.scratch)?,
                elements: request.elements,
                scales: request.scales,
                count: request.count,
                source_offset: request.source_offset,
                source_block_size: request.source_block_size,
                scale_heads: request.scale_heads,
                pages: request.pages,
            },
            stream,
        )?;

        let copy = generation.op::<ops::prefix_cache::CacheCopyRegion>(&group.copy)?;
        for field in ["prefix_cache.key", "prefix_cache.value"] {
            let mut values = None;
            let mut scale = None;
            let mut initialized = None;
            for (name, view) in request.fields {
                let Some(rest) = name.strip_prefix(field) else {
                    continue;
                };
                match rest {
                    ".values" => values = Some(view),
                    ".scale" => scale = Some(view),
                    ".initialized" => initialized = Some(view),
                    _ => {}
                }
            }
            let (Some(values), Some(initialized)) = (values, initialized) else {
                continue;
            };
            <ops::prefix_cache::CacheCopyRegion as ops::Op>::launch(
                copy,
                &group.copy,
                ops::prefix_cache::CopyRegionArgs {
                    values: Self::tensor(values)?,
                    scale: scale.map(Self::tensor).transpose()?,
                    initialized: Self::tensor(initialized)?,
                    source: Self::tensor(&request.scratch)?,
                    units: Self::tensor(&units)?,
                    offset: request.offset,
                    count: request.count,
                },
                stream,
            )?;
        }
        Ok(())
    }
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

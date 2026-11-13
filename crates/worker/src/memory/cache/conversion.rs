//! Converted page loads: pages another host holds in another representation, loaded into this
//! rank's units in its own.
//!
//! A cache that admits converted loads ([`CacheLayout::with_conversion`]: an FP8 cache, from its
//! groups' dense compute type) keeps, per attention group, a source buffer of host storage: an
//! exclusive tensor its host's Storage Server retains, into which the host's Transfer Server
//! reads a peer's pages of the dense source layout (`ipc::PoolConversion`). A page read there
//! is never committed to a host pool. The buffer pool names a page of it as a run of the
//! group's source region, whose unit `u` is slot `u` of the buffer, and a copy from there into
//! units is a converted load, which runs on the rank's copy stream, page by page:
//!
//! 1. both fields of the source page are copied into the device scratch, keys at `[0, 8E)` and
//!    values at `[8E, 16E)` (the scratch layout of `ops::prefix_cache`);
//! 2. per field, `page_conversion` decodes the field into the FP32 block at `[16E, 20E)`, and
//!    `cache_copy_region` encodes the block into the page's units with the cache's own
//!    encoding: one E4M3 scale per plane from the span's largest magnitude, the initialization
//!    flag raised last.
//!
//! The scratch holds one page, so the pages convert one after another; stream order keeps each
//! page's copy behind the previous page's kernels. The page's units are a `[units_per_page]`
//! row of a small device table, written once per batch before the first page's kernels; the
//! row is I32, the element type `cache_copy_region` reads its units in.
//!
//! The two kernels launch through `backend::launch` under a context of the conversion's own,
//! over the copy stream and the generation the rank serves from; it allocates nothing, the
//! scratch and the table being the conversion's storage.

use std::collections::BTreeSet;
use std::sync::Arc;

use ::storage::Key;
use ipc::PoolConversion;
use ops::prefix_cache::{
    CacheCopyRegion, ConversionArgs, ConversionDescriptor, CopyRegionArgs, CopyRegionDescriptor,
    Field, PageConversion, scratch_bytes,
};
use platform::{CopyBatch, DeviceId, DevicePtr, DeviceRange, SourceOrder, Stream};
use uniserve_tensor::{DType, Tensor};

use super::host::{self, HostPage};
use super::layout::{CacheLayout, GroupLayout};
use super::planes::Planes;
use crate::memory::allocation::Storage;
use crate::memory::client::StorageClient;
use crate::memory::lend;
use crate::{Error, Result};

/// The most host bytes one group's source buffer holds.
const SOURCE_BYTES: u64 = 1 << 30;

/// Numbers the source buffers this process maps, so no two share a key.
static SOURCE_BUFFERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The two fields a page holds, in the page encoding's order.
const FIELDS: [Field; 2] = [Field::Key, Field::Value];

/// The bytes the device holds for the converted loads of `layout`, `width` pages a batch at
/// most: the conversion scratch of the largest group's page and the unit table. Zero for a
/// layout without converted loads.
pub fn device_bytes(layout: &CacheLayout, width: u32) -> u64 {
    if layout.converts_from.is_none() {
        return 0;
    }
    page_scratch_bytes(layout) + u64::from(width) * u64::from(units_per_page(layout)) * 4
}

/// The scratch bytes the largest group's page converts in.
fn page_scratch_bytes(layout: &CacheLayout) -> u64 {
    let elements = layout
        .groups
        .iter()
        .map(GroupLayout::page_elements)
        .max()
        .unwrap_or(0);
    scratch_bytes(elements, 0)
}

/// The units of the largest group's page: the columns of the unit table's rows.
fn units_per_page(layout: &CacheLayout) -> u32 {
    layout
        .groups
        .iter()
        .map(|group| group.units_per_page)
        .max()
        .unwrap_or(1)
}

/// The element type the converted pages are stored in at their source.
fn source_dtype(layout: &CacheLayout) -> Result<DType> {
    layout
        .source()
        .and_then(|source| source.groups.first().map(|group| group.dtype))
        .ok_or_else(|| Error::Invalid("a cache that converts from no source".into()))
}

/// Launches the conversion of one source page of `group`, already copied into `scratch`, into
/// the page whose units `units` names, by the context installed on this thread: per field, the
/// decode into `block` and the encode into the field's planes of `planes`.
fn convert_page(
    layout: &CacheLayout,
    group: &GroupLayout,
    source: DType,
    planes: &Planes,
    scratch: &Tensor,
    block: &Tensor,
    units: &Tensor,
) -> Result<()> {
    for (index, field) in FIELDS.into_iter().enumerate() {
        backend::launch::<PageConversion>(
            &ConversionDescriptor {
                field,
                source,
                compute: group.compute_dtype,
                head_dim: group.head_dim,
                heads: group.num_kv_heads,
                layers: group.layers.len() as u32,
                scaled: false,
            },
            ConversionArgs {
                scratch: backend::operand(scratch)?,
                elements: group.page_elements(),
                scales: 0,
                block_size: group.block_size,
                scale_heads: 0,
            },
        )?;

        let (values, scale, initialized) = planes.fields(index)?;
        backend::launch::<CacheCopyRegion>(
            &CopyRegionDescriptor {
                field,
                columns: layout.columns,
                units_per_page: group.units_per_page,
                plane_bytes: layout.plane_bytes,
                heads: group.num_kv_heads,
                head_dim: group.head_dim,
                element: group.dtype,
                scaled: layout.quantized,
            },
            CopyRegionArgs {
                values: backend::operand(&values)?,
                scale: scale.as_ref().map(backend::operand).transpose()?,
                initialized: backend::operand(&initialized)?,
                source: backend::operand(block)?,
                units: backend::operand(units)?,
                offset: 0,
                count: u64::from(group.block_size),
            },
        )?;
    }
    Ok(())
}

/// The shape of the FP32 block a page of `group` decodes into.
fn block_shape(group: &GroupLayout) -> [usize; 4] {
    [
        group.layers.len(),
        group.block_size as usize,
        group.num_kv_heads as usize,
        group.head_dim as usize,
    ]
}

/// The kernels the converted loads of a cache of `layout` over `units` units launch, as
/// `(operator, key)`: one page of every group converted under a meta context, which launches
/// nothing. Empty for a layout without converted loads.
///
/// # Errors
///
/// When a group's descriptors refuse the page's tensors.
pub(crate) fn keys(
    layout: &Arc<CacheLayout>,
    units: u32,
) -> Result<BTreeSet<(&'static str, String)>> {
    if layout.converts_from.is_none() {
        return Ok(BTreeSet::new());
    }
    let source = source_dtype(layout)?;
    let planes = Planes::meta(Arc::clone(layout), units)?;

    // SAFETY: meta tensors have no data and their addresses are never read.
    let meta = |shape: &[usize], dtype| unsafe { lend(0, shape, dtype, None) };

    let context = backend::Context::meta(None);
    context.record_launches();
    {
        let _installed = context.enter();

        for group in &layout.groups {
            let scratch = meta(
                &[scratch_bytes(group.page_elements(), 0) as usize],
                DType::U8,
            )?;
            let block = meta(&block_shape(group), DType::F32)?;
            let row = meta(&[group.units_per_page as usize], DType::I32)?;

            convert_page(layout, group, source, &planes, &scratch, &block, &row)?;
        }
    }

    Ok(context
        .take_launches()
        .into_iter()
        .map(|(operator, key, _)| (operator, key))
        .collect())
}

/// The converted loads of one rank's cache: the device scratch and unit table, each group's
/// source buffer, and the context its kernels launch under.
pub(crate) struct Conversion {
    layout: Arc<CacheLayout>,
    source: DType,
    /// Each group's source buffer, mapped and registered with the device, with the source
    /// pages it holds and the bytes of one.
    buffers: Vec<(Arc<Storage>, u32, u64)>,
    /// The conversion scratch of the largest group's page, on the device.
    scratch: Arc<Storage>,
    /// The units of the pages of one batch, `[width, units_per_page]` I32 on the device.
    units: Arc<Storage>,
    /// Pages a batch converts at most.
    width: u32,
    units_per_page: u32,
    /// What the two kernels launch under: the copy stream and the serving generation.
    context: backend::Context,
}

impl Conversion {
    /// The converted loads of a cache of `layout` on `device`, `width` pages a batch at most,
    /// launching on `stream` with `kernels`, with each group's source buffer of up to `width`
    /// source pages (at most [`SOURCE_BYTES`]) allocated at the rank's host's Storage Server.
    /// `None` for a layout without converted loads. Returns with it what the rank reports of
    /// each group's conversion, in group order.
    ///
    /// # Errors
    ///
    /// When the server refuses a source buffer or a device allocation fails.
    pub(crate) fn open(
        client: &StorageClient,
        layout: &Arc<CacheLayout>,
        device: DeviceId,
        width: u32,
        stream: &Stream,
        kernels: Arc<ops::Generation>,
    ) -> Result<Option<(Self, Vec<PoolConversion>)>> {
        let Some(source_layout) = layout.source() else {
            return Ok(None);
        };
        let source = source_dtype(layout)?;

        let mut buffers = Vec::with_capacity(layout.groups.len());
        let mut reports = Vec::with_capacity(layout.groups.len());
        for index in 0..layout.groups.len() {
            let shape = host::page_shape(&source_layout, index)?;

            // The buffer is named by the rank's process, device, group and mapping: it is this
            // rank's own writable buffer, which nothing else retains or reads.
            let mut digest = ::storage::key::Digest::new(b"uniserve-converted-load-source");
            digest.number(u64::from(std::process::id()));
            digest.number(device.0 as u64);
            digest.number(index as u64);
            digest.number(SOURCE_BUFFERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
            let key = Key(digest.finish());

            // One request's pages of the widest row, at most `SOURCE_BYTES` of them: a page the
            // scheduler finds no free slot for is computed instead.
            let slots =
                (SOURCE_BYTES / shape.page_bytes.max(1)).clamp(1, u64::from(width.max(1))) as u32;
            let bytes = u64::from(slots) * shape.page_bytes;

            let (_, descriptors) = client.call("a converted-load source buffer", |client| {
                client.allocate_host(key, format!("converted-load source {index}"), bytes)
            })?;
            let descriptor = descriptors.into_iter().next().ok_or_else(|| {
                Error::Interface("a source buffer arrived without its storage".into())
            })?;
            let storage = Storage::map_host(descriptor, bytes as usize, device, true)?;
            tracing::info!(
                group = index,
                slots,
                page_bytes = shape.page_bytes,
                "mapped a converted-load source buffer"
            );

            buffers.push((storage, slots, shape.page_bytes));
            reports.push(PoolConversion {
                source: shape,
                staging: key,
                slots,
            });
        }

        let units_per_page = units_per_page(layout);
        let scratch = Storage::linear(device, page_scratch_bytes(layout) as usize)?;
        let units = Storage::linear(
            device,
            (u64::from(width) * u64::from(units_per_page) * 4) as usize,
        )?;

        // The context allocates nothing: every tensor of a conversion is over the scratch, the
        // unit table or the planes.
        let ordinal = usize::try_from(device.0)
            .map_err(|_| Error::Invalid(format!("{device} names no GPU ordinal")))?;
        let region = DeviceRange::new(DevicePtr::from_address(scratch.address()), 0);
        let context = backend::Context::new(ordinal, stream.clone(), kernels, region);

        Ok(Some((
            Self {
                layout: Arc::clone(layout),
                source,
                buffers,
                scratch,
                units,
                width,
                units_per_page,
                context,
            },
            reports,
        )))
    }

    /// The generation the conversions launch from, from the next load on: set at a switch.
    pub(crate) fn set_kernels(&self, kernels: Arc<ops::Generation>) {
        self.context.set_kernels(kernels);
    }

    /// Converts the source pages of `loads`, each a group, a slot of its source buffer and the
    /// units of the block, into their units on `stream`, the copy stream the conversion was
    /// opened on. The units are the destination of the copy alone until it is reported done;
    /// the buffer pool holds the source pages until then.
    ///
    /// # Errors
    ///
    /// When a page names a group or a slot outside the rank's source buffers, a batch converts
    /// more pages than its unit table holds, or a launch fails.
    pub(crate) fn load(&self, planes: &Planes, loads: &[HostPage], stream: &Stream) -> Result<()> {
        if loads.is_empty() {
            return Ok(());
        }
        if loads.len() > self.width as usize {
            return Err(Error::Invalid(format!(
                "a batch converts {} pages and its unit table holds {}",
                loads.len(),
                self.width
            )));
        }

        // Every page is checked before anything is issued.
        let stride = self.units_per_page as usize;
        for (group, slot, units) in loads {
            let group = *group as usize;
            let (Some((_, slots, _)), Some(layout)) =
                (self.buffers.get(group), self.layout.groups.get(group))
            else {
                return Err(Error::Invalid(format!(
                    "a converted load names cache group {group}, which converts nothing"
                )));
            };
            if *slot >= u64::from(*slots) || units.len() != layout.units_per_page as usize {
                return Err(Error::Invalid(format!(
                    "a converted load of slot {slot} with {} units, of a source buffer of \
                     {slots} slots and pages of {} units",
                    units.len(),
                    layout.units_per_page
                )));
            }
            for unit in units.clone() {
                planes.require_unit(unit)?;
            }
        }

        // Every page's units, one row of the table each, written before the first launch.
        let mut table = vec![0_i32; loads.len() * stride];
        for (row, (_, _, units)) in loads.iter().enumerate() {
            for (column, unit) in units.clone().enumerate() {
                table[row * stride + column] = i32::try_from(unit)
                    .map_err(|_| Error::Invalid(format!("unit {unit} past i32")))?;
            }
        }
        let mut write = CopyBatch::new(SourceOrder::DuringCall);
        write.push(
            self.units.address(),
            table.as_ptr() as u64,
            std::mem::size_of_val(table.as_slice()),
        );

        // SAFETY: the table is this conversion's, read only by the kernels issued after this
        // copy on the same stream; the host rows are read before the call returns.
        unsafe { write.submit(stream) }?;

        let _installed = self.context.enter();
        for (row, (group, slot, _)) in loads.iter().enumerate() {
            let (buffer, _, page_bytes) = &self.buffers[*group as usize];
            let group = &self.layout.groups[*group as usize];
            let elements = group.page_elements();

            // The source page is the host page encoding of the source: keys, then values, each
            // `elements` elements of the source type.
            let field_bytes = page_bytes / 2;
            let page = buffer.address() + slot * page_bytes;
            let mut fields = CopyBatch::new(SourceOrder::Stream);
            for index in 0..FIELDS.len() as u64 {
                fields.push(
                    self.scratch.address() + index * 8 * elements,
                    page + index * field_bytes,
                    field_bytes as usize,
                );
            }

            // SAFETY: the scratch is this conversion's, and the previous page's kernels, which
            // read it, precede this copy on the stream; the buffer pool holds the source page
            // until the copy is reported done.
            unsafe { fields.submit(stream) }?;

            // SAFETY: the three tensors are windows of the scratch and the unit table, which
            // this conversion owns and only the work it issues on its one stream addresses.
            let (scratch, block, units) = unsafe {
                (
                    self.scratch.external(
                        0,
                        &[scratch_bytes(elements, 0) as usize][..],
                        DType::U8,
                    )?,
                    self.scratch.external(
                        (16 * elements) as usize,
                        block_shape(group).as_slice(),
                        DType::F32,
                    )?,
                    self.units.external(
                        row * stride * 4,
                        &[group.units_per_page as usize][..],
                        DType::I32,
                    )?,
                )
            };
            convert_page(
                &self.layout,
                group,
                self.source,
                planes,
                &scratch,
                &block,
                &units,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ipc::PrefixCacheDtype;
    use ops::Op;

    use super::*;

    /// 64-token pages of four layers of two 128-wide heads.
    fn layout(dtype: PrefixCacheDtype) -> CacheLayout {
        CacheLayout::dense(64, 4, 2, 128, dtype)
            .validated()
            .expect("a dense layout")
    }

    /// A converting cache states one `page_conversion` and one `cache_copy_region` per field of
    /// its converted loads, so a generation holds their kernels; a cache that loads only its
    /// own pages states none.
    #[test]
    fn a_converting_cache_states_its_page_load_kernels() -> Result<()> {
        let converting = Arc::new(layout(PrefixCacheDtype::Float8E4m3Fn).with_conversion()?);

        let keys = keys(&converting, 8)?;
        let of = |operator: &str| keys.iter().filter(|(name, _)| *name == operator).count();
        assert_eq!(keys.len(), 4, "{keys:?}");
        assert_eq!(of(PageConversion::NAME), 2);
        assert_eq!(of(CacheCopyRegion::NAME), 2);

        let own = Arc::new(layout(PrefixCacheDtype::Float8E4m3Fn));
        assert!(super::keys(&own, 8)?.is_empty());
        Ok(())
    }

    /// The device holds one page's scratch and a unit row per page of a batch, and nothing for
    /// a cache that converts none.
    #[test]
    fn the_device_holds_one_pages_scratch_and_a_batchs_unit_rows() -> Result<()> {
        let converting = layout(PrefixCacheDtype::Float8E4m3Fn).with_conversion()?;
        let elements = 4 * 64 * 2 * 128;

        assert_eq!(
            device_bytes(&converting, 16),
            scratch_bytes(elements, 0) + 16 * 4
        );
        assert_eq!(device_bytes(&layout(PrefixCacheDtype::Float8E4m3Fn), 16), 0);
        Ok(())
    }
}

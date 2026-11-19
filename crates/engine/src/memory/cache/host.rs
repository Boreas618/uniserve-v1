//! The host tier of the rank's KV cache: the host regions its pages are copied through.
//!
//! Each cache group of the KV pool copies its pages through a host region of the buffer pool, the
//! deployment's on the host: the Storage Server keeps one memfd per rank index, the rank's part
//! of every unit, which every epoch that names the region maps. The rank maps its part of each
//! group's region writable and registers it for device copies ([`HostPart`]). The buffer pool
//! names a host copy of a block as a run of the group's host region, whose unit `u` is page `u`
//! of the group on every rank, at unit `u` of the rank's part; the copies a completion queues
//! ([`HostPage`]) run on the rank's copy stream with the batch that takes them. A sliding
//! window's pages are never cached, so its group has no host region.
//!
//! A host page is one page of one group on this rank, plane by plane: each field's planes of
//! every unit and column of the page, in unit then column order, the fields in the pool's order,
//! and directly after the last field that keeps page scales, the scales of those fields
//! `[scaled fields, planes]` as F32. One owner of one page fills exactly one plane, so a page of
//! keys and values (and index keys) is layer-major:
//!
//! 1. keys `[layers, block_size, kv_heads, head_dim]`;
//! 2. values, in the same shape;
//! 3. for an FP8 cache, the scales `[2, layers]` as F32;
//! 4. for a cache that keeps index keys, the index keys `[layers, block_size / ratio, width]`.
//!
//! The state group's page is one unit of the state region: every recurrent layer's checkpoint,
//! the unit's planes in their order (`CacheLayout::state_bytes`), copied in one piece, and no
//! key, value or scale.

use std::collections::BTreeMap;
use std::ops::Range;

use ::storage::Key;
use ::storage::protocol::PageShape;
use platform::{CopyBatch, DeviceId, Event, EventFlags, SourceOrder, Stream};
use tensor::DType;

use super::layout::{CacheLayout, FieldLayout, GroupLayout, PoolLayout};
use super::planes::Planes;
use super::states::StateUnits;
use crate::memory::client::StorageClient;
use crate::memory::host_pages::{self, HostPart};
use crate::worker::{Error, Result};

/// The page encoding above, as the layout and content digests describe it beside the group's
/// shape: pages encoded otherwise are described otherwise, and so are another layout and
/// another content. The spelling names the encoding of a page of keys and values, the one every
/// encoding of the rule above reduces to for them, and stays as it is: it names host regions
/// every build of a deployment shares.
const PAGE_ENCODING: &str = "keys, then values, each field's planes in unit then column order; \
                             then scales [2, planes] float32; then index keys";

/// The fields of a page of `group`: its pool's, which a draft page shares with the full
/// history's. `None` for the state group.
fn pool_of<'a>(layout: &'a CacheLayout, group: &GroupLayout) -> Option<&'a PoolLayout> {
    (!group.state()).then(|| &layout.pools[group.pool])
}

/// The planes one page of `group` spans of `field`: every column of every unit of the page.
fn field_planes(group: &GroupLayout, field: &FieldLayout) -> u64 {
    u64::from(group.units_per_page) * u64::from(field.columns())
}

/// The index of the last field of `pool` that keeps page scales, after whose planes a page holds
/// the scales of every such field; `None` for a pool that keeps none.
fn scales_after(pool: &PoolLayout) -> Option<usize> {
    pool.fields.iter().rposition(|field| field.scaled)
}

/// The bytes of one host page of `group` of a cache of `layout`: every plane of every field of
/// the page, and a scale per plane of a scaled field.
fn page_bytes(layout: &CacheLayout, group: &GroupLayout) -> u64 {
    let Some(pool) = pool_of(layout, group) else {
        return layout.state_bytes();
    };
    let tokens = pool.block_size;

    pool.fields
        .iter()
        .map(|field| {
            let planes = field_planes(group, field) / u64::from(field.columns().max(1));
            let scales = if field.scaled {
                field_planes(group, field) * 4
            } else {
                0
            };
            planes * field.unit_bytes(tokens) + scales
        })
        .sum()
}

/// The spelling of a stored element type inside a page layout digest. The digest names the
/// host region of a group's pages, which every epoch of the deployment on a host shares
/// whatever build of the engine it runs, so the spellings are fixed here and do not follow a
/// library's.
fn stored_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F32 => "float32",
        DType::F8E4M3 => "float8_e4m3fn",
        other => ops::abi::name(other),
    }
}

/// Adds what the logical content of `group`'s pages is to `digest`: the group's window, block
/// size, layers and heads, and each field beside keys and values: the shape of the index keys a
/// key/value cache keeps beside them, and any other field by its name, its record and its
/// owners.
fn content_of(digest: &mut ::storage::key::Digest, layout: &CacheLayout, group: &GroupLayout) {
    digest.number(u64::from(group.kind.window().unwrap_or(0)));
    digest.number(u64::from(group.block_size));

    let layers: Vec<u64> = group
        .layer_ids
        .iter()
        .map(|&layer| u64::from(layer))
        .collect();
    digest.numbers(&layers);
    digest.number(u64::from(group.total_kv_heads));
    digest.number(u64::from(group.head_dim));

    // Keys and values are the group's heads above; a group that keeps nothing else adds nothing,
    // so its pages keep the digest a cache of keys and values has.
    let Some(pool) = pool_of(layout, group) else {
        return;
    };
    for field in &pool.fields {
        if [nn::family::KEY, nn::family::VALUE].contains(&field.name) {
            continue;
        }
        let ratio = field.owners.first().map_or(0, |owner| owner.ratio);
        let uniform = field.owners.iter().all(|owner| owner.ratio == ratio);
        if field.name == nn::family::INDEX_KEY && field.dims[0] == 1 && uniform {
            digest.text("index_keys");
            digest.number(u64::from(field.dims[1]));
            digest.number(u64::from(ratio));
            continue;
        }

        digest.text(field.name);
        digest.numbers(&[u64::from(field.dims[0]), u64::from(field.dims[1])]);
        let owners: Vec<u64> = field
            .owners
            .iter()
            .flat_map(|owner| [u64::from(owner.layer), u64::from(owner.ratio)])
            .collect();
        digest.numbers(&owners);
    }
}

/// Adds the shape of a state page to `digest`: one unit of the state region, its checkpoint
/// planes back to back.
fn state_of(digest: &mut ::storage::key::Digest, layout: &CacheLayout, group: &GroupLayout) {
    if group.state() {
        digest.text("state-unit");
        digest.number(u64::from(layout.checkpoint_planes()));
    }
}

/// The digest of the byte layout of `group`'s host pages: the page encoding, the stored
/// representation and the group's shape and logical layers. It names the group's host region
/// (`PageShape::layout`).
fn layout_digest(layout: &CacheLayout, group: &GroupLayout) -> [u8; 32] {
    let mut digest = ::storage::key::Digest::new(b"uniserve-page-layout");
    digest.text(PAGE_ENCODING);
    digest.number(u64::from(layout.kv().quantized()));
    digest.number(layout.plane_bytes());
    digest.number(u64::from(layout.columns()));
    content_of(&mut digest, layout, group);
    digest.text(stored_name(group.dtype));

    // Every field beside keys and values is stored in its own type (index keys in the layers'
    // compute type, a block-scaled record in bytes), whatever the planes of keys and values
    // store; a group that keeps none adds nothing.
    if let Some(pool) = pool_of(layout, group) {
        for field in &pool.fields {
            if ![nn::family::KEY, nn::family::VALUE].contains(&field.name) {
                digest.text(stored_name(field.dtype));
            }
        }
    }
    state_of(&mut digest, layout, group);
    digest.finish()
}

/// The digest of the logical content of `group`'s pages: the page encoding and the group's
/// shape and logical layers, without the representation [`layout_digest`] adds. Pages of one
/// prompt computed by one model in two representations hold the same logical content.
fn content_digest(layout: &CacheLayout, group: &GroupLayout) -> [u8; 32] {
    let mut digest = ::storage::key::Digest::new(b"uniserve-page-content-layout");
    digest.text(PAGE_ENCODING);
    content_of(&mut digest, layout, group);
    state_of(&mut digest, layout, group);
    digest.finish()
}

/// The shape of the host pages of group `group` of a cache of `layout`: their layout digest,
/// the rank's head slice and their bytes.
///
/// # Errors
///
/// For a group the layout does not hold.
pub(crate) fn page_shape(layout: &CacheLayout, group: usize) -> Result<PageShape> {
    let entry = layout
        .groups
        .get(group)
        .ok_or_else(|| Error::Invalid(format!("the cache holds no group {group}")))?;

    Ok(PageShape {
        layout: Key(layout_digest(layout, entry)),
        kv_head_offset: entry.kv_head_offset,
        kv_head_count: entry.num_kv_heads,
        page_bytes: page_bytes(layout, entry),
    })
}

/// The digests the buffer pool's class of `group` states for a cache of `layout` split over
/// `ranks` tensor ranks (`storage::runtime::Class::{tensors, encoding}`): the logical content of
/// its pages ([`content_digest`]), and its byte layout on every rank ([`layout_digest`]) with the
/// number of ranks a block's units are split over. A host copy of another layout is answered with
/// both encodings and is not converted, so it counts as a miss.
///
/// # Errors
///
/// For a group the layout does not hold.
pub(crate) fn class_digests(
    layout: &CacheLayout,
    group: usize,
    ranks: u32,
) -> Result<([u8; 32], [u8; 32])> {
    let entry = layout
        .groups
        .get(group)
        .ok_or_else(|| Error::Invalid(format!("the cache holds no group {group}")))?;

    let mut encoding = ::storage::key::Digest::new(b"uniserve-page-class-encoding");
    encoding.bytes(&layout_digest(layout, entry));
    encoding.number(u64::from(ranks));
    Ok((content_digest(layout, entry), encoding.finish()))
}

/// The page identity of each of `layout`'s groups for a model whose weights are `fingerprint`
/// (`PrefixCacheInfo::identities`): the digest of the weights' identity and the group's logical
/// page content ([`content_digest`]). The scheduler seeds the group's page keys at every tier
/// from it (`storage::key::page_seed`), so a page computed by other weights or of another shape
/// is another page. The representation is not part of it: the same page in another stored type
/// has the same key, and the host regions of each representation stay apart by their layout
/// digest.
pub(crate) fn page_identities(layout: &CacheLayout, fingerprint: &str) -> Vec<Key> {
    layout
        .groups
        .iter()
        .map(|group| {
            let mut identity = ::storage::key::Digest::new(b"uniserve-page-identity");
            identity.text(fingerprint);
            identity.bytes(&content_digest(layout, group));
            identity.key()
        })
        .collect()
}

/// One page's copy between the host tier and the units of the pool: the cache group, the page
/// of the group's host region, and the units of the block on the device.
pub(crate) type HostPage = (u32, u64, Range<u32>);

/// The rank's parts of the cache groups' host regions, by cache group, and the stream its page
/// copies run on.
pub(crate) struct HostTier {
    /// Each group's part with the planes of one field a page of the group spans; a group whose
    /// pages have no host copy has none.
    parts: BTreeMap<u32, (HostPart, u32)>,
    /// The copy stream: page copies run on it beside the forwards on a batch's stream.
    stream: Stream,
}

impl HostTier {
    /// Maps part `part` (the rank's index in its worker group) of the host region `regions` names
    /// for each cache group of `layout` at the rank's host's Storage Server, registered with
    /// `device`: one page of the group a unit. A group named `None` has no host copy. The copies
    /// run on `stream`.
    ///
    /// # Errors
    ///
    /// The server's refusal of a part, or a part of no page.
    pub(crate) fn map(
        client: &StorageClient,
        layout: &CacheLayout,
        device: DeviceId,
        (regions, part): (&[Option<&str>], u32),
        stream: Stream,
    ) -> Result<Self> {
        // A part the device's rank index creates lies on the host NUMA node closest to the
        // device, so the copies between them cross no link between nodes.
        let numa_node = host_pages::numa_node(device);

        let mut parts = BTreeMap::new();
        for (index, group) in layout.groups.iter().enumerate() {
            let Some(Some(region)) = regions.get(index) else {
                continue;
            };

            if group.window() {
                return Err(Error::Invalid(format!(
                    "cache group {index} keeps a sliding window, whose pages have no host copy"
                )));
            }
            let shape = page_shape(layout, index)?;
            let held = HostPart::map(client, device, region, part, shape.page_bytes, numa_node)?;
            parts.insert(index as u32, (held, group.units_per_page));
        }

        Ok(Self { parts, stream })
    }

    /// Issues the copies a batch runs between the host tier and the units on the copy stream:
    /// `loads` from their host pages into their units, and `stores` from their units into their
    /// host pages; a page of the state
    /// group is a unit of `states`. Returns the event recorded behind them, which the batch's end
    /// waits on; `None` when there is none.
    ///
    /// No forward reads or writes what these copies write before they end: a loaded block
    /// becomes an entry of its owner's table only once the copy is reported done, and a stored
    /// block is shared, which no batch writes. Their sources are held by the buffer pool until
    /// then. So the copy stream waits for nothing of the batch's stream.
    ///
    /// Every page is checked before anything is issued. When an issue fails after others were
    /// issued, the copy stream is waited for before the failure returns, so whatever was issued
    /// has ended.
    ///
    /// # Errors
    ///
    /// A page of a group that has no host region or outside the rank's part of it; the driver's
    /// refusal.
    pub(crate) fn copy(
        &self,
        layout: &CacheLayout,
        (planes, states): (&Planes, Option<&StateUnits>),
        loads: &[HostPage],
        stores: &[HostPage],
    ) -> Result<Option<Event>> {
        if loads.is_empty() && stores.is_empty() {
            return Ok(None);
        }
        let memory = (planes, states);
        for copy in loads.iter().chain(stores) {
            self.runs(layout, memory, copy)?;
        }

        // Timing events bracket the issue: the copy stream's device time is theirs
        // (`profiling::copies`), as CUPTI records no batch of copies.
        let timing = EventFlags {
            timing: true,
            ..EventFlags::default()
        };
        let issued = || -> Result<Event> {
            let started = Event::new(timing);
            started.record(&self.stream)?;
            self.load(layout, memory, loads, &self.stream)?;
            self.store(layout, memory, stores, &self.stream)?;

            let ended = Event::new(timing);
            ended.record(&self.stream)?;
            crate::profiling::copies::record(started, ended);

            let copied = Event::new(EventFlags::default());
            copied.record(&self.stream)?;
            Ok(copied)
        };

        match issued() {
            Ok(copied) => Ok(Some(copied)),
            Err(error) => {
                // The copies issued before the failure end before it is reported; a stream
                // that cannot be waited for has failed with the device.
                if let Err(waited) = self.stream.synchronize() {
                    tracing::error!(%waited, "the copy stream failed after a copy failed");
                }
                Err(error)
            }
        }
    }

    /// One page's copies between its host page and its units, as `(device, host, bytes)`: every
    /// plane of every field in the page's encoding (module documentation), the scales after the
    /// last scaled field; for the state group, the one state unit whole.
    fn runs(
        &self,
        layout: &CacheLayout,
        (planes, states): (&Planes, Option<&StateUnits>),
        copy: &HostPage,
    ) -> Result<Vec<(u64, u64, usize)>> {
        let (group, page, units) = copy;
        let (part, units_per_page) = self
            .parts
            .get(group)
            .ok_or_else(|| Error::Invalid(format!("cache group {group} has no host region")))?;

        let laid = layout
            .groups
            .get(*group as usize)
            .ok_or_else(|| Error::Invalid(format!("the cache holds no group {group}")))?;
        let Some(pool) = pool_of(layout, laid) else {
            // A state page is one unit of the state region, its planes back to back on both
            // sides.
            let states =
                states.ok_or_else(|| Error::Invalid("the cache keeps no state units".into()))?;
            if units.len() != 1 {
                return Err(Error::Invalid(format!(
                    "a checkpoint's page is one state unit, not units {}..{}",
                    units.start, units.end
                )));
            }
            let bytes = states.unit_bytes();
            return Ok(vec![(
                states.address(units.start)?,
                part.address(*page)?,
                bytes as usize,
            )]);
        };
        if units.len() as u32 != *units_per_page {
            return Err(Error::Invalid(format!(
                "a page of group {group} spans {units_per_page} units, not units {}..{}",
                units.start, units.end
            )));
        }

        // Each field's planes in unit then column order, and after the last scaled field the
        // scales of every scaled field in the same order.
        let mut host = part.address(*page)?;
        let block = pool.block_size;
        let scaled = scales_after(pool);
        let mut runs = Vec::new();
        for (index, field) in pool.fields.iter().enumerate() {
            for unit in units.clone() {
                for column in 0..field.columns() {
                    let bytes = field.plane_bytes(column as usize, block);
                    runs.push((
                        planes.plane_address(index, column, unit)?,
                        host,
                        bytes as usize,
                    ));
                    host += bytes;
                }
            }

            if Some(index) == scaled {
                for (index, field) in pool.fields.iter().enumerate() {
                    if !field.scaled {
                        continue;
                    }
                    for unit in units.clone() {
                        for column in 0..field.columns() {
                            runs.push((planes.scale_address(index, column, unit)?, host, 4));
                            host += 4;
                        }
                    }
                }
            }
        }
        Ok(runs)
    }

    /// Loads the units of `loads` from their host pages in `stream` order. The copies overwrite
    /// every column of every field and the scales of a scaled field; a pool that keeps scales
    /// then raises the units' initialization flags, which its writer reads a page's scale by.
    pub(crate) fn load(
        &self,
        layout: &CacheLayout,
        memory: (&Planes, Option<&StateUnits>),
        loads: &[HostPage],
        stream: &Stream,
    ) -> Result<()> {
        let Some((_, first_page, first_units)) = loads.first() else {
            return Ok(());
        };

        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for copy in loads {
            for (device, host, bytes) in self.runs(layout, memory, copy)? {
                batch.push(device, host, bytes);
            }
        }
        tracing::debug!(
            pages = loads.len(),
            first_page,
            first_unit = first_units.start,
            "loading pages from the host KV tier"
        );

        // SAFETY: the buffer pool holds the host pages and gives the units to this copy alone
        // until it is reported done, which is after the batch that runs it has ended.
        unsafe { batch.submit(stream) }?;

        // The flags are the pool's: a state unit has none.
        if layout.kv().quantized() {
            let state = layout
                .state_group()
                .and_then(|group| u32::try_from(group).ok());
            let units: Vec<u32> = loads
                .iter()
                .filter(|(group, _, _)| Some(*group) != state)
                .flat_map(|(_, _, units)| units.clone())
                .collect();
            memory.0.mark_initialized(&units, stream)?;
        }
        Ok(())
    }

    /// Stores the units of `stores` into their host pages in `stream` order.
    pub(crate) fn store(
        &self,
        layout: &CacheLayout,
        memory: (&Planes, Option<&StateUnits>),
        stores: &[HostPage],
        stream: &Stream,
    ) -> Result<()> {
        let Some((_, first_page, first_units)) = stores.first() else {
            return Ok(());
        };

        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for copy in stores {
            for (device, host, bytes) in self.runs(layout, memory, copy)? {
                batch.push(host, device, bytes);
            }
        }
        tracing::debug!(
            pages = stores.len(),
            first_page,
            first_unit = first_units.start,
            "storing pages to the host KV tier"
        );

        // SAFETY: the buffer pool gives the host pages to this copy alone until it is reported
        // done, and the units hold a shared block, which no batch writes.
        unsafe { batch.submit(stream) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nn::family::{CacheField, FieldOwner, Format, History, INDEX_KEY, Record};
    use proto::request::PrefixCacheDtype;

    use super::*;

    /// Four layers of two 128-wide heads, over `block`-token pages.
    fn layout_of(block: u32, dtype: PrefixCacheDtype) -> CacheLayout {
        CacheLayout::of(block, &CacheField::keys_and_values(4, 2, 128), dtype)
            .expect("a dense layout")
    }

    /// 64-token pages of four layers of two 128-wide heads.
    fn layout(dtype: PrefixCacheDtype) -> CacheLayout {
        layout_of(64, dtype)
    }

    /// Index keys of 128 BF16 elements per four tokens over the four layers.
    fn index_keys() -> CacheField {
        CacheField {
            name: INDEX_KEY,
            owners: (0..4).map(|layer| FieldOwner { layer, ratio: 4 }).collect(),
            readers: Vec::new(),
            history: History::Full,
            record: Record::rows(Format::Elements(DType::BF16), 1, 128),
        }
    }

    /// A dense page is both fields of every layer; an FP8 page stores a byte per element and one
    /// F32 scale per field and layer.
    #[test]
    fn a_host_page_holds_both_fields_of_every_layer() -> Result<()> {
        let plane = 64 * 2 * 128 * 2;
        assert_eq!(
            page_shape(&layout(PrefixCacheDtype::BFloat16), 0)?.page_bytes,
            2 * 4 * plane
        );
        assert_eq!(
            page_shape(&layout(PrefixCacheDtype::Float8E4m3Fn), 0)?.page_bytes,
            2 * 4 * (plane / 2) + 2 * 4 * 4
        );
        assert!(page_shape(&layout(PrefixCacheDtype::BFloat16), 1).is_err());
        Ok(())
    }

    /// A page of a cache that keeps index keys holds each layer's index-key plane after its
    /// fields, sixteen keys of 128 BF16 elements for 64 tokens, and is no page of the cache
    /// that keeps none.
    #[test]
    fn a_host_page_holds_the_index_keys_of_every_layer() -> Result<()> {
        let plain = layout(PrefixCacheDtype::BFloat16);
        let mut fields = CacheField::keys_and_values(4, 2, 128);
        fields.push(index_keys());
        let keyed = CacheLayout::of(64, &fields, PrefixCacheDtype::BFloat16)?;

        let (plain, keyed) = (page_shape(&plain, 0)?, page_shape(&keyed, 0)?);
        assert_eq!(keyed.page_bytes, plain.page_bytes + 4 * (16 * 128 * 2));
        assert_ne!(keyed.layout, plain.layout);
        Ok(())
    }

    /// A page's identity names the weights and the page's logical content, not its
    /// representation: the same page stored as FP8 and as BF16 has one key, and other weights
    /// or another page shape make another.
    #[test]
    fn a_page_identity_is_the_weights_and_the_logical_content() {
        let bf16 = page_identities(&layout(PrefixCacheDtype::BFloat16), "weights-a");
        let fp8 = page_identities(&layout(PrefixCacheDtype::Float8E4m3Fn), "weights-a");

        assert_eq!(bf16.len(), 1);
        assert_eq!(bf16, fp8);

        let other_weights = page_identities(&layout(PrefixCacheDtype::BFloat16), "weights-b");
        assert_ne!(bf16, other_weights);

        let wider = layout_of(128, PrefixCacheDtype::BFloat16);
        assert_ne!(bf16, page_identities(&wider, "weights-a"));
    }

    /// A page of a field that is neither keys nor values is its owners' planes, each of its own
    /// ratio's entries, and its record and owners are part of its content: another format or
    /// another owner is another page.
    #[test]
    fn a_page_of_a_block_scaled_field_holds_each_owners_entries() -> Result<()> {
        let latent = |format, ratio| CacheField {
            name: "compressed_latent",
            owners: vec![
                FieldOwner { layer: 2, ratio },
                FieldOwner {
                    layer: 20,
                    ratio: 1,
                },
            ],
            readers: Vec::new(),
            history: History::Full,
            record: Record::rows(format, 1, 512),
        };
        let shape = |format, ratio| {
            let layout = CacheLayout::of(64, &[latent(format, ratio)], PrefixCacheDtype::BFloat16)?;
            page_shape(&layout, 0)
        };

        // Thirty-two NVFP4 entries of 288 bytes, then sixty-four.
        let nvfp4 = shape(Format::Nvfp4, 2)?;
        assert_eq!(nvfp4.page_bytes, (32 + 64) * 288);
        let mxfp8 = shape(Format::Mxfp8, 2)?;
        assert_eq!(mxfp8.page_bytes, (32 + 64) * 528);
        assert_ne!(nvfp4.layout, mxfp8.layout);
        assert_ne!(nvfp4.layout, shape(Format::Nvfp4, 4)?.layout);
        Ok(())
    }
}

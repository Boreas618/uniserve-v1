//! The host tier of the rank's KV cache: the host page pools its pages are copied through.
//!
//! Each cache group copies its pages through a page pool of the host's Storage Server: a memfd
//! of equal pages shared by every rank of the host whose pages have the same shape. The rank
//! maps each pool writable and registers it for device copies; the scheduler decides which
//! pages to copy and states them in batches (`ipc::PageCopy`), and the copies run on the rank's
//! copy stream.
//!
//! A host page is one page of one group on this rank, plane by plane: the key planes of every
//! unit and column of the page, in unit then column order, the value planes in the same order,
//! and for an FP8 cache the scales `[2, planes]` as F32. One layer of one attention page fills
//! exactly one plane, so an attention page is layer-major:
//!
//! 1. keys `[layers, block_size, kv_heads, head_dim]`;
//! 2. values, in the same shape;
//! 3. for an FP8 cache, the scales `[2, layers]` as F32;
//! 4. for a cache that keeps index keys, the index keys `[layers, block_size / ratio, width]`.
//!
//! The state group's page holds its layers' checkpoints in the cache's state planes: every
//! plane of each field run, in unit then column order, and no key, value or scale.

use std::collections::BTreeMap;
use std::sync::Arc;

use ::storage::Key;
use ::storage::protocol::PageShape;
use ipc::{HostPoolInfo, PageCopy};
use platform::{CopyBatch, DeviceId, Event, EventFlags, SourceOrder, Stream};
use uniserve_tensor::DType;

use super::conversion::Conversion;
use super::layout::{CacheLayout, GroupLayout};
use super::planes::Planes;
use crate::memory::client::StorageClient;
use crate::memory::host_pages::{self, HostPages};
use crate::{Error, Result};

/// The version of the page encoding above: pages of another version are another content.
const PAGE_ENCODING: u32 = 1;

/// The pages of the state group's block size between the checkpoints the host tier keeps of a
/// long prompt: the scheduler offloads a state group's checkpoint before every aligned
/// boundary, those of the default `kv_cache_config.boundary_interval`, and before a prompt's
/// last ones.
const STATE_SPACING: u64 = 64;

/// The bytes of one host page of `group` of a cache of `layout`.
fn page_bytes(layout: &CacheLayout, group: &GroupLayout) -> u64 {
    let planes = layout.page_planes(group);
    if group.state() {
        return u64::from(layout.state_fields()) * planes * layout.plane_bytes;
    }

    let scales = if layout.quantized { 2 * planes * 4 } else { 0 };
    2 * planes * layout.plane_bytes + scales + planes * layout.index_bytes()
}

/// The tokens one host page of `group` stands for: an attention or draft page holds its block
/// size's, and a state group keeps one checkpoint per [`STATE_SPACING`] pages.
fn page_tokens(group: &GroupLayout) -> u64 {
    let pages = if group.state() { STATE_SPACING } else { 1 };
    u64::from(group.block_size) * pages
}

/// The bytes one token of `layout`'s pages takes in the host tier over every cache group: each
/// group's page bytes over the tokens a page stands for, rounded up, so the pools of `tokens`
/// tokens take at most `tokens` times as many bytes.
pub fn token_bytes(layout: &CacheLayout) -> u64 {
    layout
        .groups
        .iter()
        .map(|group| page_bytes(layout, group).div_ceil(page_tokens(group)))
        .sum()
}

/// The spelling of a stored element type inside a pool's layout digest. The digest names a
/// page pool every rank of a host shares, whatever build of the engine it runs, so the
/// spellings are fixed here and do not follow a library's.
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
/// size, layers and heads, and the shape of the index keys its layers keep.
fn content_of(digest: &mut ::storage::key::Digest, group: &GroupLayout) {
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

    // A group that keeps none adds nothing, so its pages keep the digest they had.
    if let Some(keys) = group.index_keys {
        digest.text("index_keys");
        digest.number(u64::from(keys.width));
        digest.number(u64::from(keys.ratio));
    }
}

/// Adds the shape of a state page to `digest`: the units and field runs its checkpoints fill.
fn state_of(digest: &mut ::storage::key::Digest, layout: &CacheLayout, group: &GroupLayout) {
    if group.state() {
        digest.text("state");
        digest.number(u64::from(group.units_per_page));
        digest.number(u64::from(layout.state_fields()));
    }
}

/// The digest of the byte layout of `group`'s host pages: the page encoding, the stored
/// representation and the group's shape and logical layers. It names the group's page pool
/// (`PageShape::layout`).
fn layout_digest(layout: &CacheLayout, group: &GroupLayout) -> [u8; 32] {
    let mut digest = ::storage::key::Digest::new(b"uniserve-page-layout");
    digest.number(u64::from(PAGE_ENCODING));
    digest.number(u64::from(layout.quantized));
    digest.number(layout.plane_bytes);
    digest.number(u64::from(layout.columns));
    content_of(&mut digest, group);
    digest.text(stored_name(group.dtype));

    // The index keys are stored in the layers' compute type, whatever the planes store; a
    // group that keeps none adds nothing.
    if group.index_keys.is_some() {
        digest.text(stored_name(group.compute_dtype));
    }
    state_of(&mut digest, layout, group);
    digest.finish()
}

/// The digest of the logical content of `group`'s pages: the page encoding and the group's
/// shape and logical layers, without the representation [`layout_digest`] adds. Pages of one
/// prompt computed by one model in two representations hold the same logical content.
fn content_digest(layout: &CacheLayout, group: &GroupLayout) -> [u8; 32] {
    let mut digest = ::storage::key::Digest::new(b"uniserve-page-content-layout");
    digest.number(u64::from(PAGE_ENCODING));
    content_of(&mut digest, group);
    state_of(&mut digest, layout, group);
    digest.finish()
}

/// The shape of the host pages of group `group` of a cache of `layout`: their layout digest,
/// the rank's head slice and their bytes. A cache's own pages have it, and so do the pages a
/// converted load reads, under the source layout ([`CacheLayout::source`]).
///
/// # Errors
///
/// For a group the layout does not hold.
pub fn page_shape(layout: &CacheLayout, group: usize) -> Result<PageShape> {
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

/// The page identity of each of `layout`'s groups for a model whose weights are `fingerprint`
/// (`PrefixCacheInfo::identities`): the weights and the group's logical page content, as
/// `|`-separated parts. The scheduler seeds the group's page keys at every tier from it
/// (`storage::key::page_seed`), so a page computed by other weights or of another shape is
/// another page. The representation is not part of it: the same page in another stored type has
/// the same key, and the pools of each representation stay apart by their layout digest.
pub fn page_identities(layout: &CacheLayout, fingerprint: &str) -> Vec<String> {
    layout
        .groups
        .iter()
        .map(|group| {
            let content = Key(content_digest(layout, group)).hex();
            format!("{fingerprint}|page-content:{content}")
        })
        .collect()
}

/// The rank's host page pools, by cache group, the stream its page copies run on, and the
/// converted loads of a cache that admits them.
pub(crate) struct HostTier {
    /// Each group's pool with the planes of one field a page of the group spans.
    pools: BTreeMap<u32, (HostPages, u32)>,
    /// The copy stream: page copies run on it beside the forwards on a batch's stream.
    stream: Stream,
    conversion: Option<Conversion>,
    /// Recorded on a batch's stream behind its unit resets; the copy stream starts the batch's
    /// loads behind it, so a reset never clears a page a load wrote.
    reset: Event,
    /// Recorded on the copy stream behind a batch's loads; the batch's stream waits on it, so
    /// its forward reads the loaded pages.
    loaded: Event,
}

impl HostTier {
    /// Maps a page pool for every cache group of `layout` at the rank's host's Storage Server,
    /// registered with `device`, each holding the pages of `tokens` tokens: the count the
    /// engine sizes every rank's host tier at. A pool the deployment already has for the same
    /// pages, which an earlier epoch created, is shared at its size. The copies run on
    /// `stream`, and `conversion` is the converted loads of a cache that admits them, opened
    /// on the same stream. Returns the tier and what the rank reports of each pool, in group
    /// order, without their conversions.
    ///
    /// # Errors
    ///
    /// [`Error::Resource`] when `tokens` hold no page of a group; the server's refusal.
    pub(crate) fn map(
        client: &StorageClient,
        layout: &Arc<CacheLayout>,
        device: DeviceId,
        endpoint: Option<&str>,
        tokens: u64,
        stream: Stream,
        conversion: Option<Conversion>,
    ) -> Result<(Self, Vec<HostPoolInfo>)> {
        let mut pools = BTreeMap::new();
        let mut reports = Vec::new();

        // The pools lie on the host NUMA node closest to the device, so the copies between
        // them cross no link between nodes.
        let numa_node = host_pages::numa_node(device);

        for (index, group) in layout.groups.iter().enumerate() {
            let shape = page_shape(layout, index)?;
            let pages = tokens / page_tokens(group);
            if pages == 0 {
                return Err(Error::Resource(
                    "the host KV tier holds no page of each cache group",
                ));
            }

            let pool = HostPages::map(
                client,
                device,
                shape,
                numa_node,
                Some(pages * shape.page_bytes),
                "a host KV tier pool",
            )?;
            reports.push(HostPoolInfo {
                pool: pool.pool(),
                shape: pool.shape(),
                endpoint: endpoint.map(str::to_owned),
                conversion: None,
            });
            pools.insert(index as u32, (pool, layout.page_planes(group) as u32));
        }

        Ok((
            Self {
                pools,
                stream,
                conversion,
                reset: Event::new(EventFlags::default()),
                loaded: Event::new(EventFlags::default()),
            },
            reports,
        ))
    }

    /// The generation the converted loads launch from, from the next load on.
    pub(crate) fn set_kernels(&self, kernels: Arc<ops::Generation>) {
        if let Some(conversion) = &self.conversion {
            conversion.set_kernels(kernels);
        }
    }

    /// Issues a batch's page copies on the copy stream: loads the units of `loads` from their
    /// host pages, converting the pages of the conversion source (`PageCopy::staged`), and
    /// offloads the published units of `offloads` into theirs.
    ///
    /// The loads fill new units of the batch, which the batch reset on `stream`, the batch's
    /// own: the copy stream starts them behind that reset, and `stream` waits for them before
    /// the batch's forward. The offloads copy pages the scheduler published once it accepted
    /// the result that computed them, and it holds their units until the batch's output commits
    /// them; that output waits on the returned event, recorded behind the offloads, and nothing
    /// else does. `None` for a batch that offloads nothing.
    ///
    /// # Errors
    ///
    /// A page of another pool or shape than its group's, or a page of the conversion source on
    /// a cache that converts none.
    pub(crate) fn copy(
        &self,
        layout: &CacheLayout,
        planes: &Planes,
        loads: &[PageCopy],
        offloads: &[PageCopy],
        stream: &Stream,
    ) -> Result<Option<Event>> {
        if !loads.is_empty() {
            let (converted, pooled): (Vec<&PageCopy>, Vec<&PageCopy>) =
                loads.iter().partition(|copy| copy.staged);
            if !converted.is_empty() && self.conversion.is_none() {
                return Err(Error::Invalid(
                    "a batch loads a page of a conversion source and the cache converts none"
                        .into(),
                ));
            }

            self.reset.record(stream)?;
            self.stream.wait_event(&self.reset)?;

            self.load(layout, planes, &pooled, &self.stream)?;
            if let Some(conversion) = &self.conversion {
                conversion.load(planes, &converted, &self.stream)?;
            }

            self.loaded.record(&self.stream)?;
            stream.wait_event(&self.loaded)?;
        }

        if offloads.is_empty() {
            return Ok(None);
        }
        self.offload(layout, planes, offloads, &self.stream)?;

        let offloaded = Event::new(EventFlags::default());
        offloaded.record(&self.stream)?;
        Ok(Some(offloaded))
    }

    /// Copies `offloads` to their host pages and waits for the copies, with no batch to order
    /// them behind: what the scheduler published has been computed, and it holds the units
    /// meanwhile.
    pub(crate) fn offload_now(
        &self,
        layout: &CacheLayout,
        planes: &Planes,
        offloads: &[PageCopy],
    ) -> Result<()> {
        self.offload(layout, planes, offloads, &self.stream)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// One page's copies between its host page and its units, as `(device, host, bytes)`: every
    /// field and plane, then every scale, then every index-key plane.
    fn runs(
        &self,
        layout: &CacheLayout,
        planes: &Planes,
        copy: &PageCopy,
    ) -> Result<Vec<(u64, u64, usize)>> {
        let (pool, page_planes) = self
            .pools
            .get(&copy.group_id)
            .filter(|(pool, _)| pool.pool() == copy.pool)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "page pool {} is not the pool of cache group {}",
                    copy.pool, copy.group_id
                ))
            })?;
        let page_planes = *page_planes;

        let columns = layout.columns;
        if copy.units.len() as u32 * columns != page_planes {
            return Err(Error::Invalid(format!(
                "a page of {page_planes} planes spans {} units of {columns} columns, not {}",
                page_planes.div_ceil(columns),
                copy.units.len()
            )));
        }

        let plane = layout.plane_bytes;
        let host = pool.address(copy.host_page)?;
        let count = u64::from(page_planes);
        let unit = |index: u32| copy.units[(index / columns) as usize].0;

        let state = layout
            .groups
            .get(copy.group_id as usize)
            .is_some_and(GroupLayout::state);
        if state {
            // A state page is its checkpoints' planes, field run by field run.
            let field_runs = layout.state_fields() as usize;
            let mut runs = Vec::with_capacity(field_runs * page_planes as usize);

            for run in 0..field_runs {
                for index in 0..page_planes {
                    let device = planes.state_plane_address(run, index % columns, unit(index))?;
                    let host = host + (run as u64 * count + u64::from(index)) * plane;
                    runs.push((device, host, plane as usize));
                }
            }
            return Ok(runs);
        }

        let mut runs = Vec::with_capacity(4 * page_planes as usize);
        for field in 0..2 {
            for index in 0..page_planes {
                let device = planes.plane_address(field, index % columns, unit(index))?;
                let host = host + (field as u64 * count + u64::from(index)) * plane;
                runs.push((device, host, plane as usize));
            }
        }
        if layout.quantized {
            for field in 0..2 {
                for index in 0..page_planes {
                    let device = planes.scale_address(field, index % columns, unit(index))?;
                    let host =
                        host + 2 * count * plane + (field as u64 * count + u64::from(index)) * 4;
                    runs.push((device, host, 4));
                }
            }
        }

        let index_bytes = layout.index_bytes();
        if index_bytes > 0 {
            let scales = if layout.quantized { 2 * count * 4 } else { 0 };

            for index in 0..page_planes {
                let device = planes.index_address(index % columns, unit(index))?;
                let host = host + 2 * count * plane + scales + u64::from(index) * index_bytes;
                runs.push((device, host, index_bytes as usize));
            }
        }
        Ok(runs)
    }

    /// Loads the units of `loads` from their host pages in `stream` order. The units are new
    /// units of the batch, which the copies overwrite whole (every column of both fields, an
    /// FP8 cache's scales and the index keys of a cache that keeps them); an FP8 cache then
    /// raises their initialization flags.
    pub(crate) fn load(
        &self,
        layout: &CacheLayout,
        planes: &Planes,
        loads: &[&PageCopy],
        stream: &Stream,
    ) -> Result<()> {
        let Some(first) = loads.first() else {
            return Ok(());
        };

        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for copy in loads {
            for (device, host, bytes) in self.runs(layout, planes, copy)? {
                batch.push(device, host, bytes);
            }
        }
        tracing::debug!(
            pages = loads.len(),
            first_page = first.host_page,
            first_unit = first.units.first().map(|unit| unit.0),
            "loading pages from the host KV tier"
        );

        // SAFETY: the scheduler pins the host pages until the batch's output arrives, and the
        // units are the batch's new units, which nothing else reads or writes before the
        // batch's forward, which waits for these copies.
        unsafe { batch.submit(stream) }?;

        if layout.quantized {
            let units: Vec<u32> = loads
                .iter()
                .flat_map(|copy| copy.units.iter().map(|unit| unit.0))
                .collect();
            planes.mark_initialized(&units, stream)?;
        }
        Ok(())
    }

    /// Offloads the units of the published pages of `offloads` into their host pages in
    /// `stream` order.
    pub(crate) fn offload(
        &self,
        layout: &CacheLayout,
        planes: &Planes,
        offloads: &[PageCopy],
        stream: &Stream,
    ) -> Result<()> {
        let Some(first) = offloads.first() else {
            return Ok(());
        };

        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for copy in offloads {
            for (device, host, bytes) in self.runs(layout, planes, copy)? {
                batch.push(host, device, bytes);
            }
        }
        tracing::debug!(
            pages = offloads.len(),
            first_page = first.host_page,
            first_unit = first.units.first().map(|unit| unit.0),
            "offloading pages to the host KV tier"
        );

        // SAFETY: the host pages are reserved for this rank until the scheduler commits them
        // after the batch's output, and it holds the published units, which no batch writes,
        // until then.
        unsafe { batch.submit(stream) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ipc::PrefixCacheDtype;

    use super::*;

    /// 64-token pages of four layers of two 128-wide heads.
    fn layout(dtype: PrefixCacheDtype) -> CacheLayout {
        CacheLayout::dense(64, 4, 2, 128, dtype)
            .validated()
            .expect("a dense layout")
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

        // A token takes its share of a page, rounded up.
        assert_eq!(
            token_bytes(&layout(PrefixCacheDtype::BFloat16)),
            2 * 4 * plane / 64
        );
        Ok(())
    }

    /// A page of a cache that keeps index keys holds each layer's index-key plane after its
    /// fields, sixteen keys of 128 BF16 elements for 64 tokens, and is no page of the cache
    /// that keeps none.
    #[test]
    fn a_host_page_holds_the_index_keys_of_every_layer() -> Result<()> {
        let plain = layout(PrefixCacheDtype::BFloat16);
        let keyed = plain.clone().with_index_keys(nn::family::IndexKeys {
            width: 128,
            ratio: 4,
        })?;

        let (plain, keyed) = (page_shape(&plain, 0)?, page_shape(&keyed, 0)?);
        assert_eq!(keyed.page_bytes, plain.page_bytes + 4 * (16 * 128 * 2));
        assert_ne!(keyed.layout, plain.layout);
        Ok(())
    }

    /// A converting FP8 cache asks for exactly the pages a BF16 deployment of the same model
    /// stores: the source layout's page shape is the BF16 cache's, and differs from its own.
    #[test]
    fn an_fp8_cache_converts_the_pages_a_bf16_cache_stores() -> Result<()> {
        let fp8 = layout(PrefixCacheDtype::Float8E4m3Fn).with_conversion()?;
        let source = fp8.source().expect("a converting layout has a source");
        let bf16 = layout(PrefixCacheDtype::BFloat16);

        assert_eq!(page_shape(&source, 0)?, page_shape(&bf16, 0)?);
        assert_ne!(page_shape(&fp8, 0)?, page_shape(&bf16, 0)?);
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
        assert!(bf16[0].starts_with("weights-a|page-content:"));
        assert_eq!(bf16, fp8);

        let other_weights = page_identities(&layout(PrefixCacheDtype::BFloat16), "weights-b");
        assert_ne!(bf16, other_weights);

        let wider = CacheLayout::dense(128, 4, 2, 128, PrefixCacheDtype::BFloat16);
        assert_ne!(bf16, page_identities(&wider, "weights-a"));
    }
}

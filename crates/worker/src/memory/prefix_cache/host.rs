//! The host KV cache (L2) of the rank's prefix cache.
//!
//! Each cache group copies its pages through a page pool of the host's Storage
//! Server: a memfd of equal slots shared by every worker of the host whose
//! pages have the same shape. The rank maps each pool writable and registers it
//! for device copies; the engine decides which pages to copy and carries them
//! in batches (`Batch::page_loads`, `Batch::page_offloads`), and the copies run
//! on the rank's copy stream as soon as the batch arrives
//! (`PrefixCache::issue_pages`).
//!
//! A page is one page of one group on this rank, plane by plane: the key
//! planes of every unit and column of the page, in unit then column order,
//! the value planes in the same order, and for FP8 pools the scales `[2,
//! planes]` as `f32`. A K/V group's layer `k` lies in column `k % columns`
//! of the page's unit `k / columns`, and one layer of one page fills exactly
//! one plane of the unit pool (a group's block size is the plane over its
//! row bytes), so a K/V page is layer-major:
//!
//! 1. keys `[layers, block_size, kv_heads, head_dim]`;
//! 2. values, in the same shape;
//! 3. for FP8 pools, the scales `[2, layers]` as `f32`.
//!
//! The state group's page holds its layers' checkpoints in the pool's state
//! planes (`CachePlanes::state_views`), in the order the plan lays them out:
//! a state page copies every plane of each field run, in unit then column
//! order, and holds no key, value or scale. A K/V page copies as one
//! plane-sized run per field, unit and column, and one scale per field, unit
//! and column.

use std::collections::BTreeMap;

use ::storage::Key;
use ::storage::protocol::PageShape;
use ipc::{HostPoolInfo, PageCopy};
use platform::{CopyBatch, DeviceId, SourceOrder, Stream};

use super::plan::{CacheGroupPlan, CachePlan};
use super::planes::{CachePlanes, UnitColumn};
use crate::memory::host_pages::{self, HostPages};
use crate::memory::retained::Retained;
use crate::stream::Issue;
use crate::{Error, Result};

/// The version of the page encoding above: pages of another version are
/// another content.
const PAGE_ENCODING: u32 = 1;

/// One group's page pool, mapped into the rank.
struct GroupPool {
    pages: HostPages,
    /// Planes of one field a page spans (`CachePlan::page_planes`).
    planes: u32,
}

/// The rank's page pools, by cache group.
pub(crate) struct HostKvCache {
    pools: BTreeMap<u32, GroupPool>,
}

/// The bytes of one page of `group` on a pool of `plan`'s layout.
fn page_bytes(plan: &CachePlan, group: &CacheGroupPlan) -> u64 {
    let planes = plan.page_planes(group);
    if group.state() {
        return u64::from(plan.state_fields()) * planes * plan.plane_bytes;
    }
    let scales = if plan.quantized { 2 * planes * 4 } else { 0 };
    2 * planes * plan.plane_bytes + scales
}

/// The pages of the state group's block size between the checkpoints the
/// host KV cache keeps of a long prompt: the engine offloads a state
/// group's checkpoint before every aligned boundary, those of the default
/// `kv_cache_config.boundary_interval`, and before a prompt's last ones.
const STATE_SPACING: u64 = 64;

/// The tokens one host page of `group` stands for: a K/V group's page holds
/// its block size's, and a state group keeps one checkpoint per
/// [`STATE_SPACING`] pages.
fn page_tokens(group: &CacheGroupPlan) -> u64 {
    let pages = if group.state() { STATE_SPACING } else { 1 };
    u64::from(group.block_size) * pages
}

/// The bytes one token of `plan`'s pages takes over every cache group: each
/// group's page bytes over the tokens a page stands for, rounded up, so that
/// the pools of `tokens` tokens ([`HostKvCache::start`]) take at most
/// `tokens` times as many bytes.
pub(crate) fn token_bytes(plan: &CachePlan) -> u64 {
    plan.groups
        .iter()
        .map(|group| page_bytes(plan, group).div_ceil(page_tokens(group)))
        .sum()
}

/// The digest of the byte layout of `group`'s pages: the page encoding,
/// the cache dtype and this group's shape and logical layers. It
/// identifies the group's page pool (`PageShape::layout`) and is part of
/// the group's page identity ([`page_identities`]).
fn layout(plan: &CachePlan, group: usize) -> [u8; 32] {
    let entry = &plan.groups[group];
    let mut digest = ::storage::key::Digest::new(b"uniserve-page-layout");
    digest.number(u64::from(PAGE_ENCODING));
    digest.number(u64::from(plan.quantized));
    digest.number(plan.plane_bytes);
    digest.number(u64::from(plan.columns));
    digest.number(u64::from(entry.window().unwrap_or(0)));
    digest.number(u64::from(entry.block_size));
    let layers: Vec<u64> = entry
        .layer_ids
        .iter()
        .map(|&layer| u64::from(layer))
        .collect();
    digest.numbers(&layers);
    digest.number(u64::from(entry.total_kv_heads));
    digest.number(u64::from(entry.head_dim));
    digest.text(entry.dtype.name());
    // A state group's pages are its checkpoints' planes, as many units as
    // they fill, laid field run by field run.
    if entry.state() {
        digest.text("state");
        digest.number(u64::from(entry.units_per_page));
        digest.number(u64::from(plan.state_fields()));
    }
    digest.finish()
}

/// The digest of the logical content of `group`'s pages: the page encoding
/// and the group's shape and logical layers, without its representation
/// (the cache dtype, its quantization and the planes' packing), which
/// [`layout`] adds. Pages of one prompt computed by one model in two
/// representations hold the same logical content.
fn content(plan: &CachePlan, group: usize) -> [u8; 32] {
    let entry = &plan.groups[group];
    let mut digest = ::storage::key::Digest::new(b"uniserve-page-content-layout");
    digest.number(u64::from(PAGE_ENCODING));
    digest.number(u64::from(entry.window().unwrap_or(0)));
    digest.number(u64::from(entry.block_size));
    let layers: Vec<u64> = entry
        .layer_ids
        .iter()
        .map(|&layer| u64::from(layer))
        .collect();
    digest.numbers(&layers);
    digest.number(u64::from(entry.total_kv_heads));
    digest.number(u64::from(entry.head_dim));
    if entry.state() {
        digest.text("state");
        digest.number(u64::from(entry.units_per_page));
    }
    digest.finish()
}

/// The page identity of each of `plan`'s groups for a model whose weights
/// are `fingerprint` (`PrefixCacheInfo::identities`): the weights and the
/// group's logical page content, as `|`-separated components. The engine
/// seeds the group's page keys at every cache level from it
/// (`storage::key::page_seed`), so a page computed by other weights or of
/// another shape is another page on the device, in the host page pools and
/// in the KV storage. The representation is not part of it: the same page
/// in another cache dtype has the same key, and the pools and KV storage
/// objects of each representation stay apart by their layout
/// (`PageShape::layout`, [`layout`]).
pub fn page_identities(plan: &CachePlan, fingerprint: &str) -> Vec<String> {
    (0..plan.groups.len())
        .map(|group| {
            let content = ::storage::Key(content(plan, group)).hex();
            format!("{fingerprint}|page-content:{content}")
        })
        .collect()
}

/// What a rank whose pages the engine copies through the host KV cache
/// keeps until the engine sizes it (`uniserve.host_cache`).
pub(crate) struct HostCacheSetup {
    /// Where an engine on another host reaches the host's Storage Server.
    pub(crate) endpoint: Option<String>,
    /// The cache's device.
    pub(crate) device: DeviceId,
    /// The copy stream the pages are copied on, beside the forwards on the
    /// primary stream.
    pub(crate) copy: Stream,
}

impl HostCacheSetup {
    /// Map the page pools of `cache`, `tokens` tokens in each cache group,
    /// and copy the cache's pages through them from now on. Returns what
    /// the rank reports of each pool.
    pub(crate) fn map(
        self,
        retained: &Retained,
        cache: &mut super::PrefixCache,
        tokens: u64,
    ) -> Result<Vec<HostPoolInfo>> {
        let (host, pools) = HostKvCache::start(
            retained,
            cache.plan(),
            self.device,
            self.endpoint.as_deref(),
            tokens,
        )?;
        cache.set_host_kv_cache(host, self.copy)?;
        Ok(pools)
    }
}

impl HostKvCache {
    /// Map a page pool for every cache group of `plan` at the rank's host's
    /// Storage Server, registered with `device`. A sliding-window group's
    /// pool holds the windows the engine offloads before aligned
    /// boundaries, page by page, as a full-attention group's holds every
    /// published page.
    /// Each group's pool holds the pages of `tokens` tokens, the count the
    /// engine sizes every rank of the epoch's host KV cache at; a pool the
    /// deployment already has for the same pages, which a previous epoch
    /// created, is shared at its size.
    /// Returns the host KV cache and what the rank reports of each pool.
    pub(crate) fn start(
        retained: &Retained,
        plan: &CachePlan,
        device: DeviceId,
        endpoint: Option<&str>,
        tokens: u64,
    ) -> Result<(Self, Vec<HostPoolInfo>)> {
        let mut pools = BTreeMap::new();
        let mut infos = Vec::new();
        for (index, group) in plan.groups.iter().enumerate() {
            let layout = layout(plan, index);
            // The pool lies on the host NUMA node closest to the device, so
            // the copies between them cross no inter-node link.
            let numa_node = host_pages::numa_node(device);
            let shape = PageShape {
                layout: Key(layout),
                kv_head_offset: group.kv_head_offset,
                kv_head_count: group.num_kv_heads,
                page_bytes: page_bytes(plan, group),
            };
            let slots = tokens / page_tokens(group);
            if slots == 0 {
                return Err(Error::Resource(
                    "the host KV cache holds no page of each cache group",
                ));
            }
            let pages = HostPages::map(
                retained,
                device,
                shape,
                numa_node,
                Some(slots * shape.page_bytes),
                "a host KV cache pool",
            )?;
            let info = HostPoolInfo {
                pool: pages.pool(),
                shape: pages.shape(),
                endpoint: endpoint.map(str::to_owned),
            };
            pools.insert(
                index as u32,
                GroupPool {
                    pages,
                    planes: plan.page_planes(group) as u32,
                },
            );
            infos.push(info);
        }
        Ok((Self { pools }, infos))
    }

    /// One page's copies between its slot and its units: `(device, host,
    /// bytes)` for every field and plane, then every scale.
    fn runs(&self, planes: &CachePlanes, copy: &PageCopy) -> Result<Vec<(u64, u64, usize)>> {
        let pool = self
            .pools
            .get(&copy.group_id)
            .filter(|pool| pool.pages.pool() == copy.pool)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "page pool {} is not the pool of cache group {}",
                    copy.pool, copy.group_id
                ))
            })?;
        let plan = planes.plan();
        let columns = plan.columns;
        if copy.units.len() as u32 * columns != pool.planes {
            return Err(Error::Invalid(format!(
                "a page of {} planes spans {} units of {columns} columns, not {}",
                pool.planes,
                pool.planes.div_ceil(columns),
                copy.units.len()
            )));
        }
        let plane = plan.plane_bytes;
        let slot = pool.pages.address(copy.host_page)?;
        let count = u64::from(pool.planes);
        let state = plan
            .groups
            .get(copy.group_id as usize)
            .is_some_and(|group| group.state());
        if state {
            // A state page is its checkpoints' planes, field run by field run.
            let mut runs = Vec::with_capacity(plan.state_fields() as usize * pool.planes as usize);
            for field in 0..plan.state_fields() as usize {
                for index in 0..pool.planes {
                    let unit = copy.units[(index / columns) as usize].0;
                    let device = planes.state_plane_address(field, index % columns, unit)?;
                    let host = slot + (field as u64 * count + u64::from(index)) * plane;
                    runs.push((device, host, plane as usize));
                }
            }
            return Ok(runs);
        }
        let mut runs = Vec::with_capacity(2 * pool.planes as usize * 2);
        for field in 0..2 {
            for index in 0..pool.planes {
                let unit = copy.units[(index / columns) as usize].0;
                let device = planes.plane_address(field, index % columns, unit)?;
                let host = slot + (field as u64 * count + u64::from(index)) * plane;
                runs.push((device, host, plane as usize));
            }
        }
        if plan.quantized {
            for field in 0..2 {
                for index in 0..pool.planes {
                    let unit = copy.units[(index / columns) as usize].0;
                    let device = planes.scale_address(field, index % columns, unit)?;
                    let host =
                        slot + 2 * count * plane + (field as u64 * count + u64::from(index)) * 4;
                    runs.push((device, host, 4));
                }
            }
        }
        Ok(runs)
    }

    /// Load the units of `loads` from their slots in `stream` order. The
    /// units are new units of the batch, which the copies overwrite whole
    /// (every column of both fields, and an FP8 pool's scales); an FP8 pool
    /// then marks them initialized with `column`, bound to `stream`.
    pub(crate) fn load(
        &self,
        planes: &CachePlanes,
        loads: &[PageCopy],
        column: &UnitColumn,
        stream: impl Issue,
    ) -> Result<()> {
        if loads.is_empty() {
            return Ok(());
        }
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for copy in loads {
            for (device, host, bytes) in self.runs(planes, copy)? {
                batch.push(device, host, bytes);
            }
        }
        tracing::debug!(
            pages = loads.len(),
            first_slot = loads[0].host_page,
            first_unit = loads[0].units.first().map(|unit| unit.0),
            "loading pages from the host KV cache"
        );
        // SAFETY: the slots are pinned by the engine until the batch's output
        // arrives, and the units are the batch's new units, which no other
        // operation reads or writes before the batch's forward, which waits
        // for these copies.
        unsafe { stream.copy(&batch) }?;
        if planes.plan().quantized {
            let units: Vec<u32> = loads
                .iter()
                .flat_map(|copy| copy.units.iter().map(|unit| unit.0))
                .collect();
            column.mark_initialized(&units, stream)?;
        }
        Ok(())
    }

    /// Offload the units of the published pages of `offloads` into their
    /// slots in `stream` order; the batch's output waits for the copies.
    pub(crate) fn offload(
        &self,
        planes: &CachePlanes,
        offloads: &[PageCopy],
        stream: impl Issue,
    ) -> Result<()> {
        if offloads.is_empty() {
            return Ok(());
        }
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for copy in offloads {
            for (device, host, bytes) in self.runs(planes, copy)? {
                batch.push(host, device, bytes);
            }
        }
        tracing::debug!(
            pages = offloads.len(),
            first_slot = offloads[0].host_page,
            first_unit = offloads[0].units.first().map(|unit| unit.0),
            "offloading pages to the host KV cache"
        );
        // SAFETY: the slots are reserved for this rank until the engine
        // commits them after the batch's output, and the engine holds the
        // published units, which no batch writes, until then.
        unsafe { stream.copy(&batch) }
    }
}

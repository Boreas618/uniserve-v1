//! The rank's KV cache: the memory a causal forward reads and writes its keys and values in.
//!
//! "KV cache" is the reference engines' word for it (vLLM's `KVCacheManager`, SGLang's token to
//! KV pool); the scheduler's side allocates its pages and indexes them by prefix, and this
//! module is the rank's side. [`KvCacheMemory`] owns:
//!
//! - the **planes** ([`planes`]): the KV pool's exclusive storage at the host's Storage
//!   Server, the memory of the cache's buffer pool region, every field's planes with each
//!   plane's initialization flag and, for a field in the deployment's FP8 type, its scale;
//! - the **window planes**: each sliding-window group's pool, memory of the rank's own process
//!   sized by its request slots ([`KvCacheConfig::windows`]);
//! - the **pooled page tables** ([`tables`]): one per field group, the device copy of the
//!   group's entries of the rank's device block table, which a forward gathers its rows from;
//! - the **attention workspace**, which the paged-attention kernel uses during a forward;
//! - the **host tier** ([`host`]): the rank's parts of the host regions of the cache groups'
//!   pages, which pages are loaded from and stored to on the rank's copy stream;
//! - the **scratch pages** ([`scratch`]) a shadow forward writes;
//! - for a model that keeps a recurrent state, the **state units** ([`states`]): the memory of
//!   the scheduler's state region, one checkpoint a unit, which the state copies and the host
//!   tier read and write.
//!
//! A model reads and writes none of this directly: the runner that holds the cache lends it for
//! a forward as `nn::KvPlane`s and page-table tensors, external tensors over this memory.
//!
//! ## Two forms
//!
//! [`KvCacheMemory::open`] takes the cache's storage on a device. [`KvCacheMemory::meta`] makes
//! a cache that holds none: every tensor it lends is a meta tensor of the same shape and
//! element type, the page tables are kept on the host alone, and what needs a device is a
//! refusal. A profile run and the host tests compute over the second form.
//!
//! ## A batch's cache work
//!
//! The buffer pool's completions reach the rank's device block table
//! ([`BlockTables`](crate::memory::tables::BlockTables)), whose runs name the cache's memory by
//! region ([`CacheRegions`]). The runner that holds the cache issues a batch's cache work on
//! the worker thread, before the batch's forward, in this order:
//!
//! 1. [`KvCacheMemory::apply`]: the block table's changes since the last batch, on the batch's
//!    stream: the units the buffer pool allocated are reset, and the pooled tables' rows that
//!    changed are written;
//! 2. [`KvCacheMemory::copy`]: the copies the batch takes, between runs of the units and of
//!    the host tier, whose event the batch's end waits for;
//! 3. the restores of recurrent state
//!    ([`StateMemory::copy`](crate::memory::state::StateMemory::copy)), then the forward, then
//!    the keeps.
//!
//! The class of cache group `g` is class `g` of the buffer pool, and the classes past the
//! cache groups are no cache's. The state group's class takes its blocks from the state region,
//! whose runs name state units; a sliding-window group's from its own region of window units;
//! every other cache group's from the KV pool.
//!
//! ## Two epochs of one region
//!
//! Every storage of the planes is keyed by the region's memory, so a cache that another epoch
//! opens from the region's segment table ([`segments::of_table`]) addresses the same cells:
//! values, flags, scales and index keys; and the state units likewise by the state region's. Nothing of the cache is copied or handed
//! over at an epoch switch (`specs/updates.md` §3.4): the buffer pool keeps the two epochs'
//! blocks apart, and the units it gives an epoch are reset by that epoch's
//! [`KvCacheMemory::apply`] before a batch writes them.
//!
//! ## Index tensors
//!
//! Page tables and unit ids are `u32`. A pooled table never holds `nn::NO_SLOT`: a page a row
//! does not hold names unit 0, the padding unit, because a table entry feeds a gather. The slot
//! column of a shadow forward, which its runner numbers over the scratch set's pages, holds
//! `nn::NO_SLOT` where a token writes nothing, as the slot column of a served batch does.

pub(crate) mod host;
pub(crate) mod info;
pub(crate) mod layout;
mod planes;
pub(crate) mod scratch;
pub mod segments;
mod states;
mod tables;

use std::sync::Arc;

use self::info::PrefixCacheInfo;
use nn::{KvCache, KvPlane};
use platform::{DeviceId, Event, Stream};
use storage::runtime::Run;
use tensor::{DType, Tensor};

use self::host::{HostPage, HostTier};
pub use self::layout::{CacheLayout, Checkpoint, FieldLayout, GroupLayout, PoolLayout};
use self::planes::Planes;
use self::scratch::ScratchPages;
pub(crate) use self::segments::PlaneSegment;
use self::states::StateUnits;
use self::tables::PageTables;
use crate::memory::allocation::Storage;
use crate::memory::client::StorageClient;
use crate::memory::exclusive::ExclusiveStorage;
use crate::memory::lend;
use crate::memory::tables::Change;
use crate::worker::{Error, Result};

/// The bytes the trtllm-gen paged-attention kernel takes as its workspace: what
/// [`KvCacheConfig::workspace_bytes`] is for a model that attends through it.
pub(crate) const ATTENTION_WORKSPACE_BYTES: u64 = 512 << 20;

/// A rank's KV cache as it is opened: the model's layout and the capacity the rank resolved
/// for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// The cache's layout, validated ([`CacheLayout::validated`]).
    pub layout: Arc<CacheLayout>,
    /// Units of the pool, the padding unit included.
    pub units: u32,
    /// Request slots: slots `1..=slots` own a row of every pooled table.
    pub slots: u32,
    /// Pages per row of a pooled table: the pages a request of the model's length holds.
    pub width: u32,
    /// The segments the value planes are made of on a device ([`segments`]), which hold `units`
    /// units; `None` for planes that are not split.
    pub segments: Option<Vec<PlaneSegment>>,
    /// The bytes of the attention workspace a forward is lent.
    pub workspace_bytes: u64,
    /// For a layout with a state group, the units of the state region (one checkpoint each) and
    /// the segments they are made of on a device, `None` for units that are not split; zero
    /// units and `None` for a layout without one.
    pub states: (u32, Option<Vec<PlaneSegment>>),
    /// The units of each sliding-window group's pool, the padding unit included, in the order of
    /// the layout's pools after the KV pool ([`window_units`]); empty for a layout of no window.
    pub windows: Vec<u32>,
}

/// The units of the pool of a sliding window of `window` tokens over pages of `page` tokens for
/// `slots` request slots whose steps hold `tokens` tokens at most, the padding unit included:
/// every running request's window pages at once, beside the pages of a step's tokens.
///
/// A request holds the pages of the window before its next token and of the tokens it computes,
/// which the scheduler retires behind the window as it goes, and asks for a few pages ahead
/// (`crate::scheduler::LOW_WATERMARK`): at most `⌈window / page⌉ + 2` pages of the window and
/// of its partial pages, beside the watermark, and the step's tokens take their own pages,
/// `⌈tokens / page⌉` beyond that over every request of the step. Nothing in the pool outlives a
/// request, so it is sized for the slots alone and never grows with the KV pool.
pub(crate) fn window_units(window: u32, page: u32, slots: u32, tokens: u32) -> u32 {
    let page = page.max(1);
    let per_request = window.div_ceil(page) + 2 + crate::scheduler::LOW_WATERMARK;

    1 + slots.saturating_mul(per_request) + tokens.div_ceil(page)
}

impl KvCacheConfig {
    /// The device bytes the cache holds beside its KV planes and state units: the pooled page
    /// tables, the attention workspace and the sliding windows' pools, all of which the request
    /// slots size.
    pub(crate) fn fixed_bytes(&self) -> u64 {
        let windows: u64 = self
            .layout
            .pools
            .iter()
            .skip(1)
            .zip(&self.windows)
            .map(|(pool, units)| u64::from(*units) * pool.unit_bytes())
            .sum();

        PageTables::device_bytes(&self.layout, self.slots, self.width)
            + self.workspace_bytes
            + windows
    }

    /// The units of each pool, the padding unit included, in the layout's pool order: the KV
    /// pool's, then each window's.
    pub(crate) fn pool_units(&self) -> Vec<u32> {
        std::iter::once(self.units)
            .chain(self.windows.iter().copied())
            .collect()
    }

    /// Checks what the cache's tensors rely on: a workspace, at least one attention group, a
    /// unit count of each window pool, and state units exactly when the layout keeps a state
    /// group.
    fn check(&self) -> Result<()> {
        if self.layout.attention_groups() == 0 {
            return Err(Error::Invalid("a KV cache needs an attention group".into()));
        }
        if self.windows.len() + 1 != self.layout.pools.len() {
            return Err(Error::Invalid(format!(
                "a cache of {} pools states the units of {} sliding windows",
                self.layout.pools.len(),
                self.windows.len()
            )));
        }
        if self.layout.state_group().is_some() != (self.states.0 > 0) {
            return Err(Error::Invalid(format!(
                "a cache {} a state group states {} state units",
                if self.layout.state_group().is_some() {
                    "with"
                } else {
                    "without"
                },
                self.states.0
            )));
        }
        if self.workspace_bytes == 0 {
            return Err(Error::Invalid(
                "a KV cache needs the attention workspace a forward is lent".into(),
            ));
        }

        for (index, group) in self.layout.groups.iter().enumerate() {
            if !group.attention() {
                continue;
            }

            // A token has one cache slot per batch, `unit · block_size + offset`, and a row of
            // a pooled table one unit per page: a page of several units would need one of each
            // per unit.
            if group.units_per_page != 1 {
                return Err(Error::Unsupported(format!(
                    "attention group {index} lays a page over {} units, and a token's cache \
                     slot and a page-table entry name one",
                    group.units_per_page
                )));
            }
        }
        Ok(())
    }
}

/// The numbers the buffer pool gave the regions the cache's memory is (the answer to `Open`):
/// what tells a run a completion states, `(region, first unit, count)`, as units of the pool,
/// state units or pages of the host tier.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CacheRegions {
    /// The region of the pool's units on the device: unit `u` of it is the pool's unit `u`.
    pub units: u8,
    /// The state region, whose unit `u` is state unit `u`; `None` for a cache without a state
    /// group.
    pub states: Option<u8>,
    /// Per cache group, the host region of the group's pages: unit `u` of it is the group's host
    /// page `u`. `None` for a group whose pages have no host copy.
    pub host: Vec<Option<u8>>,
    /// The region of each sliding window's pool, in the layout's pool order after the KV pool:
    /// unit `u` of it is the pool's unit `u`.
    pub windows: Vec<u8>,
}

/// The KV cache of one rank: its planes, page tables, workspace, host tier and scratch pages.
pub struct KvCacheMemory {
    config: KvCacheConfig,
    /// The KV pool's planes.
    planes: Planes,
    /// Each sliding window's pool's planes, in the layout's pool order after the KV pool.
    windows: Vec<Planes>,
    tables: PageTables,
    /// The regions of the cache's memory, once the buffer pool numbered them
    /// ([`Self::set_regions`]).
    regions: Option<CacheRegions>,
    /// The attention workspace on the device; `None` for a cache that holds no device.
    workspace: Option<Arc<Storage>>,
    /// The host tier, once the rank's parts of the host regions are mapped
    /// ([`Self::map_host_parts`]).
    host: Option<HostTier>,
    /// The scratch pages of shadow forwards, once opened ([`Self::open_scratch`]).
    scratch: Option<ScratchPages>,
    /// The state units of a layout with a state group.
    states: Option<StateUnits>,
}

impl KvCacheMemory {
    /// A cache that holds no device: its tensors are meta tensors of the shapes and element
    /// types the cache of `config` has, its page tables are on the host alone, and its device
    /// work is refused or does nothing.
    ///
    /// # Errors
    ///
    /// A config whose layout the cache's tensors cannot state.
    pub(crate) fn meta(config: KvCacheConfig) -> Result<Self> {
        config.check()?;

        let states = match config.layout.state_group() {
            Some(_) => Some(StateUnits::meta(&config.layout, config.states.0)?),
            None => None,
        };
        let windows = config
            .layout
            .pools
            .iter()
            .skip(1)
            .zip(&config.windows)
            .map(|(pool, units)| Planes::meta(pool.clone(), *units))
            .collect::<Result<_>>()?;
        Ok(Self {
            planes: Planes::meta(config.layout.kv().clone(), config.units)?,
            windows,
            tables: PageTables::meta(
                Arc::clone(&config.layout),
                config.slots,
                config.width,
                config.pool_units(),
            )?,
            workspace: None,
            regions: None,
            host: None,
            scratch: None,
            states,
            config,
        })
    }

    /// Opens the cache of `config` on `device`: takes its KV planes and its state units from
    /// `owner` (the rank's Storage Server client, or storage local to the process), addressed by
    /// `device` and each of `peers` (`Storage::peers`), its window planes from the process,
    /// allocates its page tables and workspace, and resets what was created of the planes for
    /// this rank on `stream`, which is synchronized before returning.
    ///
    /// # Errors
    ///
    /// A config whose layout the cache's tensors cannot state, segments that do not hold the
    /// config's units, or a storage `owner` or the device refuses.
    pub fn open(
        config: KvCacheConfig,
        (device, peers): (DeviceId, &[DeviceId]),
        owner: &dyn ExclusiveStorage,
        stream: &Stream,
    ) -> Result<Self> {
        config.check()?;

        let planes = Planes::open(
            config.layout.kv().clone(),
            config.units,
            config.segments.as_deref(),
            (device, peers),
            owner,
            stream,
        )?;
        let windows = config
            .layout
            .pools
            .iter()
            .skip(1)
            .zip(&config.windows)
            .map(|(pool, units)| Planes::open_local(pool.clone(), *units, (device, peers), stream))
            .collect::<Result<_>>()?;
        let tables = PageTables::open(
            Arc::clone(&config.layout),
            config.slots,
            config.width,
            config.pool_units(),
            device,
            stream,
        )?;
        let workspace = Storage::linear(device, config.workspace_bytes as usize)?;
        let states = match config.layout.state_group() {
            Some(_) => Some(StateUnits::open(
                &config.layout,
                config.states.0,
                config.states.1.as_deref(),
                (device, peers),
                owner,
            )?),
            None => None,
        };

        Ok(Self {
            config,
            planes,
            windows,
            tables,
            workspace: Some(workspace),
            regions: None,
            host: None,
            scratch: None,
            states,
        })
    }

    /// The config the cache was opened with.
    pub(crate) fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// The cache's layout.
    pub(crate) fn layout(&self) -> &Arc<CacheLayout> {
        &self.config.layout
    }

    /// The attention group `group`'s layout.
    fn attention(&self, group: usize) -> Result<&GroupLayout> {
        self.config
            .layout
            .groups
            .get(group)
            .filter(|layout| layout.attention())
            .ok_or_else(|| Error::Invalid(format!("the cache has no attention group {group}")))
    }

    /// The draft group's layout and the columns each of its layers lies over.
    fn draft(&self) -> Result<(&GroupLayout, u32)> {
        let layout = &self.config.layout;

        layout
            .draft_group()
            .map(|group| &layout.groups[group])
            .zip(layout.draft_span())
            .ok_or_else(|| Error::Invalid("the cache keeps no draft group".into()))
    }

    /// The planes of pool `pool`: the KV pool's, or a sliding window's.
    fn pool(&self, pool: usize) -> Result<&Planes> {
        match pool {
            0 => Ok(&self.planes),
            window => self.windows.get(window - 1).ok_or_else(|| {
                Error::Invalid(format!(
                    "the cache holds {} pools, not pool {pool}",
                    self.windows.len() + 1
                ))
            }),
        }
    }

    /// The planes of attention group `group`: every owner of every field of its pool, fields in
    /// the model's order and owners ascending, each its entries `[units, block_size / ratio,
    /// heads, width]` in the field's stored type, its flags `[units, 1, 1, 1]` U8 and, for a
    /// field in the deployment's FP8 type, its scales `[units, 1, 1, 1]` F32 ([`Planes::planes`]).
    /// A page id of the group is a unit id of its pool.
    ///
    /// The tensors are over the cache's memory: they are used only while the cache lives.
    ///
    /// # Errors
    ///
    /// A group that is no attention group of the cache.
    pub fn planes(&self, group: usize) -> Result<Vec<KvPlane>> {
        let layout = self.attention(group)?;
        self.pool(layout.pool)?.planes(layout)
    }

    /// The planes of the draft group's layers, in the drafter's order, as its attention reads
    /// them: pages of `block_size / span` tokens, `[span · units, …]`, page `c · units + u`
    /// holding the `c`-th run of the draft page in unit `u`.
    ///
    /// # Errors
    ///
    /// A cache that keeps no draft group.
    pub(crate) fn draft_planes(&self) -> Result<Vec<KvPlane>> {
        let (layout, span) = self.draft()?;
        self.planes.draft_planes(layout, span)
    }

    /// The pooled page table of attention group `group`, `[slots + 1, width]` U32: row `s - 1`
    /// of request slot `s` names the unit of each of its logical pages, and the last row, the
    /// padding row, and every page a slot does not hold name unit 0. A forward gathers its
    /// batch's rows from it.
    ///
    /// # Errors
    ///
    /// A group that is no attention group of the cache.
    pub(crate) fn page_table(&self, group: usize) -> Result<Tensor> {
        self.attention(group)?;
        self.tables.table(group)
    }

    /// The pooled page table of attention group `group` as the host holds it,
    /// `[slots + 1, width]` row-major: what the device's table holds once the changes applied
    /// so far completed.
    ///
    /// # Errors
    ///
    /// A group that is no attention group of the cache.
    ///
    /// The tests' view of the device's tables: no serving path reads it, and a meta cache's
    /// tables hold no device bytes to read instead.
    #[cfg(test)]
    pub(crate) fn host_table(&self, group: usize) -> Result<&[u32]> {
        self.tables.host_table(group)
    }

    /// The attention workspace, `[workspace_bytes]` U8.
    pub(crate) fn workspace(&self) -> Result<Tensor> {
        let shape = [self.config.workspace_bytes as usize];

        match &self.workspace {
            // SAFETY: the cache owns the workspace for as long as it lives, and only the
            // forwards of the runner that holds the cache, one at a time, address it.
            Some(storage) => unsafe { storage.external(0, shape.as_slice(), DType::U8) },
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape.as_slice(), DType::U8, None) },
        }
    }

    /// Attention group `group` as a forward is lent it: `block_tables` is the batch's rows
    /// gathered from [`Self::page_table`], `[rows, width]` U32, `slot_mapping` the slots its
    /// tokens write in the group, `[tokens]` U32, and `starts` each row's first position a
    /// sliding-window group holds, `[rows]` U32, present exactly for such a group.
    ///
    /// # Errors
    ///
    /// A group that is no attention group of the cache, or starts stated for a full-history
    /// group or the reverse.
    pub(crate) fn lend(
        &self,
        group: usize,
        (block_tables, slot_mapping, starts): (Tensor, Tensor, Option<Tensor>),
    ) -> Result<KvCache> {
        let layout = self.attention(group)?;
        if layout.window() != starts.is_some() {
            return Err(Error::Invalid(format!(
                "attention group {group} is lent {} starts, and its readers read {}",
                if starts.is_some() { "its" } else { "no" },
                if layout.window() {
                    "a sliding window from each row's start"
                } else {
                    "every position"
                }
            )));
        }

        Ok(KvCache {
            block_tables,
            slot_mapping,
            starts,
            page_size: layout.block_size,
            planes: self.planes(group)?,
            workspace: self.workspace()?,
        })
    }

    /// The draft cache as a drafter's forward is lent it: `block_tables` names, per row, the
    /// drafter's pages ([`Self::draft_planes`]), which a speculative step states from the
    /// draft group's entries of the rank's device block table, and `slot_mapping` the draft
    /// slots each token writes.
    ///
    /// # Errors
    ///
    /// A cache that keeps no draft group.
    pub(crate) fn lend_draft(&self, block_tables: Tensor, slot_mapping: Tensor) -> Result<KvCache> {
        let (layout, span) = self.draft()?;

        Ok(KvCache {
            block_tables,
            slot_mapping,
            starts: None,
            page_size: layout.block_size / span,
            planes: self.draft_planes()?,
            workspace: self.workspace()?,
        })
    }

    /// The cache as the rank reports it to the engine: its layout, units, plane runs and
    /// segments.
    pub fn info(&self) -> PrefixCacheInfo {
        PrefixCacheInfo {
            plane_runs: self.planes.runs().to_vec(),
            segments: self
                .planes
                .segments()
                .map(<[_]>::to_vec)
                .unwrap_or_default(),
            held_segments: self.planes.held_segments(),
            ..self.config.layout.info(self.config.units)
        }
    }

    /// The segments the value planes are made of, in unit order; `None` for planes that are not
    /// split.
    pub fn segments(&self) -> Option<&[crate::memory::cache::info::CacheSegment]> {
        self.planes.segments()
    }

    /// The bytes the planes, the window planes and the state units map on the device; a
    /// sleeping rank's segments are not in it.
    pub(crate) fn mapped_bytes(&self) -> u64 {
        self.planes.mapped_bytes()
            + self.windows.iter().map(Planes::mapped_bytes).sum::<u64>()
            + self.states.as_ref().map_or(0, StateUnits::mapped_bytes)
    }

    /// Whether `region` is the state region, as [`Self::set_regions`] stated it.
    fn is_states(&self, region: u8) -> bool {
        self.regions
            .as_ref()
            .is_some_and(|regions| regions.states == Some(region))
    }

    /// Whether `region` is the pool's region, as [`Self::set_regions`] stated it.
    fn is_units(&self, region: u8) -> bool {
        self.regions
            .as_ref()
            .is_some_and(|regions| regions.units == region)
    }

    /// The segment table of the device region `region` as this rank holds it, with the device
    /// bytes one unit of its split memory takes, which a segment gives back per unit: the pool's
    /// value planes (`CacheLayout::split_bytes`) or the state units (`CacheLayout::state_bytes`).
    /// `None` for a region that is none of the cache's or whose memory is not split.
    pub(crate) fn segment_table(
        &self,
        region: u8,
    ) -> Option<(crate::memory::cache::info::CacheTable, u64)> {
        let (segments, held, unit_bytes) = if self.is_units(region) {
            (
                self.planes.segments()?,
                self.planes.held_segments(),
                self.planes.unit_bytes(),
            )
        } else if self.is_states(region) {
            let states = self.states.as_ref()?;
            (
                states.segments()?,
                states.held_segments(),
                states.unit_bytes(),
            )
        } else {
            return None;
        };
        let table = crate::memory::cache::info::CacheTable {
            segments: segments.to_vec(),
            held,
        };
        Some((table, unit_bytes))
    }

    /// The device bytes one unit of the pool's planes takes, which a segment gives back per unit
    /// (`CacheLayout::split_bytes`).
    pub(crate) fn unit_bytes(&self) -> u64 {
        self.planes.unit_bytes()
    }

    /// The device regions of the cache whose memory is split into segments, by number: the
    /// pool's and the state region. Empty before [`Self::set_regions`].
    pub(crate) fn split_regions(&self) -> Vec<u8> {
        let Some(regions) = &self.regions else {
            return Vec::new();
        };
        let mut split = Vec::new();
        if self.planes.segments().is_some() {
            split.push(regions.units);
        }
        if let (Some(states), Some(region)) = (&self.states, regions.states)
            && states.segments().is_some()
        {
            split.push(region);
        }
        split
    }

    /// Where the Storage Server retains every state unit's bytes, run by run: what the rank
    /// places for the state region (`storage::protocol::Request::Place`); empty for a cache
    /// without a state group or whose units are local to its process.
    pub(crate) fn state_plane_runs(&self) -> &[crate::memory::cache::info::CachePlaneRun] {
        self.states.as_ref().map_or(&[], StateUnits::runs)
    }

    /// States the regions of the cache's memory, as the buffer pool numbered them in its
    /// answer to `Open`: what [`Self::copy`] tells the runs of a copy by. They are the same
    /// for the life of the session.
    ///
    /// # Errors
    ///
    /// Regions stated for another number of cache groups than the layout's.
    pub(crate) fn set_regions(&mut self, regions: CacheRegions) -> Result<()> {
        let groups = self.config.layout.groups.len();
        if regions.host.len() != groups {
            return Err(Error::Invalid(format!(
                "host regions stated for {} cache groups of a cache of {groups}",
                regions.host.len()
            )));
        }
        if regions.windows.len() != self.windows.len() {
            return Err(Error::Invalid(format!(
                "window regions stated for {} sliding windows of a cache of {}",
                regions.windows.len(),
                self.windows.len()
            )));
        }
        let windowed = regions
            .windows
            .iter()
            .enumerate()
            .filter(|(_, window)| regions.host.contains(&Some(**window)));
        if let Some((window, region)) = windowed.into_iter().next() {
            return Err(Error::Invalid(format!(
                "region {region} is the pool of sliding window {window} and a group's host \
                 region"
            )));
        }
        if regions.states.is_some() != self.states.is_some() {
            return Err(Error::Invalid(format!(
                "a state region is {} for a cache {} a state group",
                if regions.states.is_some() {
                    "stated"
                } else {
                    "not stated"
                },
                if self.states.is_some() {
                    "with"
                } else {
                    "without"
                }
            )));
        }

        self.regions = Some(regions);
        Ok(())
    }

    /// Whether `region` is memory of the cache, as [`Self::set_regions`] stated it: its units,
    /// its state units, or a cache group's host pages. A cache whose regions are not stated
    /// holds none.
    pub(crate) fn holds(&self, region: u8) -> bool {
        self.regions.as_ref().is_some_and(|regions| {
            regions.units == region
                || regions.states == Some(region)
                || regions.host.contains(&Some(region))
                || regions.windows.contains(&region)
        })
    }

    /// Applies `changes`, the rank's device block table's changes since the last batch
    /// (`BlockTables::take_changes`), on `stream` for a cache on a device and with `None` for
    /// one that holds none: the units of each entry the buffer pool allocated since are reset
    /// (flags cleared, scales set to one, values left: a block is written before it is read),
    /// and the attention groups' pooled rows that changed are written. A change of a class
    /// past the cache groups is no cache's and is skipped, and so is the reset of a state group's
    /// entry, a unit of the state region, which a keep writes whole before a restore reads it.
    /// Every change is checked before anything is issued.
    ///
    /// The order is the one a batch needs: the forward that follows on `stream` reads the
    /// pooled rows and writes the reset units.
    ///
    /// # Errors
    ///
    /// A change the pooled tables refuse, a reset of units outside the pool or of the padding
    /// unit, or a stream stated for a cache that holds no device, or the reverse.
    pub fn apply(&mut self, changes: &[Change], stream: Option<&Stream>) -> Result<()> {
        if self.planes.device_id().is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "a cache's changes are applied on a stream exactly when it is on a device".into(),
            ));
        }

        let groups = &self.config.layout.groups;
        let mut resets = Vec::new();
        for change in changes {
            let Some((_, first, count)) = change.run.filter(|_| change.reset) else {
                continue;
            };
            let Some(group) = groups
                .get(usize::from(change.class))
                .filter(|group| !group.state())
            else {
                continue;
            };
            // A group's units are its pool's: the KV pool's, or its window's.
            let units = self.pool(group.pool)?.units();
            if first == 0 || u64::from(first) + u64::from(count) > u64::from(units) {
                return Err(Error::Invalid(format!(
                    "the block at position {} of class {} of request slot {} takes units \
                     {first}..{} of a pool of units 1..{units}",
                    change.position,
                    change.class,
                    change.slot,
                    u64::from(first) + u64::from(count),
                )));
            }
            resets.push((group.pool, first, count));
        }

        // The pooled tables check every change before they write a row.
        self.tables.apply(changes, stream)?;

        if let Some(stream) = stream {
            for (pool, first, count) in resets {
                self.pool(pool)?.reset_units(first, count, false, stream)?;
            }
        }
        Ok(())
    }

    /// Maps the cache's host tier: part `part` (the rank's index in its worker group) of each
    /// cache group's host region at the rank's host's Storage Server through `client`, whose
    /// copies run on `stream`, the rank's copy stream. `regions` names, per cache group, the host
    /// region of its pages as the session's layout states it (unit `u` of the region is page `u`
    /// of the group), or `None` for a group whose pages have no host copy; the open session must
    /// state them.
    ///
    /// # Errors
    ///
    /// A cache that holds no device, one whose host tier is mapped already, or regions stated
    /// for another number of cache groups; the server's refusal.
    pub(crate) fn map_host_parts(
        &mut self,
        client: &StorageClient,
        (regions, part): (&[Option<&str>], u32),
        stream: Stream,
    ) -> Result<()> {
        let device = self.planes.device_id().ok_or(Error::State(
            "the cache holds no device: it maps no host tier",
        ))?;
        if self.host.is_some() {
            return Err(Error::State("the cache's host tier is mapped already"));
        }

        let layout = &self.config.layout;
        if regions.len() != layout.groups.len() {
            return Err(Error::Invalid(format!(
                "host regions stated for {} cache groups of a cache of {}",
                regions.len(),
                layout.groups.len()
            )));
        }

        self.host = Some(HostTier::map(
            client,
            layout,
            device,
            (regions, part),
            stream,
        )?);
        Ok(())
    }

    /// The page identity of each cache group for weights of `fingerprint`
    /// (`PrefixCacheInfo::identities`).
    pub(crate) fn page_identities(&self, fingerprint: &str) -> Vec<storage::Key> {
        host::page_identities(&self.config.layout, fingerprint)
    }

    /// The host tier, or the refusal of a cache that maps none.
    fn host(&self) -> Result<&HostTier> {
        self.host.as_ref().ok_or_else(|| {
            Error::Invalid(
                "a copy reads or writes host pages and the cache maps no host tier".into(),
            )
        })
    }

    /// Issues `copies`, the copies of the blocks a batch runs, each from a run to a run as the
    /// buffer pool's `Copy` states them (`BlockTables::start_copy`), on `stream` for a cache on
    /// a device and with `None` for one that holds none, whose copies are checked and issue
    /// nothing. Returns the events recorded behind them on each stream they were issued on,
    /// which the batch's end waits on.
    ///
    /// The regions of the runs ([`CacheRegions`]) tell each copy:
    ///
    /// - units to units, a clone: every field of each unit is copied, in order, on `stream`
    ///   ([`Planes::copy_units`]);
    /// - a group's host page to units, a load, and units to a group's host page, a store: on
    ///   the copy stream, through the host tier; the same for the state group's host page and a
    ///   state unit, a checkpoint's load and store.
    ///
    /// A copy writes only its destination, which no table entry names until the copy is
    /// reported done, and reads a source the buffer pool holds until then: nothing orders it
    /// against the forwards of the batch. Every copy is checked before anything is issued, and
    /// when an issue fails after others were issued, the streams are waited for before the
    /// failure returns.
    ///
    /// # Errors
    ///
    /// A cache whose regions are not stated; a copy between regions that are no cache memory,
    /// of runs of different units or a host run of more than one page; units outside the pool
    /// or the padding unit, or a state unit outside the state region; a copy between state
    /// units; a host copy on a cache that maps no host tier; a stream stated for a cache that
    /// holds no device, or the reverse; the driver's refusal.
    pub(crate) fn copy(
        &self,
        copies: &[(Run, Run)],
        stream: Option<&Stream>,
    ) -> Result<Vec<Event>> {
        if copies.is_empty() {
            return Ok(Vec::new());
        }
        if self.planes.device_id().is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "a cache's copies are issued on a stream exactly when it is on a device".into(),
            ));
        }
        let regions = self.regions.as_ref().ok_or(Error::State(
            "the regions of the cache's memory are not stated: no copy names its runs",
        ))?;

        // The group whose host pages `region` holds.
        let group_of = |regions: &[Option<u8>], region: u8| {
            regions
                .iter()
                .position(|held| *held == Some(region))
                .map(|group| group as u32)
        };
        let page = |run: Run| -> Result<u64> {
            match run {
                (_, page, 1) => Ok(u64::from(page)),
                (region, first, count) => Err(Error::Invalid(format!(
                    "a host copy names {count} pages from page {first} of region {region}, and \
                     a block's host copy is one page"
                ))),
            }
        };

        // A device run is the pool's units or the state units; the state units copy only to and
        // from the state group's host pages, since a clone of a checkpoint is refused (its live
        // state is the slot's).
        let device = |region: u8| region == regions.units || regions.states == Some(region);
        let state_group = self
            .config
            .layout
            .state_group()
            .and_then(|group| u32::try_from(group).ok());
        // A page of the state group copies to and from state units alone, and every other
        // group's to and from the pool's units.
        let matches = |group: u32, region: u8| {
            (Some(group) == state_group) == (regions.states == Some(region))
        };
        let (mut units, mut loads, mut stores) = (Vec::new(), Vec::new(), Vec::new());
        for &(from, to) in copies {
            let (source, destination) = (device(from.0), device(to.0));
            if source && destination && (from.0 != regions.units || to.0 != regions.units) {
                return Err(Error::Invalid(format!(
                    "a copy from region {} to region {} copies state units, which only a host \
                     page loads and stores",
                    from.0, to.0
                )));
            }

            match (source, destination) {
                (true, true) if from.2 == to.2 => {
                    units.extend((0..from.2).map(|unit| (from.1 + unit, to.1 + unit)));
                }
                (true, true) => {
                    return Err(Error::Invalid(format!(
                        "a copy of {} units into {}",
                        from.2, to.2
                    )));
                }
                (false, true) => {
                    let block = to.1..to.1 + to.2;
                    if let Some(group) = group_of(&regions.host, from.0) {
                        if !matches(group, to.0) {
                            return Err(Error::Invalid(format!(
                                "a load of cache group {group}'s page writes region {}, which \
                                 holds no page of the group",
                                to.0
                            )));
                        }
                        loads.push((group, page(from)?, block));
                    } else {
                        return Err(Error::Invalid(format!(
                            "a copy into the cache's units reads region {}, which is none of \
                             its host pages",
                            from.0
                        )));
                    }
                }
                (true, false) => {
                    let group = group_of(&regions.host, to.0).ok_or_else(|| {
                        Error::Invalid(format!(
                            "a copy of the cache's units writes region {}, which is none of its \
                             host pages",
                            to.0
                        ))
                    })?;
                    if !matches(group, from.0) {
                        return Err(Error::Invalid(format!(
                            "a store of cache group {group}'s page reads region {}, which holds \
                             no page of the group",
                            from.0
                        )));
                    }
                    stores.push((group, page(to)?, from.1..from.1 + from.2));
                }
                (false, false) => {
                    return Err(Error::Invalid(format!(
                        "a copy from region {} to region {} touches none of the cache's units",
                        from.0, to.0
                    )));
                }
            }
        }

        // Every unit is checked before anything is issued: a page of the state group names state
        // units, every other a block of the pool, which never holds the padding unit.
        self.planes.check_copies(&units)?;
        let state = state_group;
        let paged: Vec<&HostPage> = loads.iter().chain(&stores).collect();
        for (group, _, block) in &paged {
            if Some(*group) == state {
                if block.len() != 1 || block.end > self.state_units() {
                    return Err(Error::Invalid(format!(
                        "a checkpoint's host copy names state units {}..{} of a state region of \
                         units 0..{}, and a checkpoint is one unit",
                        block.start,
                        block.end,
                        self.state_units()
                    )));
                }
            } else if block.start == 0 || block.end > self.config.units {
                return Err(Error::Invalid(format!(
                    "a host copy names units {}..{} of a pool of units 1..{}",
                    block.start, block.end, self.config.units
                )));
            }
        }
        let host = if paged.is_empty() {
            None
        } else {
            Some(self.host()?)
        };

        let Some(stream) = stream else {
            return Ok(Vec::new());
        };
        let mut events = Vec::with_capacity(2);
        if !units.is_empty() {
            let cloned = Event::new(platform::EventFlags::default());
            let issued = (|| -> Result<()> {
                self.planes.copy_units(&units, stream)?;
                cloned.record(stream)?;
                Ok(())
            })();
            if let Err(error) = issued {
                // The clones issued before the failure end before it is reported.
                if let Err(waited) = stream.synchronize() {
                    tracing::error!(%waited, "the batch's stream failed after a copy failed");
                }
                return Err(error);
            }
            events.push(cloned);
        }

        if let Some(host) = host {
            match host.copy(
                &self.config.layout,
                (&self.planes, self.states.as_ref()),
                &loads,
                &stores,
            ) {
                Ok(copied) => events.extend(copied),
                Err(error) => {
                    // The host copies settled their own stream; the clones end too before the
                    // failure is reported.
                    for event in &events {
                        if let Err(waited) = event.synchronize() {
                            tracing::error!(%waited, "a clone failed after a copy failed");
                        }
                    }
                    return Err(error);
                }
            }
        }
        Ok(events)
    }

    /// Creates the segments of the device region `region` given back again at their addresses,
    /// the first `limit` of them or all, through `owner`: the pool's value planes, whose units
    /// are reset on `stream`, which is synchronized before returning, or the state units. A wake
    /// of a region that gave nothing back, whose memory is not split or that is none of the
    /// cache's, does nothing.
    ///
    /// # Errors
    ///
    /// When `owner` or the stream fails; the region's memory is then as asleep as it was.
    pub(crate) fn wake(
        &mut self,
        region: u8,
        owner: &dyn ExclusiveStorage,
        stream: &Stream,
        limit: Option<u32>,
    ) -> Result<()> {
        if self.is_units(region) && self.planes.segments().is_some() {
            return self.planes.restore_segments(owner, stream, limit);
        }
        if self.is_states(region)
            && let Some(states) = self
                .states
                .as_mut()
                .filter(|states| states.segments().is_some())
        {
            return states.restore_segments(owner, limit);
        }
        Ok(())
    }

    /// Gives back the segments of the device region `region` from `first` on, through `owner`:
    /// the pool's value planes or the state units. The buffer pool withdrew their units and no
    /// device work addresses them.
    ///
    /// # Errors
    ///
    /// A region that is none of the cache's or whose memory is not split, or when `owner` fails.
    pub(crate) fn release_segments(
        &mut self,
        region: u8,
        first: u32,
        owner: &dyn ExclusiveStorage,
    ) -> Result<()> {
        if self.is_units(region) {
            return self.planes.release_segments(first, owner);
        }
        let states = self.is_states(region);
        match self.states.as_mut().filter(|_| states) {
            Some(states) => states.release_segments(first, owner),
            None => Err(Error::Invalid(format!(
                "region {region} is none of the cache's device regions"
            ))),
        }
    }

    /// Opens the scratch page set of shadow forwards, `pages` pages, whose table rows are as
    /// wide as the set or the cache's pooled tables, whichever is narrower: taken from the
    /// scratch pool of `allocator` and prepared on `stream` for a cache on a device
    /// ([`scratch::bytes`] of them), and with `None` a set of meta tensors for a cache that
    /// holds none.
    ///
    /// # Errors
    ///
    /// A set opened twice, a set of no pages, a device stated for a cache that holds none or
    /// the reverse, or a pool that cannot hold the set.
    pub(crate) fn open_scratch(
        &mut self,
        pages: u32,
        device: Option<(&mut platform::Allocator, &Stream)>,
    ) -> Result<()> {
        if self.scratch.is_some() {
            return Err(Error::State("the cache's scratch pages are open already"));
        }
        if self.planes.device_id().is_some() != device.is_some() {
            return Err(Error::Invalid(
                "scratch pages are taken from a device's pool exactly when the cache is on one"
                    .into(),
            ));
        }

        let layout = Arc::clone(&self.config.layout);
        let width = self.config.width;
        self.scratch = Some(match device {
            Some((allocator, stream)) => {
                ScratchPages::open(layout, pages, width, allocator, stream)?
            }
            None => ScratchPages::meta(layout, pages, width)?,
        });
        Ok(())
    }

    /// The scratch page set, or the refusal of a cache that opened none.
    fn scratch(&self) -> Result<&ScratchPages> {
        self.scratch.as_ref().ok_or(Error::State(
            "the cache's scratch pages are not open: a shadow forward has no pages to write",
        ))
    }

    /// The pages the scratch set holds, which a shadow forward's rows share; zero before it is
    /// opened.
    pub(crate) fn scratch_capacity(&self) -> u32 {
        self.scratch.as_ref().map_or(0, ScratchPages::pages)
    }

    /// The planes of attention group `group`'s layers over the scratch pages, in the shape
    /// [`Self::planes`] has with the set's pages in place of the pool's units.
    ///
    /// # Errors
    ///
    /// A cache whose scratch pages are not open, or a group that is no attention group.
    pub(crate) fn scratch_planes(&self, group: usize) -> Result<Vec<KvPlane>> {
        let layout = self.attention(group)?;
        self.scratch()?.planes(layout)
    }

    /// The scratch set's page table, `[pages, width]` U32, for every attention group: row `i`
    /// names the set's pages from page `i` on, so a row of a shadow forward whose pages begin at
    /// page `i` gathers row `i`.
    ///
    /// # Errors
    ///
    /// A cache whose scratch pages are not open.
    pub(crate) fn scratch_table(&self) -> Result<Tensor> {
        self.scratch()?.table()
    }

    /// Copies what a batch's rows have cached in attention group `group` into their scratch
    /// pages on `stream`, before a shadow forward: `slots` are the rows' request slots
    /// (`nn::NO_SLOT` for a padding row), `lengths` the keys each row has cached and `first`
    /// the first scratch page the set hands each row. A row of no keys, or a padding row, copies
    /// nothing.
    ///
    /// # Errors
    ///
    /// A cache that holds no device or whose scratch pages are not open; a row whose slot
    /// holds no entry of the group, whose keys need more pages than its slot's table names, or
    /// whose pages lie past the set's.
    pub(crate) fn copy_history(
        &self,
        group: usize,
        slots: &[u32],
        lengths: &[u32],
        first: &[u32],
        stream: &Stream,
    ) -> Result<()> {
        let layout = self.attention(group)?;
        let scratch = self.scratch()?;
        if slots.len() != lengths.len() || slots.len() != first.len() {
            return Err(Error::Invalid(format!(
                "a shadow forward names {} slots, {} lengths and {} first pages",
                slots.len(),
                lengths.len(),
                first.len()
            )));
        }

        let table = self.tables.host_table(group)?;
        let width = self.tables.width() as usize;
        let mut history = Vec::with_capacity(slots.len());
        for ((slot, length), first) in slots.iter().zip(lengths).zip(first) {
            if *slot == nn::NO_SLOT || *length == 0 {
                continue;
            }
            let held = length.div_ceil(layout.block_size) as usize;
            if held > width {
                return Err(Error::Invalid(format!(
                    "a row of {length} keys needs {held} pages of a table of {width}"
                )));
            }
            let row = self.tables.row(*slot)? as usize;
            let units = &table[row * width..row * width + held];
            if units.iter().all(|unit| *unit == 0) {
                return Err(Error::Invalid(format!(
                    "request slot {slot} holds no entry of cache group {group}"
                )));
            }
            history.push((*first, units.to_vec()));
        }

        scratch.copy_history(&self.planes, &history, stream)
    }

    /// The state units as `[units · checkpoint_planes, plane_bytes]` U8, plane `p` of unit `u`
    /// at row `u · checkpoint_planes + p` ([`Self::state_rows`]): what the copies of recurrent
    /// state gather from and scatter into.
    ///
    /// # Errors
    ///
    /// A cache without a state group.
    pub(crate) fn state_tensor(&self) -> Result<Tensor> {
        self.states()?.tensor()
    }

    /// The rows of [`Self::state_tensor`] state unit `unit`'s planes lie in, in plane order.
    ///
    /// # Errors
    ///
    /// A cache without a state group, or a unit outside the state region.
    pub(crate) fn state_rows(&self, unit: u32) -> Result<std::ops::Range<i64>> {
        self.states()?.rows(unit)
    }

    /// The units of the state region; zero for a cache without a state group.
    pub(crate) fn state_units(&self) -> u32 {
        self.states.as_ref().map_or(0, StateUnits::units)
    }

    /// The state units, or the refusal of a cache without a state group.
    fn states(&self) -> Result<&StateUnits> {
        self.states
            .as_ref()
            .ok_or_else(|| Error::Invalid("the cache keeps no state group".into()))
    }
}

#[cfg(test)]
mod tests {
    use nn::family::{CacheField, History, KEY, StateField, StateShape, VALUE};
    use proto::request::PrefixCacheDtype;

    use super::*;

    /// The regions the tests number the cache's memory by: its units, its state units, the
    /// host pages of each of up to three groups, and a sliding window's units.
    const UNITS: u8 = 0;
    const STATES: u8 = 1;
    const HOST: [u8; 3] = [4, 5, 6];
    const WINDOW: u8 = 7;

    /// A model of two attention layers of 4-token pages of one 4-wide head, a recurrent state
    /// of two layers and a drafter of one layer, over sixteen units, a state region of eight
    /// checkpoints, two request slots and rows of four pages.
    fn config() -> KvCacheConfig {
        let state = StateShape {
            layers: 2,
            fields: vec![StateField {
                name: "state",
                dtype: DType::U8,
                dims: vec![40],
                stepwise: false,
                rows: nn::family::StateRows::Whole,
            }],
        };
        let layout = CacheLayout::of(
            4,
            &CacheField::keys_and_values(2, 1, 4),
            PrefixCacheDtype::Float8E4m3Fn,
        )
        .and_then(|layout| layout.with_state(&[state]))
        .and_then(|layout| layout.with_draft(&CacheField::keys_and_values(1, 1, 4)))
        .expect("a hybrid layout with a draft group");

        KvCacheConfig {
            layout: Arc::new(layout),
            units: 16,
            slots: 2,
            width: 4,
            segments: None,
            workspace_bytes: 1 << 10,
            states: (8, None),
            windows: Vec::new(),
        }
    }

    /// A model of two attention groups of two layers each: full attention over layers 0 and 1,
    /// and a sliding window of eight tokens over layers 2 and 3, whose pool holds six units.
    fn two_groups() -> KvCacheConfig {
        let window = |name| CacheField {
            history: History::Window { tokens: 8 },
            ..CacheField::attention(name, 4, 1, 4)
        };
        let mut fields = CacheField::keys_and_values(2, 1, 4);
        for name in [KEY, VALUE] {
            let mut field = window(name);
            field.owners.drain(..2);
            fields.push(field);
        }
        let layout = CacheLayout::of(4, &fields, PrefixCacheDtype::BFloat16)
            .expect("two groups of two layers");

        KvCacheConfig {
            layout: Arc::new(layout),
            units: 16,
            slots: 2,
            width: 4,
            segments: None,
            workspace_bytes: 1 << 10,
            states: (0, None),
            windows: vec![6],
        }
    }

    /// The changes that give `slot` the blocks of `units` in `class` from position 0, one unit
    /// each, allocated and so reset.
    fn allocated(slot: u32, class: u8, units: &[u32]) -> Vec<Change> {
        units
            .iter()
            .enumerate()
            .map(|(position, unit)| Change {
                slot,
                class,
                position: position as u32,
                run: Some((UNITS, *unit, 1)),
                reset: true,
            })
            .collect()
    }

    /// The regions of a cache of `groups` cache groups: the hybrid layout of [`config`] for
    /// three, with its state region and host pages of every group, and [`two_groups`] for two,
    /// whose window has its own region and no host copy.
    fn regions(groups: usize) -> CacheRegions {
        match groups {
            3 => CacheRegions {
                units: UNITS,
                states: Some(STATES),
                host: HOST.iter().copied().map(Some).collect(),
                windows: Vec::new(),
            },
            _ => CacheRegions {
                units: UNITS,
                states: None,
                host: vec![Some(HOST[0]), None],
                windows: vec![WINDOW],
            },
        }
    }

    /// A cache that holds no device lends meta tensors of the shapes and element types the
    /// cache on a device has.
    #[test]
    fn a_cache_with_no_device_lends_tensors_of_its_shapes() -> Result<()> {
        let cache = KvCacheMemory::meta(config())?;
        assert_eq!(
            cache.workspace()?.device().location(),
            ops::meta().location()
        );

        // The attention group: the keys and the values of each layer, a column each, FP8
        // with a scale and a flag per unit.
        let planes = cache.planes(0)?;
        assert_eq!(planes.len(), 4);
        for plane in &planes {
            assert_eq!(plane.entries.dims(), [16, 4, 1, 4]);
            assert_eq!(plane.entries.dtype(), DType::F8E4M3);
            assert_eq!(plane.initialized.dims(), [16, 1, 1, 1]);
            assert_eq!(plane.initialized.dtype(), DType::U8);
            let scale = plane.scale.as_ref().expect("an FP8 cache states scales");
            assert_eq!(
                (scale.dims(), scale.dtype()),
                (&[16, 1, 1, 1][..], DType::F32)
            );
        }

        // The draft group: one layer over the two columns of a unit, pages of an attention
        // page's tokens.
        let draft = cache.draft_planes()?;
        assert_eq!(draft.len(), 2);
        assert_eq!(draft[0].entries.dims(), [2 * 16, 4, 1, 4]);
        assert_eq!(draft[1].initialized.dims(), [2 * 16, 1, 1, 1]);

        // The pooled table and the workspace.
        let table = cache.page_table(0)?;
        assert_eq!((table.dims(), table.dtype()), (&[3, 4][..], DType::U32));
        let workspace = cache.workspace()?;
        assert_eq!(
            (workspace.dims(), workspace.dtype()),
            (&[1 << 10][..], DType::U8)
        );

        // The draft and state groups have no pooled table and no attention planes.
        assert!(cache.page_table(1).is_err());
        assert!(cache.planes(2).is_err());

        // What a forward is lent: a full-history group reads no start.
        let slots = || Tensor::zeros(2, DType::U32, &tensor::Device::Cpu).unwrap();
        let lent = cache.lend(0, (cache.page_table(0)?, slots(), None))?;
        assert_eq!((lent.page_size, lent.planes.len()), (4, 4));
        assert!(
            cache
                .lend(0, (cache.page_table(0)?, slots(), Some(slots())))
                .is_err()
        );
        assert_eq!(
            cache.lend_draft(cache.page_table(0)?, slots())?.page_size,
            4
        );
        Ok(())
    }

    /// A sliding window's group lends the planes of its own pool, whose units the window
    /// sizes, over its own pooled table, with each row's start.
    #[test]
    fn a_window_group_lends_its_own_pool() -> Result<()> {
        let cache = KvCacheMemory::meta(two_groups())?;

        let planes = cache.planes(1)?;
        let owners: Vec<(&str, u32)> = planes
            .iter()
            .map(|plane| (plane.field, plane.layer))
            .collect();
        assert_eq!(owners, [(KEY, 2), (KEY, 3), (VALUE, 2), (VALUE, 3)]);
        assert!(
            planes
                .iter()
                .all(|plane| plane.entries.dims() == [6, 4, 1, 4])
        );
        assert_eq!(cache.planes(0)?[0].entries.dims(), [16, 4, 1, 4]);

        let column = || Tensor::zeros(2, DType::U32, &tensor::Device::Cpu).unwrap();
        let lent = cache.lend(1, (cache.page_table(1)?, column(), Some(column())))?;
        assert!(lent.starts.is_some());
        assert!(
            cache
                .lend(1, (cache.page_table(1)?, column(), None))
                .is_err()
        );

        // A window of eight tokens over pages of four: four pages a request beside the
        // watermark, and the pages of a step's tokens.
        assert_eq!(window_units(8, 4, 2, 16), 1 + 2 * (2 + 2 + 1) + 4);
        Ok(())
    }

    /// A model of two attention groups has a pooled table and planes per group.
    #[test]
    fn each_attention_group_has_its_planes_and_its_table() -> Result<()> {
        let mut cache = KvCacheMemory::meta(two_groups())?;

        // The keys and the values of each group's two layers.
        for group in 0..2 {
            assert_eq!(cache.planes(group)?.len(), 4);
            assert_eq!(cache.page_table(group)?.dims(), [3, 4]);
        }
        assert!(cache.draft_planes().is_err());

        // The groups' rows are apart: slot 1 holds other units in each, the window's units of
        // its own pool, which a unit past it is not.
        let window = |units: &[u32]| -> Vec<Change> {
            allocated(1, 1, units)
                .into_iter()
                .map(|change| Change {
                    run: change.run.map(|(_, unit, count)| (WINDOW, unit, count)),
                    ..change
                })
                .collect()
        };
        let changes = [allocated(1, 0, &[3, 4]), window(&[5])].concat();
        cache.apply(&changes, None)?;
        assert_eq!(cache.host_table(0)?[..4], [3, 4, 0, 0]);
        assert_eq!(cache.host_table(1)?[..4], [5, 0, 0, 0]);
        let past = cache
            .apply(&window(&[9]), None)
            .expect_err("a unit past the window's pool");
        assert!(past.to_string().contains("units 1..6"), "{past}");
        Ok(())
    }

    /// The pooled table's last row names only the padding unit, whatever the slots hold, and
    /// slot `s`'s entries are row `s - 1`, which its slot gathers.
    #[test]
    fn the_padding_row_names_only_the_padding_unit() -> Result<()> {
        let mut cache = KvCacheMemory::meta(config())?;

        let changes = [allocated(1, 0, &[5, 6, 7]), allocated(2, 0, &[8])].concat();
        cache.apply(&changes, None)?;
        // A slot outside the two the cache holds names no row.
        assert!(cache.apply(&allocated(3, 0, &[9]), None).is_err());

        let rows = cache.host_table(0)?;
        let row = |row: usize| &rows[row * 4..(row + 1) * 4];
        assert_eq!(row(0), [5, 6, 7, 0]);
        assert_eq!(row(1), [8, 0, 0, 0]);
        assert_eq!(row(2), [0, 0, 0, 0]);
        assert!(rows.iter().all(|unit| *unit != nn::NO_SLOT));

        // Entries that left name the padding unit again.
        let left: Vec<Change> = (0..3)
            .map(|position| Change {
                slot: 1,
                class: 0,
                position,
                run: None,
                reset: false,
            })
            .collect();
        cache.apply(&left, None)?;
        assert_eq!(cache.host_table(0)?[..4], [0, 0, 0, 0]);
        Ok(())
    }

    /// A reset is checked before anything is applied: an allocated block of the padding unit
    /// or past the pool changes nothing. A change of a class past the cache groups is none of
    /// the cache's.
    #[test]
    fn a_reset_of_units_outside_the_pool_is_refused() -> Result<()> {
        let mut cache = KvCacheMemory::meta(two_groups())?;
        let reset = |unit: u32, count: u32| Change {
            slot: 1,
            class: 0,
            position: 0,
            run: Some((UNITS, unit, count)),
            reset: true,
        };

        for refused in [reset(0, 1), reset(16, 1), reset(15, 2)] {
            let message = cache
                .apply(&[refused], None)
                .expect_err("the reset is refused")
                .to_string();
            assert!(message.contains("units 1..16"), "{message}");
        }
        assert!(cache.host_table(0)?.iter().all(|unit| *unit == 0));

        let product = Change {
            class: 9,
            ..reset(40, 3)
        };
        cache.apply(&[product], None)?;
        Ok(())
    }

    /// A copy is told by the regions of its runs, and refused whole, naming its regions or
    /// units and the reason, when one of them is no copy of the cache's memory.
    #[test]
    fn a_copy_is_refused_by_its_regions_and_units() -> Result<()> {
        let mut cache = KvCacheMemory::meta(config())?;
        let clone = ((UNITS, 3, 1), (UNITS, 4, 1));

        // Runs mean nothing before the regions are stated, which name a state region exactly
        // for a cache with a state group.
        let unstated = cache.copy(&[clone], None).err().expect("no regions");
        assert!(matches!(unstated, Error::State(_)), "{unstated}");
        assert!(cache.set_regions(regions(2)).is_err());
        let stateless = CacheRegions {
            states: None,
            ..regions(3)
        };
        assert!(cache.set_regions(stateless).is_err());
        cache.set_regions(regions(3))?;

        let refused = |copy: (Run, Run)| {
            cache
                .copy(&[clone, copy], None)
                .err()
                .expect("the copy is refused")
                .to_string()
        };
        assert!(refused(((UNITS, 3, 1), (UNITS, 0, 1))).contains("padding unit"));
        assert!(refused(((UNITS, 3, 1), (UNITS, 3, 1))).contains("the destination is the source"));
        assert!(refused(((UNITS, 3, 1), (UNITS, 16, 1))).contains("outside the pool"));
        assert!(refused(((UNITS, 3, 2), (UNITS, 6, 1))).contains("2 units into 1"));
        assert!(refused(((9, 3, 1), (UNITS, 6, 1))).contains("region 9"));
        assert!(refused(((UNITS, 3, 1), (9, 6, 1))).contains("region 9"));
        assert!(refused(((HOST[0], 3, 1), (HOST[1], 6, 1))).contains("none of the cache's"));
        assert!(refused(((HOST[0], 3, 2), (UNITS, 6, 1))).contains("one page"));

        // A clone moves blocks of the pool of any unit count; a checkpoint is never cloned, and a
        // checkpoint's host page is one state unit, and no unit of the pool.
        cache.copy(&[clone, ((UNITS, 5, 2), (UNITS, 8, 2))], None)?;
        assert!(refused(((STATES, 3, 1), (STATES, 4, 1))).contains("copies state units"));
        assert!(refused(((UNITS, 3, 1), (STATES, 4, 1))).contains("copies state units"));
        assert!(refused(((HOST[2], 0, 1), (UNITS, 6, 1))).contains("holds no page of the group"));
        assert!(refused(((HOST[0], 0, 1), (STATES, 6, 1))).contains("holds no page of the group"));
        assert!(refused(((STATES, 6, 1), (HOST[0], 0, 1))).contains("holds no page of the group"));
        assert!(refused(((HOST[2], 0, 1), (STATES, 6, 2))).contains("one unit"));
        assert!(refused(((HOST[2], 0, 1), (STATES, 8, 1))).contains("units 0..8"));

        // A copy through the host tier needs one, the first checkpoint's included.
        let load = cache
            .copy(&[((HOST[2], 0, 1), (STATES, 0, 1))], None)
            .err()
            .expect("no host tier");
        assert!(load.to_string().contains("no host tier"), "{load}");
        Ok(())
    }

    /// What needs a device is refused by a cache that holds none, and what a profile run reads
    /// is not.
    #[test]
    fn a_cache_with_no_device_refuses_device_work() -> Result<()> {
        let mut cache = KvCacheMemory::meta(config())?;
        let local = crate::memory::exclusive::LocalStorage;

        assert_eq!(cache.mapped_bytes(), 0);
        assert!(cache.segments().is_none());
        assert!(cache.release_segments(UNITS, 0, &local).is_err());

        cache.set_regions(regions(3))?;
        assert!(cache.copy(&[], None)?.is_empty());
        let refused = cache
            .copy(&[((UNITS, 1, 1), (HOST[0], 0, 1))], None)
            .err()
            .expect("no host tier is mapped");
        assert!(refused.to_string().contains("no host tier"), "{refused}");

        // The report states the layout and no storage.
        let info = cache.info();
        assert_eq!((info.num_units, info.groups.len()), (16, 3));
        assert!(info.plane_runs.is_empty() && info.segments.is_empty());
        Ok(())
    }

    /// The scratch page set of a cache with no device is a set of meta tensors in the cache's
    /// page shape; a cache with a sliding window keeps none.
    #[test]
    fn scratch_pages_have_the_caches_page_shape() -> Result<()> {
        let mut windowed = KvCacheMemory::meta(two_groups())?;
        assert!(windowed.open_scratch(12, None).is_err());

        let mut config = config();
        config.layout = Arc::new(CacheLayout::of(
            4,
            &CacheField::keys_and_values(2, 1, 4),
            PrefixCacheDtype::BFloat16,
        )?);
        config.states = (0, None);
        let mut cache = KvCacheMemory::meta(config)?;
        assert_eq!(cache.scratch_capacity(), 0);
        assert!(cache.scratch_planes(0).is_err());

        cache.open_scratch(12, None)?;
        assert_eq!(cache.scratch_capacity(), 12);
        assert!(cache.open_scratch(12, None).is_err());

        let planes = cache.scratch_planes(0)?;
        assert_eq!(planes.len(), 4);
        assert_eq!(planes[0].entries.dims(), [12, 4, 1, 4]);

        // A table row for every page of the set, as wide as the set or the pooled tables.
        let width = cache.config().width.min(12) as usize;
        assert_eq!(cache.scratch_table()?.dims(), [12, width]);
        Ok(())
    }

    /// A runner is built on one thread and moved to the worker thread with the cache and the
    /// recurrent state it holds.
    #[test]
    fn the_cache_and_the_state_move_to_the_worker_thread() {
        fn moves<T: Send>() {}

        moves::<KvCacheMemory>();
        moves::<crate::memory::state::StateMemory>();
    }

    /// A layout the cache's tensors cannot state is refused when the cache is made.
    #[test]
    fn a_layout_the_tensors_cannot_state_is_refused() {
        // An attention page of two units.
        let mut config = two_groups();
        let mut layout = (*config.layout).clone();
        layout.groups[0].units_per_page = 2;
        config.layout = Arc::new(layout);
        let refused = KvCacheMemory::meta(config)
            .err()
            .expect("a page of two units");
        assert!(matches!(refused, Error::Unsupported(_)), "{refused}");

        // A pool of the padding unit alone.
        let mut config = two_groups();
        config.units = 1;
        assert!(KvCacheMemory::meta(config).is_err());

        // No request slot.
        let mut config = two_groups();
        config.slots = 0;
        assert!(KvCacheMemory::meta(config).is_err());

        // No attention workspace.
        let mut config = two_groups();
        config.workspace_bytes = 0;
        assert!(KvCacheMemory::meta(config).is_err());
    }

    /// The fixed bytes are the tables, the workspace and the window's pool: six units of the
    /// keys and values of two layers, a 32-byte plane and a flag each.
    #[test]
    fn the_fixed_bytes_are_the_tables_the_workspace_and_the_windows() {
        let config = two_groups();
        assert_eq!(
            config.fixed_bytes(),
            2 * 3 * 4 * 4 + (1 << 10) + 6 * 2 * 2 * (32 + 1)
        );
    }

    /// A cache on a device takes its planes from the owner, lends tensors on its GPU, copies a
    /// unit's bytes into the unit a clone names, mirrors a slot's row on the device and resets
    /// the units the buffer pool allocated: their flags read as empty.
    #[test]
    #[ignore = "requires a GPU"]
    fn a_cache_on_a_device_lends_its_planes_and_resets_what_it_is_given() -> Result<()> {
        let device = DeviceId(0);
        let stream = Stream::new(device)?;
        let mut cache = KvCacheMemory::open(
            two_groups(),
            (device, &[]),
            &crate::memory::exclusive::LocalStorage,
            &stream,
        )?;
        assert_ne!(
            cache.workspace()?.device().location(),
            ops::meta().location()
        );
        assert!(cache.mapped_bytes() > 0);
        assert_eq!(cache.planes(0)?[1].entries.dims(), [16, 4, 1, 4]);

        cache.set_regions(regions(2))?;
        cache.apply(&allocated(1, 0, &[3, 4]), Some(&stream))?;

        // A clone moves a unit's bytes into the unit its copy names. A column's planes lie unit
        // after unit, so the key plane of unit 1 in column 0 is the layout's plane bytes (four
        // tokens of one 4-wide BF16 head, 32 bytes) and the next unit's plane follows it.
        let plane = cache.layout().plane_bytes() as usize;
        let pattern: Vec<u8> = (0..plane as u8).collect();
        let (source, destination) = (
            cache.planes.plane_address(0, 0, 1)?,
            cache.planes.plane_address(0, 0, 5)?,
        );
        // SAFETY: the key plane of unit 1 in column 0, which the cache owns and nothing else
        // writes; the host pattern is read before the stream is synchronized.
        unsafe { platform::memcpy_async(source, pattern.as_ptr() as u64, plane, &stream)? };
        stream.synchronize()?;
        for cloned in cache.copy(&[((UNITS, 1, 1), (UNITS, 5, 1))], Some(&stream))? {
            cloned.synchronize()?;
        }
        let mut moved = vec![0_u8; plane];
        // SAFETY: as above, the key plane of unit 5, read behind the copy; the host buffer
        // holds it.
        unsafe { platform::memcpy_async(moved.as_mut_ptr() as u64, destination, plane, &stream)? };
        stream.synchronize()?;
        assert_eq!(moved, pattern);

        // The device row is the host mirror's.
        let table = cache.page_table(0)?;
        let mut device_row = [0_u32; 4];
        // SAFETY: the row is the first four entries of the pooled table, which the cache owns
        // and nothing writes after the synchronized change; the host array holds four.
        unsafe {
            platform::memcpy_async(
                device_row.as_mut_ptr() as u64,
                ops::operand(&table)?.ptr.address(),
                16,
                &stream,
            )?;
        }
        stream.synchronize()?;
        assert_eq!(device_row, cache.host_table(0)?[..4]);

        // A unit a forward initialized is reset when the buffer pool gives it again.
        let flags = cache.planes(0)?[0].initialized.clone();
        let unit = |unit: u64| -> Result<u8> {
            let mut flag = [0_u8; 1];
            // SAFETY: the flag of `unit` in column 0 is one byte of the cache's flag storage,
            // read behind every write on the stream; the host array holds one.
            unsafe {
                platform::memcpy_async(
                    flag.as_mut_ptr() as u64,
                    ops::operand(&flags)?.ptr.address() + unit,
                    1,
                    &stream,
                )?;
            }
            stream.synchronize()?;
            Ok(flag[0])
        };
        cache.planes.mark_initialized(&[6], &stream)?;
        assert_eq!(unit(6)?, 1);
        cache.apply(&allocated(2, 0, &[6]), Some(&stream))?;
        assert_eq!(unit(6)?, 0);
        Ok(())
    }

    /// A cache whose planes split gives its segments back in a sleep and creates them again in
    /// a wake, at the same addresses.
    #[test]
    #[ignore = "requires a GPU"]
    fn a_sleep_gives_the_segments_back_and_a_wake_restores_them() -> Result<()> {
        let device = DeviceId(0);
        let stream = Stream::new(device)?;
        let local = crate::memory::exclusive::LocalStorage;

        let layout = Arc::new(CacheLayout::of(
            64,
            &CacheField::keys_and_values(2, 2, 256),
            PrefixCacheDtype::BFloat16,
        )?);
        let granularity = platform::DeviceAllocation::granularity(device)? as u64;
        let units = segments::segmented_units(4096, layout.segment_plane(), granularity);
        let config = KvCacheConfig {
            segments: segments::own(1, units, layout.segment_plane(), granularity),
            layout,
            units,
            slots: 2,
            width: 4,
            workspace_bytes: 1 << 20,
            states: (0, None),
            windows: Vec::new(),
        };
        let mut cache = KvCacheMemory::open(config, (device, &[]), &local, &stream)?;
        cache.set_regions(CacheRegions {
            units: UNITS,
            states: None,
            host: vec![None],
            windows: Vec::new(),
        })?;
        let held_of = |cache: &KvCacheMemory| {
            cache
                .segment_table(UNITS)
                .map_or(0, |(table, _)| table.held)
        };
        let held = held_of(&cache);
        let mapped = cache.mapped_bytes();
        assert!(held > 0);

        // A sleep gives back every segment.
        cache.release_segments(UNITS, 0, &local)?;
        assert_eq!(held_of(&cache), 0);
        assert!(cache.mapped_bytes() < mapped);

        cache.wake(UNITS, &local, &stream, Some(1))?;
        assert_eq!(held_of(&cache), 1);
        cache.wake(UNITS, &local, &stream, None)?;
        assert_eq!((held_of(&cache), cache.mapped_bytes()), (held, mapped));
        Ok(())
    }
}

//! The rank's KV cache: the memory a causal forward reads and writes its keys and values in.
//!
//! "KV cache" is the reference engines' word for it (vLLM's `KVCacheManager`, SGLang's token to
//! KV pool); the scheduler's side allocates its pages and indexes them by prefix, and this
//! module is the rank's side. [`KvCacheMemory`] owns:
//!
//! - the **planes** ([`planes`]): the unit pool's exclusive storage at the host's Storage
//!   Server, the memory of the cache's buffer pool region, with each plane's initialization
//!   flag and, for an FP8 cache, its scale, and the index keys of a model whose attention reads
//!   the tokens an indexer selects;
//! - the **pooled page tables** ([`tables`]): one per attention group, the device copy of the
//!   group's entries of the rank's device block table, which a forward gathers its rows from;
//! - the **attention workspace**, which the paged-attention kernel uses during a forward;
//! - the **host tier** ([`host`]): the rank's parts of the host regions of the cache groups'
//!   pages, which pages are loaded from and stored to on the rank's copy stream, with the
//!   converted loads ([`conversion`]) of a cache that admits them;
//! - the **scratch pages** ([`scratch`]) a shadow forward writes;
//! - for a model that keeps a recurrent state, the **state units** ([`states`]): the memory of
//!   the scheduler's state region, one checkpoint a unit, which the state copies and the host
//!   tier read and write.
//!
//! A model reads and writes none of this directly: the runner that holds the cache lends it for
//! a forward as `nn::KvPlanes` and page-table tensors, external tensors over this memory.
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
//! whose runs name state units; every other cache group's from the pool.
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

pub(crate) mod conversion;
pub(crate) mod host;
pub(crate) mod info;
pub(crate) mod layout;
mod planes;
pub(crate) mod scratch;
pub mod segments;
mod states;
mod tables;

use std::collections::BTreeSet;
use std::sync::Arc;

use self::info::PrefixCacheInfo;
use nn::{KvCache, KvPlanes};
use platform::{DeviceId, Event, Stream};
use storage::runtime::Run;
use tensor::{DType, Tensor};

use self::conversion::Conversion;
use self::host::{HostPage, HostTier};
pub use self::layout::{CacheLayout, Checkpoint, GroupLayout};
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
}

impl KvCacheConfig {
    /// The device bytes the cache holds beside its planes: the pooled page tables, the
    /// attention workspace, and the scratch and unit table of its converted loads.
    pub(crate) fn fixed_bytes(&self) -> u64 {
        PageTables::device_bytes(&self.layout, self.slots, self.width)
            + self.workspace_bytes
            + conversion::device_bytes(&self.layout, self.width)
    }

    /// Checks what the cache's tensors rely on: a workspace, at least one attention group, each
    /// of pages of one unit that fill a plane, and state units exactly when the layout keeps a
    /// state group.
    fn check(&self) -> Result<()> {
        if self.layout.attention_groups() == 0 {
            return Err(Error::Invalid("a KV cache needs an attention group".into()));
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
            if u64::from(group.block_size) * group.row_bytes() != self.layout.plane_bytes {
                return Err(Error::Invalid(format!(
                    "attention group {index}'s page of {} tokens of {} bytes does not fill a \
                     plane of {} bytes",
                    group.block_size,
                    group.row_bytes(),
                    self.layout.plane_bytes
                )));
            }
        }
        Ok(())
    }
}

/// The numbers the buffer pool gave the regions the cache's memory is (the answer to `Open`):
/// what tells a run a completion states, `(region, first unit, count)`, as units of the pool,
/// state units, pages of the host tier or slots of a converted-load source buffer.
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
    /// Per cache group, the region of the group's converted-load source buffer: unit `u` of it
    /// is slot `u` of the buffer. `None` for a group that converts none.
    pub sources: Vec<Option<u8>>,
}

/// The KV cache of one rank: its planes, page tables, workspace, host tier and scratch pages.
pub struct KvCacheMemory {
    config: KvCacheConfig,
    planes: Planes,
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
        Ok(Self {
            planes: Planes::meta(Arc::clone(&config.layout), config.units)?,
            tables: PageTables::meta(
                Arc::clone(&config.layout),
                config.slots,
                config.width,
                config.units,
            )?,
            workspace: None,
            regions: None,
            host: None,
            scratch: None,
            states,
            config,
        })
    }

    /// Opens the cache of `config` on `device`: takes its planes and its state units from
    /// `owner` (the rank's Storage Server client, or storage local to the process), allocates its
    /// page tables and workspace, and resets what was created of the planes for this rank on
    /// `stream`, which is synchronized before returning.
    ///
    /// # Errors
    ///
    /// A config whose layout the cache's tensors cannot state, segments that do not hold the
    /// config's units, or a storage `owner` or the device refuses.
    pub fn open(
        config: KvCacheConfig,
        device: DeviceId,
        owner: &dyn ExclusiveStorage,
        stream: &Stream,
    ) -> Result<Self> {
        config.check()?;

        let planes = Planes::open(
            Arc::clone(&config.layout),
            config.units,
            config.segments.as_deref(),
            device,
            owner,
            stream,
        )?;
        let tables = PageTables::open(
            Arc::clone(&config.layout),
            config.slots,
            config.width,
            config.units,
            device,
            stream,
        )?;
        let workspace = Storage::linear(device, config.workspace_bytes as usize)?;
        let states = match config.layout.state_group() {
            Some(_) => Some(StateUnits::open(
                &config.layout,
                config.states.0,
                config.states.1.as_deref(),
                device,
                owner,
            )?),
            None => None,
        };

        Ok(Self {
            config,
            planes,
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

    /// The planes of attention group `group`'s layers, in model order: each layer's keys and
    /// values `[units, block_size, kv_heads, head_dim]` in the group's stored type, its flags
    /// `[units, 1, 1, 1]` U8, for an FP8 cache its scales `[units, 1, 1, 1]` F32, and for a
    /// cache that keeps index keys its index keys `[units, block_size / ratio, width]` in the
    /// group's compute type. A page id of the group is a unit id.
    ///
    /// The tensors are over the cache's memory: they are used only while the cache lives.
    ///
    /// # Errors
    ///
    /// A group that is no attention group of the cache.
    pub fn planes(&self, group: usize) -> Result<Vec<KvPlanes>> {
        let layout = self.attention(group)?;

        (0..layout.layers.len())
            .map(|layer| self.planes.layer(layout, layer))
            .collect()
    }

    /// The planes of the draft group's layers, in the drafter's order, as its attention reads
    /// them: pages of `block_size / span` tokens, `[span · units, …]`, page `c · units + u`
    /// holding the `c`-th run of the draft page in unit `u`.
    ///
    /// # Errors
    ///
    /// A cache that keeps no draft group.
    pub(crate) fn draft_planes(&self) -> Result<Vec<KvPlanes>> {
        let (layout, span) = self.draft()?;

        (0..layout.layers.len())
            .map(|layer| self.planes.draft_layer(layout, layer, span))
            .collect()
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
    /// gathered from [`Self::page_table`], `[rows, width]` U32.
    ///
    /// # Errors
    ///
    /// A group that is no attention group of the cache.
    pub(crate) fn lend(&self, group: usize, block_tables: Tensor) -> Result<KvCache> {
        Ok(KvCache {
            block_tables,
            page_size: self.attention(group)?.block_size,
            layers: self.planes(group)?,
            workspace: self.workspace()?,
        })
    }

    /// The draft cache as a drafter's forward is lent it: `block_tables` names, per row, the
    /// drafter's pages ([`Self::draft_planes`]), which a speculative step states from the
    /// draft group's entries of the rank's device block table.
    ///
    /// # Errors
    ///
    /// A cache that keeps no draft group.
    pub(crate) fn lend_draft(&self, block_tables: Tensor) -> Result<KvCache> {
        let (layout, span) = self.draft()?;

        Ok(KvCache {
            block_tables,
            page_size: layout.block_size / span,
            layers: self.draft_planes()?,
            workspace: self.workspace()?,
        })
    }

    /// The cache as the rank reports it to the engine: its layout, units, plane runs and
    /// segments. The rank adds its identities and endpoint.
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

    /// The bytes the planes and the state units map on the device; a sleeping rank's segments
    /// are not in it.
    pub(crate) fn mapped_bytes(&self) -> u64 {
        self.planes.mapped_bytes() + self.states.as_ref().map_or(0, StateUnits::mapped_bytes)
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
        if regions.host.len() != groups || regions.sources.len() != groups {
            return Err(Error::Invalid(format!(
                "regions stated for {} and {} cache groups of a cache of {groups}",
                regions.host.len(),
                regions.sources.len()
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
    /// its state units, or a cache group's host pages or converted-load source buffer. A cache
    /// whose regions are not stated holds none.
    pub(crate) fn holds(&self, region: u8) -> bool {
        self.regions.as_ref().is_some_and(|regions| {
            regions.units == region
                || regions.states == Some(region)
                || regions.host.contains(&Some(region))
                || regions.sources.contains(&Some(region))
        })
    }

    /// Whether `region` is a cache group's converted-load source region, whose pages the cache
    /// converts into its own representation as it loads them ([`Self::copy`]).
    pub(crate) fn converts_from(&self, region: u8) -> bool {
        self.regions
            .as_ref()
            .is_some_and(|regions| regions.sources.contains(&Some(region)))
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

        let groups = self.config.layout.groups.len();
        let state = self.config.layout.state_group();
        let mut resets = Vec::new();
        for change in changes {
            let Some((_, first, count)) = change.run.filter(|_| change.reset) else {
                continue;
            };
            if usize::from(change.class) >= groups || Some(usize::from(change.class)) == state {
                continue;
            }
            if first == 0 || u64::from(first) + u64::from(count) > u64::from(self.config.units) {
                return Err(Error::Invalid(format!(
                    "the block at position {} of class {} of request slot {} takes units \
                     {first}..{} of a pool of units 1..{}",
                    change.position,
                    change.class,
                    change.slot,
                    u64::from(first) + u64::from(count),
                    self.config.units
                )));
            }
            resets.push((first, count));
        }

        // The pooled tables check every change before they write a row.
        self.tables.apply(changes, stream)?;

        if let Some(stream) = stream {
            for (first, count) in resets {
                self.planes.reset_units(first, count, false, stream)?;
            }
        }
        Ok(())
    }

    /// Maps the cache's host tier: part `part` (the rank's index in its worker group) of each
    /// cache group's host region at the rank's host's Storage Server through `client`, whose
    /// copies run on `stream`, the rank's copy stream. `regions` names, per cache group, the host
    /// region of its pages as the session's layout states it (unit `u` of the region is page `u`
    /// of the group), or `None` for a group whose pages have no host copy, and `sources` its
    /// converted-load source region (unit `u` is slot `u` of the group's source buffer), or
    /// `None` for a group that converts none; the open session must state them. A cache that
    /// admits converted loads takes its part of each source region as the group's source buffer,
    /// with its conversion scratch, and launches its conversions with `kernels`.
    ///
    /// # Errors
    ///
    /// A cache that holds no device, one whose host tier is mapped already, or regions stated
    /// for another number of cache groups; the server's refusal.
    pub(crate) fn map_host_parts(
        &mut self,
        client: &StorageClient,
        (regions, sources, part): (&[Option<&str>], &[Option<&str>], u32),
        stream: Stream,
        kernels: Arc<ops::Generation>,
    ) -> Result<()> {
        let device = self.planes.device_id().ok_or(Error::State(
            "the cache holds no device: it maps no host tier",
        ))?;
        if self.host.is_some() {
            return Err(Error::State("the cache's host tier is mapped already"));
        }

        let layout = &self.config.layout;
        if regions.len() != layout.groups.len() || sources.len() != layout.groups.len() {
            return Err(Error::Invalid(format!(
                "host regions stated for {} and source regions for {} cache groups of a cache \
                 of {}",
                regions.len(),
                sources.len(),
                layout.groups.len()
            )));
        }

        let conversion = Conversion::open(
            client,
            layout,
            device,
            (sources, part),
            self.config.width,
            &stream,
            kernels,
        )?;
        self.host = Some(HostTier::map(
            client,
            layout,
            device,
            (regions, part),
            stream,
            conversion,
        )?);
        Ok(())
    }

    /// The page identity of each cache group for weights of `fingerprint`
    /// (`PrefixCacheInfo::identities`).
    pub(crate) fn page_identities(&self, fingerprint: &str) -> Vec<String> {
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
    ///   state unit, a checkpoint's load and store;
    /// - a slot of a group's converted-load source to units, a converted load: on the copy
    ///   stream.
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

        // The group whose host pages, or whose source slots, `region` holds.
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
        let (mut units, mut loads, mut converted, mut stores) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
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
                    } else if let Some(group) = group_of(&regions.sources, from.0) {
                        converted.push((group, page(from)?, block));
                    } else {
                        return Err(Error::Invalid(format!(
                            "a copy into the cache's units reads region {}, which is none of \
                             its host pages or source buffers",
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
        let paged: Vec<&HostPage> = loads.iter().chain(&converted).chain(&stores).collect();
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
                &converted,
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

    /// The generation the cache's own kernels launch from, from the next converted load on:
    /// set at a generation switch.
    pub(crate) fn set_kernels(&self, kernels: Arc<ops::Generation>) {
        if let Some(host) = &self.host {
            host.set_kernels(kernels);
        }
    }

    /// The kernels the cache's own work launches, as `(operator, key)`: those of its converted
    /// loads, stated under a meta context, so a cache that holds no device states them too.
    /// Empty for a cache that converts none.
    ///
    /// # Errors
    ///
    /// When a conversion's descriptors refuse the cache's layout.
    pub(crate) fn keys(&self) -> Result<BTreeSet<(&'static str, String)>> {
        conversion::keys(&self.config.layout, self.config.units)
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
    pub(crate) fn scratch_planes(&self, group: usize) -> Result<Vec<KvPlanes>> {
        let layout = self.attention(group)?;
        let scratch = self.scratch()?;

        (0..layout.layers.len())
            .map(|layer| scratch.layer(layout, layer))
            .collect()
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

        scratch.copy_history(&self.planes, layout, &history, stream)
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
    use nn::family::{StateField, StateShape};
    use proto::request::PrefixCacheDtype;

    use super::*;

    /// The regions the tests number the cache's memory by: its units, its state units, and the
    /// host pages of each of up to three groups.
    const UNITS: u8 = 0;
    const STATES: u8 = 1;
    const HOST: [u8; 3] = [4, 5, 6];

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
            }],
        };
        let layout = CacheLayout::hybrid(4, 2, 1, 4, PrefixCacheDtype::Float8E4m3Fn, &[state])
            .and_then(|layout| layout.with_draft(1, 1, 4))
            .expect("a hybrid layout with a draft group");

        KvCacheConfig {
            layout: Arc::new(layout),
            units: 16,
            slots: 2,
            width: 4,
            segments: None,
            workspace_bytes: 1 << 10,
            states: (8, None),
        }
    }

    /// A model of two attention groups of two layers each: full attention and a sliding
    /// window.
    fn two_groups() -> KvCacheConfig {
        let mut layout = CacheLayout::dense(4, 2, 1, 4, PrefixCacheDtype::BFloat16);
        let mut window = layout.groups[0].clone();
        window.kind = crate::batch::cache::CacheGroupKind::SlidingWindow { window: 8 };
        window.layer_ids = vec![2, 3];
        layout.groups.push(window);

        KvCacheConfig {
            layout: Arc::new(layout.validated().expect("two groups of two layers")),
            units: 16,
            slots: 2,
            width: 4,
            segments: None,
            workspace_bytes: 1 << 10,
            states: (0, None),
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

    /// The regions of a cache of `groups` cache groups whose pages have host copies: the hybrid
    /// layout of [`config`] for three, with its state region, and [`two_groups`] for two.
    fn regions(groups: usize) -> CacheRegions {
        CacheRegions {
            units: UNITS,
            states: (groups == 3).then_some(STATES),
            host: HOST[..groups].iter().copied().map(Some).collect(),
            sources: vec![None; groups],
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

        // The attention group: one column per layer, FP8 with a scale and a flag per unit.
        let planes = cache.planes(0)?;
        assert_eq!(planes.len(), 2);
        for layer in &planes {
            assert_eq!(layer.key.dims(), [16, 4, 1, 4]);
            assert_eq!(layer.value.dims(), [16, 4, 1, 4]);
            assert_eq!(layer.key.dtype(), DType::F8E4M3);
            assert_eq!(layer.key_initialized.dims(), [16, 1, 1, 1]);
            assert_eq!(layer.key_initialized.dtype(), DType::U8);
            let scale = layer
                .value_scale
                .as_ref()
                .expect("an FP8 cache states scales");
            assert_eq!(
                (scale.dims(), scale.dtype()),
                (&[16, 1, 1, 1][..], DType::F32)
            );
        }

        // The draft group: one layer over the two columns of a unit, pages of an attention
        // page's tokens.
        let draft = cache.draft_planes()?;
        assert_eq!(draft.len(), 1);
        assert_eq!(draft[0].key.dims(), [2 * 16, 4, 1, 4]);
        assert_eq!(draft[0].key_initialized.dims(), [2 * 16, 1, 1, 1]);

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

        // What a forward is lent.
        let lent = cache.lend(0, cache.page_table(0)?)?;
        assert_eq!((lent.page_size, lent.layers.len()), (4, 2));
        assert_eq!(cache.lend_draft(cache.page_table(0)?)?.page_size, 4);
        Ok(())
    }

    /// A model of two attention groups has a pooled table and planes per group.
    #[test]
    fn each_attention_group_has_its_planes_and_its_table() -> Result<()> {
        let mut cache = KvCacheMemory::meta(two_groups())?;

        for group in 0..2 {
            assert_eq!(cache.planes(group)?.len(), 2);
            assert_eq!(cache.page_table(group)?.dims(), [3, 4]);
        }
        assert!(cache.draft_planes().is_err());

        // The groups' rows are apart: slot 1 holds other units in each.
        let changes = [allocated(1, 0, &[3, 4]), allocated(1, 1, &[9])].concat();
        cache.apply(&changes, None)?;
        assert_eq!(cache.host_table(0)?[..4], [3, 4, 0, 0]);
        assert_eq!(cache.host_table(1)?[..4], [9, 0, 0, 0]);
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

        // A cache that converts nothing launches no kernel of its own.
        assert!(cache.keys()?.is_empty());
        Ok(())
    }

    /// The scratch page set of a cache with no device is a set of meta tensors in the cache's
    /// page shape.
    #[test]
    fn scratch_pages_have_the_caches_page_shape() -> Result<()> {
        let mut cache = KvCacheMemory::meta(two_groups())?;
        assert_eq!(cache.scratch_capacity(), 0);
        assert!(cache.scratch_planes(0).is_err());

        cache.open_scratch(12, None)?;
        assert_eq!(cache.scratch_capacity(), 12);
        assert!(cache.open_scratch(12, None).is_err());

        let planes = cache.scratch_planes(1)?;
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0].key.dims(), [12, 4, 1, 4]);

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

    /// The fixed bytes are the tables, the workspace and the conversion's scratch and units.
    #[test]
    fn the_fixed_bytes_are_the_tables_and_the_workspace() {
        let config = two_groups();
        assert_eq!(config.fixed_bytes(), 2 * 3 * 4 * 4 + (1 << 10));
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
            device,
            &crate::memory::exclusive::LocalStorage,
            &stream,
        )?;
        assert_ne!(
            cache.workspace()?.device().location(),
            ops::meta().location()
        );
        assert!(cache.mapped_bytes() > 0);
        assert_eq!(cache.planes(0)?[1].key.dims(), [16, 4, 1, 4]);

        cache.set_regions(regions(2))?;
        cache.apply(&allocated(1, 0, &[3, 4]), Some(&stream))?;

        // A clone moves a unit's bytes into the unit its copy names. A column's planes lie unit
        // after unit, so the key plane of unit 1 in column 0 is the layout's plane bytes (four
        // tokens of one 4-wide BF16 head, 32 bytes) and the next unit's plane follows it.
        let plane = cache.layout().plane_bytes as usize;
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
        let flags = cache.planes(0)?[0].key_initialized.clone();
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

        let layout =
            Arc::new(CacheLayout::dense(64, 2, 2, 256, PrefixCacheDtype::BFloat16).validated()?);
        let granularity = platform::DeviceAllocation::granularity(device)? as u64;
        let units = segments::segmented_units(4096, layout.plane_bytes, granularity);
        let config = KvCacheConfig {
            segments: segments::own(1, units, layout.plane_bytes, granularity),
            layout,
            units,
            slots: 2,
            width: 4,
            workspace_bytes: 1 << 20,
            states: (0, None),
        };
        let mut cache = KvCacheMemory::open(config, device, &local, &stream)?;
        cache.set_regions(CacheRegions {
            units: UNITS,
            states: None,
            host: vec![None],
            sources: vec![None],
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

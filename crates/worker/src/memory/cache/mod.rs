//! The rank's KV cache: the memory a causal forward reads and writes its keys and values in.
//!
//! "KV cache" is the reference engines' word for it (vLLM's `KVCacheManager`, SGLang's token to
//! KV pool); the scheduler's side allocates its pages and indexes them by prefix, and this
//! module is the rank's side. [`KvCacheMemory`] owns:
//!
//! - the **planes** ([`planes`]): the unit pool's exclusive storage at the host's Storage
//!   Server, with each plane's initialization flag and, for an FP8 cache, its scale;
//! - the **page tables** ([`tables`]): the scheduler's page assignments per request slot and
//!   cache group, and one pooled table per attention group that a forward gathers its rows
//!   from;
//! - the **attention workspace**, which the paged-attention kernel uses during a forward;
//! - the **host tier** ([`host`]): the host page pools pages are loaded from and offloaded to
//!   on the rank's copy stream, with the converted loads ([`conversion`]) of a cache that
//!   admits them;
//! - the **scratch pages** ([`scratch`]) a shadow forward writes.
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
//! The runner that holds the cache issues a batch's cache work on the batch's stream, on the
//! executor thread, before the batch's forward, in this order:
//!
//! 1. [`KvCacheMemory::install`]: the batch's rows of the pooled tables;
//! 2. [`KvCacheMemory::apply_units`]: the reset of the units the scheduler allocated for the
//!    batch, then the copies of a leased prefix into them;
//! 3. [`KvCacheMemory::copy_pages`]: the loads from the host tier into those units, which the
//!    stream waits for, and the offloads, whose event the batch's output waits for;
//! 4. the restores of recurrent state
//!    ([`StateMemory::copy`](crate::memory::state::StateMemory::copy)), then the forward, then
//!    the keeps.
//!
//! ## Index tensors
//!
//! Page tables and unit ids are `u32`. A pooled table never holds `nn::NO_SLOT`: a page a row
//! does not hold names unit 0, the padding unit, because a table entry feeds a gather. The slot
//! column of a shadow forward, which its runner numbers over the scratch set's pages, holds
//! `nn::NO_SLOT` where a token writes nothing, as the slot column of a served batch does.

pub mod conversion;
pub mod host;
pub mod layout;
mod planes;
pub mod scratch;
pub mod segments;
mod tables;

use std::collections::BTreeSet;
use std::sync::Arc;

use ipc::{BlockTable, CacheUnitAllocation, HostPoolInfo, PageCopy, PrefixCacheInfo};
use nn::{KvCache, KvPlanes};
use platform::{DeviceId, Event, Stream};
use uniserve_tensor::{DType, Device, Tensor};

use self::conversion::Conversion;
use self::host::HostTier;
pub use self::layout::{CacheLayout, Checkpoint, GroupLayout};
use self::planes::Planes;
use self::scratch::ScratchPages;
pub use self::segments::PlaneSegment;
use self::tables::PageTables;
use crate::memory::allocation::Storage;
use crate::memory::client::StorageClient;
use crate::memory::exclusive::ExclusiveStorage;
use crate::memory::lend;
use crate::{Error, Result};

/// The bytes the trtllm-gen paged-attention kernel takes as its workspace: what
/// [`KvCacheConfig::workspace_bytes`] is for a model that attends through it.
pub const ATTENTION_WORKSPACE_BYTES: u64 = 512 << 20;

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
    /// The segments the value planes are made of on a device ([`segments`]), which hold
    /// `units` units; `None` for planes that are not split.
    pub segments: Option<Vec<PlaneSegment>>,
    /// The bytes of the attention workspace a forward is lent.
    pub workspace_bytes: u64,
}

impl KvCacheConfig {
    /// The device bytes the cache holds beside its planes: the pooled page tables, the
    /// attention workspace, and the scratch and unit table of its converted loads.
    pub fn fixed_bytes(&self) -> u64 {
        PageTables::device_bytes(&self.layout, self.slots, self.width)
            + self.workspace_bytes
            + conversion::device_bytes(&self.layout, self.width)
    }

    /// Checks what the cache's tensors rely on: a workspace, and at least one attention group,
    /// each of pages of one unit that fill a plane.
    fn check(&self) -> Result<()> {
        if self.layout.attention_groups() == 0 {
            return Err(Error::Invalid("a KV cache needs an attention group".into()));
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

/// The KV cache of one rank: its planes, page tables, workspace, host tier and scratch pages.
pub struct KvCacheMemory {
    config: KvCacheConfig,
    planes: Planes,
    tables: PageTables,
    /// The attention workspace on the device; `None` for a cache that holds no device.
    workspace: Option<Arc<Storage>>,
    /// The host tier, once the engine sized it ([`Self::map_host_tier`]).
    host: Option<HostTier>,
    /// The scratch pages of shadow forwards, once opened ([`Self::open_scratch`]).
    scratch: Option<ScratchPages>,
}

impl KvCacheMemory {
    /// A cache that holds no device: its tensors are meta tensors of the shapes and element
    /// types the cache of `config` has, its page tables are on the host alone, and its device
    /// work is refused or does nothing.
    ///
    /// # Errors
    ///
    /// A config whose layout the cache's tensors cannot state.
    pub fn meta(config: KvCacheConfig) -> Result<Self> {
        config.check()?;

        Ok(Self {
            planes: Planes::meta(Arc::clone(&config.layout), config.units)?,
            tables: PageTables::meta(
                Arc::clone(&config.layout),
                config.slots,
                config.width,
                config.units,
            )?,
            workspace: None,
            host: None,
            scratch: None,
            config,
        })
    }

    /// Opens the cache of `config` on `device`: takes its planes from `owner` (the rank's
    /// Storage Server client, or storage local to the process), allocates its page tables and
    /// workspace, and resets what was created for this rank on `stream`, which is synchronized
    /// before returning.
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

        Ok(Self {
            config,
            planes,
            tables,
            workspace: Some(workspace),
            host: None,
            scratch: None,
        })
    }

    /// The config the cache was opened with.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// The cache's layout.
    pub fn layout(&self) -> &Arc<CacheLayout> {
        &self.config.layout
    }

    /// The device the cache's tensors are on: its GPU, or the meta device.
    pub fn device(&self) -> Device {
        match self.planes.device_id() {
            Some(device) => backend::cuda(device.0 as usize),
            None => backend::meta(),
        }
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
    /// `[units, 1, 1, 1]` U8 and, for an FP8 cache, its scales `[units, 1, 1, 1]` F32. A page
    /// id of the group is a unit id.
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
    pub fn draft_planes(&self) -> Result<Vec<KvPlanes>> {
        let (layout, span) = self.draft()?;

        (0..layout.layers.len())
            .map(|layer| self.planes.draft_layer(layout, layer, span))
            .collect()
    }

    /// The pooled page table of attention group `group`, `[slots + 1, width]` U32: row
    /// [`Self::table_row`] of a request slot names the unit of each of its logical pages, and
    /// row [`Self::padding_row`] and every page a slot does not hold name unit 0. A forward
    /// gathers its batch's rows from it.
    ///
    /// # Errors
    ///
    /// A group that is no attention group of the cache.
    pub fn page_table(&self, group: usize) -> Result<Tensor> {
        self.attention(group)?;
        self.tables.table(group)
    }

    /// The row of the pooled tables that request slot `slot` owns.
    ///
    /// # Errors
    ///
    /// A slot outside `1..=slots`.
    pub fn table_row(&self, slot: u32) -> Result<u32> {
        self.tables.row(slot)
    }

    /// The row of the pooled tables that names only the padding unit: what a bucket's padding
    /// rows gather.
    pub fn padding_row(&self) -> u32 {
        self.tables.padding_row()
    }

    /// The pooled page table of attention group `group` as the host holds it,
    /// `[slots + 1, width]` row-major: what the device's table holds once the installs issued
    /// so far completed.
    ///
    /// # Errors
    ///
    /// A group that is no attention group of the cache.
    pub fn host_table(&self, group: usize) -> Result<&[u32]> {
        self.tables.host_table(group)
    }

    /// The pages request slot `slot` holds in cache group `group`, of any kind, as the
    /// scheduler last stated them: its first logical page and the units of the pages from
    /// there on, page-major. `None` when the slot holds no table of the group.
    pub fn installed_pages(&self, slot: u32, group: u32) -> Option<(u32, &[u32])> {
        self.tables.installed(slot, group)
    }

    /// The attention workspace, `[workspace_bytes]` U8.
    pub fn workspace(&self) -> Result<Tensor> {
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
    pub fn lend(&self, group: usize, block_tables: Tensor) -> Result<KvCache> {
        Ok(KvCache {
            block_tables,
            page_size: self.attention(group)?.block_size,
            layers: self.planes(group)?,
            workspace: self.workspace()?,
        })
    }

    /// The draft cache as a drafter's forward is lent it: `block_tables` names, per row, the
    /// drafter's pages ([`Self::draft_planes`]), which a speculative step states from
    /// [`Self::installed_pages`].
    ///
    /// # Errors
    ///
    /// A cache that keeps no draft group.
    pub fn lend_draft(&self, block_tables: Tensor) -> Result<KvCache> {
        let (layout, span) = self.draft()?;

        Ok(KvCache {
            block_tables,
            page_size: layout.block_size / span,
            layers: self.draft_planes()?,
            workspace: self.workspace()?,
        })
    }

    /// The cache as the rank reports it to the engine: its layout, units, plane runs and
    /// segments. The rank adds its page pools, identities and endpoint.
    pub fn info(&self) -> PrefixCacheInfo {
        let inherited = self.planes.inherited();

        PrefixCacheInfo {
            plane_runs: self.planes.runs().to_vec(),
            segments: self
                .planes
                .segments()
                .map(<[_]>::to_vec)
                .unwrap_or_default(),
            held_segments: self.planes.held_segments(),
            inherited_start: inherited.start,
            inherited_units: inherited.end - inherited.start,
            ..self.config.layout.info(self.config.units)
        }
    }

    /// The segments the value planes are made of, in unit order; `None` for planes that are
    /// not split.
    pub fn segments(&self) -> Option<&[ipc::CacheSegment]> {
        self.planes.segments()
    }

    /// The segments the planes hold, `[0, held)`; zero for planes that are not split.
    pub fn held_segments(&self) -> u32 {
        self.planes.held_segments()
    }

    /// The device bytes one unit of both value planes holds over every column.
    pub fn unit_bytes(&self) -> u64 {
        self.planes.unit_bytes()
    }

    /// The bytes the planes map on the device; a sleeping rank's segments are not in it.
    pub fn mapped_bytes(&self) -> u64 {
        self.planes.mapped_bytes()
    }

    /// Installs the scheduler's page tables of a batch ([`ipc::BlockTable`]) into the mirror
    /// and the pooled tables, on `stream` for a cache on a device and with `None` for one that
    /// holds none. Every table is checked first, and nothing changes when one is refused.
    ///
    /// # Errors
    ///
    /// A table of a slot or group the cache does not hold, of units outside the pool, of more
    /// pages than a row or of more tokens than its pages, by slot, group and reason.
    pub fn install(&mut self, tables: &[BlockTable], stream: Option<&Stream>) -> Result<()> {
        self.tables.install(tables, stream)
    }

    /// Forgets the page tables of `slots` and names the padding unit in their rows again, on
    /// `stream` for a cache on a device, ordered behind every batch that reads those rows.
    ///
    /// # Errors
    ///
    /// A slot outside the cache's request slots.
    pub fn release_slots(&mut self, slots: &[u32], stream: Option<&Stream>) -> Result<()> {
        self.tables.release(slots, stream)
    }

    /// Applies what the scheduler allocated for a batch's pages ([`ipc::CacheUnitAllocation`]),
    /// on `stream` for a cache on a device: the new units' flags are cleared and their scales
    /// set to one, then the units of a leased prefix are copied into them. Their values are
    /// left as they are: a page is written before it is read.
    ///
    /// The order is the only correct one: a destination is reset first, which is what makes
    /// the page new, and the copies follow. They read sources the scheduler still leases, and
    /// the forward that reads the destinations follows on the same stream.
    ///
    /// Every allocation is checked before anything is issued.
    ///
    /// # Errors
    ///
    /// A group the cache does not hold; a unit that is the padding unit or outside the pool; a
    /// copy whose destination the allocation did not name, whose destination is its source, or
    /// which writes a read-only unit of the predecessor epoch; a copy in a cache with a state
    /// group, whose state planes a unit copy does not move.
    pub fn apply_units(
        &mut self,
        units: &[CacheUnitAllocation],
        stream: Option<&Stream>,
    ) -> Result<()> {
        if self.planes.device_id().is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "cache units are applied on a stream exactly when the cache is on a device".into(),
            ));
        }

        let groups = self.config.layout.groups.len();
        let pool = self.config.units;
        for allocation in units {
            let group = allocation.group_id as usize;
            if group >= groups {
                return Err(Error::Invalid(format!(
                    "the scheduler allocated units of cache group {group}, and the cache has \
                     {groups} group(s)"
                )));
            }

            let held = |unit: ipc::UnitId, what: &str| -> Result<()> {
                if unit.0 == 0 || unit.0 >= pool {
                    return Err(Error::Invalid(format!(
                        "the scheduler named cache unit {} of group {group} as {what}, outside \
                         the pool's units 1..{pool} (unit 0 is the padding unit)",
                        unit.0
                    )));
                }
                Ok(())
            };
            for &unit in &allocation.unit_ids {
                held(unit, "a new unit")?;
            }
            for &(source, destination) in &allocation.copies {
                held(source, "a copy's source")?;
                held(destination, "a copy's destination")?;

                if !allocation.unit_ids.contains(&destination) {
                    return Err(Error::Invalid(format!(
                        "the scheduler copies into cache unit {} of group {group}, which the \
                         allocation did not name among its new units",
                        destination.0
                    )));
                }
            }
            self.planes.check_copies(&allocation.copies)?;

            // A page of a cache with a state group holds its layers' checkpoints in planes of
            // their own, which a unit copy does not move: the destination would look
            // initialized while its state planes hold another page's.
            if !allocation.copies.is_empty() && self.config.layout.state_fields() > 0 {
                return Err(Error::Unsupported(
                    "a unit copy in a cache with a state group does not move the state planes"
                        .into(),
                ));
            }
        }

        let Some(stream) = stream else {
            return Ok(());
        };
        for allocation in units {
            let acquired: Vec<u32> = allocation.unit_ids.iter().map(|unit| unit.0).collect();
            for (first, count) in planes::unit_runs(&acquired) {
                self.planes.reset_units(first, count, false, stream)?;
            }
            self.planes.copy_units(&allocation.copies, stream)?;
        }
        Ok(())
    }

    /// Maps the cache's host tier: a page pool of `tokens` tokens per cache group at the
    /// rank's host's Storage Server through `client`, whose copies run on `stream`, the rank's
    /// copy stream. A cache that admits converted loads also takes its source buffers and
    /// conversion scratch, and launches its conversions with `kernels`. Returns what the rank
    /// reports of each pool, in group order; `endpoint` is where an engine on another host
    /// reaches the server.
    ///
    /// # Errors
    ///
    /// A cache that holds no device, or one whose host tier is mapped already; the server's
    /// refusal.
    pub fn map_host_tier(
        &mut self,
        client: &StorageClient,
        stream: Stream,
        endpoint: Option<&str>,
        tokens: u64,
        kernels: Arc<ops::Generation>,
    ) -> Result<Vec<HostPoolInfo>> {
        let device = self.planes.device_id().ok_or(Error::State(
            "the cache holds no device: it maps no host tier",
        ))?;
        if self.host.is_some() {
            return Err(Error::State("the cache's host tier is mapped already"));
        }

        let layout = &self.config.layout;
        let conversion =
            Conversion::open(client, layout, device, self.config.width, &stream, kernels)?;
        let (conversion, converted) = match conversion {
            Some((conversion, converted)) => (Some(conversion), converted),
            None => (None, Vec::new()),
        };

        let (tier, mut pools) =
            HostTier::map(client, layout, device, endpoint, tokens, stream, conversion)?;
        for (pool, converted) in pools.iter_mut().zip(converted) {
            pool.conversion = Some(converted);
        }

        self.host = Some(tier);
        Ok(pools)
    }

    /// The bytes one token of the cache's pages takes in the host tier, over every group.
    pub fn host_token_bytes(&self) -> u64 {
        host::token_bytes(&self.config.layout)
    }

    /// The page identity of each cache group for weights of `fingerprint`
    /// (`PrefixCacheInfo::identities`).
    pub fn page_identities(&self, fingerprint: &str) -> Vec<String> {
        host::page_identities(&self.config.layout, fingerprint)
    }

    /// The host tier, or the refusal of a cache that maps none.
    fn host(&self) -> Result<&HostTier> {
        self.host.as_ref().ok_or_else(|| {
            Error::Invalid("a batch copies pages and the cache maps no host tier".into())
        })
    }

    /// Issues a batch's page copies ([`ipc::PageCopy`]) on the copy stream: loads the units of
    /// `loads` from the host tier, converting the pages of a conversion source, and offloads
    /// the published units of `offloads` into it.
    ///
    /// `stream` is the batch's stream, on which [`Self::apply_units`] reset the loaded units:
    /// the loads start behind that reset and `stream` waits for them, so the forward issued
    /// next reads the loaded pages. Returns the event recorded behind the offloads, which the
    /// batch's output waits on; `None` for a batch that offloads nothing.
    ///
    /// # Errors
    ///
    /// A batch that copies pages on a cache that maps no host tier; a page of another pool or
    /// shape than its group's; a page of a conversion source on a cache that converts none.
    pub fn copy_pages(
        &mut self,
        loads: &[PageCopy],
        offloads: &[PageCopy],
        stream: &Stream,
    ) -> Result<Option<Event>> {
        if loads.is_empty() && offloads.is_empty() {
            return Ok(None);
        }
        self.host()?
            .copy(&self.config.layout, &self.planes, loads, offloads, stream)
    }

    /// Copies the published units of `offloads` to the host tier and waits for the copies on
    /// the copy stream, with no batch to order them behind: the scheduler holds the units
    /// meanwhile. The executor runs it as a job.
    ///
    /// # Errors
    ///
    /// As [`Self::copy_pages`].
    pub fn offload_pages(&mut self, offloads: &[PageCopy]) -> Result<()> {
        if offloads.is_empty() {
            return Ok(());
        }
        self.host()?
            .offload_now(&self.config.layout, &self.planes, offloads)
    }

    /// The generation the cache's own kernels launch from, from the next converted load on:
    /// set at a generation switch.
    pub fn set_kernels(&self, kernels: Arc<ops::Generation>) {
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
    pub fn keys(&self) -> Result<BTreeSet<(&'static str, String)>> {
        conversion::keys(&self.config.layout, self.config.units)
    }

    /// Gives every segment of the value planes back to the device, through `owner`, which
    /// retained them: what a sleep releases. No request holds a unit and no device work
    /// addresses the planes. Planes that are not split have nothing to give back, which is not
    /// an error.
    ///
    /// # Errors
    ///
    /// When `owner` fails.
    pub fn sleep(&mut self, owner: &dyn ExclusiveStorage) -> Result<()> {
        if self.planes.segments().is_none() {
            tracing::info!("the cache's planes are not split, so a sleep gives back nothing");
            return Ok(());
        }
        self.planes.release_segments(0, owner)
    }

    /// Creates the segments given back again at their addresses, the first `limit` of them or
    /// all, through `owner`, and resets their units on `stream`, which is synchronized before
    /// returning. A wake of a cache that gave nothing back, or whose planes are not split,
    /// does nothing.
    ///
    /// # Errors
    ///
    /// When `owner` or the stream fails; the cache is then as asleep as it was.
    pub fn wake(
        &mut self,
        owner: &dyn ExclusiveStorage,
        stream: &Stream,
        limit: Option<u32>,
    ) -> Result<()> {
        if self.planes.segments().is_none() {
            return Ok(());
        }
        self.planes.restore_segments(owner, stream, limit)
    }

    /// Gives back the segments from `first` on, through `owner`: a successor epoch needs
    /// their storage. The scheduler withdrew their units and no device work addresses them.
    ///
    /// # Errors
    ///
    /// For planes that are not split, for segments the predecessor epoch writes, or when
    /// `owner` fails.
    pub fn release_segments(&mut self, first: u32, owner: &dyn ExclusiveStorage) -> Result<()> {
        self.planes.release_segments(first, owner)
    }

    /// Adopts the predecessor epoch's pages in the units `units` (half-open ranges): the value
    /// planes are the predecessor's segments, mapped whole, so its pages lie at their own
    /// units, and what is copied is their flags and scales, from the predecessor's storages
    /// into this rank's, as a task of the host's Transfer Server. Returns the units copied:
    /// none for a standby, whose storages are the predecessor's.
    ///
    /// # Errors
    ///
    /// A cache that holds no device; the server's refusal or the task's failure, after which
    /// the caller adopts none of the units.
    pub fn adopt(&self, client: &StorageClient, units: &[[u32; 2]]) -> Result<u32> {
        let device = self.planes.device_id().ok_or(Error::State(
            "the cache holds no device: it adopts no pages",
        ))?;

        let mut storages: Vec<(&str, u64)> = planes::FLAGS.iter().map(|name| (*name, 1)).collect();
        if self.config.layout.quantized {
            storages.extend(planes::SCALES.iter().map(|name| (*name, 4)));
        }

        client.copy_predecessor_cells(
            device,
            &storages,
            self.config.layout.columns,
            self.config.units,
            units,
        )
    }

    /// The pages a scratch set of `bytes` of the device's scratch pool holds
    /// ([`scratch::pages_within`]).
    pub fn scratch_pages_within(&self, bytes: u64) -> u32 {
        u32::try_from(scratch::pages_within(&self.config.layout, bytes)).unwrap_or(u32::MAX)
    }

    /// Opens the scratch page set of shadow forwards, `pages` pages: taken from the scratch
    /// pool of `allocator` and prepared on `stream` for a cache on a device, and with `None`
    /// a set of meta tensors for a cache that holds none.
    ///
    /// # Errors
    ///
    /// A set opened twice, a set of no pages, a device stated for a cache that holds none or
    /// the reverse, or a pool that cannot hold the set.
    pub fn open_scratch(
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
        self.scratch = Some(match device {
            Some((allocator, stream)) => ScratchPages::open(layout, pages, allocator, stream)?,
            None => ScratchPages::meta(layout, pages)?,
        });
        Ok(())
    }

    /// The scratch page set, or the refusal of a cache that opened none.
    fn scratch(&self) -> Result<&ScratchPages> {
        self.scratch.as_ref().ok_or(Error::State(
            "the cache's scratch pages are not open: a shadow forward has no pages to write",
        ))
    }

    /// The pages the scratch set holds over all of a shadow forward's rows; zero before it is
    /// opened.
    pub fn scratch_capacity(&self) -> u32 {
        self.scratch.as_ref().map_or(0, ScratchPages::pages)
    }

    /// The planes of attention group `group`'s layers over the scratch pages, in the shape
    /// [`Self::planes`] has with the set's pages in place of the pool's units.
    ///
    /// # Errors
    ///
    /// A cache whose scratch pages are not open, or a group that is no attention group.
    pub fn scratch_planes(&self, group: usize) -> Result<Vec<KvPlanes>> {
        let layout = self.attention(group)?;
        let scratch = self.scratch()?;

        (0..layout.layers.len())
            .map(|layer| scratch.layer(layout, layer))
            .collect()
    }

    /// The scratch set's page table for a shadow forward of `rows` rows of `pages` pages each,
    /// `[rows, pages]` U32, for every attention group: row `r` names the scratch pages
    /// `r · pages .. (r + 1) · pages`, which the runner numbers the row's slots in.
    ///
    /// # Errors
    ///
    /// A cache whose scratch pages are not open, or rows that need more pages than the set
    /// holds.
    pub fn scratch_table(&self, rows: u32, pages: u32) -> Result<Tensor> {
        self.scratch()?.table(rows, pages)
    }

    /// Copies what a batch's rows have cached in attention group `group` into their scratch
    /// pages on `stream`, before a shadow forward of `pages` pages a row: `slots` are the rows'
    /// request slots (`nn::NO_SLOT` for a padding row) and `lengths` the keys each row reads. A
    /// row of no keys, or a padding row, copies nothing.
    ///
    /// # Errors
    ///
    /// A cache that holds no device or whose scratch pages are not open; a row whose slot
    /// holds no table of the group, or whose keys need more than `pages` pages.
    pub fn copy_history(
        &self,
        group: usize,
        slots: &[u32],
        lengths: &[u32],
        pages: u32,
        stream: &Stream,
    ) -> Result<()> {
        let layout = self.attention(group)?;
        let scratch = self.scratch()?;
        if slots.len() != lengths.len() {
            return Err(Error::Invalid(format!(
                "a shadow forward names {} slots and {} lengths",
                slots.len(),
                lengths.len()
            )));
        }

        let table = self.tables.host_table(group)?;
        let width = self.tables.width() as usize;
        let mut history = Vec::with_capacity(slots.len());
        for (slot, length) in slots.iter().zip(lengths) {
            if *slot == nn::NO_SLOT || *length == 0 {
                history.push(Vec::new());
                continue;
            }
            if self.tables.installed(*slot, group as u32).is_none() {
                return Err(Error::Invalid(format!(
                    "request slot {slot} holds no page table of cache group {group}"
                )));
            }

            let held = length.div_ceil(layout.block_size) as usize;
            if held > pages as usize || held > width {
                return Err(Error::Invalid(format!(
                    "a row of {length} keys needs {held} pages of a shadow forward of {pages} \
                     pages a row"
                )));
            }
            let row = self.tables.row(*slot)? as usize;
            history.push(table[row * width..row * width + held].to_vec());
        }

        scratch.copy_history(&self.planes, layout, &history, pages, stream)
    }

    /// The state group's checkpoint planes, one tensor per field run of a state page:
    /// `[columns · units, plane_bytes]` U8, plane `(column, unit)` at row
    /// `column · units + unit`. The copies of recurrent state read and write state pages
    /// through them. Empty for a cache without a state group.
    pub(crate) fn state_runs(&self) -> Result<Vec<Tensor>> {
        self.planes.state_runs()
    }
}

#[cfg(test)]
mod tests {
    use ipc::{PrefixCacheDtype, UnitId};
    use nn::family::{StateField, StateShape};

    use super::*;

    /// A model of two attention layers of 4-token pages of one 4-wide head, a recurrent state
    /// of two layers and a drafter of one layer, over sixteen units, two request slots and
    /// rows of four pages.
    fn config() -> KvCacheConfig {
        let state = StateShape {
            layers: 2,
            fields: vec![StateField {
                name: "state",
                dtype: DType::U8,
                dims: vec![40],
            }],
        };
        let layout = CacheLayout::hybrid(4, 2, 1, 4, PrefixCacheDtype::Float8E4m3Fn, &state)
            .and_then(|layout| layout.with_draft(1, 1, 4))
            .expect("a hybrid layout with a draft group");

        KvCacheConfig {
            layout: Arc::new(layout),
            units: 16,
            slots: 2,
            width: 4,
            segments: None,
            workspace_bytes: 1 << 10,
        }
    }

    /// A model of two attention groups of two layers each: full attention and a sliding
    /// window.
    fn two_groups() -> KvCacheConfig {
        let mut layout = CacheLayout::dense(4, 2, 1, 4, PrefixCacheDtype::BFloat16);
        let mut window = layout.groups[0].clone();
        window.kind = ipc::CacheGroupKind::SlidingWindow { window: 8 };
        window.layer_ids = vec![2, 3];
        layout.groups.push(window);

        KvCacheConfig {
            layout: Arc::new(layout.validated().expect("two groups of two layers")),
            units: 16,
            slots: 2,
            width: 4,
            segments: None,
            workspace_bytes: 1 << 10,
        }
    }

    fn table(slot: u32, group: u32, units: &[u32], tokens: u32) -> BlockTable {
        BlockTable {
            request_pool_idx: slot,
            group_id: group,
            start_page: 0,
            unit_ids: units.iter().copied().map(UnitId).collect(),
            allocated_tokens: tokens,
        }
    }

    /// A cache that holds no device lends meta tensors of the shapes and element types the
    /// cache on a device has.
    #[test]
    fn a_cache_with_no_device_lends_tensors_of_its_shapes() -> Result<()> {
        let cache = KvCacheMemory::meta(config())?;
        assert_eq!(cache.device().location(), backend::meta().location());

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
        cache.install(&[table(1, 0, &[3, 4], 8), table(1, 1, &[9], 4)], None)?;
        assert_eq!(cache.host_table(0)?[..4], [3, 4, 0, 0]);
        assert_eq!(cache.host_table(1)?[..4], [9, 0, 0, 0]);
        Ok(())
    }

    /// The pooled table's last row names only the padding unit, whatever is installed, and an
    /// installed table is the row its slot gathers.
    #[test]
    fn the_padding_row_names_only_the_padding_unit() -> Result<()> {
        let mut cache = KvCacheMemory::meta(config())?;
        assert_eq!(cache.padding_row(), 2);
        assert_eq!((cache.table_row(1)?, cache.table_row(2)?), (0, 1));
        assert!(cache.table_row(0).is_err() && cache.table_row(3).is_err());

        cache.install(&[table(1, 0, &[5, 6, 7], 12), table(2, 0, &[8], 4)], None)?;

        let rows = cache.host_table(0)?;
        let row = |row: u32| &rows[row as usize * 4..(row as usize + 1) * 4];
        assert_eq!(row(cache.table_row(1)?), [5, 6, 7, 0]);
        assert_eq!(row(cache.table_row(2)?), [8, 0, 0, 0]);
        assert_eq!(row(cache.padding_row()), [0, 0, 0, 0]);
        assert!(rows.iter().all(|unit| *unit != nn::NO_SLOT));

        assert_eq!(cache.installed_pages(1, 0), Some((0, &[5, 6, 7][..])));
        cache.release_slots(&[1], None)?;
        assert_eq!(cache.installed_pages(1, 0), None);
        assert_eq!(cache.host_table(0)?[..4], [0, 0, 0, 0]);
        Ok(())
    }

    /// A unit allocation is checked whole before anything is applied, and each refusal names
    /// the unit and the reason.
    #[test]
    fn a_unit_allocation_is_refused_by_unit_and_reason() -> Result<()> {
        let mut cache = KvCacheMemory::meta(two_groups())?;
        let allocation = |group: u32, units: &[u32], copies: &[(u32, u32)]| CacheUnitAllocation {
            request_pool_idx: 1,
            group_id: group,
            unit_ids: units.iter().copied().map(UnitId).collect(),
            copies: copies
                .iter()
                .map(|(from, to)| (UnitId(*from), UnitId(*to)))
                .collect(),
        };
        let mut refused = |allocation: CacheUnitAllocation| {
            cache
                .apply_units(&[allocation], None)
                .expect_err("the allocation is refused")
                .to_string()
        };

        assert!(refused(allocation(2, &[1], &[])).contains("cache group 2"));
        assert!(refused(allocation(0, &[0], &[])).contains("padding unit"));
        assert!(refused(allocation(0, &[16], &[])).contains("cache unit 16"));
        assert!(refused(allocation(0, &[2], &[(1, 3)])).contains("did not name"));
        assert!(refused(allocation(0, &[2], &[(2, 2)])).contains("the destination is the source"));

        // A valid allocation applies; with no device nothing is issued.
        cache.apply_units(&[allocation(0, &[2, 3], &[(1, 2)])], None)?;

        // A unit copy does not move a state group's planes.
        let mut hybrid = KvCacheMemory::meta(config())?;
        let copy = hybrid
            .apply_units(&[allocation(0, &[2], &[(1, 2)])], None)
            .expect_err("a unit copy beside a state group");
        assert!(copy.to_string().contains("state planes"), "{copy}");
        hybrid.apply_units(&[allocation(0, &[2], &[])], None)?;
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
        assert!(cache.offload_pages(&[]).is_ok());
        assert!(cache.release_segments(0, &local).is_err());
        // A sleep of planes that are not split gives back nothing.
        cache.sleep(&local)?;

        let copy = PageCopy {
            rank: 0,
            pool: 1,
            group_id: 0,
            host_page: 0,
            units: vec![UnitId(1)],
            staged: false,
        };
        let refused = cache
            .offload_pages(&[copy])
            .expect_err("no host tier is mapped");
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
        assert_eq!(cache.scratch_table(3, 4)?.dims(), [3, 4]);
        assert!(cache.scratch_table(4, 4).is_err());
        Ok(())
    }

    /// A runner is built on one thread and moved to the executor thread with the cache and the
    /// recurrent state it holds.
    #[test]
    fn the_cache_and_the_state_move_to_the_executor_thread() {
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

    /// A cache on a device takes its planes from the owner, lends tensors on its GPU and
    /// mirrors an installed row on the device.
    #[test]
    #[ignore = "requires a GPU"]
    fn a_cache_on_a_device_lends_its_planes_and_installs_rows() -> Result<()> {
        let device = DeviceId(0);
        let stream = Stream::new(device)?;
        let mut cache = KvCacheMemory::open(
            two_groups(),
            device,
            &crate::memory::exclusive::LocalStorage,
            &stream,
        )?;
        assert_ne!(cache.device().location(), backend::meta().location());
        assert!(cache.mapped_bytes() > 0);
        assert_eq!(cache.planes(0)?[1].key.dims(), [16, 4, 1, 4]);

        cache.install(&[table(1, 0, &[3, 4], 8)], Some(&stream))?;
        cache.apply_units(
            &[CacheUnitAllocation {
                request_pool_idx: 1,
                group_id: 0,
                unit_ids: vec![UnitId(3), UnitId(4)],
                copies: vec![(UnitId(1), UnitId(3))],
            }],
            Some(&stream),
        )?;
        stream.synchronize()?;

        // The device row is the host mirror's.
        let table = cache.page_table(0)?;
        let mut device_row = [0_u32; 4];
        // SAFETY: the row is the first four entries of the pooled table, which the cache owns
        // and nothing writes after the synchronized install; the host array holds four.
        unsafe {
            platform::memcpy_async(
                device_row.as_mut_ptr() as u64,
                backend::operand(&table)?.ptr.address(),
                16,
                &stream,
            )?;
        }
        stream.synchronize()?;
        assert_eq!(device_row, cache.host_table(0)?[..4]);
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
        };
        let mut cache = KvCacheMemory::open(config, device, &local, &stream)?;
        let held = cache.held_segments();
        let mapped = cache.mapped_bytes();
        assert!(held > 0);

        cache.sleep(&local)?;
        assert_eq!(cache.held_segments(), 0);
        assert!(cache.mapped_bytes() < mapped);

        cache.wake(&local, &stream, Some(1))?;
        assert_eq!(cache.held_segments(), 1);
        cache.wake(&local, &stream, None)?;
        assert_eq!(
            (cache.held_segments(), cache.mapped_bytes()),
            (held, mapped)
        );
        Ok(())
    }
}

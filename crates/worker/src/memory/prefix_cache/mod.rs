//! The rank's prefix cache: unit pool, block tables, accesses and imports.
//!
//! [`PrefixCache`] composes the owners the executor drives per batch:
//!
//! - [`CachePlanes`]: the pool's device storage;
//! - [`BlockTables`] (host) and [`BlockTableStorage`] (device): the
//!   scheduler's page assignments and their numerical tables;
//! - [`TransferBases`]: resident transfers and the bases of incremental
//!   transfers;
//! - [`UnitAccesses`]: the physical accesses (executions, exports, imports)
//!   that decide unit reuse;
//! - the import lane: bounded workspaces whose host tasks copy other ranks'
//!   exports into assigned pages;
//! - [`PageTier`]: the host page pools the engine copies pages through, on
//!   the rank's copy lane ([`PrefixCache::issue_pages`]).
//!
//! Every device write runs on a stream the caller supplies; nothing reads a
//! "current" stream. All validation precedes the first device write, and
//! failed or unknown device completions keep their storage retained.

mod access;
pub(super) mod block_tables;
mod imports;
pub(crate) mod pages;
pub(super) mod plan;
pub(super) mod planes;
mod transfers;

pub(crate) use access::UnitAccesses;
use imports::Imports;
pub use imports::PendingImport;
pub(crate) use transfers::TransferBases;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use cuda::{DeviceId, Event, EventFlags, Stream};
use ipc::CallId;
use ipc::{BlockTable, BufferId, CacheTransfer, PageCopy, RequestKey, Wake};
use tvm_ffi::Any;

use self::block_tables::{AttentionRow, BlockTableStorage, BlockTables, GroupTable, TablePages};
pub(crate) use self::pages::PageTier;
use self::plan::{CACHE, CachePlan};
use self::planes::{CachePlanes, UnitColumn};
use super::allocation::StorageView;
use crate::ffi::{Handle, Interpreter};
use crate::host::HostLane;
use crate::lane::Job;
use crate::stream::Issue;
use crate::transfer::cache_export::{self, CacheExport};
use crate::transfer::exports::Location;
use crate::transfer::{
    CacheImport, CacheImporter, ImportCopy, ImportEntry, ImportWorkspace, Retired, Retirement,
    Transports,
};
use crate::{Error, HostTask, Result, lock};

/// The most import copy threads (and workspaces) a rank runs.
const IMPORT_WORKERS: usize = 4;

/// The host page tier and the copy lane its copies run on: the lane's
/// stream and its unit column.
struct PageLane {
    tier: PageTier,
    stream: Stream,
    column: UnitColumn,
}

/// A batch's page copies, recorded for the copy lane
/// ([`PrefixCache::issue_pages`]).
pub(crate) struct IssuedPages {
    /// The job to submit to the copy lane.
    pub(crate) job: Job,
    /// The device of the cache, whose streams record the fences.
    pub(crate) device: DeviceId,
    /// Recorded behind the reads: the batch's forward waits on it.
    pub(crate) reads: Option<Arc<Event>>,
    /// Recorded behind the writes: the batch's output waits on it.
    pub(crate) writes: Option<Arc<Event>>,
}

/// The rank's prefix cache owners.
pub struct PrefixCache {
    planes: Arc<CachePlanes>,
    storage: BlockTableStorage,
    tables: BlockTables,
    accesses: UnitAccesses,
    bases: TransferBases,
    imports: Imports,
    /// The host page tier and the copy lane its copies run on, when the
    /// engine copies pages through it.
    pages: Option<PageLane>,
    /// The most units warmup lent as scratch (`1..=` this), which the rank
    /// reports so that a standby worker's warmup stays inside them.
    scratch_lent: u32,
}

impl PrefixCache {
    /// Import workers for `capacity` concurrently admitted imports.
    pub(crate) fn import_workers(capacity: usize) -> usize {
        capacity.clamp(1, IMPORT_WORKERS)
    }

    /// Fixed device bytes besides the pool itself: block tables, import
    /// scratch and per-stream unit columns (`unit_columns` streams besides
    /// the import workers').
    pub(crate) fn fixed_bytes(
        plan: &CachePlan,
        num_units: u32,
        request_pool_size: u32,
        width: u32,
        import_capacity: usize,
        unit_columns: usize,
    ) -> Result<u64> {
        let workers = Self::import_workers(import_capacity) as u64;
        let (elements, scales) = plan.page_extents();
        Ok(
            BlockTableStorage::bytes(&plan.group_shapes()?, request_pool_size, width)
                + workers * ImportWorkspace::scratch_bytes(elements, scales)
                + (workers + unit_columns as u64) * UnitColumn::bytes(num_units),
        )
    }

    /// Allocate the pool, the tables and the import workspaces on `device`
    /// (host memory when `None`), resetting device storage on `stream`.
    ///
    /// Every import workspace constructs the library's unit resets and
    /// page conversion on its own stream.
    #[expect(
        clippy::too_many_arguments,
        reason = "a pool's shape, its device and stream, and where its planes are taken"
    )]
    pub fn new(
        plan: Arc<CachePlan>,
        num_units: u32,
        request_pool_size: u32,
        width: u32,
        import_capacity: usize,
        device: Option<DeviceId>,
        stream: Option<&Stream>,
        memory: &mut dyn crate::memory::retained::ExclusiveMemory,
    ) -> Result<Self> {
        let shapes = plan.group_shapes()?;
        let planes = Arc::new(CachePlanes::allocate(
            plan, device, num_units, stream, memory,
        )?);
        let storage =
            BlockTableStorage::allocate(&shapes, request_pool_size, width, device, stream)?;
        let tables = BlockTables::new(shapes, request_pool_size, width as usize, num_units)?;

        let workers = Self::import_workers(import_capacity);
        let workspaces = (0..workers)
            .map(|_| ImportWorkspace::bind(&planes).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        let importer = Arc::new(CacheImporter::new(workspaces));
        let lane = HostLane::new(import_capacity.max(1), workers, "worker-cache-import")?;
        let retired = Arc::new(Retired {
            importer: Arc::downgrade(&importer),
            wake: Mutex::new(None),
        });

        Ok(Self {
            planes,
            storage,
            tables,
            accesses: UnitAccesses::default(),
            bases: TransferBases::default(),
            imports: Imports {
                importer,
                lane,
                retired,
            },
            pages: None,
            scratch_lent: 0,
        })
    }

    /// Copy pages through `tier` from now on, on the copy lane's `stream`.
    pub(crate) fn set_page_tier(&mut self, tier: PageTier, stream: Stream) -> Result<()> {
        let column = UnitColumn::bind(&self.planes, Some(&stream))?;
        self.pages = Some(PageLane {
            tier,
            stream,
            column,
        });
        Ok(())
    }

    /// Record a batch's page copies on the copy lane, to be issued as soon
    /// as the batch arrives: fill the units of `reads` from their slots, and
    /// copy the published units of `writes` into theirs.
    ///
    /// The reads fill new units of the batch, which are idle (no in-flight
    /// export, import or execution addresses them, checked here) and which
    /// the reads overwrite whole, so the batch does not recycle them; its
    /// forward waits on the returned read fence. The writes copy pages the
    /// engine published once it accepted the result of the call that computed
    /// them, so the forward that produced them completed before this batch
    /// was sent, and the engine holds their units until the batch's output
    /// commits them; that output waits on the returned write fence, and
    /// nothing else does.
    ///
    /// Fails for a rank that maps no page pool, or when a read's unit is
    /// in use.
    pub(crate) fn issue_pages(
        &mut self,
        reads: &[PageCopy],
        writes: &[PageCopy],
    ) -> Result<IssuedPages> {
        let lane = self.pages.as_ref().ok_or_else(|| {
            Error::Invalid("a batch copies pages but the rank maps no page pool".into())
        })?;
        let device = self
            .planes
            .device()
            .ok_or(Error::Invariant("a page pool maps a device cache".into()))?;
        let unit_tokens = self.planes.plan().unit_tokens();
        let spans: Vec<_> = reads
            .iter()
            .flat_map(|copy| copy.units.iter())
            .map(|unit| (unit.0, 0, unit_tokens))
            .collect();
        self.accesses.require_reusable(&spans)?;

        let job = Job::new();
        let issue = job.on(Some(&lane.stream));
        let fence = |issue: crate::lane::Recorder<'_>| -> Result<Arc<Event>> {
            let event = Arc::new(Event::new(EventFlags::default()));
            issue.record(&event)?;
            Ok(event)
        };
        let reads_fence = if reads.is_empty() {
            None
        } else {
            lane.tier.read(&self.planes, reads, &lane.column, issue)?;
            Some(fence(issue)?)
        };
        let writes_fence = if writes.is_empty() {
            None
        } else {
            lane.tier.write(&self.planes, writes, issue)?;
            Some(fence(issue)?)
        };
        Ok(IssuedPages {
            job,
            device,
            reads: reads_fence,
            writes: writes_fence,
        })
    }

    pub(crate) fn planes(&self) -> &Arc<CachePlanes> {
        &self.planes
    }

    pub(crate) fn plan(&self) -> &Arc<CachePlan> {
        self.planes.plan()
    }

    pub(crate) fn storage(&self) -> &BlockTableStorage {
        &self.storage
    }

    pub fn tables(&self) -> &BlockTables {
        &self.tables
    }

    /// Select `rows`' attention pages. Every row that writes is checked
    /// against the live exports, imports and executions of its units.
    pub(crate) fn prepare_attention(&mut self, rows: &[AttentionRow]) -> Result<Vec<TablePages>> {
        let accesses = &mut self.accesses;
        self.tables.prepare_attention(rows, |table, prefix, query| {
            accesses.require_writable(&table.spans(u64::from(prefix), u64::from(query))?)
        })
    }

    /// Retain `slot`'s extent below `length` tokens, from each group's first
    /// resident page, for `request` until `retirement`: the physical end of
    /// the batch that reads or writes it.
    pub(crate) fn retain(
        &mut self,
        request: RequestKey,
        slot: u32,
        length: u64,
        retirement: &Arc<Retirement>,
    ) -> Result<()> {
        let mut spans = Vec::new();
        for group in 0..self.tables.groups().len() as u32 {
            let table = self.tables.table(slot, group)?;
            let start = u64::from(table.start_page) * u64::from(table.shape.block_size);
            if length > start {
                spans.extend(table.spans(start, length - start)?);
            }
        }
        self.accesses
            .retain_execution(request, &spans, Arc::clone(retirement));
        Ok(())
    }

    /// The transfer resident under `buffer`.
    pub(crate) fn resident(&self, buffer: BufferId) -> Option<&Arc<CacheTransfer>> {
        self.bases.resident(buffer)
    }

    /// The conditioning export `buffer` of `request`, covered by `slot`'s
    /// installed keys and values (`visible` tokens).
    pub(crate) fn validate_conditioning(
        &self,
        request: RequestKey,
        buffer: BufferId,
        slot: u32,
        visible: u64,
    ) -> Result<&Arc<CacheTransfer>> {
        self.bases
            .validate_conditioning(request, buffer, slot, visible, &self.tables)
    }

    /// Check a batch's exports and installs against their destinations'
    /// accepted bases, before the batch's commit point.
    pub fn validate_transfers(
        &self,
        exports: &[(BufferId, CacheTransfer)],
        installations: &[(BufferId, BufferId, CacheTransfer)],
    ) -> Result<()> {
        self.bases.validate_exports(exports, installations)
    }

    /// Make a checked batch's exports and installs resident.
    pub fn apply_transfers(
        &mut self,
        exports: Vec<(BufferId, CacheTransfer)>,
        installations: Vec<(BufferId, BufferId, CacheTransfer)>,
    ) {
        self.bases.apply_exports(exports, installations);
    }

    /// The transfer and locations of `export`, once the work it was issued
    /// with was performed (`CacheExport::locate`).
    pub fn locate_export(&mut self, export: CacheExport) -> Result<(CacheTransfer, Vec<Location>)> {
        export.locate(&mut self.accesses)
    }

    /// Revoke `export`, which is not published ([`CacheExport::revoke`]).
    pub(crate) fn revoke_export(&mut self, export: CacheExport) {
        export.revoke(&mut self.accesses);
    }

    /// Whether any physical access is outstanding.
    pub(crate) fn has_pending_accesses(&mut self) -> bool {
        self.accesses.has_pending_accesses()
    }

    /// The completions a batch writing `spans` waits for.
    pub(crate) fn write_dependencies(&mut self, spans: &[(u32, u32, u32)]) -> Vec<Arc<Retirement>> {
        self.accesses
            .write_dependencies(spans)
            .into_iter()
            .map(Arc::clone)
            .collect()
    }

    /// Release the physical exports of `buffers`.
    pub fn release_exports(&mut self, buffers: &[BufferId]) {
        self.accesses.release_exports(buffers);
    }

    /// Release the resident transfers of finished calls and their exports.
    pub fn release_calls(&mut self, calls: &[(RequestKey, CallId)]) {
        let released = self.bases.release_calls(calls);
        self.accesses.release_exports(&released);
    }

    /// Release the imports and exports of `buffers`.
    pub(crate) fn release_buffers(&mut self, buffers: &HashSet<BufferId>) -> Result<()> {
        self.release_imports(buffers)?;
        self.accesses
            .release_exports(&buffers.iter().copied().collect::<Vec<_>>());
        Ok(())
    }

    /// Release the imports and exports of finished `requests`, except the
    /// `retained` buffers.
    pub(crate) fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<()> {
        self.cancel_request_imports(requests, retained)?;
        let exports: Vec<BufferId> = self
            .accesses
            .exported_buffers()
            .into_iter()
            .filter(|buffer| requests.contains(&buffer.owner) && !retained.contains(buffer))
            .collect();
        self.accesses.release_exports(&exports);
        Ok(())
    }

    /// Whether the physical accesses of `buffers` and of `requests` (but
    /// the `retained` buffers) have retired.
    pub(crate) fn retirement_ready(
        &mut self,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> bool {
        self.accesses.retirement_ready(buffers, requests, retained)
    }

    /// Keep `slot`, a guidance prefix slot `request` reads, until the
    /// request releases its prefixes.
    pub fn retain_prefix(&mut self, request: RequestKey, slot: u32) {
        self.tables.retain_prefix(request, slot);
    }

    /// Forget a retired request epoch's resident transfers and accepted
    /// bases; its physical accesses have drained.
    pub(crate) fn drop_request(&mut self, request: RequestKey) {
        self.bases.drop_request(request);
    }

    /// Every bound tensor of the cache: the planes and the block tables,
    /// keyed as numerical bindings receive them.
    pub fn views(&self) -> crate::Result<Vec<(String, StorageView)>> {
        let mut views = self.planes.views();
        views.extend(self.storage.views());
        Ok(views)
    }

    /// The attention layers' view of the unit pool (the library's `pool`
    /// over the plan, the unit count and the unit fields by name), which
    /// execution contexts read prefixes through.
    pub(crate) fn pool(&self) -> Result<Handle> {
        let mut arguments = vec![
            self.plan().handle.to_any(),
            Any::from(i64::from(self.planes.num_units())),
        ];
        arguments.extend(self.planes.field_arguments()?);
        Handle::new(Interpreter::import(&format!("{CACHE}.pool"))?.call_values(&arguments)?)
    }

    /// Bind the unit resets to another stream (one per executing lane).
    pub fn bind_units(&self, stream: Option<&Stream>) -> Result<UnitColumn> {
        UnitColumn::bind(&self.planes, stream)
    }

    /// Install scheduler page assignments: validate them all, write the
    /// changed rows on `stream`, then commit them on the host.
    pub fn install(&mut self, assignments: &[BlockTable], stream: impl Issue) -> Result<()> {
        let update = self.tables.prepare(assignments)?;
        self.storage.install(&update, stream)?;
        self.tables.commit(update);
        Ok(())
    }

    /// Prepare newly assigned units for a new owner on `column`'s stream.
    /// Fails if an export, import or execution still overlaps any of them,
    /// or a unit lies outside the pool (`UnitColumn::recycle`).
    pub fn recycle(
        &mut self,
        units: &[u32],
        column: &UnitColumn,
        stream: impl Issue,
    ) -> Result<()> {
        let unit_tokens = self.plan().unit_tokens();
        let spans: Vec<_> = units.iter().map(|&unit| (unit, 0, unit_tokens)).collect();
        self.accesses.require_reusable(&spans)?;
        column.recycle(&self.planes, units, stream)
    }

    /// Clear released slots' rows on `stream` (ordered after every batch
    /// that read them) and forget them on the host.
    pub fn release_slots(&mut self, slots: &[u32], stream: impl Issue) -> Result<()> {
        let slots = self.tables.release_slots(slots)?;
        self.storage.clear(&slots, stream)?;
        self.tables.release(&slots);
        Ok(())
    }

    /// Clear guidance prefix slots `slots` of `request` on `stream`
    /// (ordered after every batch that read them) and forget them.
    pub(crate) fn release_prefixes(
        &mut self,
        request: RequestKey,
        slots: &[u32],
        stream: impl Issue,
    ) -> Result<()> {
        let slots = self.tables.release_slots(slots)?;
        self.storage.clear(&slots, stream)?;
        self.tables.release_prefixes(request, &slots);
        Ok(())
    }

    /// Clear every guidance prefix slot `request` still holds.
    pub fn release_request_prefixes(
        &mut self,
        request: RequestKey,
        stream: impl Issue,
    ) -> Result<()> {
        let slots = self.tables.prefix_slots(request);
        if slots.is_empty() {
            return Ok(());
        }
        self.release_prefixes(request, &slots, stream)
    }

    /// Export a slot's visible cache extent (see `cache_export::export`). Its
    /// transfer and locations exist once the issuer's work was performed
    /// ([`Self::locate_export`]).
    #[expect(clippy::too_many_arguments, reason = "one export's complete inputs")]
    pub fn export(
        &mut self,
        slot: u32,
        visible: u32,
        destination: &str,
        buffer: BufferId,
        transports: &Transports,
        consumers: &[usize],
        stream: impl Issue,
    ) -> Result<CacheExport> {
        cache_export::export(
            &mut self.accesses,
            &self.bases,
            &self.tables,
            &self.planes,
            slot,
            visible,
            destination,
            buffer,
            transports,
            consumers,
            stream,
        )
    }

    /// Reserve an import of `export` into `slot`'s `tables` (the batch's
    /// view of the slot, see [`BlockTables::for_batch`]) and submit its copy.
    /// `initialized` are the newly assigned units the import resets itself.
    pub fn reserve_import(
        &mut self,
        export: Arc<CacheTransfer>,
        slot: u32,
        tables: Vec<Arc<GroupTable>>,
        initialized: Vec<u32>,
        transports: Arc<Transports>,
    ) -> Result<PendingImport> {
        self.bases.validate_install(&export)?;
        let plan = self.planes.plan();
        for (group, target) in export.groups.iter().zip(&plan.groups) {
            let expected = [
                u64::from(export.exported_extent.saturating_sub(group.start)),
                u64::from(target.axis.total),
                u64::from(target.total_kv_heads),
                u64::from(target.head_dim),
            ];
            if group
                .tensors
                .first()
                .is_some_and(|tensor| tensor.shape != expected)
            {
                return Err(Error::Invalid(
                    "cache transfer shape does not match destination layers".into(),
                ));
            }
        }

        let ranges = self
            .tables
            .import_spans(slot, &tables, &export, &initialized)?;
        let copy =
            !initialized.is_empty() || export.groups.iter().any(|group| !group.tensors.is_empty());
        let task = copy.then(|| self.imports.lane.reserve()).transpose()?;
        let write = Arc::new(CacheImport::new(export.source, slot as usize, copy));
        let entry = ImportEntry::new(Arc::clone(&write));

        let withdraw = |task: &Option<Arc<HostTask<ImportCopy>>>| {
            if let Some(task) = task
                && let Err(cleanup) = task.cancel(true)
            {
                tracing::warn!("failed to withdraw a KV import task: {cleanup}");
            }
        };
        if let Some(task) = &task {
            let configured = task.configure(ImportCopy {
                write: Arc::clone(&write),
                export: Arc::clone(&export),
                tables: tables.clone(),
                initialized: initialized.clone(),
                planes: Arc::clone(&self.planes),
                transports,
                importer: Arc::clone(&self.imports.importer),
                retired: Arc::clone(&self.imports.retired),
            });
            if let Err(error) = configured {
                withdraw(&Some(Arc::clone(task)));
                return Err(error);
            }
        }

        let accesses = &mut self.accesses;
        let retirement = Arc::clone(entry.retirement());
        if let Err(error) = self.imports.importer.reserve(entry.clone(), || {
            accesses.reserve_import(export.source, &ranges, retirement)
        }) {
            withdraw(&task);
            return Err(error);
        }

        if let Some(task) = &task
            && let Err(error) = task.submit()
        {
            withdraw(&Some(Arc::clone(task)));
            self.imports.cancel(self.imports.importer.abandon(&write))?;
            return Err(error);
        }
        Ok(PendingImport {
            entry,
            tables,
            export,
            task,
        })
    }

    /// Accept a finished import as the slot's installed prefix under
    /// `installed`: recheck the accepted base and the slot's tables (another
    /// import may have advanced them while the copy ran), then publish the
    /// imported length in `stream`'s order.
    pub fn adopt_import(
        &mut self,
        import: &PendingImport,
        installed: BufferId,
        stream: impl Issue,
    ) -> Result<Arc<CacheTransfer>> {
        if installed.owner != import.export.source.owner {
            return Err(Error::Invalid(
                "installed cache buffer does not belong to its source".into(),
            ));
        }
        self.bases.validate_install(&import.export)?;
        let slot = import.slot() as u32;
        for (group, expected) in import.tables.iter().enumerate() {
            if self.tables.table(slot, group as u32)? != *expected {
                return Err(Error::Invalid(
                    "cache installation scheduler block table changed".into(),
                ));
            }
        }
        if !import.done() {
            return Err(Error::State(
                "cache import was observed before input readiness",
            ));
        }
        import.result()?;
        self.imports.importer.adopt(import.entry.write())?;
        self.storage
            .set_verified(slot, import.export.exported_extent, stream)?;
        self.imports.retired.reap()?;
        Ok(Arc::clone(&import.export))
    }

    /// Give up an import: cancel its reads and release its destination once
    /// its physical accesses retire.
    pub fn abandon_import(&mut self, import: &PendingImport) -> Result<()> {
        if let Some(task) = &import.task {
            task.cancel(false)?;
        }
        self.imports
            .cancel(self.imports.importer.abandon(import.entry.write()))
    }

    /// Cancel imports into released buffers.
    pub(crate) fn release_imports(&mut self, buffers: &HashSet<BufferId>) -> Result<()> {
        self.imports.cancel(self.imports.importer.release(buffers))
    }

    /// Cancel retired requests' imports, except retained buffers.
    pub(crate) fn cancel_request_imports(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<()> {
        self.imports
            .cancel(self.imports.importer.cancel_requests(requests, retained))
    }

    /// Notify retired imports' observers.
    pub fn reap_imports(&self) -> Result<()> {
        self.imports.retired.reap()
    }

    /// Install the service wake fired when imports finish or retire.
    pub(crate) fn set_wake(&self, wake: Option<Wake>) {
        self.imports.lane.set_wake(wake.clone());
        *lock(&self.imports.retired.wake) = wake;
    }

    /// Lend scratch units to request slots `1..` as resident block tables
    /// of `lengths` tokens, issued to `stream`: the units are reset, then
    /// the tables installed. A startup forward over them stages its rows
    /// from the resident tables, as serving does, so what a graph is
    /// captured over is what serving stages.
    pub(crate) fn install_scratch(
        &mut self,
        lengths: &[usize],
        stream: impl Issue,
    ) -> Result<ScratchUnits> {
        let shapes = self.plan().group_shapes()?;
        // Whole pages of every group for each row, at least one.
        let units_of = |length: usize| -> usize {
            shapes
                .iter()
                .map(|shape| {
                    length.max(1).div_ceil(shape.block_size as usize)
                        * shape.units_per_page as usize
                })
                .sum()
        };
        let count: usize = lengths.iter().map(|&length| units_of(length)).sum();
        let scratch = self.scratch_units(count as u32, stream)?;
        let mut unit = *scratch.units().start();
        let mut tables = Vec::new();
        for (row, &length) in lengths.iter().enumerate() {
            for (group, shape) in shapes.iter().enumerate() {
                let pages = length.max(1).div_ceil(shape.block_size as usize);
                let units = pages * shape.units_per_page as usize;
                tables.push(BlockTable {
                    request_pool_idx: row as u32 + 1,
                    group_id: group as u32,
                    start_page: 0,
                    unit_ids: (unit..unit + units as u32).map(ipc::UnitId).collect(),
                    allocated_tokens: (pages * shape.block_size as usize) as u32,
                });
                unit += units as u32;
            }
        }
        self.install(&tables, stream)?;
        Ok(scratch)
    }

    /// Clear the scratch tables of slots `1..=rows` and return their units,
    /// issued to `stream`: the stream waits for what wrote the units, resets
    /// them and waits for the resets.
    pub(crate) fn release_scratch(
        &mut self,
        scratch: ScratchUnits,
        rows: usize,
        stream: impl Issue,
    ) -> Result<()> {
        let slots: Vec<u32> = (1..=rows as u32).collect();
        let cleared = self.release_slots(&slots, stream);
        let finished = scratch.finish(self, stream);
        cleared.and(finished)
    }

    /// Lend units `1..=count` as warmup scratch: no access may be pending,
    /// and the units are reset natively, issued to `stream`, now and when
    /// the scratch is returned.
    fn scratch_units(&mut self, count: u32, stream: impl Issue) -> Result<ScratchUnits> {
        if self.accesses.has_pending_accesses() {
            return Err(Error::State("cache scratch requires an idle cache"));
        }
        if count >= self.planes.num_units() {
            return Err(Error::Invalid(
                "cache scratch exceeds the pool's usable units".into(),
            ));
        }
        self.planes.zero_leading(count, stream)?;
        self.scratch_lent = self.scratch_lent.max(count);
        Ok(ScratchUnits { count })
    }

    /// The most units warmup lent as scratch: units `1..=` this.
    pub(crate) fn scratch_lent(&self) -> u32 {
        self.scratch_lent
    }

    /// Cancel every import and join the import copy threads, so no import
    /// reads through the transports any more. Idempotent.
    pub(crate) fn stop_imports(&mut self) -> Result<()> {
        let cancelled = self.imports.cancel(self.imports.importer.stop());
        let mut stopped = Ok(());
        for error in self.imports.lane.close() {
            stopped = stopped.and(Err(error));
        }
        cancelled.and(stopped).and(self.imports.retired.reap())
    }

    /// Stop imports, release exports and require every access retired.
    /// Exports retire once their transports revoked them, so the caller
    /// closes the transports first. Failed retirements keep their storage:
    /// the caller must not reuse it.
    pub fn close(&mut self) -> Result<()> {
        let stopped = self.stop_imports();
        let exports = self.accesses.exported_buffers();
        self.accesses.release_exports(&exports);
        stopped
            .and(self.imports.importer.require_retired())
            .and(self.accesses.require_retired())?;
        self.bases.clear();
        self.tables.clear();
        Ok(())
    }
}

/// Units `1..=count` lent to warmup batches.
pub(crate) struct ScratchUnits {
    count: u32,
}

impl ScratchUnits {
    pub(crate) fn units(&self) -> std::ops::RangeInclusive<u32> {
        1..=self.count
    }

    /// Wait for the warmup's writes on `stream`, reset the units again and
    /// wait for the resets, all issued to `stream`.
    pub(crate) fn finish(self, cache: &PrefixCache, stream: impl Issue) -> Result<()> {
        Issue::synchronize(&stream)?;
        cache.planes.zero_leading(self.count, stream)?;
        Issue::synchronize(&stream)
    }
}

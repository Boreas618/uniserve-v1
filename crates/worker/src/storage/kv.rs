//! The rank's KV cache: unit pool, block tables, accesses and imports.
//!
//! [`KVCache`] composes the owners the executor drives per batch:
//!
//! - [`KVPlanes`]: the pool's device storage;
//! - [`BlockTables`] (host) and [`BlockTableStorage`] (device): the
//!   scheduler's page assignments and their numerical tables;
//! - [`KVCacheManager`]: resident transfers, transfer bases and the physical
//!   accesses (executions, exports, imports) that decide unit reuse;
//! - the import lane: bounded workspaces whose host tasks copy other ranks'
//!   exports into assigned pages.
//!
//! Every device write runs on a stream the caller supplies; nothing reads a
//! "current" stream. All validation precedes the first device write, and
//! failed or unknown device completions keep their storage retained.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use uniserve_cuda::Stream;
use uniserve_ipc::{BlockTable, BufferId, KvTransfer, RequestKey, Wake};

use super::kv_plan::KVPlan;
use super::kv_planes::{KVPlanes, UnitColumn};
use super::memory::StorageView;
use super::{BlockTableStorage, BlockTables, GroupTable, KVCacheManager};
use crate::ffi::Function;
use crate::host::HostLane;
use crate::transfer::exports::Location;
use crate::transfer::{
    CacheImporter, ImportCopy, ImportEntry, ImportTask, ImportWorkspace, KVImport, KVImportBackend,
    Retired, Retirement, Transports, kv_export,
};
use crate::{Error, HostTask, Outcome, Result};

/// The most import copy threads (and workspaces) a rank runs.
const IMPORT_WORKERS: usize = 4;

/// A reserved KV import: the destination tables and units it resets, the
/// export it installs, and its copy task (none when nothing is copied).
pub struct PendingImport {
    entry: ImportEntry,
    tables: Vec<Arc<GroupTable>>,
    export: Arc<KvTransfer>,
    task: Option<Arc<HostTask<ImportTask>>>,
}

impl PendingImport {
    pub fn buffer(&self) -> BufferId {
        self.entry.buffer
    }

    pub fn slot(&self) -> usize {
        self.entry.request_pool_idx
    }

    pub fn export(&self) -> &Arc<KvTransfer> {
        &self.export
    }

    /// Whether the copy task finished (successfully or not).
    pub fn done(&self) -> bool {
        self.task.as_ref().is_none_or(|task| task.completion.done())
    }

    /// The import's physical retirement: it completes once the import was
    /// adopted or abandoned and every read and the workspace stream drained.
    pub fn retirement(&self) -> &Arc<Retirement> {
        self.entry.retirement()
    }

    /// Run `callback` once the copy task finished.
    pub fn subscribe(&self, callback: Box<dyn FnOnce() + Send>) {
        match &self.task {
            Some(task) => {
                if let Some(callback) = task.completion.subscribe(callback) {
                    callback();
                }
            }
            None => callback(),
        }
    }

    /// The copy's result, without waiting.
    fn result(&self) -> Result<()> {
        let Some(task) = &self.task else {
            return Ok(());
        };
        match task.completion.outcome() {
            Some(Outcome::Success(_)) => Ok(()),
            Some(Outcome::Failed(error)) => {
                Err(Error::Transport(format!("KV import failed: {error}")))
            }
            Some(Outcome::Cancelled) => Err(Error::Resource("KV import was cancelled")),
            None => Err(Error::State(
                "KV import was observed before input readiness",
            )),
        }
    }
}

/// Import workspaces, their copy lane and the retirement observers.
struct Imports {
    importer: Arc<CacheImporter>,
    lane: HostLane<ImportTask>,
    retired: Arc<Retired>,
}

impl Imports {
    fn cancel(&self, reads: Vec<Arc<crate::transfer::ReadRef>>) -> Result<()> {
        let mut result = Ok(());
        for read in reads {
            result = result.and(read.read().cancel());
        }
        result.and(self.retired.reap())
    }
}

/// The rank's KV cache owners.
pub struct KVCache {
    planes: Arc<KVPlanes>,
    storage: BlockTableStorage,
    tables: BlockTables,
    manager: KVCacheManager<Arc<Retirement>>,
    imports: Imports,
}

impl KVCache {
    /// Import workers for `capacity` concurrently admitted imports.
    pub fn import_workers(capacity: usize) -> usize {
        capacity.clamp(1, IMPORT_WORKERS)
    }

    /// Fixed device bytes besides the pool itself: block tables, import
    /// scratch and per-stream unit columns (`unit_columns` streams besides
    /// the import workers').
    pub fn fixed_bytes(
        plan: &KVPlan,
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
                + (workers + unit_columns as u64) * u64::from(num_units) * 8,
        )
    }

    /// Allocate the pool, the tables and the import workspaces on `device`
    /// (host memory when `None`), resetting device storage on `stream`.
    ///
    /// `bind_units` and `bind_conversion` are the interface's
    /// `cache.bind_units` and `cache.bind_conversion`; every import
    /// workspace binds both on its own stream.
    #[expect(
        clippy::too_many_arguments,
        reason = "the pool's complete startup facts"
    )]
    pub fn new(
        plan: Arc<KVPlan>,
        num_units: u32,
        request_pool_size: u32,
        width: u32,
        import_capacity: usize,
        device: Option<i32>,
        stream: Option<&Stream>,
        bind_units: &Function,
        bind_conversion: &Function,
    ) -> Result<Self> {
        let shapes = plan.group_shapes()?;
        let planes = Arc::new(KVPlanes::allocate(plan, device, num_units, stream)?);
        let storage =
            BlockTableStorage::allocate(&shapes, request_pool_size, width, device, stream)?;
        let tables = BlockTables::new(shapes, request_pool_size, width as usize, num_units)?;

        let workers = Self::import_workers(import_capacity);
        let workspaces = (0..workers)
            .map(|_| ImportWorkspace::bind(&planes, bind_units, bind_conversion).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        let importer = Arc::new(CacheImporter::new(workspaces));
        let lane = HostLane::new(import_capacity.max(1), workers, "worker-kv-import")?;
        let retired = Arc::new(Retired {
            importer: Arc::downgrade(&importer),
            wake: Mutex::new(None),
        });

        Ok(Self {
            planes,
            storage,
            tables,
            manager: KVCacheManager::default(),
            imports: Imports {
                importer,
                lane,
                retired,
            },
        })
    }

    pub fn planes(&self) -> &Arc<KVPlanes> {
        &self.planes
    }

    pub fn plan(&self) -> &Arc<KVPlan> {
        self.planes.plan()
    }

    pub fn storage(&self) -> &BlockTableStorage {
        &self.storage
    }

    pub fn tables(&self) -> &BlockTables {
        &self.tables
    }

    pub fn tables_mut(&mut self) -> &mut BlockTables {
        &mut self.tables
    }

    pub fn manager(&mut self) -> &mut KVCacheManager<Arc<Retirement>> {
        &mut self.manager
    }

    /// The block tables and the access manager together, for a batch that
    /// resolves attention pages while checking writes.
    pub fn split(&mut self) -> (&BlockTables, &mut KVCacheManager<Arc<Retirement>>) {
        (&self.tables, &mut self.manager)
    }

    /// Every bound tensor of the cache: the planes and the block tables,
    /// keyed as numerical bindings receive them.
    pub fn views(&self) -> crate::Result<Vec<(String, StorageView)>> {
        let mut views = self.planes.views()?;
        views.extend(self.storage.views());
        Ok(views)
    }

    /// The plan map with the unit count, as cache bindings receive it.
    pub fn layout(&self) -> Result<tvm_ffi::Map<tvm_ffi::String, tvm_ffi::Any>> {
        self.plan().layout(self.planes.num_units())
    }

    /// Bind the unit resets to another stream (one per executing lane).
    pub fn bind_units(&self, bind_units: &Function, stream: Option<&Stream>) -> Result<UnitColumn> {
        UnitColumn::bind(&self.planes, bind_units, stream)
    }

    /// Install scheduler page assignments: validate them all, write the
    /// changed rows on `stream`, then commit them on the host.
    pub fn install(&mut self, assignments: &[BlockTable], stream: Option<&Stream>) -> Result<()> {
        let update = self.tables.prepare(assignments)?;
        self.storage.install(&update, stream)?;
        self.tables.commit(update);
        Ok(())
    }

    /// Prepare newly assigned units for a new owner on `column`'s stream.
    /// Fails if an export, import or execution still overlaps any of them.
    pub fn recycle(
        &mut self,
        units: &[u32],
        column: &UnitColumn,
        stream: Option<&Stream>,
    ) -> Result<()> {
        if units.iter().collect::<HashSet<_>>().len() != units.len() {
            return Err(Error::Invalid("recycled KV units repeat a unit".into()));
        }
        let unit_tokens = self.plan().unit_tokens();
        let spans: Vec<_> = units.iter().map(|&unit| (unit, 0, unit_tokens)).collect();
        self.manager.require_reusable(&spans)?;
        column.recycle(&self.planes, units, stream)
    }

    /// Clear released slots' rows on `stream` (ordered after every batch
    /// that read them) and forget them on the host.
    pub fn release_slots(&mut self, slots: &[u32], stream: Option<&Stream>) -> Result<()> {
        let slots = self.tables.release_slots(slots)?;
        self.storage.clear(&slots, stream)?;
        self.tables.release(&slots);
        Ok(())
    }

    /// Export a slot's visible KV extent (see [`kv_export::export`]).
    #[expect(clippy::too_many_arguments, reason = "one export's complete inputs")]
    pub fn export(
        &mut self,
        slot: u32,
        visible: u32,
        destination: &str,
        buffer: BufferId,
        transports: &Transports,
        consumers: &[usize],
        stream: Option<&Stream>,
    ) -> Result<(KvTransfer, Vec<Location>)> {
        kv_export::export(
            &mut self.manager,
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
        export: Arc<KvTransfer>,
        slot: u32,
        tables: Vec<Arc<GroupTable>>,
        initialized: Vec<u32>,
        transports: Arc<Transports>,
    ) -> Result<PendingImport> {
        self.manager.validate_install(&export)?;
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
                    "KV transfer shape does not match destination layers".into(),
                ));
            }
        }

        let ranges = self
            .tables
            .import_spans(slot, &tables, &export, &initialized)?;
        let copy =
            !initialized.is_empty() || export.groups.iter().any(|group| !group.tensors.is_empty());
        let task = copy.then(|| self.imports.lane.reserve()).transpose()?;
        let write = Arc::new(KVImport::new(export.source, slot as usize, copy));
        let entry = ImportEntry::new(Arc::clone(&write));

        let withdraw = |task: &Option<Arc<HostTask<ImportTask>>>| {
            if let Some(task) = task
                && let Err(cleanup) = task.cancel(true)
            {
                eprintln!("failed to withdraw a KV import task: {cleanup}");
            }
        };
        if let Some(task) = &task {
            let configured = task.configure(ImportCopy::new(
                Arc::clone(&self.imports.importer),
                Arc::clone(&write),
                KVImportBackend {
                    write: Arc::clone(&write),
                    export: Arc::clone(&export),
                    tables: tables.clone(),
                    initialized: initialized.clone(),
                    planes: Arc::clone(&self.planes),
                    transports,
                    importer: Arc::clone(&self.imports.importer),
                    retired: Arc::clone(&self.imports.retired),
                },
            ));
            if let Err(error) = configured {
                withdraw(&Some(Arc::clone(task)));
                return Err(error);
            }
        }

        let manager = &mut self.manager;
        let retirement = Arc::clone(entry.retirement());
        if let Err(error) = self.imports.importer.reserve(entry.clone(), || {
            manager.reserve_import(export.source, &ranges, retirement)
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
    /// imported length on `stream`.
    pub fn adopt_import(
        &mut self,
        import: &PendingImport,
        installed: BufferId,
        stream: Option<&Stream>,
    ) -> Result<Arc<KvTransfer>> {
        if installed.owner != import.export.source.owner {
            return Err(Error::Invalid(
                "installed KV buffer does not belong to its source".into(),
            ));
        }
        self.manager.validate_install(&import.export)?;
        let slot = import.slot() as u32;
        for (group, expected) in import.tables.iter().enumerate() {
            if self.tables.table(slot, group as u32)? != *expected {
                return Err(Error::Invalid(
                    "KV installation scheduler block table changed".into(),
                ));
            }
        }
        if !import.done() {
            return Err(Error::State(
                "KV import was observed before input readiness",
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
    pub fn release_imports(&mut self, buffers: &HashSet<BufferId>) -> Result<()> {
        self.imports.cancel(self.imports.importer.release(buffers))
    }

    /// Cancel retired requests' imports, except retained buffers.
    pub fn cancel_request_imports(
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
    pub fn set_wake(&self, wake: Option<Wake>) {
        self.imports.lane.set_wake(wake.clone());
        *self
            .imports
            .retired
            .wake
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = wake;
    }

    /// Lend units `1..=count` as warmup scratch: no access may be pending,
    /// and the units are reset natively on `stream` now and when the
    /// scratch is returned.
    pub fn scratch_units(&mut self, count: u32, stream: Option<&Stream>) -> Result<ScratchUnits> {
        if self.manager.has_pending_accesses() {
            return Err(Error::State("KV scratch requires an idle cache"));
        }
        if count >= self.planes.num_units() {
            return Err(Error::Invalid(
                "KV scratch exceeds the pool's usable units".into(),
            ));
        }
        self.planes.zero_leading(count, stream)?;
        Ok(ScratchUnits { count })
    }

    /// Stop imports, release exports and require every access retired.
    /// Failed retirements keep their storage: the caller must not reuse it.
    pub fn close(&mut self) -> Result<()> {
        let cancelled = self.imports.cancel(self.imports.importer.stop());
        let mut stopped = Ok(());
        for error in self.imports.lane.close() {
            stopped = stopped.and(Err(error));
        }
        let reaped = self.imports.retired.reap();
        let exports = self.manager.exported_buffers();
        self.manager.release_exports(&exports);
        cancelled
            .and(stopped)
            .and(reaped)
            .and(self.imports.importer.require_retired())
            .and(self.manager.require_retired())?;
        self.manager.clear_resident();
        self.tables.clear();
        Ok(())
    }
}

/// Units `1..=count` lent to warmup batches.
pub struct ScratchUnits {
    count: u32,
}

impl ScratchUnits {
    pub fn units(&self) -> std::ops::RangeInclusive<u32> {
        1..=self.count
    }

    /// Wait for the warmup's writes on `stream`, then reset the units again.
    pub fn finish(self, cache: &KVCache, stream: Option<&Stream>) -> Result<()> {
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        cache.planes.zero_leading(self.count, stream)?;
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        Ok(())
    }
}

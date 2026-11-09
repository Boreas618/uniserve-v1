//! The rank's prefix cache: unit pool, block tables, accesses and imports.
//!
//! [`PrefixCache`] composes the owners the worker drives per batch:
//!
//! - [`CachePlanes`]: the pool's device storage;
//! - [`BlockTables`] (host) and [`BlockTableStorage`] (device): the
//!   scheduler's page assignments and their numerical tables;
//! - the import pool: bounded workspaces whose host tasks copy other ranks'
//!   exports into assigned pages;
//! - [`HostKvCache`]: the host page pools the engine copies pages through, on
//!   the rank's copy stream ([`PrefixCache::issue_pages`]).
//!
//! Every device write runs on a stream the caller supplies; nothing reads a
//! "current" stream. All validation precedes the first device write, and
//! failed or unknown device completions keep their storage retained.

pub(super) mod block_tables;
pub mod conversion;
pub(crate) mod host;
mod imports;
mod manifest;
pub(super) mod plan;
pub(crate) mod planes;
pub mod scratch;

pub use conversion::Unimported;
use imports::Imports;

use std::sync::{Arc, Mutex};

use ipc::{BlockTable, PageCopy};
use platform::{DeviceId, Event, EventFlags, Stream};

use self::block_tables::{BlockTableStorage, BlockTables};
pub(crate) use self::host::HostKvCache;
use self::plan::CachePlan;
use self::planes::{CachePlanes, UnitColumn};
use super::allocation::StorageView;
use crate::host::HostPool;
use crate::memory::retained::ExclusiveStorage;
use crate::stream::Issue;
use crate::transfer::{CacheImporter, ImportWorkspace, Retired};
use crate::{Error, Result};

/// The most import copy threads (and workspaces) a rank runs.
const IMPORT_WORKERS: usize = 4;

/// The host KV cache, the copy stream its copies run on and the unit column
/// bound on that stream.
struct PageCopies {
    cache: HostKvCache,
    stream: Stream,
    column: UnitColumn,
}

/// A batch's page copies, recorded on the copy stream
/// ([`PrefixCache::issue_pages`]).
pub(crate) struct IssuedPages {
    /// Recorded behind the loads: the batch's forward waits on it.
    pub(crate) loads: Option<Arc<Event>>,
    /// Recorded behind the offloads: the batch's output waits on it.
    pub(crate) offloads: Option<Arc<Event>>,
}

/// The rank's prefix cache owners.
/// One layer's share of the cache planes, as the model reads it.
///
/// The scales and flags are views of the pool's per-cell planes, borrowed for
/// as long as the layer's pages are referenced; the page geometry is stated
/// once per group and repeated here so a caller needs no second lookup.
pub struct LayerPlanes {
    /// Tokens one page holds of every layer in the group.
    pub block_size: u32,
    /// Pages the pool holds for the layer's group.
    pub pages: u32,
    /// The element type the group's value planes hold, as a kernel operand.
    pub dtype: ops::DType,
    /// The layer's key values, `[pages, block_size, kv_heads, head_dim]`.
    pub key: StorageView,
    /// The layer's value values, as `key`.
    pub value: StorageView,
    /// The layer's key scales, `[pages,1,1,1]`; `None` on a dense pool.
    pub key_scale: Option<StorageView>,
    /// The layer's value scales, as `key_scale`.
    pub value_scale: Option<StorageView>,
    /// The layer's key "initialized" flags, `[pages,1,1,1]`.
    pub key_initialized: StorageView,
    /// The layer's value "initialized" flags.
    pub value_initialized: StorageView,
}

pub struct PrefixCache {
    planes: Arc<CachePlanes>,
    storage: BlockTableStorage,
    tables: BlockTables,
    imports: Imports,
    /// The host KV cache and the copy stream its copies run on, when the engine
    /// copies pages through it.
    pages: Option<PageCopies>,
}

impl PrefixCache {
    /// Import workers for `capacity` concurrently admitted imports.
    pub(crate) fn import_workers(capacity: usize) -> usize {
        capacity.clamp(1, IMPORT_WORKERS)
    }

    /// Fixed device bytes besides the pool itself: the block tables, the import
    /// scratch and the per-stream unit columns.
    ///
    /// The unit columns are the cache's index, one per stream that writes a
    /// unit, and each holds one `i64` per unit; the block tables are the batch
    /// plan's, one row per request slot at the widest table. Both are charged
    /// here so a pool's fit is not optimistic by their bytes per unit.
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
        // Every unit column holds one index per unit, charged with each unit:
        // the import workers', the primary stream's and the copy stream's of a
        // rank that copies pages through the host KV cache.
        let columns = u64::try_from(unit_columns).map_err(|_| {
            crate::Error::Invariant("a rank holds more stream columns than fit".into())
        })? * UnitColumn::bytes(num_units);
        Ok(
            BlockTableStorage::bytes(&plan.group_shapes()?, request_pool_size, width)
                + workers * ImportWorkspace::scratch_bytes(elements, scales)
                + columns,
        )
    }

    /// Allocate the pool, the tables and the import workspaces on `device`
    /// (host storage when `None`), resetting device storage on `stream`.
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
        layout: Option<&planes::PlaneLayout>,
        request_pool_size: u32,
        width: u32,
        import_capacity: usize,
        device: Option<DeviceId>,
        stream: Option<&Stream>,
        storage: &mut dyn ExclusiveStorage,
        conversion: Arc<dyn conversion::PageConversion>,
    ) -> Result<Self> {
        let shapes = plan.group_shapes()?;
        let planes = Arc::new(CachePlanes::allocate(
            plan, device, num_units, layout, stream, storage,
        )?);
        let storage =
            BlockTableStorage::allocate(&shapes, request_pool_size, width, device, stream)?;
        let tables = BlockTables::new(shapes, request_pool_size, width as usize, num_units)?;

        let workers = Self::import_workers(import_capacity);
        let workspaces = (0..workers)
            .map(|_| ImportWorkspace::bind(&planes, Arc::clone(&conversion)).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        let importer = Arc::new(CacheImporter::new(workspaces));
        let pool = HostPool::new(import_capacity.max(1), workers, "worker-cache-import")?;
        let retired = Arc::new(Retired {
            importer: Arc::downgrade(&importer),
            wake: Mutex::new(None),
        });

        Ok(Self {
            planes,
            storage,
            tables,
            imports: Imports {
                importer,
                pool,
                retired,
            },
            pages: None,
        })
    }

    /// Whether the rank copies pages through the host KV cache.
    pub(crate) fn copies_pages(&self) -> bool {
        self.pages.is_some()
    }

    /// Copy pages through `cache` from now on, on the copy stream `stream`.
    pub(crate) fn set_host_kv_cache(&mut self, cache: HostKvCache, stream: Stream) -> Result<()> {
        let column = UnitColumn::bind(&self.planes, Some(&stream))?;
        self.pages = Some(PageCopies {
            cache,
            stream,
            column,
        });
        Ok(())
    }

    /// Record a batch's page copies on the copy stream, to be issued as soon as
    /// the batch arrives: load the units of `loads` from their slots, and
    /// offload the published units of `offloads` into theirs.
    ///
    /// The loads fill new units of the batch, which are idle (no in-flight
    /// export, import or execution addresses them, checked here) and which
    /// the loads overwrite whole, so the batch does not recycle them; its
    /// forward waits on the returned load fence. The offloads copy pages the
    /// engine published once it accepted the result of the call that computed
    /// them, so the forward that produced them completed before this batch
    /// was sent, and the engine holds their units until the batch's output
    /// commits them; that output waits on the returned offload fence, and
    /// nothing else does.
    ///
    /// Fails for a rank that maps no page pool, or when a load's unit is
    /// in use.
    pub(crate) fn issue_pages(
        &mut self,
        loads: &[PageCopy],
        offloads: &[PageCopy],
        issue: impl Issue,
    ) -> Result<IssuedPages> {
        let copies = self.pages.as_ref().ok_or_else(|| {
            Error::Invalid("a batch copies pages but the rank maps no page pool".into())
        })?;
        self.planes
            .device()
            .ok_or(Error::Invariant("a page pool maps a device cache".into()))?;

        /// Record the fence a batch's forward or output waits on.
        fn fence(issue: impl Issue) -> Result<Arc<Event>> {
            let event = Arc::new(Event::new(EventFlags::default()));
            issue.record(&event)?;
            Ok(event)
        }
        let loads_fence = if loads.is_empty() {
            None
        } else {
            copies
                .cache
                .load(&self.planes, loads, &copies.column, issue)?;
            Some(fence(issue)?)
        };
        let offloads_fence = if offloads.is_empty() {
            None
        } else {
            copies.cache.offload(&self.planes, offloads, issue)?;
            Some(fence(issue)?)
        };
        Ok(IssuedPages {
            loads: loads_fence,
            offloads: offloads_fence,
        })
    }

    /// The stream the rank's page copies run on, when it copies pages through a host KV cache.
    pub(crate) fn copy_stream(&self) -> Option<Stream> {
        self.pages.as_ref().map(|copies| copies.stream.clone())
    }

    /// Copies `offloads` to their host slots and waits for the copies, with no batch to order them
    /// behind: what the engine published has been computed, and it holds the units meanwhile.
    pub(crate) fn copy_pages(&mut self, offloads: &[PageCopy]) -> Result<()> {
        let stream = self.copy_stream().ok_or_else(|| {
            Error::Invalid("a batch copies pages but the rank maps no page pool".into())
        })?;
        self.issue_pages(&[], offloads, &stream)?;
        stream.synchronize()?;
        Ok(())
    }

    pub(crate) fn planes(&self) -> &Arc<CachePlanes> {
        &self.planes
    }

    /// Copy leased immutable units before a batch starts using their new
    /// writable pages. Destination units have no outstanding device access.
    pub(crate) fn copy_units(
        &mut self,
        copies: &[(ipc::UnitId, ipc::UnitId)],
        issue: impl Issue,
    ) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        // A page of a pool with a state group holds its layers' checkpoints as planes of their own
        // storages, and this copy moves the K/V planes, the flags and the scales only. Copying such
        // a page here would hand the destination a page that looks initialized while its state
        // planes are stale, so it is refused by name.
        if self.plan().state_fields() > 0 {
            return Err(Error::Unsupported(
                "a page copy of a pool with a state group must copy its state planes through the \
                 block copy, not the K/V copy"
                    .into(),
            ));
        }
        self.planes.copy_units(copies, issue)
    }

    pub(crate) fn plan(&self) -> &Arc<CachePlan> {
        self.planes.plan()
    }

    /// How many attention groups the plan holds, which the model's layers index.
    pub fn groups(&self) -> usize {
        self.planes.groups()
    }

    /// How many layers read `group`, in model order: the bound of the
    /// `layer_planes` loop and of every per-layer view.
    pub fn layers(&self, group: usize) -> usize {
        self.planes.layers(group)
    }

    /// The pages `group`'s pool holds; `LayerPlanes::pages` repeats this per
    /// layer, so a caller that needs it once asks here.
    pub fn pages(&self, group: usize) -> Result<u32> {
        self.planes.pages(group)
    }

    /// One layer's pages as the model's cache reads them: the page geometry
    /// and the layer's own cell of each page.
    ///
    /// A layer occupies one cell of a page - column `layer % columns`, unit
    /// `layer / columns` - so its scale and flag views are `[pages,1,1,1]`,
    /// borrowed from the per-cell planes and never copied. A dense pool answers
    /// `None` for its scales because it holds none.
    /// The device's unit tables, as
    /// [`BlockTableStorage::unit_tables`](block_tables::BlockTableStorage::unit_tables)
    /// states them: `[table, slot, width]` i32 and the two dimensions.
    pub fn unit_tables(&self) -> (StorageView, u32, u32) {
        self.storage.unit_tables()
    }

    pub fn layer_planes(&self, group: usize, layer: usize) -> Result<LayerPlanes> {
        Ok(LayerPlanes {
            block_size: self.planes.page_size(group)?,
            pages: self.planes.pages(group)?,
            dtype: self.planes.dtype(group)?,
            key: self.planes.key_values(group, layer)?,
            value: self.planes.value_values(group, layer)?,
            key_scale: self.planes.key_scale(group, layer)?,
            value_scale: self.planes.value_scale(group, layer)?,
            key_initialized: self.planes.key_initialized(group, layer)?,
            value_initialized: self.planes.value_initialized(group, layer)?,
        })
    }

    pub fn tables(&self) -> &BlockTables {
        &self.tables
    }

    /// Every bound tensor of the cache: the planes and the block tables,
    /// keyed as numerical bindings receive them.
    pub fn views(&self) -> crate::Result<Vec<(String, StorageView)>> {
        let mut views = self.planes.views();
        views.extend(self.storage.views());
        Ok(views)
    }

    /// The units a gather for `slot`'s pages of `group` reads, in page order.
    ///
    /// This is the row the attention kernel gathers through: the same units
    /// [`install`](Self::install) wrote, or a refusal when the slot holds no
    /// table for the group. A caller that has just installed an allocation can
    /// ask here what the batch's gather will actually read.
    pub fn slot_pages(&self, slot: u32, group: u32) -> Result<Vec<u32>> {
        let table = self.tables.table(slot, group)?;
        Ok(table.units.iter().map(|unit| unit.0).collect())
    }

    /// Install scheduler page assignments: validate them all, write the
    /// changed rows on `stream`, then commit them on the host.
    pub fn install(&mut self, assignments: &[BlockTable], stream: impl Issue) -> Result<()> {
        let update = self.tables.prepare(assignments)?;
        self.storage.install(&update, stream)?;
        self.tables.commit(update);
        Ok(())
    }

    /// Make the given units of every group fresh for a new owner on `stream`: flags zero and scales
    /// one, and the values left as they are.
    ///
    /// A unit the mechanism acquires for a request is written before it is read: attention reads a
    /// page only below the request's length, and a page the request continues from an earlier one
    /// is copied in first. So zeroing its values is a fill of every layer's page that no read
    /// observes, issued on the batch stream ahead of the forward, one fill per column for each run
    /// of units. The flags and scales are what a later import and a quantized read do observe.
    pub fn acquire_units(&mut self, units: &[u32], stream: impl Issue) -> Result<()> {
        let mut units = units.to_vec();
        units.sort_unstable();
        units.dedup();
        if units.first() == Some(&0) {
            return Err(Error::Invalid(
                "unit zero is the block tables' sentinel and cannot be acquired".to_owned(),
            ));
        }
        let mut start = 0;
        while start < units.len() {
            let mut end = start + 1;
            while end < units.len() && units[end] == units[end - 1] + 1 {
                end += 1;
            }
            self.planes
                .reset_units(units[start], (end - start) as u32, false, stream)?;
            start = end;
        }
        Ok(())
    }

    /// Cancel every import and join the import copy threads, so no import
    /// reads through the transports any more. Idempotent.
    pub(crate) fn stop_imports(&mut self) -> Result<()> {
        let cancelled = self.imports.cancel(self.imports.importer.stop());
        let mut stopped = Ok(());
        for error in self.imports.pool.close() {
            stopped = stopped.and(Err(error));
        }
        cancelled.and(stopped).and(self.imports.retired.reap())
    }

    /// Stop imports and require every import retired. Failed retirements
    /// keep their storage: the caller must not reuse it.
    pub fn close(&mut self) -> Result<()> {
        let stopped = self.stop_imports();
        stopped.and(self.imports.importer.require_retired())?;
        self.tables.clear();
        Ok(())
    }
}

#[cfg(test)]
mod fixed_bytes_tests {
    use super::*;

    fn plan() -> CachePlan {
        CachePlan {
            columns: 1,
            plane_bytes: 16,
            quantized: false,
            unit_bytes: 16,
            dtype: ipc::PrefixCacheDtype::BFloat16,
            groups: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    /// The two fixed figures are charged separately: one column's bytes per
    /// unit for the index, and nothing at all when no stream writes one.
    #[test]
    fn unit_columns_are_charged_per_unit_and_per_stream() -> Result<()> {
        let plan = plan();
        let at = |units: u32, columns: usize| {
            PrefixCache::fixed_bytes(&plan, units, 4, 8, 0, columns).expect("fixed bytes")
        };
        let one_column = at(100, 1);
        let no_column = at(100, 0);

        // One column adds exactly one column's bytes per unit.
        assert_eq!(one_column - no_column, 100 * 8);
        // Doubling the units adds that same figure again, so the term is per unit.
        assert_eq!(at(200, 1) - one_column, 100 * 8);
        // A second stream adds one more column, not a re-count of the first.
        assert_eq!(at(100, 2) - one_column, 100 * 8);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::retained::LocalStorage;
    use crate::stream::EventPool;
    use crate::stream::recorded::Batch;

    /// A pool of one dense group of two layers with 8 tokens per page.
    fn plan() -> Arc<CachePlan> {
        Arc::new(
            (CachePlan {
                columns: 2,
                plane_bytes: 128,
                quantized: false,
                unit_bytes: 2 * 128 + 2,
                dtype: ipc::PrefixCacheDtype::BFloat16,
                groups: vec![plan::CacheGroupPlan {
                    window: None,
                    state: false,
                    draft: false,
                    block_size: 8,
                    units_per_page: 1,
                    layers: vec!["layers.0".into(), "layers.1".into()],
                    layer_ids: vec![0, 1],
                    axis: plan::GroupAxis {
                        offset: 0,
                        total: 2,
                    },
                    num_kv_heads: 2,
                    total_kv_heads: 2,
                    kv_head_offset: 0,
                    head_dim: 8,
                    dtype: crate::memory::allocation::ElementType::BF16,
                    compute_dtype: crate::memory::allocation::ElementType::BF16,
                }],
                checkpoints: Vec::new(),
            })
            .validated()
            .expect("a dense plan of two layers"),
        )
    }

    /// The row a batch's gather reads is the units the allocation named - not zeros, and not
    /// another request's pages. This is the assertion whose absence let a graph run clean against
    /// the wrong pages: only `install` writes a request slot's row.
    #[test]
    fn an_installed_allocation_is_the_row_the_batch_gathers() -> Result<()> {
        let mut cache = PrefixCache::new(
            plan(),
            /* units */ 8,
            None,
            /* request_pool_size */ 2,
            /* width */ 4,
            /* import_capacity */ 1,
            None,
            None,
            &mut LocalStorage,
            Arc::new(conversion::Unimported),
        )?;
        let batch = Batch::new(None, EventPool::new());

        // Slot 1 is the first usable slot: slot 0 is the block tables' sentinel row.
        let allocation = ipc::BlockTable {
            request_pool_idx: 1,
            group_id: 0,
            start_page: 0,
            unit_ids: vec![ipc::UnitId(3), ipc::UnitId(4), ipc::UnitId(5)],
            allocated_tokens: 24,
        };
        cache.install(std::slice::from_ref(&allocation), batch.on().host())?;
        assert_eq!(
            cache.slot_pages(1, 0)?,
            vec![3, 4, 5],
            "the gather does not read the units the allocation named"
        );

        // A later batch grows the page run: the row becomes the units it now names.
        let extended = ipc::BlockTable {
            unit_ids: vec![
                ipc::UnitId(3),
                ipc::UnitId(4),
                ipc::UnitId(5),
                ipc::UnitId(6),
            ],
            allocated_tokens: 32,
            ..allocation
        };
        cache.install(std::slice::from_ref(&extended), batch.on().host())?;
        assert_eq!(
            cache.slot_pages(1, 0)?,
            vec![3, 4, 5, 6],
            "the row did not follow the allocation that grew it"
        );

        // The sentinel row is never a request's: a batch that names slot 0 is refused by name.
        let sentinel = ipc::BlockTable {
            request_pool_idx: 0,
            ..extended
        };
        let refused = cache
            .install(&[sentinel], batch.on().host())
            .expect_err("slot 0 is not a request slot");
        assert!(
            refused.to_string().contains("block table"),
            "the refusal does not say what was wrong: {refused}"
        );

        // A slot that holds nothing answers a refusal, not a row of zeros.
        assert!(cache.slot_pages(2, 0).is_err());
        Ok(())
    }
}

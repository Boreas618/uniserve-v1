//! What the scheduler allocates for a call: the cache page tables of its
//! request ([`BlockTable`], with the units newly acquired for them in
//! [`CacheUnitAllocation`]), its forward rows ([`ForwardBatch`]), latent
//! trajectories ([`LatentParams`]), decode cursors ([`DecodeRange`]),
//! persistent buffer spans ([`BufferAllocation`]) and the host cache copies
//! ([`PageCopy`], [`LatentOffload`]).
//!
//! The batch these travel in is [`crate::execution::Batch`]: the one batch the
//! scheduler submits and the worker serves.

use super::*;

/// One request slot's complete unit table in one cache group.
///
/// `unit_ids` lists the units of logical pages `start_page..`, page-major:
/// the group's `units_per_page` units of page `start_page` come first. Pages
/// before `start_page` have been released by a sliding-window or state
/// group, and a worker reads none of them. A state group's page that holds
/// no checkpoint names the padding sentinel, unit 0, for each of its units;
/// a worker reads and writes none of them. `allocated_tokens` is the absolute token extent
/// the table covers, up to the end of its last page; the worker checks it
/// against the group's page shape, which this record does not carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockTable {
    /// Scheduler-assigned request-state row that owns the table.
    pub request_pool_idx: u32,
    /// Cache group addressed by the table.
    pub group_id: u32,
    /// First logical page the units cover; earlier pages are retired.
    pub start_page: u32,
    /// Physical unit identifiers of pages `start_page..`, page-major.
    pub unit_ids: Vec<UnitId>,
    /// Absolute token extent the table covers.
    pub allocated_tokens: u32,
}

/// Physical units newly acquired for one installed block table.
///
/// The worker resets each unit, or copies its immutable source, before a
/// call reads or writes it. Copy sources remain leased through completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheUnitAllocation {
    /// Scheduler-assigned request-state row receiving the units.
    pub request_pool_idx: u32,
    /// Cache group receiving the units.
    pub group_id: u32,
    /// Newly allocated physical unit identifiers.
    pub unit_ids: Vec<UnitId>,
    /// `(source, destination)` pairs within this cache pool. Destinations
    /// are newly allocated units; sources may be inherited read-only units.
    pub copies: Vec<(UnitId, UnitId)>,
}

/// One page copied between a rank's cache units and a slot of its host
/// page pool.
///
/// A read fills `units` (one page of `group_id`, page-major as in a
/// [`BlockTable`]) from the slot before the batch's forward; the units are
/// new units of the batch, which the worker resets first. A write copies
/// the units of a published page into the slot; the rank's output for the
/// batch reports the write complete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageCopy {
    /// The rank of the worker group that copies the page.
    pub rank: u32,
    /// The page pool at the rank's host's Storage Server.
    pub pool: u64,
    /// The cache group of the page.
    pub group_id: u32,
    /// The pool slot.
    pub slot: u64,
    /// The page's units.
    pub units: Vec<UnitId>,
}

impl PageCopy {
    /// Validates the page's units.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.units.is_empty() && distinct_units(&self.units),
            "a page copy is empty, repeats a unit, or carries unit zero"
        );
        Ok(())
    }
}

/// One page of a latent trajectory copied from a rank's latent banks to a
/// slot of the rank's host latent pool, a page pool of the rank's host's
/// Storage Server ([`LatentOffload`]): page `page` of the trajectory's
/// pages in the bank holding its committed step goes to the pool slot,
/// which the engine reserved. A video trajectory's pages are its request
/// slot's; an image trajectory's are the pages its calls name, in logical
/// order. A pool page holds as many bytes as a bank page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentPageCopy {
    /// The rank of the worker group that copies the page.
    pub rank: u32,
    /// The latent pool at the rank's host's Storage Server.
    pub pool: u64,
    /// The pool slot.
    pub slot: u64,
    /// The page's index among the trajectory's pages, from zero.
    pub page: u32,
}

/// A preempted request's trajectory (a video request's, or the image a
/// token request was denoising), written from the bank holding its
/// committed step into the host latent pools of its denoising ranks (one
/// [`LatentPageCopy`] per page of the trajectory and rank). The request's
/// calls have completed, so the committed step is final; the rank's output
/// for the batch reports the copies complete, and the engine then commits
/// the pool slots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentOffload {
    /// The request whose trajectory, in its slot, is written.
    pub request_key: RequestKey,
    pub pages: Vec<LatentPageCopy>,
}

/// The stored trajectory a resumed request's latent preparation opens at
/// (a video request's, or the image a token request had begun): the call
/// prepares the trajectory as for step zero, then each rank copies the
/// parts of the stored pages its own pages hold into the bank it
/// initializes, as the copy manifest from the stored ranks' layouts to its
/// own directs, and commits the trajectory at step `steps`. The engine
/// pinned the pool slots until the call's result.
///
/// The stored ranks may have laid the samples out under another parallel
/// configuration than the restoring ranks: every rank plans its copies
/// from the sources it reads ([`LatentSource::readers`]) to its own pages,
/// and pages laid out alike are copied whole.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentRestore {
    /// Denoising steps the stored trajectory took; positive.
    pub steps: u32,
    /// The digest of what the stored values are, which keys every page
    /// with the part of the samples it holds
    /// (`storage::manifest::Storage::paged`): what computed the
    /// trajectory, the trajectory and the steps it took.
    pub content: [u8; 32],
    /// The stored ranks' pages, in their ranks' order.
    pub sources: Vec<LatentSource>,
}

/// One stored rank's pages of a trajectory a latent preparation restores.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentSource {
    /// The restoring ranks that read these pages: the ranks on the host
    /// whose Storage Server holds the pool.
    pub readers: Vec<u32>,
    /// The pool the pages lie in.
    pub pool: LatentPoolRef,
    /// The stored rank's pages of the trajectory as one storage
    /// (`storage::manifest::StorageLayout`, JSON): the part of each latent
    /// sample it held and where its values lie in the pages.
    pub layout: String,
    /// The pool slot of each stored page, in page order, pinned.
    pub slots: Vec<u64>,
}

/// A host latent pool, as a rank of any epoch on its host maps it again:
/// its identity at the Storage Server and the shape and NUMA node a page
/// pool request finds it by (`storage::protocol::Request::PagePool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentPoolRef {
    pub pool: u64,
    /// The pool's `PageShape::layout` digest.
    pub shape: [u8; 32],
    pub numa_node: Option<u32>,
    pub page_bytes: u64,
}

impl LatentRestore {
    /// Validates the restored step and the sources.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.steps > 0 && !self.sources.is_empty(),
            "a latent restore names no step or no source"
        );
        for source in &self.sources {
            ensure_valid!(
                !source.slots.is_empty() && source.pool.page_bytes > 0,
                "a latent restore's source names no page"
            );
            let readers = source.readers.iter().collect::<HashSet<_>>();
            ensure_valid!(
                readers.len() == source.readers.len(),
                "a latent restore's source names a reader twice"
            );
        }
        Ok(())
    }
}

/// Whether `units` are real units (never the zero sentinel) without repeats.
fn distinct_units(units: &[UnitId]) -> bool {
    units.iter().all(|unit| unit.0 > 0) && units.iter().collect::<HashSet<_>>().len() == units.len()
}

impl BlockTable {
    /// Validates table identity and unit uniqueness: a real unit appears
    /// once, and the padding sentinel only for a page without units.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "block table request slot must be positive"
        );
        let held: Vec<UnitId> = self
            .unit_ids
            .iter()
            .copied()
            .filter(|unit| unit.0 > 0)
            .collect();
        ensure_valid!(distinct_units(&held), "block table repeats a unit");
        Ok(())
    }
}

impl CacheUnitAllocation {
    /// Validates newly allocated unit identities and ownership.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "cache-unit allocation request slot must be positive"
        );
        ensure_valid!(
            !self.unit_ids.is_empty() && distinct_units(&self.unit_ids),
            "cache-unit allocation is empty, repeats a unit, or carries unit zero"
        );
        let mut destinations = HashSet::new();
        ensure_valid!(
            self.copies.iter().all(|(source, destination)| {
                source.0 > 0
                    && !self.unit_ids.contains(source)
                    && self.unit_ids.contains(destination)
                    && destinations.insert(*destination)
            }),
            "cache-unit copies require distinct new destinations and immutable sources"
        );
        Ok(())
    }
}

/// Aligned columns defining the model's forward rows, independent of wire framing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardBatch {
    /// Physical call index for each row; logical call IDs remain unchanged.
    pub call_indices: Vec<u32>,
    /// Scheduler-owned request slots, including alternative CFG prefixes.
    pub request_pool_indices: Vec<u32>,
    /// Total attention lengths (cached prefix plus query), not allocated capacities.
    pub seq_lens: Vec<u32>,
    /// Query tokens evaluated by each row.
    pub query_lens: Vec<u32>,
    /// Whether the query interval persists in the prefix cache after its visible prefix.
    pub write_cache: Vec<bool>,
}

impl ForwardBatch {
    /// Append one model row to every column in the same order.
    pub fn push(
        &mut self,
        call_index: u32,
        request_pool_index: u32,
        seq_len: u32,
        query_len: u32,
        write_cache: bool,
    ) {
        self.call_indices.push(call_index);
        self.request_pool_indices.push(request_pool_index);
        self.seq_lens.push(seq_len);
        self.query_lens.push(query_len);
        self.write_cache.push(write_cache);
    }

    /// Validate column alignment and numerical ranges before indexing model inputs.
    pub fn validate(&self, call_count: usize) -> ValidationResult<()> {
        let rows = self.call_indices.len();
        ensure_valid!(
            self.request_pool_indices.len() == rows
                && self.seq_lens.len() == rows
                && self.query_lens.len() == rows
                && self.write_cache.len() == rows,
            "forward input columns have different lengths"
        );
        ensure_valid!(
            self.call_indices
                .iter()
                .all(|index| (*index as usize) < call_count),
            "forward call index is outside its batch"
        );
        ensure_valid!(
            self.request_pool_indices.iter().all(|slot| *slot > 0),
            "forward input carries the reserved request slot"
        );
        ensure_valid!(
            self.query_lens.iter().all(|length| *length > 0),
            "forward query length must be positive"
        );
        ensure_valid!(
            self.seq_lens
                .iter()
                .zip(&self.query_lens)
                .all(|(total, query)| total >= query),
            "forward sequence length is shorter than its query"
        );
        Ok(())
    }
}

/// Raster geometry and paged storage of one image request's trajectory, for
/// a call that addresses a latent product ([`Call::addresses_latent_pages`]).
///
/// The scheduler's latent allocator assigns the pages, so the rank learns
/// them here. The rank's latent pool keeps the trajectory's step ledger, so
/// no step is stated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentParams {
    /// Request that owns the trajectory.
    pub request_key: RequestKey,
    /// Call that addresses the trajectory.
    pub call_id: CallId,
    /// Physical latent pages in logical order; empty for request-owned tensors.
    pub page_table: Vec<u32>,
    /// Logical units stored in pages; zero when the trajectory uses request tensors.
    pub latent_units: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
}

impl LatentParams {
    /// Validates latent page identities, shape, and byte bounds.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.call_id.batch_id > 0,
            "latent params call id must be positive"
        );
        ensure_valid!(
            self.height > 0 && self.width > 0,
            "latent params geometry must be positive"
        );
        ensure_valid!(
            (!self.page_table.is_empty() == (self.latent_units > 0))
                && self.page_table.iter().all(|page| *page > 0)
                && self.page_table.iter().collect::<HashSet<_>>().len() == self.page_table.len(),
            "latent params page table disagrees with its units, repeats a page, or carries page zero"
        );
        Ok(())
    }
}

/// Cursor and unit bounds for one diffusion decode call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeRange {
    /// Request that owns the media output.
    pub request_key: RequestKey,
    /// Decode call receiving the params.
    pub call_id: CallId,
    /// First decoder unit assigned to the call.
    pub cursor: u32,
    /// Maximum decoder units the call may process.
    pub max_units: u32,
    /// Ranks of a distributed component that hold the call's units, in unit
    /// order: each takes the next `units_per_rank` units and the last what
    /// remains. Empty deals the units to the component's ranks in configured
    /// order, the deal of every call whose units follow that order.
    pub ranks: Vec<u32>,
}

impl DecodeRange {
    /// Validates decode identity, unit capacity, and that no rank holds two
    /// runs of the call's units.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.call_id.batch_id > 0,
            "decode params call id must be positive"
        );
        ensure_valid!(
            self.max_units > 0,
            "decode params unit bound must be positive"
        );
        ensure_valid!(
            self.ranks.iter().collect::<HashSet<_>>().len() == self.ranks.len(),
            "decode params name each rank once"
        );
        Ok(())
    }
}

/// Scheduler-selected address span for one cross-call persistent buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferAllocation {
    /// Persistent buffer identity receiving the address span.
    pub buffer: BufferId,
    /// Byte offset within the persistent-buffer arena.
    pub offset: u64,
    /// Length of the reserved span in bytes.
    pub bytes: u64,
}

impl BufferAllocation {
    /// Validates the buffer identity and a non-empty span whose end does not
    /// overflow `u64`.
    pub fn validate(self) -> ValidationResult<()> {
        self.buffer.validate()?;
        ensure_valid!(self.bytes > 0, "buffer params byte extent must be positive");
        ensure_valid!(
            self.offset.checked_add(self.bytes).is_some(),
            "buffer params span overflows"
        );
        Ok(())
    }
}

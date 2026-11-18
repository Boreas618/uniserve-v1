//! A causal runner over the rank's owners: the KV cache and the recurrent state.
//!
//! The cache and the state belong to the rank ([`KvCacheMemory`], [`StateMemory`]). This
//! module is the one place their tensors and their per-batch work are joined to a runner, so
//! an engine states neither the tensors a forward is lent nor the order of a batch's page
//! work:
//!
//! - [`CausalLmRunner::over`] builds a runner from the owners: each attention group's planes,
//!   pooled page table and workspace, the state's live and checkpoint buffers, and the scratch
//!   page set when the cache opened one;
//! - [`page_work`] builds the hook that issues a batch's page work on the runner's stream, in
//!   the order the cache module documents.
//!
//! Both work over owners that hold no device (`KvCacheMemory::meta`, `StateMemory::meta`): the
//! runner then computes over meta tensors, and the hook keeps the cache's host copy of the
//! pooled tables and issues nothing.

use std::sync::{Arc, Mutex};

use crate::batch::{Batch, StateDirection};
use nn::family::TextModel;
use nn::{KvCache, NO_SLOT};
use platform::{Event, Stream};
use storage::runtime::Run;

use super::speculation::DraftPool;
use super::{CausalLmConfig, CausalLmRunner, PageWork, PageWorkHook};
use crate::memory::cache::KvCacheMemory;
use crate::memory::state::StateMemory;
use crate::memory::tables::BlockTables;
use crate::runner::graphs::Graphs;
use crate::worker::{Error, Result};
use service::lock;

impl CausalLmRunner {
    /// A runner of `model` under `config` over the rank's `cache` and recurrent `state`.
    ///
    /// The runner is lent every attention group of the cache in the cache's order, with the
    /// group's pooled page table, and the state's live and checkpoint buffers; `graphs` and
    /// `stream` are as [`CausalLmRunner::new`] takes them. A cache whose scratch pages are
    /// open lends them for paired forwards, when the config states scratch pages
    /// ([`CausalLmRunner::lend_scratch`]).
    ///
    /// The owners outlive the runner, and their memory is written only on `stream`.
    ///
    /// # Errors
    ///
    /// As [`CausalLmRunner::new`]: a cache of other request slots or pages than the config's,
    /// or a state the model does not keep.
    pub(crate) fn over(
        model: Arc<dyn TextModel>,
        config: CausalLmConfig,
        graphs: Graphs,
        stream: Option<Stream>,
        cache: &KvCacheMemory,
        state: Option<&StateMemory>,
    ) -> Result<Self> {
        if cache.config().slots != config.slots {
            return Err(Error::Invalid(format!(
                "a runner of {} request slots over a cache of {}",
                config.slots,
                cache.config().slots
            )));
        }

        let mut groups = Vec::new();
        for group in attention_groups(cache) {
            groups.push(cache.lend(group, cache.page_table(group)?)?);
        }
        let buffers = match state {
            Some(state) => Some((state.live()?, state.checkpoint()?)),
            None => None,
        };

        let mut runner = Self::new(model, config, graphs, stream, groups, buffers)?;
        if cache.scratch_capacity() > 0 && runner.config.scratch_pages > 0 {
            runner.lend_scratch(cache)?;
        }
        // A runner over a cache that lays the draft cache out keeps it, whether or not it
        // speculates now: a kernel generation of other drafts verifies over it.
        if runner.config.drafts > 0 || cache.layout().draft_group().is_some() {
            runner.lend_draft(cache, state)?;
        }
        Ok(runner)
    }

    /// Lends the runner the draft cache of `cache`, and a runner that speculates the trail of
    /// `state` for a model that keeps a recurrent state and the drafter's state that `state`
    /// keeps ([`CausalLmRunner::set_draft_cache`]); a state whose requests own a slot per
    /// verified token keeps no trail.
    ///
    /// # Errors
    ///
    /// A cache that keeps no draft group or lies otherwise in the unit pool than the runner's
    /// config states, or what [`CausalLmRunner::set_draft_cache`] refuses: a state opened
    /// with no trail for a runner whose forwards lend no verify slots, or the reverse, or
    /// without the drafter's state for a drafter that keeps one.
    pub(crate) fn lend_draft(
        &mut self,
        cache: &KvCacheMemory,
        state: Option<&StateMemory>,
    ) -> Result<()> {
        let pool = DraftPool::of(cache.layout(), cache.config().units)?;
        if self.config.draft_pool != Some(pool) {
            return Err(Error::Invalid(format!(
                "the runner's config states the draft pool {:?} and the cache lays it out as \
                 {pool:?}",
                self.config.draft_pool
            )));
        }

        // The drafter's forwards state their rows' pages themselves: the table lent here is
        // not read.
        let unread = attention_groups(cache)
            .next()
            .ok_or_else(|| Error::Invalid("a draft cache beside no attention group".into()))?;
        let draft = cache.lend_draft(cache.page_table(unread)?)?;

        let (trail, drafter) = match state {
            Some(state) if self.config.drafts > 0 => (state.trail()?, state.drafter()?),
            _ => (None, None),
        };
        self.set_draft_cache(draft, trail, drafter)
    }

    /// Lends the runner the scratch page set `cache` opened, for its paired forwards: each
    /// attention group's planes over the scratch pages, and the set's page table, whose row `i`
    /// names the set's pages from page `i` on.
    ///
    /// # Errors
    ///
    /// A cache whose scratch pages are not open, or what [`CausalLmRunner::set_scratch`]
    /// refuses: a set of other pages than the config's.
    pub(crate) fn lend_scratch(&mut self, cache: &KvCacheMemory) -> Result<()> {
        let table = cache.scratch_table()?;

        let mut groups = Vec::new();
        for group in attention_groups(cache) {
            groups.push(KvCache {
                layers: cache.scratch_planes(group)?,
                ..cache.lend(group, table.clone())?
            });
        }
        self.set_scratch(groups)
    }
}

/// The indices of the attention groups of `cache`, in the order a model's layers index them.
fn attention_groups(cache: &KvCacheMemory) -> impl Iterator<Item = usize> + '_ {
    let groups = &cache.layout().groups;

    (0..groups.len()).filter(|group| groups[*group].attention())
}

/// The hook that issues a batch's page work for a runner over `cache` and `state`
/// ([`CausalLmRunner::set_page_work`]), on `stream`, the runner's; `None` for owners that hold
/// no device, whose work is checked, kept in the cache's host copy of the pooled tables and
/// not issued.
///
/// - [`PageWork::Tables`]: the cache applies the device block table's changes: the units the
///   buffer pool allocated are reset, and the pooled rows that changed are written.
/// - [`PageWork::Copies`]: the cache issues the copies of the blocks the batch takes; their
///   events are kept for [`PageWork::Ended`].
/// - [`PageWork::BeforeReplay`]: the batch's state copies are resolved through the tables,
///   each checkpoint's position to the one unit of the state region its entry holds in the
///   slot's table of the state group's class (the class of cache group `g` is class `g`), and
///   refused unless every one names an entry of one unit and every keep a writable one; then
///   every restore is issued.
/// - [`PageWork::AfterReplay`]: the keeps resolved before the replay are issued.
/// - [`PageWork::Ended`]: the copies issued since the last end are waited for.
/// - [`PageWork::Holds`]: the flag is set when the region is the cache's
///   (`KvCacheMemory::holds`).
/// - [`PageWork::BeforeShadow`]: what the rows of a paired forward have cached is copied into
///   the scratch pages each row takes, from the first page the hook is given for it, for every
///   attention group.
///
/// The cache is locked before the state, for the time of one call.
pub(crate) fn page_work(
    cache: Arc<Mutex<KvCacheMemory>>,
    state: Option<Arc<Mutex<StateMemory>>>,
    stream: Option<Stream>,
) -> PageWorkHook {
    // The events behind the copies issued for the batch being served.
    let mut copied: Vec<Event> = Vec::new();
    // The keeps of the batch whose replay is issued, resolved before it.
    let mut keeps: StateRuns = Vec::new();

    Box::new(move |work: PageWork<'_>| {
        let mut cache = lock(&cache);

        match work {
            PageWork::Tables(changes) => cache.apply(changes, stream.as_ref())?,
            PageWork::Copies(copies) => {
                // The cache copies a unit's bytes as its own layout holds them: a copy of another
                // encoding is refused, and the worker reports it failed.
                if let Some(copy) = copies.iter().find(|copy| copy.encodings.is_some()) {
                    return Err(Error::Unsupported(format!(
                        "the copies of request {} read a block of another encoding, which the \
                         cache does not convert",
                        copy.request.0
                    )));
                }
                let runs: Vec<(Run, Run)> = copies
                    .iter()
                    .flat_map(|copy| copy.runs.iter().copied())
                    .collect();
                copied.extend(cache.copy(&runs, stream.as_ref())?);
            }
            PageWork::BeforeReplay(batch, tables) => {
                let (restores, kept) = state_copies(&cache, batch, tables)?;
                keeps = kept;

                if let Some(state) = &state {
                    lock(state).copy(&cache, &restores, StateDirection::Restore)?;
                }
            }
            PageWork::AfterReplay => {
                let kept = std::mem::take(&mut keeps);
                if let Some(state) = &state {
                    lock(state).copy(&cache, &kept, StateDirection::Keep)?;
                }
            }
            PageWork::Ended => {
                keeps.clear();
                for event in copied.drain(..) {
                    event.synchronize()?;
                }
            }
            PageWork::Holds(region, held) => held.set(cache.holds(region)),
            PageWork::BeforeShadow(batch, first) => {
                let Some(stream) = &stream else {
                    return Ok(());
                };

                // A row's history is the keys before its own tokens, which the forward writes
                // itself; a row on no slot copies nothing.
                let rows = &batch.columns.rows;
                let slots: Vec<u32> = batch
                    .columns
                    .slots
                    .iter()
                    .map(|slot| u32::try_from(*slot).unwrap_or(NO_SLOT))
                    .collect();
                let lengths: Vec<u32> = rows
                    .seq_lens
                    .iter()
                    .zip(rows.query_offsets.windows(2))
                    .map(|(keys, span)| (keys - (span[1] - span[0])).max(0) as u32)
                    .collect();

                for group in attention_groups(&cache) {
                    cache.copy_history(group, &slots, &lengths, first, stream)?;
                }
            }
        }
        Ok(())
    })
}

/// State copies resolved through the device block table: each a request slot and the state unit
/// of the block its checkpoint's position names.
pub(crate) type StateRuns = Vec<(u32, u32)>;

/// The state copies of `batch` resolved through `tables`: `(restores, keeps)`, each a request
/// slot and the state unit of the block its checkpoint's position names in the slot's table of
/// the state group's class, one unit of the state region (`specs/engine-worker-port.md` §1.6).
///
/// # Errors
///
/// [`Error::Invalid`] for a batch that copies state over a cache without a state group, a
/// checkpoint whose position holds no entry or an entry of other than one unit, and a keep into
/// one a batch may not write: the batch is refused before anything of it is launched.
pub(crate) fn state_copies(
    cache: &KvCacheMemory,
    batch: &Batch,
    tables: &BlockTables,
) -> Result<(StateRuns, StateRuns)> {
    let copies = &batch.columns.state_copies;
    if copies.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let class = cache
        .layout()
        .state_group()
        .and_then(|group| u8::try_from(group).ok())
        .ok_or_else(|| {
            Error::Invalid(format!(
                "batch {} copies recurrent state, and the cache keeps no state group",
                batch.id.0
            ))
        })?;

    let (mut restores, mut keeps) = (Vec::new(), Vec::new());
    for copy in copies {
        let units = tables
            .units(copy.slot, class, copy.position)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "batch {} copies the state of slot {} at position {} of class {class}, \
                     which holds no entry",
                    batch.id.0, copy.slot, copy.position
                ))
            })?;
        if units.len() != 1 {
            return Err(Error::Invalid(format!(
                "batch {} copies the state of slot {} at position {} of class {class}, whose \
                 entry holds units {}..{}, and a checkpoint is one unit of the state region",
                batch.id.0, copy.slot, copy.position, units.start, units.end
            )));
        }
        let unit = units.start;

        match copy.direction {
            StateDirection::Restore => restores.push((copy.slot, unit)),
            StateDirection::Keep if tables.writable(copy.slot, class, copy.position) => {
                keeps.push((copy.slot, unit));
            }
            StateDirection::Keep => {
                return Err(Error::Invalid(format!(
                    "batch {} keeps the state of slot {} at position {} of class {class}, \
                     which is read-only",
                    batch.id.0, copy.slot, copy.position
                )));
            }
        }
    }
    Ok((restores, keeps))
}

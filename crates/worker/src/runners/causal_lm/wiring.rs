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
//!   the order the cache module documents;
//! - [`draft_pages`] builds what states a request slot's installed draft pages for a runner
//!   that speculates.
//!
//! Both work over owners that hold no device (`KvCacheMemory::meta`, `StateMemory::meta`): the
//! runner then computes over meta tensors, and the hook keeps the cache's host mirror of the
//! page tables and issues nothing.

use std::sync::{Arc, Mutex};

use ipc::execution::{Batch, StateDirection};
use nn::family::TextModel;
use nn::{KvCache, NO_SLOT};
use platform::{Event, Stream};

use super::speculation::{DraftPages, DraftPool};
use super::{CausalLmConfig, CausalLmRunner, PageWork, PageWorkHook};
use crate::graphs::Graphs;
use crate::memory::cache::KvCacheMemory;
use crate::memory::state::StateMemory;
use crate::{Error, Result, lock};

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
    pub fn over(
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
        if runner.config.drafts > 0 {
            runner.lend_draft(cache, state)?;
        }
        Ok(runner)
    }

    /// Lends a runner that speculates the draft cache of `cache`, and the trail of `state`
    /// for a model that keeps a recurrent state ([`CausalLmRunner::set_draft_cache`]).
    ///
    /// # Errors
    ///
    /// A cache that keeps no draft group or lies otherwise in the unit pool than the runner's
    /// config states, or what [`CausalLmRunner::set_draft_cache`] refuses: a state opened
    /// with no trail.
    pub fn lend_draft(&mut self, cache: &KvCacheMemory, state: Option<&StateMemory>) -> Result<()> {
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

        let trail = match state {
            Some(state) => state.trail()?,
            None => None,
        };
        self.set_draft_cache(draft, trail)
    }

    /// Lends the runner the scratch page set `cache` opened, for its paired forwards: each
    /// attention group's planes over the scratch pages, and the set's page table for the rows
    /// of the runner's largest bucket at the config's scratch pages a row.
    ///
    /// # Errors
    ///
    /// A cache whose scratch pages are not open or hold fewer pages than the runner's rows
    /// address, or what [`CausalLmRunner::set_scratch`] refuses.
    pub fn lend_scratch(&mut self, cache: &KvCacheMemory) -> Result<()> {
        let rows = self.scratch_rows();
        let table = cache.scratch_table(rows, self.config.scratch_pages)?;

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
/// no device, whose work is checked, kept in the cache's host mirror and not issued.
///
/// - [`PageWork::BeforeReplay`]: the batch's rows of the pooled tables are installed, its new
///   units reset and filled from a leased prefix, its pages loaded from and offloaded to the
///   host tier, and its recurrent state restored, in that order.
/// - [`PageWork::AfterReplay`]: the batch's recurrent state is kept, and the offloads issued
///   before the replay are waited for, so the batch's completion is behind them.
/// - [`PageWork::BeforeShadow`]: what the rows of a paired forward have cached is copied into
///   their scratch pages, `scratch_pages` pages a row, for every attention group.
///
/// The cache is locked before the state, for the time of one call.
pub fn page_work(
    cache: Arc<Mutex<KvCacheMemory>>,
    state: Option<Arc<Mutex<StateMemory>>>,
    stream: Option<Stream>,
    scratch_pages: u32,
) -> PageWorkHook {
    // The event behind the offloads of the batch whose replay is issued.
    let mut offloads: Option<Event> = None;

    Box::new(move |batch: &Batch, work: PageWork| {
        let mut cache = lock(&cache);
        let copies = &batch.columns.state_copies;

        match work {
            PageWork::BeforeReplay => {
                cache.install(&batch.block_tables, stream.as_ref())?;
                cache.apply_units(&batch.cache_units, stream.as_ref())?;

                offloads = match &stream {
                    Some(stream) => {
                        cache.copy_pages(&batch.page_loads, &batch.page_offloads, stream)?
                    }
                    None if batch.page_loads.is_empty() && batch.page_offloads.is_empty() => None,
                    None => {
                        return Err(Error::Invalid(format!(
                            "batch {} copies pages of a cache that holds no device",
                            batch.id.0
                        )));
                    }
                };

                if let Some(state) = &state {
                    lock(state).copy(&cache, copies, StateDirection::Restore)?;
                }
            }
            PageWork::AfterReplay => {
                if let Some(state) = &state {
                    lock(state).copy(&cache, copies, StateDirection::Keep)?;
                }
                if let Some(event) = offloads.take() {
                    event.synchronize()?;
                }
            }
            PageWork::BeforeShadow => {
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
                    cache.copy_history(group, &slots, &lengths, scratch_pages, stream)?;
                }
            }
        }
        Ok(())
    })
}

/// What states a request slot's installed pages of the draft group of `cache`, for a runner
/// that speculates ([`CausalLmRunner::set_draft_pages`]): the slot's first draft page and the
/// units of its pages from there on, as the scheduler last installed them, and `None` for a
/// slot that holds none or a cache that keeps no draft group.
pub fn draft_pages(cache: Arc<Mutex<KvCacheMemory>>) -> DraftPages {
    Box::new(move |slot: u32| {
        let cache = lock(&cache);
        let group = cache.layout().draft_group()? as u32;

        cache
            .installed_pages(slot, group)
            .map(|(first, units)| (first, units.to_vec()))
    })
}

//! One step's work across the boundary from the scheduling mechanism to the worker (2.4, 4.4).
//!
//! The mechanism decides this step's work and writes it in the layout the kernels read; the worker
//! pads it to a bucket and serves it. This module is that crossing, and the one statement of it the
//! scheduler and the worker both read:
//!
//! - [`Batch`] is one phase of one step: the host columns the mechanism wrote, the calls whose
//!   progress the batch advances, the [`Shape`] the runner pads to, and the generation it serves
//!   with;
//! - [`BatchColumns`] and [`BatchRows`] are the token batch's host-side layout: plain host vectors,
//!   because the mechanism owns no device and no stream. Staging them into the pinned arena,
//!   uploading them and building the device-side token batch and cache views is the runner's work;
//! - [`MediaColumns`] are a media computation's columns, per request as the model prepared them,
//!   and [`StagedRequest`] is the mechanism's record of one such request;
//! - [`BatchOutput`] and [`CallOutput`] are what the step produced, and [`Submission`] is what the
//!   caller holds until the worker responses;
//! - [`Batches`] is what the mechanism asks of the worker that serves it, which the worker
//!   implements over its devices.
//!
//! Everything here is host data and host synchronization: no device, no kernel, no model. The
//! commit table's and the sampler's columns are the module vocabulary's
//! (`proto::module::{commit, sampler}`), and a denoising step's latent layout is the video
//! vocabulary's (`proto::video::Layout`).

use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// The parts this batch has in common with the scheduler's batch (`crate::batch`), which states
// them once.
pub use crate::batch::{BatchId, Extent, StateDirection};
use crate::{
    BufferId, CallStatus, ErrorCode, MainProcessInfo, PhaseId, RequestKey, TimingCounters, UnitId,
    ValidationError, ValidationResult,
};

/// A failure the worker reports: what a failed [`Outcome`] carries and what a refused request of
/// [`Batches`] responses. The scheduler reads it as text; the worker that raised it can downcast it
/// to its own error.
pub type Fault = dyn std::error::Error + Send + Sync;

/// How an operation of the worker ended.
///
/// `E` is the failure's type: the worker's own error inside the worker, and a [`Fault`] once the
/// outcome crosses to the mechanism in a [`Submission`].
#[derive(Debug)]
pub enum Outcome<T = (), E: ?Sized = Fault> {
    /// The operation completed with its value.
    Success(T),
    /// The operation failed.
    Failed(Arc<E>),
    /// The operation was cancelled before it completed.
    Cancelled,
}

impl<T: Clone, E: ?Sized> Clone for Outcome<T, E> {
    fn clone(&self) -> Self {
        match self {
            Self::Success(value) => Self::Success(value.clone()),
            Self::Failed(error) => Self::Failed(Arc::clone(error)),
            Self::Cancelled => Self::Cancelled,
        }
    }
}

/// One completion shared by an operation and its consumers: its outcome, recorded once, the
/// waiters it wakes and the observers it responses.
///
/// `C` is the observer type (the worker's retirement notifications and host task callbacks; a
/// submission has none), `T` the operation's value and `E` its failure. Completing an operation
/// responds to its callbacks so the owner can dispatch them after releasing its lock.
pub struct Completion<C, T = (), E: ?Sized = Fault> {
    state: Mutex<CompletionState<C, T, E>>,
    /// Signalled when the outcome lands, so a waiter sleeps instead of polling.
    ready: Condvar,
}

struct CompletionState<C, T, E: ?Sized> {
    outcome: Option<Outcome<T, E>>,
    callbacks: Vec<C>,
}

/// A completion that already has its outcome: the second outcome is not recorded.
#[derive(Debug, thiserror::Error)]
#[error("operation has already completed")]
pub struct Settled;

impl<C, T, E: ?Sized> Default for Completion<C, T, E> {
    fn default() -> Self {
        Self {
            state: Mutex::new(CompletionState {
                outcome: None,
                callbacks: Vec::new(),
            }),
            ready: Condvar::new(),
        }
    }
}

impl<C, T: Clone, E: ?Sized> Completion<C, T, E> {
    fn state(&self) -> MutexGuard<'_, CompletionState<C, T, E>> {
        // No user code mutates state under the lock, so a panicking visitor cannot leave a
        // partially completed operation.
        lock(&self.state)
    }

    /// Whether the operation ended.
    pub fn done(&self) -> bool {
        self.state().outcome.is_some()
    }

    /// Whether the operation ended successfully.
    pub fn succeeded(&self) -> bool {
        matches!(self.state().outcome, Some(Outcome::Success(_)))
    }

    /// The outcome, once the operation ended.
    pub fn outcome(&self) -> Option<Outcome<T, E>> {
        self.state().outcome.clone()
    }

    /// Waits for the operation's outcome.
    pub fn wait(&self) -> Outcome<T, E> {
        let mut state = self.state();
        loop {
            if let Some(outcome) = &state.outcome {
                return outcome.clone();
            }
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Waits at most `timeout` for the outcome: what a serve loop does between batches, where
    /// `Duration::ZERO` is a poll.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Outcome<T, E>> {
        let deadline = Instant::now() + timeout;
        let mut state = self.state();
        while state.outcome.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            state = self
                .ready
                .wait_timeout(state, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        state.outcome.clone()
    }

    /// Registers a callback, or responds to it for immediate dispatch when the operation ended.
    pub fn subscribe(&self, callback: C) -> Option<C> {
        let mut state = self.state();
        if state.outcome.is_some() {
            return Some(callback);
        }
        state.callbacks.push(callback);
        None
    }

    /// Records `outcome` when the operation has none yet, wakes every waiter, and responds to the
    /// callbacks to run; `None` when an outcome was already recorded, which this one does not
    /// replace.
    pub fn settle(&self, outcome: Outcome<T, E>) -> Option<Vec<C>> {
        let mut state = self.state();
        if state.outcome.is_some() {
            return None;
        }
        state.outcome = Some(outcome);
        let callbacks = std::mem::take(&mut state.callbacks);
        drop(state);
        self.ready.notify_all();
        Some(callbacks)
    }

    /// Cancels an unresolved completion, responding whether it is now cancelled and the callbacks
    /// to run. This does not stop its producer.
    pub fn cancel(&self) -> (bool, Vec<C>) {
        let mut state = self.state();
        match state.outcome {
            Some(Outcome::Cancelled) => return (true, Vec::new()),
            Some(_) => return (false, Vec::new()),
            None => {}
        }
        state.outcome = Some(Outcome::Cancelled);
        let callbacks = std::mem::take(&mut state.callbacks);
        drop(state);
        self.ready.notify_all();
        (true, callbacks)
    }
}

impl<C, T: Clone, E> Completion<C, T, E> {
    /// Records the operation's result, as [`Completion::settle`] does.
    ///
    /// # Errors
    ///
    /// [`Settled`] when the operation already has an outcome.
    pub fn complete(&self, result: Result<T, E>) -> Result<Vec<C>, Settled> {
        let outcome = match result {
            Ok(value) => Outcome::Success(value),
            Err(error) => Outcome::Failed(Arc::new(error)),
        };
        self.settle(outcome).ok_or(Settled)
    }
}

/// What the interface returns by itself, in plain text: a worker that serves no such work, or a
/// submission that completed without an outcome.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Refusal(pub &'static str);

/// Locks `mutex`, reading through a poisoned lock: no code here mutates state in a way a panicking
/// waiter could leave half done.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The scheduling mechanism's name for one submission: monotonic in the order the mechanism made
/// them, so the control thread can report which batch a stalled job belongs to (2.4).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SubmissionId(pub u64);

/// The mechanism's name for one call: a request's phase in this batch, not the request itself.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct CallId(pub u64);

/// One call's completion, as the device half of 4.4 states it.
///
/// A batch advances a set of calls and each of them gets one entry here, whether or not it sampled
/// a token, because the mechanism's completion path is per call. A call this worker refuses never
/// reaches this type: [`Batches::submit`] refuses
/// synchronously for a batch it cannot serve, so an entry here is what execution did, and a
/// [`CallStatus::Error`] entry is always a device failure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallOutput {
    /// The call the batch advanced.
    pub call: CallId,
    /// Whether execution performed the call.
    pub status: CallStatus,
    /// Why it failed, when it did: the class the mechanism acts on (`NumericalFault` fails the
    /// rows that computed non-finite logits, `InputError` rejects the request with this rank's
    /// message, `ResourceError` fails the call).
    pub error_code: Option<ErrorCode>,
    /// What this rank said about a refused input, when it refused one.
    pub error_message: Option<String>,
    /// The tokens the sampler accepted for this call, in the order the call's sampling rows took
    /// them. A call whose rows sampled nothing (a prefill chunk that ends no prompt, a padded row)
    /// has none, which is not the same as a failure.
    pub tokens: Vec<u32>,
    /// The slot's committed count for a text call; a latent trajectory reports its own step here
    /// once `slot_commit` (`proto::module::commit`) gains a per-slot step column, and until then this field is the commit
    /// count for every call kind.
    ///
    /// It is read from the commit table's sampling position, which a text slot advances once per
    /// committed token and a latent slot advances once per denoising step.
    pub step_index: u32,
    /// The cache extent execution initialized, rejected speculative positions included: the slot's
    /// logical position after the commit.
    pub cache_computed_len: u32,
    /// The accepted prefix a successor may read: the slot's sampling position after the commit.
    pub cache_visible_len: u32,
    /// The scores of each accepted token, aligned with [`CallOutput::tokens`], on the raw
    /// distribution: `log_softmax` of the model's logits before penalties, temperature and
    /// filtering (vLLM's default `raw_logprobs`).
    ///
    /// A token's entries are the token itself - its log-probability and its one-based competition
    /// rank - then its position's most likely tokens, the largest first, as many as the
    /// deployment's `max_logprobs`; the token may be among them. A causal-LM step reports them for
    /// every token it sampled, whether or not the request asked, because a captured graph cannot
    /// branch on a request parameter; the scheduler keeps the candidates each request asks for and
    /// drops the rest. A call that sampled nothing reports none.
    pub token_logprobs: Vec<Vec<crate::TokenLogprob>>,
    /// The scores of each prompt position the call scored, in input order, in the form of
    /// [`CallOutput::token_logprobs`]: the prompt token itself, then the position's most likely
    /// tokens.
    pub prompt_logprobs: Vec<Vec<crate::TokenLogprob>>,
    /// The finished artifact and its storage lifetime, when the call produced one.
    pub media_output: Option<crate::MediaOutput>,
    /// The drafter's proposals after the last committed token.
    pub draft_token_ids: Vec<u32>,
    /// A video latent preparation's per-rank page layout.
    pub latent_layouts: Vec<crate::RankLatentLayout>,
}

impl CallOutput {
    /// A completed call that sampled `tokens` and stands at `step_index`, from the slot state
    /// `state` the commit table left.
    pub fn completed(call: CallId, tokens: Vec<u32>, state: RowState) -> Self {
        Self {
            call,
            status: CallStatus::Ok,
            error_code: None,
            error_message: None,
            tokens,
            step_index: state.step_index,
            cache_computed_len: state.cache_computed_len,
            cache_visible_len: state.cache_visible_len,
            token_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            // A text call produces no artifact, drafts no token and lays out no latent page: the
            // mechanism reads these as empty, which is what happened.
            media_output: None,
            draft_token_ids: Vec::new(),
            latent_layouts: Vec::new(),
        }
    }
}

/// Where the commit table left one call's slot, as the completion reads it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowState {
    /// The slot's trajectory position: the denoising trajectory's next step.
    pub step_index: u32,
    /// The cache extent execution initialized, rejected speculative positions included.
    pub cache_computed_len: u32,
    /// The accepted prefix a successor may read.
    pub cache_visible_len: u32,
}

/// What one serving step produced (4.4).
///
/// The mechanism learns it from the batch's [`Submission`], builds the next step's input ids from
/// the tokens, and folds the timing into the statistics of the calls' domain. A step that samples
/// nothing (a chunk of a prefill that ends no prompt, a warmup, a generation command) completes
/// with no call and nothing measured.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BatchOutput {
    /// One entry per call the batch advanced, in the order the batch named them.
    pub calls: Vec<CallOutput>,
    /// What the batch measured, in the segments this worker can observe.
    ///
    /// `queued_us` is the wait from the mechanism's submission to this device thread starting the
    /// batch, `device_us` the replay's launch to the stream's completion, and `host_us` the rest of
    /// the step on the host. `copy_us` is zero in-process and states a fact rather than a
    /// measurement: nothing is transferred for the completion, since the mechanism reads it here.
    pub timing: TimingCounters,
    /// What this rank's counters, probes and profiling session accumulated since its last result,
    /// which the scheduler adds into its statistics. Empty and absent while nothing observes it.
    pub forward_stats: Option<crate::ForwardStats>,
}

/// One batch: one phase of one step, as the mechanism assembles it.
///
/// `shape` is what the runner pads to. It is the mechanism's statement of the batch's size — the
/// rows and tokens it wrote — and the runner serves the batch from the smallest captured bucket
/// that holds it; a shape no bucket holds is an error, never an eager fallback (4.5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    /// The batch's number, for the stall report and the completion the mechanism reads (2.4).
    pub id: BatchId,
    /// The phase this batch advances, as the scheduler addresses it (`crate::PhaseId`).
    pub phase: PhaseId,
    /// The worker group whose ranks run the batch, by the name its [`Inventory`] states: a
    /// process whose devices serve several groups, such as a deployment's prefill and decode
    /// groups, hands the batch to the devices of this group alone.
    pub group: String,
    /// The calls whose progress this batch advances, in scheduling order.
    pub calls: Vec<CallId>,
    /// The rows and tokens the mechanism wrote, and what the batch computes.
    pub shape: Shape,
    /// The host columns the mechanism wrote, in the layout the phase's kernels read.
    pub columns: BatchColumns,
    /// The generation whose kernels serve this batch (4.2, 7.3).
    pub generation: u64,
    /// What the mechanism allocated for this batch's cache pages: per group, the units it acquired
    /// and the `(source, destination)` copies a leased prefix needs before a call reads it.
    ///
    /// The mechanism computes these and owns the source lease, which runs to completion; the
    /// worker applies them on the batch's stream before the forward, so a destination holds the
    /// source's bytes by stream order and the graph needs no extra barrier. A batch that reads a
    /// fresh prefix carries no allocation, and one that reuses a cached prefix cannot be served
    /// correctly without it ([`CacheUnitAllocation`]).
    pub cache_units: Vec<crate::CacheUnitAllocation>,
    /// The batch's slot rows: one `crate::BlockTable` per request slot and cache group, as the
    /// mechanism's own allocation of pages states them.
    ///
    /// The worker installs these into the cache's pooled table before the replay, because the
    /// gather that builds the attention kernels' block-table column reads **row `slot` of that
    /// table**. The row is the only place a request's pages live for the gather: a batch whose rows
    /// are not installed gathers whatever the slot held before - zeros, or another request's pages -
    /// and the graph then runs clean against the wrong memory. That failure looks like a model
    /// defect rather than a cache one, so it is refused at submit rather than left to the graph.
    pub block_tables: Vec<crate::BlockTable>,
    /// Pages of the host KV cache the batch's new units are filled with, on the runner's copy stream
    /// before its forward: a request whose prefix the host held continues from them.
    pub page_loads: Vec<crate::PageCopy>,
    /// Published pages the batch copies to the host KV cache beside its forward; the batch's output
    /// waits for the copies, and the mechanism commits the pages once it read that output.
    pub page_offloads: Vec<crate::PageCopy>,
    /// The stored trajectories the batch's latent preparations open at, by request: the pages of
    /// the host latent pools the engine pinned for each resumed request, which the denoiser copies
    /// into its trajectory before the call completes (`crate::Call::latent_restore`). A preparation
    /// with none opens its trajectory at step zero.
    pub latent_restores: Vec<(crate::RequestKey, crate::LatentRestore)>,
    /// The encoder outputs of the batch's requests in the host encoder pool, by request
    /// (`crate::Call::host_output`): an import reads the output's rows from the pages instead of
    /// computing them, and an export writes the rows it computed to the reserved pages before the
    /// call completes.
    pub host_outputs: Vec<(crate::RequestKey, crate::HostOutput)>,
    /// The images the batch's vision encodings read, by call: each encodes into the feature
    /// product its call declares, which later prefills read ([`VisionRows`]).
    pub images: Vec<ImageEncoding>,
    /// Every persistent product the batch's calls read or write, once per backing rank: the rank
    /// of the batch's group whose buffer pool holds the span, and the span there.
    ///
    /// The scheduler places every product; the worker holds no record of where one lies, so a
    /// runner finds a product it names ([`MediaColumns::reads`], [`MediaColumns::writes`],
    /// [`ImageEncoding::output`], [`VisionRows::feature`]) here, at the span of its own rank, and
    /// the span of each other rank it copies an output to.
    pub buffers: Vec<(u32, crate::BufferAllocation)>,
    /// The persistent products whose last reader has completed, which the rank releases before it
    /// serves the batch (`crate::BatchCommand::Free`).
    pub frees: Vec<crate::BufferId>,
    /// A media computation's columns, and `None` for every causal-LM batch.
    ///
    /// One batch is one computation (2.4): a media batch states its inputs here and carries no token
    /// column a language model reads, and a token batch carries no media inputs. The runner that
    /// receives one knows which class it serves, so the field is read by the media runners alone.
    pub media: Option<MediaColumns>,
}

impl Batch {
    /// Checks the rules that span the batch's parts, so a batch that crosses a process boundary is
    /// refused before any of it is staged.
    ///
    /// Every nested record passes its own `validate`, and across them:
    ///
    /// - the batch names each call once;
    /// - a token batch has one call row per call and one call per request slot (a request runs one
    ///   call per step), and its columns fit its shape ([`BatchColumns::check`]);
    /// - each `(slot, group)` has at most one block table and one unit allocation, and an
    ///   allocation's units lie inside the table of the same slot and group;
    /// - a page read fills only units the batch newly allocates, which the worker resets first;
    /// - a recurrent-state copy moves a page of one of its slot's tables, restores precede
    ///   checkpoints, and each checkpoint responds to a row's checkpoint token count;
    /// - a request has at most one stored trajectory to restore and one host encoder output;
    /// - a product has at most one span on each rank.
    pub fn validate(&self) -> ValidationResult<()> {
        let mut calls = std::collections::HashSet::with_capacity(self.calls.len());
        ensure_valid!(
            self.calls.iter().all(|call| calls.insert(*call)),
            "a batch repeats a call"
        );

        // A token batch's columns state one row range per call; a media batch carries its inputs
        // in `media` and token columns only when it reads a prompt.
        if self.media.is_none() {
            ensure_valid!(
                self.columns.calls() == self.calls.len(),
                "a token batch states a different number of call rows than calls"
            );
            self.columns.check(&self.shape)?;
        }
        let mut slots = std::collections::HashSet::with_capacity(self.columns.calls());
        ensure_valid!(
            self.columns
                .rows
                .calls
                .iter()
                .all(|row| slots.insert(row.request)),
            "a batch carries two calls for one request"
        );
        if let Some(media) = &self.media {
            media.check()?;
        }

        let mut tables = std::collections::HashMap::with_capacity(self.block_tables.len());
        for table in &self.block_tables {
            table.validate()?;
            ensure_valid!(
                tables
                    .insert((table.request_pool_idx, table.group_id), table)
                    .is_none(),
                "a batch repeats a block table"
            );
        }

        let mut allocated = std::collections::HashSet::with_capacity(self.cache_units.len());
        for allocation in &self.cache_units {
            allocation.validate()?;
            let identity = (allocation.request_pool_idx, allocation.group_id);
            ensure_valid!(
                allocated.insert(identity),
                "a batch repeats a cache-unit allocation"
            );
            let table = tables.get(&identity).ok_or_else(|| {
                invalid_message!("a cache-unit allocation has no matching block table")
            })?;
            ensure_valid!(
                allocation
                    .unit_ids
                    .iter()
                    .all(|unit| table.unit_ids.contains(unit)),
                "a cache-unit allocation is outside its block table"
            );
        }

        // A page read overwrites whole units, which is only sound for units this batch acquired.
        let new_units = self
            .cache_units
            .iter()
            .flat_map(|allocation| allocation.unit_ids.iter())
            .collect::<std::collections::HashSet<_>>();
        for copy in self.page_loads.iter().chain(&self.page_offloads) {
            copy.validate()?;
        }
        ensure_valid!(
            self.page_loads
                .iter()
                .all(|copy| copy.units.iter().all(|unit| new_units.contains(unit))),
            "a page read fills units that are not new units of the batch"
        );

        self.check_state_copies(&slots, &tables)?;

        let mut restored = std::collections::HashSet::with_capacity(self.latent_restores.len());
        for (request, restore) in &self.latent_restores {
            restore.validate()?;
            ensure_valid!(
                restored.insert(*request),
                "a batch restores one trajectory twice"
            );
        }
        let mut hosted = std::collections::HashSet::with_capacity(self.host_outputs.len());
        ensure_valid!(
            self.host_outputs
                .iter()
                .all(|(request, _)| hosted.insert(*request)),
            "a batch names two host encoder outputs for one request"
        );

        // A product has one span in each rank's pool: two spans for one product on one rank would
        // leave the runner unable to say which one it reads.
        let mut placed = std::collections::HashSet::with_capacity(self.buffers.len());
        for (rank, allocation) in &self.buffers {
            allocation.validate()?;
            ensure_valid!(
                placed.insert((*rank, allocation.buffer)),
                "a batch places one product twice on one rank"
            );
        }
        Ok(())
    }

    /// Checks the recurrent-state copies against the calls and tables of the batch: a copy moves a
    /// state page of one of the batch's request slots, every restore precedes every checkpoint (the
    /// restores run before the forward and the checkpoints after it), a slot restores at most once,
    /// and each checkpoint responds to one row's non-zero `checkpoints` entry.
    fn check_state_copies(
        &self,
        slots: &std::collections::HashSet<u32>,
        tables: &std::collections::HashMap<(u32, u32), &crate::BlockTable>,
    ) -> ValidationResult<()> {
        let copies = &self.columns.state_copies;
        let mut restored = std::collections::HashSet::with_capacity(copies.len());
        for copy in copies {
            ensure_valid!(!copy.units.is_empty(), "a state copy names no unit");
            ensure_valid!(
                slots.contains(&copy.slot),
                "a state copy names a slot no call of the batch runs on"
            );
            // The copy's page is one of the slot's tables' pages; the copy does not name its
            // group, so any table of the slot may hold it.
            ensure_valid!(
                tables.iter().any(|((slot, _), table)| *slot == copy.slot
                    && copy.units.iter().all(|unit| table.unit_ids.contains(unit))),
                "a state copy's units lie in no block table of its slot"
            );
            if copy.direction == StateDirection::Restore {
                ensure_valid!(
                    restored.insert(copy.slot),
                    "a batch restores one slot's state twice"
                );
            }
        }
        ensure_valid!(
            copies.is_sorted_by_key(|copy| copy.direction != StateDirection::Restore),
            "a state checkpoint precedes a restore"
        );
        let keeps = copies
            .iter()
            .filter(|copy| copy.direction == StateDirection::Keep)
            .count();
        let kept_rows = self
            .columns
            .checkpoints
            .iter()
            .filter(|tokens| **tokens != 0)
            .count();
        ensure_valid!(
            keeps == kept_rows,
            "the batch's state checkpoints do not respond to its rows' checkpoint tokens"
        );
        Ok(())
    }
}

impl CallOutput {
    /// Checks that the completion's fields agree with its status.
    ///
    /// Rules that relate a completion to the kind of the call it responds belong to the mechanism,
    /// which holds the call.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.cache_visible_len <= self.cache_computed_len,
            "a completion's visible cache length exceeds its computed length"
        );
        match self.status {
            CallStatus::Error => ensure_valid!(
                self.error_code.is_some(),
                "an error completion must carry an error code"
            ),
            CallStatus::Ok | CallStatus::Predicated => ensure_valid!(
                self.error_code.is_none(),
                "a non-error completion must not carry an error code"
            ),
        }
        ensure_valid!(
            self.error_message.is_none() || self.error_code == Some(ErrorCode::InputError),
            "only an input-error completion states why the rank refused its request"
        );
        ensure_valid!(
            self.token_logprobs.is_empty() || self.token_logprobs.len() == self.tokens.len(),
            "a completion's token log-probabilities do not align with its tokens"
        );
        if self.status != CallStatus::Ok {
            ensure_valid!(
                self.latent_layouts.is_empty(),
                "latent layouts belong to a successful completion"
            );
        }
        if self.status == CallStatus::Predicated {
            ensure_valid!(
                self.tokens.is_empty() && self.draft_token_ids.is_empty(),
                "a predicated completion must select its parent without semantic output"
            );
        }
        if let Some(output) = &self.media_output {
            output.handle.validate()?;
            ensure_valid!(
                output.bytes > 0,
                "a media output's byte length must be positive"
            );
        }
        Ok(())
    }
}

impl BatchOutput {
    /// Checks every completion, and that the output responds only calls `batch` carried, each once.
    pub fn validate(&self, batch: &Batch) -> ValidationResult<()> {
        let mut responded = std::collections::HashSet::with_capacity(self.calls.len());
        for output in &self.calls {
            output.validate()?;
            ensure_valid!(
                batch.calls.contains(&output.call),
                "a completion responds to a call its batch did not carry"
            );
            ensure_valid!(
                responded.insert(output.call),
                "a batch output repeats a call"
            );
        }
        Ok(())
    }
}

/// One host-side call of a batch: which request, which phase, which rows it occupies.
///
/// The mechanism builds these from the scheduler's batch; the worker uploads the columns
/// [`BatchRows`] carries into the device views a model reads (`nn::TokenBatch`). A call's `rows`
/// are the half-open row range it owns in the batch's device columns.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct BatchRow {
    /// The request slot the call belongs to.
    pub request: u32,
    /// What the call does.
    pub kind: RowKind,
    /// Which attention group the call's keys and values belong to.
    pub cache_group: u32,
    /// The rows this call owns in the batch's device columns.
    pub rows: std::ops::Range<u32>,
}

/// What one call of a batch computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum RowKind {
    /// Prompt tokens of a request.
    Prefill,
    /// One generated token per request.
    Decode,
    /// Speculative verification rows: drafted tokens whose logits the model produces.
    Verify,
    /// One denoising step over a block of tokens of a diffusion language model: one row per
    /// call, which reads the cache and writes none.
    Denoise,
}

/// The host columns of one batch, as the mechanism writes them and the worker uploads them.
///
/// Every column is already in the layout the device views of `nn::TokenBatch` have: the worker
/// copies these into the pinned arena and uploads them before the graph replay. The columns are
/// long-lived buffers the worker reuses every step, so assembling a batch does not allocate.
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchRows {
    /// The calls of the batch, in issue order.
    pub calls: Vec<BatchRow>,
    /// `[tokens]` rotary positions.
    pub positions: Vec<i32>,
    /// `[tokens]` cache slots; `-1` writes nothing.
    pub slot_mapping: Vec<i32>,
    /// `[rows]` keys visible to each row's last query.
    pub seq_lens: Vec<i32>,
    /// `[rows + 1]` packed query offsets.
    pub query_offsets: Vec<i32>,
    /// `[rows]` hidden rows whose logits are wanted.
    pub logits_rows: Vec<i32>,
}

impl BatchRows {
    /// Drop every column, keeping the buffers' capacity.
    pub fn clear(&mut self) {
        self.calls.clear();
        self.positions.clear();
        self.slot_mapping.clear();
        self.seq_lens.clear();
        self.query_offsets.clear();
        self.logits_rows.clear();
    }

    /// The number of packed query tokens.
    pub fn tokens(&self) -> usize {
        self.positions.len()
    }

    /// The number of rows (sequences) of the batch.
    pub fn rows(&self) -> usize {
        self.seq_lens.len()
    }
}

/// The host columns of one batch, in the layout the phase's kernels read.
///
/// The mechanism writes these from the request state it owns; the worker stages them into the
/// pinned arena, uploads them on the batch's stream and builds the device views the model reads.
/// The row layout is [`BatchRows`], the batch's host half — one definition, no conversion in the
/// mechanism. The runner pads to the bucket it serves and never reorders what the mechanism
/// wrote; a padded row carries `slot_mapping = -1` and a commit slot below zero, so it writes no
/// cache page and keeps no token state.
///
/// # The three counts
///
/// Every column here follows exactly one of three counts, and a column compared with the wrong one
/// is a mis-sized batch:
///
/// - **tokens** ([`BatchColumns::tokens`]): `input_ids` and the `[tokens]` columns of `rows` —
///   positions, slot mapping;
/// - **rows** ([`BatchColumns::rows`]): the sequences. `slots`, `parameters` and the `[rows]`
///   columns of `rows` — lengths, query offsets, the sampling rows — are one entry per row, and a
///   call may own several: a verification of three candidates is one call with three rows, and a
///   call's `BatchRow::rows` span names them;
/// - **calls** ([`BatchColumns::calls`]): the commit table, one row per call, because a call states
///   once how its step advances the slot's device state however many rows it evaluated.
///
/// A mechanism that fills these columns and a worker that stages them both read this, so the
/// rule is stated once; [`BatchColumns::check`] enforces it before a launch.
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchColumns {
    /// The rows and their calls, as the model reads them: positions, slot mapping, lengths, query
    /// offsets, the sampling rows and the calls.
    pub rows: BatchRows,
    /// `[tokens]` the ids the embedding op reads; the runner stages them to the descriptor's id
    /// element type.
    pub input_ids: Vec<i32>,
    /// `[rows]` the sampler's slot view: the runner's per-slot sampling state.
    pub slots: Vec<i32>,
    /// `rows x proto::module::sampler::PARAMETER_COLUMNS` f32 request parameters, row-major.
    pub parameters: Vec<f32>,
    /// `[rows]` where each row's draw lies: its request's Philox stream and the logical position
    /// of the token it samples (`crate::philox`). The runner turns each into the row's uniform
    /// before the replay, so a sampled token depends on its request's seed and position alone. A
    /// greedy row and a row that samples nothing carry one too; its draw is never read.
    pub draws: Vec<crate::philox::DrawAddress>,
    /// `slot_commit` (`proto::module::commit`)'s column table, column-major `[columns][rows]` i64; a negative commit slot
    /// belongs to a padded row and keeps its token state.
    pub commit: Vec<i64>,
    /// The KV page size in tokens, the same number the policy sees in `Room`.
    pub page_size: u32,
    /// The attention descriptor's static bound on a sequence's KV length.
    pub max_kv_len: u32,
    /// `[rows]` the absolute token count after which each row's recurrent state is kept for the
    /// prefix cache, zero for a row that keeps none; empty for a model that keeps no recurrent
    /// state. The model's recurrent layers write that state into the row's slot of the checkpoint
    /// banks, and the worker moves it into the page `state_copies` names.
    pub checkpoints: Vec<i32>,
    /// The recurrent-state copies between the request slots' banks and the cache's state pages
    /// this batch needs: every restore before the forward, every checkpoint after it.
    pub state_copies: Vec<StateCopy>,
    /// One per call, in call order: what the call does for its request's draft cache
    /// (`crate::Call::draft`), which a worker that speculates with its model's drafter reads; empty
    /// for a batch whose calls draft nothing.
    pub drafts: Vec<crate::DraftWork>,
    /// The image blocks the batch's prefill rows write, in token order: each block's token rows
    /// carry no id the embedding reads, and the worker writes the feature product's rows there.
    pub vision: Vec<VisionRows>,
    /// The logit adjustments the sampler applies before penalties, temperature and filtering:
    /// the slots' logit bias and allowed sets, and each row's banned tokens.
    pub adjustments: LogitAdjustments,
    /// `[tokens]` the prompt token each token's logits score, `-1` for a token that scores none;
    /// empty for a batch that scores no prompt token. A call that scores its prompt
    /// ([`crate::Call::prompt_scores`]) names, for each of its tokens, the prompt token after it:
    /// the next input token, and for its last the first prompt token after the call, so a prompt
    /// is scored at every position but its first however it is chunked (vLLM's alignment). A
    /// batch that names one is served by [`Variant::PrefillScored`].
    pub targets: Vec<i32>,
    /// The request slots this batch starts a request's sampling state on, strictly increasing by
    /// slot: the batch carrying a request's first token call on a worker names its slot, and the
    /// worker sets the slot's penalty counts to the request's own history before the batch
    /// samples ([`FreshSlot`]), so a request's penalties count its own tokens - wherever it
    /// generated them - and never those of the slot's previous request.
    pub fresh_slots: Vec<FreshSlot>,
    /// One entry per call of a [`Variant::TokenDenoise`] batch, and none for any other batch: the
    /// block of its request the call denoises and the step it runs. The rank holds each slot's
    /// block between steps; a step of a block that already stopped computes nothing and reports
    /// [`crate::CallStatus::Predicated`], so the mechanism may queue a step behind the one in
    /// flight without knowing whether that one stops the block.
    #[serde(default)]
    pub block_steps: Vec<crate::BlockStep>,
}

/// One request slot a batch starts a request's sampling state on.
///
/// A request does not always start on a worker with nothing generated: a request given from its
/// prefill group to a decode group generated its first token on the prefill group, and a migrated
/// or preempted request generated its tokens before it was admitted again. Its penalties count
/// those tokens all the same, so the slot's counts start at the request's history rather than at
/// zero: one count per occurrence of each token in `history`.
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshSlot {
    /// The request slot whose penalty counts start.
    pub slot: u32,
    /// The tokens the request generated before this batch, which its penalties count: empty for a
    /// request that generated none, or whose penalties count nothing.
    pub history: Vec<u32>,
}

/// One vision encoding of a batch: the image a call reads and the feature product it writes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageEncoding {
    /// The request the call belongs to.
    pub request: crate::RequestKey,
    /// The image's encoded bytes, base64 (`crate::Call::input_image`).
    pub image: std::sync::Arc<str>,
    /// The feature product the call declares (`crate::Call::encoder_output`).
    pub output: crate::BufferId,
}

/// One image block of a prefill batch: `rows` consecutive token rows from `token`, holding the
/// rows of the vision-encoder product `feature`, which rotate from logical position `start`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionRows {
    /// The batch's token index of the block's first row.
    pub token: u32,
    /// The block's rows: the feature's rows, one cache position each.
    pub rows: u32,
    /// The logical position the block's first row rotates from: the position after the context
    /// before it.
    pub start: u32,
    /// The feature product the rows are.
    pub feature: crate::BufferId,
}

/// The logit adjustments of one batch: what the sampler adds to a row's logits before its
/// penalties, its temperature and its candidate filtering (`ops::sampler::LogitsShape`).
///
/// Two lifetimes, so two forms:
///
/// - a request's **logit bias** and **allowed set** are fixed for its lifetime, so they live in a
///   per-slot table on the worker, one `[vocabulary]` f32 row per request slot: the bias of each
///   biased token, `-inf` at every token outside an allowed set, `0` elsewhere. The batch carrying a
///   request's first sampling call on a worker states its row once ([`SlotBias`]), and every row of
///   the request reads it by its flag in [`Self::biased`];
/// - a row's **banned tokens** move with its history: the end-of-sequence tokens under a
///   `min_tokens` floor and the last token of every bad word whose prefix the generated tokens end
///   with (the scheduler's `SamplingState::suppressed_token_ids`). They are stated per row and per
///   batch, and a banned token's logit is `-inf`.
///
/// A batch none of whose requests asks for any of them states nothing here, and its rows sample
/// exactly as they would without the mechanism.
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogitAdjustments {
    /// The slot rows of the bias table this batch writes before it samples.
    pub installs: Vec<SlotBias>,
    /// `[rows]` whether each row adds its slot's bias row; empty when no row does.
    pub biased: Vec<bool>,
    /// `[rows]` the tokens each row may not sample, strictly increasing; empty when no row bans
    /// any.
    pub banned: Vec<Vec<u32>>,
}

impl LogitAdjustments {
    /// Whether the batch adjusts no row's logits and installs nothing.
    pub fn is_empty(&self) -> bool {
        self.installs.is_empty()
            && !self.biased.iter().any(|biased| *biased)
            && self.banned.iter().all(Vec::is_empty)
    }
}

/// One request slot's row of the worker's bias table: the additive term of every token of the
/// slot's request, written before the batch that states it samples.
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlotBias {
    /// The request slot whose row this is: the sampler slot of the request's rows.
    pub slot: u32,
    /// `(token, bias)` pairs, strictly increasing in token; a token outside the vocabulary is
    /// ignored, as no logit carries it.
    pub bias: Vec<(u32, f32)>,
    /// The tokens the request may sample, strictly increasing, when it restricts them: every other
    /// token's row entry is `-inf`, whatever its bias.
    pub allowed: Option<Vec<u32>>,
}

/// One copy of a request slot's recurrent state between its banks and a state page of the cache.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateCopy {
    /// The request slot whose banks the copy reads or writes.
    pub slot: u32,
    /// The state page's units, page-major as its block table names them.
    pub units: Vec<crate::UnitId>,
    /// Which way the state moves.
    pub direction: StateDirection,
}

/// One media computation's columns: the buffers the model prepared, per request (4.5).
///
/// A media computation is not staged from typed token columns. What a denoiser, an encoder or a
/// decoder reads is what the model itself prepared for each request (`Denoiser::prepare_evaluation`
/// and its siblings), so a media batch's columns are data: the worker copies them to device
/// storage sized from the computation's extent and never interprets them, which is what keeps model
/// mathematics out of the worker. The computation travels with them so the columns can state the
/// rules of their own family without a shared struct growing a field only one family reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MediaColumns {
    /// The computation these columns are for, and the extent its kernels are captured for.
    pub computation: Computation,
    /// Per request, in batch order, the buffers the model prepared for it.
    pub inputs: Vec<Vec<Vec<u8>>>,
    /// Per request, in batch order, the persistent products its call reads, in
    /// `crate::Call::buffer_inputs` order; each one's span is in [`Batch::buffers`].
    pub reads: Vec<Vec<crate::BufferId>>,
    /// Per request, in batch order, the persistent products its call writes, in
    /// `crate::Call::buffer_outputs` order; each one's span is in [`Batch::buffers`].
    pub writes: Vec<Vec<crate::BufferId>>,
}

impl MediaColumns {
    /// Whether the columns state the computation they are for (4.5).
    ///
    /// The rules are the family's own: one set of inputs, reads and writes per request, no token
    /// column a language model reads, and - except for an encode, which is sized by rows and
    /// tokens - an extent, because a bucket captured for one resolution serves no other.
    pub fn check(&self) -> ValidationResult<()> {
        let computation = self.computation;
        if !computation.stated() {
            return Err(ValidationError::message(format!(
                "a {:?} batch is a computation no variant states",
                computation.call
            )));
        }
        let rows = computation.rows as usize;
        if self.inputs.len() != rows {
            return Err(ValidationError::message(format!(
                "a {:?} batch of {rows} rows carries {} request(s) of media inputs",
                computation.call,
                self.inputs.len()
            )));
        }
        if self.reads.len() != rows || self.writes.len() != rows {
            return Err(ValidationError::message(format!(
                "a {:?} batch of {rows} rows states the products of {} and {} request(s)",
                computation.call,
                self.reads.len(),
                self.writes.len()
            )));
        }
        // Three computations are sized by no extent at all: an encode by rows and tokens, and the
        // artifact's two host calls by the request's own products - an audio track's samples come
        // from the decoder's window, and a mux reads what the encodes already wrote. A device
        // computation, by contrast, is captured for one resolution, so a batch that states none cannot
        // say which graph serves it.
        let host = matches!(
            computation.variant(),
            Some(variant)
                if variant.host() || variant == Variant::Encode
        );
        if computation.extent == Extent::None && !host {
            return Err(ValidationError::message(format!(
                "a {:?} batch states no extent: its kernels are captured for one resolution, and a                  batch cannot say which",
                computation.call
            )));
        }
        Ok(())
    }

    /// The columns of a media batch, from the facts the mechanism states per request.
    ///
    /// The mechanism owns the computation and, per request, the request's own stated parameters:
    /// its conditioning as the model family prepared it, the diffusion step it runs, and the seed
    /// and resolution its noise is drawn from. It does not own what the model prepares on the device
    /// — the encoded conditioning rows, the latent trajectory — so a staged set carries the first
    /// and never a placeholder for the second.
    ///
    /// `calls` are the batch's calls in the order of `requests`, one per request: each row reads
    /// and writes the persistent products its call declares.
    pub fn staged(
        computation: Computation,
        requests: &[StagedRequest],
        calls: &[&crate::Call],
    ) -> MediaColumns {
        MediaColumns {
            computation,
            inputs: requests.iter().map(StagedRequest::encode).collect(),
            reads: calls
                .iter()
                .map(|call| {
                    call.buffer_inputs()
                        .map(crate::TensorRef::buffer_id)
                        .collect()
                })
                .collect(),
            writes: calls
                .iter()
                .map(|call| {
                    call.buffer_outputs()
                        .map(crate::TensorRef::buffer_id)
                        .collect()
                })
                .collect(),
        }
    }

    /// One request's staged inputs, as [`MediaColumns::staged`] wrote them.
    ///
    /// The rows are this batch's requests in batch order, so a runner reads the row the call it
    /// serves was placed at. A row that does not carry the two buffers this spelling states, or
    /// whose parameter record is the wrong width, is refused by name: a runner that read a
    /// truncated record would run a step index nobody stated.
    pub fn request(&self, index: usize) -> ValidationResult<StagedRequest> {
        let Some(row) = self.inputs.get(index) else {
            return Err(ValidationError::message(format!(
                "a {:?} batch carries {} request(s); row {index} is not one of them",
                self.computation.call,
                self.inputs.len()
            )));
        };
        StagedRequest::decode(row)
    }

    /// The bytes of every request's every buffer: what the staging path allocates for.
    pub fn bytes(&self) -> usize {
        self.inputs
            .iter()
            .flat_map(|request| request.iter())
            .map(Vec::len)
            .sum()
    }
}

/// One request's staged media inputs, as the mechanism states them (4.5).
///
/// The record is the mechanism-to-model seam for one request of a media computation, and it holds
/// what the mechanism owns and nothing else: the request's conditioning exactly as the model family
/// prepared it, the diffusion step the batch is, the seed and resolution the request states, and the
/// request's identity. The model's own device-side preparation (the encoded conditioning rows, the
/// latent trajectory) is the model's and is not carried here, because a mechanism that filled it
/// would be inventing either a tensor it cannot read or a shape the model never stated.
///
/// # The bytes
///
/// A staged request is two buffers, in this order:
///
/// - **0**, the conditioning blob, byte for byte as `DiffusionRequest::conditioning` holds it;
/// - **1**, sixty bytes little-endian: `step: u32`, `steps: u32`, `seed: u64`, `width: u32`,
///   `height: u32`, `frames: u32`, then the identity - `engine_id: u64`, `request_id: u64`,
///   `request_epoch: u64` - and `given: u32` and `unit: u32`.
///
/// Little-endian and fixed width so that a worker and an engine of different architectures read one
/// record, and so that a runner reads offsets rather than a parser.
///
/// # Why the identity travels with the row
///
/// An in-process batch numbers its calls by their position (`Batch::calls` is a list of indices), so
/// a row tells a runner *where* a request is and never *which* request it is. A step that must find
/// what the previous step of the same request left behind - a latent trajectory, an encoded
/// conditioning - has nothing to key on without this, and a request whose row moved between two
/// batches would take another request's state. The mechanism owns the key, so the mechanism states
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagedRequest {
    /// The request this row's state belongs to: the only name a step has for what it must find
    /// again, since a batch numbers its calls by position.
    pub request: crate::RequestKey,
    /// The request's conditioning, as the model family prepared it.
    pub conditioning: Vec<u8>,
    /// The zero-based diffusion step this batch is.
    pub step: u32,
    /// The steps the request's schedule runs in total.
    pub steps: u32,
    /// The request's noise seed: the draw is addressed by it and the element's index.
    pub seed: u64,
    /// The output raster's width: the request's resolution.
    pub width: u32,
    /// The output raster's height.
    pub height: u32,
    /// The output frames after the model's alignment.
    pub frames: u32,
    /// The encoded rounds this call hands over, which only a muxing call states: the final muxing
    /// call hands none, and that is what makes it the one that finishes the artifact.
    pub rounds: u32,
    /// The first timeline unit a unit-measured call covers: the clip a video decode is. Zero for
    /// every other call.
    pub unit: u32,
}

impl StagedRequest {
    /// The two buffers a staged request is (see the type's byte layout).
    pub fn encode(&self) -> Vec<Vec<u8>> {
        let mut record = Vec::with_capacity(PARAMETER_BYTES);
        record.extend_from_slice(&self.step.to_le_bytes());
        record.extend_from_slice(&self.steps.to_le_bytes());
        record.extend_from_slice(&self.seed.to_le_bytes());
        record.extend_from_slice(&self.width.to_le_bytes());
        record.extend_from_slice(&self.height.to_le_bytes());
        record.extend_from_slice(&self.frames.to_le_bytes());
        record.extend_from_slice(&self.request.engine_id.to_le_bytes());
        record.extend_from_slice(&self.request.request_id.0.to_le_bytes());
        record.extend_from_slice(&self.request.request_epoch.to_le_bytes());
        record.extend_from_slice(&self.rounds.to_le_bytes());
        record.extend_from_slice(&self.unit.to_le_bytes());
        vec![self.conditioning.clone(), record]
    }

    /// Reads a staged request's two buffers.
    pub fn decode(row: &[Vec<u8>]) -> ValidationResult<Self> {
        let [conditioning, record] = row else {
            return Err(ValidationError::message(format!(
                "a media request's row carries {} buffer(s); a staged request is the conditioning \
                 blob and its parameter record",
                row.len()
            )));
        };
        if record.len() != PARAMETER_BYTES {
            // The refusal names the **shape**, not the width it expects: the width is
            // `PARAMETER_BYTES`, which every column this record gains moves by a field, and a
            // refusal that prints it states a fact about the build rather than about what was
            // given over. What the reader needs is which record this is and what arrived.
            return Err(ValidationError::message(format!(
                "a media request's parameter record is step, steps, seed, width, height, frames, \
                 the request's three identities, the rounds it hands over and its first unit; this \
                 one is {} bytes, which is not that shape",
                record.len()
            )));
        }
        let word = |at: usize| {
            u32::from_le_bytes([record[at], record[at + 1], record[at + 2], record[at + 3]])
        };
        let word64 = |at: usize| {
            u64::from_le_bytes([
                record[at],
                record[at + 1],
                record[at + 2],
                record[at + 3],
                record[at + 4],
                record[at + 5],
                record[at + 6],
                record[at + 7],
            ])
        };
        Ok(Self {
            request: crate::RequestKey::new(word64(28), crate::RequestId(word64(36)), word64(44)),
            conditioning: conditioning.clone(),
            step: word(0),
            steps: word(4),
            seed: word64(8),
            width: word(16),
            height: word(20),
            frames: word(24),
            rounds: word(52),
            unit: word(56),
        })
    }
}

/// The width of a staged request's parameter record: `step`, `steps`, `seed`, `width`, `height` and
/// `frames`, then the request's identity (`engine_id`, `request_id`, `request_epoch`), then `given`
/// and `unit`.
const PARAMETER_BYTES: usize = (4 + 4 + 8 + 4 + 4 + 4) + (8 + 8 + 8) + 4 + 4;

impl BatchColumns {
    /// The number of rows (sequences) the columns describe.
    ///
    /// A row is one sequence's query, which is not the same as a call: a verify batch carries
    /// several rows for one call, and the per-row columns (`slots`, `parameters`) are as long as
    /// the rows. `rows.calls` counts the calls instead, which sized the sampler's views to one row
    /// when a call held three.
    pub fn rows(&self) -> usize {
        self.rows.rows()
    }

    /// The calls the columns describe.
    pub fn calls(&self) -> usize {
        self.rows.calls.len()
    }

    /// The number of query tokens the columns describe.
    pub fn tokens(&self) -> usize {
        self.input_ids.len()
    }

    /// Whether every column has the length its field promises for `shape`, so that a mis-sized
    /// batch fails before a launch rather than after one.
    pub fn check(&self, shape: &Shape) -> ValidationResult<()> {
        let tokens = self.rows.tokens();
        // Two counts, and a column belongs to one of them: `slots`, `parameters`, `draws`,
        // `seq_lens` and `logits_rows` are per **row** (a verify or branch call carries several
        // rows for one call),
        // while the commit table states one row per **call** (`crates/scheduler`'s `columns.rs`).
        // Sizing one against the other refuses a valid multi-row call and lets a wrong commit table
        // through.
        let rows = self.rows();
        let calls = self.calls();
        let parameters = proto::module::sampler::PARAMETER_COLUMNS as usize;
        for (name, actual, expected) in [
            ("positions", self.rows.positions.len(), tokens),
            ("slot_mapping", self.rows.slot_mapping.len(), tokens),
            ("input_ids", self.input_ids.len(), tokens),
            ("seq_lens", self.rows.seq_lens.len(), rows),
            ("logits_rows", self.rows.logits_rows.len(), rows),
            ("slots", self.slots.len(), rows),
            ("parameters", self.parameters.len(), rows * parameters),
            ("draws", self.draws.len(), rows),
            (
                "commit",
                self.commit.len(),
                proto::module::commit::COMMIT_FIELDS * calls,
            ),
        ] {
            if actual != expected {
                return Err(ValidationError::message(format!(
                    "{name} has {actual} entries, the batch has {rows} rows, {calls} calls and \
                     {tokens} tokens"
                )));
            }
        }
        if tokens > shape.tokens as usize || rows > shape.rows as usize {
            return Err(ValidationError::message(format!(
                "the batch holds {rows} rows and {tokens} tokens, more than its shape's {} and {}",
                shape.rows, shape.tokens
            )));
        }
        if self.page_size == 0 {
            return Err(ValidationError::message("the batch's page size is zero"));
        }
        if !self.targets.is_empty() && self.targets.len() != tokens {
            return Err(ValidationError::message(format!(
                "targets has {} entries, the batch has {tokens} tokens",
                self.targets.len()
            )));
        }
        if self.targets.iter().any(|target| *target >= 0) != shape.variant.scores() {
            return Err(ValidationError::message(format!(
                "a {} batch {} prompt targets",
                shape.variant,
                if shape.variant.scores() {
                    "names no"
                } else {
                    "names"
                }
            )));
        }
        if !self.checkpoints.is_empty() && self.checkpoints.len() != rows {
            return Err(ValidationError::message(format!(
                "checkpoints has {} entries, the batch has {rows} rows",
                self.checkpoints.len()
            )));
        }
        if !self.drafts.is_empty() && self.drafts.len() != calls {
            return Err(ValidationError::message(format!(
                "drafts has {} entries, the batch has {calls} calls",
                self.drafts.len()
            )));
        }
        let denoises = shape.variant == Variant::TokenDenoise;
        if self.block_steps.len() != if denoises { calls } else { 0 } {
            return Err(ValidationError::message(format!(
                "block_steps has {} entries, and a {} batch of {calls} calls states {}",
                self.block_steps.len(),
                shape.variant,
                if denoises { "one per call" } else { "none" }
            )));
        }
        if denoises
            != self
                .rows
                .calls
                .iter()
                .any(|call| call.kind == RowKind::Denoise)
            || (denoises
                && self
                    .rows
                    .calls
                    .iter()
                    .any(|call| call.kind != RowKind::Denoise))
        {
            return Err(ValidationError::message(format!(
                "a {} batch carries {} calls that denoise a block of tokens",
                shape.variant,
                if denoises { "other than" } else { "some" }
            )));
        }
        let adjustments = &self.adjustments;
        for (name, actual) in [
            ("biased", adjustments.biased.len()),
            ("banned", adjustments.banned.len()),
        ] {
            if actual != 0 && actual != rows {
                return Err(ValidationError::message(format!(
                    "{name} has {actual} entries, the batch has {rows} rows"
                )));
            }
        }
        let increasing = |tokens: &[u32]| tokens.windows(2).all(|pair| pair[0] < pair[1]);
        for install in &adjustments.installs {
            let biased: Vec<u32> = install.bias.iter().map(|(token, _)| *token).collect();
            if !increasing(&biased)
                || install
                    .allowed
                    .as_deref()
                    .is_some_and(|allowed| !increasing(allowed))
                || install.bias.iter().any(|(_, bias)| !bias.is_finite())
            {
                return Err(ValidationError::message(format!(
                    "slot {}'s bias row is not canonical: its tokens must be strictly increasing \
                     and its biases finite",
                    install.slot
                )));
            }
        }
        if !self
            .fresh_slots
            .windows(2)
            .all(|pair| pair[0].slot < pair[1].slot)
        {
            return Err(ValidationError::message(
                "the batch's fresh slots are not strictly increasing",
            ));
        }
        if !adjustments.banned.iter().all(|tokens| increasing(tokens)) {
            return Err(ValidationError::message(
                "a row's banned tokens are not strictly increasing",
            ));
        }
        Ok(())
    }
}

/// The kind of forward a captured graph serves.
///
/// A bucket only ever holds a forward of its own variant: a decode row cannot be served by a
/// prefill graph, and a prefill that projects logits has different outputs from one that only
/// writes the KV cache.
///
/// # The catalog, as it is
///
/// | variant | writes | sampler |
/// --- | --- | ---
/// | [`Variant::PrefillWrite`] | the KV cache | runs over every row, and every row's draw is discarded (`logits_rows` are all `-1`) |
/// | [`Variant::PrefillOutput`] | the KV cache **and** every sampling row's logits | runs over every row of the bucket; a row that ends a prompt keeps its token |
/// | [`Variant::PrefillGreedy`] | the same, for a batch whose prompt endings are all greedy | runs, greedily |
/// | [`Variant::PrefillScored`] | the same, and the scores of every prompt token the batch's `targets` name | runs over every row, one draw per row |
/// | [`Variant::DecodeGreedy`] | one token per row | runs, greedily |
/// | [`Variant::DecodeSampled`] | one token per row | runs, one draw per row |
///
/// A prefill bucket carries the sampler because the logits are computed anyway, and because it is
/// what makes a first token possible without a decode step spent only on choosing it. The draw is
/// the expensive tail: its candidate selection visits every row of the bucket for milliseconds, which
/// a batch of greedy prompt endings and a batch that ends no prompt both have no use for. So those
/// two take the greedy tail, and only a batch with a drawing row pays for the draw.
///
/// **A greedy row in a drawing bucket is served by the draw:** the worker stages it with a top-k
/// width of one, so its candidate set is its largest shaped logit and its draw selects it, the
/// token the greedy launch sequence would choose (ties to the lowest id on both paths). A batch may
/// therefore mix greedy and drawn rows, prompt endings or decodes, and is served as a drawing one.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum Variant {
    /// Decode rows drawn from the slot state, sampled greedily.
    DecodeGreedy,
    /// Decode rows drawn from the slot state, sampled with the bucket's sampler graph.
    DecodeSampled,
    /// Prefill rows of host tokens writing the KV cache, keeping no hidden state.
    PrefillWrite,
    /// Prefill rows of host tokens that also project each row's last hidden state.
    PrefillOutput,
    /// A [`Self::PrefillOutput`] whose rows that end a prompt are all greedy: each row's first
    /// token is the argmax of its shaped logits, and nothing is drawn.
    PrefillGreedy,
    /// A [`Self::PrefillOutput`] that also scores prompt tokens (`prompt_logprobs`): every token
    /// whose target the batch names ([`BatchColumns::targets`]) has its logits projected and its
    /// target's log-probability, rank and candidates reported on the raw distribution.
    ///
    /// A variant of its own because the scoring projects every token of the bucket rather than
    /// one per row, which a prefill that scores nothing does not pay for: a batch none of whose
    /// requests asks for prompt logprobs is served by the other prefill variants, with their
    /// launches. Its prompt endings are drawn, a greedy one as a draw over its largest candidate.
    PrefillScored,
    /// One encode of prompt rows: the text model's and the vision tower's outputs, kept as the
    /// conditioning the denoiser reads.
    Encode,
    /// One denoising step of one latent layout: the diffusion model's velocity, applied to the
    /// latents in place (a request's steps are one batch each, never a loop inside a batch).
    Denoise,
    /// One clip decoded from latents into a raster.
    Decode,
    /// One audio track decoded from latents into interleaved PCM.
    ///
    /// A variant of its own because the graph is: a clip's decode is a stack of three-dimensional
    /// convolutions over a voxel grid, and a track's is the vocoder's upsampling and dilated
    /// one-dimensional convolutions. One bucket cannot hold both, and a capture is keyed by the
    /// graph a bucket runs.
    AudioDecode,
    /// One video unit encoded on the host: the postprocessor's `[frames, rows, columns, 3]` raster
    /// into the encoded unit the assembly appends.
    VideoEncode,
    /// One request's audio track encoded on the host: the decoder's interleaved PCM into the track
    /// the assembly carries.
    AudioEncode,
    /// One request's encoded units and track assembled into its MP4, on the host.
    Mux,
    /// One request's trajectory opened. The denoiser draws the request's noise at its first step,
    /// so this call computes nothing: it is the mechanism's step between a prompt's encoding and
    /// the first denoising call.
    LatentPrepare,
    /// One request's condition media read on the host: its bytes decoded and laid out as the
    /// model's encoders read them, in storage a later call of the request reads.
    MediaRead,
    /// A condition encoder's own call, whose work a later call computes: the vision tower runs
    /// inside the text encoding that splices its rows, and the latent encoders inside the
    /// denoiser's first step, whose conditioner writes their rows. So this call computes nothing:
    /// it is the mechanism's step between the media read and the call that consumes it.
    ConditionEncode,
    /// A rank's part in an expert step it has no tokens for: every expert layer, and nothing else.
    ExpertJoin,
    /// One denoising step of a diffusion language model over each row's block of tokens
    /// ([`BatchColumns::block_steps`]): the rows read their requests' cached context and write no
    /// cache page, and the step that stops a block reports the block's tokens.
    TokenDenoise,
    /// One verification per row: a committed token followed by its drafts, each token evaluated
    /// as the decode step at its position and sampled greedily; the row commits the drafts its
    /// own samples agree with and the sample after them. A bucket of `rows` verifications holds
    /// `rows` times one more than the deployment's drafts tokens.
    VerifyGreedy,
    /// A [`Self::VerifyGreedy`] whose tokens are drawn, a greedy row as a draw over its one
    /// largest candidate.
    VerifySampled,
    /// The drafter's forward behind a prefill or a verification of the same rows and tokens:
    /// its entries over the positions whose next token the step before it committed, then the
    /// drafts each row proposes, chained on the device. No batch states it: the worker issues
    /// it behind the batch it follows.
    Draft,
}

impl Variant {
    /// The call kind this variant serves, as a profiling session's `stages` and the fence probes'
    /// histograms name it (`crate::CallKind`).
    pub const fn stage(self) -> &'static str {
        match self {
            Self::DecodeGreedy | Self::DecodeSampled => "decode",
            Self::VerifyGreedy | Self::VerifySampled => "verify",
            Self::Draft => "draft",
            Self::PrefillWrite
            | Self::PrefillOutput
            | Self::PrefillGreedy
            | Self::PrefillScored => "prefill",
            Self::Encode => "text_encoding",
            Self::Denoise => "denoising",
            Self::Decode => "video_decoding",
            Self::AudioDecode => "audio_decoding",
            Self::VideoEncode => "video_encoding",
            Self::AudioEncode => "audio_encoding",
            Self::Mux => "muxing",
            Self::LatentPrepare => "latent_preparation",
            Self::MediaRead => "media_reading",
            Self::ConditionEncode => "condition_encoding",
            Self::ExpertJoin => "expert_join",
            Self::TokenDenoise => "token_denoising",
        }
    }

    /// The variant's name, as a capture report prints it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DecodeGreedy => "decode_greedy",
            Self::DecodeSampled => "decode_sampled",
            Self::PrefillWrite => "prefill_write",
            Self::PrefillOutput => "prefill_output",
            Self::PrefillGreedy => "prefill_greedy",
            Self::PrefillScored => "prefill_scored",
            Self::Encode => "encode",
            Self::Denoise => "denoise",
            Self::Decode => "decode",
            Self::AudioDecode => "audio_decode",
            Self::VideoEncode => "video_encode",
            Self::AudioEncode => "audio_encode",
            Self::Mux => "mux",
            Self::LatentPrepare => "latent_prepare",
            Self::MediaRead => "media_read",
            Self::ConditionEncode => "condition_encode",
            Self::ExpertJoin => "expert_join",
            Self::TokenDenoise => "token_denoise",
            Self::VerifyGreedy => "verify_greedy",
            Self::VerifySampled => "verify_sampled",
            Self::Draft => "draft",
        }
    }

    /// Whether this variant is a media computation rather than a causal-LM forward.
    ///
    /// The two families never share a bucket: a media bucket carries an [`Extent`] the language
    /// model has no use for, and a causal-LM runner refuses one by name rather than serving it
    /// from a graph shaped for another computation.
    pub const fn media(self) -> bool {
        matches!(
            self,
            Self::Encode
                | Self::Denoise
                | Self::Decode
                | Self::AudioDecode
                | Self::VideoEncode
                | Self::AudioEncode
                | Self::Mux
                | Self::LatentPrepare
                | Self::MediaRead
                | Self::ConditionEncode
        )
    }

    /// Whether this variant is performed in this process rather than on a device (§2.4).
    ///
    /// The artifact's three host calls capture no graph: their work is encoding and muxing, which
    /// the process's media thread pool does. A bucket of one states which device serves the call and never
    /// an extent, so no resolution narrows it.
    pub const fn host(self) -> bool {
        matches!(
            self,
            Self::VideoEncode
                | Self::AudioEncode
                | Self::Mux
                | Self::MediaRead
                | Self::ConditionEncode
        )
    }

    /// The computation a media call is, and `None` for a call no variant states yet.
    ///
    /// The three decoders produce one raster for one muxer. The text encoding is the one encode a
    /// device runs as a call of its own; the vision and latent encodings are computed by the calls
    /// that consume them ([`Variant::ConditionEncode`]). A call with no variant is not served, which
    /// the mechanism reads as a capability the worker does not state rather than as a batch to
    /// send anyway.
    pub const fn of_media(call: crate::MediaCall) -> Option<Self> {
        match call {
            crate::MediaCall::TextEncoding => Some(Self::Encode),
            crate::MediaCall::VisionEncoding | crate::MediaCall::LatentEncoding => {
                Some(Self::ConditionEncode)
            }
            crate::MediaCall::Denoising => Some(Self::Denoise),
            crate::MediaCall::VideoDecoding => Some(Self::Decode),
            // The audio half of a response: its own bucket, matching the `AudioDecoding` call this
            // file mapped to nothing while the model served video alone.
            crate::MediaCall::AudioDecoding => Some(Self::AudioDecode),
            // The three host calls: their work is encoding and muxing in this process, and each is
            // a phase of its own because each is a call the mechanism submits and waits on.
            crate::MediaCall::VideoEncoding => Some(Self::VideoEncode),
            crate::MediaCall::AudioEncoding => Some(Self::AudioEncode),
            crate::MediaCall::Muxing => Some(Self::Mux),
            crate::MediaCall::LatentPreparation => Some(Self::LatentPrepare),
            crate::MediaCall::MediaReading => Some(Self::MediaRead),
        }
    }

    /// Whether a bucket of this variant picks each row's token by taking the largest candidate
    /// rather than by drawing one (4.4's greedy path).
    ///
    /// A greedy request states temperature zero, whose shaping divides by one, so the row's largest
    /// candidate *is* its token. A bucket of greedy rows takes the argmax launch sequence; one with
    /// a drawn row draws, and serves its greedy rows as a draw over their one largest candidate. A
    /// prefill that ends no prompt keeps no token, so it takes the cheaper of the two.
    pub const fn greedy(self) -> bool {
        matches!(
            self,
            Self::DecodeGreedy | Self::PrefillGreedy | Self::PrefillWrite | Self::VerifyGreedy
        )
    }

    /// Whether a bucket of this variant runs the sampler (4.4).
    ///
    /// Every bucket the causal-LM runner serves does, which is every variant this predicate is asked
    /// about: a media bucket is refused by name before a graph is chosen for it (`Bucket::causal_lm`),
    /// so it never reaches this question. A prefill's sampler is a tail over logits the forward
    /// computes anyway, so it costs the graph almost nothing and makes a sampled first token possible
    /// without a decode step spent only on drawing; the mechanism says which rows end a prompt by
    /// putting them in `logits_rows`, and a row that ends none has its draw discarded. See
    /// [`Variant`]'s catalog, and the one rule under which a captured draw may be replaced by the
    /// host's argmax.
    pub const fn samples(self) -> bool {
        true
    }

    /// Whether a bucket of this variant scores the prompt tokens its batch's targets name.
    pub const fn scores(self) -> bool {
        matches!(self, Self::PrefillScored)
    }

    /// Whether a forward of this variant draws its inputs from the slot state one token per row.
    pub const fn is_decode(self) -> bool {
        matches!(self, Self::DecodeGreedy | Self::DecodeSampled)
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// One forward's size: what it asks a captured graph to hold.
///
/// `tokens` is the number of tokens the forward writes to the KV cache and `rows` the number of
/// sequences it evaluates. A decode forward evaluates one token per row, so its `tokens == rows`;
/// a prefill evaluates a chunk of host tokens, which may be more than its rows.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct Shape {
    pub rows: u32,
    pub tokens: u32,
    pub variant: Variant,
}

impl Shape {
    /// A decode forward of `rows` rows.
    pub const fn decode(rows: u32, variant: Variant) -> Self {
        Self {
            rows,
            tokens: rows,
            variant,
        }
    }

    /// A prefill or encode forward of `rows` rows holding `tokens` tokens.
    pub const fn prefill(rows: u32, tokens: u32, variant: Variant) -> Self {
        Self {
            rows,
            tokens,
            variant,
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}/{} rows/{} tokens",
            self.variant, self.rows, self.tokens
        )
    }
}

/// One media computation as the mechanism states it, so that a batch can say whether it serves it.
///
/// The three fields are what a media graph's capture depends on: which computation it is, how many
/// requests it holds, and - for a decode or a denoising step - the shape its kernels are captured
/// for. An encode states rows and tokens like a prefill, so its extent is [`Extent::None`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Computation {
    /// The media call the mechanism is about to run.
    pub call: crate::MediaCall,
    /// The requests in the batch.
    pub rows: u32,
    /// An encode's tokens; zero for a denoising step and a decode, which are sized by their extent.
    pub tokens: u32,
    /// The latent layout of a denoising step, the raster of a decode, and [`Extent::None`] for an
    /// encode.
    pub extent: Extent,
}

impl Computation {
    /// The variant that computes this call, or `None` when no variant states it.
    pub const fn variant(self) -> Option<Variant> {
        Variant::of_media(self.call)
    }

    /// Whether this is a computation the worker serves at all, before any batch is consulted.
    pub const fn stated(self) -> bool {
        self.variant().is_some()
    }
}

/// What a scheduling mechanism asks of the worker that serves it (4.4, 4.5, 7.3).
///
/// The scheduler thread builds a step's batch and hands it here: `submit` serves it on the device
/// of the batch's phase, and `shadow` runs the same batch's paired forwards against the generation
/// it names without serving it. The main process implements this once over its per-device
/// workers; [`crate::device::Worker`] keeps the same two calls as inherent methods, so a
/// mechanism with a single device needs no router.
pub trait Batches: Send + Sync {
    /// Serves one batch: stage its inputs, choose its bucket, replay its graph.
    fn submit(&self, batch: Batch) -> Result<Submission, Box<Fault>>;

    /// Runs one shadow batch (7.3 step 2): the paired forwards of `batch.generation`'s changed
    /// operators against the serving generation's kernels, compared by each operator's tolerance.
    ///
    /// A shadow batch never writes running memory and never serves a request; its failure is the
    /// update's, which the main process reports as `PrepareFailed`.
    fn shadow(&self, batch: Batch) -> Result<Submission, Box<Fault>>;

    /// The bucket this worker would serve `computation` in, or `None` when it serves no such
    /// computation (2.4's stated capability).
    ///
    /// The mechanism asks before it forms a media batch: a media call is served only where a
    /// runner's batch states the computation, and the response is the bucket that step prepared it
    /// into, so the refusal the mechanism makes names a capability the worker does not state
    /// rather than a check that was deleted. A missing phase is its own refusal: a batch cannot be
    /// sent to a device that does not exist.
    fn media(
        &self,
        _phase: crate::PhaseId,
        _computation: &Computation,
    ) -> Result<Option<Extent>, Box<Fault>> {
        Ok(None)
    }

    /// The devices `batch` runs on, each by its place among [`Batches::device_count`] devices.
    ///
    /// A device runs the batches it is given from a queue of its own, and the launch's
    /// `queue_depth` bounds that queue, as the capacity reservation holds each device's in-flight calls
    /// by it. A mechanism that counted every batch of a rank against one bound would hold one
    /// device's clip behind another device's, so it counts the devices a batch takes. A worker of
    /// one device runs every batch there.
    fn devices(&self, _batch: &Batch) -> Result<Vec<usize>, Box<Fault>> {
        Ok(vec![0])
    }

    /// How many devices run this worker's batches, one queue each.
    fn device_count(&self) -> usize {
        1
    }

    /// The pools this worker serves (4.2, 7.3): what the mechanism schedules against.
    ///
    /// The mechanism builds its `WorkerInfo` from this and nothing else, so a pool whose ranks,
    /// cache geometry or model length is absent here is a pool it refuses to serve rather than one
    /// it assumes: a scheduler that guesses a rank's cache makes the host KV path disappear at
    /// runtime. A worker that states none cannot be served in process, and the mechanism says so
    /// when it builds its info.
    fn inventory(&self) -> &[Inventory] {
        &[]
    }

    /// Performs the storage `operation` on every rank, while no batch runs (7.6), and responds to each
    /// rank's JSON result.
    ///
    /// A worker with no storage of its own has nothing to change and nothing to report, which
    /// is an empty response: the mechanism reads it as a cache that is not split into segments.
    fn operate(&self, _operation: &EpochOperation) -> Result<Vec<String>, Box<Fault>> {
        Ok(Vec::new())
    }

    /// Puts the prepared expert redistribution of `order` in place, while no batch runs (7.6), and
    /// responds to each rank's result.
    fn commit_experts(
        &self,
        _order: &std::collections::BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<String>, Box<Fault>> {
        Ok(Vec::new())
    }

    /// Maps the host KV cache's page pools on this rank's devices, `tokens` tokens of each cache
    /// group, and copies pages through them from now on. Responds to each pool's name with its ranks'
    /// pools, in rank order, as the mechanism keys the host KV cache by them. A worker that keeps
    /// no host KV cache maps nothing, and the mechanism prepares no host write.
    fn map_host_cache(&self, _tokens: u64) -> Result<HostPools, Box<Fault>> {
        Ok(Vec::new())
    }

    /// Copies `offloads` to the host KV cache with no batch to carry them: the mechanism's pages
    /// that wait while nothing runs. The submission's output means the copies finished.
    fn copy_pages(&self, _offloads: Vec<crate::PageCopy>) -> Result<Submission, Box<Fault>> {
        Err(Box::new(Refusal(
            "this worker keeps no host KV cache to copy pages to",
        )))
    }

    /// Copies the trajectories of preempted requests to the host latent pools the engine reserved
    /// slots in, with no batch to carry them (`crate::LatentOffload`): every call of each request
    /// completed, so each copy waits only for the batches that advanced the trajectory. The
    /// submission's output means the copies finished.
    fn offload_latents(
        &self,
        _offloads: Vec<crate::LatentOffload>,
    ) -> Result<Submission, Box<Fault>> {
        Err(Box::new(Refusal(
            "this worker maps no host latent pool to offload a trajectory to",
        )))
    }
}

/// A storage operation of a serving epoch, which the mechanism performs on every rank at a batch
/// boundary ([`Batches::operate`]): no batch is in flight while it runs, so no running job shares
/// a mapping it changes (7.6).
///
/// Each variant carries its message from [`crate::memory`], whose [`crate::memory::Operation`]
/// impl names the reply every rank responds (one JSON text per rank, [`crate::memory::replies`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum EpochOperation {
    /// Reports how the rank's KV cache storage is split into segments and how many it holds.
    CacheStorage(crate::memory::CacheStorage),
    /// Gives back the KV cache's segments from one segment on.
    ReleaseCache(crate::memory::ReleaseCache),
    /// Creates given-back segments again.
    RestoreCache(crate::memory::RestoreCache),
    /// Gives back the graph storage and the KV cache while the epoch sleeps.
    Sleep(crate::memory::Sleep),
    /// Restores what [`EpochOperation::Sleep`] gave back.
    Wake(crate::memory::Wake),
    /// Puts the weights the engine loaded beside the serving batches in place.
    UpdateWeights(crate::memory::UpdateWeights),
    /// Takes over the KV cache units a predecessor epoch lent.
    AdoptCache(crate::memory::AdoptCache),
    /// Publishes a cache warming's claimed pages, which the ranks then serve as cached prefixes.
    PublishCache(crate::memory::PublishCache),
    /// Describes where the ranks hold a cache warming's pages, for a successor's cache manifest.
    CacheManifest(crate::memory::CacheManifest),
}

/// The host KV cache's page pools a worker mapped: each pool's name with its ranks' pools, in rank
/// order ([`Batches::map_host_cache`]).
pub type HostPools = Vec<(String, Vec<Vec<crate::HostPoolInfo>>)>;

/// One pool of the deployment a worker serves (4.2, 7.3).
#[derive(Clone, Debug, Default)]
pub struct Inventory {
    /// The deployment's name for the pool, the identity the mechanism's `WorkerGroupId` carries.
    pub group: String,
    /// The pool's ranks in launch order, each with what it reported at handshake.
    pub ranks: Vec<MainProcessInfo>,
    /// What the deployment launched the pool with; the ranks' reports do not carry it back.
    pub layout: PoolLayout,
    /// The longest sequence the pool's model was built for: the attention descriptor's static bound
    /// on the keys one row reads.
    pub max_model_len: u32,
}

/// What a deployment launched one pool with (4.2): the values the deployment decides, which the
/// ranks' reports do not carry back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoolLayout {
    /// Unresolved physical batches the pool admits at once, counted as at least one.
    pub queue_depth: usize,
    /// Ranks the pool was launched with, counted as at least one.
    pub ranks: usize,
    /// The pool's stages by name, each with its member ranks, parallel degrees and distribution.
    pub stages: std::collections::BTreeMap<String, proto::config::BlockConfig>,
}

/// What the caller of a submission holds until its jobs ended.
///
/// A submission completes once the device worker has performed every job's work and the work
/// completed on the device (the worker thread waits on its own stream's work before reporting a
/// job, and with it the submission). A serving submission completes with the step's
/// [`BatchOutput`] — the tokens its sampling rows sampled; every other submission completes with an
/// empty one. The scheduler thread waits on the submission of a step at the next step's transfer,
/// never on the device itself.
pub struct Submission {
    id: SubmissionId,
    state: Arc<SubmissionState>,
    /// The same batch on the other ranks that serve it: the submission completes when they all did.
    peers: Vec<Submission>,
}

/// What a submission's completion calls once its outcome lands ([`Submission::on_settled`]).
///
/// It runs on the thread that completed the submission (a device's worker thread), so it must
/// only signal: wake a waiter, never do the waiter's work.
pub type Waker = Arc<dyn Fn() + Send + Sync>;

/// The completion state of one submission, which the worker's queue completes.
pub struct SubmissionState {
    /// The submission's own outcome, which `Completion` records once and waits on, with the
    /// wakers [`Submission::on_settled`] registered.
    completion: Completion<Waker, BatchOutput>,
    /// What the submission was doing when it failed, written where the failure happened: the
    /// caller cannot know it, and an error alone does not say which bucket or operator broke.
    failure: Mutex<Option<String>>,
}

impl Default for SubmissionState {
    fn default() -> Self {
        Self {
            completion: Completion::default(),
            failure: Mutex::new(None),
        }
    }
}

impl Submission {
    /// A submission that has already completed with `outcome`.
    pub fn completed(id: SubmissionId, outcome: Outcome<BatchOutput>) -> Self {
        let state = Arc::new(SubmissionState::default());
        state.completion.settle(outcome);
        Self {
            id,
            state,
            peers: Vec::new(),
        }
    }

    /// This submission and the same batch's on `peers`, as one.
    ///
    /// The ranks of a group run one batch together, so it has ended when every one of them ended:
    /// this rank's outcome is the response, and a peer's failure replaces it, because a rank that
    /// failed leaves the others' output without its part.
    #[must_use]
    pub fn joined(mut self, peers: Vec<Submission>) -> Self {
        self.peers = peers;
        self
    }

    /// A submission waiting for its jobs' completion, with the state the worker's queue completes.
    pub fn pending(id: SubmissionId) -> (Self, Arc<SubmissionState>) {
        let state = Arc::new(SubmissionState::default());
        (
            Self {
                id,
                state: Arc::clone(&state),
                peers: Vec::new(),
            },
            state,
        )
    }

    /// The state the worker's queue completes, and the submission's number.
    pub fn parts(&self) -> (Arc<SubmissionState>, SubmissionId) {
        (Arc::clone(&self.state), self.id)
    }

    /// Waits until the submission completed and responds to its outcome.
    pub fn wait(&self) -> Outcome<BatchOutput> {
        let own = self.wait_own();
        let peers: Vec<Outcome<BatchOutput>> = self.peers.iter().map(Self::wait).collect();
        Self::joined_outcome(own, peers)
    }

    /// This rank's outcome unless a peer's says the batch did not complete.
    fn joined_outcome(
        own: Outcome<BatchOutput>,
        peers: Vec<Outcome<BatchOutput>>,
    ) -> Outcome<BatchOutput> {
        match own {
            Outcome::Success(_) => peers
                .into_iter()
                .find(|peer| !matches!(peer, Outcome::Success(_)))
                .unwrap_or(own),
            failed => failed,
        }
    }

    /// Waits until this rank's part completed and responds to its outcome.
    fn wait_own(&self) -> Outcome<BatchOutput> {
        self.state.completion.wait()
    }

    /// Waits at most `timeout` for the submission's completion.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Outcome<BatchOutput>> {
        let deadline = Instant::now() + timeout;
        let own = self.wait_own_until(deadline)?;
        let mut peers = Vec::with_capacity(self.peers.len());
        for peer in &self.peers {
            peers.push(peer.wait_timeout(deadline.saturating_duration_since(Instant::now()))?);
        }
        Some(Self::joined_outcome(own, peers))
    }

    /// Waits until `deadline` for this rank's part.
    fn wait_own_until(&self, deadline: Instant) -> Option<Outcome<BatchOutput>> {
        self.state
            .completion
            .wait_timeout(deadline.saturating_duration_since(Instant::now()))
    }

    /// Calls `wake` when this rank's part or any peer's part settles, at once for a part that
    /// already has: a caller holding several submissions waits on one signal of its own and reads
    /// [`Submission::outcome`] of each when it fires, instead of polling them.
    ///
    /// `wake` may be called once per part, so a waiter re-checks the outcome rather than counting
    /// calls.
    pub fn on_settled(&self, wake: &Waker) {
        if let Some(wake) = self.state.completion.subscribe(Arc::clone(wake)) {
            wake();
        }
        for peer in &self.peers {
            peer.on_settled(wake);
        }
    }

    /// The submission's outcome, when it completed.
    pub fn outcome(&self) -> Option<Outcome<BatchOutput>> {
        let own = self.state.completion.outcome()?;
        let mut peers = Vec::with_capacity(self.peers.len());
        for peer in &self.peers {
            peers.push(peer.outcome()?);
        }
        Some(Self::joined_outcome(own, peers))
    }

    /// What the submission was doing when it failed, when it did.
    ///
    /// The device thread writes it where the failure happens - the job it was performing and the
    /// bucket, generation or operator that named it - because the caller holds only the bare error.
    pub fn failure_label(&self) -> Option<String> {
        lock(&self.state.failure)
            .clone()
            .or_else(|| self.peers.iter().find_map(Self::failure_label))
    }
}

impl SubmissionState {
    /// Records what this submission was doing when it failed, for the report that reads it.
    pub fn note_failure(&self, label: String) {
        let mut slot = lock(&self.failure);
        if slot.is_none() {
            *slot = Some(label);
        }
    }

    /// Completes the submission with `outcome`; the first completion wins.
    pub fn finish(self: Arc<Self>, outcome: Outcome<BatchOutput>) {
        // The first outcome stands; a later one (a cancel racing a completion) is dropped.
        for wake in self.completion.settle(outcome).unwrap_or_default() {
            wake();
        }
    }
}

impl std::fmt::Debug for Submission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Submission")
            .field("id", &self.id)
            .field("completed", &self.outcome().is_some())
            .finish()
    }
}

impl std::fmt::Debug for SubmissionState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SubmissionState")
    }
}

// What the scheduler allocates for a call, in the batch above: the cache page tables of its
// request (`BlockTable`, with the units newly acquired for them in `CacheUnitAllocation`),
// decode cursors (`DecodeRange`), persistent buffer spans (`BufferAllocation`) and the host
// cache copies (`PageCopy`, `LatentOffload`).

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
/// [`BlockTable`]) from the host page before the batch's forward; the units
/// are new units of the batch, which the worker resets first. A write copies
/// the units of a published page into the host page; the rank's output for
/// the batch reports the write complete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageCopy {
    /// The rank of the worker group that copies the page.
    pub rank: u32,
    /// The page pool at the rank's host's Storage Server.
    pub pool: u64,
    /// The cache group of the page.
    pub group_id: u32,
    /// The page's index in the host page pool, or in the group's staging
    /// tensor when `staged`.
    pub host_page: u64,
    /// The page's units.
    pub units: Vec<UnitId>,
    /// A load of a page in the group's conversion source representation
    /// from slot `host_page` of the rank's staging tensor
    /// (`PoolConversion`), converted into the units as it is loaded.
    pub staged: bool,
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
/// page of the rank's host latent pool, a page pool of the rank's host's
/// Storage Server ([`LatentOffload`]): page `page` of the trajectory's
/// pages in the bank holding its committed step goes to the host page,
/// which the engine reserved. A video trajectory's pages are its request
/// slot's. A pool page holds as many bytes as a bank page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentPageCopy {
    /// The rank of the worker group that copies the page.
    pub rank: u32,
    /// The latent pool at the rank's host's Storage Server.
    pub pool: u64,
    /// The page's index in the host latent pool.
    pub host_page: u64,
    /// The page's index among the trajectory's pages, from zero.
    pub page: u32,
}

/// A preempted video request's trajectory, written from the bank holding its
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

/// The stored trajectory a resumed video request's latent preparation opens
/// at: the call
/// prepares the trajectory as for step zero, then each rank copies the
/// parts of the stored pages its own pages hold into the bank it
/// initializes, as the copy manifest from the stored ranks' layouts to its
/// own directs, and commits the trajectory at step `steps`. The engine
/// pinned the pool slots until the call's result.
///
/// The stored ranks may have laid the samples out under another parallel
/// configuration than the restoring ranks: every rank prepares its copies
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
/// pool request finds it by (`storage::protocol::Request::Pool`).
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

/// Cursor and unit bounds for one diffusion decode call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeRange {
    /// Request that owns the media output.
    pub request_key: RequestKey,
    /// Decode call receiving the params.
    pub call_id: crate::CallId,
    /// First decoder unit assigned to the call.
    pub cursor: u32,
    /// Maximum decoder units the call may process.
    pub max_units: u32,
    /// Ranks of a distributed stage that hold the call's units, in unit
    /// order: each takes the next `units_per_rank` units and the last what
    /// remains. Empty deals the units to the stage's ranks in configured
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

#[cfg(test)]
mod joined_tests {
    use super::*;

    fn done(id: u64) -> Submission {
        Submission::completed(SubmissionId(id), Outcome::Success(BatchOutput::default()))
    }

    /// A batch a group runs together has ended when every rank's part ended, and the rank that
    /// failed decides the response: the others' output lacks its part.
    #[test]
    fn a_joined_submission_waits_for_every_rank_and_returns_a_peers_failure() {
        let (pending, state) = Submission::pending(SubmissionId(2));
        let joined = done(1).joined(vec![pending]);
        assert!(joined.outcome().is_none(), "a peer has not ended");
        state.finish(Outcome::Failed(Arc::new(Refusal("a peer failed"))));
        assert!(matches!(joined.outcome(), Some(Outcome::Failed(_))));
        assert!(matches!(joined.wait(), Outcome::Failed(_)));

        let joined = done(3).joined(vec![done(4), done(5)]);
        assert!(matches!(joined.wait(), Outcome::Success(_)));
    }
}

#[cfg(test)]
mod media_tests {
    use super::*;
    use proto::video::{LatentShape, Layout};

    fn layout() -> Layout {
        Layout {
            video: LatentShape {
                channels: 16,
                frames: 13,
                height: 60,
                width: 34,
            },
            audio: None,
            text_rows: 256,
            condition_rows: 0,
            condition_elements: 0,
        }
    }

    /// A media batch's columns are the model's prepared buffers, one set per request, and the
    /// family's rules are its own: a computation no variant states, a request count the batch does
    /// not carry, and a denoising step without its resolution are each refused before a launch - while
    /// an encode, which is sized by rows and tokens, states no extent and is accepted.
    #[test]
    fn a_media_batch_states_one_set_of_inputs_per_request_and_its_own_extent() {
        let denoise = MediaColumns {
            computation: Computation {
                call: crate::MediaCall::Denoising,
                rows: 1,
                tokens: 0,
                extent: Extent::Latents(layout()),
            },
            inputs: vec![vec![vec![0u8; 64]]],
            reads: vec![Vec::new()],
            writes: vec![Vec::new()],
        };
        assert_eq!(denoise.bytes(), 64);
        denoise.check().expect("one request, its resolution stated");

        // A mux reads what the encodes already wrote, so no resolution sizes it: a **host** call
        // states `Extent::None` and is accepted, which is what separates it from the device
        // computation whose extent is asserted below. The case that used to sit here made muxing
        // the example of a call no variant states, and that stopped being true when it gained one.
        let muxed = MediaColumns {
            computation: Computation {
                call: crate::MediaCall::Muxing,
                rows: 1,
                tokens: 0,
                extent: Extent::None,
            },
            inputs: vec![Vec::new()],
            reads: vec![Vec::new()],
            writes: vec![Vec::new()],
        };
        muxed
            .check()
            .expect("a host call is sized by the request's own products, not a resolution");
        assert_eq!(muxed.bytes(), 0, "a mux stages no extent of its own");

        let wrong_rows = MediaColumns {
            inputs: vec![Vec::new(), Vec::new()],
            ..denoise.clone()
        };
        let refusal = wrong_rows
            .check()
            .expect_err("two request sets for a one-row batch");
        assert!(
            refusal.to_string().contains("2 request(s)"),
            "the refusal names the count: {refusal}"
        );

        // A row whose products are not stated would leave its runner without the spans it reads.
        let unplaced = MediaColumns {
            reads: Vec::new(),
            ..denoise.clone()
        };
        let refusal = unplaced
            .check()
            .expect_err("a one-row batch states the products of one request");
        assert!(
            refusal.to_string().contains("products of 0 and 1"),
            "the refusal names the counts: {refusal}"
        );

        let no_resolution = MediaColumns {
            computation: Computation {
                call: crate::MediaCall::Denoising,
                rows: 1,
                tokens: 0,
                extent: Extent::None,
            },
            ..denoise.clone()
        };
        let refusal = no_resolution
            .check()
            .expect_err("a denoising step states its resolution");
        assert!(
            refusal.to_string().contains("states no extent"),
            "the refusal names the missing extent: {refusal}"
        );

        let encode = MediaColumns {
            computation: Computation {
                call: crate::MediaCall::TextEncoding,
                rows: 1,
                tokens: 300,
                extent: Extent::None,
            },
            inputs: vec![vec![vec![1u8; 8]]],
            reads: vec![Vec::new()],
            writes: vec![Vec::new()],
        };
        assert_eq!(encode.computation.variant(), Some(Variant::Encode));
        encode
            .check()
            .expect("an encode is sized by rows and tokens");
    }

    /// A staged request is the mechanism's record of one request, and it round-trips through the
    /// two buffers it is: the conditioning verbatim and the parameters little-endian at fixed
    /// offsets, so a runner reads offsets rather than a parser. A row of the wrong shape is
    /// refused by name - a truncated record would be a step index nobody stated.
    #[test]
    fn a_staged_request_is_the_conditioning_blob_and_its_parameter_record() {
        let staged = StagedRequest {
            unit: 0,
            request: crate::RequestKey::new(7, crate::RequestId(41), 3),
            conditioning: vec![9, 8, 7],
            step: 3,
            steps: 12,
            seed: 0x0102_0304_0506_0708,
            width: 768,
            height: 512,
            frames: 121,
            rounds: 0,
        };
        let row = staged.encode();
        assert_eq!(row.len(), 2, "a staged request is two buffers");
        assert_eq!(
            row[0],
            vec![9, 8, 7],
            "the conditioning is carried verbatim"
        );
        assert_eq!(row[1].len(), PARAMETER_BYTES);
        assert_eq!(&row[1][0..4], &3u32.to_le_bytes(), "step");
        assert_eq!(&row[1][4..8], &12u32.to_le_bytes(), "steps");
        assert_eq!(
            &row[1][8..16],
            &0x0102_0304_0506_0708u64.to_le_bytes(),
            "seed"
        );
        let decoded = StagedRequest::decode(&row).expect("the record round-trips");
        assert_eq!(decoded, staged);
        assert_eq!(
            decoded.request,
            crate::RequestKey::new(7, crate::RequestId(41), 3),
            "the identity is the row's name for what a step must find again"
        );

        let columns = MediaColumns::staged(
            Computation {
                call: crate::MediaCall::Denoising,
                rows: 1,
                tokens: 0,
                extent: Extent::Latents(layout()),
            },
            std::slice::from_ref(&staged),
            &[&crate::Call::new(
                staged.request,
                crate::CallId::new(1, 0),
                crate::CallKind::Media(crate::MediaCall::Denoising),
            )],
        );
        assert_eq!(columns.request(0).expect("one row"), staged);
        let out_of_range = columns.request(1).expect_err("one row, asked for two");
        assert!(
            format!("{out_of_range}").contains("row 1 is not one of them"),
            "{out_of_range}"
        );
    }

    /// A staged request's two buffers are both load-bearing, and the refusals name which one is
    /// wrong: a document's worth of buffers, or a record of another width, is not this record.
    #[test]
    fn a_staged_request_of_the_wrong_shape_is_refused_by_name() {
        let short = vec![vec![0u8; 3], vec![0u8; PARAMETER_BYTES - 1]];
        let error = StagedRequest::decode(&short).expect_err("a truncated record");
        let text = format!("{error}");
        assert!(
            text.contains("step, steps, seed, width, height, frames"),
            "the refusal does not name the record's shape: {text}"
        );
        assert!(
            text.contains(&format!("{} bytes", PARAMETER_BYTES - 1)),
            "the refusal does not name the record it was given: {text}"
        );

        let one = vec![vec![1u8; 3]];
        let error = StagedRequest::decode(&one).expect_err("one buffer is not the record");
        assert!(
            format!("{error}").contains("carries 1 buffer(s)"),
            "{error}"
        );
    }
}

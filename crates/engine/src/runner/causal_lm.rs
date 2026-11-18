//! `CausalLmRunner`: the runner of every causal language model.
//!
//! The runner holds its model through `nn::family::TextModel` and never names a concrete one.
//! It owns the input tensors its forwards read, the request slots' device state and the
//! sampler over it, and its [`Graphs`]; it borrows the key/value cache and the recurrent state
//! as tensors their owners lend.
//!
//! ## The forward
//!
//! One function computes every bucket, on the meta device for the profile run, under a capture,
//! and again before each replay ([`Runner::forward`]). It is tensor operations and model calls
//! over the input tensors alone:
//!
//! ```text
//! pooled page tables [slots + 1, width] ─ index_select(slot rows) → KvCache::block_tables
//! next input tokens  [slots + 1, …]     ─ index_select(slot rows) → a decode's input ids
//! contexts           [slots + 1, w]     ─ index_select(slot rows) → TokenBatch::token_history
//! accepted counts    [slots + 1]        ─ index_select(slot rows) → VerifySlots::num_accepted
//! input ids                             → embed_input_ids         → hidden [tokens, hidden]
//! hidden, batch, cache, state           → forward                 → hidden [tokens, hidden]
//! hidden                                ─ index_select(logits rows) → compute_logits
//! logits [rows, vocabulary]             → Sampler::greedy | forward → tokens [rows]
//! logits, tokens                        → Sampler::score          → logprobs, ranks, candidates
//! hidden, targets (a scoring bucket)    → compute_logits, Sampler::score, in chunks
//! tokens, commit table                  → Sampler::commit_        → the slots' state
//! contexts, input ids                   ─ index_select(next contexts) → scatter(slots) → contexts
//! one per row                           ─ scatter(slots)          → accepted counts
//! positions [slots + 1]                 ─ index_select(call slots) → each call's positions
//! ```
//!
//! A decode row's input token is the one the step before it committed into its slot, read on
//! the device, so a step can be issued behind the step it continues. Every bucket selects a
//! token per row and scores it: whether a row's request asked is not a value a captured graph
//! branches on, and the completion keeps what the batch asked for. The outputs and their order
//! are stated in [`output`].
//!
//! A model whose layers read token ids (`TextModel::token_history`) is lent the ids the
//! forward embeds and the `w` ids before each row its history declares, which its slot keeps
//! and the forward advances to the row's last `w` (`columns`, token ids). A runner that
//! speculates over a
//! model whose state keeps a slot per verified token (`nn::family::StateField::stepwise`) lends
//! every forward its state rows' verify slots (`nn::VerifySlots`; `columns`, verify slots):
//! their slots, which an input column states, and how many tokens of their slots' last steps
//! stand, which the slots keep. A forward of a prompt or a decode leaves one standing, and a
//! verification its accepted tokens (`speculation`). A paired forward advances neither.
//!
//! A runner lent a feature buffer ([`CausalLmRunner::set_features`]) serves prefills with image
//! rows: before the replay the encoder output each image block names is copied from the block
//! the rank's tables hold into the buffer, and the forward of every prefill bucket takes the
//! marked rows' embeddings from it and lends the batch its rotary coordinates (`features`).
//!
//! ## A step
//!
//! Before a batch, the worker hands the runner the changes of the rank's device block table
//! ([`Runner::write_tables`]) and the copies that start with the batch ([`Runner::copy`]).
//! [`Runner::execute`] then serves the batch over that table:
//!
//! 1. the batch is checked and the smallest captured bucket that holds it is chosen;
//! 2. the cache slot each token writes is computed from its cache position through the table,
//!    and a call that would write an entry that is missing or read-only is refused
//!    (`columns::slot_mapping`);
//! 3. the batch's columns are padded to the bucket ([`columns`]);
//! 4. the slots the batch starts a request on are started, and its ban masks copied
//!    ([`slots`]);
//! 5. the batch's page work that precedes the forward is issued ([`PageWork::BeforeReplay`]);
//! 6. the padded columns are written into the input block, with the host values the forward
//!    computes, and the bucket's graph is replayed;
//! 7. the page work that follows the forward is issued ([`PageWork::AfterReplay`]);
//! 8. the step's outputs are copied to page-locked host memory behind it (`readback`);
//! 9. the event behind that copy is waited for, and the copies the batch took
//!    ([`PageWork::Ended`]); one `CallOutput` per call is assembled from the outputs on the
//!    host, in the batch's order, a refused call's as a failure.
//!
//! Steps 4 to 8 are issued on one stream in that order, which is the only synchronization
//! between them: the replay reads the slot state, the pooled page tables and the pages as the
//! copies before it left them.
//!
//! ## Steps in flight
//!
//! A batch that overlaps (`CausalLmRunner::overlaps`: no speculation, no expert union, no
//! feature buffer) is served in two turns of the worker. [`Runner::start`] issues steps 1 to 8
//! and returns; the runner's lander thread waits for the event behind the step's read and wakes
//! the worker, which meanwhile issues its next job, the next step included, on the stream
//! behind it. [`Runner::execute`] of the batch then lands the step: step 9. The host's work for
//! a step (its columns, the forward that writes the host values, the launch) is thus done while
//! the device computes the step before it, as the scheduler's queue depth intends: the next
//! decode reads its input token from the slot state the previous one committed on the device.
//! The step's outputs are read from the host copy, which no later replay writes, and the slot
//! state's write buffer is written again only once the stream has passed its last copies.
//!
//! A batch with no rows, which a group with copies queued and no other work is sent
//! (`specs/scheduler-design.md` §2.1), takes only step 9: the table's changes and the copies
//! reached the runner before it, nothing is replayed, the copies are waited for, and its output
//! names no call. A step that fails waits for the copies all the same before it returns.
//!
//! ## Page work
//!
//! The key/value cache and the recurrent state belong to the rank, which two runners of a
//! device may share. The runner takes their tensors at construction and never copies a page:
//! the device block table's changes, the copies of blocks and the state copies are issued by
//! the rank's cache through the hook given to [`CausalLmRunner::set_page_work`]. A runner with
//! no hook serves the batches that carry no page work and refuses one that carries some.
//! [`wiring`] builds the runner and the hook from the rank's cache and state modules.
//!
//! ## Paired forwards
//!
//! Before a generation switch the worker computes a served batch again as a paired forward
//! (`Graphs::shadow`), whose inputs [`CausalLmRunner::prepare_shadow_inputs`] writes. It is the
//! same forward over other pages: the rows write and read the scratch page set the cache lends
//! ([`CausalLmRunner::set_scratch`]), the commit table is inert and no slot is started, so
//! nothing a serving request reads is written. A model that keeps a recurrent state has no
//! scratch state to advance, and its paired forwards are refused.
//!
//! The scratch page set is one set the rows of a batch share: `ceil(T / page) + R` pages for
//! the `T` tokens and `R` rows of the runner's largest bucket
//! ([`CausalLmRunner::scratch_pages`]). It hands its pages to the rows in order, each row the
//! pages of the context it reads, its history copied in before the forward
//! ([`PageWork::BeforeShadow`]). A batch whose rows' contexts need no more than `T` tokens
//! always fits, since a row wastes less than one page. A batch whose rows need more pages than
//! the set holds has no paired forward: it is reported uncompared (`Graphs::unpaired`), and the
//! batch itself was served as it was.
//!
//! ## Expert unions
//!
//! A runner whose model's experts are split across ranks is a member of an expert union
//! ([`CausalLmRunner::join_union`], [`experts`]). Its buckets are captured once per capacity
//! class, with the join bucket beside them. Before the replay of a batch the member agrees
//! the step's capacity with the other members, which blocks until each has entered, and
//! replays the smallest class of its bucket that holds it; a member with no batch is driven
//! by a join batch (`Variant::ExpertJoin`) and replays the join bucket, the expert layers
//! alone. Everything of a step that can refuse the batch is done before the agreement: once
//! a step is agreed, every member launches it, or the others wait on their devices.
//!
//! ## Speculation
//!
//! A runner whose config states drafts speculates with its model's drafter ([`speculation`]).
//! Its decode buckets are verification buckets, and every prefill and verification bucket has
//! the drafter's bucket of its rows and tokens beside it. A step replays two graphs back to
//! back: the batch's own, which for a verification samples every token, accepts the drafts
//! its samples agree with and commits them, and the drafter's, which writes the draft cache
//! and proposes each row's next drafts. Nothing is read back between them: what the second
//! graph and the completion read of the first is kept in the runner's memory. A drafter whose
//! layers keep a state (`nn::Drafter::state_layers`) is lent its live rows by each of its forwards,
//! and every proposal is the argmax of the model's head over the drafter's own read of its
//! output (`nn::Drafter::logits_rows`).

mod columns;
pub(crate) mod experts;
mod features;
mod output;
mod readback;
mod slots;
pub mod speculation;
pub(crate) mod wiring;

#[cfg(test)]
mod tests;

use std::cell::Cell;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use crate::batch::{Batch, BatchOutput, Phase, Shape as BatchShape, TimingCounters, Variant};
use nn::family::{StateField, StateShape, TextModel};
use nn::{KvCache, RecurrentState, TokenBatch, VerifySlots};
use platform::{Event, Stream};
use proto::module::commit::COMMIT_FIELDS;
use proto::module::sampler::{PARAMETER_COLUMNS, TEMPERATURE};
use tensor::{DType, Shape, Tensor};

use self::columns::{Column, Layout, Refusal};
use self::experts::Member;
use self::features::Features;
use self::readback::{Pending, Readback};
use self::slots::Slots;
use self::speculation::{DraftPool, DraftTable, Entry, Kept, Speculation};
use crate::memory::buffers::BufferMemory;
use crate::memory::cache::{CacheLayout, scratch};
use crate::memory::state::layer_fields;
use crate::memory::tables::{BlockCopy, BlockTables, Change};
use crate::runner::DeviceCopy;
use crate::runner::Runner;
use crate::runner::buckets::{Bucket, Buckets};
use crate::runner::graphs::Graphs;
use crate::runner::sampler::{self, Sampler, SamplerConfig};
use crate::worker::completion::Waker;
use crate::worker::{Error, Result};

/// The tokens one chunk of a scoring bucket's prompt projection holds: a chunk's logits and
/// log-probabilities are `[SCORE_ROWS, vocabulary]` FP32 each (39 MB over 151936 tokens), and
/// the next chunk takes the memory the one before it returned.
const SCORE_ROWS: usize = 64;

/// What a deployment fixes of a causal runner. Immutable once the runner is built.
#[derive(Clone, Debug)]
pub struct CausalLmConfig {
    /// The phases the runner serves, as the scheduler names them in a batch.
    pub phases: Vec<Phase>,
    /// The buckets the runner captures (`Buckets::for_batch`):
    /// their largest rows and tokens are the batch bound the rank admits.
    pub buckets: Buckets,
    /// Request slots: the rows of the slot state and of the pooled page tables, which hold one
    /// row more, the padding row.
    pub slots: u32,
    /// Tokens of one cache page.
    pub page_size: u32,
    /// The longest key reduction the deployment serves: the attention descriptor's bound,
    /// one value for every step.
    pub max_kv_len: u32,
    /// The most likely tokens a scored row reports beside its own: the deployment's
    /// `max_logprobs`, and zero to report none.
    pub candidates: u32,
    /// The pages of the scratch page set a paired forward's rows share
    /// ([`CausalLmRunner::scratch_pages`]), and zero for a runner that computes no paired
    /// forward. A batch whose rows' contexts need more is not compared.
    pub scratch_pages: u32,
    /// The drafts a verification holds at most (`speculative_config.num_draft_tokens`), and
    /// zero for a runner that speculates nothing. A runner that states some serves its decode
    /// steps as verifications and runs its model's drafter behind every step.
    pub drafts: u32,
    /// How the draft cache lies in the cache's unit pool ([`DraftPool::of`]): stated exactly
    /// when `drafts` is above zero.
    pub draft_pool: Option<DraftPool>,
}

/// Which part of a batch's page work the rank's cache issues, with what that part reads.
#[derive(Clone, Copy)]
pub enum PageWork<'a> {
    /// The rank's device block table changed at these positions since the last batch
    /// ([`BlockTables::take_changes`]): the units the buffer pool allocated are reset and the
    /// pooled page tables' rows that changed are written, before the next forward reads them.
    Tables(&'a [Change]),
    /// The copies of blocks that start with the next batch ([`BlockTables::start_copy`]): each
    /// from its first run to its second, ended before the batch is.
    Copies(&'a [BlockCopy]),
    /// What the forward of the batch reads: its state copies are resolved through the tables,
    /// a checkpoint's position to the run of units its entry holds, and refused where an entry
    /// is missing or a keep's is read-only; each restored slot's recurrent state is copied into
    /// the live buffers.
    BeforeReplay(&'a Batch, &'a BlockTables),
    /// What the forward left: each kept checkpoint of the recurrent state is copied to the
    /// block its position names.
    AfterReplay,
    /// The batch ended, or failed, or the batch another runner served that the copies started
    /// with ([`Runner::wait_copies`]): the copies issued since the last end are waited for, so
    /// the device has completed them when the runner returns.
    Ended,
    /// Whether the cache's memory holds the region of this number (`KvCacheMemory::holds`): the
    /// hook sets the flag when it does, and issues nothing.
    Holds(u8, &'a Cell<bool>),
    /// What a paired forward of the batch reads: the pages its rows have cached are copied into
    /// the rows' scratch pages, row `r`'s from the first page of the set it takes, the slice's
    /// entry `r` (`columns::scratch_spans`).
    BeforeShadow(&'a Batch, &'a [u32]),
}

/// The buffers of a recurrent state, as `nn::RecurrentState` lays them out: one entry per
/// recurrent layer, in the model's order of them, each the layer's fields as `[slots, …]`
/// tensors.
pub(crate) type StateBuffers = Vec<Vec<Tensor>>;

/// The seam through which the rank's cache issues a batch's page work on the runner's stream
/// ([`CausalLmRunner::set_page_work`]).
pub(crate) type PageWorkHook = Box<dyn FnMut(PageWork<'_>) -> Result<()> + Send>;

/// The runner of a causal language model (module documentation).
pub struct CausalLmRunner {
    model: Arc<dyn TextModel>,
    config: CausalLmConfig,
    graphs: Graphs,
    /// Where each input tensor lies in the input block.
    layout: Layout,
    /// The request slots' device state.
    slots: Slots,
    /// The token selection over the slots' count and bias tables.
    sampler: Sampler,
    /// The model's attention groups, as the cache lends them: each group's `block_tables` is
    /// the group's pooled page table.
    cache: Vec<KvCache>,
    /// The live and the checkpoint buffers of the recurrent state, when the model keeps one.
    state: Option<(StateBuffers, StateBuffers)>,
    /// What issues a batch's page work; `None` for a runner that serves no batch with any.
    page_work: Option<PageWorkHook>,
    /// The scratch page set of paired forwards, group for group as `cache`; empty for a
    /// runner that was lent none.
    scratch: Vec<KvCache>,
    /// Whether the prepared inputs are a paired forward's: the forward then computes over
    /// `scratch`. Set by [`CausalLmRunner::prepare_shadow_inputs`] and cleared by every other
    /// fill of the input block, so no capture and no replay computes over the scratch set.
    paired: Cell<bool>,
    /// The buckets the runner states: the config's, or their capacity classes and the join
    /// bucket for a member of an expert union.
    buckets: Buckets,
    /// This rank's member of the expert union, for a model whose experts are split across
    /// ranks.
    union: Option<Member>,
    /// What a runner that speculates holds beside its sampler: the draft cache it is lent,
    /// the selection of its verifications and what its forwards keep.
    speculation: Option<Speculation>,
    /// The stream the graphs replay on; `None` for graphs that hold no device.
    stream: Option<Stream>,
    /// The feature buffer of prefills with image rows, for a runner lent one.
    features: Option<Features>,
    /// The bytes of the input block of the step being prepared: a batch's columns are padded
    /// here, where a refusal costs nothing, before anything of the step is issued.
    staged: Vec<u8>,
    /// The page-locked memory each step's outputs are copied to behind it; `None` for graphs
    /// that hold no device.
    readback: Option<Readback>,
    /// How many steps the runner has launched: the number of the last one.
    launched: u64,
    /// The steps [`Runner::start`] launched and whose landing turn has not come, oldest first.
    flights: VecDeque<Flight>,
    /// The thread that wakes the worker once a step in `flights` completed on the device,
    /// started with the runner's first such step.
    lander: Option<Lander>,
}

/// A step the runner launched and has not landed: what its landing reads beside its batch.
struct Flight {
    /// The batch whose step this is: its landing is [`Runner::execute`] of the same batch.
    batch: crate::batch::BatchId,
    /// The step's number among the runner's launches.
    number: u64,
    /// The bucket that served the batch.
    bucket: Bucket,
    /// The calls refused before the step was issued, by their index among the batch's calls.
    refused: Vec<Refusal>,
    /// The step's outputs, and behind them the drafter's proposals, on their way to the host.
    pending: Pending,
    /// How many of the read tensors are the drafter's proposals, the last ones.
    proposals: usize,
    /// When the runner took the batch, and when it launched the step.
    started: Instant,
    launched: Instant,
}

/// The thread that waits, for each step a runner launched ahead of its landing, until the device
/// completed it, and then wakes the worker for the step's landing turn ([`Runner::start`]).
/// It is what keeps the worker thread from blocking on a step while it could issue the next
/// one: the wait for the device moves here, and the worker blocks only on its queue.
struct Lander {
    steps: Option<crossbeam_channel::Sender<(Arc<Event>, Waker)>>,
    thread: Option<JoinHandle<()>>,
}

impl Lander {
    /// Starts the thread, which ends once the lander is dropped and every step handed to it was
    /// waited for.
    fn spawn() -> Result<Self> {
        let (steps, landing) = crossbeam_channel::unbounded::<(Arc<Event>, Waker)>();
        let thread = std::thread::Builder::new()
            .name("lander".to_owned())
            .spawn(move || {
                for (event, resume) in landing {
                    // A failed wait is the landing's to report: it waits on the same event, and
                    // the device's failure fails the batch there.
                    let _ = event.synchronize();
                    resume();
                }
            })
            .map_err(|error| Error::transport_source("starting a runner's lander thread", error))?;

        Ok(Self {
            steps: Some(steps),
            thread: Some(thread),
        })
    }

    /// Calls `resume` once the work `event` captured has completed, after every step handed
    /// over before it.
    fn watch(&self, event: Arc<Event>, resume: Waker) -> Result<()> {
        self.steps
            .as_ref()
            .and_then(|steps| steps.send((event, resume)).ok())
            .ok_or(Error::State("a runner's lander thread has stopped"))
    }
}

impl Drop for Lander {
    fn drop(&mut self) {
        drop(self.steps.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl CausalLmRunner {
    /// The bytes of the input tensors of a runner of `model` under `config`: the `inputs` its
    /// [`Graphs`] are made with.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket.
    pub fn input_bytes(config: &CausalLmConfig, model: &dyn TextModel) -> Result<u64> {
        Ok(stated(config, model)?.1.bytes())
    }

    /// The pages of the scratch page set paired forwards of a runner of `buckets` compute over,
    /// for pages of `page_size` tokens: `ceil(T / page_size) + R` for the `T` tokens and `R`
    /// rows of the largest bucket (`Buckets::bounds`).
    ///
    /// The rows of a batch share the set and take its pages in order, each the pages of the
    /// context it reads, and a row wastes less than one page, so every batch whose rows' contexts
    /// sum to at most `T` tokens fits: a prompt's chunks with no history always do.
    pub(crate) fn scratch_pages(buckets: &Buckets, page_size: u32) -> u32 {
        let (rows, tokens) = buckets.bounds();

        tokens.div_ceil(page_size.max(1)).saturating_add(rows)
    }

    /// The bytes a cache write of a paired forward of a runner under `config`, over a cache of
    /// `layout`, writes in place beyond what the same call of a profiled forward writes, which
    /// the shadow region holds beside the largest profiled call (`Graphs::shadow_peak`): one
    /// layer of the scratch page set (`memory::cache::scratch::in_place`), the pages a paired
    /// forward's cache writes write instead of the cache's. Zero for a runner that computes no
    /// paired forward. What every other paired call writes in place, the request slots' state
    /// and a speculating runner's kept outputs included, is the profiled call's own.
    pub(crate) fn cache_writes(config: &CausalLmConfig, layout: &CacheLayout) -> u64 {
        scratch::in_place(layout, config.scratch_pages)
    }

    /// The bytes of device memory a runner of `model` that speculates under `config` keeps its
    /// target forwards' outputs in: each bucket's chosen tokens and scores and the rows the
    /// largest one's forward returns, `TextModel::stream_width` wide in the model's element
    /// type, which the drafter's forward reads. Zero for a config that states no drafts.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket, or one that states drafts and no draft
    /// pool.
    pub(crate) fn kept_bytes(config: &CausalLmConfig, model: &dyn TextModel) -> Result<u64> {
        if config.drafts == 0 {
            return Ok(0);
        }
        let layout = stated(config, model)?.1;

        Kept::bytes(
            layout.rows(),
            layout.tokens(),
            config.candidates,
            model.stream_width(),
            model.dtype(),
        )
    }

    /// The bytes of device memory the request slots' state takes for a runner of `model` under
    /// `config`: the count and bias tables over the model's vocabulary, the commit state, the
    /// ban masks, and the contexts and accepted counts where its forwards read them, which the
    /// runner allocates when its graphs hold a device.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket, or a model whose state keeps a slot per
    /// verified token under a config that states no drafts.
    pub(crate) fn slot_bytes(config: &CausalLmConfig, model: &dyn TextModel) -> Result<u64> {
        let layout = stated(config, model)?.1;
        let owned = owned_slots(model, config)?;

        Ok(Slots::bytes(
            config.slots,
            model.vocabulary(),
            layout.rows() as u32,
            1,
            model.token_history(),
            owned.is_some(),
        ))
    }

    /// A runner of `model` under `config`.
    ///
    /// - `graphs` are made for [`CausalLmRunner::input_bytes`] bytes of input tensors.
    /// - `stream` is the stream `graphs` replay on, on which the runner issues the slot
    ///   state's writes; `None` for graphs that hold no device (`Graphs::meta`), whose runner
    ///   computes over meta tensors.
    /// - `cache` is the model's attention groups, in the order its attention layers index
    ///   them, as the rank's cache lends them: each group's planes and workspace, and in
    ///   `block_tables` the group's **pooled** page table, `[slots + 1, width]` U32, row
    ///   `slot` holding the pages of the request on that slot and the last row naming the
    ///   padding unit alone. A forward gathers its rows' tables from it.
    /// - `state` is the `(live, checkpoint)` buffers of the recurrent state, in the layout of
    ///   `nn::RecurrentState`, for a model that keeps one: a buffer per field of every layer of
    ///   the model's runs (`TextModel::state_layers`), the live buffer of a field that keeps a
    ///   slot per verified token holding `drafts + 1` rows a request slot when the config
    ///   states drafts (`StateMemory`).
    ///
    /// The caller keeps the memory of `cache` and `state` valid and at its addresses for as
    /// long as the runner exists, and writes it only on `stream`.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a bucket that is no causal forward, input tensors `graphs` do
    /// not hold, a stream without a device or the reverse, a cache that is not the model's
    /// layers over `config`'s page size and slots, a state that is not the model's over the
    /// config's slots, a model whose state keeps a slot per verified token under a config that
    /// states no drafts, or a config that states drafts for a model that loaded no drafter or
    /// without the draft pool of its cache; the refusals of the slot state's allocation and
    /// of the sampler.
    pub fn new(
        model: Arc<dyn TextModel>,
        config: CausalLmConfig,
        graphs: Graphs,
        stream: Option<Stream>,
        cache: Vec<KvCache>,
        state: Option<(StateBuffers, StateBuffers)>,
    ) -> Result<Self> {
        for bucket in config.buckets.iter() {
            bucket.causal_lm()?;
        }
        let (buckets, layout) = stated(&config, &*model)?;

        if graphs.enter().is_ok() != stream.is_some() {
            return Err(Error::Invalid(
                "a causal runner takes the stream of its graphs, and none for graphs that hold \
                 no device"
                    .into(),
            ));
        }
        check_cache(&*model, &config, &cache)?;
        let owned = owned_slots(&*model, &config)?;
        check_state(&*model, &config, owned, state.as_ref())?;

        let vocabulary = model.vocabulary();
        let slots = Slots::new(
            stream.clone(),
            config.slots,
            vocabulary,
            layout.rows() as u32,
            1,
            model.token_history(),
            owned.is_some(),
        )?;
        let sampler = Sampler::new(
            SamplerConfig {
                vocabulary,
                slots: config.slots,
                // A request may ask for any top-k, so a drawn row keeps the whole vocabulary
                // as its candidates.
                width: vocabulary,
                candidates: config.candidates,
                extras: 0,
            },
            slots.counts().clone(),
            slots.bias().clone(),
        )?;

        let speculation = match config.draft_pool {
            Some(pool) if config.drafts > 0 => {
                if model.drafter().is_none() {
                    return Err(Error::Invalid(
                        "the deployment speculates and its model loaded no drafter".into(),
                    ));
                }

                // A verification's rows count the drafts before the position they sample.
                let verifier = Sampler::new(
                    SamplerConfig {
                        extras: config.drafts,
                        ..*sampler.config()
                    },
                    slots.counts().clone(),
                    slots.bias().clone(),
                )?;
                // The drafter reads the rows the model's forward returns, which are its
                // streams for a model that carries several.
                let kept = Kept::new(
                    stream.as_ref(),
                    layout.rows(),
                    layout.tokens(),
                    config.candidates,
                    model.stream_width(),
                    model.dtype(),
                )?;

                Some(Speculation {
                    drafts: config.drafts,
                    pool,
                    kept,
                    verifier,
                    draft: None,
                    trail: None,
                    state: None,
                })
            }
            _ => None,
        };

        let readback = stream.as_ref().map(|stream| Readback::new(stream.device()));
        let runner = Self {
            buckets,
            speculation,
            model,
            config,
            graphs,
            layout,
            slots,
            sampler,
            cache,
            state,
            page_work: None,
            scratch: Vec::new(),
            paired: Cell::new(false),
            union: None,
            stream,
            features: None,
            staged: Vec::new(),
            readback,
            launched: 0,
            flights: VecDeque::new(),
            lander: None,
        };

        // Every input tensor lies inside the block the graphs were made for.
        let block = runner.layout.bytes() as usize;
        if runner.graphs.input(0, block, DType::U8).is_err() {
            return Err(Error::Invalid(format!(
                "the graphs hold fewer than the {} bytes of the runner's input tensors",
                runner.layout.bytes()
            )));
        }
        Ok(runner)
    }

    /// Lends a runner that speculates the draft cache its drafter's forwards compute over,
    /// the trail of the recurrent state when its model keeps one and its forwards lend no
    /// verify slots, and the drafter's state when the drafter keeps one.
    ///
    /// - `draft` is the draft group as the rank's cache lends it: the drafter's planes, pages
    ///   of the draft pool's page size, and the workspace. Its `block_tables` is not read: a
    ///   forward states each row's pages as an input column.
    /// - `trail` is the trail buffers of the recurrent state, in the layout of
    ///   `nn::RecurrentState::trail`, with a row for each token of the largest verification
    ///   bucket. A runner whose forwards lend verify slots keeps the state after each token in
    ///   its requests' slots, and is lent none.
    /// - `drafter` is the live buffers of the drafter's state (`nn::Drafter::state_layers`), in the
    ///   layout of `nn::RecurrentState::live`, a home row per request slot of each field
    ///   (`StateMemory::drafter`); `None` for a drafter that keeps no state. The drafter's
    ///   forwards are lent them and advance them in place.
    ///
    /// The caller keeps their memory as it keeps the cache's and the state's.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a runner that speculates nothing, a draft cache that is not the
    /// drafter's layers over the draft pool's pages, a trail that is not the model's
    /// recurrent state over the tokens of a verification, or that the runner keeps none of,
    /// or a drafter's state that is not the drafter's fields over the config's request slots.
    pub fn set_draft_cache(
        &mut self,
        draft: KvCache,
        trail: Option<StateBuffers>,
        drafter: Option<StateBuffers>,
    ) -> Result<()> {
        let (layers, drafter_runs) = match self.model.drafter() {
            Some(drafter) => (Some(drafter.cache().layers), drafter.state_layers()),
            None => (None, Vec::new()),
        };
        let Some(speculation) = self.speculation.as_mut() else {
            return Err(Error::Invalid(
                "a draft cache lent to a runner whose config states no drafts".into(),
            ));
        };

        if Some(draft.layers.len() as u32) != layers || draft.page_size != speculation.pool.page {
            return Err(Error::Invalid(format!(
                "a draft cache of {} layers over pages of {} tokens, and the drafter's {layers:?} \
                 layers read pages of {}",
                draft.layers.len(),
                draft.page_size,
                speculation.pool.page
            )));
        }

        // The trail holds the state after every token of the largest verification.
        let tokens = self
            .buckets
            .iter()
            .filter(|bucket| verifies(bucket.variant))
            .map(|bucket| bucket.tokens as usize)
            .max()
            .unwrap_or(0);
        let runs = self.model.state_layers();
        let layers: Vec<&[StateField]> = layer_fields(&runs).collect();
        let trails = !layers.is_empty() && self.slots.accepted().is_none();

        let trailed = match &trail {
            None => !trails,
            Some(trail) => {
                trails
                    && trail.len() == layers.len()
                    && trail.iter().zip(&layers).all(|(layer, fields)| {
                        layer.len() == fields.len()
                            && layer.iter().all(|field| {
                                field.dims().first().is_some_and(|rows| *rows >= tokens)
                            })
                    })
            }
        };
        if !trailed {
            return Err(Error::Invalid(format!(
                "a verification of a model that keeps a recurrent state keeps a trail of its \
                 {tokens} tokens for every field unless its forwards lend verify slots, and any \
                 other verification is lent none"
            )));
        }
        check_drafter_state(&drafter_runs, self.config.slots, drafter.as_ref())?;

        speculation.draft = Some(draft);
        speculation.trail = trail;
        speculation.state = drafter;
        Ok(())
    }

    /// Sets what issues a batch's page work on the runner's stream.
    ///
    /// The runner calls `hook` on the worker thread: with [`PageWork::Tables`] for the
    /// changes of the device block table the worker hands it, with [`PageWork::Copies`] for
    /// the copies that start with a batch, and for every batch it serves with
    /// [`PageWork::BeforeReplay`] after the slot state's writes and before the replay, with
    /// [`PageWork::AfterReplay`] after the replay and before its stream event is waited for,
    /// and with [`PageWork::Ended`] once the batch ended or failed; between batches, with
    /// [`PageWork::Holds`] for the regions the worker asks the runner whether it maps, and
    /// with [`PageWork::Ended`] when the worker waits for its copies. The hook issues on the
    /// stream the runner's graphs replay on, or on the rank's copy stream, and waits for
    /// nothing but at the end: the stream's order places the table writes, unit resets and
    /// state restores ahead of the replay that reads the pooled tables, the planes and the
    /// live state, and the state keeps behind it. A step's completion is returned after the
    /// end, so it is behind every copy of its batch.
    ///
    /// A hook's error fails the batch.
    pub fn set_page_work(&mut self, hook: PageWorkHook) {
        self.page_work = Some(hook);
    }

    /// The bytes of device memory the feature buffer of a runner of `model` under `config`
    /// takes ([`CausalLmRunner::set_features`]): zero for a model without a vision tower.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket.
    pub(crate) fn image_bytes(config: &CausalLmConfig, model: &dyn TextModel) -> Result<u64> {
        Features::bytes(model, stated(config, model)?.1.tokens())
    }

    /// Lends the runner a feature buffer, [`CausalLmRunner::image_bytes`] of device memory it
    /// allocates on its stream's device, through which its prefills place image rows: the
    /// encoder output of an image block is the block of class `class` that the rank's tables
    /// hold at the block's position, in `memory`, the rank's media memory. The runner also
    /// issues through `memory` the copies of encoder outputs that start with its batches.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a model without a vision tower; the driver's refusal of the
    /// allocation.
    pub(crate) fn set_features(&mut self, memory: Arc<BufferMemory>, class: u8) -> Result<()> {
        let features = Features::new(
            &*self.model,
            self.layout.tokens(),
            self.stream.as_ref(),
            memory,
            class,
        )?;

        self.features = Some(features);
        Ok(())
    }

    /// The copies that place `batch`'s image rows through `tables` (`features`); none for a
    /// runner lent no feature buffer.
    fn image_copies(&self, batch: &Batch, tables: &BlockTables) -> Result<Vec<DeviceCopy>> {
        match &self.features {
            Some(features) => features.copies(batch, tables),
            None => Ok(Vec::new()),
        }
    }

    /// Issues `copies`, those of [`CausalLmRunner::image_copies`], and marks `batch`'s image
    /// rows, on the runner's stream ahead of the replay.
    fn place_images(&mut self, batch: &Batch, copies: &[DeviceCopy]) -> Result<()> {
        match self.features.as_mut() {
            Some(features) => features.issue(batch, copies, self.stream.as_ref()),
            None => Ok(()),
        }
    }

    /// Makes the runner `member` of an expert union: its buckets become the capacity classes
    /// of the config's buckets and the join bucket (`experts::classes`), its forwards are lent
    /// the member's exchange, and its steps are agreed with the other members.
    ///
    /// It is called before the profile run and the captures, which state the buckets, and for
    /// graphs that hold a device after the members created their transport
    /// (`Member::create_transport`).
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a model that splits no experts, a member whose largest step is
    /// smaller than the runner's largest bucket, or a runner that holds captured graphs;
    /// [`Error::Unsupported`] for a runner that speculates.
    pub(crate) fn join_union(&mut self, member: Member) -> Result<()> {
        if self.speculation.is_some() {
            return Err(Error::Unsupported(
                "speculative decoding of a model whose experts are split across ranks".into(),
            ));
        }
        let (_, tokens) = self.config.buckets.bounds();
        if self.model.experts().is_none() || member.capacity() < tokens {
            return Err(Error::Invalid(format!(
                "an expert union whose members send at most {} rows a step, for a model that \
                 routes {} and buckets of up to {tokens} tokens",
                member.capacity(),
                match self.model.experts() {
                    Some(_) => "experts",
                    None => "no experts",
                }
            )));
        }
        if !self.graphs.buckets().is_empty() {
            return Err(Error::Invalid(
                "a runner joins an expert union before its buckets are captured".into(),
            ));
        }

        self.buckets = experts::classes(&self.config.buckets, member.capacity(), member.beside());
        self.graphs
            .set_exchange(member.transport(), Some(member.meet()));
        self.union = Some(member);
        Ok(())
    }

    /// Lends the scratch page set paired forwards compute over: one `KvCache` per attention
    /// group, in the order and with the layers of the runner's cache, over the config's
    /// `scratch_pages` pages, whose `block_tables` is the set's own page table: `[pages, width]`
    /// U32, row `i` naming the pages `i, i + 1, …` of the set's planes and the set's last page
    /// where they run past its end, `width` the set's pages or, when fewer, the width of the
    /// group's pooled table, which bounds the pages any served row reads. A row of a paired
    /// forward whose pages begin at page `i` gathers row `i`.
    ///
    /// The caller keeps the set's memory as it keeps the cache's. Before each paired forward
    /// the pages the batch's rows read are copied, on the runner's stream, into the pages the
    /// set hands each row (page `k` of row `r`'s request into the `k`-th page row `r` takes), so
    /// the forward attends over the context the served batch attended over: the runner asks the
    /// page-work hook for it ([`PageWork::BeforeShadow`]).
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config that states no scratch pages, or a set that is not the
    /// cache's groups and layers over the config's page size and scratch pages.
    pub(crate) fn set_scratch(&mut self, scratch: Vec<KvCache>) -> Result<()> {
        let pages = self.config.scratch_pages as usize;
        let same_groups = scratch.len() == self.cache.len()
            && scratch
                .iter()
                .zip(&self.cache)
                .all(|(lent, group)| lent.layers.len() == group.layers.len());

        let tabled = scratch.iter().zip(&self.cache).all(|(lent, group)| {
            let table = &lent.block_tables;
            let width = pages.min(group.block_tables.dims().get(1).copied().unwrap_or(0));

            table.dims() == [pages, width]
                && table.dtype() == DType::U32
                && lent.page_size == self.config.page_size
        });
        if pages == 0 || !same_groups || !tabled {
            return Err(Error::Invalid(format!(
                "a scratch page set of {} groups for a cache of {}, whose tables are not u32 \
                 rows of the {pages} pages of the set",
                scratch.len(),
                self.cache.len()
            )));
        }

        self.scratch = scratch;
        Ok(())
    }

    /// The head of input tensor `column`, as `shape`.
    fn column(&self, column: Column, shape: impl Into<Shape>) -> Result<Tensor> {
        self.graphs
            .input(self.layout.offset(column), shape, column.dtype())
    }

    /// Checks `batch` against what the runner serves and returns the smallest captured bucket
    /// that holds it.
    fn choose(&self, batch: &Batch) -> Result<Bucket> {
        let columns = &batch.columns;
        let shape = batch.shape;

        if !batch.media.is_empty() || shape.variant.media() {
            return Err(Error::Invalid(format!(
                "batch {} is a media computation, which a causal runner serves none of",
                batch.id.0
            )));
        }
        batch.check()?;

        // The page size and the key bound are part of every captured launch.
        if columns.page_size != self.config.page_size || columns.max_kv_len > self.config.max_kv_len
        {
            return Err(Error::Invalid(format!(
                "batch {} states pages of {} tokens under a key bound of {}, and the runner's \
                 graphs are captured for pages of {} under a bound of {}",
                batch.id.0,
                columns.page_size,
                columns.max_kv_len,
                self.config.page_size,
                self.config.max_kv_len
            )));
        }

        let verify = match &self.speculation {
            Some(speculation) => {
                // A refill evaluates positions behind its slot's own, and leaves the slot's
                // state, the model's and the drafter's, as it was.
                let refills = columns.drafts.iter().any(|work| work.refill.is_some());
                if refills && (self.state.is_some() || speculation.state.is_some()) {
                    return Err(Error::Unsupported(format!(
                        "batch {} refills the draft cache of a model whose layers or drafter \
                         keep a recurrent state, which needs a scratch state for each refill row",
                        batch.id.0
                    )));
                }
                speculation::verifies(columns, speculation.drafts)?
            }
            None => {
                let verifies = columns
                    .rows
                    .calls
                    .iter()
                    .any(|call| call.kind == Phase::Verify);
                if verifies || !columns.drafts.is_empty() {
                    return Err(Error::Unsupported(format!(
                        "batch {} verifies or drafts tokens, and this runner's config states no \
                         drafts",
                        batch.id.0
                    )));
                }
                false
            }
        };

        refuse_drawn_rows(shape.variant, &columns.parameters)?;

        // Image rows are placed by the prefill of a runner lent a feature buffer, and work the
        // runner states no bucket for is refused by its name, never served as text.
        let prefill = !verify && !shape.variant.is_decode();
        if !columns.vision.is_empty() && (self.features.is_none() || !prefill) {
            return Err(Error::Unsupported(format!(
                "batch {} carries image rows, which this runner places only in a prefill and \
                 only when it is lent a feature buffer",
                batch.id.0
            )));
        }

        // A verification is served from the verification bucket of its selection, whether its
        // batch states a decode's shape or a verification's.
        let shape = if verify {
            let stated = matches!(
                shape.variant,
                Variant::DecodeGreedy | Variant::DecodeSampled
            ) || verifies(shape.variant);
            if !stated {
                return Err(Error::Invalid(format!(
                    "batch {} verifies drafts and states the shape {shape}",
                    batch.id.0
                )));
            }

            BatchShape::prefill(
                columns.calls() as u32,
                columns.tokens() as u32,
                match shape.variant.greedy() {
                    true => Variant::VerifyGreedy,
                    false => Variant::VerifySampled,
                },
            )
        } else {
            shape
        };

        self.graphs
            .select(|bucket| bucket.holds(&shape))
            .ok_or_else(|| {
                let stated = self.config.buckets.with_drafts(self.config.drafts);

                match stated.select(&shape) {
                    Some(bucket) if verifies(bucket.variant) => {
                        Error::State("a verification bucket that holds the batch is not captured")
                    }
                    Some(bucket) => Error::State(match bucket.variant.is_decode() {
                        true => "a decode bucket that holds the batch is not captured",
                        false => "a prefill bucket that holds the batch is not captured",
                    }),
                    None => {
                        let (rows, tokens) = stated.bounds();
                        Error::Invalid(format!(
                            "no bucket holds {shape}: the runner's buckets top at {rows} rows \
                             and {tokens} tokens, and a batch past that bound is admitted as \
                             smaller ones"
                        ))
                    }
                }
            })
    }

    /// The cache slot each token of `batch` writes through `tables`, and the calls refused for
    /// a token whose entry may not be written (`columns::slot_mapping`). A call writes the
    /// attention group it names: the model's attention groups come first among the cache
    /// groups, in the order the runner is lent them.
    fn slot_mapping(
        &self,
        batch: &Batch,
        tables: &BlockTables,
    ) -> Result<(Vec<u32>, Vec<Refusal>)> {
        columns::slot_mapping(
            &batch.columns.rows,
            tables,
            self.config.page_size,
            self.cache.len() as u32,
        )
        .map_err(|error| Error::Invalid(format!("batch {}: {error}", batch.id.0)))
    }

    /// The draft pages of each row of `batch`, in row order, from its slot's entries of the
    /// draft group's class of `tables` ([`DraftPool::table`]); a row on no slot states none.
    /// Empty for a runner that speculates nothing.
    fn draft_tables(&self, batch: &Batch, tables: &BlockTables) -> Result<Vec<DraftTable>> {
        let Some(speculation) = &self.speculation else {
            return Ok(Vec::new());
        };

        let mut drafted = Vec::with_capacity(batch.columns.slots.len());
        for held in &batch.columns.slots {
            match u32::try_from(*held).ok().filter(|slot| *slot > 0) {
                Some(slot) => drafted.push(speculation.pool.table(tables, slot)?),
                None => drafted.push((0, Vec::new(), Vec::new())),
            }
        }
        Ok(drafted)
    }

    /// Prepares the inputs of a step of `batch` in `bucket` over `tables`: its padded columns,
    /// with `mapping` the cache slot each token writes and, for a paired forward, `scratch` the
    /// scratch pages each row takes (`columns::fill`), and the host values the bucket's forward
    /// computes.
    fn fill(
        &mut self,
        batch: &Batch,
        bucket: &Bucket,
        tables: &BlockTables,
        mapping: &[u32],
        scratch: Option<&[Range<u32>]>,
    ) -> Result<()> {
        self.stage(batch, bucket, tables, mapping, scratch)?;
        self.prepare_staged(bucket)
    }

    /// Pads `batch`'s columns to `bucket` in the runner's own bytes of the input block, with
    /// `mapping` the cache slot each of its tokens writes and `scratch` the scratch pages each
    /// row of a paired forward takes, and returns the ban masks of the step's sampling rows
    /// (`sampler::adjustments`), empty for a step that bans nothing. The columns of a bucket
    /// are those of every capacity class of it. A runner that speculates also pads the columns
    /// of the drafter's forward behind the bucket, which lie in the same block, with each row's
    /// draft pages from `tables`.
    fn stage(
        &mut self,
        batch: &Batch,
        bucket: &Bucket,
        tables: &BlockTables,
        mapping: &[u32],
        scratch: Option<&[Range<u32>]>,
    ) -> Result<Vec<u32>> {
        self.staged.resize(self.layout.bytes() as usize, 0);
        let vocabulary = self.model.vocabulary();
        let verify = verifies(bucket.variant);

        let masks = if verify {
            // Each token of a verification is a sampling row.
            let rows = speculation::token_rows(&batch.columns);

            speculation::fill_verification(
                &self.layout,
                self.config.slots,
                vocabulary,
                &batch.columns,
                &rows,
                bucket,
                mapping,
                &mut self.staged,
            )?;
            sampler::adjustments(
                &rows.biased,
                &rows.banned,
                bucket.tokens as usize,
                vocabulary,
            )?
            .1
        } else {
            columns::fill(
                &self.layout,
                self.config.slots,
                vocabulary,
                &batch.columns,
                bucket,
                mapping,
                scratch,
                &mut self.staged,
            )?;
            let columns = &batch.columns;
            sampler::adjustments(
                &columns.biased,
                &columns.banned,
                bucket.rows as usize,
                vocabulary,
            )?
            .1
        };

        let drafted = self.draft_tables(batch, tables)?;
        if let Some(speculation) = &self.speculation {
            speculation::fill_drafter(
                &self.layout,
                &speculation.pool,
                self.config.slots,
                vocabulary,
                &batch.columns,
                &drafted,
                bucket,
                &mut self.staged,
            )?;
        }
        Ok(masks)
    }

    /// Prepares the step of `bucket` over the staged columns: they are written into the input
    /// block, with the host values the bucket's forward computes.
    fn prepare_staged(&self, bucket: &Bucket) -> Result<()> {
        // The meta forward that writes the host values computes over the serving cache: a
        // forward's host values follow from the bucket alone.
        self.paired.set(false);

        self.graphs.prepare_inputs(self, bucket, |block| {
            block[..self.staged.len()].copy_from_slice(&self.staged);
            Ok(())
        })
    }

    /// One step of a member with no batch of its own: the expert layers of the step `batch`
    /// asks it to join, or nothing when the member already took part in that step.
    fn join_step(&mut self, batch: &Batch, started: Instant) -> Result<BatchOutput> {
        let member = self.union.as_mut().ok_or_else(|| {
            Error::Invalid(format!(
                "batch {} asks a runner outside an expert union to join a step",
                batch.id.0
            ))
        })?;
        let Some(capacity) = member.join(batch)? else {
            return Ok(BatchOutput::default());
        };

        let bucket = at_capacity(&self.graphs, &Bucket::expert_join(), capacity)?;
        if let Some(stream) = &self.stream {
            member.write(capacity, stream)?;
        }

        self.paired.set(false);
        self.graphs.prepare_inputs(&*self, &bucket, |_| Ok(()))?;

        let launched = Instant::now();
        self.graphs.replay(&bucket)?;
        self.graphs.wait()?;

        Ok(BatchOutput {
            calls: Vec::new(),
            timing: TimingCounters {
                queued_us: 0,
                device_us: launched.elapsed().as_micros() as u64,
                copy_us: 0,
                host_us: launched.duration_since(started).as_micros() as u64,
            },
            forward_stats: None,
        })
    }

    /// Serves one step of `batch` over `tables` and waits for it: everything of
    /// [`Runner::execute`] but waiting for the copies the batch took.
    fn serve(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        let started = Instant::now();

        // A join names no call: it is this member's part in a step of the union.
        if batch.shape.variant == Variant::ExpertJoin {
            return self.join_step(batch, started);
        }

        // Any other batch that names no call has no rows: it carries the table's changes and the
        // copies alone, which reached the runner before it, and computes nothing. Its output
        // names no call, and `execute` ends the copies with it.
        if batch.requests.is_empty() && batch.columns.calls() == 0 {
            return Ok(BatchOutput::default());
        }

        let flight = self.launch(batch, tables, started)?;
        self.land(batch, flight)
    }

    /// Whether the step of `batch` may land after the runner launched the next one
    /// ([`Runner::start`]): a batch with rows, on a runner whose graphs hold a device, that
    /// neither speculates (a verification never pipelines, and the drafter's graph reads what
    /// the step kept), nor is a member of an expert union (whose members agree one step at a
    /// time), nor was lent a feature buffer (whose image rows each batch places for its own
    /// step). Every other batch is served by [`Runner::execute`] alone.
    fn overlaps(&self, batch: &Batch) -> bool {
        self.readback.is_some()
            && self.speculation.is_none()
            && self.union.is_none()
            && self.features.is_none()
            && batch.shape.variant != Variant::ExpertJoin
            && !(batch.requests.is_empty() && batch.columns.calls() == 0)
    }

    /// Issues one step of `batch`, a batch with rows, over `tables`, which the runner took at
    /// `started`: steps 1 to 7 of the module documentation, then the copy of the step's
    /// outputs to the host behind them. Nothing of the step is waited for; the returned flight
    /// is what [`CausalLmRunner::land`] reads once the device completed it.
    fn launch(&mut self, batch: &Batch, tables: &BlockTables, started: Instant) -> Result<Flight> {
        if self.union.as_ref().is_some_and(Member::experts_only) {
            return Err(Error::Invalid(format!(
                "batch {} reached a member of an expert union that holds experts alone and \
                 serves no request",
                batch.id.0
            )));
        }

        if !batch.columns.state_copies.is_empty() && self.page_work.is_none() {
            return Err(Error::Invalid(format!(
                "batch {} carries state copies, and the runner was given nothing that issues \
                 them",
                batch.id.0
            )));
        }
        // Everything that can refuse the batch comes first: the choice of a bucket, the cache
        // slots its tokens write and the padding of the columns.
        let base = self.choose(batch)?;

        // A runner that speculates issues the drafter's graph behind the batch's own.
        let drafter = match &self.speculation {
            Some(_) => {
                let drafter = Bucket::draft(base.rows, base.tokens);
                if self.graphs.select(|held| *held == drafter).is_none() {
                    return Err(Error::State(
                        "the drafter's bucket behind the batch's is not captured",
                    ));
                }
                Some(drafter)
            }
            None => None,
        };

        // A refused call writes nothing and fails alone; the others are served.
        let (mapping, refused) = self.slot_mapping(batch, tables)?;
        let masks = self.stage(batch, &base, tables, &mapping, None)?;
        let images = self.image_copies(batch, tables)?;

        // The slot state's writes, ahead of the replay that reads them. A step launched while
        // an earlier one's writes may still be read waits for the stream here (`Slots::begin`).
        self.slots.begin()?;
        self.slots.start(
            &self.sampler,
            &batch.columns,
            &batch.new_slots,
            base.variant.is_decode(),
        )?;
        self.slots.restore(&batch.columns.state_copies)?;
        if !masks.is_empty() {
            self.slots.ban(&masks)?;
        }
        self.place_images(batch, &images)?;

        if let Some(hook) = self.page_work.as_mut() {
            hook(PageWork::BeforeReplay(batch, tables))?;
        }

        // A member of an expert union agrees the step with the other members, which blocks
        // until each has entered it, and runs the capacity class that holds it.
        let bucket = match self.union.as_mut() {
            Some(member) => {
                let tokens = batch.columns.tokens() as u32;
                let capacity = member.enter(tokens, batch)?;

                let bucket = at_capacity(&self.graphs, &base, capacity)?;
                if let Some(stream) = &self.stream {
                    member.write(capacity, stream)?;
                }
                bucket
            }
            None => base,
        };
        self.prepare_staged(&bucket)?;

        let launched = Instant::now();
        let outputs = self.graphs.replay(&bucket)?;
        self.launched += 1;

        // The drafter's columns lie in the block the batch's graph was issued with, and its
        // graph reads what that graph kept: the stream's order is their synchronization.
        let proposals = match &drafter {
            Some(drafter) => {
                self.prepare_staged(drafter)?;
                self.graphs.replay(drafter)?
            }
            None => Vec::new(),
        };

        if let Some(hook) = self.page_work.as_mut() {
            hook(PageWork::AfterReplay)?;
        }

        // The outputs, then the drafter's proposals, are copied to the host behind everything
        // of the step, the state keeps included, before anything of a later step is issued:
        // the next replay of the bucket writes the same memory again.
        let (Some(readback), Some(stream)) = (self.readback.as_mut(), self.stream.as_ref()) else {
            return Err(Error::State(
                "the runner's graphs hold no device to read a step from",
            ));
        };
        let read: Vec<Tensor> = outputs
            .into_iter()
            .chain(proposals.iter().cloned())
            .collect();
        let pending = readback.issue(&read, stream)?;

        Ok(Flight {
            batch: batch.id,
            number: self.launched,
            bucket,
            refused,
            pending,
            proposals: proposals.len(),
            started,
            launched,
        })
    }

    /// Lands `flight`, the step the runner launched for `batch`: waits until the device completed
    /// it and its outputs reached the host, and assembles one `CallOutput` per call, in the
    /// batch's order, a refused call's as a failure (step 8 of the module documentation, but for
    /// the copies the batch took).
    fn land(&mut self, batch: &Batch, flight: Flight) -> Result<BatchOutput> {
        let Flight {
            number,
            bucket,
            refused,
            pending,
            proposals,
            started,
            launched,
            ..
        } = flight;

        let mut outputs = self
            .readback
            .as_mut()
            .ok_or(Error::State(
                "the runner's graphs hold no device to read a step from",
            ))?
            .read(pending)?;
        let landed = Instant::now();

        // The stream has passed every slot write since `Slots::begin` only when no later step
        // began writing again.
        if number == self.launched {
            self.slots.settled();
        }

        let proposals = outputs.split_off(outputs.len().saturating_sub(proposals));
        let vocabulary = self.model.vocabulary();
        let candidates = self.config.candidates;

        let mut calls = if verifies(bucket.variant) {
            output::verification_output(&batch.columns, &bucket, vocabulary, candidates, &outputs)?
        } else {
            output::calls_output(&batch.columns, &bucket, vocabulary, candidates, &outputs)?
        };
        if self.speculation.is_some() {
            speculation::propose(batch, vocabulary, &proposals, &mut calls)?;
        }
        output::refuse(&mut calls, &refused);

        let host = launched.duration_since(started) + landed.elapsed();
        Ok(BatchOutput {
            calls,
            timing: TimingCounters {
                // The worker states the wait it saw before it called the runner.
                queued_us: 0,
                device_us: landed.duration_since(launched).as_micros() as u64,
                // Nothing is transferred for a completion: the scheduler reads it here.
                copy_us: 0,
                host_us: host.as_micros() as u64,
            },
            forward_stats: None,
        })
    }

    /// The attention groups of a forward: each group's page-table rows gathered by
    /// `table_rows`, from the pooled table of the serving cache or, in a paired forward, from
    /// the scratch set's table.
    fn gathered_cache(&self, table_rows: &Tensor) -> Result<Vec<KvCache>> {
        let groups = if self.paired.get() {
            &self.scratch
        } else {
            &self.cache
        };

        groups
            .iter()
            .map(|group| {
                Ok(KvCache {
                    block_tables: group.block_tables.index_select(table_rows, 0)?,
                    ..group.clone()
                })
            })
            .collect()
    }

    /// The verify slots of a forward's `rows` state rows (`nn::VerifySlots`): their slots, an
    /// input column, and their slots' accepted counts, gathered by `slot_rows`, each row's row
    /// of the slot state (the padding row for a row on no slot). `None` for a runner whose
    /// forwards lend none.
    fn verify_slots(&self, slot_rows: &Tensor, rows: usize) -> Result<Option<VerifySlots>> {
        let Some(accepted) = self.slots.accepted() else {
            return Ok(None);
        };
        let owned = self.layout.drafts() + 1;

        Ok(Some(VerifySlots {
            state_indices: self.column(Column::StateIndices, (rows, owned))?,
            num_accepted: accepted.index_select(slot_rows, 0)?,
        }))
    }

    /// Scores every token of a scoring bucket against its target: `[logprobs, ranks]`, then
    /// the candidates' `[values, ids]` when the sampler reports any, each over the bucket's
    /// tokens.
    ///
    /// Every token is projected, in chunks of [`SCORE_ROWS`], so the working set does not grow
    /// with the bucket: a chunk is a fixed sequence of launches and the graph holds one per
    /// chunk. A token that names no target is projected all the same and reports rank zero.
    fn score_prompt(&self, bucket: &Bucket, hidden: &Tensor) -> Result<Vec<Tensor>> {
        let tokens = bucket.tokens as usize;
        let targets = self.column(Column::Targets, tokens)?;

        let mut chunks: Vec<Vec<Tensor>> = Vec::new();
        let mut start = 0;
        while start < tokens {
            let count = SCORE_ROWS.min(tokens - start);

            // The chunk's logits are returned to the allocator before the next chunk's are
            // taken.
            let scores = {
                let logits = self
                    .model
                    .compute_logits(&hidden.narrow(0, start, count)?)?;
                self.sampler
                    .score(&logits, &targets.narrow(0, start, count)?)?
            };

            for (kind, part) in scores.into_iter().enumerate() {
                match chunks.get_mut(kind) {
                    Some(parts) => parts.push(part),
                    None => chunks.push(vec![part]),
                }
            }
            start += count;
        }

        chunks
            .iter()
            .map(|parts| Ok(Tensor::cat(parts, 0)?))
            .collect()
    }
}

impl Runner for CausalLmRunner {
    fn phases(&self) -> &[Phase] {
        &self.config.phases
    }

    /// A warm-up's forwards are run in step by every member of the runner's expert union: each
    /// exchange launch of such a forward first meets the other members on the host. A runner
    /// outside a union ignores it.
    fn set_warming(&self, warming: bool) {
        if let Some(member) = &self.union {
            member.set_warming(warming);
        }
    }

    fn buckets(&self) -> Vec<Bucket> {
        self.buckets.capture_order().copied().collect()
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        if !self.buckets.contains(bucket) {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is none the runner states"
            )));
        }

        // A runner that speculates states the verification buckets and the drafter's.
        if verifies(bucket.variant) {
            return self.forward_verification(bucket);
        }
        if bucket.variant == Variant::Draft {
            return self.forward_drafter(bucket);
        }

        // A member of an expert union is lent the exchange of the bucket's capacity class. Its
        // join, and every bucket of a member that holds experts alone, is the expert layers
        // of a step with no tokens of the member's own, and returns no output.
        let exchange = match &self.union {
            Some(member) => Some(member.exchange(bucket.capacity)?),
            None => None,
        };
        let joins = bucket.variant == Variant::ExpertJoin
            || self.union.as_ref().is_some_and(Member::experts_only);
        if joins {
            let exchange = exchange.ok_or_else(|| {
                Error::Invariant(format!(
                    "bucket {bucket} joins an expert step in a runner outside an expert union"
                ))
            })?;

            self.model.join_expert_step(&exchange)?;
            return Ok(Vec::new());
        }
        let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
        let decode = bucket.variant.is_decode();

        // ---- What the rows read through their slots ----

        let slots = self.column(Column::Slots, rows)?;
        let slot_rows = self.column(Column::SlotRows, rows)?;
        let cache = self.gathered_cache(&self.column(Column::TableRows, rows)?)?;

        // A decode row's input token is the one the step before it committed into its slot.
        let ids = if decode {
            self.slots
                .future()
                .narrow(1, 0, 1)?
                .index_select(&slot_rows, 0)?
                .reshape(rows)?
        } else {
            self.column(Column::InputIds, tokens)?
        };

        // ---- The model ----

        // A model that reads token ids reads the two ids before each row, which its slot keeps.
        let contexts = match self.slots.context() {
            Some(table) => Some(table.index_select(&slot_rows, 0)?),
            None => None,
        };

        // A prefill of a runner lent a feature buffer takes its image rows from it.
        let images = self.features.as_ref().filter(|_| !decode);
        let positions = self.column(Column::Positions, tokens)?;

        let batch = TokenBatch {
            token_ids: contexts.as_ref().map(|_| ids.clone()),
            token_history: contexts.clone(),
            rope_positions: match images {
                Some(features) => Some(features.rope(&positions, tokens)?),
                None => None,
            },
            positions,
            slot_mapping: self.column(Column::SlotMapping, tokens)?,
            seq_lens: self.column(Column::SeqLens, rows)?,
            query_offsets: self.column(Column::QueryOffsets, rows + 1)?,
            logits_rows: self.column(Column::LogitsRows, rows)?,
            phase: if decode {
                Phase::Decode
            } else {
                Phase::Prefill
            },
            page_size: self.config.page_size,
            num_tokens: bucket.tokens,
            max_kv_len: self.config.max_kv_len,
            draft_lens: None,
            live_tokens: Some(self.column(Column::LiveTokens, 1)?),
        };
        batch.check()?;

        // A batch's state rows are its attention rows.
        let state = match &self.state {
            Some((live, checkpoint)) => Some(RecurrentState {
                verify: self.verify_slots(&slot_rows, rows)?,
                query_offsets: batch.query_offsets.clone(),
                seq_lens: batch.seq_lens.clone(),
                slots: slots.clone(),
                checkpoints: self.column(Column::Checkpoints, rows)?,
                live: live.clone(),
                checkpoint: checkpoint.clone(),
                trail: None,
            }),
            None => None,
        };

        let hidden = {
            let embedded = self.model.embed_input_ids(&ids)?;
            let embedded = match images {
                Some(features) => features.merge(&embedded, tokens)?,
                None => embedded,
            };
            self.model
                .forward(&batch, &cache, state.as_ref(), exchange.as_ref(), &embedded)?
        };

        // ---- The selection of one token per row ----

        let logits = {
            let gathered = hidden.index_select(&batch.logits_rows, 0)?;
            self.model.compute_logits(&gathered)?
        };

        let parameters = self.column(Column::Parameters, (rows, PARAMETER_COLUMNS as usize))?;
        let adjustments = self.column(Column::Adjustments, (rows, 2))?;
        let bans = self.slots.bans().narrow(0, 0, rows)?;

        // A bucket whose rows are all greedy takes each row's largest shaped logit; any other
        // draws, and serves a greedy row as a draw over its one largest candidate. A greedy
        // selection reports no validity, and every task of it counts.
        let (chosen, valid) = if bucket.variant.greedy() {
            let chosen =
                self.sampler
                    .greedy(&logits, &slots, &parameters, &adjustments, &bans, None)?;
            (chosen, None)
        } else {
            let draws = self.column(Column::Draws, rows)?;
            let (chosen, valid) = self.sampler.forward(
                &logits,
                &slots,
                &parameters,
                &draws,
                &adjustments,
                &bans,
                None,
            )?;
            (chosen, Some(valid))
        };

        let scores = self.sampler.score(&logits, &chosen)?;
        drop(logits);

        let prompt = if bucket.variant.scores() {
            self.score_prompt(bucket, &hidden)?
        } else {
            Vec::new()
        };

        // ---- The commit, and each call's positions after it ----

        let table = self
            .column(Column::Commit, (COMMIT_FIELDS, self.layout.rows()))?
            .narrow(1, 0, rows)?;
        let counted = match &valid {
            Some(valid) => valid.clone(),
            None => self.column(Column::Valid, rows)?,
        };
        let [future, predicates, logical, sampling, verified] = self.slots.committed()?;

        self.sampler.commit_(
            &table,
            &chosen,
            &counted,
            &self.column(Column::Active, rows)?,
            &self.column(Column::Continuation, rows)?,
            &self.column(Column::Points, rows)?,
            &future,
            &predicates,
            &logical,
            &sampling,
            &verified,
        )?;

        let call_slots = self.column(Column::CallSlots, rows)?;
        let logical = self.slots.logical().index_select(&call_slots, 0)?;
        let sampling = self.slots.sampling().index_select(&call_slots, 0)?;

        // ---- What each row's slot continues from: its last ids, and one standing token ----

        // A paired forward writes nothing a serving request reads.
        if !self.paired.get() {
            if let (Some(table), Some(contexts)) = (self.slots.context(), &contexts) {
                let index = self.column(Column::NextContext, (rows, table.dim(1)?))?;
                write_rows(table, &slots, &history_ids(contexts, &ids, &index)?)?;
            }
            if let Some(accepted) = self.slots.accepted() {
                write_rows(accepted, &slots, &slots.ones_like()?)?;
            }
        }

        // The order `output` states, each entry with its place among the kept tensors.
        let mut scores = scores.into_iter();
        let mut outputs = vec![(Entry::Chosen, chosen)];

        outputs.extend(
            [Entry::Logprobs, Entry::Ranks]
                .into_iter()
                .zip(scores.by_ref()),
        );
        outputs.extend([(Entry::Logical, logical), (Entry::Sampling, sampling)]);
        outputs.extend([Entry::TopValues, Entry::TopIds].into_iter().zip(scores));
        outputs.extend(valid.map(|valid| (Entry::Valid, valid)));
        outputs.extend(
            [
                Entry::PromptLogprobs,
                Entry::PromptRanks,
                Entry::PromptValues,
                Entry::PromptIds,
            ]
            .into_iter()
            .zip(prompt),
        );

        // The drafter's forward follows this one and takes the allocator's memory again: a
        // runner that speculates keeps the outputs and the hidden rows in its own.
        match &self.speculation {
            Some(speculation) => {
                speculation.kept.keep(Entry::Hidden, &hidden)?;

                outputs
                    .into_iter()
                    .map(|(entry, output)| speculation.kept.keep(entry, &output))
                    .collect()
            }
            None => Ok(outputs.into_iter().map(|(_, output)| output).collect()),
        }
    }

    fn idle_inputs(&self, bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        if !self.buckets.contains(bucket) {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is none the runner states"
            )));
        }

        // A join reads no input tensor.
        if bucket.variant == Variant::ExpertJoin {
            return Ok(());
        }

        // An idle step is a serving step's forward: what a capture records.
        self.paired.set(false);

        let vocabulary = self.model.vocabulary();
        let (page_size, max_kv_len) = (self.config.page_size, self.config.max_kv_len);

        if verifies(bucket.variant) {
            let idle =
                speculation::idle_verification(bucket, self.config.drafts, page_size, max_kv_len);

            return speculation::fill_verification(
                &self.layout,
                self.config.slots,
                vocabulary,
                &idle,
                &speculation::token_rows(&idle),
                bucket,
                &[],
                columns,
            );
        }

        // The drafter's idle step follows an idle prefill of its rows and tokens: entries that
        // write no page of the draft cache, and rows that propose nothing.
        let target = match bucket.variant {
            Variant::Draft => Bucket::prefill(bucket.rows, bucket.tokens, Variant::PrefillWrite),
            _ => *bucket,
        };
        let idle = columns::idle(&target, page_size, max_kv_len);
        columns::fill(
            &self.layout,
            self.config.slots,
            vocabulary,
            &idle,
            &target,
            &[],
            None,
            columns,
        )?;

        match &self.speculation {
            Some(speculation) if bucket.variant == Variant::Draft => speculation::fill_drafter(
                &self.layout,
                &speculation.pool,
                self.config.slots,
                vocabulary,
                &idle,
                &[],
                bucket,
                columns,
            ),
            _ => Ok(()),
        }
    }

    /// Prepares the inputs of a step of `batch` over `tables`.
    ///
    /// # Errors
    ///
    /// As [`Runner::prepare_inputs`]: a call one of whose tokens names a cache entry that is
    /// missing or read-only refuses the batch here, since a preparation reports no call; a
    /// served step fails that call alone ([`Runner::execute`]).
    fn prepare_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        // Outside a step, a member of an expert union prepares the smallest capacity class of
        // the bucket: the class a step runs follows from its agreement (`execute`).
        let bucket = self.choose(batch)?;

        let (mapping, refused) = self.slot_mapping(batch, tables)?;
        if let Some((_, reason)) = refused.first() {
            return Err(Error::Invalid(format!("batch {}: {reason}", batch.id.0)));
        }
        let images = self.image_copies(batch, tables)?;

        self.fill(batch, &bucket, tables, &mapping, None)?;
        self.place_images(batch, &images)?;
        Ok(bucket)
    }

    /// Prepares the inputs of a paired forward of `batch`, a batch the runner served, and
    /// returns the bucket the worker passes to `Graphs::shadow`.
    ///
    /// The scratch page set hands each row the pages of the context it reads, in row order
    /// (`columns::scratch_spans`). The columns are the batch's own, padded as a serving step's,
    /// except that every token writes its row's scratch pages at its cache position, every row
    /// gathers the row of the set's page table that begins at its first page, and the commit
    /// table is inert (`columns`). No slot is started and none of the batch's own page work is
    /// issued: the forward reads the slots' state as it stands and writes none of it.
    ///
    /// A batch whose rows need more pages than the set holds prepares nothing: its graphs are
    /// told it has no paired forward (`Graphs::unpaired`), and each paired forward the worker
    /// then asks of the bucket reports its operator uncompared.
    ///
    /// # Errors
    ///
    /// As [`Runner::prepare_inputs`]; [`Error::Unsupported`] for a model that keeps a
    /// recurrent state, which has no scratch state to advance, for a member of an expert
    /// union, whose exchange launches wait for the other members', for a verification, and
    /// for a runner that was lent no scratch page set.
    fn prepare_shadow_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        if self.union.is_some() {
            return Err(Error::Unsupported(
                "a paired forward of a member of an expert union would wait on its device for \
                 exchange launches the other members do not make"
                    .into(),
            ));
        }
        if self.state.is_some() {
            return Err(Error::Unsupported(
                "a paired forward of a model that keeps a recurrent state has no scratch state \
                 to advance"
                    .into(),
            ));
        }
        if self.scratch.is_empty() {
            return Err(Error::Unsupported(
                "the runner was lent no scratch page set to compute a paired forward over".into(),
            ));
        }
        let bucket = self.choose(batch)?;
        if verifies(bucket.variant) {
            return Err(Error::Unsupported(
                "a paired forward of a verification: its token rows share one request's pages, \
                 and the scratch page set gives each row its own"
                    .into(),
            ));
        }
        if !batch.columns.vision.is_empty() {
            return Err(Error::Unsupported(
                "a paired forward of a prefill with image rows, whose feature buffer holds the \
                 images of the batch served last"
                    .into(),
            ));
        }

        let rows = &batch.columns.rows;
        let spans = columns::scratch_spans(rows, self.config.page_size)?;
        let needed = spans.last().map_or(0, |span| span.end);
        if needed > self.config.scratch_pages {
            self.paired.set(false);
            self.graphs.unpaired(
                &bucket,
                format!(
                    "batch {}: its {} rows' contexts take {needed} scratch pages of {} tokens and \
                     the scratch page set holds {}",
                    batch.id.0,
                    spans.len(),
                    self.config.page_size,
                    self.config.scratch_pages
                ),
            );
            return Ok(bucket);
        }

        let mapping = columns::scratch_column(
            &rows.cache_positions,
            &rows.query_offsets,
            bucket.tokens as usize,
            &spans,
            self.config.page_size,
        )?;
        self.fill(batch, &bucket, tables, &mapping, Some(&spans))?;
        // No row an earlier batch's image marked stays marked.
        self.place_images(batch, &[])?;

        // The rows' cached history reaches their scratch pages ahead of the forward.
        let first: Vec<u32> = spans.iter().map(|span| span.start).collect();
        if let Some(hook) = self.page_work.as_mut() {
            hook(PageWork::BeforeShadow(batch, &first))?;
        }
        self.paired.set(true);
        Ok(bucket)
    }

    /// Launches the step of `batch` without waiting for it, when the batch overlaps
    /// ([`CausalLmRunner::overlaps`]): steps 1 to 7 of the module documentation and the copy of
    /// its outputs to the host are issued, and `resume` is called from the runner's lander thread
    /// once the device has completed them. The worker meanwhile takes its next job, which may
    /// be the next step, issued on the stream behind this one; [`Runner::execute`] of the batch
    /// then lands it. A batch that does not overlap starts nothing.
    fn start(&mut self, batch: &Batch, tables: &BlockTables, resume: Waker) -> Result<bool> {
        if !self.overlaps(batch) {
            return Ok(false);
        }
        if self.lander.is_none() {
            self.lander = Some(Lander::spawn()?);
        }

        let flight = self.launch(batch, tables, Instant::now())?;
        let event = self
            .readback
            .as_ref()
            .map(|readback| readback.event(&flight.pending))
            .ok_or(Error::State(
                "the runner's graphs hold no device to read a step from",
            ))?;
        let watched = self
            .lander
            .as_ref()
            .ok_or(Error::State("a runner's lander thread has stopped"))
            .and_then(|lander| lander.watch(event, resume));

        // A step no thread waits for would never land: it is read here, which frees its turn,
        // and the batch fails.
        if let Err(error) = watched {
            let _ = self.land(batch, flight);
            return Err(error);
        }
        self.flights.push_back(flight);
        Ok(true)
    }

    /// One step of `batch` over `tables` (module documentation), or the landing of the step
    /// [`Runner::start`] launched for it. It returns once the device completed the step and the
    /// copies [`Runner::copy`] issued for it, whether the step was served or failed.
    fn execute(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput> {
        let launched = self
            .flights
            .iter()
            .position(|flight| flight.batch == batch.id)
            .and_then(|at| self.flights.remove(at));
        let served = match launched {
            Some(flight) => self.land(batch, flight),
            None => self.serve(batch, tables),
        };

        // The copies the batch took end with it, also when it failed.
        let ended = self.wait_copies();
        let output = served?;
        ended?;
        Ok(output)
    }

    /// Writes the device block table's `changes` into the pooled page tables and resets the
    /// units the buffer pool allocated, through the page-work hook ([`PageWork::Tables`]).
    ///
    /// # Errors
    ///
    /// What the hook refuses; [`Error::Invalid`] for changes given to a runner that was given
    /// nothing that writes its tables.
    fn write_tables(&mut self, changes: &[Change]) -> Result<()> {
        match self.page_work.as_mut() {
            Some(hook) => hook(PageWork::Tables(changes)),
            None if changes.is_empty() => Ok(()),
            None => Err(Error::Invalid(format!(
                "{} changes of the device block table reach a runner that was given nothing \
                 that writes its page tables",
                changes.len()
            ))),
        }
    }

    /// The cache's memory, asked through the page-work hook ([`PageWork::Holds`]), and for a
    /// runner lent a feature buffer the rank's media memory, where the encoder outputs lie.
    fn maps(&mut self, region: u8) -> bool {
        let media = self
            .features
            .as_ref()
            .is_some_and(|features| features.holds(region));

        let cached = Cell::new(false);
        if let Some(hook) = self.page_work.as_mut()
            && let Err(error) = hook(PageWork::Holds(region, &cached))
        {
            tracing::warn!(region, %error, "the page-work hook does not state the cache's regions");
        }
        media || cached.get()
    }

    /// Issues `copies` through the page-work hook ([`PageWork::Copies`]), and those of encoder
    /// outputs, for a runner lent a feature buffer, through the rank's media memory on its
    /// stream; the batch's end covers them ([`Runner::wait_copies`]).
    ///
    /// # Errors
    ///
    /// What the hook or the media memory refuses; [`Error::Unsupported`] for a copy with runs
    /// given to a runner that was given nothing that issues copies.
    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        let copies = match self.features.as_mut() {
            Some(features) => features.copy(copies, self.stream.as_ref())?,
            None => copies.to_vec(),
        };
        let copies = copies.as_slice();

        match self.page_work.as_mut() {
            Some(hook) => hook(PageWork::Copies(copies)),
            None => match copies.iter().find(|copy| !copy.runs.is_empty()) {
                Some(copy) => Err(Error::Unsupported(format!(
                    "the copies of request {} reach a runner that was given nothing that \
                     issues them",
                    copy.request.0
                ))),
                None => Ok(()),
            },
        }
    }

    /// Waits for the copies the cache issued ([`PageWork::Ended`]) and for those of encoder
    /// outputs, on the runner's stream.
    fn wait_copies(&mut self) -> Result<()> {
        let ended = match self.page_work.as_mut() {
            Some(hook) => hook(PageWork::Ended),
            None => Ok(()),
        };
        let settled = match self.features.as_mut() {
            Some(features) => features.settle(self.stream.as_ref()),
            None => Ok(()),
        };

        ended?;
        settled
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

/// The buckets a runner of `config` states outside an expert union, and the layout of its
/// input tensors.
fn stated(config: &CausalLmConfig, model: &dyn TextModel) -> Result<(Buckets, Layout)> {
    let buckets = config.buckets.with_drafts(config.drafts);
    let history = model.token_history().map_or(0, |history| history.width);

    let pages = match (config.drafts, &config.draft_pool) {
        (0, _) => 0,
        (_, Some(pool)) => pool.width(config.max_kv_len),
        (_, None) => {
            return Err(Error::Invalid(
                "a config that states drafts states the draft pool of its cache".into(),
            ));
        }
    };
    let layout = Layout::of(buckets.iter(), config.drafts, pages, history)?;

    Ok((buckets, layout))
}

/// Whether a bucket of `variant` is a verification.
const fn verifies(variant: Variant) -> bool {
    matches!(variant, Variant::VerifyGreedy | Variant::VerifySampled)
}

/// The smallest captured capacity class of `bucket` that holds a step of `capacity` rows a
/// member.
fn at_capacity(graphs: &Graphs, bucket: &Bucket, capacity: u32) -> Result<Bucket> {
    let stated = bucket.at_capacity(0);

    graphs
        .select(|held| held.at_capacity(0) == stated && held.capacity >= capacity)
        .ok_or_else(|| {
            Error::Invariant(format!(
                "no captured graph of bucket {bucket} holds an expert step of {capacity} rows a \
                 member"
            ))
        })
}

/// Refuses a greedy bucket a row that draws.
///
/// A bucket's variant fixes the selection of every row. A drawing bucket serves a greedy row
/// as well, as a draw over its one largest candidate (`sampler::parameters`). The converse is
/// not so: an argmax is not what a row that states a temperature asked for, so a greedy
/// variant holding one is refused by the row.
fn refuse_drawn_rows(variant: Variant, parameters: &[f32]) -> Result<()> {
    if !variant.greedy() {
        return Ok(());
    }

    let width = PARAMETER_COLUMNS as usize;
    for (row, stated) in parameters.chunks_exact(width).enumerate() {
        let temperature = stated[TEMPERATURE as usize];
        if temperature != 0.0 {
            return Err(Error::Invalid(format!(
                "row {row} of a {variant} batch states temperature {temperature}: its token is \
                 drawn from its distribution, and a greedy bucket takes each row's argmax"
            )));
        }
    }
    Ok(())
}

/// Refuses a cache that is not `model`'s layers over `config`'s page size and slots.
fn check_cache(model: &dyn TextModel, config: &CausalLmConfig, cache: &[KvCache]) -> Result<()> {
    let shape = model.cache();

    let layers: usize = cache.iter().map(|group| group.layers.len()).sum();
    if layers != shape.layers as usize {
        return Err(Error::Invalid(format!(
            "the cache lends {layers} layers in {} groups, and the model's layers store keys \
             and values in {}",
            cache.len(),
            shape.layers
        )));
    }

    for (index, group) in cache.iter().enumerate() {
        let table = &group.block_tables;
        let pooled = table.rank() == 2
            && table.dims()[0] == config.slots as usize + 1
            && table.dtype() == DType::U32;
        if !pooled || group.page_size != config.page_size {
            return Err(Error::Invalid(format!(
                "group {index} of the cache lends a page table {:?} over pages of {} tokens, \
                 and the runner gathers u32 rows of {} slots and the padding row over pages of \
                 {}",
                table.dims(),
                group.page_size,
                config.slots,
                config.page_size
            )));
        }
    }
    Ok(())
}

/// The slots a request owns of each stepwise field of `model`'s state under `config`
/// (`nn::family::StateField::stepwise`): `drafts + 1` when the config states drafts and a field
/// is stepwise, the forwards then lending verify slots (`nn::VerifySlots`), and `None` when no
/// field is stepwise.
///
/// # Errors
///
/// [`Error::Invalid`] for a stepwise field under a config that states no drafts: the model was
/// loaded for a deployment that speculates.
fn owned_slots(model: &dyn TextModel, config: &CausalLmConfig) -> Result<Option<u32>> {
    let runs = model.state_layers();
    let stepwise = layer_fields(&runs).flatten().any(|field| field.stepwise);

    match (stepwise, config.drafts) {
        (false, _) => Ok(None),
        (true, 0) => Err(Error::Invalid(
            "the model's state keeps a slot per verified token, and the runner's config states \
             no drafts to verify"
                .into(),
        )),
        (true, drafts) => Ok(Some(drafts + 1)),
    }
}

/// Refuses recurrent state buffers that are not `model`'s over `config`'s request slots: a
/// buffer per field of every layer of the model's runs (`TextModel::state_layers`), in order,
/// whose live buffers hold every row a forward addresses: a home row per slot, and `owned`
/// rows a slot of a stepwise field when the forwards lend verify slots ([`owned_slots`]).
fn check_state(
    model: &dyn TextModel,
    config: &CausalLmConfig,
    owned: Option<u32>,
    state: Option<&(StateBuffers, StateBuffers)>,
) -> Result<()> {
    let runs = model.state_layers();
    let layers: Vec<&[StateField]> = layer_fields(&runs).collect();

    let (live, checkpoint) = match (layers.is_empty(), state) {
        (true, None) => return Ok(()),
        (false, Some(state)) => state,
        (false, None) => {
            return Err(Error::Invalid(
                "the model keeps a recurrent state and the runner was lent none".into(),
            ));
        }
        (true, Some(_)) => {
            return Err(Error::Invalid(
                "the runner was lent a recurrent state its model does not keep".into(),
            ));
        }
    };

    // The verify slots name every row a request owns of a stepwise field; a checkpoint keeps
    // a home row per slot.
    let slots = config.slots as usize;
    check_buffers("live", live, &layers, |field| {
        if field.stepwise {
            slots * owned.unwrap_or(1) as usize
        } else {
            slots
        }
    })?;
    check_buffers("checkpoint", checkpoint, &layers, |_| slots)
}

/// Refuses the live buffers of a drafter's state that are not those of `runs`, the drafter's
/// (`nn::Drafter::state_layers`), over `slots` request slots: a home row per slot of each field of
/// every layer of the runs, and nothing for a drafter that keeps no state. A drafter's field
/// keeps one slot per request, so a field that keeps one per verified token is refused.
fn check_drafter_state(
    runs: &[StateShape],
    slots: u32,
    state: Option<&StateBuffers>,
) -> Result<()> {
    let layers: Vec<&[StateField]> = layer_fields(runs).collect();

    let live = match (layers.is_empty(), state) {
        (true, None) => return Ok(()),
        (false, Some(live)) => live,
        (false, None) => {
            return Err(Error::Invalid(
                "the model's drafter keeps a state and the runner was lent none".into(),
            ));
        }
        (true, Some(_)) => {
            return Err(Error::Invalid(
                "the runner was lent a drafter's state its model's drafter does not keep".into(),
            ));
        }
    };

    if let Some(field) = layers
        .iter()
        .copied()
        .flatten()
        .find(|field| field.stepwise)
    {
        return Err(Error::Invalid(format!(
            "the drafter's state field {} keeps a slot per verified token, and the drafter's \
             forwards are lent one slot per request",
            field.name
        )));
    }
    check_buffers("drafter's live", live, &layers, |_| slots as usize)
}

/// Refuses `buffers`, the `what` buffers of a state, unless they hold a buffer per field of
/// every layer of `layers`, in order, each of at least the rows `rows` states for its field: a
/// forward addresses a row of a buffer by an index no kernel bounds.
fn check_buffers(
    what: &str,
    buffers: &[Vec<Tensor>],
    layers: &[&[StateField]],
    rows: impl Fn(&StateField) -> usize,
) -> Result<()> {
    let layered = buffers.len() == layers.len()
        && buffers
            .iter()
            .zip(layers)
            .all(|(layer, fields)| layer.len() == fields.len());
    if !layered {
        return Err(Error::Invalid(format!(
            "the state lends {} {what} layers, and {} layers keep a state, each its run's fields",
            buffers.len(),
            layers.len()
        )));
    }

    for (buffers, fields) in buffers.iter().zip(layers) {
        for (buffer, field) in buffers.iter().zip(fields.iter()) {
            let rows = rows(field);

            if buffer.dims().first().is_none_or(|held| *held < rows) {
                return Err(Error::Invalid(format!(
                    "the {what} buffer of field {} is {:?}, and the runner's request slots \
                     address {rows} rows of it",
                    field.name,
                    buffer.dims()
                )));
            }
        }
    }
    Ok(())
}

/// The ids `index` names in the id history of a forward (`columns`, token ids): `contexts`,
/// `[rows, w]` U32, the `w` ids before each of its rows, then `ids`, `[tokens]` U32, its token
/// ids. The result has `index`'s shape.
///
/// # Errors
///
/// Tensors of other shapes or element types.
pub(crate) fn history_ids(contexts: &Tensor, ids: &Tensor, index: &Tensor) -> Result<Tensor> {
    let history = Tensor::cat(&[&contexts.flatten_all()?, ids], 0)?;

    Ok(history
        .index_select(&index.flatten_all()?, 0)?
        .reshape(index.shape())?)
}

/// Writes `values`, one entry along the leading axis for each entry of `slots`, `[rows]` U32,
/// into the rows of `table`, a per-slot buffer, that `slots` names; an entry of `slots` that
/// names no row (`nn::NO_SLOT`, a row on no request slot) writes nothing.
///
/// # Errors
///
/// Tensors of other shapes or element types.
pub(crate) fn write_rows(table: &Tensor, slots: &Tensor, values: &Tensor) -> Result<()> {
    let mut broadcast = vec![1; values.rank()];
    broadcast[0] = slots.elem_count();
    let index = slots.reshape(broadcast)?.broadcast_as(values.shape())?;

    Ok(table.scatter_set(&index, values, 0)?)
}

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
//! input ids                             → embed_input_ids         → hidden [tokens, hidden]
//! hidden, batch, cache, state           → forward                 → hidden [tokens, hidden]
//! hidden                                ─ index_select(logits rows) → compute_logits
//! logits [rows, vocabulary]             → Sampler::greedy | forward → tokens [rows]
//! logits, tokens                        → Sampler::score          → logprobs, ranks, candidates
//! hidden, targets (a scoring bucket)    → compute_logits, Sampler::score, in chunks
//! tokens, commit table                  → Sampler::commit_        → the slots' state
//! positions [slots + 1]                 ─ index_select(call slots) → each call's positions
//! ```
//!
//! A decode row's input token is the one the step before it committed into its slot, read on
//! the device, so a step can be issued behind the step it continues. Every bucket selects a
//! token per row and scores it: whether a row's request asked is not a value a captured graph
//! branches on, and the completion keeps what the batch asked for. The outputs and their order
//! are stated in [`output`].
//!
//! ## A step
//!
//! [`Runner::execute`] serves one batch:
//!
//! 1. the batch is checked and the smallest captured bucket that holds it is chosen;
//! 2. the batch's columns are padded to the bucket ([`columns`]);
//! 3. the slots the batch starts a request on are started, and its ban masks copied
//!    ([`slots`]);
//! 4. the batch's page work that precedes the forward is issued ([`PageWork::BeforeReplay`]);
//! 5. the padded columns are written into the input block, with the host values the forward
//!    computes, and the bucket's graph is replayed;
//! 6. the page work that follows the forward is issued ([`PageWork::AfterReplay`]);
//! 7. the stream event behind the replay is waited for, the outputs are read and one
//!    `CallOutput` per call is assembled.
//!
//! Steps 3 to 6 are issued on one stream in that order, which is the only synchronization
//! between them: the replay reads the slot state, the pooled page tables and the pages as the
//! copies before it left them.
//!
//! ## Page work
//!
//! The key/value cache and the recurrent state belong to the rank, which two runners of a
//! device may share. The runner takes their tensors at construction and never copies a page:
//! a batch's table installs, unit copies, page loads and offloads and state copies are issued
//! by the rank's cache through the hook given to [`CausalLmRunner::set_page_work`]. A runner
//! with no hook serves the batches that carry no page work and refuses one that carries some.
//! [`wiring`] builds the runner and the hook from the rank's cache and state modules.
//!
//! ## Paired forwards
//!
//! Before a generation switch the executor computes a served batch again as a paired forward
//! (`Graphs::shadow`), whose inputs [`CausalLmRunner::prepare_shadow_inputs`] writes. It is the
//! same forward over other pages: the rows write and read the scratch page set the cache lends
//! ([`CausalLmRunner::set_scratch`]) in its own numbering, the commit table is inert and no
//! slot is started, so nothing a serving request reads is written. A model that keeps a
//! recurrent state has no scratch state to advance, and its paired forwards are refused.
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
//! graph and the completion read of the first is kept in the runner's memory.

mod columns;
pub mod experts;
mod output;
mod slots;
pub mod speculation;
pub mod wiring;

#[cfg(test)]
mod tests;

use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;

use ipc::execution::{Batch, BatchOutput, RowKind, Shape as BatchShape, Variant};
use ipc::{PhaseId, TimingCounters};
use nn::family::TextModel;
use nn::{KvCache, PhaseKind, RecurrentState, TokenBatch};
use platform::Stream;
use proto::module::commit::COMMIT_FIELDS;
use proto::module::sampler::{PARAMETER_COLUMNS, TEMPERATURE};
use uniserve_tensor::{DType, Shape, Tensor};

use self::columns::{Column, Layout};
use self::experts::Member;
use self::slots::Slots;
use self::speculation::{DraftPages, DraftPool, Kept, Speculation};
use crate::buckets::{Bucket, Buckets};
use crate::graphs::Graphs;
use crate::runner::Runner;
use crate::sampler::{self, Sampler, SamplerConfig};
use crate::{Error, Result};

/// The tokens one chunk of a scoring bucket's prompt projection holds: a chunk's logits and
/// log-probabilities are `[SCORE_ROWS, vocabulary]` FP32 each (39 MB over 151936 tokens), and
/// the next chunk takes the memory the one before it returned.
const SCORE_ROWS: usize = 64;

/// What a deployment fixes of a causal runner. Immutable once the runner is built.
#[derive(Clone, Debug)]
pub struct CausalLmConfig {
    /// The phases the runner serves, as the scheduler addresses them in a batch.
    pub phases: Vec<PhaseId>,
    /// The buckets the runner captures (`Buckets::for_batch`, `Buckets::default_causal_lm`):
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
    /// The pages each row of a paired forward addresses in the scratch page set, and zero for
    /// a runner that computes no paired forward. A row whose tokens lie past them is refused.
    pub scratch_pages: u32,
    /// The drafts a verification holds at most (`speculative_config.num_draft_tokens`), and
    /// zero for a runner that speculates nothing. A runner that states some serves its decode
    /// steps as verifications and runs its model's drafter behind every step.
    pub drafts: u32,
    /// How the draft cache lies in the cache's unit pool ([`DraftPool::of`]): stated exactly
    /// when `drafts` is above zero.
    pub draft_pool: Option<DraftPool>,
}

/// Which part of a batch's page work the rank's cache issues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PageWork {
    /// What the forward reads: the batch's rows of the pooled page tables are installed, its
    /// new units reset and filled with the copies a reused prefix needs, its page loads are
    /// issued, and each restored slot's recurrent state is copied into the live buffers.
    BeforeReplay,
    /// What the forward left: each kept checkpoint of the recurrent state is copied to its
    /// page, and the batch's page offloads are waited for.
    AfterReplay,
    /// What a paired forward reads: the pages its rows have cached are copied into the rows'
    /// scratch pages.
    BeforeShadow,
}

/// The buffers of a recurrent state, as `nn::RecurrentState` lays them out: one entry per
/// recurrent layer, in the model's order of them, each the layer's fields as `[slots, …]`
/// tensors.
pub type StateBuffers = Vec<Vec<Tensor>>;

/// The seam through which the rank's cache issues a batch's page work on the runner's stream
/// ([`CausalLmRunner::set_page_work`]).
pub type PageWorkHook = Box<dyn FnMut(&Batch, PageWork) -> Result<()> + Send>;

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
    /// The bytes of the input block of the step being prepared: a batch's columns are padded
    /// here, where a refusal costs nothing, before anything of the step is issued.
    staged: Vec<u8>,
}

impl CausalLmRunner {
    /// The bytes of the runner's input tensors for `config`: the `inputs` its [`Graphs`] are
    /// made with.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket.
    pub fn input_bytes(config: &CausalLmConfig) -> Result<u64> {
        Ok(stated(config)?.1.bytes())
    }

    /// The bytes of device memory a runner that speculates keeps its target forwards' outputs
    /// in, for a model of `hidden`-wide rows of `dtype`: each bucket's chosen tokens and
    /// scores and the hidden rows of the largest, which the drafter's forward reads. Zero for
    /// a config that states no drafts.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket, or one that states drafts and no draft
    /// pool.
    pub fn kept_bytes(config: &CausalLmConfig, hidden: u32, dtype: DType) -> Result<u64> {
        if config.drafts == 0 {
            return Ok(0);
        }
        let layout = stated(config)?.1;

        Kept::bytes(
            layout.rows(),
            layout.tokens(),
            config.candidates,
            hidden,
            dtype,
        )
    }

    /// The bytes of device memory the request slots' state takes for `config` over a
    /// `vocabulary`-wide logit row: the count and bias tables, the commit state and the ban
    /// masks, which the runner allocates when its graphs hold a device.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket.
    pub fn slot_bytes(config: &CausalLmConfig, vocabulary: u32) -> Result<u64> {
        let layout = stated(config)?.1;

        Ok(Slots::bytes(
            config.slots,
            vocabulary,
            layout.rows() as u32,
            1,
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
    ///   `nn::RecurrentState`, for a model that keeps one.
    ///
    /// The caller keeps the memory of `cache` and `state` valid and at its addresses for as
    /// long as the runner exists, and writes it only on `stream`.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a bucket that is no causal forward, input tensors `graphs` do
    /// not hold, a stream without a device or the reverse, a cache that is not the model's
    /// layers over `config`'s page size and slots, a state that is not the model's, or a
    /// config that states drafts for a model that loaded no drafter or without the draft pool
    /// of its cache; the refusals of the slot state's allocation and of the sampler.
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
        let (buckets, layout) = stated(&config)?;

        if graphs.enter().is_ok() != stream.is_some() {
            return Err(Error::Invalid(
                "a causal runner takes the stream of its graphs, and none for graphs that hold \
                 no device"
                    .into(),
            ));
        }
        check_cache(&*model, &config, &cache)?;
        check_state(&*model, state.as_ref())?;

        let vocabulary = model.vocabulary();
        let slots = Slots::new(
            stream.clone(),
            config.slots,
            vocabulary,
            layout.rows() as u32,
            1,
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
                let kept = Kept::new(
                    stream.as_ref(),
                    layout.rows(),
                    layout.tokens(),
                    config.candidates,
                    model.hidden(),
                    model.dtype(),
                )?;

                Some(Speculation {
                    drafts: config.drafts,
                    pool,
                    kept,
                    verifier,
                    draft: None,
                    trail: None,
                    pages: None,
                })
            }
            _ => None,
        };

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
            staged: Vec::new(),
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
    /// and the trail of the recurrent state when its model keeps one.
    ///
    /// - `draft` is the draft group as the rank's cache lends it: the drafter's planes, pages
    ///   of the draft pool's page size, and the workspace. Its `block_tables` is not read: a
    ///   forward states each row's pages as an input column.
    /// - `trail` is the trail buffers of the recurrent state, in the layout of
    ///   `nn::RecurrentState::trail`, with a row for each token of the largest verification
    ///   bucket.
    ///
    /// The caller keeps their memory as it keeps the cache's and the state's.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a runner that speculates nothing, a draft cache that is not the
    /// drafter's layers over the draft pool's pages, or a trail that is not the model's
    /// recurrent state over the tokens of a verification.
    pub fn set_draft_cache(&mut self, draft: KvCache, trail: Option<StateBuffers>) -> Result<()> {
        let layers = self.model.drafter().map(|drafter| drafter.cache().layers);
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
        let fields = self.model.state().map(|shape| shape.fields.len());
        let trailed = match (&trail, fields) {
            (None, None) => true,
            (Some(trail), Some(fields)) => trail.iter().all(|layer| {
                layer.len() == fields
                    && layer
                        .iter()
                        .all(|field| field.dims().first().is_some_and(|rows| *rows >= tokens))
            }),
            _ => false,
        };
        if !trailed {
            return Err(Error::Invalid(format!(
                "a verification of a model that keeps a recurrent state keeps a trail of its \
                 {tokens} tokens for every field, and a model that keeps none is lent none"
            )));
        }

        speculation.draft = Some(draft);
        speculation.trail = trail;
        Ok(())
    }

    /// Sets what states a request slot's installed pages of the draft group
    /// ([`DraftPages`]): what each drafter row's page table is built from. A runner given
    /// none reads the tables its batches carry alone. A runner that speculates nothing
    /// ignores it.
    pub fn set_draft_pages(&mut self, pages: DraftPages) {
        if let Some(speculation) = self.speculation.as_mut() {
            speculation.pages = Some(pages);
        }
    }

    /// Sets what issues a batch's page work on the runner's stream.
    ///
    /// The runner calls `hook` on the executor thread, twice per batch that carries page work
    /// (block tables, cache units, page loads or offloads, state copies): with
    /// [`PageWork::BeforeReplay`] after the slot state's writes and before the replay, and
    /// with [`PageWork::AfterReplay`] after the replay and before its stream event is waited
    /// for. The hook issues on the stream the runner's graphs replay on and waits for
    /// nothing: the stream's order places the table installs, unit copies, page loads and
    /// state restores ahead of the replay that reads the pooled tables, the planes and the
    /// live state, and the state keeps and page offloads behind it. The step's outputs are
    /// read after both, so a completion is behind every copy of its batch.
    ///
    /// A hook's error fails the batch.
    pub fn set_page_work(&mut self, hook: PageWorkHook) {
        self.page_work = Some(hook);
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
    pub fn join_union(&mut self, member: Member) -> Result<()> {
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

        self.buckets = experts::classes(&self.config.buckets, member.capacity());
        self.graphs
            .set_exchange(member.transport(), Some(member.meet()));
        self.union = Some(member);
        Ok(())
    }

    /// States whether the forwards that follow are a warm-up's, which every member of the
    /// runner's expert union runs in step: each exchange launch of such a forward first meets
    /// the other members on the host. The executor sets it around the captures of a warm-up
    /// and clears it before the first batch. A runner outside a union ignores it.
    pub fn set_warming(&self, warming: bool) {
        if let Some(member) = &self.union {
            member.set_warming(warming);
        }
    }

    /// Lends the scratch page set paired forwards compute over: one `KvCache` per attention
    /// group, in the order and with the layers of the runner's cache, whose `block_tables` is
    /// the set's own page table, `[rows, scratch_pages]` U32 for at least the rows of the
    /// runner's largest bucket, row `r` naming the pages `r · scratch_pages ..
    /// (r + 1) · scratch_pages` of the set's planes.
    ///
    /// The caller keeps the set's memory as it keeps the cache's. Before each paired forward
    /// the pages the batch's rows read are copied, on the runner's stream, into the rows'
    /// scratch pages (page `k` of row `r`'s request into page `r · scratch_pages + k`), so the
    /// forward attends over the context the served batch attended over: the runner asks the
    /// page-work hook for it ([`PageWork::BeforeShadow`]).
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config that states no scratch pages, or a set that is not the
    /// cache's groups and layers over the config's page size and scratch pages.
    pub fn set_scratch(&mut self, scratch: Vec<KvCache>) -> Result<()> {
        let pages = self.config.scratch_pages as usize;
        let rows = self.scratch_rows() as usize;
        let same_groups = scratch.len() == self.cache.len()
            && scratch
                .iter()
                .zip(&self.cache)
                .all(|(lent, group)| lent.layers.len() == group.layers.len());

        let tabled = scratch.iter().all(|group| {
            let table = &group.block_tables;

            table.rank() == 2
                && table.dims()[0] >= rows
                && table.dims()[1] == pages
                && table.dtype() == DType::U32
                && group.page_size == self.config.page_size
        });
        if pages == 0 || !same_groups || !tabled {
            return Err(Error::Invalid(format!(
                "a scratch page set of {} groups for a cache of {}, whose tables are not u32 \
                 rows of {pages} pages for the {rows} rows of the largest bucket",
                scratch.len(),
                self.cache.len()
            )));
        }

        self.scratch = scratch;
        Ok(())
    }

    /// The rows of a paired forward: those of the largest bucket whose rows are sequences. A
    /// verification's token rows share their request's pages and have no paired forward.
    pub(crate) fn scratch_rows(&self) -> u32 {
        self.config.buckets.bounds().0
    }

    /// The runner's config.
    pub const fn config(&self) -> &CausalLmConfig {
        &self.config
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

        if batch.media.is_some() {
            return Err(Error::Invalid(format!(
                "batch {} is a media computation, which a causal runner serves none of",
                batch.id.0
            )));
        }
        columns.check(&shape)?;
        if batch.calls.len() != columns.calls() {
            return Err(Error::Invalid(format!(
                "batch {} names {} calls and states {} call rows",
                batch.id.0,
                batch.calls.len(),
                columns.calls()
            )));
        }

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

        // Work the runner states no bucket for is refused by its name, never served as text.
        if !columns.vision.is_empty() || !batch.images.is_empty() {
            return Err(Error::Unsupported(format!(
                "batch {} carries image rows, and this runner places no image features",
                batch.id.0
            )));
        }
        let verify = match &self.speculation {
            Some(speculation) => {
                let refills = columns.drafts.iter().any(|work| work.refill.is_some());
                if refills && self.state.is_some() {
                    return Err(Error::Unsupported(format!(
                        "batch {} refills the draft cache of a model that keeps a recurrent \
                         state, which needs a scratch state for each refill row",
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
                    .any(|call| call.kind == RowKind::Verify);
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

        refuse_drawn_rows(&shape.variant, &columns.parameters)?;

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

    /// The installed draft pages of each row of `batch`, `(first page, units)` in row order:
    /// the table the batch itself carries for the row's slot, or the one the cache holds.
    fn draft_tables(&mut self, batch: &Batch) -> Vec<(u32, Vec<u32>)> {
        let Some(speculation) = self.speculation.as_mut() else {
            return Vec::new();
        };
        let group = speculation.pool.group;

        let mut tables = Vec::with_capacity(batch.columns.slots.len());
        for held in &batch.columns.slots {
            let Some(slot) = u32::try_from(*held).ok().filter(|slot| *slot > 0) else {
                tables.push((0, Vec::new()));
                continue;
            };

            let carried = batch
                .block_tables
                .iter()
                .find(|table| table.request_pool_idx == slot && table.group_id == group)
                .map(|table| {
                    let units = table.unit_ids.iter().map(|unit| unit.0).collect();
                    (table.start_page, units)
                });
            let installed = match (carried, speculation.pages.as_mut()) {
                (Some(carried), _) => Some(carried),
                (None, Some(pages)) => pages(slot),
                (None, None) => None,
            };

            tables.push(installed.unwrap_or_default());
        }
        tables
    }

    /// Prepares the inputs of a step of `batch` in `bucket`: its padded columns, and the host
    /// values the bucket's forward computes; `scratch` is a paired forward's slot mapping.
    fn fill(&mut self, batch: &Batch, bucket: &Bucket, scratch: Option<&[u32]>) -> Result<()> {
        self.stage(batch, bucket, scratch)?;
        self.prepare_staged(bucket)
    }

    /// Pads `batch`'s columns to `bucket` in the runner's own bytes of the input block, and
    /// returns the ban masks of the step's sampling rows (`sampler::adjustments`), empty for
    /// a step that bans nothing. The columns of a bucket are those of every capacity class of
    /// it. A runner that speculates also pads the columns of the drafter's forward behind the
    /// bucket, which lie in the same block.
    fn stage(
        &mut self,
        batch: &Batch,
        bucket: &Bucket,
        scratch: Option<&[u32]>,
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
                &mut self.staged,
            )?;
            sampler::adjustments(&rows.adjustments, bucket.tokens as usize, vocabulary)?.1
        } else {
            columns::fill(
                &self.layout,
                self.config.slots,
                vocabulary,
                &batch.columns,
                bucket,
                scratch,
                &mut self.staged,
            )?;
            sampler::adjustments(&batch.columns.adjustments, bucket.rows as usize, vocabulary)?.1
        };

        let tables = self.draft_tables(batch);
        if let Some(speculation) = &self.speculation {
            speculation::fill_drafter(
                &self.layout,
                &speculation.pool,
                vocabulary,
                &batch.columns,
                &tables,
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
    fn phases(&self) -> &[PhaseId] {
        &self.config.phases
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

        let batch = TokenBatch {
            token_ids: None,
            ngram_context: None,
            positions: self.column(Column::Positions, tokens)?,
            slot_mapping: self.column(Column::SlotMapping, tokens)?,
            seq_lens: self.column(Column::SeqLens, rows)?,
            query_offsets: self.column(Column::QueryOffsets, rows + 1)?,
            logits_rows: self.column(Column::LogitsRows, rows)?,
            phase: if decode {
                PhaseKind::Decode
            } else {
                PhaseKind::Prefill
            },
            page_size: self.config.page_size,
            num_tokens: bucket.tokens,
            max_kv_len: self.config.max_kv_len,
            draft_lens: None,
            rope_positions: None,
            live_tokens: Some(self.column(Column::LiveTokens, 1)?),
        };
        batch.check()?;

        // A batch's state rows are its attention rows.
        let state = match &self.state {
            Some((live, checkpoint)) => Some(RecurrentState {
                verify: None,
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

        // The order `output` states, each entry with its place among the kept tensors.
        let mut scores = scores.into_iter();
        let mut outputs = vec![(speculation::CHOSEN, chosen)];

        outputs.extend(
            [speculation::LOGPROBS, speculation::RANKS]
                .into_iter()
                .zip(scores.by_ref()),
        );
        outputs.extend([
            (speculation::LOGICAL, logical),
            (speculation::SAMPLING, sampling),
        ]);
        outputs.extend(
            [speculation::TOP_VALUES, speculation::TOP_IDS]
                .into_iter()
                .zip(scores),
        );
        outputs.extend(valid.map(|valid| (speculation::VALID, valid)));
        outputs.extend(
            [
                speculation::PROMPT_LOGPROBS,
                speculation::PROMPT_RANKS,
                speculation::PROMPT_VALUES,
                speculation::PROMPT_IDS,
            ]
            .into_iter()
            .zip(prompt),
        );

        // The drafter's forward follows this one and takes the allocator's memory again: a
        // runner that speculates keeps the outputs and the hidden rows in its own.
        match &self.speculation {
            Some(speculation) => {
                speculation.kept.keep(speculation::HIDDEN, &hidden)?;

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
            None,
            columns,
        )?;

        match &self.speculation {
            Some(speculation) if bucket.variant == Variant::Draft => speculation::fill_drafter(
                &self.layout,
                &speculation.pool,
                vocabulary,
                &idle,
                &[],
                bucket,
                columns,
            ),
            _ => Ok(()),
        }
    }

    fn prepare_inputs(&mut self, batch: &Batch) -> Result<Bucket> {
        // Outside a step, a member of an expert union prepares the smallest capacity class of
        // the bucket: the class a step runs follows from its agreement (`execute`).
        let bucket = self.choose(batch)?;

        self.fill(batch, &bucket, None)?;
        Ok(bucket)
    }

    /// Prepares the inputs of a paired forward of `batch`, a batch the runner served, and
    /// returns the bucket the executor passes to `Graphs::shadow`.
    ///
    /// The columns are the batch's own, padded as a serving step's, except that every token
    /// writes its row's scratch pages, every row gathers its own row of the scratch set's
    /// page table, and the commit table is inert (`columns`). No slot is started and none of
    /// the batch's own page work is issued: the forward reads the slots' state as it stands
    /// and writes none of it.
    ///
    /// # Errors
    ///
    /// As [`Runner::prepare_inputs`]; [`Error::Unsupported`] for a model that keeps a
    /// recurrent state, which has no scratch state to advance, for a member of an expert
    /// union, whose exchange launches wait for the other members', for a verification, and
    /// for a runner that was lent no scratch page set; [`Error::Invalid`] for a row whose
    /// tokens lie past its scratch pages.
    fn prepare_shadow_inputs(&mut self, batch: &Batch) -> Result<Bucket> {
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

        let rows = &batch.columns.rows;
        let scratch = columns::scratch_column(
            &rows.positions,
            &rows.query_offsets,
            &rows.slot_mapping,
            bucket.tokens as usize,
            self.config.scratch_pages,
            self.config.page_size,
        )?;

        self.fill(batch, &bucket, Some(&scratch))?;

        // The rows' cached history reaches their scratch pages ahead of the forward.
        if let Some(hook) = self.page_work.as_mut() {
            hook(batch, PageWork::BeforeShadow)?;
        }
        self.paired.set(true);
        Ok(bucket)
    }

    fn execute(&mut self, batch: &Batch) -> Result<BatchOutput> {
        let started = Instant::now();

        // A join names no call: it is this member's part in a step of the union.
        if batch.shape.variant == Variant::ExpertJoin {
            return self.join_step(batch, started);
        }

        // Any other batch that names no call computes nothing: it carries releases alone.
        if batch.calls.is_empty() && batch.columns.calls() == 0 {
            return Ok(BatchOutput::default());
        }
        if self.union.as_ref().is_some_and(Member::experts_only) {
            return Err(Error::Invalid(format!(
                "batch {} reached a member of an expert union that holds experts alone and \
                 serves no request",
                batch.id.0
            )));
        }

        let work = carries_page_work(batch);
        if work && self.page_work.is_none() {
            return Err(Error::Invalid(format!(
                "batch {} carries page work (block tables, cache units, page copies or state \
                 copies), and the runner was given nothing that issues it",
                batch.id.0
            )));
        }
        // Everything that can refuse the batch comes first: the choice of a bucket and the
        // padding of the columns.
        let base = self.choose(batch)?;
        let verify = verifies(base.variant);

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
        let masks = self.stage(batch, &base, None)?;

        // The slot state's writes, ahead of the replay that reads them.
        self.slots.begin()?;
        self.slots
            .start(&self.sampler, &batch.columns, base.variant.is_decode())?;
        if !masks.is_empty() {
            self.slots.ban(&masks)?;
        }

        if let (true, Some(hook)) = (work, self.page_work.as_mut()) {
            hook(batch, PageWork::BeforeReplay)?;
        }

        // A member of an expert union agrees the step with the other members, which blocks
        // until each has entered it, and runs the capacity class that holds it.
        let bucket = match self.union.as_mut() {
            Some(member) => {
                let tokens = batch.columns.tokens() as u32;
                let capacity = member.enter(tokens, batch.phase, batch.generation)?;

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

        // The drafter's columns lie in the block the batch's graph was issued with, and its
        // graph reads what that graph kept: the stream's order is their synchronization.
        let proposals = match &drafter {
            Some(drafter) => {
                self.prepare_staged(drafter)?;
                Some(self.graphs.replay(drafter)?)
            }
            None => None,
        };

        if let (true, Some(hook)) = (work, self.page_work.as_mut()) {
            hook(batch, PageWork::AfterReplay)?;
        }

        self.graphs.wait()?;
        let landed = Instant::now();
        self.slots.settled();

        // The read waits for the stream, and so for the copies issued behind the replay.
        let calls = {
            let _guard = self.graphs.enter()?;
            let vocabulary = self.model.vocabulary();
            let candidates = self.config.candidates;

            let mut calls = if verify {
                output::verification_output(
                    &batch.calls,
                    &batch.columns,
                    &bucket,
                    vocabulary,
                    candidates,
                    &outputs,
                )?
            } else {
                output::calls_output(
                    &batch.calls,
                    &batch.columns,
                    &bucket,
                    vocabulary,
                    candidates,
                    &outputs,
                )?
            };
            if let Some(proposals) = &proposals {
                speculation::propose(batch, vocabulary, proposals, &mut calls)?;
            }
            calls
        };

        let host = launched.duration_since(started) + landed.elapsed();
        Ok(BatchOutput {
            calls,
            timing: TimingCounters {
                // The executor states the wait it saw before it called the runner.
                queued_us: 0,
                device_us: landed.duration_since(launched).as_micros() as u64,
                // Nothing is transferred for a completion: the scheduler reads it here.
                copy_us: 0,
                host_us: host.as_micros() as u64,
            },
            forward_stats: None,
        })
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

/// The buckets a runner of `config` states outside an expert union, and the layout of its
/// input tensors.
fn stated(config: &CausalLmConfig) -> Result<(Buckets, Layout)> {
    let buckets = config.buckets.with_drafts(config.drafts);

    let pages = match (config.drafts, &config.draft_pool) {
        (0, _) => 0,
        (_, Some(pool)) => pool.width(config.max_kv_len),
        (_, None) => {
            return Err(Error::Invalid(
                "a config that states drafts states the draft pool of its cache".into(),
            ));
        }
    };
    let layout = Layout::of(buckets.iter(), config.drafts, pages)?;

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

/// Whether `batch` carries work on the cache's pages or the recurrent state's.
fn carries_page_work(batch: &Batch) -> bool {
    !(batch.block_tables.is_empty()
        && batch.cache_units.is_empty()
        && batch.page_loads.is_empty()
        && batch.page_offloads.is_empty()
        && batch.columns.state_copies.is_empty())
}

/// Refuses a greedy bucket a row that draws.
///
/// A bucket's variant fixes the selection of every row. A drawing bucket serves a greedy row
/// as well, as a draw over its one largest candidate (`sampler::parameters`). The converse is
/// not so: an argmax is not what a row that states a temperature asked for, so a greedy
/// variant holding one is refused by the row.
fn refuse_drawn_rows(variant: &ipc::execution::Variant, parameters: &[f32]) -> Result<()> {
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

/// Refuses recurrent state buffers that are not `model`'s.
fn check_state(model: &dyn TextModel, state: Option<&(StateBuffers, StateBuffers)>) -> Result<()> {
    match (model.state(), state) {
        (None, None) => Ok(()),
        (Some(shape), Some((live, checkpoint))) => {
            let layered = |buffers: &[Vec<Tensor>]| {
                buffers.len() == shape.layers as usize
                    && buffers
                        .iter()
                        .all(|layer| layer.len() == shape.fields.len())
            };
            if layered(live) && layered(checkpoint) {
                return Ok(());
            }

            Err(Error::Invalid(format!(
                "the recurrent state lends {} live and {} checkpoint layers, and the model \
                 keeps {} layers of {} fields",
                live.len(),
                checkpoint.len(),
                shape.layers,
                shape.fields.len()
            )))
        }
        (Some(_), None) => Err(Error::Invalid(
            "the model keeps a recurrent state and the runner was lent none".into(),
        )),
        (None, Some(_)) => Err(Error::Invalid(
            "the runner was lent a recurrent state its model does not keep".into(),
        )),
    }
}

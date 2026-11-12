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
//! 2. the batch's columns are padded to the bucket and written into the input block
//!    ([`columns`]), with the host values the forward computes;
//! 3. the slots the batch starts a request on are started, and its ban masks copied
//!    ([`slots`]);
//! 4. the batch's page work that precedes the forward is issued ([`PageWork::BeforeReplay`]);
//! 5. the bucket's graph is replayed;
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
//!
//! ## Paired forwards
//!
//! Before a generation switch the executor computes a served batch again as a paired forward
//! (`Graphs::shadow`), whose inputs [`CausalLmRunner::prepare_shadow_inputs`] writes. It is the
//! same forward over other pages: the rows write and read the scratch page set the cache lends
//! ([`CausalLmRunner::set_scratch`]) in its own numbering, the commit table is inert and no
//! slot is started, so nothing a serving request reads is written. A model that keeps a
//! recurrent state has no scratch state to advance, and its paired forwards are refused.

mod columns;
mod output;
mod slots;

#[cfg(test)]
mod tests;

use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;

use ipc::execution::{Batch, BatchOutput, RowKind};
use ipc::{PhaseId, TimingCounters};
use nn::family::TextModel;
use nn::{KvCache, PhaseKind, RecurrentState, TokenBatch};
use platform::Stream;
use proto::module::commit::COMMIT_FIELDS;
use proto::module::sampler::{PARAMETER_COLUMNS, TEMPERATURE};
use uniserve_tensor::{DType, Shape, Tensor};

use self::columns::{Column, Layout};
use self::slots::Slots;
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
}

/// Which part of a batch's page work the rank's cache issues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PageWork {
    /// What the forward reads: the batch's rows of the pooled page tables are installed, its
    /// new units reset and filled with the copies a reused prefix needs, its page loads are
    /// issued, and each restored slot's recurrent state is copied into the live buffers.
    BeforeReplay,
    /// What the forward left: each kept checkpoint of the recurrent state is copied to its
    /// page, and the batch's page offloads are issued.
    AfterReplay,
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
}

impl CausalLmRunner {
    /// The bytes of the runner's input tensors for `config`: the `inputs` its [`Graphs`] are
    /// made with.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket.
    pub fn input_bytes(config: &CausalLmConfig) -> Result<u64> {
        Ok(Layout::of(config.buckets.iter())?.bytes())
    }

    /// The bytes of device memory the request slots' state takes for `config` over a
    /// `vocabulary`-wide logit row: the count and bias tables, the commit state and the ban
    /// masks, which the runner allocates when its graphs hold a device.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config with no bucket.
    pub fn slot_bytes(config: &CausalLmConfig, vocabulary: u32) -> Result<u64> {
        let layout = Layout::of(config.buckets.iter())?;

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
    /// layers over `config`'s page size and slots, or a state that is not the model's; the
    /// refusals of the slot state's allocation and of the sampler.
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
        let layout = Layout::of(config.buckets.iter())?;

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
        let slots = Slots::new(stream, config.slots, vocabulary, layout.rows() as u32, 1)?;
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

        let runner = Self {
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

    /// Lends the scratch page set paired forwards compute over: one `KvCache` per attention
    /// group, in the order and with the layers of the runner's cache, whose `block_tables` is
    /// the set's own page table, `[rows, scratch_pages]` U32 for at least the rows of the
    /// runner's largest bucket, row `r` naming the pages `r · scratch_pages ..
    /// (r + 1) · scratch_pages` of the set's planes.
    ///
    /// The caller keeps the set's memory as it keeps the cache's. Before each paired forward
    /// it copies, on the runner's stream, the pages the batch's rows read into the rows'
    /// scratch pages (page `k` of row `r`'s request into page `r · scratch_pages + k`), so the
    /// forward attends over the context the served batch attended over.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a config that states no scratch pages, or a set that is not the
    /// cache's groups and layers over the config's page size and scratch pages.
    pub fn set_scratch(&mut self, scratch: Vec<KvCache>) -> Result<()> {
        let pages = self.config.scratch_pages as usize;
        let same_groups = scratch.len() == self.cache.len()
            && scratch
                .iter()
                .zip(&self.cache)
                .all(|(lent, group)| lent.layers.len() == group.layers.len());

        let tabled = scratch.iter().all(|group| {
            let table = &group.block_tables;

            table.rank() == 2
                && table.dims()[0] >= self.layout.rows()
                && table.dims()[1] == pages
                && table.dtype() == DType::U32
                && group.page_size == self.config.page_size
        });
        if pages == 0 || !same_groups || !tabled {
            return Err(Error::Invalid(format!(
                "a scratch page set of {} groups for a cache of {}, whose tables are not u32 \
                 rows of {pages} pages for the {} rows of the largest bucket",
                scratch.len(),
                self.cache.len(),
                self.layout.rows()
            )));
        }

        self.scratch = scratch;
        Ok(())
    }

    /// Prepares the inputs of a paired forward of `batch`, a batch the runner served, and
    /// returns the bucket the executor passes to `Graphs::shadow`.
    ///
    /// The columns are the batch's own, padded as a serving step's, except that every token
    /// writes its row's scratch pages, every row gathers its own row of the scratch set's
    /// page table, and the commit table is inert (`columns`). No slot is started and no page
    /// work is issued: the forward reads the slots' state as it stands and writes none of it.
    ///
    /// # Errors
    ///
    /// As [`Runner::prepare_inputs`]; [`Error::Unsupported`] for a model that keeps a
    /// recurrent state, which has no scratch state to advance, and for a runner that was lent
    /// no scratch page set; [`Error::Invalid`] for a row whose tokens lie past its scratch
    /// pages.
    pub fn prepare_shadow_inputs(&mut self, batch: &Batch) -> Result<Bucket> {
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
        self.paired.set(true);
        Ok(bucket)
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
        let verifies = columns
            .rows
            .calls
            .iter()
            .any(|call| call.kind == RowKind::Verify);
        if verifies || !columns.drafts.is_empty() {
            return Err(Error::Unsupported(format!(
                "batch {} verifies or drafts tokens, and this runner speculates nothing",
                batch.id.0
            )));
        }

        refuse_drawn_rows(&shape.variant, &columns.parameters)?;

        self.graphs
            .select(|bucket| bucket.holds(&shape))
            .ok_or_else(|| match self.config.buckets.select(&shape) {
                Some(bucket) => Error::State(match bucket.variant.is_decode() {
                    true => "a decode bucket that holds the batch is not captured",
                    false => "a prefill bucket that holds the batch is not captured",
                }),
                None => {
                    let (rows, tokens) = self.config.buckets.bounds();
                    Error::Invalid(format!(
                        "no bucket holds {shape}: the runner's buckets top at {rows} rows and \
                         {tokens} tokens, and a batch past that bound is admitted as smaller ones"
                    ))
                }
            })
    }

    /// Writes `batch`'s columns, padded to `bucket`, into the input block, with the host
    /// values the bucket's forward computes; `scratch` is a paired forward's slot mapping.
    fn fill(&self, batch: &Batch, bucket: &Bucket, scratch: Option<&[u32]>) -> Result<()> {
        // The meta forward that writes the host values computes over the serving cache: a
        // forward's host values follow from the bucket alone.
        self.paired.set(false);

        self.graphs.prepare_inputs(self, bucket, |block| {
            columns::fill(
                &self.layout,
                self.config.slots,
                self.model.vocabulary(),
                &batch.columns,
                bucket,
                scratch,
                block,
            )
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
        self.config.buckets.capture_order().copied().collect()
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        if !self.config.buckets.contains(bucket) {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is none the runner states"
            )));
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
                .forward(&batch, &cache, state.as_ref(), None, &embedded)?
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

        // The order `output` states.
        let mut outputs = Vec::with_capacity(12);
        let mut scores = scores.into_iter();
        outputs.push(chosen);
        outputs.extend(scores.by_ref().take(2));
        outputs.extend([logical, sampling]);
        outputs.extend(scores);
        outputs.extend(valid);
        outputs.extend(prompt);
        Ok(outputs)
    }

    fn idle_inputs(&self, bucket: &Bucket, columns: &mut [u8]) -> Result<()> {
        if !self.config.buckets.contains(bucket) {
            return Err(Error::Invariant(format!(
                "bucket {bucket} is none the runner states"
            )));
        }

        // An idle step is a serving step's forward: what a capture records.
        self.paired.set(false);

        let idle = columns::idle(bucket, self.config.page_size, self.config.max_kv_len);
        columns::fill(
            &self.layout,
            self.config.slots,
            self.model.vocabulary(),
            &idle,
            bucket,
            None,
            columns,
        )
    }

    fn prepare_inputs(&mut self, batch: &Batch) -> Result<Bucket> {
        let bucket = self.choose(batch)?;

        self.fill(batch, &bucket, None)?;
        Ok(bucket)
    }

    fn execute(&mut self, batch: &Batch) -> Result<BatchOutput> {
        let started = Instant::now();

        // A batch that names no call computes nothing: it carries releases alone.
        if batch.calls.is_empty() && batch.columns.calls() == 0 {
            return Ok(BatchOutput::default());
        }

        let work = carries_page_work(batch);
        if work && self.page_work.is_none() {
            return Err(Error::Invalid(format!(
                "batch {} carries page work (block tables, cache units, page copies or state \
                 copies), and the runner was given nothing that issues it",
                batch.id.0
            )));
        }
        let bucket = self.prepare_inputs(batch)?;

        // The slot state's writes, ahead of the replay that reads them.
        self.slots.begin()?;
        self.slots
            .start(&self.sampler, &batch.columns, bucket.variant.is_decode())?;
        if batch
            .columns
            .adjustments
            .banned
            .iter()
            .any(|row| !row.is_empty())
        {
            let (_, masks) = sampler::adjustments(
                &batch.columns.adjustments,
                bucket.rows as usize,
                self.model.vocabulary(),
            )?;
            self.slots.ban(&masks)?;
        }

        if let (true, Some(hook)) = (work, self.page_work.as_mut()) {
            hook(batch, PageWork::BeforeReplay)?;
        }

        let launched = Instant::now();
        let outputs = self.graphs.replay(&bucket)?;

        if let (true, Some(hook)) = (work, self.page_work.as_mut()) {
            hook(batch, PageWork::AfterReplay)?;
        }

        self.graphs.wait()?;
        let landed = Instant::now();
        self.slots.settled();

        // The read waits for the stream, and so for the copies issued behind the replay.
        let calls = {
            let _guard = self.graphs.enter()?;
            output::calls_output(
                batch,
                &bucket,
                self.model.vocabulary(),
                self.config.candidates,
                &outputs,
            )?
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

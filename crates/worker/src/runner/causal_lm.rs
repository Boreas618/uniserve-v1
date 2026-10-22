//! Text forwards of a causal language model on one lane binding.
//!
//! A [`CausalLMRunner`] owns one binding's input and output tensors
//! ([`InputBuffers`]), its graph shapes and captured graphs, and the Python
//! runner ([`Methods`]). It runs on the control thread and issues nothing
//! itself: it records a forward into a [`Job`], which the lane's execution
//! thread performs. Per forward it:
//!
//! 1. resolves the rows' attention pages against the installed block tables
//!    (checking every write against the cache's live accesses);
//! 2. selects a resident graph of the smallest bucket that holds the rows,
//!    before the rank is sealed a graph to capture, or an eager run for
//!    multimodal rows and rows no bucket holds;
//! 3. records the writes of the forward's inputs on its stream, one batched
//!    copy ([`CausalLMRunner::stage`]);
//! 4. records the graph's launch, or the eager call of the runner's forward
//!    method ([`CausalLMRunner::forward`]), then the eager selection for
//!    rows that sample, verify drafts or score their prompt
//!    ([`CausalLMRunner::select`]), and the capture of a graph the eager
//!    call warmed;
//! 5. once the job completed, stores the captured graph and counts how the
//!    forward ran ([`CausalLMRunner::complete`]).
//!
//! A decode forward evaluates one token per row from the slot state: its
//! inputs are each row's slot, forced-finish flag and finish tokens, and a
//! row whose token comes from the host has it written to its slot first.
//! Its graph gathers, runs the model and selects: greedily, or, in the
//! bucket's sampling graph, each row from its top-k candidates with the
//! row's parameters and draw, which are inputs too. A decode whose rows
//! all select this way runs no Python. A decode bucket pads with rows of
//! slot 0.
//!
//! A prefill forward evaluates rows of host tokens (prompt chunks and
//! verification rows): its inputs are the rows' packed columns, tokens and
//! positions. Its graph gathers, runs the model and, unless the forward
//! only writes the prefix cache, keeps the hidden states and projects each
//! row's last position. A prefill bucket adds one sequence holding every
//! padding token, then empty sequences.
//!
//! Padding rows never write K/V (their slot mapping is -1).
//!
//! A rank's runners and the request state they share are
//! [`CausalLMRunners`] ([`runners`]): a batch of token calls reserves its
//! storage ([`batch`]), turns its calls into rows, sampling tasks and prompt
//! scoring ([`calls`], [`context`], [`step`]), records one forward per
//! expert microbatch and the commit of its outputs ([`commit`]) as the jobs
//! of its lane. [`start`] builds the runners and [`warmup`] captures their
//! graphs.

mod batch;
mod calls;
mod commit;
mod context;
pub(crate) mod input_buffers;
pub(crate) mod methods;
pub(crate) mod rows;
mod runners;
pub(crate) mod start;
mod step;
pub(crate) mod warmup;

pub(crate) use input_buffers::{Dimensions, FINISH_IDS};
pub(crate) use methods::{Construction, Functions};
pub(crate) use rows::{RowSource, TokenRow, logit_layout};
pub(crate) use runners::{CausalLMRunners, Pending, Step, complete, forward};

use cuda::{CopyBatch, DeviceId, SourceOrder, Stream};
use ipc::ForwardMode;

use super::TokenSelection;
use super::input_buffers::InputBuffers;
use super::methods::{Argument, MethodCall, Methods};
use crate::cuda_graph::{GraphBreaks, GraphMode, GraphTable, Lookup, bucket, task_buckets};
use crate::cuda_graph::{PrefillVariant, ShapeKey, TextForward, TextShapes};
use crate::ffi::{DType, Handle};
use crate::lane::{Capture, Completion, Job};
use crate::memory::allocation::StorageView;
use crate::memory::{AttentionRow, PrefixCache};
use crate::request::decode_state::DecodeState;
use crate::runner::attention::pack_rows;
use crate::sampling::{
    GREEDY, NEUTRAL, RowSampling, SamplingMetadata, bitmask_words, candidates, task_bucket,
};
use crate::stream::{Fill, Issue};
use crate::{Error, Result};
use runners::COMPLETION_FIELDS;

/// The commit mode of a row that only sets its slot's cache length
/// (`uniserve.sampling.state.CACHE`).
pub(crate) const COMMIT_CACHE: i64 = 3;

/// A numerical block table's page layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableShape {
    pub(crate) block_size: u32,
    pub(crate) windowed: bool,
}

/// The prompt chunks a prefill forward scores, one entry per chunk, which
/// the runner expands into prompt rows ([`prompt_rows`]).
#[derive(Debug, Clone, Default)]
pub(crate) struct PromptChunks {
    /// The chunk's first candidate row and its candidate rows; a chunk of
    /// none only retains the logits of its row.
    pub(crate) rows: Vec<i64>,
    pub(crate) row_counts: Vec<i64>,
    /// Its first token in the forward.
    pub(crate) tokens: Vec<i64>,
    /// What predicts its first token: nothing (0), the slot's retained
    /// logits (1), or candidate row `continued - 2`.
    pub(crate) continued: Vec<i64>,
    pub(crate) slots: Vec<i64>,
    /// Where its words start in the logprob tensor.
    pub(crate) offsets: Vec<i64>,
    /// The row of the tail tensor its last logits go to.
    pub(crate) tails: Vec<i64>,
    /// Its top count, which its words report.
    pub(crate) tops: Vec<i64>,
    /// The top bucket every chunk is scored at: that of its widest chunk.
    pub(crate) top: usize,
}

/// The prompt rows of `chunks`, which `score_prompt` scores: per row its
/// hidden position (the token position of a candidate row of
/// `logit_rows`), its retained slot (or -1), its target token's index in
/// the forward, its chunk's first word, its index in the chunk, the chunk's
/// scored rows and their top count, and its tail row. A chunk's scored rows
/// predict its tokens in order, a first chunk's from its second token: the
/// first from what its `continued` names, each next from the candidate row
/// before it. Its last candidate row then retains its logits in its tail
/// row. A row without a target has the count -1 and reports nothing, and a
/// row without a tail copies into the `spare` tail row.
fn prompt_rows(chunks: &PromptChunks, logit_rows: &[i64], spare: i64) -> Result<Vec<[i64; 8]>> {
    let position = |row: i64| -> Result<i64> {
        usize::try_from(row)
            .ok()
            .and_then(|row| logit_rows.get(row).copied())
            .ok_or_else(|| {
                Error::Invariant("a prompt chunk names a row its forward does not project".into())
            })
    };
    let mut rows = Vec::new();
    for chunk in 0..chunks.rows.len() {
        let (first, count) = (chunks.rows[chunk], chunks.row_counts[chunk]);
        let (token, continued) = (chunks.tokens[chunk], chunks.continued[chunk]);
        // A first chunk scores its tokens after the first.
        let fresh = i64::from(continued == 0);
        let scored = (count - fresh).max(0);
        for index in 0..scored {
            // The candidate row predicting the chunk's token `index + fresh`,
            // counted from the chunk's first; -1 is what precedes it.
            let before = index + fresh - 1;
            let (hidden, slot) = match before {
                0.. => (position(first + before)?, -1),
                _ if continued == 1 => (0, chunks.slots[chunk]),
                _ => (position(continued - 2)?, -1),
            };
            rows.push([
                hidden,
                slot,
                token + index + fresh,
                chunks.offsets[chunk],
                index,
                scored,
                chunks.tops[chunk],
                spare,
            ]);
        }
        let last = first + (count - 1).max(0);
        rows.push([position(last)?, -1, 0, 0, 0, 0, -1, chunks.tails[chunk]]);
    }
    Ok(rows)
}

/// A forward's selection inputs padded to its task bucket
/// ([`CausalLMRunner::select`]): padding tasks select greedily over row 0
/// in slot 0, every token allowed, and their results are not read. A
/// verification, which selects eagerly, is not padded.
struct Selection {
    tasks: usize,
    rows: Vec<i64>,
    paths: Vec<i64>,
    draws: Vec<f32>,
    params: Vec<f32>,
    slots: Vec<i64>,
    gates: Vec<i64>,
    mask_rows: Vec<i64>,
    biased: Vec<u8>,
    force: Vec<u8>,
    drafts: Vec<i64>,
    terminal: Vec<i64>,
    /// The reporting tasks and their top counts, padded to their bucket
    /// with rows of task zero and count -1, which report nothing.
    logprob_tasks: Vec<i64>,
    logprob_counts: Vec<i64>,
}

impl Selection {
    /// `metadata`'s inputs on a binding of `capacity` rows.
    fn pad(metadata: &SamplingMetadata, capacity: usize) -> Self {
        let tasks = if metadata.width == 1 {
            task_bucket(metadata.tasks, capacity)
        } else {
            metadata.tasks
        };
        let width = metadata.width;
        let extra = tasks - metadata.tasks;
        fn padded<T: Copy>(values: &[T], fill: T, count: usize) -> Vec<T> {
            let mut values = values.to_vec();
            values.resize(values.len() + count, fill);
            values
        }
        let mut params = metadata.params.clone();
        for _ in 0..extra * width {
            params.extend_from_slice(&NEUTRAL);
        }
        let (rows, _) = metadata.logprob_bucket;
        let reporting = metadata.logprob_tasks.len();
        let counts: Vec<i64> = metadata.logprobs.as_ref().map_or_else(Vec::new, |layout| {
            layout.counts.iter().map(|&count| count as i64).collect()
        });
        Self {
            tasks,
            rows: padded(&metadata.rows, 0, extra * width),
            paths: padded(&metadata.paths, GREEDY, extra),
            draws: padded(&metadata.draws, 0.0, extra * width),
            params,
            slots: padded(&metadata.slots, 0, extra),
            gates: padded(&metadata.gates, 0, extra),
            mask_rows: padded(&metadata.mask_rows, -1, extra * width),
            biased: padded(&metadata.biased, 0, extra),
            force: padded(&metadata.force, 0, extra),
            drafts: padded(&metadata.drafts, -1, extra * (width - 1)),
            terminal: padded(&metadata.terminal, 0, extra),
            logprob_tasks: padded(&metadata.logprob_tasks, 0, rows - reporting),
            logprob_counts: padded(&counts, -1, rows - reporting),
        }
    }

    /// The inputs of `tasks` padding tasks, none reporting.
    fn padding(tasks: usize) -> Self {
        Self {
            tasks,
            rows: vec![0; tasks],
            paths: vec![GREEDY; tasks],
            draws: vec![0.0; tasks],
            params: NEUTRAL.repeat(tasks),
            slots: vec![0; tasks],
            gates: vec![0; tasks],
            mask_rows: vec![-1; tasks],
            biased: vec![0; tasks],
            force: vec![0; tasks],
            drafts: Vec::new(),
            terminal: vec![0; tasks],
            logprob_tasks: vec![0; tasks],
            logprob_counts: vec![-1; tasks],
        }
    }
}

/// Precomputed embeddings replacing the token embeddings of consecutive
/// tokens of an invocation: one vision block's features.
pub(crate) struct Embeddings {
    /// The first token, counted from the invocation's first token.
    pub(crate) token: usize,
    /// `[tokens, hidden]` features in the model dtype.
    pub(crate) values: StorageView,
}

/// One invocation request.
pub(crate) struct StepRequest<'a> {
    /// What the rows are: a prefill, a decode or a verification.
    pub(crate) mode: ForwardMode,
    pub(crate) rows: &'a [TokenRow],
    /// Each decode row's finish tokens, padded with -1; a row beyond the
    /// slice has none. A decode forward without `sampling` selects with
    /// them.
    pub(crate) finish_ids: &'a [[i64; FINISH_IDS]],
    /// The rows' draws and parameters of a decode forward that samples its
    /// rows itself; absent when its own selection is greedy.
    pub(crate) row_sampling: Option<&'a RowSampling>,
    /// The tasks the runner's selection selects; absent when no row
    /// samples or a decode forward's own selections are final.
    pub(crate) sampling: Option<&'a SamplingMetadata>,
    /// The prompt chunks the forward scores.
    pub(crate) prompt: &'a PromptChunks,
    /// Vision features placed over some of the rows' tokens.
    pub(crate) embeddings: &'a [Embeddings],
}

/// The storage a step borrows.
pub(crate) struct StepContext<'a> {
    pub(crate) prefix_cache: &'a mut PrefixCache,
    pub(crate) state: &'a DecodeState,
}

/// How an invocation ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Executed {
    pub(crate) mode: GraphMode,
    pub(crate) live_tokens: usize,
    pub(crate) padded_tokens: usize,
}

/// How an invocation runs, before its expert capacity selects a graph
/// variant.
enum Choice {
    Eager,
    Graph(ShapeKey),
}

/// A planned forward whose input writes its job holds, awaiting its expert
/// step capacity.
pub(crate) struct Staged {
    plan: Plan,
}

impl Staged {
    /// Tokens the forward sends to each expert layer: its graph's padded
    /// tokens, or its live tokens when it runs eagerly.
    pub(crate) fn expert_tokens(&self) -> usize {
        self.plan.tokens
    }
}

/// One planned forward: its shapes and host columns.
struct Plan {
    mode: ForwardMode,
    rows: usize,
    live_tokens: usize,
    tokens: usize,
    widths: Vec<i64>,
    queries: Vec<i64>,
    prefixes: Vec<i64>,
    starts: Vec<i64>,
    causal: Vec<i64>,
    /// The forward reads each row's causality from the device column
    /// (`causal_values`): its graph serves rows of either causality, or
    /// its rows differ.
    device_causality: bool,
    /// The forward reads the embedding columns: vision features replace the
    /// embeddings of the tokens the mask marks, or its graph reads them.
    embeddings: bool,
    /// Rotary position axes the rows carry: 1, or 3 when a row places
    /// positions explicitly.
    axes: usize,
    /// Candidate rows the eager selection projects at `logit_rows`: every
    /// position of a verification, or of a prefill that scores its prompt.
    logit_count: usize,
    /// A prefill keeps hidden states and last-position logits.
    outputs: bool,
    choice: Choice,
    /// Host column words of the row gather (absent for decode).
    row_columns: Vec<i64>,
    token_ids: Vec<i64>,
    positions: Vec<i64>,
    slots: Vec<i64>,
    force: Vec<u8>,
    /// `[rows, FINISH_IDS]` finish tokens of a decode, padded with -1.
    finish_ids: Vec<i64>,
    /// Candidates each row of a decode samples among, or zero for its
    /// greedy selection; then the rows' draws and parameters.
    candidates: usize,
    draws: Vec<f32>,
    params: Vec<f32>,
    logit_rows: Vec<i64>,
    /// The prompt rows the forward scores ([`prompt_rows`]).
    prompt: Vec<[i64; 8]>,
    /// `(token offset, slot)` device copies of a prefill row's first token
    /// from its slot's next input token.
    relays: Vec<(usize, u32)>,
    /// Slot state a decode row's host source supplies: `(address, value)`
    /// of a next input token or a logical position.
    slot_words: Vec<(u64, i64)>,
}

/// One lane binding of a causal language model.
pub(crate) struct CausalLMRunner {
    device: Option<DeviceId>,
    stream: Option<Stream>,
    methods: Methods,
    functions: Functions,
    /// An attention operator plans each launch on the host.
    plans_attention: bool,
    /// Where the distributed expert weights note the copies a captured
    /// forward performs between its segments.
    breaks: Option<GraphBreaks>,
    buffers: InputBuffers<Dimensions>,
    tables: Vec<TableShape>,
    /// Configured per-table widths every graph addresses.
    widths: Vec<i64>,
    shapes: TextShapes,
    graphs: GraphTable<ShapeKey>,
    pool: Option<Handle>,
}

/// What the runner is built from.
pub(crate) struct RunnerParts {
    /// Names the runner in errors.
    pub(crate) component: String,
    pub(crate) device: Option<DeviceId>,
    pub(crate) stream: Option<Stream>,
    pub(crate) methods: Methods,
    pub(crate) functions: Functions,
    pub(crate) plans_attention: bool,
    pub(crate) breaks: Option<GraphBreaks>,
    pub(crate) buffers: InputBuffers<Dimensions>,
    pub(crate) tables: Vec<TableShape>,
    pub(crate) widths: Vec<i64>,
    pub(crate) shapes: TextShapes,
    /// The graph pool, absent when graphs are off.
    pub(crate) pool: Option<Handle>,
}

impl CausalLMRunner {
    pub(crate) fn new(parts: RunnerParts) -> Self {
        let graphs = GraphTable::new(
            format!("text runner of {}", parts.component),
            parts.methods.graphs.release.clone(),
        );
        Self {
            device: parts.device,
            stream: parts.stream,
            methods: parts.methods,
            functions: parts.functions,
            plans_attention: parts.plans_attention,
            breaks: parts.breaks,
            buffers: parts.buffers,
            tables: parts.tables,
            widths: parts.widths,
            shapes: parts.shapes,
            graphs,
            pool: parts.pool,
        }
    }

    pub(crate) fn device(&self) -> Option<DeviceId> {
        self.device
    }

    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    pub(crate) fn shapes(&self) -> &TextShapes {
        &self.shapes
    }

    /// The runner's graph pool, absent when graphs are off.
    pub(crate) fn pool(&self) -> Option<&Handle> {
        self.pool.as_ref()
    }

    pub(crate) fn dimensions(&self) -> Dimensions {
        self.buffers.dimensions
    }

    pub(crate) fn column(&self, name: &str) -> Result<&StorageView> {
        self.buffers.column(name)
    }

    /// Graphs captured so far.
    pub(crate) fn captured(&self) -> usize {
        self.graphs.len()
    }

    /// Refuse further captures.
    pub(crate) fn seal(&mut self) {
        self.graphs.seal();
    }

    /// Release every graph and accept captures again
    /// ([`GraphTable::reopen`]).
    pub(crate) fn reopen(&mut self) -> Result<()> {
        self.graphs.reopen()
    }

    /// Plan `request` and record the writes of its inputs on the runner's
    /// stream in `job`.
    pub(crate) fn stage(
        &self,
        request: &StepRequest<'_>,
        context: &mut StepContext<'_>,
        job: &Job,
    ) -> Result<Staged> {
        let plan = self.plan(request, context)?;
        self.copy_columns(&plan, request, context.state, job)?;
        Ok(Staged { plan })
    }

    /// The columns `names` as method arguments, in order.
    fn tensors(&self, names: &[&str]) -> Result<Vec<Argument>> {
        names
            .iter()
            .map(|name| self.buffers.column(name).map(Argument::from))
            .collect()
    }

    /// The runner's forward method of a staged forward at `capacity`:
    /// `decode`, or `prefill` for prefill and verification rows.
    fn method(&self, staged: &Staged, capacity: Option<usize>) -> Result<MethodCall> {
        let plan = &staged.plan;
        let capacity = capacity.unwrap_or(0);
        if plan.mode == ForwardMode::Decode {
            let mut arguments = vec![
                plan.rows.into(),
                plan.widths.as_slice().into(),
                plan.prefixes.as_slice().into(),
                plan.starts.as_slice().into(),
                capacity.into(),
                plan.candidates.into(),
            ];
            arguments.extend(self.tensors(&[
                "request_pool_indices",
                "decode_force_finish",
                "finish_ids",
                "logits",
                "draws",
                "params",
                "completion",
                "continuation",
                "tagged",
                "points",
                "transition",
            ])?);
            return Ok(self.methods.call(&self.functions.decode, arguments));
        }

        let mut arguments = vec![
            plan.rows.into(),
            plan.tokens.into(),
            plan.widths.as_slice().into(),
            plan.queries.as_slice().into(),
            plan.prefixes.as_slice().into(),
            plan.starts.as_slice().into(),
            plan.causal.as_slice().into(),
            plan.axes.into(),
            plan.outputs.into(),
            capacity.into(),
        ];
        arguments.extend(self.tensors(&["row_columns", "input_ids", "positions"])?);
        arguments.push(self.causal_values(plan)?);
        if plan.embeddings {
            arguments.extend(self.tensors(&["input_embeddings", "embedding_mask"])?);
        } else {
            arguments.extend([Argument::None, Argument::None]);
        }
        arguments.extend(self.tensors(&["hidden", "logits"])?);
        Ok(self.methods.call(&self.functions.prefill, arguments))
    }

    /// The runner's method planning the attention launches of `plan`'s
    /// forward on the host: `plan_decode` or `plan_prefill`.
    fn plan_attention(&self, plan: &Plan) -> Result<MethodCall> {
        if plan.mode == ForwardMode::Decode {
            let mut arguments = vec![
                plan.rows.into(),
                plan.widths.as_slice().into(),
                plan.prefixes.as_slice().into(),
                plan.starts.as_slice().into(),
            ];
            arguments.extend(self.tensors(&["request_pool_indices"])?);
            return Ok(self.methods.call(&self.functions.plan_decode, arguments));
        }

        let mut arguments = vec![
            plan.rows.into(),
            plan.tokens.into(),
            plan.widths.as_slice().into(),
            plan.queries.as_slice().into(),
            plan.prefixes.as_slice().into(),
            plan.starts.as_slice().into(),
            plan.causal.as_slice().into(),
        ];
        arguments.extend(self.tensors(&["row_columns"])?);
        arguments.push(self.causal_values(plan)?);
        Ok(self.methods.call(&self.functions.plan_prefill, arguments))
    }

    /// The device column a prefill reads its rows' causality from, or none
    /// when the rows' host causality applies to all of them.
    fn causal_values(&self, plan: &Plan) -> Result<Argument> {
        if !plan.device_causality {
            return Ok(Argument::None);
        }
        Ok(self
            .tensors(&["causal_values"])?
            .pop()
            .unwrap_or(Argument::None))
    }

    /// The task buckets of the runner's selections.
    pub(crate) fn task_buckets(&self) -> Vec<usize> {
        task_buckets(self.buffers.dimensions.rows)
    }

    /// The key of the selection graph of `metadata`'s tasks: their task
    /// bucket and the samplers they take.
    fn selection_key(&self, metadata: &SamplingMetadata) -> ShapeKey {
        ShapeKey::Select {
            tasks: task_bucket(metadata.tasks, self.buffers.dimensions.rows),
            top_k: metadata.top_k,
            categorical: metadata.categorical.then_some(metadata.sorts),
        }
    }

    /// The runner's `select` of `key`'s tasks (a [`ShapeKey::Select`]).
    pub(crate) fn select_call(&self, key: &ShapeKey) -> Result<MethodCall> {
        let ShapeKey::Select {
            tasks,
            top_k,
            categorical,
        } = *key
        else {
            return Err(Error::Invariant("a selection has a select key".into()));
        };
        let mut arguments: Vec<Argument> = vec![
            tasks.into(),
            candidates(self.buffers.dimensions.vocab).into(),
            top_k.into(),
            categorical.is_some().into(),
            categorical.unwrap_or(false).into(),
        ];
        arguments.extend(self.tensors(&[
            "logits",
            "select_rows",
            "select_paths",
            "draws",
            "params",
            "tasks",
            "select_masks",
            "select_mask_rows",
            "select_biased",
            "select_force",
            "select_finish",
            "select_transitions",
            "completion",
            "continuation",
            "tagged",
            "points",
            "transition",
            "scored",
        ])?);
        Ok(self.methods.call(&self.functions.select, arguments))
    }

    /// The runner's `score` of `key`'s rows (a [`ShapeKey::Score`]).
    pub(crate) fn score_call(&self, key: &ShapeKey) -> Result<MethodCall> {
        let ShapeKey::Score { rows, top } = *key else {
            return Err(Error::Invariant("a scoring has a score key".into()));
        };
        let dump = self.buffers.dimensions.dump_words();
        let mut arguments: Vec<Argument> = vec![rows.into(), top.into(), dump.into()];
        arguments.extend(self.tensors(&[
            "logits",
            "logprob_tasks",
            "logprob_counts",
            "scored",
            "completion",
            "logprobs",
        ])?);
        Ok(self.methods.call(&self.functions.score, arguments))
    }

    /// The runner's `score_prompt` of `key`'s block of prompt rows (a
    /// [`ShapeKey::Prompt`]), from the block column's first row.
    pub(crate) fn score_prompt_call(&self, key: &ShapeKey) -> Result<MethodCall> {
        let ShapeKey::Prompt { rows, top } = *key else {
            return Err(Error::Invariant("a prompt scoring has a prompt key".into()));
        };
        let dump = self.buffers.dimensions.dump_words();
        let mut arguments: Vec<Argument> = vec![rows.into(), top.into(), dump.into()];
        arguments.extend(self.tensors(&[
            "hidden",
            "prompt_rows",
            "prompt_block",
            "input_ids",
            "logprobs",
            "prompt_tail",
        ])?);
        Ok(self.methods.call(&self.functions.score_prompt, arguments))
    }

    /// The runner's `verify` of a verification forward's tasks, which
    /// projects every candidate position and selects eagerly.
    fn verify_call(&self, plan: &Plan, metadata: &SamplingMetadata) -> Result<MethodCall> {
        let (reported, top) = metadata.logprob_bucket;
        let mut arguments: Vec<Argument> = vec![
            plan.logit_count.into(),
            metadata.tasks.into(),
            metadata.width.into(),
            candidates(self.buffers.dimensions.vocab).into(),
            metadata.sorts.into(),
            top.into(),
            reported.into(),
            self.buffers.dimensions.dump_words().into(),
        ];
        arguments.extend(self.tensors(&[
            "hidden",
            "logit_rows",
            "select_rows",
            "draws",
            "params",
            "tasks",
            "select_masks",
            "select_mask_rows",
            "select_biased",
            "select_force",
            "select_finish",
            "select_transitions",
            "select_drafts",
            "select_terminal",
            "logprob_tasks",
            "logprob_counts",
            "completion",
            "continuation",
            "tagged",
            "points",
            "transition",
            "scored",
            "logprobs",
        ])?);
        Ok(self.methods.call(&self.functions.verify, arguments))
    }

    /// Record in `job` the launch of `key`'s resident graph, or, without a
    /// pool, `call` eagerly. Selections are captured at startup
    /// ([`Self::capture_alone`]), so a key without a graph is an error.
    fn alone(&self, key: &ShapeKey, call: MethodCall, job: &Job) -> Result<()> {
        if self.pool.is_none() {
            job.call(call);
            return Ok(());
        }
        match self.graphs.lookup(key, None)? {
            Lookup::Replay(graph) => job.on(self.stream.as_ref()).launch(graph),
            Lookup::Capture => Err(Error::Invariant(format!(
                "the {key:?} graph is captured at startup"
            ))),
        }
    }

    /// Record in `job` the eager run of `call`, then its capture, which the
    /// job's completion carries ([`Self::store_alone`]); `capture` is the
    /// same call.
    pub(crate) fn capture_alone(
        &self,
        call: MethodCall,
        capture: MethodCall,
        job: &Job,
    ) -> Result<()> {
        job.call(call);
        job.capture(self.capture(capture)?);
        Ok(())
    }

    /// Store the graph of `key` the completed job of [`Self::capture_alone`]
    /// carries.
    pub(crate) fn store_alone(&mut self, key: ShapeKey, completion: &mut Completion) -> Result<()> {
        self.graphs.insert(key, None, completion.captured()?)
    }

    /// Record a staged forward at `capacity` in `job`: the launch of its
    /// resident graph, or the eager call of its method. Where the eager
    /// call is the warmup of a capture, the capture is recorded in `then`,
    /// which runs behind the forward's selection ([`Self::select`]): at the
    /// end of the job, or after the forwards of every microbatch of an
    /// expert step. Without `launch`, a resident graph's shapes run eagerly
    /// instead.
    ///
    /// Returns how the forward runs. An eager forward enters Python for the
    /// model's forward, which passes the host turn at its expert dispatch
    /// points; a launched one submits its whole step at once.
    pub(crate) fn forward(
        &self,
        staged: &Staged,
        capacity: Option<usize>,
        launch: bool,
        job: &Job,
        then: &Job,
    ) -> Result<GraphMode> {
        let (mode, graph) = match &staged.plan.choice {
            Choice::Eager => (GraphMode::Eager, None),
            Choice::Graph(key) => match self.graphs.lookup(key, capacity)? {
                Lookup::Replay(graph) if launch => (GraphMode::Replay, Some(graph)),
                Lookup::Replay(_) => (GraphMode::Eager, None),
                // The eager run is the capture's warmup and produces this
                // forward's results; capture records without executing.
                Lookup::Capture => (GraphMode::Capture, None),
            },
        };
        match graph {
            Some(graph) => {
                if self.plans_attention {
                    job.call(self.plan_attention(&staged.plan)?);
                }
                job.on(self.stream.as_ref()).launch(graph)?;
            }
            None => job.call(self.method(staged, capacity)?),
        }

        if mode == GraphMode::Capture {
            // The plans a capture reads are built outside it. Capture
            // submits no device work and no communication.
            then.call(self.plan_attention(&staged.plan)?);
            then.capture(self.capture(self.method(staged, capacity)?)?);
        }
        Ok(mode)
    }

    /// Record in `job` the selection of a staged forward's sampled tasks
    /// and the scoring its reporting tasks and prompt chunks ask, which
    /// read the logits and hidden states the forward left: the launches of
    /// the runner's `select` and `score` graphs and of its `score_prompt`
    /// graph once per block of prompt rows, or their eager calls without a
    /// pool. A verification projects and selects eagerly, which graphs
    /// refuse.
    pub(crate) fn select(
        &self,
        staged: &Staged,
        request: &StepRequest<'_>,
        job: &Job,
    ) -> Result<()> {
        let plan = &staged.plan;
        if let Some(metadata) = request.sampling {
            if plan.mode == ForwardMode::Verify {
                if self.pool.is_some() {
                    return Err(Error::Unsupported(
                        "a verification's selection has no captured graph".into(),
                    ));
                }
                job.call(self.verify_call(plan, metadata)?);
            } else {
                let key = self.selection_key(metadata);
                self.alone(&key, self.select_call(&key)?, job)?;
                if metadata.logprobs.is_some() {
                    let (rows, top) = metadata.logprob_bucket;
                    let key = ShapeKey::Score { rows, top };
                    self.alone(&key, self.score_call(&key)?, job)?;
                }
            }
        }

        // The prompt rows score in calls of one block each, the call's first
        // row written to the block column before it.
        if !plan.prompt.is_empty() {
            let top = request.prompt.top;
            let rows = self.buffers.dimensions.prompt_block(top);
            let key = ShapeKey::Prompt { rows, top };
            let block = self.buffers.column("prompt_block")?.address();
            for first in (0..plan.prompt.len()).step_by(rows) {
                // SAFETY: the binding's block column, ordered on its stream
                // before the call that reads it.
                unsafe {
                    job.on(self.stream.as_ref()).fill(Fill {
                        address: block,
                        value: first as u32,
                        width: 4,
                        count: 1,
                    })
                }?;
                self.alone(&key, self.score_prompt_call(&key)?, job)?;
            }
        }
        Ok(())
    }

    /// The capture of whatever `method` enqueues on the runner's stream as
    /// one graph in the runner's pool.
    fn capture(&self, method: MethodCall) -> Result<Capture> {
        let (Some(pool), Some(stream), Some(device)) = (&self.pool, &self.stream, self.device)
        else {
            return Err(Error::State(
                "graph capture requires a graph pool on a CUDA stream",
            ));
        };
        Ok(Capture {
            functions: self.methods.graphs.clone(),
            pool: pool.clone(),
            peers: Vec::new(),
            stream: stream.raw(),
            device,
            method,
            breaks: self.breaks.clone(),
        })
    }

    /// Apply the completion of a forward that ran in `mode` at `capacity`:
    /// store the graph its job captured, and count the mode.
    pub(crate) fn complete(
        &mut self,
        staged: Staged,
        capacity: Option<usize>,
        mode: GraphMode,
        completion: &mut Completion,
    ) -> Result<Executed> {
        let plan = &staged.plan;
        if mode == GraphMode::Capture
            && let Choice::Graph(key) = &plan.choice
        {
            self.graphs
                .insert(key.clone(), capacity, completion.captured()?)?;
        }
        Ok(Executed {
            mode,
            live_tokens: plan.live_tokens,
            padded_tokens: plan.tokens - plan.live_tokens,
        })
    }

    /// Resolve pages, the graph choice and the host columns of `request`.
    fn plan(&self, request: &StepRequest<'_>, context: &mut StepContext<'_>) -> Result<Plan> {
        let rows = request.rows;
        let decode = request.mode == ForwardMode::Decode;
        let dimensions = self.buffers.dimensions;
        let live = rows.len();
        if live == 0 || live > dimensions.rows {
            return Err(Error::Invalid(format!(
                "text invocation has {live} rows; the binding holds {}",
                dimensions.rows
            )));
        }
        for row in rows {
            row.validate()?;
        }
        let live_tokens: usize = rows.iter().map(|row| row.query as usize).sum();
        if live_tokens > dimensions.tokens {
            return Err(Error::Invalid(format!(
                "text invocation has {live_tokens} tokens; the binding holds {}",
                dimensions.tokens
            )));
        }
        if decode && rows.iter().any(|row| row.query != 1) {
            return Err(Error::Invalid(
                "decode rows evaluate exactly one token".into(),
            ));
        }

        // Pages each row selects, with every write checked against live
        // exports, imports and executions of its units.
        let attention: Vec<AttentionRow> = rows
            .iter()
            .map(|row| AttentionRow {
                slot: row.slot,
                prefix: row.prefix,
                query: row.query,
                write: true,
            })
            .collect();
        let pages = context.prefix_cache.prepare_attention(&attention)?;
        if pages.len() != self.tables.len() {
            return Err(Error::Invariant(
                "attention pages disagree with the binding's tables".into(),
            ));
        }

        let cache_only = rows
            .iter()
            .all(|row| row.selection == TokenSelection::Cache);
        let last_logits = rows
            .iter()
            .all(|row| row.selection == TokenSelection::LastLogits);

        // Multimodal prompts (explicit positions, vision features or
        // non-causal blocks) run eagerly.
        let placed = rows
            .iter()
            .any(|row| matches!(row.source, RowSource::Placed { .. }));
        let multimodal =
            placed || !request.embeddings.is_empty() || rows.iter().any(|row| !row.causal);
        let spatial = rows
            .iter()
            .any(|row| matches!(row.source, RowSource::Placed { spatial: true, .. }));
        let axes = if spatial { 3 } else { 1 };
        // A prompt placing vision blocks takes the variant whose graphs read
        // each row's causality and the embedding columns on the device;
        // causal text takes the text variant.
        let variant = if multimodal {
            PrefillVariant::media(axes)
        } else {
            PrefillVariant::TEXT
        };

        // Graph selection depends only on call metadata, so every rank of a
        // group selects the same graph. Under graphs the catalogs hold every
        // batch the worker admits, so a forward no captured bucket holds is
        // refused; without graphs every forward runs eagerly at its live
        // shape.
        let mut choice = Choice::Eager;
        let mut bucket = (live, live_tokens);
        if self.pool.is_some() {
            let queries: Vec<usize> = rows.iter().map(|row| row.query as usize).collect();
            let selected = self.shapes.select(&TextForward {
                queries: &queries,
                tokens: live_tokens,
                decode,
                variant,
                last_logits,
                cache_only,
            })?;
            let Some((bucket_rows, bucket_tokens, decode, outputs)) = selected else {
                return Err(Error::Invalid(format!(
                    "no captured text graph holds a {:?} forward of {live} rows and \
                     {live_tokens} tokens ({variant:?})",
                    request.mode
                )));
            };
            choice = Choice::Graph(if decode {
                ShapeKey::Decode {
                    rows: bucket_rows,
                    sampled: request.row_sampling.is_some(),
                }
            } else {
                ShapeKey::Prefill {
                    rows: bucket_rows,
                    tokens: bucket_tokens,
                    outputs,
                    variant,
                }
            });
            bucket = (bucket_rows, bucket_tokens);
        }
        let graphed = !matches!(choice, Choice::Eager);
        let (padded_rows, padded_tokens) = bucket;
        if padded_rows > dimensions.rows || padded_tokens > dimensions.tokens {
            return Err(Error::Invariant(
                "graph bucket exceeds the binding's columns".into(),
            ));
        }

        // Graphs address the configured widths; an eager call its own.
        let mut widths = Vec::with_capacity(pages.len());
        for (table, selected) in pages.iter().enumerate() {
            let live_width = selected.width();
            let width = if graphed {
                let configured = self.widths[table] as usize;
                if live_width > configured {
                    return Err(Error::Invalid(format!(
                        "table {table} selects {live_width} pages; its graph width is {configured}"
                    )));
                }
                configured
            } else if decode {
                live_width.next_power_of_two().min(dimensions.width)
            } else {
                live_width
            };
            if width > dimensions.width {
                return Err(Error::Invalid(format!(
                    "table {table} selects {width} pages; the binding holds {}",
                    dimensions.width
                )));
            }
            widths.push(width as i64);
        }

        // Host lengths with padding: decode buckets add inert one-token
        // rows; prefill buckets one sequence holding all padding tokens,
        // then empty sequences.
        let mut queries: Vec<i64> = rows.iter().map(|row| i64::from(row.query)).collect();
        let mut prefixes: Vec<i64> = rows.iter().map(|row| i64::from(row.prefix)).collect();
        let mut slots: Vec<i64> = rows.iter().map(|row| i64::from(row.slot)).collect();
        if padded_rows > live {
            if decode {
                queries.resize(padded_rows, 1);
            } else {
                queries.push((padded_tokens - live_tokens) as i64);
                queries.resize(padded_rows, 0);
            }
            prefixes.resize(padded_rows, 0);
            slots.resize(padded_rows, 0);
        } else if padded_tokens > live_tokens {
            return Err(Error::Invariant(
                "token padding requires an additional sequence".into(),
            ));
        }
        if queries.iter().sum::<i64>() as usize != padded_tokens {
            return Err(Error::Invariant(
                "padded query lengths disagree with the graph bucket".into(),
            ));
        }
        let mut starts = vec![0_i64; pages.len() * padded_rows];
        for (table, selected) in pages.iter().enumerate() {
            for (row, &start) in selected.starts.iter().enumerate() {
                starts[table * padded_rows + row] = i64::from(start);
            }
        }
        let mut causal: Vec<i64> = rows.iter().map(|row| i64::from(row.causal)).collect();
        causal.resize(padded_rows, 1);
        let gated_all = rows.iter().all(|row| row.gated);
        let mut force: Vec<u8> = rows
            .iter()
            .map(|row| u8::from(gated_all && row.force_finish))
            .collect();
        force.resize(padded_rows, 0);

        // A decode reads its rows from the slot state: a row whose token or
        // position comes from the host has it written to its slot. A decode
        // that samples its rows reads each row's draw and parameters; its
        // padding rows select greedily.
        let mut finish_ids = Vec::new();
        let mut slot_words = Vec::new();
        let (mut sampled_candidates, mut draws, mut params) = (0, Vec::new(), Vec::new());
        if let Some(sampled) = request.row_sampling.filter(|_| decode) {
            sampled_candidates = candidates(dimensions.vocab);
            (draws, params) = sampled.padded(padded_rows);
        }
        if decode {
            finish_ids = vec![-1_i64; padded_rows * FINISH_IDS];
            for (row, ids) in request.finish_ids.iter().enumerate() {
                finish_ids[row * FINISH_IDS..][..FINISH_IDS].copy_from_slice(ids);
            }
            for row in rows {
                let slot = row.slot as usize;
                match &row.source {
                    RowSource::Indexed => {}
                    RowSource::Host { tokens, position } => {
                        slot_words.push((context.state.future_token(slot)?, i64::from(tokens[0])));
                        slot_words.push((context.state.logical_length(slot)?, *position as i64));
                    }
                    RowSource::Relay { position, .. } => {
                        slot_words.push((context.state.logical_length(slot)?, *position as i64));
                    }
                    RowSource::Placed { .. } => {
                        return Err(Error::Invalid("a decode row places no positions".into()));
                    }
                }
            }
        }

        // The row gather's host columns: per-row sections of stride R + 1,
        // then each token's row ([`pack_rows`]).
        let mut row_columns = Vec::new();
        let mut token_ids = Vec::new();
        let mut positions = Vec::new();
        let mut relays = Vec::new();
        if !decode {
            let padded: Vec<AttentionRow> = (0..padded_rows)
                .map(|row| AttentionRow {
                    slot: slots[row] as u32,
                    prefix: prefixes[row] as u32,
                    query: queries[row] as u32,
                    // Live rows write the cache; padding rows do not.
                    write: row < live,
                })
                .collect();
            row_columns = pack_rows(&padded, &pages, dimensions.rows + 1, padded_tokens);

            token_ids = vec![0_i64; padded_tokens];
            // Axis-major: axis `a` of token `t` at `a * padded_tokens + t`.
            positions = vec![0_i64; axes * padded_tokens];
            let mut offset = 0;
            for row in rows {
                match &row.source {
                    RowSource::Host { tokens, position } => {
                        for (index, &token) in tokens.iter().enumerate() {
                            token_ids[offset + index] = i64::from(token);
                            positions[offset + index] = (position + index as u64) as i64;
                        }
                    }
                    RowSource::Relay { drafts, position } => {
                        relays.push((offset, row.slot));
                        for index in 0..=drafts.len() {
                            if index > 0 {
                                token_ids[offset + index] = i64::from(drafts[index - 1]);
                            }
                            positions[offset + index] = (position + index as u64) as i64;
                        }
                    }
                    RowSource::Indexed => {
                        return Err(Error::Invalid(
                            "only a decode row reads its position from its slot".into(),
                        ));
                    }
                    RowSource::Placed {
                        tokens,
                        positions: placed,
                        ..
                    } => {
                        for (index, &token) in tokens.iter().enumerate() {
                            token_ids[offset + index] = i64::from(token);
                            for (axis, values) in placed.iter().take(axes).enumerate() {
                                positions[axis * padded_tokens + offset + index] = values[index];
                            }
                        }
                    }
                }
                offset += row.query as usize;
            }
        }

        // The token positions the eager selection projects into candidate
        // rows: every position of a verification row, and of a prefill row
        // that scores its prompt.
        let mut logit_rows = Vec::new();
        if request.mode == ForwardMode::Verify || !request.prompt.rows.is_empty() {
            let mut offset = 0;
            for row in rows {
                let stop = offset + row.query as usize;
                logit_rows.extend((stop - row.logit_rows()..stop).map(|position| position as i64));
                offset = stop;
            }
        }

        let prompt = prompt_rows(
            request.prompt,
            &logit_rows,
            self.buffers.dimensions.rows as i64,
        )?;

        // A forward that only writes the cache keeps no outputs, unless the
        // graph serving it computes them.
        let outputs = !cache_only
            || matches!(
                choice,
                Choice::Graph(ShapeKey::Prefill { outputs: true, .. })
            );

        // A graph of the media variant reads the device columns whatever
        // its rows hold; an eager forward reads them for rows that differ in
        // causality, and for vision features.
        let media_graph = graphed && variant.embeddings;
        let device_causality = media_graph || causal.iter().any(|&value| value != causal[0]);
        let embeddings = media_graph || !request.embeddings.is_empty();

        Ok(Plan {
            mode: request.mode,
            rows: padded_rows,
            live_tokens,
            tokens: padded_tokens,
            widths,
            queries,
            prefixes,
            starts,
            causal,
            device_causality,
            embeddings,
            axes,
            logit_count: logit_rows.len(),
            outputs,
            choice,
            row_columns,
            token_ids,
            positions,
            slots,
            force,
            finish_ids,
            candidates: sampled_candidates,
            draws,
            params,
            logit_rows,
            prompt,
            relays,
            slot_words,
        })
    }

    /// Record the copies of the plan's host columns and sampling inputs to
    /// the runner's columns, then the relays of slot-state tokens, in stream
    /// order.
    fn copy_columns(
        &self,
        plan: &Plan,
        request: &StepRequest<'_>,
        state: &DecodeState,
        job: &Job,
    ) -> Result<()> {
        let column = |name: &str| self.buffers.column(name).map(StorageView::address);
        let mut entries: Vec<(u64, u64, usize)> = Vec::new();
        let mut push = |destination: u64, source: &[u8]| {
            entries.push((destination, source.as_ptr() as u64, source.len()));
        };
        let mut causal_values = Vec::new();
        if plan.mode == ForwardMode::Decode {
            push(
                column("request_pool_indices")?,
                bytemuck::cast_slice(&plan.slots),
            );
            push(column("decode_force_finish")?, &plan.force);
            push(
                column("finish_ids")?,
                bytemuck::cast_slice(&plan.finish_ids),
            );
            if plan.candidates > 0 {
                push(column("draws")?, bytemuck::cast_slice(&plan.draws));
                push(column("params")?, bytemuck::cast_slice(&plan.params));
            }
            for (address, value) in &plan.slot_words {
                push(*address, bytemuck::cast_slice(std::slice::from_ref(value)));
            }
        } else {
            push(
                column("row_columns")?,
                bytemuck::cast_slice(&plan.row_columns),
            );
            push(column("input_ids")?, bytemuck::cast_slice(&plan.token_ids));
            // Each axis is one row of the `[3, T]` column.
            let axis_words = plan.positions.len() / plan.axes.max(1);
            let stride = 8 * self.buffers.dimensions.tokens as u64;
            for (axis, values) in plan.positions.chunks(axis_words.max(1)).enumerate() {
                push(
                    column("positions")? + axis as u64 * stride,
                    bytemuck::cast_slice(values),
                );
            }
            if plan.device_causality {
                causal_values = plan.causal.iter().map(|&value| value as i32).collect();
            }
        }
        if !causal_values.is_empty() {
            push(
                column("causal_values")?,
                bytemuck::cast_slice(&causal_values),
            );
        }
        if !plan.logit_rows.is_empty() {
            push(
                column("logit_rows")?,
                bytemuck::cast_slice(&plan.logit_rows),
            );
        }
        // The selection's inputs, padded to its task bucket: padding tasks
        // select greedily over row 0 in slot 0, and their results are not
        // read. Token sets start empty; their set words follow.
        let selection = request
            .sampling
            .map(|metadata| Selection::pad(metadata, self.buffers.dimensions.rows));
        let sets = match &selection {
            Some(selection) => self.push_selection(selection, request.sampling, &mut push)?,
            None => Vec::new(),
        };
        // The prompt rows, padded to whole calls with rows that score and
        // copy into the spare words and tail row, field by field.
        let mut prompt_fields: [Vec<i64>; 8] = Default::default();
        if !plan.prompt.is_empty() {
            let dimensions = &self.buffers.dimensions;
            let block = dimensions.prompt_block(request.prompt.top);
            let calls = plan.prompt.len().div_ceil(block);
            let padding = self.prompt_padding();
            for row in plan.prompt.iter().chain(std::iter::repeat_n(
                &padding,
                calls * block - plan.prompt.len(),
            )) {
                for (field, &value) in prompt_fields.iter_mut().zip(row) {
                    field.push(value);
                }
            }
            let stride = 8 * dimensions.prompt_rows() as u64;
            for (index, field) in prompt_fields.iter().enumerate() {
                push(
                    column("prompt_rows")? + index as u64 * stride,
                    bytemuck::cast_slice(field),
                );
            }
        }

        // Vision features replace their tokens' embeddings: the mask is
        // cleared over the invocation, then set over each block.
        let mut features = Vec::new();
        let mut masks = Vec::new();
        if plan.embeddings {
            let values = self.buffers.column("input_embeddings")?;
            let row_bytes = values.nbytes() / self.buffers.dimensions.tokens.max(1) as u64;
            let mask = self.buffers.column("embedding_mask")?.address();
            masks.push((mask, 0_u8, plan.tokens));
            for block in request.embeddings {
                let count = block.values.shape().first().copied().unwrap_or(0) as usize;
                if block.values.dtype() != values.dtype()
                    || block.values.nbytes() != count as u64 * row_bytes
                    || block.token + count > plan.live_tokens
                {
                    return Err(Error::Invalid(
                        "vision features disagree with the embedding column".into(),
                    ));
                }
                features.push((
                    values.address() + block.token as u64 * row_bytes,
                    block.values.address(),
                    block.values.nbytes() as usize,
                ));
                masks.push((mask + block.token as u64, 1, count));
            }
        }

        let mut relays = Vec::new();
        for &(offset, slot) in &plan.relays {
            relays.push((
                column("input_ids")? + 8 * offset as u64,
                state.future_token(slot as usize)?,
                8,
            ));
        }

        let stream = job.on(self.stream.as_ref());
        for &(address, count) in &sets {
            // SAFETY: a token set column of this binding, ordered on its
            // stream before the set words the batch below writes.
            unsafe {
                stream.fill(Fill {
                    address,
                    value: 0,
                    width: 4,
                    count,
                })
            }?;
        }
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        for &(destination, source, size) in &entries {
            batch.push(destination, source, size);
        }
        // SAFETY: destinations are distinct columns of this binding, ordered
        // on its stream after earlier readers; sources are host vectors read
        // during the call.
        unsafe { stream.copy(&batch) }?;

        let mut device = CopyBatch::new(SourceOrder::Stream);
        for &(destination, source, size) in &relays {
            device.push(destination, source, size);
        }
        // SAFETY: each destination is one token or position of this
        // invocation, written after the host columns above; sources are slot
        // state committed earlier on this stream.
        unsafe { stream.copy(&device) }?;

        for &(address, value, count) in &masks {
            // SAFETY: bytes of the binding's mask column, ordered on its
            // stream.
            unsafe {
                stream.fill(Fill {
                    address,
                    value: u32::from(value),
                    width: 1,
                    count,
                })
            }?;
        }

        let mut leased = CopyBatch::new(SourceOrder::Stream);
        for &(destination, source, size) in &features {
            leased.push(destination, source, size);
        }
        // SAFETY: rows of the binding's embedding column; the features are
        // leased products whose producers this stream waited on.
        unsafe { stream.copy(&leased) }
    }

    /// Push the column writes of `selection`, and of `metadata`'s token
    /// sets; returns the set regions, `(address, words)`, which are cleared
    /// before the set words are written.
    fn push_selection(
        &self,
        selection: &Selection,
        metadata: Option<&SamplingMetadata>,
        push: &mut dyn FnMut(u64, &[u8]),
    ) -> Result<Vec<(u64, usize)>> {
        let column = |name: &str| self.buffers.column(name).map(StorageView::address);
        let rows = self.buffers.dimensions.rows as u64;
        push(column("draws")?, bytemuck::cast_slice(&selection.draws));
        push(column("params")?, bytemuck::cast_slice(&selection.params));
        push(column("tasks")?, bytemuck::cast_slice(&selection.slots));
        push(
            column("tasks")? + 8 * rows,
            bytemuck::cast_slice(&selection.gates),
        );
        push(
            column("select_rows")?,
            bytemuck::cast_slice(&selection.rows),
        );
        push(
            column("select_paths")?,
            bytemuck::cast_slice(&selection.paths),
        );
        push(
            column("select_mask_rows")?,
            bytemuck::cast_slice(&selection.mask_rows),
        );
        push(column("select_biased")?, &selection.biased);
        push(column("select_force")?, &selection.force);
        push(
            column("select_drafts")?,
            bytemuck::cast_slice(&selection.drafts),
        );
        push(
            column("select_terminal")?,
            bytemuck::cast_slice(&selection.terminal),
        );
        push(
            column("logprob_tasks")?,
            bytemuck::cast_slice(&selection.logprob_tasks),
        );
        push(
            column("logprob_counts")?,
            bytemuck::cast_slice(&selection.logprob_counts),
        );
        let mut sets = Vec::new();
        if let Some(metadata) = metadata {
            push(
                column("select_masks")?,
                bytemuck::cast_slice(&metadata.masks),
            );
            let words = bitmask_words(self.buffers.dimensions.vocab);
            for (name, values) in [
                ("select_finish", &metadata.finish),
                ("select_transitions", &metadata.transitions),
            ] {
                let base = column(name)?;
                sets.push((base, selection.tasks * words));
                for (index, word) in values.iter().enumerate() {
                    if *word != 0 {
                        push(
                            base + 4 * index as u64,
                            bytemuck::cast_slice(std::slice::from_ref(word)),
                        );
                    }
                }
            }
        }
        Ok(sets)
    }

    /// A prompt row that pads a forward's rows to whole calls: it reports
    /// nothing and copies into the spare tail row.
    fn prompt_padding(&self) -> [i64; 8] {
        [0, -1, 0, 0, 0, 0, -1, self.buffers.dimensions.rows as i64]
    }

    /// Record in `job` the inputs of a selection whose every task pads (as
    /// [`Selection::pad`] pads), and of prompt rows that all pad, over
    /// cleared logits, token sets, scored rows and hidden states.
    /// Startup captures the selection and scoring graphs over them: the
    /// values do not choose their kernels.
    pub(crate) fn pad_selection(&self, job: &Job) -> Result<()> {
        let selection = Selection::padding(self.buffers.dimensions.rows);
        let mut entries: Vec<(u64, u64, usize)> = Vec::new();
        self.push_selection(&selection, None, &mut |destination, source| {
            entries.push((destination, source.as_ptr() as u64, source.len()));
        })?;
        let rows = self.buffers.dimensions.prompt_rows();
        let fields: Vec<Vec<i64>> = self
            .prompt_padding()
            .iter()
            .map(|&value| vec![value; rows])
            .collect();
        let prompt = self.buffers.column("prompt_rows")?.address();
        for (index, field) in fields.iter().enumerate() {
            let source: &[u8] = bytemuck::cast_slice(field);
            entries.push((
                prompt + (8 * rows * index) as u64,
                source.as_ptr() as u64,
                source.len(),
            ));
        }
        let stream = job.on(self.stream.as_ref());
        for name in [
            "logits",
            "select_masks",
            "select_finish",
            "select_transitions",
            "scored",
            "hidden",
            "prompt_block",
        ] {
            let view = self.buffers.column(name)?;
            // SAFETY: a column of this binding, ordered on its stream before
            // the batch below.
            unsafe {
                stream.fill(Fill {
                    address: view.address(),
                    value: 0,
                    width: 1,
                    count: view.nbytes() as usize,
                })
            }?;
        }
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        for &(destination, source, size) in &entries {
            batch.push(destination, source, size);
        }
        // SAFETY: destinations are distinct columns of this binding, ordered
        // on its stream; sources are host vectors read during the call.
        unsafe { stream.copy(&batch) }
    }

    /// The commit bucket of `rows` rows: the smallest decode bucket holding
    /// them, else the binding's rows, which a batch of more calls than a
    /// decode batch holds commits at.
    fn commit_bucket(&self, rows: usize) -> usize {
        bucket(self.shapes.decode(), rows).unwrap_or(self.buffers.dimensions.rows)
    }

    /// Record the installation of `rows` commit rows (`[slot, mode, task,
    /// cache, logical, sampling, visible, penalties]` each) into the slot
    /// state: the launch of the commit graph of their bucket
    /// ([`Self::commit_bucket`]), or the runner's `commit` without a pool.
    ///
    /// Before the runner is sealed, a bucket without a graph commits
    /// eagerly and captures it: the bucket's key is returned, and the job's
    /// completion carries the graph for [`Self::complete_commit`].
    pub(crate) fn commit(&self, rows: &[[i64; 8]], job: &Job) -> Result<Option<ShapeKey>> {
        if rows.is_empty() {
            return Ok(None);
        }
        let capacity = self.buffers.dimensions.rows;
        if rows.len() > capacity {
            return Err(Error::Invariant("commit rows exceed the binding".into()));
        }
        let key = self.pool.as_ref().map(|_| ShapeKey::Commit {
            rows: self.commit_bucket(rows.len()),
        });

        // A bucket pads with rows that store the padding slot's zero cache
        // length again.
        let count = match &key {
            Some(ShapeKey::Commit { rows }) => *rows,
            _ => rows.len(),
        };
        let padding = [0, COMMIT_CACHE, -1, 0, 0, 0, 0, 0];
        let base = self.buffers.column("commit")?.address();
        let fields: Vec<Vec<i64>> = (0..8)
            .map(|field| {
                rows.iter()
                    .chain(std::iter::repeat_n(&padding, count - rows.len()))
                    .map(|row| row[field])
                    .collect()
            })
            .collect();
        let stream = job.on(self.stream.as_ref());
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        for (field, values) in fields.iter().enumerate() {
            let source: &[u8] = bytemuck::cast_slice(values);
            batch.push(
                base + (8 * capacity * field) as u64,
                source.as_ptr() as u64,
                source.len(),
            );
        }
        // SAFETY: distinct rows of the commit column; sources are read
        // during the call.
        unsafe { stream.copy(&batch) }?;

        let method = || -> Result<MethodCall> {
            let mut arguments = vec![count.into()];
            arguments.extend(self.tensors(&["commit", "completion", "continuation", "points"])?);
            Ok(self.methods.call(&self.functions.commit, arguments))
        };
        let Some(key) = key else {
            job.call(method()?);
            return Ok(None);
        };
        if let Lookup::Replay(graph) = self.graphs.lookup(&key, None)? {
            stream.launch(graph)?;
            return Ok(None);
        }
        // The eager commit installs these rows and compiles the kernel;
        // the capture records without executing.
        job.call(method()?);
        job.capture(self.capture(method()?)?);
        Ok(Some(key))
    }

    /// Store the commit graph the job of [`Self::commit`] captured for
    /// `key`.
    pub(crate) fn complete_commit(
        &mut self,
        key: ShapeKey,
        completion: &mut Completion,
    ) -> Result<()> {
        self.graphs.insert(key, None, completion.captured()?)
    }

    /// Release the graphs. The caller drained the lane.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        self.graphs.close(aborted)
    }
}

/// The element type of a model dtype name.
pub(crate) fn model_dtype(name: &str) -> Result<DType> {
    let dtype = DType::from_name(name.trim_start_matches("torch."))
        .ok_or_else(|| Error::Unsupported(format!("unsupported model dtype {name:?}")))?;
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) {
        return Err(Error::Unsupported(format!(
            "text models compute in a floating dtype, not {name:?}"
        )));
    }
    Ok(dtype)
}

#[cfg(test)]
mod tests {
    use super::{PromptChunks, prompt_rows};
    use crate::Result;

    #[test]
    fn prompt_rows_predict_each_chunks_tokens_in_order() -> Result<()> {
        // Candidate rows 0-2 project positions 10-12, a first chunk of three
        // tokens from token 0; rows 3-4 project 20-21, a chunk from token 3
        // continued from slot 5's retained logits; rows 5-6 project 30-31,
        // a chunk from token 5 continued from candidate row 4 (the last row
        // of a vision block); row 7 projects 40, a chunk that only retains.
        // The chunks report 4, 2, 1 and no top tokens.
        let logit_rows = [10, 11, 12, 20, 21, 30, 31, 40];
        let chunks = PromptChunks {
            rows: vec![0, 3, 5, 7],
            row_counts: vec![3, 2, 2, 0],
            tokens: vec![0, 3, 5, 0],
            continued: vec![0, 1, 2 + 4, 0],
            slots: vec![1, 5, 1, 1],
            offsets: vec![100, 200, 300, 400],
            tails: vec![0, 1, 2, 3],
            tops: vec![4, 2, 1, 0],
            top: 4,
        };
        let spare = 64;

        let rows = prompt_rows(&chunks, &logit_rows, spare)?;

        // Per row: hidden position, retained slot, target token, chunk
        // words, index, scored rows, their top count and tail row.
        assert_eq!(
            rows,
            vec![
                // The first chunk's first two rows score its second and
                // third tokens; its last row retains its logits.
                [10, -1, 1, 100, 0, 2, 4, spare],
                [11, -1, 2, 100, 1, 2, 4, spare],
                [12, -1, 0, 0, 0, 0, -1, 0],
                // The slot's retained logits score the next chunk's first
                // token.
                [0, 5, 3, 200, 0, 2, 2, spare],
                [20, -1, 4, 200, 1, 2, 2, spare],
                [21, -1, 0, 0, 0, 0, -1, 1],
                // The vision block's last row scores the first token after
                // it.
                [21, -1, 5, 300, 0, 2, 1, spare],
                [30, -1, 6, 300, 1, 2, 1, spare],
                [31, -1, 0, 0, 0, 0, -1, 2],
                [40, -1, 0, 0, 0, 0, -1, 3],
            ]
        );

        // A chunk naming a row its forward does not project is refused.
        let mut beyond = chunks;
        beyond.rows[0] = 8;
        assert!(prompt_rows(&beyond, &logit_rows, spare).is_err());
        Ok(())
    }
}

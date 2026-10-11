//! Text forwards of a causal language model on one lane binding.
//!
//! A [`CausalLMRunner`] owns one binding's input and output tensors
//! ([`InputBuffers`]), its graph shapes and captured graphs, and the Python
//! runner ([`Methods`]). Per forward it:
//!
//! 1. resolves the rows' attention pages against the installed block tables
//!    (checking every write against the cache's live accesses);
//! 2. selects an eager run, a resident graph or, before the rank is sealed,
//!    a graph to capture;
//! 3. writes the forward's inputs on the lane stream in one batched copy;
//! 4. launches the graph, or calls the runner's forward method;
//! 5. calls the runner's eager selection for rows that sample, verify
//!    drafts or score their prompt.
//!
//! A decode forward evaluates one token per row from the slot state: its
//! inputs are each row's slot, forced-finish flag and finish tokens, and a
//! row whose token comes from the host has it written to its slot first.
//! Its graph gathers, runs the model and selects greedily, so a greedy
//! decode runs no Python. A decode bucket pads with rows of slot 0.
//!
//! A prefill forward evaluates rows of host tokens (prompt chunks and
//! verification rows): its inputs are the rows' packed columns, tokens and
//! positions. Its graph gathers, runs the model and, unless the forward
//! only writes the prefix cache, keeps the hidden states and projects each
//! row's last position. A prefill bucket adds one sequence holding every
//! padding token, then empty sequences.
//!
//! Padding rows never write K/V (their write addresses are -1).

pub(crate) mod input_buffers;
pub(crate) mod methods;
pub(crate) mod rows;

pub(crate) use input_buffers::{Dimensions, FINISH_IDS, InputBuffers};
pub(crate) use methods::{Construction, Methods};
pub(crate) use rows::{RowSource, TokenRow, logit_layout};

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, SourceOrder, Stream};
use uniserve_ipc::ForwardMode;

use super::{TextShapes, TokenSelection};
use crate::cuda_graph::{CUDAGraph, GraphBreaks, GraphMode, GraphTable, Lookup, bucket};
use crate::ffi::{DType, Handle};
use crate::parallel::MicrobatchCall;
use crate::request::decode_state::DecodeState;
use crate::storage::memory::StorageView;
use crate::storage::{AttentionRow, BlockTables, KVCacheManager};
use crate::transfer::Retirement;
use crate::{Error, Result};

use input_buffers::ROW_SECTIONS;
use methods::{Argument, MethodCall};

/// The commit mode of a row that only sets its slot's cache length
/// (`uniserve_models.sampling.state.CACHE`).
pub(crate) const COMMIT_CACHE: i64 = 3;

/// A numerical block table's page layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableShape {
    pub(crate) page_tokens: u32,
    pub(crate) windowed: bool,
}

/// What identifies a captured text graph besides the runner's fixed table
/// widths and tensors: the method and its scalar arguments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ShapeKey {
    /// `decode` of a row bucket.
    Decode { rows: usize },
    /// `prefill` of a row and token bucket, with outputs or writing only
    /// the prefix cache.
    Prefill {
        rows: usize,
        tokens: usize,
        outputs: bool,
    },
    /// `commit` of a row bucket.
    Commit { rows: usize },
}

/// The sampling inputs of one invocation (see `crate::sampling::plan`).
pub(crate) struct SamplingColumns<'a> {
    /// The host descriptor passed to the step.
    pub(crate) words: &'a [i64],
    /// One draw per candidate row.
    pub(crate) draws: &'a [f32],
    /// Eight parameters per candidate row.
    pub(crate) params: &'a [f32],
    /// Each task's slot, then each task's gate (1 when gated).
    pub(crate) slots: &'a [i64],
    pub(crate) gates: &'a [i64],
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
    /// The tasks the runner's `sample` selects; absent when no row samples
    /// or a decode forward's greedy selections are final.
    pub(crate) sampling: Option<SamplingColumns<'a>>,
    /// The prompt scoring descriptor; empty when no row scores its prompt.
    pub(crate) prompt: &'a [i64],
    /// Vision features placed over some of the rows' tokens.
    pub(crate) embeddings: &'a [Embeddings],
}

/// The storage a step borrows.
pub(crate) struct StepContext<'a> {
    pub(crate) tables: &'a BlockTables,
    pub(crate) cache: &'a mut KVCacheManager<Arc<Retirement>>,
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

/// A planned forward whose inputs are written on the runner's stream,
/// awaiting its expert step capacity.
pub(crate) struct Staged {
    plan: Plan,
    /// The sampling descriptor; empty when no row samples.
    sampling: Vec<i64>,
    prompt: Vec<i64>,
    /// Vision features replace some of the tokens' embeddings.
    embedded: bool,
}

impl Staged {
    /// Tokens the forward sends to each expert layer: its graph's padded
    /// tokens, or its live tokens when it runs eagerly.
    pub(crate) fn expert_tokens(&self) -> usize {
        self.plan.tokens
    }
}

/// A staged forward's numerical work at its expert capacity.
pub(crate) struct Launch {
    pub(crate) mode: GraphMode,
    /// The forward: a graph launch, or the runner method (also a capture's
    /// warmup), then `sample` for rows that sample. Run on the lane's
    /// execution thread, or on a microbatch thread in its turn of an expert
    /// step.
    pub(crate) call: MicrobatchCall<()>,
    /// The call enters Python for the model forward, which passes the host
    /// turn at its expert dispatch points.
    pub(crate) yields: bool,
}

/// One planned forward: its shapes and host columns.
struct Plan {
    mode: ForwardMode,
    live: usize,
    rows: usize,
    live_tokens: usize,
    tokens: usize,
    widths: Vec<i64>,
    queries: Vec<i64>,
    prefixes: Vec<i64>,
    starts: Vec<i64>,
    causal: Vec<i64>,
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
    logit_rows: Vec<i64>,
    /// `(token offset, slot)` device copies of a prefill row's first token
    /// from its slot's next input token.
    relays: Vec<(usize, u32)>,
    /// Slot state a decode row's host source supplies: `(address, value)`
    /// of a next input token or a logical position.
    slot_words: Vec<(u64, i64)>,
}

/// One lane binding of a causal language model.
pub(crate) struct CausalLMRunner {
    device: Option<i32>,
    stream: Option<Stream>,
    methods: Methods,
    /// An attention operator plans each launch on the host.
    plans_attention: bool,
    /// Where the distributed expert weights note the copies a captured
    /// forward performs between its segments.
    breaks: Option<GraphBreaks>,
    buffers: InputBuffers,
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
    pub(crate) device: Option<i32>,
    pub(crate) stream: Option<Stream>,
    pub(crate) methods: Methods,
    pub(crate) plans_attention: bool,
    pub(crate) breaks: Option<GraphBreaks>,
    pub(crate) buffers: InputBuffers,
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

    pub(crate) fn device(&self) -> Option<i32> {
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

    /// Plan `request` and write its inputs on the runner's stream.
    pub(crate) fn stage(
        &self,
        request: &StepRequest<'_>,
        context: &mut StepContext<'_>,
    ) -> Result<Staged> {
        let plan = self.plan(request, context)?;
        self.copy_columns(&plan, request, context.state)?;
        Ok(Staged {
            plan,
            sampling: request
                .sampling
                .as_ref()
                .map_or_else(Vec::new, |sampling| sampling.words.to_vec()),
            prompt: request.prompt.to_vec(),
            embedded: !request.embeddings.is_empty(),
        })
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
    fn forward(&self, staged: &Staged, capacity: Option<usize>) -> Result<MethodCall> {
        let plan = &staged.plan;
        let capacity = capacity.unwrap_or(0);
        if plan.mode == ForwardMode::Decode {
            let mut arguments = vec![
                plan.rows.into(),
                plan.widths.as_slice().into(),
                plan.prefixes.as_slice().into(),
                plan.starts.as_slice().into(),
                capacity.into(),
            ];
            arguments.extend(self.tensors(&[
                "request_pool_indices",
                "decode_force_finish",
                "finish_ids",
                "logits",
                "completion",
                "continuation",
                "tagged",
                "points",
                "transition",
            ])?);
            return Ok(self.methods.call(&self.methods.decode, arguments));
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
        arguments.extend(self.tensors(&[
            "row_columns",
            "input_ids",
            "positions",
            "causal_values",
        ])?);
        if staged.embedded {
            arguments.extend(self.tensors(&["input_embeddings", "embedding_mask"])?);
        } else {
            arguments.extend([Argument::None, Argument::None]);
        }
        arguments.extend(self.tensors(&["hidden", "logits"])?);
        Ok(self.methods.call(&self.methods.prefill, arguments))
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
            return Ok(self.methods.call(&self.methods.plan_decode, arguments));
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
        arguments.extend(self.tensors(&["row_columns", "causal_values"])?);
        Ok(self.methods.call(&self.methods.plan_prefill, arguments))
    }

    /// The eager methods that follow a staged forward: `sample` for rows
    /// that sample from the logits the forward projected (`verify` for a
    /// verification, which projects every candidate position), and
    /// `prompt_logprobs` for scored prompt chunks.
    fn selection(&self, staged: &Staged) -> Result<Vec<MethodCall>> {
        let plan = &staged.plan;
        let outputs = [
            "draws",
            "params",
            "tasks",
            "completion",
            "continuation",
            "tagged",
            "points",
            "transition",
            "logprobs",
        ];
        let mut calls = Vec::new();
        if !staged.sampling.is_empty() {
            let words = staged.sampling.as_slice();
            if plan.mode == ForwardMode::Verify {
                let mut arguments = vec![plan.logit_count.into(), words.into()];
                arguments.extend(self.tensors(&["hidden", "logit_rows"])?);
                arguments.extend(self.tensors(&outputs)?);
                calls.push(self.methods.call(&self.methods.verify, arguments));
            } else {
                let mut arguments = vec![plan.live.into(), words.into()];
                arguments.extend(self.tensors(&["logits"])?);
                arguments.extend(self.tensors(&outputs)?);
                calls.push(self.methods.call(&self.methods.sample, arguments));
            }
        }
        if !staged.prompt.is_empty() {
            let mut arguments = vec![plan.logit_count.into(), staged.prompt.as_slice().into()];
            arguments.extend(self.tensors(&[
                "hidden",
                "logit_rows",
                "input_ids",
                "logprobs",
                "prompt_tail",
            ])?);
            calls.push(self.methods.call(&self.methods.prompt_logprobs, arguments));
        }
        Ok(calls)
    }

    /// The numerical work of a staged forward at `capacity`: the launch of
    /// its resident graph, or the eager method (also the warmup of a
    /// capture [`Self::finish`] then records), followed by the eager
    /// selection of rows that sample or score. Without `launch`, a resident
    /// graph's shapes run eagerly instead.
    pub(crate) fn launch(
        &self,
        staged: &Staged,
        capacity: Option<usize>,
        launch: bool,
    ) -> Result<Launch> {
        let (mode, graph) = match &staged.plan.choice {
            Choice::Eager => (GraphMode::Eager, None),
            Choice::Graph(key) => match self.graphs.lookup(key, capacity)? {
                Lookup::Replay(graph) if launch => (GraphMode::Replay, Some(graph.clone())),
                Lookup::Replay(_) => (GraphMode::Eager, None),
                // The eager run is the capture's warmup and produces this
                // forward's results; capture records without executing.
                Lookup::Capture => (GraphMode::Capture, None),
            },
        };
        let selection = self.selection(staged)?;
        let select = move || {
            selection
                .into_iter()
                .try_for_each(|call| call.invoke().map(drop))
        };
        match graph {
            Some(graph) => {
                let stream = self
                    .stream
                    .clone()
                    .ok_or(Error::State("a graph is launched on a CUDA stream"))?;
                let planned = self
                    .plans_attention
                    .then(|| self.plan_attention(&staged.plan))
                    .transpose()?;
                Ok(Launch {
                    mode,
                    call: Box::new(move || {
                        if let Some(call) = planned {
                            call.invoke()?;
                        }
                        graph.launch(&stream)?;
                        select()
                    }),
                    yields: false,
                })
            }
            None => {
                let call = self.forward(staged, capacity)?;
                Ok(Launch {
                    mode,
                    call: Box::new(move || {
                        call.invoke()?;
                        select()
                    }),
                    yields: true,
                })
            }
        }
    }

    /// Capture whatever `method` enqueues on the runner's stream as one
    /// graph in the runner's pool.
    fn capture(&self, method: MethodCall) -> Result<CUDAGraph> {
        let (Some(pool), Some(stream), Some(device)) = (&self.pool, &self.stream, self.device)
        else {
            return Err(Error::State(
                "graph capture requires a graph pool on a CUDA stream",
            ));
        };
        let captured = self
            .methods
            .graphs
            .capture(pool, stream.raw(), device, || method.invoke().map(drop));

        // The breaks the forward reached belong to this capture, whether or
        // not it succeeded.
        let breaks = self
            .breaks
            .as_ref()
            .map_or_else(Vec::new, GraphBreaks::take);
        Ok(captured?.with_breaks(breaks))
    }

    /// Complete a forward whose call ran: capture its graph after the
    /// eager warmup when its launch chose capture, and count the mode.
    /// Capture submits no device work and no communication.
    pub(crate) fn finish(
        &mut self,
        staged: Staged,
        capacity: Option<usize>,
        mode: GraphMode,
    ) -> Result<Executed> {
        let plan = &staged.plan;
        if mode == GraphMode::Capture
            && let Choice::Graph(key) = &plan.choice
        {
            // The plans a capture reads are built outside it.
            self.plan_attention(plan)?.invoke()?;
            let graph = self.capture(self.forward(&staged, capacity)?)?;
            self.graphs.insert(key.clone(), capacity, graph)?;
        }
        self.graphs.record(mode);
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
        let cache = &mut *context.cache;
        let pages = context
            .tables
            .prepare_attention(&attention, |table, prefix, query| {
                cache.require_writable(&table.spans(u64::from(prefix), u64::from(query))?)
            })?;
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

        // Graph selection depends only on call metadata, so every rank of a
        // group selects the same graph.
        let mut choice = Choice::Eager;
        let mut bucket = (live, live_tokens);
        if self.pool.is_some() && !multimodal {
            let queries: Vec<usize> = rows.iter().map(|row| row.query as usize).collect();
            if let Some((bucket_rows, bucket_tokens, decode, outputs)) = self.shapes.select(
                &queries,
                live_tokens,
                decode,
                Some(true),
                false,
                last_logits,
                cache_only,
            )? {
                choice = Choice::Graph(if decode {
                    ShapeKey::Decode { rows: bucket_rows }
                } else {
                    ShapeKey::Prefill {
                        rows: bucket_rows,
                        tokens: bucket_tokens,
                        outputs,
                    }
                });
                bucket = (bucket_rows, bucket_tokens);
            }
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
        // position comes from the host has it written to its slot.
        let mut finish_ids = Vec::new();
        let mut slot_words = Vec::new();
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
        // then each token's row.
        let mut row_columns = Vec::new();
        let mut token_ids = Vec::new();
        let mut positions = Vec::new();
        let mut relays = Vec::new();
        if !decode {
            let stride = dimensions.rows + 1;
            row_columns = vec![0_i64; (ROW_SECTIONS + pages.len()) * stride + padded_tokens];
            let mut query_offset = 0;
            let mut prefix_offset = 0;
            for row in 0..padded_rows {
                row_columns[row] = slots[row];
                row_columns[stride + row] = prefixes[row];
                row_columns[2 * stride + row] = queries[row];
                // Live rows write the cache; padding rows do not.
                row_columns[3 * stride + row] = i64::from(row < live);
                row_columns[4 * stride + row] = query_offset;
                row_columns[5 * stride + row] = prefix_offset;
                for (table, selected) in pages.iter().enumerate() {
                    row_columns[(ROW_SECTIONS + table) * stride + row] =
                        selected.lengths.get(row).copied().unwrap_or(0) as i64;
                }
                let token_section = (ROW_SECTIONS + pages.len()) * stride;
                for token in 0..queries[row] as usize {
                    row_columns[token_section + query_offset as usize + token] = row as i64;
                }
                query_offset += queries[row];
                prefix_offset += prefixes[row];
            }
            row_columns[4 * stride + padded_rows] = query_offset;
            row_columns[5 * stride + padded_rows] = prefix_offset;

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
        if request.mode == ForwardMode::Verify || !request.prompt.is_empty() {
            let mut offset = 0;
            for row in rows {
                let stop = offset + row.query as usize;
                logit_rows.extend((stop - row.logit_rows()..stop).map(|position| position as i64));
                offset = stop;
            }
        }

        // A forward that only writes the cache keeps no outputs, unless the
        // graph serving it computes them.
        let outputs = !cache_only
            || matches!(
                choice,
                Choice::Graph(ShapeKey::Prefill { outputs: true, .. })
            );

        Ok(Plan {
            mode: request.mode,
            live,
            rows: padded_rows,
            live_tokens,
            tokens: padded_tokens,
            widths,
            queries,
            prefixes,
            starts,
            causal,
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
            logit_rows,
            relays,
            slot_words,
        })
    }

    /// Copy the plan's host columns and sampling inputs to the device, then
    /// relay slot-state tokens, in stream order.
    fn copy_columns(
        &self,
        plan: &Plan,
        request: &StepRequest<'_>,
        state: &DecodeState,
    ) -> Result<()> {
        let column = |name: &str| self.buffers.column(name).map(StorageView::address);
        let rows = self.buffers.dimensions.rows as u64;
        let mut entries: Vec<(u64, u64, usize)> = Vec::new();
        let mut push = |destination: u64, source: &[u8]| {
            entries.push((destination, source.as_ptr() as u64, source.len()));
        };
        let mut causal_values = Vec::new();
        if plan.mode == ForwardMode::Decode {
            push(column("request_pool_indices")?, bytes(&plan.slots));
            push(column("decode_force_finish")?, &plan.force);
            push(column("finish_ids")?, bytes(&plan.finish_ids));
            for (address, value) in &plan.slot_words {
                push(*address, bytes(std::slice::from_ref(value)));
            }
        } else {
            push(column("row_columns")?, bytes(&plan.row_columns));
            push(column("input_ids")?, bytes(&plan.token_ids));
            // Each axis is one row of the `[3, T]` column.
            let axis_words = plan.positions.len() / plan.axes.max(1);
            let stride = 8 * self.buffers.dimensions.tokens as u64;
            for (axis, values) in plan.positions.chunks(axis_words.max(1)).enumerate() {
                push(column("positions")? + axis as u64 * stride, bytes(values));
            }
            if plan.causal.iter().any(|&value| value != plan.causal[0]) {
                // Mixed causality is read from the device column.
                causal_values = plan.causal.iter().map(|&value| value as i32).collect();
            }
        }
        if !causal_values.is_empty() {
            push(column("causal_values")?, bytes_i32(&causal_values));
        }
        if !plan.logit_rows.is_empty() {
            push(column("logit_rows")?, bytes(&plan.logit_rows));
        }
        if let Some(sampling) = &request.sampling {
            push(column("draws")?, bytes_f32(sampling.draws));
            push(column("params")?, bytes_f32(sampling.params));
            push(column("tasks")?, bytes(sampling.slots));
            push(column("tasks")? + 8 * rows, bytes(sampling.gates));
        }

        // Vision features replace their tokens' embeddings: the mask is
        // cleared over the invocation, then set over each block.
        let mut features = Vec::new();
        let mut masks = Vec::new();
        if !request.embeddings.is_empty() {
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

        match &self.stream {
            Some(stream) => {
                let mut batch = CopyBatch::new(SourceOrder::DuringCall);
                for &(destination, source, size) in &entries {
                    batch.push(destination, source, size);
                }
                // SAFETY: destinations are distinct columns of this binding,
                // ordered on its stream after earlier readers; sources are
                // host vectors the driver reads during the call.
                unsafe { batch.submit(stream) }?;
                let mut device = CopyBatch::new(SourceOrder::Stream);
                for &(destination, source, size) in &relays {
                    device.push(destination, source, size);
                }
                // SAFETY: each destination is one token or position of this
                // invocation, written after the host columns above; sources
                // are slot state committed earlier on this stream.
                unsafe { device.submit(stream) }?;
                for &(address, value, count) in &masks {
                    // SAFETY: bytes of the binding's mask column, ordered on
                    // its stream.
                    unsafe { uniserve_cuda::memset_d8(address, value, count, stream) }?;
                }
                let mut leased = CopyBatch::new(SourceOrder::Stream);
                for &(destination, source, size) in &features {
                    leased.push(destination, source, size);
                }
                // SAFETY: rows of the binding's embedding column; the
                // features are leased products whose producers this stream
                // waited on.
                unsafe { leased.submit(stream) }?;
            }
            None => {
                for &(address, value, count) in &masks {
                    // SAFETY: bytes of a host column of this binding.
                    unsafe { std::ptr::write_bytes(address as *mut u8, value, count) };
                }
                for &(destination, source, size) in entries.iter().chain(&relays).chain(&features) {
                    // SAFETY: host columns and state owned by this rank; the
                    // executor thread serializes their writers and readers.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            source as *const u8,
                            destination as *mut u8,
                            size,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Install `rows` commit rows (`[slot, mode, task, cache, logical,
    /// sampling, visible, penalties]` each) into the slot state: the launch
    /// of the commit graph of the smallest decode bucket holding them, or
    /// the runner's `commit` where no graph serves.
    pub(crate) fn commit(&mut self, rows: &[[i64; 8]]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let capacity = self.buffers.dimensions.rows;
        if rows.len() > capacity {
            return Err(Error::Invariant("commit rows exceed the binding".into()));
        }
        let key = self
            .pool
            .as_ref()
            .and_then(|_| bucket(self.shapes.decode(), rows.len()))
            .map(|rows| ShapeKey::Commit { rows });

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
        match &self.stream {
            Some(stream) => {
                let mut batch = CopyBatch::new(SourceOrder::DuringCall);
                for (field, values) in fields.iter().enumerate() {
                    let source = bytes(values);
                    batch.push(
                        base + (8 * capacity * field) as u64,
                        source.as_ptr() as u64,
                        source.len(),
                    );
                }
                // SAFETY: distinct rows of the commit column; sources are
                // read during the call.
                unsafe { batch.submit(stream) }?;
            }
            None => {
                for (field, values) in fields.iter().enumerate() {
                    // SAFETY: a host column of this binding.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            values.as_ptr(),
                            (base + (8 * capacity * field) as u64) as *mut i64,
                            values.len(),
                        );
                    }
                }
            }
        }

        let method = || -> Result<MethodCall> {
            let mut arguments = vec![count.into()];
            arguments.extend(self.tensors(&["commit", "completion", "continuation", "points"])?);
            Ok(self.methods.call(&self.methods.commit, arguments))
        };
        let Some(key) = key else {
            return method()?.invoke().map(drop);
        };
        if let Lookup::Replay(graph) = self.graphs.lookup(&key, None)? {
            let stream = self
                .stream
                .as_ref()
                .ok_or(Error::State("a graph is launched on a CUDA stream"))?;
            return graph.launch(stream);
        }
        // The eager commit installs these rows and compiles the kernel;
        // the capture records without executing.
        method()?.invoke()?;
        let graph = self.capture(method()?)?;
        self.graphs.insert(key, None, graph)
    }

    /// Release the graphs. The caller drained the lane.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        self.graphs.close(aborted)
    }
}

fn bytes(values: &[i64]) -> &[u8] {
    // SAFETY: plain integers reinterpreted as their bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn bytes_i32(values: &[i32]) -> &[u8] {
    // SAFETY: plain integers reinterpreted as their bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn bytes_f32(values: &[f32]) -> &[u8] {
    // SAFETY: plain floats reinterpreted as their bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
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

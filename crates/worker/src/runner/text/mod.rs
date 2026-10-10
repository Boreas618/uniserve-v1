//! Text invocations of a causal language model on one lane binding.
//!
//! A [`TextRunner`] owns one binding's fixed columns ([`TextBuffers`]), its
//! graph shapes and captured graphs, and the typed Python steps
//! ([`TextModel`]). Per invocation it:
//!
//! 1. resolves the rows' attention pages against the installed block tables
//!    (checking every write against the cache's live accesses);
//! 2. selects an eager run, a resident graph (replay) or, before the rank is
//!    sealed, a graph to capture;
//! 3. packs the rows' host columns, including graph padding, and copies
//!    them to the device in one batched copy on the lane stream, followed by
//!    device copies of relayed tokens and positions from the slot state;
//! 4. calls the binding's `step` once.
//!
//! Rows are packed in order. Graph padding follows the captured shapes:
//! decode buckets pad with inert one-token rows of slot 0; prefill buckets
//! add one sequence holding every padding token, then empty sequences. Inert
//! rows never write K/V (their write addresses are -1).

pub(crate) mod buffers;
pub(crate) mod rows;

pub(crate) use buffers::{Dimensions, TextBuffers};
pub(crate) use rows::{RowSource, TokenRow, logit_layout};

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, SourceOrder, Stream};

use super::{TextShapes, TokenSelection};
use crate::cuda_graph::{GraphMode, GraphTable, Lookup};
use crate::ffi::{DType, DeferredCall, Handle};
use crate::model::text::{CaptureArgs, StepArgs, TextModel};
use crate::request::decode_state::DecodeState;
use crate::storage::memory::StorageView;
use crate::storage::{AttentionRow, BlockTables, KVCacheManager};
use crate::transfer::Retirement;
use crate::{Error, Result};

use buffers::ROW_SECTIONS;

/// A numerical block table's page layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableShape {
    pub(crate) page_tokens: u32,
    pub(crate) windowed: bool,
}

/// What a captured text graph depends on besides the runner's fixed table
/// widths: the decode or prefill body, the bucket and whether a prefill
/// family computes hidden outputs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TextGraphKey {
    pub(crate) decode: bool,
    pub(crate) rows: usize,
    pub(crate) tokens: usize,
    pub(crate) outputs: bool,
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
    /// A decode graph's greedy selections are final.
    pub(crate) greedy: bool,
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
    pub(crate) decode: bool,
    pub(crate) rows: &'a [TokenRow],
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
    Graph(TextGraphKey),
}

/// A planned invocation whose host columns are staged on the runner's
/// stream, awaiting its expert step capacity.
pub(crate) struct Staged {
    plan: Plan,
    sampling: Vec<i64>,
    greedy: bool,
    prompt: Vec<i64>,
    /// Vision features replace some of the tokens' embeddings.
    embedded: bool,
}

impl Staged {
    /// Tokens the invocation sends to each expert layer: its graph's padded
    /// tokens, or its live tokens when it runs eagerly.
    pub(crate) fn expert_tokens(&self) -> usize {
        self.plan.tokens
    }
}

/// A staged invocation's numerical call at its expert capacity.
pub(crate) struct Launch {
    pub(crate) mode: GraphMode,
    /// The step: an eager forward (a capture's warmup included) or a graph
    /// replay. Runs on the caller's thread or a microbatch thread.
    pub(crate) call: DeferredCall,
}

/// One planned invocation: its shapes and host columns.
struct Plan {
    decode: bool,
    live: usize,
    rows: usize,
    live_tokens: usize,
    tokens: usize,
    indexed: bool,
    widths: Vec<i64>,
    queries: Vec<i64>,
    prefixes: Vec<i64>,
    starts: Vec<i64>,
    causal: Vec<i64>,
    /// Rotary position axes the rows carry: 1, or 3 when a row places
    /// positions explicitly.
    axes: usize,
    logit_count: usize,
    all_last: bool,
    cache_only: bool,
    choice: Choice,
    /// Host column words of the row gather (absent for indexed decode).
    row_columns: Vec<i64>,
    token_ids: Vec<i64>,
    positions: Vec<i64>,
    slots: Vec<i64>,
    force: Vec<u8>,
    logit_rows: Vec<i64>,
    /// `(token offset, slot, position from state)` device copies.
    relays: Vec<(usize, u32, bool)>,
}

/// One lane binding of a causal language model.
pub(crate) struct TextRunner {
    device: Option<i32>,
    stream: Option<Stream>,
    model: TextModel,
    buffers: TextBuffers,
    tables: Vec<TableShape>,
    /// Configured per-table widths every graph addresses.
    widths: Vec<i64>,
    shapes: TextShapes,
    graphs: GraphTable<TextGraphKey>,
    pool: Option<Handle>,
    indexed: bool,
}

/// What the runner is built from.
pub(crate) struct RunnerParts {
    /// Names the runner in errors.
    pub(crate) component: String,
    pub(crate) device: Option<i32>,
    pub(crate) stream: Option<Stream>,
    pub(crate) model: TextModel,
    pub(crate) buffers: TextBuffers,
    pub(crate) tables: Vec<TableShape>,
    pub(crate) widths: Vec<i64>,
    pub(crate) shapes: TextShapes,
    /// The graph pool, absent when graphs are off.
    pub(crate) pool: Option<Handle>,
}

impl TextRunner {
    pub(crate) fn new(parts: RunnerParts) -> Self {
        let indexed = parts.device.is_some();
        let graphs = GraphTable::new(
            format!("text runner of {}", parts.component),
            parts.model.release().clone(),
        );
        Self {
            device: parts.device,
            stream: parts.stream,
            model: parts.model,
            buffers: parts.buffers,
            tables: parts.tables,
            widths: parts.widths,
            shapes: parts.shapes,
            graphs,
            pool: parts.pool,
            indexed,
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

    /// Decode rows of this runner can gather their inputs from slot state.
    pub(crate) fn indexed(&self) -> bool {
        self.indexed
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

    /// Plan `request` and stage its host columns on the runner's stream.
    pub(crate) fn stage(
        &self,
        request: &StepRequest<'_>,
        context: &mut StepContext<'_>,
    ) -> Result<Staged> {
        let plan = self.plan(request, context)?;
        self.copy_columns(&plan, request, context.state)?;
        let (sampling, greedy) = match &request.sampling {
            Some(sampling) => (sampling.words.to_vec(), sampling.greedy),
            None => (Vec::new(), false),
        };
        Ok(Staged {
            plan,
            sampling,
            greedy,
            prompt: request.prompt.to_vec(),
            embedded: !request.embeddings.is_empty(),
        })
    }

    /// The numerical call of a staged invocation at `capacity`: a replay
    /// of the resident graph variant, or an eager forward (also the warmup
    /// of a capture [`Self::finish`] then records). Without `replay`, a
    /// resident graph's shapes run eagerly instead.
    pub(crate) fn launch(
        &self,
        staged: &Staged,
        capacity: Option<usize>,
        replay: bool,
    ) -> Result<Launch> {
        let (mode, graph) = match &staged.plan.choice {
            Choice::Eager => (GraphMode::Eager, None),
            Choice::Graph(key) => match self.graphs.lookup(key, capacity)? {
                Lookup::Replay(graph) if replay => {
                    (GraphMode::Replay, Some(graph.handle().clone()))
                }
                Lookup::Replay(_) => (GraphMode::Eager, None),
                // The eager run is the capture's warmup and produces this
                // invocation's results; capture records without executing.
                Lookup::Capture => (GraphMode::Capture, None),
            },
        };
        let plan = &staged.plan;
        let call = self.model.step_call(
            &StepArgs {
                graph: graph.as_ref(),
                decode: plan.decode,
                rows: plan.rows,
                live: plan.live,
                tokens: plan.tokens,
                indexed: plan.indexed,
                widths: &plan.widths,
                queries: &plan.queries,
                prefixes: &plan.prefixes,
                starts: &plan.starts,
                causal: &plan.causal,
                writes: true,
                logit_count: plan.logit_count,
                all_last: plan.all_last,
                cache_only: plan.cache_only,
                sampling: &staged.sampling,
                greedy: staged.greedy,
                prompt: &staged.prompt,
                axes: plan.axes,
                embedded: staged.embedded,
            },
            capacity.unwrap_or(0),
        );
        Ok(Launch { mode, call })
    }

    /// Complete an invocation whose call ran: capture its graph after the
    /// eager warmup when its launch chose capture, and count the mode.
    /// Capture submits no device work and no communication.
    pub(crate) fn finish(
        &mut self,
        staged: Staged,
        capacity: Option<usize>,
        mode: GraphMode,
    ) -> Result<Executed> {
        let plan = staged.plan;
        if mode == GraphMode::Capture
            && let Choice::Graph(key) = &plan.choice
        {
            let pool = self
                .pool
                .as_ref()
                .ok_or(Error::State("graph capture requires a graph pool"))?;
            let graph = self.model.capture(
                &CaptureArgs {
                    decode: plan.decode,
                    rows: plan.rows,
                    tokens: plan.tokens,
                    widths: &plan.widths,
                    queries: &plan.queries,
                    prefixes: &plan.prefixes,
                    starts: &plan.starts,
                    causal: &plan.causal,
                    writes: true,
                    cache_only: plan.cache_only && !key.outputs,
                },
                pool,
                capacity.unwrap_or(0),
            )?;
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
        if request.decode && rows.iter().any(|row| row.query != 1) {
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

        let indexed = self.indexed && rows.iter().all(|row| row.source == RowSource::Indexed);
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
                request.decode,
                Some(true),
                false,
                last_logits,
                cache_only,
            )? {
                choice = Choice::Graph(TextGraphKey {
                    decode,
                    rows: bucket_rows,
                    tokens: bucket_tokens,
                    outputs,
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
            } else if indexed {
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
        let mut writes = vec![1_i64; live];
        let mut slots: Vec<i64> = rows.iter().map(|row| i64::from(row.slot)).collect();
        if padded_rows > live {
            if request.decode {
                queries.resize(padded_rows, 1);
            } else {
                queries.push((padded_tokens - live_tokens) as i64);
                queries.resize(padded_rows, 0);
            }
            prefixes.resize(padded_rows, 0);
            writes.resize(padded_rows, 0);
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

        // The row gather's host columns: per-row sections of stride R + 1,
        // then each token's row.
        let mut row_columns = Vec::new();
        let mut token_ids = Vec::new();
        let mut positions = Vec::new();
        let mut relays = Vec::new();
        if !indexed {
            let stride = dimensions.rows + 1;
            row_columns = vec![0_i64; (ROW_SECTIONS + pages.len()) * stride + padded_tokens];
            let mut query_offset = 0;
            let mut prefix_offset = 0;
            for row in 0..padded_rows {
                row_columns[row] = slots[row];
                row_columns[stride + row] = prefixes[row];
                row_columns[2 * stride + row] = queries[row];
                row_columns[3 * stride + row] = writes[row];
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
                        relays.push((offset, row.slot, false));
                        for index in 0..=drafts.len() {
                            if index > 0 {
                                token_ids[offset + index] = i64::from(drafts[index - 1]);
                            }
                            positions[offset + index] = (position + index as u64) as i64;
                        }
                    }
                    RowSource::Resident | RowSource::Indexed => {
                        relays.push((offset, row.slot, true));
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

        // Candidate logits positions of a non-decode invocation.
        let layout = logit_layout(rows);
        let logit_count: usize = layout.iter().map(|&(_, count)| count).sum();
        let mut logit_rows = Vec::new();
        let all_last = last_logits;
        if !request.decode && !all_last && logit_count > 0 {
            let mut offset = 0;
            for row in rows {
                let stop = offset + row.query as usize;
                logit_rows.extend((stop - row.logit_rows()..stop).map(|position| position as i64));
                offset = stop;
            }
        }

        Ok(Plan {
            decode: request.decode,
            live,
            rows: padded_rows,
            live_tokens,
            tokens: padded_tokens,
            indexed,
            widths,
            queries,
            prefixes,
            starts,
            causal,
            axes,
            logit_count: if request.decode { live } else { logit_count },
            all_last,
            cache_only,
            choice,
            row_columns,
            token_ids,
            positions,
            slots,
            force,
            logit_rows,
            relays,
        })
    }

    /// Copy the plan's host columns and sampling inputs to the device, then
    /// relay slot-state tokens and positions, in stream order.
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
        push(column("request_pool_indices")?, bytes(&plan.slots));
        push(column("decode_force_finish")?, &plan.force);
        let mut causal_values = Vec::new();
        if !plan.indexed {
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
        for &(offset, slot, resident) in &plan.relays {
            let slot = slot as usize;
            relays.push((
                column("input_ids")? + 8 * offset as u64,
                state.future_token(slot)?,
                8,
            ));
            if resident {
                relays.push((
                    column("positions")? + 8 * offset as u64,
                    state.logical_length(slot)?,
                    8,
                ));
            }
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
    /// sampling, visible, penalties]` each) into the slot state.
    pub(crate) fn commit(&self, rows: &[[i64; 8]]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let capacity = self.buffers.dimensions.rows;
        if rows.len() > capacity {
            return Err(Error::Invariant("commit rows exceed the binding".into()));
        }
        let base = self.buffers.column("commit")?.address();
        let fields: Vec<Vec<i64>> = (0..8)
            .map(|field| rows.iter().map(|row| row[field]).collect())
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
        self.model.commit(rows.len())
    }

    /// Release graphs and the binding's views. The caller drained the lane.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = self.graphs.close(aborted);
        let closed = self.model.close(aborted);
        graphs.and(closed)
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

//! `DiffusionLMRunner`: the forwards of a diffusion language model on one
//! lane, the Rust half of
//! `uniserve_models.runner.diffusion_lm.DiffusionLMRunner`.
//!
//! A readout evaluates blocks of tokens over their requests' cached
//! prefixes and reads candidate log-probabilities at answer slots: one
//! `attend` over every token, then `readout` over the slot rows. A
//! denoising step denoises each row's resident block by one step of block
//! diffusion (`denoise`). The runner owns the lane's buffers, the resident
//! banks, the graph table and the Python half. For one batch
//! ([`forward`]) it writes the forward's inputs into the buffers on the
//! lane's stream, gathers resident bank rows natively, pads to a graph
//! bucket, launches the bucket's graph and copies the outputs to the
//! batch's output lease.
//!
//! With a graph pool, startup captures (largest first):
//!
//! - an `attend` graph per row and length bucket, over the bucket's tokens
//!   in up to twice as many sequences ([`shapes::attend_padding`]);
//! - on the final pipeline stage of a model whose `readout` exchanges no
//!   tokens, a `readout` graph per slot bucket; a forward with more slots
//!   launches one per chunk;
//! - on generating ranks, a `denoise` graph per row bucket and first-step
//!   flag: the whole step, from the model through the sampler to the commit
//!   of the stepped state to the banks.
//!
//! A graph is captured by calling its method between the begin and the end
//! of a capture, after one eager call of the same shape, and launched
//! natively; a distributed expert weight prefetch is a graph break its
//! launch performs between two segments. Without a pool (CPU lanes, graph
//! policy off) every forward calls its method on the Python thread. A
//! `readout` that exchanges tokens with expert peers also runs there, once
//! over every slot of the forward.
//!
//! With expert-parallel layers, a readout (`attend` and an exchanging
//! `readout`) and a denoising step are each one expert step: the caller's
//! coordinator ([`Coordinate`]) turns the forward's token extent into the
//! step's agreed capacity, and graphs are captured and launched per
//! capacity.

mod buffers;
mod calls;
mod facts;
mod forward;
mod shapes;
mod start;

use std::ops::Range;
use std::sync::Arc;

use uniserve_core::CanvasSampling;
use uniserve_cuda::Stream;

use super::input_buffers::InputBuffers;
use super::methods::{Argument, Methods};
use crate::cuda_graph::{GraphBreaks, GraphMode, GraphTable, Lookup};
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::ffi::{Backing, DType, Handle, Record};
use crate::parallel::ExpertUnion;
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{GroupShape, GroupTable, KVCache, OutputPool, TablePages, table_pages};
use crate::warmup::WarmupHooks;
use crate::{Error, Result};

use buffers::{AttentionMetadata, Banks, ColumnWrites, DenoiseViews, Dimensions};
use calls::{DenoiseRow, ReadoutPass, Sequence};
use facts::Facts;
use shapes::{SLOT_BUCKETS, ShapeKey, Shapes};

pub(crate) use forward::execute;
pub(crate) use start::{KIND, Startup, reserved_bytes};

/// Turns a forward's expert token extent into its expert step's capacity;
/// `None` without expert-parallel layers.
type Coordinate<'a> = &'a mut dyn FnMut(usize) -> Result<Option<usize>>;

/// Candidates one `readout` graph selects: the size of the candidate
/// buffers.
const CANDIDATES: usize = 1 << 16;

/// The import path of the runner's Python half.
const RUNNER: &str = "uniserve_models.runner.diffusion_lm.DiffusionLMRunner";

/// The methods of the Python half the rank calls.
const METHODS: [&str; 6] = [
    "attend",
    "readout",
    "denoise",
    "plan_attention",
    "plans_attention",
    "kernels",
];

/// The runner of one diffusion language model component on the rank.
pub(crate) struct DiffusionLMRunner {
    /// The runner's place among the rank's diffusion LM runners, which
    /// tells their expert steps apart.
    index: usize,
    /// Names the runner in errors, graph tables and kernel records.
    owner: String,
    /// The component whose calls the runner serves.
    component: String,
    methods: Methods,
    facts: Facts,
    buffers: InputBuffers<Dimensions>,
    /// The state rows of a denoising step, on a generating rank.
    state: Option<DenoiseViews>,
    /// The resident banks of a generating rank.
    banks: Option<Banks>,
    /// The block-diffusion sampling a generating rank serves.
    sampling: Option<CanvasSampling>,
    /// Blocks the sampler steps at a time.
    chunk_rows: usize,
    shapes: Shapes,
    /// Page columns every table's rows hold in a graph.
    graph_widths: Vec<usize>,
    /// The cache groups of the numerical tables, in table order.
    groups: Vec<GroupShape>,
    /// The allocator pool graphs are captured into; `None` runs eagerly.
    pool: Option<Handle>,
    /// The lane's stream; `None` for a CPU lane.
    stream: Option<Stream>,
    graphs: GraphTable<ShapeKey>,
    /// The graph breaks of a forward being captured, noted by the
    /// distributed expert weights its layers prefetch.
    breaks: Option<GraphBreaks>,
    /// Segments of the captured graph with the most graph breaks.
    segments: usize,
    /// An attention operator plans each launch on the host, so a launch of
    /// `attend` or `denoise` follows `plan_attention`. Known once every
    /// shape has run; until then every launch plans.
    plans_attention: bool,
    /// The NCCL communicators of the runner's component, destroyed after
    /// its graphs.
    collectives: Vec<crate::parallel::StreamCollectives>,
    /// One output lease per batch in flight.
    outputs: OutputPool,
}

/// How one forward ran.
struct Executed {
    mode: GraphMode,
    tokens: usize,
    padded_tokens: usize,
}

/// The host lists of a method's attention arguments, in parameter order.
fn mirrors(metadata: &AttentionMetadata) -> [Argument; 4] {
    [
        metadata.widths.as_slice().into(),
        metadata.queries.as_slice().into(),
        metadata.prefixes.as_slice().into(),
        metadata.starts.as_slice().into(),
    ]
}

impl DiffusionLMRunner {
    /// The warmup and capture plan every source of an expert union must
    /// share: the runner's row and length buckets and whether it denoises.
    pub(crate) fn capture_plan(&self) -> serde_json::Value {
        serde_json::json!({
            "rows": self.shapes.rows(),
            "lengths": self.shapes.lengths(),
            "steps": self.sampling.is_some(),
        })
    }

    /// The runner's owner name and kernel records.
    pub(crate) fn kernels(&self) -> Result<(String, Vec<Record>)> {
        let records = self.methods.call("kernels", Vec::new())?.invoke()?;
        let records = tvm_ffi::Array::<tvm_ffi::Any>::try_from(records)
            .map_err(|_| Error::Interface(format!("{RUNNER}.kernels returns a list")))?
            .iter()
            .enumerate()
            .map(|(index, record)| Record::new(format!("{RUNNER}.kernels[{index}]"), record))
            .collect::<Result<_>>()?;
        Ok((self.owner.clone(), records))
    }

    /// Call the method `name` on the Python thread: an eager forward, or
    /// the forward a capture in progress records.
    fn call(&self, name: &str, arguments: &[Argument]) -> Result<()> {
        self.methods
            .call(name, arguments.to_vec())?
            .invoke()
            .map(drop)
    }

    /// Refuse further captures; serving may only launch.
    pub(crate) fn seal(&mut self) {
        self.graphs.seal();
    }

    /// Release graphs and output storage after the lane drained.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = self.graphs.close(aborted);
        for collectives in &mut self.collectives {
            result = result.and(collectives.close(aborted));
        }
        self.outputs.begin_close();
        if !aborted {
            self.outputs.finish_close();
        }
        result
    }

    fn submit(&self, writes: ColumnWrites) -> Result<()> {
        writes.submit(self.buffers.device(), self.stream.as_ref())
    }

    fn tensor(&self, name: &str) -> Result<Argument> {
        Ok(Argument::Tensor(self.buffers.column(name)?.clone()))
    }

    /// The attention metadata tensors of a method call, in parameter order.
    fn attention_tensors(&self) -> Result<Vec<Argument>> {
        [
            "attention.query_lengths",
            "attention.query_offsets",
            "attention.prefix_lengths",
            "attention.prefix_offsets",
            "attention.block_tables",
            "attention.start_pages",
        ]
        .into_iter()
        .map(|name| self.tensor(name))
        .collect()
    }

    /// The final stage's attend state, which `attend` writes and `readout`
    /// reads.
    fn attend_state(&self) -> Result<Vec<Argument>> {
        (0..self.facts.attend_widths.len())
            .map(|index| self.tensor(&format!("attend_state.{index}")))
            .collect()
    }

    /// Run one forward of `method`: launch the graph resident for `key` at
    /// expert `capacity`, or call the method and capture it; without a pool
    /// call the method. `plan` is the `plan_attention` call of a method
    /// that attends.
    fn run(
        &mut self,
        key: ShapeKey,
        capacity: Option<usize>,
        method: &'static str,
        arguments: &[Argument],
        plan: Option<&[Argument]>,
    ) -> Result<GraphMode> {
        let (Some(pool), Some(stream), Some(device)) = (
            self.pool.clone(),
            self.stream.clone(),
            self.buffers.device(),
        ) else {
            self.call(method, arguments)?;
            self.graphs.record(GraphMode::Eager);
            return Ok(GraphMode::Eager);
        };
        let mode = match self.graphs.lookup(&key, capacity)? {
            Lookup::Replay(graph) => {
                if let (true, Some(plan)) = (self.plans_attention, plan) {
                    self.call("plan_attention", plan)?;
                }
                graph.launch(&stream)?;
                GraphMode::Replay
            }
            Lookup::Capture => {
                // The eager call produces this forward's results, plans its
                // attention and warms the capture, which records the same
                // call without executing it.
                self.call(method, arguments)?;
                let captured = self
                    .methods
                    .graphs
                    .capture(&pool, stream.raw(), device, || self.call(method, arguments));

                // The breaks the forward reached (weight prefetches) belong
                // to this capture, whether or not it succeeded.
                let breaks = self
                    .breaks
                    .as_ref()
                    .map_or_else(Vec::new, GraphBreaks::take);
                self.segments = self.segments.max(breaks.len() + 1);
                self.graphs
                    .insert(key, capacity, captured?.with_breaks(breaks))?;
                GraphMode::Capture
            }
        };
        self.graphs.record(mode);
        Ok(mode)
    }

    /// Evaluate a readout and read every candidate's log-probability.
    ///
    /// `sink(range, values)` receives the FP32 values of the candidates
    /// `range` (in row, slot and candidate order), which later device work
    /// on the stream overwrites. Every pipeline stage receives the same
    /// values.
    fn readout(
        &mut self,
        pass: &ReadoutPass<'_>,
        coordinate: Coordinate<'_>,
        mut sink: impl FnMut(Range<usize>, &StorageView) -> Result<()>,
    ) -> Result<Executed> {
        let layout = self.buffers.dimensions.clone();
        let tokens = pass.tokens.len();
        if pass.rows.is_empty() || pass.rows.iter().map(|row| row.query).sum::<usize>() != tokens {
            return Err(Error::Invalid(
                "readout rows must partition their tokens".into(),
            ));
        }
        if pass.slots.is_empty()
            || pass.slots.len() > layout.tokens
            || pass.slot_candidates.len() != pass.slots.len() + 1
        {
            return Err(Error::Invalid(
                "a readout reads between one slot and its token capacity".into(),
            ));
        }
        // Every stage refuses the same calls, before any collective.
        let vocab = self.facts.vocab_size as i64;
        if pass.candidates.iter().any(|&id| id < 0 || id >= vocab) {
            return Err(Error::Invalid(format!(
                "a readout candidate is outside the {vocab}-token vocabulary"
            )));
        }

        // `attend`: the smallest bucket holding the rows, or the live
        // shape without graphs.
        let graph = self.pool.is_some();
        let (rows, visible, padding, widths) = if graph {
            let (rows, length) = self.shapes.attend(pass.rows.len(), pass.longest())?;
            (
                rows,
                length,
                shapes::attend_padding(pass.rows.len(), tokens, rows, length)?,
                self.graph_widths.clone(),
            )
        } else {
            (
                pass.rows.len(),
                pass.longest(),
                Vec::new(),
                pass.pages.iter().map(TablePages::width).collect(),
            )
        };
        let key = ShapeKey::Attend {
            rows,
            length: visible,
        };
        let padded_tokens = if graph { rows * visible } else { tokens };
        if padded_tokens > layout.tokens {
            return Err(Error::Invalid(format!(
                "a readout of {padded_tokens} tokens exceeds its {} token capacity",
                layout.tokens
            )));
        }
        let mut writes = ColumnWrites::new();
        let ids = self.buffers.column("input_ids")?.clone();
        writes.write(&ids, 0, &pass.tokens)?;
        writes.zero(&ids, tokens, padded_tokens - tokens)?;
        buffers::write_positions(&self.buffers, &mut writes, &pass.rows, padded_tokens)?;
        let metadata = buffers::write_attention(
            &self.buffers,
            &mut writes,
            &pass.rows,
            &padding,
            pass.pages,
            &widths,
        )?;
        self.submit(writes)?;

        // The expert step carries the tokens and, when `readout`
        // exchanges, the slot rows.
        let local = self.facts.local_tail;
        let extent = if local {
            padded_tokens
        } else {
            padded_tokens.max(pass.slots.len())
        };
        let capacity = coordinate(extent)?;
        let expert_capacity = capacity.unwrap_or(0);
        let state = self.attend_state()?;
        let attention = self.attention_tensors()?;

        let mut plan: Vec<Argument> = vec![metadata.rows().into(), visible.into()];
        plan.extend(mirrors(&metadata));
        plan.extend(attention.iter().cloned());
        let mut arguments: Vec<Argument> =
            vec![metadata.rows().into(), padded_tokens.into(), visible.into()];
        arguments.extend(mirrors(&metadata));
        arguments.push(expert_capacity.into());
        arguments.push(self.tensor("input_ids")?);
        arguments.push(self.tensor("positions")?);
        arguments.extend(attention);
        arguments.extend(state.iter().cloned());
        let mode = self.run(key, capacity, "attend", &arguments, Some(&plan))?;

        let slot_rows = self.buffers.column("slot_rows")?.clone();
        let index = self.buffers.column("candidate_index")?.clone();
        let values = self.buffers.column("candidate_logprobs")?.clone();
        let readout =
            |slots: usize, candidates: usize, index: &StorageView, values: &StorageView| {
                let mut arguments: Vec<Argument> = vec![
                    slots.into(),
                    candidates.into(),
                    expert_capacity.into(),
                    (&slot_rows).into(),
                    index.into(),
                    values.into(),
                ];
                arguments.extend(state.iter().cloned());
                arguments
            };
        // The flat index of candidate `candidate` in the `[slots, vocab]`
        // log-probabilities of the slots from `base` on.
        let flat = |range: Range<usize>, base: usize| {
            let mut slot = base;
            range
                .map(|candidate| {
                    while pass.slot_candidates[slot + 1] <= candidate {
                        slot += 1;
                    }
                    (slot - base) as i64 * vocab + pass.candidates[candidate]
                })
                .collect::<Vec<i64>>()
        };

        if graph && local {
            // One `readout` per slot chunk and window of its candidates.
            // Every stage runs the same windows, whose candidate values the
            // final stage broadcasts.
            for (start, live, bucket) in shapes::slot_chunks(pass.slots.len()) {
                let first = pass.slot_candidates[start];
                let last = pass.slot_candidates[start + live];
                for window in (first..last).step_by(CANDIDATES) {
                    let end = (window + CANDIDATES).min(last);
                    let selected = flat(window..end, start);
                    let mut writes = ColumnWrites::new();
                    writes.write(&slot_rows, 0, &pass.slots[start..start + live])?;
                    writes.zero(&slot_rows, live, bucket - live)?;
                    // Unused candidates select the first log-probability.
                    writes.write(&index, 0, &selected)?;
                    writes.zero(&index, selected.len(), CANDIDATES - selected.len())?;
                    self.submit(writes)?;
                    let arguments = readout(bucket, CANDIDATES, &index, &values);
                    if self.facts.last_stage() {
                        let key = ShapeKey::Readout { slots: bucket };
                        self.run(key, None, "readout", &arguments, None)?;
                    } else {
                        self.call("readout", &arguments)?;
                    }
                    sink(window..end, &values.prefix((end - window) as i64)?)?;
                }
            }
        } else {
            // One `readout` over every slot and candidate, on the Python
            // thread: an exchanging `readout` completes the expert step
            // `attend` began.
            let total = pass.candidates.len();
            let device = self.buffers.device();
            // A forward with more candidates than the candidate buffers
            // hold selects into storage of its own.
            let larger = if total > CANDIDATES {
                let memory = match device {
                    Some(device) => Memory::device(device, total * 12)?,
                    None => Memory::host(total * 12),
                };
                let backing = Arc::clone(&memory) as Arc<dyn Backing>;
                let shape = [total as i64];
                Some((
                    StorageView::contiguous(Arc::clone(&backing), 0, DType::I64, &shape)?,
                    StorageView::contiguous(backing, total * 8, DType::F32, &shape)?,
                ))
            } else {
                None
            };
            let (index, values) = larger
                .as_ref()
                .map_or((&index, &values), |(index, values)| (index, values));
            let mut writes = ColumnWrites::new();
            writes.write(&slot_rows, 0, &pass.slots)?;
            writes.write(index, 0, &flat(0..total, 0))?;
            self.submit(writes)?;
            self.call("readout", &readout(pass.slots.len(), total, index, values))?;
            sink(0..total, &values.prefix(total as i64)?)?;
            if let (Some(_), Some(stream)) = (&larger, &self.stream) {
                // The forward's own storage is freed once its copies ran.
                stream.synchronize()?;
            }
        }
        Ok(Executed {
            mode,
            tokens,
            padded_tokens,
        })
    }

    /// Denoise the resident blocks of `rows` by one step. The rows' int64
    /// `[1 + length]` results are the leading rows of `denoise.results` and
    /// their continuation flags those of `denoise.continued`.
    fn denoise(
        &mut self,
        rows: &[DenoiseRow],
        pages: &[TablePages],
        coordinate: Coordinate<'_>,
    ) -> Result<Executed> {
        let (Some(banks), Some(state)) = (&self.banks, &self.state) else {
            return Err(Error::Invalid(
                "this worker keeps no generating blocks".into(),
            ));
        };
        let length = self.facts.sequence_length;
        let live = rows.len();
        let graph = self.pool.is_some();
        let padded = if graph {
            self.shapes.denoise(live)?
        } else {
            live
        };
        if live == 0 || padded > state.rows {
            return Err(Error::Invalid(format!(
                "a denoising step of {live} rows exceeds its {} rows",
                state.rows
            )));
        }

        let mut writes = ColumnWrites::new();
        let columns = self.buffers.column("denoise.columns")?.clone();
        let fields: [Vec<i64>; 4] = [
            rows.iter().map(|row| i64::from(row.slot)).collect(),
            // Seeds are u64 carried bit for bit in int64 columns.
            rows.iter().map(|row| row.seed as i64).collect(),
            rows.iter().map(|row| i64::from(row.block)).collect(),
            rows.iter().map(|row| i64::from(row.step)).collect(),
        ];
        for (field, values) in fields.iter().enumerate() {
            // A padding row starts a block of seed and block zero in the
            // sentinel slot zero, which its commit skips.
            writes.write(&columns, field * state.rows, values)?;
            writes.zero(&columns, field * state.rows + live, padded - live)?;
        }
        let sequences: Vec<Sequence> = rows
            .iter()
            .map(|row| Sequence {
                query: length,
                prefix: row.prefix,
                position: row.position,
            })
            .collect();
        buffers::write_positions(&self.buffers, &mut writes, &sequences, padded * length)?;
        let widths: Vec<usize> = if graph {
            self.graph_widths.clone()
        } else {
            pages.iter().map(TablePages::width).collect()
        };
        let metadata = buffers::write_attention(
            &self.buffers,
            &mut writes,
            &sequences,
            &vec![length; padded - live],
            pages,
            &widths,
        )?;
        self.submit(writes)?;
        let slots: Vec<u32> = rows.iter().map(|row| row.slot).collect();
        banks.gather(&slots, padded, state, self.stream.as_ref())?;

        let first = rows.iter().all(|row| row.step == 0);
        let capacity = coordinate(padded * length)?;
        let attention = self.attention_tensors()?;
        let mut plan: Vec<Argument> = vec![padded.into(), length.into()];
        plan.extend(mirrors(&metadata));
        plan.extend(attention.iter().cloned());
        let mut arguments: Vec<Argument> = vec![padded.into(), first.into()];
        arguments.extend(mirrors(&metadata));
        arguments.push(capacity.unwrap_or(0).into());
        arguments.push(self.tensor("denoise.columns")?);
        arguments.push(self.tensor("positions")?);
        arguments.extend(attention);
        for name in [
            "denoise.tokens",
            "denoise.history",
            "denoise.self_conditioning",
            "denoise.results",
            "denoise.continued",
        ] {
            arguments.push(self.tensor(name)?);
        }
        let key = ShapeKey::Denoise {
            rows: padded,
            first,
        };
        let mode = self.run(key, capacity, "denoise", &arguments, Some(&plan))?;
        Ok(Executed {
            mode,
            tokens: live * length,
            padded_tokens: padded * length,
        })
    }

    /// Run every forward shape serving can select, over scratch prefixes
    /// of the prefix cache: row `r` of a synthetic forward reads a
    /// one-token prefix from its own page of every cache group. Denoising
    /// steps run in slots `1..`; their synthetic state is harmless because
    /// a request's first step starts its block before reading any of it.
    ///
    /// With a graph pool, every configured graph is captured, largest
    /// first, each after its method ran eagerly. Without one, a generating
    /// CUDA lane runs one eager step of every sampler chunk shape, whose
    /// first self-conditioning product chooses its algorithm and
    /// synchronizes the device, so serving never does.
    ///
    /// With expert-parallel layers every graph is captured at every expert
    /// capacity holding its tokens, largest first, each warmup an announced
    /// expert step; every rank warms the same sequence.
    fn warm(&mut self, kv: &mut KVCache, experts: Option<&ExpertUnion>) -> Result<()> {
        let row_units: usize = self
            .groups
            .iter()
            .map(|group| group.units_per_page as usize)
            .sum();
        let scratch = kv.scratch_units(
            (self.shapes.capacity() * row_units) as u32,
            self.stream.as_ref(),
        )?;
        let first = *scratch.units().start() as usize;
        let warmed = self.warm_shapes(first, row_units, experts);
        let finished = scratch.finish(kv, self.stream.as_ref());
        warmed.and(finished)?;
        if let Some(stream) = &self.stream {
            stream.synchronize()?;
        }
        let plans = self.methods.call("plans_attention", Vec::new())?.invoke()?;
        self.plans_attention = bool::try_from(plans)
            .map_err(|_| Error::Interface(format!("{RUNNER}.plans_attention returns a flag")))?;

        let mut counts = [0; 3];
        for (key, _) in self.graphs.keys() {
            counts[match key {
                ShapeKey::Attend { .. } => 0,
                ShapeKey::Readout { .. } => 1,
                ShapeKey::Denoise { .. } => 2,
            }] += 1;
        }
        tracing::info!(
            target: "uniserve_worker.warmup",
            "{}: captured {} graphs of up to {} segments: {} attend, {} readout and {} denoise buckets; attention launches are planned on the {}",
            self.owner,
            counts.iter().sum::<usize>(),
            self.segments,
            counts[0],
            counts[1],
            counts[2],
            if self.plans_attention {
                "host"
            } else {
                "device"
            },
        );
        Ok(())
    }

    fn warm_shapes(
        &mut self,
        first_unit: usize,
        row_units: usize,
        experts: Option<&ExpertUnion>,
    ) -> Result<()> {
        // The expert capacities a forward of `tokens` is captured at.
        let variants = |tokens: usize| -> Vec<Option<usize>> {
            match experts {
                None => vec![None],
                Some(experts) => experts
                    .parallel()
                    .capacities()
                    .iter()
                    .rev()
                    .copied()
                    .filter(|&capacity| capacity >= tokens)
                    .map(Some)
                    .collect(),
            }
        };
        let length = self.facts.sequence_length;
        if self.pool.is_none() {
            if self.stream.is_none() || self.sampling.is_none() {
                return Ok(());
            }
            for count in 1..=self.chunk_rows {
                let pages = self.scratch_pages(count, length, first_unit, row_units)?;
                self.denoise(
                    &synthetic_steps(count, true),
                    &pages,
                    &mut forward::coordinator(experts, self.index, None),
                )?;
            }
            return Ok(());
        }

        let rows: Vec<usize> = self.shapes.rows().iter().rev().copied().collect();
        let lengths: Vec<usize> = self.shapes.lengths().iter().rev().copied().collect();
        let local = self.facts.local_tail;
        if local && self.facts.last_stage() {
            for slots in SLOT_BUCKETS {
                self.warm_readout(slots)?;
            }
        }
        for &count in &rows {
            for &length in &lengths {
                let pages = self.scratch_pages(count, length, first_unit, row_units)?;
                let pass = ReadoutPass {
                    rows: vec![
                        Sequence {
                            query: length,
                            prefix: 1,
                            position: 1,
                        };
                        count
                    ],
                    tokens: vec![0; count * length],
                    slots: (0..count).map(|row| (row * length) as i64).collect(),
                    slot_candidates: (0..=count).collect(),
                    candidates: vec![0; count],
                    pages: &pages,
                };
                let (rows, length) = self.shapes.attend(count, length)?;
                let tokens = rows * length;
                let extent = if local { tokens } else { tokens.max(count) };
                for capacity in variants(extent) {
                    self.readout(
                        &pass,
                        &mut forward::coordinator(experts, self.index, capacity),
                        |_, _| Ok(()),
                    )?;
                }
            }
        }
        if self.sampling.is_some() {
            for &count in &rows {
                let pages = self.scratch_pages(count, length, first_unit, row_units)?;
                let padded = self.shapes.denoise(count)?;
                for first in [true, false] {
                    for capacity in variants(padded * length) {
                        self.denoise(
                            &synthetic_steps(count, first),
                            &pages,
                            &mut forward::coordinator(experts, self.index, capacity),
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Capture the `readout` graph of `slots` slot rows.
    fn warm_readout(&mut self, slots: usize) -> Result<()> {
        let slot_rows = self.buffers.column("slot_rows")?.clone();
        let index = self.buffers.column("candidate_index")?.clone();
        let mut writes = ColumnWrites::new();
        writes.write(&slot_rows, 0, &(0..slots as i64).collect::<Vec<_>>())?;
        writes.zero(&index, 0, CANDIDATES)?;
        self.submit(writes)?;
        // A local `readout` exchanges no tokens: it has no expert capacity.
        let mut arguments: Vec<Argument> = vec![
            slots.into(),
            CANDIDATES.into(),
            0usize.into(),
            (&slot_rows).into(),
            (&index).into(),
            self.tensor("candidate_logprobs")?,
        ];
        arguments.extend(self.attend_state()?);
        let key = ShapeKey::Readout { slots };
        self.run(key, None, "readout", &arguments, None).map(drop)
    }

    /// Pages of `count` rows of `length` tokens reading a one-token prefix
    /// on scratch units: row `r` reads units `first_unit + r * row_units
    /// ..`, one page of every group in table order.
    fn scratch_pages(
        &self,
        count: usize,
        length: usize,
        first_unit: usize,
        row_units: usize,
    ) -> Result<Vec<TablePages>> {
        let tables = (0..count)
            .map(|row| {
                let mut next = (first_unit + row * row_units..).map(|unit| unit as u32);
                self.groups
                    .iter()
                    .map(|&shape| {
                        let units = (&mut next)
                            .take(shape.units_per_page as usize)
                            .map(uniserve_core::UnitId)
                            .collect();
                        Arc::new(GroupTable {
                            shape,
                            start_page: 0,
                            units,
                            allocated_tokens: shape.page_tokens,
                        })
                    })
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        table_pages(&tables, &vec![1; count], &vec![length as u32; count])
    }
}

/// Denoising steps of slots `1..=count` with a one-token prefix, at step
/// zero (`first`) or one.
fn synthetic_steps(count: usize, first: bool) -> Vec<DenoiseRow> {
    (0..count)
        .map(|row| DenoiseRow {
            slot: row as u32 + 1,
            seed: row as u64 + 1,
            block: 0,
            step: u32::from(!first),
            prefix: 1,
            position: 1,
        })
        .collect()
}

/// The runner's warmup work.
pub(crate) struct DiffusionLMWarmup;

impl WarmupHooks for DiffusionLMWarmup {
    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let parts = backend.parts();
        let Some(storage) = parts.storage else {
            return Ok(());
        };
        for runner in parts.diffusion_lm {
            let kv = storage
                .kv
                .as_mut()
                .ok_or(Error::State("diffusion LM warmup requires a KV cache"))?;
            runner.warm(kv, parts.experts).map_err(|error| {
                WorkerError::classify(error, Some("diffusion LM graph capture"))
                    .with_phase("warmup")
            })?;
        }
        Ok(())
    }
}

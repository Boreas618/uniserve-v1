//! `DiffusionLMRunner`: the forwards of a diffusion language model on one
//! lane, the Rust half of
//! `uniserve.runner.diffusion_lm.DiffusionLMRunner`.
//!
//! A readout evaluates blocks of tokens over their requests' cached
//! prefixes and reads candidate log-probabilities at answer slots: one
//! `attend` over every token, one `finish` of the final layer at the slot
//! rows, then `readout` of the finished rows. A denoising step denoises
//! each row's resident block by one step of block diffusion (`denoise`).
//! The runner owns the lane's buffers, the resident banks, the graph table
//! and the Python half. For one batch
//! ([`forward`]) it records a job of its lane: the writes of the forward's
//! inputs into the buffers, the native gather of resident bank rows, the
//! launch of the graph of the bucket the forward pads to, and the copies of
//! the outputs to the batch's output lease. The lane's execution thread
//! performs the job, and the runner applies its completion
//! ([`DiffusionLMRunner::complete`]).
//!
//! With a graph pool, startup captures (largest first) a graph of every
//! forward serving can form:
//!
//! - an `attend` graph per row and token bucket, over the bucket's tokens
//!   in up to twice as many sequences ([`shapes::attend_padding`]), so
//!   every pass of at most the engine's rows and token budget fits one;
//! - on the final pipeline stage, a `finish` graph per slot bucket, up to
//!   the slots of a pass of the token budget;
//! - on every stage, a `readout` graph per slot bucket; a forward with more
//!   slots launches one per chunk;
//! - on generating ranks, a `denoise` graph per row bucket and first-step
//!   flag: the whole step, from the model through the sampler to the commit
//!   of the stepped state to the banks.
//!
//! A graph is captured by calling its method between the begin and the end
//! of a capture, after one eager call of the same shape, and launched
//! natively; a distributed expert weight prefetch is a graph break its
//! launch performs between two segments. The capture is an operation of
//! the forward's job, whose completion carries the graph. A forward that no
//! captured bucket holds is refused. Without a pool (CPU lanes, graph
//! policy off) every forward calls its method on the builder thread.
//!
//! With expert-parallel layers, a readout and a denoising step are each one
//! expert step of the job ([`Expert`]): the execution thread agrees on the
//! step's capacity for the forward's token extent, and the job carries the
//! forward at every capacity the agreement can choose. When the final
//! layer exchanges tokens with expert peers, the step spans `attend` and
//! `finish`. Graphs are captured and launched per capacity.

mod batch;
mod buffers;
mod calls;
mod facts;
mod shapes;
mod start;

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use cuda::Stream;
use ipc::CanvasSampling;

use super::input_buffers::InputBuffers;
use super::methods::{Argument, Methods};
use crate::cuda_graph::{GraphBreaks, GraphMode, GraphTable, Lookup};
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::ffi::Function;
use crate::ffi::Handle;
use crate::lane::{
    BLOCK_ON_TIMEOUT, Capture, Completion, ExpertStep, ExpertVariant, Job, JobQueues,
};
use crate::memory::allocation::StorageView;
use crate::memory::{AttentionRow, OutputPool, PrefixCache, TablePages};
use crate::parallel::{ExpertUnion, MicrobatchForward};
use crate::profiling::kernels::KernelRecord;
use crate::warmup::WarmupHooks;
use crate::{Error, Result};

use buffers::{AttentionMetadata, Banks, ColumnWrites, DenoiseViews, Dimensions};
use calls::{DenoiseRow, ReadoutPass, Sequence};
use facts::Facts;
use shapes::{SLOT_BUCKETS, ShapeKey, Shapes};

pub(crate) use batch::{Pending, complete, forward};
pub(crate) use start::{KIND, Startup, reserved_bytes};

/// The expert steps of a runner's forwards.
struct Expert<'a> {
    union: &'a Arc<ExpertUnion>,
    /// The runner's call kind in the union's agreement.
    kind: i64,
    /// The capacity a warmup step runs at; `None` runs it at the smallest
    /// capacity holding the forward.
    forced: Option<usize>,
}

/// Candidates one `readout` graph selects: the size of the candidate
/// buffers.
const CANDIDATES: usize = 1 << 16;

/// The import path of the runner's Python half.
const RUNNER: &str = "uniserve.runner.diffusion_lm.DiffusionLMRunner";

/// The methods of the Python half the rank calls.
struct Functions {
    attend: Function,
    finish: Function,
    readout: Function,
    denoise: Function,
    plan_attention: Function,
    plans_attention: Function,
    kernels: Function,
    close: Function,
}

impl Functions {
    /// Resolve the methods of `methods`' runner.
    fn resolve(methods: &Methods) -> Result<Self> {
        let method = |name: &str| methods.method(name);
        Ok(Self {
            attend: method("attend")?,
            finish: method("finish")?,
            readout: method("readout")?,
            denoise: method("denoise")?,
            plan_attention: method("plan_attention")?,
            plans_attention: method("plans_attention")?,
            kernels: method("kernels")?,
            close: method("close")?,
        })
    }
}

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
    functions: Functions,
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

/// A method of the Python half as a forward runs it.
struct Method<'a> {
    /// The graph bucket of the forward.
    key: &'a ShapeKey,
    function: &'a Function,
    /// The arguments of the `plan_attention` call a launch of a method
    /// that attends follows.
    plan: Option<&'a [Argument]>,
}

/// How a method runs at every capacity its expert step can be agreed at;
/// one entry of no capacity without an expert step.
type Modes = Vec<(Option<usize>, GraphMode)>;

/// A method's arguments at an expert capacity.
type Arguments<'a> = &'a dyn Fn(usize) -> Vec<Argument>;

/// One forward recorded into a job, which [`DiffusionLMRunner::complete`]
/// applies the job's completion for.
struct Recorded {
    /// The graph bucket of the forward's method.
    key: ShapeKey,
    modes: Modes,
    /// The `finish` that completes the expert step of `attend` when the
    /// final layer exchanges tokens: its bucket and modes.
    finish: Option<(ShapeKey, Modes)>,
    /// The buckets of the graphs the job captures behind the forward's step
    /// (a local `finish` and the slot-chunk `readout`s), in job order; their
    /// graphs are stored at completion.
    captured: Vec<ShapeKey>,
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
    /// Blocks one sampler chunk of the runner's denoising step covers, every
    /// row count of which the engine warms through its scheduler; zero on a
    /// rank that generates no blocks.
    pub(crate) fn step_rows(&self) -> u32 {
        if self.sampling.is_some() {
            self.chunk_rows as u32
        } else {
            0
        }
    }

    /// The runner's owner name and kernel records.
    pub(crate) fn kernels(&self) -> Result<(String, Vec<KernelRecord>)> {
        let records = self
            .methods
            .call(&self.functions.kernels, Vec::new())
            .invoke()?;
        let records = tvm_ffi::Array::<tvm_ffi::Any>::try_from(records)
            .map_err(|_| Error::Interface(format!("{RUNNER}.kernels returns a list")))?
            .iter()
            .enumerate()
            .map(|(index, record)| KernelRecord::new(format!("{RUNNER}.kernels[{index}]"), record))
            .collect::<Result<_>>()?;
        Ok((self.owner.clone(), records))
    }

    /// Call `method` on the builder thread, outside any job: startup and
    /// retirement.
    fn call(&self, method: &Function, arguments: &[Argument]) -> Result<()> {
        self.methods
            .call(method, arguments.to_vec())
            .invoke()
            .map(drop)
    }

    /// Refuse further captures; serving may only launch.
    pub(crate) fn seal(&mut self) {
        self.graphs.seal();
    }

    /// Release every graph and accept captures again
    /// ([`GraphTable::reopen`]).
    pub(crate) fn reopen(&mut self) -> Result<()> {
        self.graphs.reopen()
    }

    /// The NCCL communicators the runner's collectives run on.
    pub(crate) fn collectives(&self) -> &[crate::parallel::StreamCollectives] {
        &self.collectives
    }

    /// Release graphs, the Python runner's execution context and output
    /// storage after the lane drained. An aborted close keeps the context:
    /// device work may still address its workspace.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = self.graphs.close(aborted);
        if !aborted {
            result = result.and(self.call(&self.functions.close, &[]));
        }
        for collectives in &mut self.collectives {
            result = result.and(collectives.close(aborted));
        }
        self.outputs.begin_close();
        if !aborted {
            self.outputs.finish_close();
        }
        result
    }

    /// Record `writes` to the runner's buffers in `job`.
    fn submit(&self, writes: ColumnWrites, job: &Job) -> Result<()> {
        writes.submit(job.on(self.stream.as_ref()))
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

    /// Record one forward of `method` at expert `capacity`: into `forward`
    /// the launch of the resident graph (behind the `plan_attention` call
    /// of a method that attends), or the method's eager call; into `then`
    /// the capture that follows an eager call. Without a pool the method is
    /// called eagerly; with one, a bucket sealed without a graph is an
    /// error ([`GraphTable::lookup`]).
    fn method(
        &self,
        method: &Method<'_>,
        capacity: Option<usize>,
        arguments: Vec<Argument>,
        forward: &Job,
        then: &Job,
    ) -> Result<GraphMode> {
        let Method {
            key,
            function,
            plan,
        } = *method;
        let (Some(pool), Some(stream), Some(device)) =
            (&self.pool, &self.stream, self.buffers.device())
        else {
            forward.call(self.methods.call(function, arguments));
            return Ok(GraphMode::Eager);
        };
        match self.graphs.lookup(key, capacity)? {
            Lookup::Replay(graph) => {
                if let (true, Some(plan)) = (self.plans_attention, plan) {
                    forward.call(
                        self.methods
                            .call(&self.functions.plan_attention, plan.to_vec()),
                    );
                }
                forward.on(Some(stream)).launch(graph)?;
                Ok(GraphMode::Replay)
            }
            Lookup::Capture => {
                // The eager call produces this forward's results, plans its
                // attention and warms the capture, which records the same
                // call without executing it.
                forward.call(self.methods.call(function, arguments.clone()));
                then.capture(Capture {
                    functions: self.methods.graphs.clone(),
                    pool: pool.clone(),
                    peers: Vec::new(),
                    stream: stream.raw(),
                    device,
                    method: self.methods.call(function, arguments),
                    // The breaks the forward reaches (weight prefetches)
                    // belong to this capture.
                    breaks: self.breaks.clone(),
                });
                Ok(GraphMode::Capture)
            }
        }
    }

    /// Record the forward of `method` over `extent` expert tokens in `job`.
    ///
    /// `arguments(capacity)` are the method's arguments at an expert
    /// capacity (zero without expert-parallel layers), and `completes` the
    /// method and arguments that complete the method's expert step behind
    /// it (the `finish` of a final layer that exchanges tokens). Without
    /// `expert` the forward is the next operations of `job`. With it the
    /// forward is one expert step: the job carries the method at every
    /// capacity the step can be agreed at, each with the capture and the
    /// completing method that follow it at that capacity. Returns the modes
    /// of the method and of the completing method.
    fn step(
        &self,
        method: &Method<'_>,
        arguments: Arguments<'_>,
        completes: Option<(&Method<'_>, Arguments<'_>)>,
        expert: Option<&Expert<'_>>,
        extent: usize,
        job: &Job,
    ) -> Result<(Modes, Modes)> {
        let Some(expert) = expert else {
            let mode = self.method(method, None, arguments(0), job, job)?;
            let mut completed = Vec::new();
            if let Some((completing, arguments)) = completes {
                let mode = self.method(completing, None, arguments(0), job, job)?;
                completed.push((None, mode));
            }
            return Ok((vec![(None, mode)], completed));
        };

        let lane = self
            .stream
            .clone()
            .ok_or(Error::State("expert steps run on a CUDA stream"))?;
        let union = expert.union;
        // The capacities the step can run at: the one a warmup step
        // announces, or every capacity an agreement can choose for a rank
        // that sends `extent` tokens.
        let holding = union.parallel().capacities_for(extent);
        let capacities: Vec<usize> = if union.warming() {
            match expert.forced {
                Some(capacity) => vec![capacity],
                None => holding.take(1).collect(),
            }
        } else {
            holding.collect()
        };
        if capacities.is_empty() {
            return Err(Error::Invalid(format!(
                "{extent} tokens exceed the expert exchange's tokens per rank"
            )));
        }
        let mut variants = Vec::with_capacity(capacities.len());
        let mut modes = Vec::with_capacity(capacities.len());
        let mut completed = Vec::new();
        for capacity in capacities {
            let (forward, then) = (Job::new(), Job::new());
            let mode = self.method(method, Some(capacity), arguments(capacity), &forward, &then)?;
            if let Some((completing, arguments)) = completes {
                // The completing method follows the method's capture, with
                // its own eager call and capture.
                let mode = self.method(
                    completing,
                    Some(capacity),
                    arguments(capacity),
                    &then,
                    &then,
                )?;
                completed.push((Some(capacity), mode));
            }
            variants.push(ExpertVariant {
                capacity,
                forwards: vec![Some(MicrobatchForward {
                    job: forward,
                    // An eager forward enters Python, which passes the host
                    // turn at its expert dispatch points.
                    yields: mode != GraphMode::Replay,
                })],
                then,
            });
            modes.push((Some(capacity), mode));
        }
        job.expert_step(ExpertStep {
            union: Arc::clone(union),
            lane,
            tokens: extent,
            kind: expert.kind,
            forced: expert.forced,
            variants,
            after: vec![None],
        });
        Ok((modes, completed))
    }

    /// Apply the completion of the job that ran `recorded`: store the
    /// graphs its captures produced and report how its forward ran.
    fn complete(&mut self, recorded: Recorded, completion: &mut Completion) -> Result<Executed> {
        let Recorded {
            key,
            modes,
            finish,
            captured,
            tokens,
            padded_tokens,
        } = recorded;
        let capacity = match modes.first() {
            Some((None, _)) => None,
            _ => Some(completion.capacity()?),
        };
        // How a method ran at the capacity the step was agreed at.
        let ran = |modes: &Modes| -> Result<GraphMode> {
            modes
                .iter()
                .find(|(variant, _)| *variant == capacity)
                .map(|&(_, mode)| mode)
                .ok_or_else(|| {
                    Error::Invariant("an expert step ran at an unplanned capacity".into())
                })
        };
        let mode = ran(&modes)?;
        let finished = match &finish {
            Some((key, modes)) => Some((*key, ran(modes)?)),
            None => None,
        };
        let mut store = |key: ShapeKey, capacity: Option<usize>| -> Result<()> {
            let graph = completion.captured()?;
            self.segments = self.segments.max(graph.segments());
            self.graphs.insert(key, capacity, graph)
        };
        // Captures complete in job order: the step's method, its completing
        // `finish`, then the captures behind the step.
        if mode == GraphMode::Capture {
            store(key, capacity)?;
        }
        if let Some((key, GraphMode::Capture)) = finished {
            store(key, capacity)?;
        }
        for key in captured {
            store(key, None)?;
        }
        Ok(Executed {
            mode,
            tokens,
            padded_tokens,
        })
    }

    /// Record in `job` the evaluation of a readout and the read of every
    /// candidate's log-probability.
    ///
    /// `sink(range, values)` records the copy of the FP32 values of the
    /// candidates `range` (in row, slot and candidate order) out of
    /// `values`, which later operations of the job overwrite. Every
    /// pipeline stage receives the same values.
    fn readout(
        &self,
        pass: &ReadoutPass<'_>,
        expert: Option<&Expert<'_>>,
        job: &Job,
        mut sink: impl FnMut(Range<usize>, &StorageView) -> Result<()>,
    ) -> Result<Recorded> {
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

        // `attend`: the bucket holding the rows and tokens, or the live
        // shape without graphs. A graph's launches are sized for sequences
        // of a whole block.
        let graph = self.pool.is_some();
        let length = self.facts.sequence_length;
        let (rows, padded_tokens, visible, padding, widths) = if graph {
            let (rows, total) = self
                .shapes
                .attend(pass.rows.len(), tokens, pass.longest())?;
            (
                rows,
                total,
                length,
                shapes::attend_padding(pass.rows.len(), tokens, rows, total, length)?,
                self.graph_widths.clone(),
            )
        } else {
            (
                pass.rows.len(),
                tokens,
                pass.longest(),
                Vec::new(),
                pass.pages.iter().map(TablePages::width).collect(),
            )
        };
        let key = ShapeKey::Attend {
            rows,
            tokens: padded_tokens,
        };
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

        // `finish` completes the final layer at every slot's token row, once
        // per forward, on the final stage; padding slots read row zero.
        let slots = pass.slots.len();
        let finished_slots = if graph {
            self.shapes.finish(slots)?
        } else {
            slots
        };
        let last = self.facts.last_stage();
        let slot_rows = self.buffers.column("slot_rows")?.clone();
        if last {
            writes.write(&slot_rows, 0, &pass.slots)?;
            writes.zero(&slot_rows, slots, finished_slots - slots)?;
        }
        self.submit(writes, job)?;

        // The expert step carries the tokens and, when `finish` exchanges,
        // the slot rows it completes, padding included.
        let local = self.facts.local_tail;
        let extent = if local {
            padded_tokens
        } else {
            padded_tokens.max(finished_slots)
        };
        let state = self.attend_state()?;
        let attention = self.attention_tensors()?;

        let mut plan: Vec<Argument> = vec![metadata.rows().into(), visible.into()];
        plan.extend(mirrors(&metadata));
        plan.extend(attention.iter().cloned());
        let (input_ids, positions) = (self.tensor("input_ids")?, self.tensor("positions")?);
        let attend = |capacity: usize| {
            let mut arguments: Vec<Argument> =
                vec![metadata.rows().into(), padded_tokens.into(), visible.into()];
            arguments.extend(mirrors(&metadata));
            arguments.push(capacity.into());
            arguments.push(input_ids.clone());
            arguments.push(positions.clone());
            arguments.extend(attention.iter().cloned());
            arguments.extend(state.iter().cloned());
            arguments
        };
        let finished = if last {
            self.tensor("finished")?
        } else {
            Argument::None
        };
        let finish = |capacity: usize| {
            let mut arguments: Vec<Argument> = vec![
                finished_slots.into(),
                capacity.into(),
                (&slot_rows).into(),
                finished.clone(),
            ];
            arguments.extend(state.iter().cloned());
            arguments
        };

        let attending = Method {
            key: &key,
            function: &self.functions.attend,
            plan: Some(&plan),
        };
        let finish_key = ShapeKey::Finish {
            slots: finished_slots,
        };
        let finishing = Method {
            key: &finish_key,
            function: &self.functions.finish,
            plan: None,
        };
        // A `finish` whose final layer exchanges tokens completes the expert
        // step `attend` began; a local one follows the step.
        let exchanging = last && !local;
        let completes = exchanging.then_some((&finishing, &finish as Arguments<'_>));
        let (modes, finish_modes) =
            self.step(&attending, &attend, completes, expert, extent, job)?;
        let mut captured = Vec::new();
        let mut capturing = HashSet::new();
        if last
            && local
            && self.method(&finishing, None, finish(0), job, job)? == GraphMode::Capture
        {
            capturing.insert(finish_key);
            captured.push(finish_key);
        }

        // One `readout` per slot chunk and window of its candidates: the
        // final stage projects the chunk's finished rows, and every stage
        // receives the candidate values the final stage broadcasts.
        let finished_rows = self.buffers.column("finished_rows")?.clone();
        let index = self.buffers.column("candidate_index")?.clone();
        let values = self.buffers.column("candidate_logprobs")?.clone();
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
        for (start, live, bucket) in shapes::slot_chunks(slots) {
            let first = pass.slot_candidates[start];
            let end_candidate = pass.slot_candidates[start + live];
            let chunk: Vec<i64> = (start..start + live).map(|slot| slot as i64).collect();
            for window in (first..end_candidate).step_by(CANDIDATES) {
                let end = (window + CANDIDATES).min(end_candidate);
                let selected = flat(window..end, start);
                let mut writes = ColumnWrites::new();
                writes.write(&finished_rows, 0, &chunk)?;
                writes.zero(&finished_rows, live, bucket - live)?;
                // Unused candidates select the first log-probability.
                writes.write(&index, 0, &selected)?;
                writes.zero(&index, selected.len(), CANDIDATES - selected.len())?;
                self.submit(writes, job)?;
                let arguments: Vec<Argument> = vec![
                    bucket.into(),
                    CANDIDATES.into(),
                    (&finished_rows).into(),
                    finished.clone(),
                    (&index).into(),
                    (&values).into(),
                ];
                let key = ShapeKey::Readout { slots: bucket };
                if capturing.contains(&key) {
                    // A bucket this startup job already captures is stored
                    // only at its completion, so its later chunks run its
                    // method again.
                    job.call(self.methods.call(&self.functions.readout, arguments));
                } else {
                    let reading = Method {
                        key: &key,
                        function: &self.functions.readout,
                        plan: None,
                    };
                    if self.method(&reading, None, arguments, job, job)? == GraphMode::Capture {
                        capturing.insert(key);
                        captured.push(key);
                    }
                }
                sink(window..end, &values.prefix((end - window) as i64)?)?;
            }
        }
        Ok(Recorded {
            key,
            modes,
            finish: exchanging.then_some((finish_key, finish_modes)),
            captured,
            tokens,
            padded_tokens,
        })
    }

    /// Record in `job` one denoising step of the resident blocks of
    /// `rows`. The rows' int64 `[1 + length]` results are the leading rows
    /// of `denoise.results` and their continuation flags those of
    /// `denoise.continued`.
    fn denoise(
        &self,
        rows: &[DenoiseRow],
        pages: &[TablePages],
        expert: Option<&Expert<'_>>,
        job: &Job,
    ) -> Result<Recorded> {
        let (Some(banks), Some(state)) = (&self.banks, &self.state) else {
            return Err(Error::Invalid(
                "this worker keeps no generating blocks".into(),
            ));
        };
        let length = self.facts.sequence_length;
        let live = rows.len();
        // The smallest bucket holding the rows, or the live shape without
        // graphs.
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
        self.submit(writes, job)?;
        let slots: Vec<u32> = rows.iter().map(|row| row.slot).collect();
        banks.gather(&slots, padded, state, job.on(self.stream.as_ref()))?;

        let first = rows.iter().all(|row| row.step == 0);
        let attention = self.attention_tensors()?;
        let mut plan: Vec<Argument> = vec![padded.into(), length.into()];
        plan.extend(mirrors(&metadata));
        plan.extend(attention.iter().cloned());
        let columns = [self.tensor("denoise.columns")?, self.tensor("positions")?];
        let outputs = [
            "denoise.tokens",
            "denoise.history",
            "denoise.self_conditioning",
            "denoise.results",
            "denoise.continued",
        ]
        .into_iter()
        .map(|name| self.tensor(name))
        .collect::<Result<Vec<_>>>()?;
        let arguments = |capacity: usize| {
            let mut arguments: Vec<Argument> = vec![padded.into(), first.into()];
            arguments.extend(mirrors(&metadata));
            arguments.push(capacity.into());
            arguments.extend(columns.iter().cloned());
            arguments.extend(attention.iter().cloned());
            arguments.extend(outputs.iter().cloned());
            arguments
        };
        let key = ShapeKey::Denoise {
            rows: padded,
            first,
        };
        let denoising = Method {
            key: &key,
            function: &self.functions.denoise,
            plan: Some(&plan),
        };
        let (modes, _) = self.step(&denoising, &arguments, None, expert, padded * length, job)?;
        Ok(Recorded {
            key,
            modes,
            finish: None,
            captured: Vec::new(),
            tokens: live * length,
            padded_tokens: padded * length,
        })
    }

    /// Run every forward shape serving can select, over scratch tables of
    /// the prefix cache: the scratch tables of slots `1..=n`, `n` the blocks
    /// a denoising step holds, each hold a one-token prefix, one page of
    /// every cache group, and row `r` of a synthetic forward reads the table
    /// of slot `r % n + 1`, staged as serving stages a readout's rows (whose
    /// rows of one request share its slot). Denoising steps run in slots
    /// `1..`; their synthetic state is harmless because a request's first
    /// step starts its block before reading any of it.
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
    ///
    /// Every forward is a job of the runner's lane (`lanes`). The scratch
    /// tables are startup work this call issues itself, between jobs.
    fn warm(
        &mut self,
        prefix_cache: &mut PrefixCache,
        experts: Option<&Arc<ExpertUnion>>,
        lanes: &JobQueues,
    ) -> Result<()> {
        // One page of every group for each of the largest denoising
        // bucket's rows, as resident tables of slots `1..`.
        let rows = self.shapes.capacity();
        let scratch = prefix_cache.install_scratch(&vec![1; rows], self.stream.as_ref())?;
        let warmed = self.warm_shapes(prefix_cache, rows, experts, lanes);
        let released = prefix_cache.release_scratch(scratch, rows, self.stream.as_ref());
        warmed.and(released)?;
        if let Some(stream) = &self.stream {
            stream.synchronize()?;
        }
        let plans = self
            .methods
            .call(&self.functions.plans_attention, Vec::new())
            .invoke()?;
        self.plans_attention = bool::try_from(plans)
            .map_err(|_| Error::Interface(format!("{RUNNER}.plans_attention returns a flag")))?;

        let mut counts = [0; 4];
        for (key, _) in self.graphs.keys() {
            counts[match key {
                ShapeKey::Attend { .. } => 0,
                ShapeKey::Finish { .. } => 1,
                ShapeKey::Readout { .. } => 2,
                ShapeKey::Denoise { .. } => 3,
            }] += 1;
        }
        tracing::info!(
            "{}: captured {} graphs of up to {} segments: {} attend, {} finish, {} readout and {} denoise buckets; attention launches are planned on the {}",
            self.owner,
            counts.iter().sum::<usize>(),
            self.segments,
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            if self.plans_attention {
                "host"
            } else {
                "device"
            },
        );
        Ok(())
    }

    /// Submit `job`, which recorded `recorded`, to the runner's lane and
    /// apply its completion.
    fn perform(&mut self, lanes: &JobQueues, job: Job, recorded: Recorded) -> Result<()> {
        let completion = lanes.block_on(KIND, job, BLOCK_ON_TIMEOUT)?;
        let mut completion = completion.check().map_err(|failure| failure.error)?;
        self.complete(recorded, &mut completion).map(drop)
    }

    fn warm_shapes(
        &mut self,
        prefix_cache: &mut PrefixCache,
        installed: usize,
        experts: Option<&Arc<ExpertUnion>>,
        lanes: &JobQueues,
    ) -> Result<()> {
        // The expert step of a warmup forward at `capacity`.
        let index = self.index;
        let step = |capacity: Option<usize>| {
            experts.map(|union| Expert {
                union,
                kind: batch::expert_kind(index),
                forced: capacity,
            })
        };
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
                let pages = scratch_pages(prefix_cache, &vec![length; count], installed)?;
                let job = Job::new();
                let recorded = self.denoise(
                    &synthetic_steps(count, true),
                    &pages,
                    step(None).as_ref(),
                    &job,
                )?;
                self.perform(lanes, job, recorded)?;
            }
            return Ok(());
        }

        // The graphs that run alone: every stage's `readout`, and on the
        // final stage the `finish` of a final layer that exchanges no
        // tokens.
        let local = self.facts.local_tail;
        let last = self.facts.last_stage();
        for slots in SLOT_BUCKETS.into_iter().rev() {
            self.warm_readout(slots, lanes)?;
        }
        if local && last {
            for slots in self.shapes.finish_buckets().to_vec().into_iter().rev() {
                self.warm_finish(slots, lanes)?;
            }
        }

        // Every `attend` bucket, filled by blocks of one slot each.
        let buckets: Vec<(usize, usize)> = self.shapes.attend_buckets().collect();
        for &(count, tokens) in buckets.iter().rev() {
            let queries = spread(tokens, count);
            let pages = scratch_pages(prefix_cache, &queries, installed)?;
            let mut slots = Vec::with_capacity(count);
            let mut offset = 0;
            for &query in &queries {
                slots.push(offset as i64);
                offset += query;
            }
            let pass = ReadoutPass {
                rows: queries
                    .iter()
                    .map(|&query| Sequence {
                        query,
                        prefix: 1,
                        position: 1,
                    })
                    .collect(),
                tokens: vec![0; tokens],
                slots,
                slot_candidates: (0..=count).collect(),
                candidates: vec![0; count],
                pages: &pages,
            };
            let extent = if local {
                tokens
            } else {
                tokens.max(self.shapes.finish(count)?)
            };
            for capacity in variants(extent) {
                let job = Job::new();
                let recorded = self.readout(&pass, step(capacity).as_ref(), &job, |_, _| Ok(()))?;
                self.perform(lanes, job, recorded)?;
            }
        }

        // A `finish` whose final layer exchanges tokens runs in the expert
        // step of its readout: each slot bucket is captured at every
        // capacity holding the bucket, in a pass of one shortest block whose
        // fewest slots for the bucket read its token rows in turn. Every
        // stage runs the passes, whose `attend` spans the pipeline.
        if !local {
            let finish = self.shapes.finish_buckets().to_vec();
            let (_, short) = self.shapes.attend(1, 1, 1)?;
            let pages = scratch_pages(prefix_cache, &[short], installed)?;
            for bucket in (0..finish.len()).rev() {
                let slots = if bucket == 0 {
                    1
                } else {
                    finish[bucket - 1] + 1
                };
                let pass = ReadoutPass {
                    rows: vec![Sequence {
                        query: short,
                        prefix: 1,
                        position: 1,
                    }],
                    tokens: vec![0; short],
                    slots: (0..slots).map(|slot| (slot % short) as i64).collect(),
                    slot_candidates: (0..=slots).collect(),
                    candidates: vec![0; slots],
                    pages: &pages,
                };
                for capacity in variants(short.max(finish[bucket])) {
                    let job = Job::new();
                    let recorded =
                        self.readout(&pass, step(capacity).as_ref(), &job, |_, _| Ok(()))?;
                    self.perform(lanes, job, recorded)?;
                }
            }
        }

        if self.sampling.is_some() {
            let rows: Vec<usize> = self.shapes.rows().iter().rev().copied().collect();
            for &count in &rows {
                let pages = scratch_pages(prefix_cache, &vec![length; count], installed)?;
                let padded = self.shapes.denoise(count)?;
                for first in [true, false] {
                    for capacity in variants(padded * length) {
                        let job = Job::new();
                        let recorded = self.denoise(
                            &synthetic_steps(count, first),
                            &pages,
                            step(capacity).as_ref(),
                            &job,
                        )?;
                        self.perform(lanes, job, recorded)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Capture the `readout` graph of `slots` finished rows alone.
    fn warm_readout(&mut self, slots: usize, lanes: &JobQueues) -> Result<()> {
        let finished_rows = self.buffers.column("finished_rows")?.clone();
        let index = self.buffers.column("candidate_index")?.clone();
        let mut writes = ColumnWrites::new();
        writes.write(&finished_rows, 0, &(0..slots as i64).collect::<Vec<_>>())?;
        writes.zero(&index, 0, CANDIDATES)?;
        let finished = if self.facts.last_stage() {
            self.tensor("finished")?
        } else {
            Argument::None
        };
        let arguments: Vec<Argument> = vec![
            slots.into(),
            CANDIDATES.into(),
            (&finished_rows).into(),
            finished,
            (&index).into(),
            self.tensor("candidate_logprobs")?,
        ];
        let key = ShapeKey::Readout { slots };
        self.warm_alone(
            key,
            |functions| &functions.readout,
            writes,
            arguments,
            lanes,
        )
    }

    /// Capture the `finish` graph of `slots` slot rows alone, on the final
    /// stage of a model whose final layer exchanges no tokens: it has no
    /// expert capacity.
    fn warm_finish(&mut self, slots: usize, lanes: &JobQueues) -> Result<()> {
        let slot_rows = self.buffers.column("slot_rows")?.clone();
        let mut writes = ColumnWrites::new();
        writes.zero(&slot_rows, 0, slots)?;
        let mut arguments: Vec<Argument> = vec![
            slots.into(),
            0usize.into(),
            (&slot_rows).into(),
            self.tensor("finished")?,
        ];
        arguments.extend(self.attend_state()?);
        let key = ShapeKey::Finish { slots };
        self.warm_alone(key, |functions| &functions.finish, writes, arguments, lanes)
    }

    /// Capture the graph of `key` alone, as a job of the runner's lane that
    /// performs `writes`, then calls the method `function` names with
    /// `arguments`.
    fn warm_alone(
        &mut self,
        key: ShapeKey,
        function: fn(&Functions) -> &Function,
        writes: ColumnWrites,
        arguments: Vec<Argument>,
        lanes: &JobQueues,
    ) -> Result<()> {
        let job = Job::new();
        self.submit(writes, &job)?;
        let method = Method {
            key: &key,
            function: function(&self.functions),
            plan: None,
        };
        let mode = self.method(&method, None, arguments, &job, &job)?;
        let completion = lanes.block_on(KIND, job, BLOCK_ON_TIMEOUT)?;
        let mut completion = completion.check().map_err(|failure| failure.error)?;
        if mode == GraphMode::Capture {
            let graph = completion.captured()?;
            self.segments = self.segments.max(graph.segments());
            self.graphs.insert(key, None, graph)?;
        }
        Ok(())
    }
}

/// `tokens` tokens over `rows` blocks, the first `tokens % rows` one token
/// longer than the others.
fn spread(tokens: usize, rows: usize) -> Vec<usize> {
    let (each, longer) = (tokens / rows, tokens % rows);
    (0..rows)
        .map(|row| each + usize::from(row < longer))
        .collect()
}

/// The pages rows of `queries` tokens read over a one-token prefix, row `r`
/// through the scratch table of slot `r % installed + 1`, selected from the
/// resident tables as a serving readout's are.
fn scratch_pages(
    prefix_cache: &mut PrefixCache,
    queries: &[usize],
    installed: usize,
) -> Result<Vec<TablePages>> {
    let rows: Vec<AttentionRow> = queries
        .iter()
        .enumerate()
        .map(|(row, &query)| AttentionRow {
            slot: (row % installed.max(1)) as u32 + 1,
            prefix: 1,
            query: query as u32,
            write: false,
        })
        .collect();
    prefix_cache.prepare_attention(&rows)
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
        let lanes = backend.job_queues().clone();
        let experts = backend.experts().cloned();
        let parts = backend.parts();
        let Some(storage) = parts.storage else {
            return Ok(());
        };
        for runner in parts.diffusion_lm {
            let prefix_cache = storage
                .prefix_cache
                .as_mut()
                .ok_or(Error::State("diffusion LM warmup requires a prefix cache"))?;
            runner
                .warm(prefix_cache, experts.as_ref(), &lanes)
                .map_err(|error| {
                    WorkerError::classify(error, Some("diffusion LM graph capture"))
                        .with_phase("warmup")
                })?;
        }
        Ok(())
    }

    fn steps_experts(&self) -> bool {
        true
    }
}

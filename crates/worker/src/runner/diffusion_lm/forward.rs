//! One batch of the diffusion LM runner: readouts or denoising steps.
//!
//! A batch runs in four stages on its lane's execution thread:
//!
//! 1. **Reservation.** Bind every call's pending output and the batch's
//!    output lease; install its block tables and retain the prefix cache
//!    spans it reads until its device work retires; bind the completion
//!    flags of its denoising steps. A readout whose host predicate is false
//!    is skipped (`Predicated`). A denoising step's predicate is never
//!    read: whether its block continues is decided on the device from its
//!    slot's resident continuation flag, so a step queued behind the one
//!    that stopped its block runs as a no-op and reports itself skipped.
//! 2. **Forward.** Validate the calls into one readout or one list of
//!    denoising rows (a batch carries one form) and run it. A step advances
//!    its request's submitted block and step.
//! 3. **Outputs and commit.** Copy candidate values or step results into
//!    the output lease, publish the steps' completion flags, then commit
//!    products and pending calls together. A failure before the commit
//!    point retires everything the batch reserved and reports every call as
//!    failed; a failure after it is fatal.
//! 4. **Completion** ([`crate::executor::forward::launch::materialize`]).
//!    Once the lease's device fences complete, decode each call's result
//!    and accept its progress. These calls write no prefix cache: each
//!    keeps its visible length.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferId, CallKind, CallStatus, DType, ForwardStats, RequestKey, TensorRef,
    WorkerInfo,
};

use super::calls::{self, DenoiseRow, ReadoutPass, Readouts};
use super::{DiffusionLMRunner, Executed, KIND};
use crate::cuda_graph::GraphMode;
use crate::error::WorkerError;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::forward::experts::ExpertRuntime;
use crate::executor::forward::launch::{
    Launched, StreamOrder, error_code, failed_output, lock, read_flag, reports_output, seal,
};
use crate::executor::lifecycle::RequestStorage;
use crate::executor::{BatchResult, PendingOutput};
use crate::ffi::Device;
use crate::request::RequestPool;
use crate::storage::{KVCache, OutputLease, OutputPlacement, ProductBuffer, TensorStore};
use crate::stream::EventPool;
use crate::transfer::Retirement;
use crate::worker::RankStorage;
use crate::{Error, Result};

/// What a batch borrows from the rank.
struct Context<'a> {
    pub(crate) info: &'a WorkerInfo,
    pub(crate) rank: usize,
    pub(crate) requests: &'a mut RequestPool,
    pub(crate) kv: &'a mut KVCache,
    pub(crate) tensors: &'a TensorStore,
    pub(crate) events: &'a Arc<EventPool>,
    /// The text runtime's batch order: these forwards read the prefix
    /// cache its prefills write, possibly on another stream.
    pub(crate) text: Option<&'a StreamOrder>,
    /// The rank's expert runtime, when forwards are expert steps.
    pub(crate) experts: Option<&'a ExpertRuntime>,
}

/// The expert call kind of diffusion LM runner `index`: distinct from the
/// text runtime's kinds, so the union's ready kinds take turns.
fn expert_kind(index: usize) -> i64 {
    64 + index as i64
}

/// The coordinator of runner `index`'s expert steps: agreement while
/// serving, announced or `forced` capacities during warmup.
pub(super) fn coordinator<'a>(
    experts: Option<&'a ExpertRuntime>,
    index: usize,
    forced: Option<usize>,
) -> impl FnMut(usize) -> Result<Option<usize>> + 'a {
    move |extent| {
        experts
            .map(|experts| experts.capacity(extent, expert_kind(index), forced))
            .transpose()
    }
}

/// A failure, the batch phase it happened in and whether the batch had
/// passed its commit point.
struct Failure {
    error: Error,
    phase: &'static str,
    committed: bool,
}

impl Failure {
    fn new(phase: &'static str) -> impl Fn(Error) -> Self {
        move |error| Self {
            error,
            phase,
            committed: false,
        }
    }

    fn fatal(error: Error) -> Self {
        Self {
            error,
            phase: "batch commit",
            committed: true,
        }
    }
}

/// One call's state through the batch.
struct CallState {
    output: PendingOutput,
    slot: u32,
    active: bool,
    completion: Option<ProductBuffer>,
}

/// Resources a batch reserved, retired on failure.
struct Reserved {
    writes: Vec<ProductBuffer>,
    /// Signals the physical end of the batch's KV reads.
    retirement: Arc<Retirement>,
}

impl DiffusionLMRunner {
    /// Run one batch of diffusion LM calls up to its commit point.
    fn forward(
        &mut self,
        context: Context<'_>,
        plan: &BatchPlan,
    ) -> std::result::Result<Launched, Failure> {
        let started = Instant::now();
        let reserve = Failure::new("batch registration");
        let Context {
            info,
            rank,
            requests,
            kv,
            tensors,
            events,
            text,
            experts,
        } = context;
        let calls = &plan.calls;
        if calls
            .iter()
            .any(|call| call.code != KIND || call.component != self.component)
        {
            return Err(reserve(Error::Invalid(format!(
                "a diffusion LM batch carries token-denoising calls of component {:?}",
                self.component
            ))));
        }
        let device = self.buffers.device();
        let stream = self.stream.clone();
        let stream = stream.as_ref();
        let reports = reports_output(info, rank, &self.component).map_err(&reserve)?;

        // Output rows, bound to their requests' live epochs and slots.
        let bound = calls
            .iter()
            .map(|call| {
                let request = requests.get(call.request_key.request_id.0)?;
                Ok((call.request_key, call.call_id, request.slot()))
            })
            .collect::<Result<Vec<_>>>()
            .and_then(|rows| requests.bind_calls(&rows))
            .map_err(&reserve)?;
        let mut states = Vec::with_capacity(calls.len());
        for (row, (call, request)) in calls.iter().zip(&bound).enumerate() {
            states.push(CallState {
                output: PendingOutput::new(call, request.progress().map_err(&reserve)?, row)
                    .map_err(&reserve)?,
                slot: request.slot() as u32,
                active: true,
                completion: None,
            });
        }

        // Readouts gate on host predicates. A step's continuation stays on
        // the device: reading it here would serialize steps that can run.
        if let Some(text) = text {
            text.wait(stream).map_err(&reserve)?;
        }
        for (call, state) in calls.iter().zip(&mut states) {
            if let Some(predicate) = &call.predicate
                && predicate.dtype == DType::U8
                && call.canvas.is_none()
                && !read_flag(tensors, call, predicate, device, stream).map_err(&reserve)?
            {
                state.active = false;
                state.output.output.status = CallStatus::Predicated;
            }
        }

        // The output lease: a step's outcome and block, or a readout's
        // candidate values, per call.
        let length = self.facts.sequence_length;
        let words = calls.iter().fold(0usize, |total, call| {
            total.saturating_add(match &call.readout {
                Some(readout) => readout.candidate_count().max(1),
                None => 1 + length,
            })
        });
        let devices: Vec<i32> = device.into_iter().collect();
        let buffer = self
            .outputs
            .lease(calls.len(), words.max(calls.len()), &devices, false)
            .map_err(&reserve)?;
        if let Some(stream) = stream {
            lock(&buffer)
                .begin_device(device, stream, events)
                .map_err(&reserve)?;
        }

        let mut reserved = Reserved {
            writes: Vec::new(),
            retirement: Arc::new(Retirement::default()),
        };
        let batch = Batch {
            runner: self,
            experts,
            requests,
            kv,
            tensors,
            events,
            plan,
            stream,
            device,
            reports,
        };
        match batch.run(&mut states, &buffer, &mut reserved) {
            Ok(mut stats) => {
                stats.component_us.insert(
                    "canvas_execute".into(),
                    started.elapsed().as_micros() as u64,
                );
                Ok(Launched {
                    outputs: states.into_iter().map(|state| state.output).collect(),
                    buffer,
                    started,
                    stats,
                    timer: "canvas_materialize",
                    host: None,
                })
            }
            Err(failure) if failure.committed => Err(failure),
            Err(failure) => {
                // Retire every reservation on the consuming stream; storage
                // returns once the fences recorded here complete.
                let abandoned = tensors.abandon_writes(&reserved.writes);
                let sealed = if lock(&buffer).sealed() {
                    Ok(())
                } else {
                    seal(events, &buffer, stream, &reserved.retirement)
                };
                lock(&buffer).abandon();
                if let Err(cleanup) = abandoned.and(sealed) {
                    return Err(Failure {
                        error: cleanup,
                        phase: "batch cleanup",
                        committed: true,
                    });
                }
                Err(failure)
            }
        }
    }
}

impl RequestStorage for DiffusionLMRunner {
    fn admit_slots(&mut self, _slots: &[usize]) -> Result<()> {
        // A request's first denoising step starts its block in the banks
        // before any step reads them; admission resets nothing.
        Ok(())
    }

    fn release_buffers(&mut self, _buffers: &HashSet<BufferId>) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<()> {
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        _buffers: &HashSet<BufferId>,
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(true)
    }

    fn retire_slots(&mut self, _slots: &[usize]) -> Result<()> {
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        self.outputs.reap();
        Ok(())
    }
}

/// One batch on the runner.
struct Batch<'a> {
    runner: &'a mut DiffusionLMRunner,
    experts: Option<&'a ExpertRuntime>,
    requests: &'a mut RequestPool,
    kv: &'a mut KVCache,
    tensors: &'a TensorStore,
    events: &'a Arc<EventPool>,
    plan: &'a BatchPlan,
    stream: Option<&'a Stream>,
    device: Option<i32>,
    reports: bool,
}

impl Batch<'_> {
    fn run(
        mut self,
        states: &mut [CallState],
        buffer: &OutputLease,
        reserved: &mut Reserved,
    ) -> std::result::Result<ForwardStats, Failure> {
        let reserve = Failure::new("batch registration");
        let plan = self.plan;
        let calls = &plan.calls;
        let slots: HashSet<u32> = states
            .iter()
            .filter(|state| state.active)
            .map(|state| state.slot)
            .collect();
        if plan
            .new_cache_units
            .iter()
            .any(|allocation| slots.contains(&allocation.request_pool_idx))
        {
            return Err(reserve(Error::Invalid(
                "diffusion LM calls write no prefix cache and receive no new units".into(),
            )));
        }
        let assignments: Vec<_> = plan
            .block_tables
            .iter()
            .filter(|table| slots.contains(&table.request_pool_idx))
            .cloned()
            .collect();
        self.kv
            .install(&assignments, self.stream)
            .map_err(&reserve)?;

        // Every active call retains the prefix it reads until the batch's
        // device work retires. A windowed group starts at its first
        // resident page.
        for (call, state) in calls.iter().zip(states.iter()) {
            if !state.active {
                continue;
            }
            let visible = state.output.progress.kv_visible_len;
            let tables = self.kv.tables();
            tables.coordinates(state.slot, visible).map_err(&reserve)?;
            let mut spans = Vec::new();
            for group in 0..tables.groups().len() as u32 {
                let table = tables.table(state.slot, group).map_err(&reserve)?;
                let start = u64::from(table.start_page) * u64::from(table.shape.page_tokens);
                if visible > start {
                    spans.extend(table.spans(start, visible - start).map_err(&reserve)?);
                }
            }
            self.kv.manager().retain_execution(
                call.request_key,
                &spans,
                Arc::clone(&reserved.retirement),
            );
        }

        // Completion flags of denoising steps; these calls publish nothing
        // else.
        let requests: HashMap<RequestKey, usize> = calls
            .iter()
            .zip(states.iter())
            .map(|(call, state)| (call.request_key, state.slot as usize))
            .collect();
        let target = self.device.map_or(Device::Cpu, Device::Cuda);
        let mut flags: Vec<(TensorRef, Device)> = Vec::new();
        for call in calls {
            if !call.outputs.is_empty()
                || call.token_output.is_some()
                || call.transition_output.is_some()
                || call.image_output.is_some()
                || call.encoder_output.is_some()
                || call.kv_output.is_some()
            {
                return Err(reserve(Error::Unsupported(
                    "token-denoising calls publish only a completion flag".into(),
                )));
            }
            if let Some(flag) = &call.completion_output {
                if flag.dtype != DType::U8 {
                    return Err(reserve(Error::Invalid(
                        "a denoising step's completion flag is a uint8 scalar".into(),
                    )));
                }
                flags.push((flag.clone(), target));
            }
        }
        let bound = self
            .tensors
            .bind_output_groups(
                &[flags],
                OutputPlacement {
                    request_slots: Some(&requests),
                    ..OutputPlacement::default()
                },
            )
            .map_err(&reserve)?;
        for write in bound.into_iter().flatten() {
            let id = write.id();
            reserved.writes.push(write.clone());
            if let Some(state) = calls
                .iter()
                .zip(states.iter_mut())
                .find(|(call, _)| {
                    call.completion_output
                        .as_ref()
                        .is_some_and(|flag| flag.buffer_id() == id)
                })
                .map(|(_, state)| state)
            {
                state.completion = Some(write);
            }
        }
        // Skipped calls still forward a false decision to their consumers.
        for state in states.iter().filter(|state| !state.active) {
            if let Some(write) = &state.completion {
                self.tensors
                    .write_scalar(write, crate::storage::Scalar::Int(0), self.stream)
                    .map_err(&reserve)?;
            }
        }

        let active: Vec<usize> = (0..calls.len()).filter(|&row| states[row].active).collect();
        let execute = Failure::new("batch execution");
        let mut stats = ForwardStats::default();
        if let Some(&first) = active.first() {
            let steps = calls[first].canvas.is_some();
            if active
                .iter()
                .any(|&row| calls[row].canvas.is_some() != steps)
            {
                return Err(execute(Error::Invalid(
                    "a batch carries readouts or denoising steps, not both".into(),
                )));
            }
            let rows = calls::rows_by_call(plan).map_err(&execute)?;
            let mode = if steps {
                self.steps(states, &active, &rows, buffer)?
            } else {
                self.readouts(states, &active, &rows, buffer)?
            };
            record(&mut stats, mode);
        }
        self.commit(states, buffer, reserved)?;
        Ok(stats)
    }

    /// Run the active readouts as one pass and capture their candidates.
    fn readouts(
        &mut self,
        states: &mut [CallState],
        active: &[usize],
        rows: &[Vec<usize>],
        buffer: &OutputLease,
    ) -> std::result::Result<Executed, Failure> {
        let execute = Failure::new("batch execution");
        let calls = &self.plan.calls;
        let mut readouts = Readouts::new();
        let mut ranges = Vec::with_capacity(active.len());
        for &row in active {
            let state = &states[row];
            let progress = state.output.progress;
            ranges.push(
                readouts
                    .add(
                        &self.plan.forward,
                        &calls[row],
                        &rows[row],
                        state.slot,
                        progress.kv_visible_len,
                        progress.logical_position,
                    )
                    .map_err(&execute)?,
            );
        }
        let pages = self
            .kv
            .tables()
            .prepare_attention(&readouts.attention, |_, _, _| Ok(()))
            .map_err(&execute)?;
        let pass = ReadoutPass {
            rows: std::mem::take(&mut readouts.rows),
            tokens: std::mem::take(&mut readouts.tokens),
            slots: std::mem::take(&mut readouts.slots),
            slot_candidates: std::mem::take(&mut readouts.slot_candidates),
            candidates: std::mem::take(&mut readouts.candidates),
            pages: &pages,
        };

        // Candidate values arrive in passes over the bound buffer; each pass
        // is copied to the lease per call, so a call's words stay adjacent.
        let mut spans: Vec<Option<(usize, usize)>> = vec![None; active.len()];
        let (stream, events) = (self.stream, self.events);
        let mut coordinate = coordinator(self.experts, self.runner.index, None);
        let executed = {
            let mut output = lock(buffer);
            self.runner
                .readout(&pass, &mut coordinate, |range, values| {
                    for (call, candidates) in ranges.iter().enumerate() {
                        let start = range.start.max(candidates.start);
                        let end = range.end.min(candidates.end);
                        if start >= end {
                            continue;
                        }
                        // FP32 values read back as their 32-bit patterns.
                        let source = values
                            .narrow(0, (start - range.start) as i64, (end - start) as i64)?
                            .reshape(crate::ffi::DType::U32, &[(end - start) as i64])?;
                        let (offset, count) = output.capture(&source, stream, events)?;
                        spans[call] = Some(match spans[call] {
                            None => (offset, count),
                            Some((first, total)) if first + total == offset => {
                                (first, total + count)
                            }
                            Some(_) => {
                                return Err(Error::Invariant(
                                    "a readout's candidate words are not adjacent".into(),
                                ));
                            }
                        });
                    }
                    Ok(())
                })
        }
        .map_err(&execute)?;
        for ((&row, span), candidates) in active.iter().zip(spans).zip(&ranges) {
            let state = &mut states[row];
            let (offset, count) = span.unwrap_or((0, 0));
            if count != candidates.len() {
                return Err(execute(Error::Invariant(
                    "a readout lost candidate values".into(),
                )));
            }
            state.output.candidate_range = Some((offset, count));
            let visible = state.output.progress.kv_visible_len;
            state.output.set_cache_length(visible);
        }
        Ok(executed)
    }

    /// Step the active calls' blocks once and capture their results.
    fn steps(
        &mut self,
        states: &mut [CallState],
        active: &[usize],
        rows: &[Vec<usize>],
        buffer: &OutputLease,
    ) -> std::result::Result<Executed, Failure> {
        let execute = Failure::new("batch execution");
        let calls = &self.plan.calls;
        let served = self.runner.sampling;
        let length = self.runner.facts.sequence_length;
        let mut steps: Vec<DenoiseRow> = Vec::with_capacity(active.len());
        let mut attention = Vec::with_capacity(active.len());
        for &row in active {
            let call = &calls[row];
            let state = &states[row];
            let request = self
                .requests
                .get(call.request_key.request_id.0)
                .map_err(&execute)?;
            let progress = state.output.progress;
            let (coordinates, step) = calls::step_row(
                &self.plan.forward,
                call,
                &rows[row],
                request.admission(),
                served.as_ref(),
                progress.kv_visible_len,
                progress.logical_position,
            )
            .map_err(&execute)?;
            self.requests
                .advance_canvas(call.request_key, coordinates)
                .map_err(&execute)?;
            attention.push(crate::storage::AttentionRow {
                slot: step.slot,
                prefix: step.prefix,
                query: length as u32,
                write: false,
            });
            steps.push(step);
        }
        let pages = self
            .kv
            .tables()
            .prepare_attention(&attention, |_, _, _| Ok(()))
            .map_err(&execute)?;
        let mut coordinate = coordinator(self.experts, self.runner.index, None);
        let executed = self
            .runner
            .denoise(&steps, &pages, &mut coordinate)
            .map_err(&execute)?;

        // One copy of every row's `[outcome | block]` result.
        let results = self
            .runner
            .buffers
            .column("denoise.results")
            .and_then(|results| results.prefix(steps.len() as i64))
            .map_err(&execute)?;
        let (offset, _) = lock(buffer)
            .capture(&results, self.stream, self.events)
            .map_err(&execute)?;
        let width = 1 + length;
        for (position, &row) in active.iter().enumerate() {
            let state = &mut states[row];
            state.output.canvas_range = Some((offset + position * width, width));
            let visible = state.output.progress.kv_visible_len;
            state.output.set_cache_length(visible);
        }

        // Publish each step's continuation flag, in runs of adjacent rows.
        let continued = self
            .runner
            .buffers
            .column("denoise.continued")
            .map_err(&execute)?;
        let mut start = 0;
        while start < active.len() {
            if states[active[start]].completion.is_none() {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < active.len() && states[active[end]].completion.is_some() {
                end += 1;
            }
            let writes: Vec<ProductBuffer> = active[start..end]
                .iter()
                .filter_map(|&row| states[row].completion.clone())
                .collect();
            let values = continued
                .narrow(0, start as i64, (end - start) as i64)
                .and_then(|values| {
                    values.reshape(crate::ffi::DType::U8, &[(end - start) as i64, 1])
                })
                .map_err(&execute)?;
            self.tensors
                .write_scalars(&writes, &values, self.stream)
                .map_err(&execute)?;
            start = end;
        }
        Ok(executed)
    }

    /// Validate every owner's update, then commit them together.
    fn commit(
        &mut self,
        states: &mut [CallState],
        buffer: &OutputLease,
        reserved: &mut Reserved,
    ) -> std::result::Result<(), Failure> {
        let fail = Failure::new("batch commit");
        {
            let output = lock(buffer);
            for (state, call) in states.iter().zip(&self.plan.calls) {
                state.output.validate_output(&output, call).map_err(&fail)?;
            }
        }
        self.tensors
            .validate_writes(&reserved.writes)
            .map_err(&fail)?;
        seal(self.events, buffer, self.stream, &reserved.retirement).map_err(&fail)?;
        for state in states.iter_mut() {
            state.output.reports_output = self.reports;
        }
        let pending: Vec<_> = self
            .plan
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id, call.advances_state()))
            .collect();
        self.requests.validate_pending(&pending).map_err(&fail)?;

        // Every owner accepted its update. Any failure from here is fatal.
        let writes = std::mem::take(&mut reserved.writes);
        self.tensors
            .commit_writes(&writes)
            .map_err(Failure::fatal)?;
        self.requests.add_pending(&pending).map_err(Failure::fatal)
    }
}

fn record(stats: &mut ForwardStats, executed: Executed) {
    let mode = KIND.as_str();
    *stats.mode_counts.entry(mode.into()).or_default() += 1;
    *stats.mode_tokens.entry(mode.into()).or_default() += executed.tokens as u64;
    match executed.mode {
        GraphMode::Capture => stats.cuda_graph_captures += 1,
        GraphMode::Replay => stats.cuda_graph_replays += 1,
        GraphMode::Eager => {}
    }
    if executed.mode != GraphMode::Eager {
        *stats
            .cuda_graph_runtime_mode_counts
            .entry(mode.into())
            .or_default() += 1;
        stats.cuda_graph_unpadded_tokens += executed.tokens as u64;
        stats.cuda_graph_padded_tokens += executed.padded_tokens as u64;
    }
}

/// Execute a batch of `kind` calls: launch it, or report every call failed
/// when it fails before its commit point.
pub(crate) fn execute(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<(), WorkerError> {
    let plan = Arc::clone(batch.shared_plan());
    let predecessors = batch.predecessors().to_vec();
    let parts = backend.parts();
    // A batch carries calls of one component, which names its runner.
    let component = plan.calls.first().map(|call| call.component.as_str());
    let runner = parts
        .diffusion_lm
        .iter_mut()
        .find(|runner| Some(runner.component.as_str()) == component);
    let Some(runner) = runner else {
        return Err(WorkerError::classify(
            Error::Unsupported(format!(
                "component {:?} has no diffusion LM runner on this rank",
                component.unwrap_or_default()
            )),
            None,
        ));
    };
    let Some(storage) = parts.storage else {
        return Err(crate::executor::forward::unsupported(kind, "diffusion LM"));
    };
    let RankStorage {
        kv: Some(kv),
        tensors,
        events,
        ..
    } = storage
    else {
        return Err(crate::executor::forward::unsupported(kind, "diffusion LM"));
    };
    let context = Context {
        info: parts.info,
        rank: parts.rank,
        requests: parts.requests,
        kv,
        tensors,
        events,
        text: parts.text.map(|text| text.order()),
        experts: parts.experts,
    };
    let failure = match runner.forward(context, &plan) {
        Ok(launched) => {
            batch.launch(launched);
            return Ok(());
        }
        Err(failure) => failure,
    };
    let mut error = WorkerError::classify(failure.error, Some(failure.phase));
    if failure.committed {
        // Products and pending calls may already be visible; the rank cannot
        // roll them back.
        error.fatal = true;
    }
    if error.fatal {
        return Err(error);
    }
    tracing::warn!(
        target: "uniserve_worker.execution",
        "diffusion LM batch {} failed: {error}{}",
        plan.batch_id,
        error
            .traceback()
            .map(|trace| format!("\n{trace}"))
            .unwrap_or_default()
    );
    let output = failed_output(
        backend.requests(),
        &plan,
        &predecessors,
        error_code(error.code),
    )?;
    batch.complete(BatchResult {
        output,
        media: Vec::new(),
    });
    Ok(())
}

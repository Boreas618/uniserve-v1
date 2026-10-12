//! One batch of the diffusion LM runner: readouts or denoising steps.
//!
//! The control thread prepares a batch and applies its results; the lane's
//! execution thread issues its stream work, which the batch records as jobs
//! ([`forward`] prepares the next one, [`complete`] applies its
//! completion). A batch is one job, unless a readout has a host-gated
//! predicate: the opening job reads those, and its completion decides the
//! active calls. Its stages:
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
//!    denoising rows (a batch carries one form) and record it. A step
//!    advances its request's submitted block and step.
//! 3. **Outputs and commit.** Record the copies of candidate values or
//!    step results into the output lease and the publication of the steps'
//!    completion flags, validate every owner's update and seal the lease.
//!    Products and pending calls are committed together once the job
//!    completed. A failure before that commit retires everything the batch
//!    reserved and reports every call as failed; a failure of the commit
//!    itself is fatal.
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
use super::{DiffusionLMRunner, Executed, Expert, KIND, Recorded};
use crate::cuda_graph::GraphMode;
use crate::error::WorkerError;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::forward::Stage;
use crate::executor::forward::launch::{
    Launched, StreamOrder, error_code, failed_output, lock, reports_output, seal,
};
use crate::executor::lifecycle::RequestStorage;
use crate::executor::{BatchResult, PendingOutput};
use crate::ffi::Device;
use crate::lane::{Completion, Job};
use crate::parallel::ExpertUnion;
use crate::request::RequestPool;
use crate::storage::{KVCache, OutputLease, OutputPlacement, ProductBuffer, TensorStore};
use crate::stream::EventPool;
use crate::transfer::Retirement;
use crate::worker::RankStorage;
use crate::{Error, Result};

/// What a batch borrows from the rank.
struct Context<'a> {
    info: &'a WorkerInfo,
    rank: usize,
    runner: &'a mut DiffusionLMRunner,
    requests: &'a mut RequestPool,
    kv: &'a mut KVCache,
    tensors: &'a TensorStore,
    events: &'a Arc<EventPool>,
    plan: &'a BatchPlan,
    /// The text runtime's batch order: these forwards read the prefix
    /// cache its prefills write, possibly on another stream.
    text: Option<&'a StreamOrder>,
    /// The rank's expert union member, when forwards are expert steps.
    experts: Option<&'a Arc<ExpertUnion>>,
}

/// The expert call kind of diffusion LM runner `index`: distinct from the
/// text runtime's kinds, so the union's ready kinds take turns.
pub(super) fn expert_kind(index: usize) -> i64 {
    64 + index as i64
}

/// A failure, the batch phase it happened in and whether the batch had
/// begun its commit.
pub(crate) struct Failure {
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

    fn fatal(phase: &'static str) -> impl Fn(Error) -> Self {
        move |error| Self {
            error,
            phase,
            committed: true,
        }
    }
}

/// A batch whose outputs are bound, before its output storage is leased:
/// its host-gated predicates decide first which calls are active.
pub(crate) struct Opening {
    states: Vec<CallState>,
    /// The calls whose host-gated predicate the opening job reads, in the
    /// order of its reads.
    predicated: Vec<usize>,
    /// This rank reports the component's host results.
    reports: bool,
    started: Instant,
}

/// A batch whose last job is on its lane: what it reserved, and what the
/// job's completion is applied to.
pub(crate) struct Committing {
    states: Vec<CallState>,
    buffer: OutputLease,
    /// The completion flags the batch publishes.
    writes: Vec<ProductBuffer>,
    /// The forward the job runs; `None` when every call was skipped.
    recorded: Option<Recorded>,
    reports: bool,
    started: Instant,
}

/// A batch between two of its jobs.
pub(crate) enum Pending {
    /// The job on the lane reads the batch's host-gated predicates.
    Predicates(Opening),
    /// The predicates are known: the batch reserves, steps and commits
    /// next.
    Opened(Opening),
    /// The job on the lane is the batch's last.
    Committing(Box<Committing>),
    /// The job on the lane ends what the failed batch recorded: its
    /// fences, and the seal of its lease.
    Discarding(Failure),
    /// The batch failed and holds nothing: its calls are reported next.
    Failed(Failure),
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

impl Opening {
    /// Bind the batch's outputs to their requests, then record in `job`
    /// the batch's order after the text runtime's batches and the read of
    /// every readout's host-gated predicate: the job's completion carries
    /// one byte each.
    fn open(context: &mut Context<'_>, job: &Job<'_>) -> Result<Self> {
        let started = Instant::now();
        let Context {
            info,
            rank,
            runner,
            requests,
            tensors,
            plan,
            text,
            ..
        } = context;
        let calls = &plan.calls;
        if calls
            .iter()
            .any(|call| call.code != KIND || call.component != runner.component)
        {
            return Err(Error::Invalid(format!(
                "a diffusion LM batch carries token-denoising calls of component {:?}",
                runner.component
            )));
        }
        let device = runner.buffers.device();
        let reports = reports_output(info, *rank, &runner.component)?;

        // Output rows, bound to their requests' live epochs and slots.
        let rows = calls
            .iter()
            .map(|call| {
                let request = requests.get(call.request_key.request_id.0)?;
                Ok((call.request_key, call.call_id, request.slot()))
            })
            .collect::<Result<Vec<_>>>()?;
        let bound = requests.bind_calls(&rows)?;
        let mut states = Vec::with_capacity(calls.len());
        for (row, (call, request)) in calls.iter().zip(&bound).enumerate() {
            states.push(CallState {
                output: PendingOutput::new(call, request.progress()?, row)?,
                slot: request.slot() as u32,
                active: true,
                completion: None,
            });
        }

        // Readouts gate on host predicates. A step's continuation stays on
        // the device: reading it here would serialize steps that can run.
        // Each predicate is read after its producer's fence, and its lease
        // ends behind the read.
        let issue = job.on(runner.stream.as_ref());
        if let Some(text) = text {
            text.wait(issue)?;
        }
        let mut predicated = Vec::new();
        for (index, call) in calls.iter().enumerate() {
            let Some(predicate) = call
                .predicate
                .as_ref()
                .filter(|predicate| predicate.dtype == DType::U8 && call.canvas.is_none())
            else {
                continue;
            };
            let read = tensors.consume(predicate, call.call_id, device.map(Device::Cuda), issue)?;
            issue.read(read.value().address(), 1);
            tensors.complete_reads(&[read], issue, &[])?;
            predicated.push(index);
        }
        Ok(Self {
            states,
            predicated,
            reports,
            started,
        })
    }

    /// Apply the opening job's completion: a readout whose predicate read
    /// false is skipped.
    fn predicates(&mut self, completion: &mut Completion) -> Result<()> {
        for index in std::mem::take(&mut self.predicated) {
            let active = match completion.read()?[..] {
                [0] => false,
                [1] => true,
                _ => {
                    return Err(Error::Invalid(
                        "call predicate is not a canonical boolean".into(),
                    ));
                }
            };
            if !active {
                let state = &mut self.states[index];
                state.active = false;
                state.output.output.status = CallStatus::Predicated;
            }
        }
        Ok(())
    }

    /// Lease the opened batch's output storage and record its reservation,
    /// forward, outputs and seal in `job`: the batch's last job, or the job
    /// that ends what a failed batch recorded.
    fn run<'a>(self, context: Context<'_>, job: Job<'a>) -> (Option<Job<'a>>, Pending) {
        let Self {
            mut states,
            reports,
            started,
            ..
        } = self;
        let Context {
            runner,
            requests,
            kv,
            tensors,
            events,
            plan,
            experts,
            ..
        } = context;
        let reserve = Failure::new("batch registration");
        let ended = |job: Job<'a>, failure: Failure| {
            if job.is_empty() {
                (None, Pending::Failed(failure))
            } else {
                // What the batch recorded before it failed is still due:
                // the fences of its predicate reads among it.
                (Some(job), Pending::Discarding(failure))
            }
        };
        let calls = &plan.calls;
        let device = runner.buffers.device();
        let stream = runner.stream.clone();

        // The output lease: a step's outcome and block, or a readout's
        // candidate values, per call.
        let length = runner.facts.sequence_length;
        let words = calls.iter().fold(0usize, |total, call| {
            total.saturating_add(match &call.readout {
                Some(readout) => readout.candidate_count().max(1),
                None => 1 + length,
            })
        });
        let devices: Vec<i32> = device.into_iter().collect();
        let buffer =
            match runner
                .outputs
                .lease(calls.len(), words.max(calls.len()), &devices, false)
            {
                Ok(buffer) => buffer,
                Err(error) => return ended(job, reserve(error)),
            };
        if stream.is_some()
            && let Err(error) = lock(&buffer).begin_device(device, job.on(stream.as_ref()), events)
        {
            lock(&buffer).abandon();
            return ended(job, reserve(error));
        }

        let mut reserved = Reserved {
            writes: Vec::new(),
            retirement: Arc::new(Retirement::default()),
        };
        let batch = Batch {
            runner,
            experts,
            requests,
            kv,
            tensors,
            events,
            plan,
            stream: stream.as_ref(),
            job: &job,
        };
        match batch.run(&mut states, &buffer, &mut reserved) {
            Ok(recorded) => (
                Some(job),
                Pending::Committing(Box::new(Committing {
                    states,
                    buffer,
                    writes: reserved.writes,
                    recorded,
                    reports,
                    started,
                })),
            ),
            Err(failure) => {
                // Retire every reservation; the lease is sealed in the job,
                // and its storage returns once the fences recorded there
                // complete.
                let abandoned = tensors.abandon_writes(&reserved.writes);
                let sealed = if lock(&buffer).sealed() {
                    Ok(())
                } else {
                    seal(
                        events,
                        &buffer,
                        job.on(stream.as_ref()),
                        &reserved.retirement,
                    )
                };
                lock(&buffer).abandon();
                match abandoned.and(sealed) {
                    Ok(()) => ended(job, failure),
                    Err(cleanup) => ended(job, Failure::fatal("batch cleanup")(cleanup)),
                }
            }
        }
    }
}

impl Committing {
    /// Apply the completion of the batch's last job: store what its
    /// forward captured, then commit products and pending calls together.
    fn commit(
        mut self,
        runner: &mut DiffusionLMRunner,
        requests: &mut RequestPool,
        tensors: &TensorStore,
        plan: &BatchPlan,
        completed: Result<Completion>,
    ) -> std::result::Result<Launched, Failure> {
        let executed = completed.and_then(|mut completion| {
            self.recorded
                .take()
                .map(|recorded| runner.complete(recorded, &mut completion))
                .transpose()
        });
        let executed = match executed {
            Ok(executed) => executed,
            Err(error) => {
                // The job recorded the lease's seal, and the lane makes a
                // failed job's remaining event records, so only the
                // reservations of the host are left to retire.
                let abandoned = tensors.abandon_writes(&self.writes);
                lock(&self.buffer).abandon();
                return Err(match abandoned {
                    Ok(()) => Failure::new("batch execution")(error),
                    Err(cleanup) => Failure::fatal("batch cleanup")(cleanup),
                });
            }
        };

        // Every owner accepted its update when the job was recorded. Any
        // failure from here is fatal.
        let fatal = Failure::fatal("batch commit");
        for state in &mut self.states {
            state.output.reports_output = self.reports;
        }
        let pending: Vec<_> = plan
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id, call.advances_state()))
            .collect();
        tensors.commit_writes(&self.writes).map_err(&fatal)?;
        requests.add_pending(&pending).map_err(&fatal)?;

        let mut stats = ForwardStats::default();
        if let Some(executed) = executed {
            record(&mut stats, executed);
        }
        stats.component_us.insert(
            "canvas_execute".into(),
            self.started.elapsed().as_micros() as u64,
        );
        Ok(Launched {
            outputs: self.states.into_iter().map(|state| state.output).collect(),
            buffer: self.buffer,
            started: self.started,
            stats,
            timer: "canvas_materialize",
            host: None,
        })
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

/// One batch on the runner, recording into its job.
struct Batch<'a, 'j> {
    runner: &'a DiffusionLMRunner,
    experts: Option<&'a Arc<ExpertUnion>>,
    requests: &'a mut RequestPool,
    kv: &'a mut KVCache,
    tensors: &'a TensorStore,
    events: &'a Arc<EventPool>,
    plan: &'a BatchPlan,
    stream: Option<&'a Stream>,
    job: &'a Job<'j>,
}

impl Batch<'_, '_> {
    /// The expert steps of this runner's forwards.
    fn expert(&self) -> Option<Expert<'_>> {
        self.experts.map(|union| Expert {
            union,
            kind: expert_kind(self.runner.index),
            forced: None,
        })
    }

    /// Record the batch up to the seal of its lease; the forward it
    /// recorded, when a call is active.
    fn run(
        mut self,
        states: &mut [CallState],
        buffer: &OutputLease,
        reserved: &mut Reserved,
    ) -> std::result::Result<Option<Recorded>, Failure> {
        let reserve = Failure::new("batch registration");
        let plan = self.plan;
        let issue = self.job.on(self.stream);
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
        self.kv.install(&assignments, issue).map_err(&reserve)?;

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
        let target = self
            .runner
            .buffers
            .device()
            .map_or(Device::Cpu, Device::Cuda);
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
                    .write_scalar(write, crate::storage::Scalar::Int(0), issue)
                    .map_err(&reserve)?;
            }
        }

        let active: Vec<usize> = (0..calls.len()).filter(|&row| states[row].active).collect();
        let execute = Failure::new("batch execution");
        let mut recorded = None;
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
            recorded = Some(if steps {
                self.steps(states, &active, &rows, buffer)?
            } else {
                self.readouts(states, &active, &rows, buffer)?
            });
        }
        self.seal(states, buffer, reserved)?;
        Ok(recorded)
    }

    /// Record the active readouts as one pass and the copies of their
    /// candidates.
    fn readouts(
        &mut self,
        states: &mut [CallState],
        active: &[usize],
        rows: &[Vec<usize>],
        buffer: &OutputLease,
    ) -> std::result::Result<Recorded, Failure> {
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
        let (issue, events) = (self.job.on(self.stream), self.events);
        let expert = self.expert();
        let executed = {
            let mut output = lock(buffer);
            self.runner
                .readout(&pass, expert.as_ref(), self.job, |range, values| {
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
                        let (offset, count) = output.capture(&source, issue, events)?;
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

    /// Record one step of the active calls' blocks and the copies of
    /// their results.
    fn steps(
        &mut self,
        states: &mut [CallState],
        active: &[usize],
        rows: &[Vec<usize>],
        buffer: &OutputLease,
    ) -> std::result::Result<Recorded, Failure> {
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
        let issue = self.job.on(self.stream);
        let expert = self.expert();
        let executed = self
            .runner
            .denoise(&steps, &pages, expert.as_ref(), self.job)
            .map_err(&execute)?;

        // One copy of every row's `[outcome | block]` result.
        let results = self
            .runner
            .buffers
            .column("denoise.results")
            .and_then(|results| results.prefix(steps.len() as i64))
            .map_err(&execute)?;
        let (offset, _) = lock(buffer)
            .capture(&results, issue, self.events)
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
                .write_scalars(&writes, &values, issue)
                .map_err(&execute)?;
            start = end;
        }
        Ok(executed)
    }

    /// Validate every owner's update and record the seal of the lease.
    /// The updates are committed once the job completed
    /// ([`Committing::commit`]).
    fn seal(
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
        seal(
            self.events,
            buffer,
            self.job.on(self.stream),
            &reserved.retirement,
        )
        .map_err(&fail)?;
        let pending: Vec<_> = self
            .plan
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id, call.advances_state()))
            .collect();
        self.requests.validate_pending(&pending).map_err(&fail)
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

/// The owners a diffusion LM batch of `plan` runs on, or the refusal of
/// a rank that has no runner for its component.
fn context<'a>(
    parts: crate::executor::backend::Parts<'a>,
    plan: &'a BatchPlan,
    experts: Option<&'a Arc<ExpertUnion>>,
    kind: CallKind,
) -> std::result::Result<Context<'a>, WorkerError> {
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
    let Some(RankStorage {
        kv: Some(kv),
        tensors,
        events,
        ..
    }) = parts.storage
    else {
        return Err(crate::executor::forward::unsupported(kind, "diffusion LM"));
    };
    Ok(Context {
        info: parts.info,
        rank: parts.rank,
        runner,
        requests: parts.requests,
        kv,
        tensors,
        events,
        plan,
        text: parts.text.map(|text| text.order()),
        experts,
    })
}

/// The next job of a diffusion LM batch on its lane, or `None` when the
/// batch advanced without one: it failed and its calls are reported.
pub(crate) fn forward<'a>(
    backend: &'a mut WorkerBackend,
    batch: &'a mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<Option<Job<'a>>, WorkerError> {
    let experts = backend.experts().cloned();
    let state = match batch.denoising_parts().1.take() {
        Some(Pending::Failed(failure)) => return fail(backend, batch, failure).map(|()| None),
        state => state,
    };
    let (plan, pending) = batch.denoising_parts();
    let mut context = context(backend.parts(), plan, experts.as_ref(), kind)?;

    let job = Job::new();
    let (job, next) = match state {
        None => match Opening::open(&mut context, &job) {
            Ok(opening) if !opening.predicated.is_empty() => {
                (Some(job), Pending::Predicates(opening))
            }
            Ok(opening) => opening.run(context, job),
            Err(error) => {
                let failure = Failure::new("batch registration")(error);
                if job.is_empty() {
                    (None, Pending::Failed(failure))
                } else {
                    // The fences the opening recorded are still due.
                    (Some(job), Pending::Discarding(failure))
                }
            }
        },
        Some(Pending::Opened(opening)) => opening.run(context, job),
        Some(
            state @ (Pending::Predicates(_)
            | Pending::Committing(_)
            | Pending::Discarding(_)
            | Pending::Failed(_)),
        ) => {
            // A job of the batch is on the lane: its completion comes
            // before the next job.
            *pending = Some(state);
            return Err(WorkerError::classify(
                Error::Invariant(
                    "a diffusion LM batch's job was prepared before its last completed".into(),
                ),
                Some("diffusion LM batch"),
            ));
        }
    };
    *pending = Some(next);
    Ok(job)
}

/// Apply the completion of the diffusion LM batch's job on the lane.
pub(crate) fn complete(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    completion: Completion,
) -> std::result::Result<(), WorkerError> {
    let (_, pending) = batch.denoising_parts();
    let next = match (pending.take(), completion.check()) {
        // A predicate that could not be read fails the batch, which has
        // reserved nothing yet.
        (Some(Pending::Predicates(_)), Err(failure)) => {
            Pending::Failed(Failure::new("batch registration")(failure.error))
        }
        (Some(Pending::Predicates(mut opening)), Ok(mut completion)) => {
            match opening.predicates(&mut completion) {
                Ok(()) => Pending::Opened(opening),
                Err(error) => Pending::Failed(Failure::new("batch registration")(error)),
            }
        }
        (Some(Pending::Committing(committing)), completed) => {
            let (plan, _) = batch.denoising_parts();
            let Context {
                runner,
                requests,
                tensors,
                ..
            } = context(backend.parts(), plan, None, KIND)?;
            let completed = completed.map_err(|failure| failure.error);
            match committing.commit(runner, requests, tensors, plan, completed) {
                Ok(launched) => {
                    batch.launch(launched);
                    batch.set_stage(Stage::Done);
                    return Ok(());
                }
                Err(failure) => Pending::Failed(failure),
            }
        }
        // What the failed batch recorded could not be ended: storage it
        // fenced stays reserved, and the rank cannot reuse it safely.
        (Some(Pending::Discarding(_)), Err(failure)) => {
            Pending::Failed(Failure::fatal("batch cleanup")(failure.error))
        }
        (Some(Pending::Discarding(failure)), Ok(_)) => Pending::Failed(failure),
        (state, _) => {
            *batch.denoising_parts().1 = state;
            return Err(WorkerError::classify(
                Error::Invariant("a diffusion LM batch has no job on its lane".into()),
                Some("diffusion LM batch"),
            ));
        }
    };
    *batch.denoising_parts().1 = Some(next);
    Ok(())
}

/// Report a failed batch: a failure of its commit stops the rank, any
/// other completes the batch with its calls failed.
fn fail(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    failure: Failure,
) -> std::result::Result<(), WorkerError> {
    let plan = Arc::clone(batch.shared_plan());
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
        batch.predecessors(),
        error_code(error.code),
    )?;
    batch.complete(BatchResult {
        output,
        media: Vec::new(),
    });
    batch.set_stage(Stage::Done);
    Ok(())
}

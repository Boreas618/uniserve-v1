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
//! 4. **Completion** ([`crate::executor::batch::materialize`]).
//!    Once the lease's device fences complete, decode each call's result
//!    and accept its progress. These calls write no prefix cache: each
//!    keeps its visible length.

use std::collections::HashSet;
use std::sync::Arc;

use uniserve_cuda::Stream;
use uniserve_ipc::{Batch as BatchPlan, BufferId, CallKind, DType, ForwardStats, RequestKey};

use super::calls::{self, DenoiseRow, ReadoutPass, Readouts};
use super::{DiffusionLMRunner, Executed, Expert, KIND, Recorded};
use crate::config::placement::GroupPlacement;
use crate::cuda_graph::GraphMode;
use crate::error::WorkerError;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::batch::{
    Batch, Failure, Launched, Opening, Stage, StreamOrder, fail, unsupported,
};
use crate::executor::lifecycle::{LaneJobs, RequestStorage};
use crate::lane::{Completion, Job};
use crate::parallel::ExpertUnion;
use crate::request::RequestPool;
use crate::storage::{PrefixCache, ProductBuffer, ProductStore};
use crate::stream::EventPool;
use crate::worker::RankStorage;
use crate::{Error, Result, lock};

/// What a batch borrows from the rank.
struct Context<'a> {
    placement: &'a GroupPlacement,
    rank: usize,
    runner: &'a mut DiffusionLMRunner,
    requests: &'a mut RequestPool,
    prefix_cache: &'a mut PrefixCache,
    tensors: &'a ProductStore,
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

/// A batch whose last job is on its lane: what it reserved, and what the
/// job's completion is applied to.
pub(crate) struct Committing {
    batch: Batch,
    /// The forward the job runs; `None` when every call was skipped.
    recorded: Option<Recorded>,
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

/// Bind the batch's outputs to their requests, then record in `job` the
/// batch's order after the text runtime's batches and the read of every
/// readout's host-gated predicate: the job's completion carries one byte
/// each. Readouts gate on host predicates; a step's continuation stays on
/// the device: reading it here would serialize steps that can run.
fn open(context: &mut Context<'_>, job: &Job) -> Result<Opening> {
    let Context {
        placement,
        rank,
        runner,
        requests,
        tensors,
        plan,
        text,
        ..
    } = context;
    if plan
        .calls
        .iter()
        .any(|call| call.kind != KIND || call.component != runner.component)
    {
        return Err(Error::Invalid(format!(
            "a diffusion LM batch carries token-denoising calls of component {:?}",
            runner.component
        )));
    }
    let issue = job.on(runner.stream.as_ref());
    if let Some(text) = text {
        text.wait(issue)?;
    }
    Opening::open(
        requests,
        tensors,
        placement,
        *rank,
        plan,
        runner.buffers.device(),
        issue,
        |call| call.canvas.is_none(),
    )
}

/// Lease the opened batch's output storage and record its reservation,
/// forward, outputs and seal in `job`: the batch's last job, or the job that
/// ends what a failed batch recorded.
fn run(opening: Opening, context: Context<'_>, job: Job) -> (Option<Job>, Pending) {
    let Context {
        runner,
        requests,
        prefix_cache,
        tensors,
        events,
        plan,
        experts,
        ..
    } = context;
    let reserve = Failure::new("batch registration");
    let ended = |job: Job, failure: Failure| {
        if job.is_empty() {
            (None, Pending::Failed(failure))
        } else {
            // What the batch recorded before it failed is still due: the
            // fences of its predicate reads among it.
            (Some(job), Pending::Discarding(failure))
        }
    };
    let calls = &plan.calls;

    // The output lease: a step's outcome and block, or a readout's
    // candidate values, per call.
    let length = runner.facts.sequence_length;
    let words = calls.iter().fold(0usize, |total, call| {
        total.saturating_add(match &call.readout {
            Some(readout) => readout.candidate_count().max(1),
            None => 1 + length,
        })
    });
    let mut batch = match Batch::lease(
        opening,
        &mut runner.outputs,
        words.max(calls.len()),
        runner.stream.clone(),
        runner.buffers.device(),
        events,
        &job,
    ) {
        Ok(batch) => batch,
        Err(error) => return ended(job, reserve(error)),
    };

    let runner: &DiffusionLMRunner = runner;
    let owners = Owners {
        runner,
        stream: runner.stream.as_ref(),
        experts,
        requests,
        prefix_cache,
        tensors,
        events,
        plan,
        job: &job,
    };
    match owners.run(&mut batch) {
        Ok(recorded) => (
            Some(job),
            Pending::Committing(Box::new(Committing { batch, recorded })),
        ),
        Err(failure) => {
            // Retire every reservation; the lease is sealed in the job, and
            // its storage returns once the fences recorded there complete.
            match batch.discard(tensors, events, &job) {
                Ok(()) => ended(job, failure),
                Err(cleanup) => ended(job, Failure::fatal("batch cleanup")(cleanup)),
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
        tensors: &ProductStore,
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
                let writes = std::mem::take(&mut self.batch.writes);
                let abandoned = tensors.abandon_writes(&writes);
                lock(&self.batch.buffer).abandon();
                return Err(match abandoned {
                    Ok(()) => Failure::new("batch execution")(error),
                    Err(cleanup) => Failure::fatal("batch cleanup")(cleanup),
                });
            }
        };

        // Every owner accepted its update when the job was recorded. Any
        // failure from here is fatal.
        self.batch.commit(plan, tensors, requests)?;
        let mut stats = ForwardStats::default();
        if let Some(executed) = executed {
            record(&mut stats, executed);
        }
        stats.component_us.insert(
            "canvas_execute".into(),
            self.batch.started.elapsed().as_micros() as u64,
        );
        Ok(self.batch.launch(stats, "canvas_materialize", None))
    }
}

impl RequestStorage for DiffusionLMRunner {
    fn admit_slots(&mut self, _slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
        // A request's first denoising step starts its block in the banks
        // before any step reads them; admission resets nothing.
        Ok(())
    }

    fn release_buffers(&mut self, _buffers: &HashSet<BufferId>, _jobs: &LaneJobs) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
        _jobs: &LaneJobs,
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

    fn retire_slots(&mut self, _slots: &[usize], _jobs: &LaneJobs) -> Result<()> {
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        self.outputs.reap();
        Ok(())
    }
}

/// The owners a batch borrows while it records into its job.
struct Owners<'a> {
    runner: &'a DiffusionLMRunner,
    experts: Option<&'a Arc<ExpertUnion>>,
    requests: &'a mut RequestPool,
    prefix_cache: &'a mut PrefixCache,
    tensors: &'a ProductStore,
    events: &'a Arc<EventPool>,
    plan: &'a BatchPlan,
    stream: Option<&'a Stream>,
    job: &'a Job,
}

impl Owners<'_> {
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
    fn run(mut self, batch: &mut Batch) -> std::result::Result<Option<Recorded>, Failure> {
        let reserve = Failure::new("batch registration");
        let plan = self.plan;
        let issue = self.job.on(self.stream);
        let calls = &plan.calls;
        let slots: HashSet<u32> = batch
            .calls
            .iter()
            .filter(|state| state.active)
            .map(|state| state.slot as u32)
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
        self.prefix_cache
            .install(&assignments, issue)
            .map_err(&reserve)?;

        // Every active call retains the prefix it reads until the batch's
        // device work retires.
        for (call, state) in calls.iter().zip(&batch.calls) {
            if !state.active {
                continue;
            }
            let slot = state.slot as u32;
            let visible = state.output.progress.cache_visible_len;
            self.prefix_cache
                .tables()
                .coordinates(slot, visible)
                .map_err(&reserve)?;
            self.prefix_cache
                .retain(call.request_key, slot, visible, &batch.retirement)
                .map_err(&reserve)?;
        }

        // Completion flags of denoising steps; these calls publish nothing
        // else. Skipped calls still forward a false decision to their
        // consumers.
        for call in calls {
            if !call.outputs.is_empty()
                || call.token_output.is_some()
                || call.transition_output.is_some()
                || call.image_output.is_some()
                || call.encoder_output.is_some()
                || call.cache_output.is_some()
            {
                return Err(reserve(Error::Unsupported(
                    "token-denoising calls publish only a completion flag".into(),
                )));
            }
            if call
                .completion_output
                .as_ref()
                .is_some_and(|flag| flag.dtype != DType::U8)
            {
                return Err(reserve(Error::Invalid(
                    "a denoising step's completion flag is a uint8 scalar".into(),
                )));
            }
        }
        batch
            .bind_flags(plan, self.tensors, self.job)
            .map_err(&reserve)?;

        let active = batch.active();
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
                self.steps(batch, &active, &rows)?
            } else {
                self.readouts(batch, &active, &rows)?
            });
        }
        self.seal(batch)?;
        Ok(recorded)
    }

    /// Record the active readouts as one pass and the copies of their
    /// candidates.
    fn readouts(
        &mut self,
        batch: &mut Batch,
        active: &[usize],
        rows: &[Vec<usize>],
    ) -> std::result::Result<Recorded, Failure> {
        let execute = Failure::new("batch execution");
        let calls = &self.plan.calls;
        let mut readouts = Readouts::new();
        let mut ranges = Vec::with_capacity(active.len());
        for &row in active {
            let state = &batch.calls[row];
            let progress = state.output.progress;
            ranges.push(
                readouts
                    .add(
                        &self.plan.forward,
                        &calls[row],
                        &rows[row],
                        state.slot as u32,
                        progress.cache_visible_len,
                        progress.logical_position,
                    )
                    .map_err(&execute)?,
            );
        }
        let pages = self
            .prefix_cache
            .prepare_attention(&readouts.attention)
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
            let mut output = lock(&batch.buffer);
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
            let state = &mut batch.calls[row];
            let (offset, count) = span.unwrap_or((0, 0));
            if count != candidates.len() {
                return Err(execute(Error::Invariant(
                    "a readout lost candidate values".into(),
                )));
            }
            state.output.candidate_range = Some((offset, count));
            let visible = state.output.progress.cache_visible_len;
            state.output.set_cache_length(visible);
        }
        Ok(executed)
    }

    /// Record one step of the active calls' blocks and the copies of
    /// their results.
    fn steps(
        &mut self,
        batch: &mut Batch,
        active: &[usize],
        rows: &[Vec<usize>],
    ) -> std::result::Result<Recorded, Failure> {
        let execute = Failure::new("batch execution");
        let calls = &self.plan.calls;
        let served = self.runner.sampling;
        let length = self.runner.facts.sequence_length;
        let mut steps: Vec<DenoiseRow> = Vec::with_capacity(active.len());
        let mut attention = Vec::with_capacity(active.len());
        for &row in active {
            let call = &calls[row];
            let state = &batch.calls[row];
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
                progress.cache_visible_len,
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
            .prefix_cache
            .prepare_attention(&attention)
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
        let (offset, _) = lock(&batch.buffer)
            .capture(&results, issue, self.events)
            .map_err(&execute)?;
        let width = 1 + length;
        for (position, &row) in active.iter().enumerate() {
            let state = &mut batch.calls[row];
            state.output.canvas_range = Some((offset + position * width, width));
            let visible = state.output.progress.cache_visible_len;
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
            if batch.calls[active[start]].completion_write.is_none() {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < active.len() && batch.calls[active[end]].completion_write.is_some() {
                end += 1;
            }
            let writes: Vec<ProductBuffer> = active[start..end]
                .iter()
                .filter_map(|&row| batch.calls[row].completion_write.clone())
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
    fn seal(&mut self, batch: &mut Batch) -> std::result::Result<(), Failure> {
        let fail = Failure::new("batch commit");
        batch.validate(self.plan, self.tensors).map_err(&fail)?;
        batch.seal(self.events, self.job).map_err(&fail)?;
        self.requests
            .validate_pending(&Batch::<()>::pending(self.plan))
            .map_err(&fail)
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
        prefix_cache: Some(prefix_cache),
        tensors,
        events,
        ..
    }) = parts.storage
    else {
        return Err(unsupported(kind, "diffusion LM"));
    };
    Ok(Context {
        placement: parts.placement,
        rank: parts.rank,
        runner,
        requests: parts.requests,
        prefix_cache,
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
) -> std::result::Result<Option<Job>, WorkerError> {
    let experts = backend.experts().cloned();
    let state = match batch.denoising_parts().1.take() {
        Some(Pending::Failed(failure)) => return fail(backend, batch, failure).map(|()| None),
        state => state,
    };
    let (plan, pending) = batch.denoising_parts();
    let mut context = context(backend.parts(), plan, experts.as_ref(), kind)?;

    let job = Job::new();
    let (job, next) = match state {
        None => match open(&mut context, &job) {
            Ok(opening) if opening.reads_predicates() => (Some(job), Pending::Predicates(opening)),
            Ok(opening) => run(opening, context, job),
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
        Some(Pending::Opened(opening)) => run(opening, context, job),
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

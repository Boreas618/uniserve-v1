//! The text runtime: prefill, decode and speculative-verify batches of causal
//! language models.
//!
//! A token batch runs in four stages on the executor thread:
//!
//! 1. **Reservation** ([`batch`]): bind every call's pending output and the
//!    batch's output lease; install its block tables, reset newly assigned
//!    units and retain the KV extents it reads and writes until its device
//!    work retires; bind its scalar products (token relays, transition and
//!    completion flags); lease its predicates. A call whose host predicate
//!    is false is skipped (`Predicated`) and only publishes false flags.
//! 2. **Step** ([`step`]): one input row per active call, the sampling tasks
//!    and prompt scoring, and one invocation on the call kind's runner.
//! 3. **Capture and commit** ([`commit`]): advance each call's progress, copy
//!    completion and logprob words to the output lease, relay sampled tokens
//!    and transitions into their products, then commit products, slot state
//!    and pending calls together. A failure before the commit point retires
//!    everything the batch reserved and reports every call as failed; a
//!    failure after it is fatal.
//! 4. **Materialization** ([`poll`]): once the lease's device fences
//!    complete, decode each call's result and accept its progress.

mod batch;
mod calls;
mod commit;
pub(crate) mod start;
mod step;
pub(crate) mod warmup;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::{Event, EventFlags, Stream};
use uniserve_ipc::{
    Batch as BatchPlan, BatchOutput, BufferId, CallKind, CallStatus, ErrorCode, ForwardMode,
    ForwardStats, RequestKey, Wake,
};

use super::super::backend::{Parts, WorkerBackend, WorkerBatch};
use super::Running;
use crate::Result;
use crate::error::WorkerError;
use crate::executor::lifecycle::RequestStorage;
use crate::executor::{BatchResult, PendingOutput, request_output};
use crate::request::decode_state::DecodeState;
use crate::runner::graphs::GraphMode;
use crate::runner::text::TextRunner;
use crate::storage::{OutputLease, OutputPool, UnitColumn};

use batch::{TokenBatch, lock};
use commit::Failure;
use step::Stepped;

/// Fields of a call's completion section: valid, active, token, accepted.
const COMPLETION_FIELDS: usize = 4;

/// The text runtime of a rank: one runner per lane binding, the slot
/// continuation state they share, and their output storage.
pub(crate) struct TextRuntime {
    runners: Vec<TextRunner>,
    /// The unit reset column of each runner's stream.
    columns: Vec<UnitColumn>,
    routes: HashMap<(String, CallKind), usize>,
    state: DecodeState,
    outputs: OutputPool,
    /// Global vocabulary of sampling.
    vocab: usize,
    /// Orders slot state across runner streams: recorded after each text
    /// batch's last write, waited on by the next batch on another stream.
    order: Option<(usize, Event)>,
}

impl TextRuntime {
    fn route(&self, component: &str, kind: CallKind) -> Option<usize> {
        self.routes.get(&(component.to_owned(), kind)).copied()
    }

    /// Seal every runner's graphs.
    pub(crate) fn seal(&mut self) {
        for runner in &mut self.runners {
            runner.seal();
        }
    }

    /// Order `stream` after the previous text batch on another stream.
    fn order(&self, stream: Option<&Stream>) -> Result<()> {
        wait_order(&self.order, stream)
    }

    /// Record the end of this stream's text work for other streams.
    fn ordered(&mut self, stream: Option<&Stream>) -> Result<()> {
        record_order(&mut self.order, stream)
    }

    pub(crate) fn set_wake(&self, _wake: Option<Wake>) {}

    /// Release graphs, steps and output storage after the lanes drained.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        for runner in &mut self.runners {
            result = result.and(runner.close(aborted));
        }
        self.outputs.begin_close();
        if !aborted {
            self.outputs.finish_close();
        }
        result
    }
}

fn wait_order(order: &Option<(usize, Event)>, stream: Option<&Stream>) -> Result<()> {
    let Some(stream) = stream else {
        return Ok(());
    };
    if let Some((previous, event)) = order
        && *previous != stream.raw()
    {
        stream.wait_event(event)?;
    }
    Ok(())
}

fn record_order(order: &mut Option<(usize, Event)>, stream: Option<&Stream>) -> Result<()> {
    let Some(stream) = stream else {
        return Ok(());
    };
    let event = match order.take() {
        Some((_, event)) => event,
        None => Event::new(EventFlags::default()),
    };
    event.record(stream)?;
    *order = Some((stream.raw(), event));
    Ok(())
}

impl RequestStorage for TextRuntime {
    fn admit_slots(&mut self, slots: &[usize]) -> Result<()> {
        // Reset on the first runner's stream; other streams order after it
        // before their next batch.
        let stream = self.runners.first().and_then(TextRunner::stream);
        wait_order(&self.order, stream)?;
        self.state.reset(slots, stream)?;
        record_order(&mut self.order, stream)
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

/// A token batch launched on the device, awaiting its output fences.
pub(crate) struct Launched {
    outputs: Vec<PendingOutput>,
    buffer: OutputLease,
    started: Instant,
    stats: ForwardStats,
}

/// Execute a batch of `kind` calls.
pub(crate) fn execute(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<(), WorkerError> {
    let plan = Arc::clone(batch.shared_plan());
    let predecessors = batch.predecessors().to_vec();
    let failure = match run(backend.parts(), &plan) {
        Ok(Some(launched)) => {
            batch.launch(Running::Token(launched));
            return Ok(());
        }
        Ok(None) => return Err(super::unsupported(kind, "text")),
        Err(failure) => failure,
    };
    let mut error = WorkerError::classify(failure.error, Some("token batch"));
    if failure.committed {
        // Products and slot state may already be visible; the rank cannot
        // roll them back.
        error.fatal = true;
    }
    if error.fatal {
        return Err(error);
    }
    tracing::warn!(
        target: "uniserve_worker.execution",
        "text batch {} failed: {error}{}",
        plan.batch_id,
        error
            .traceback()
            .map(|trace| format!("\n{trace}"))
            .unwrap_or_default()
    );
    let code = match error.code {
        crate::ErrorCode::ResourceError => ErrorCode::ResourceExhausted,
        crate::ErrorCode::ComputeError => ErrorCode::ComputeError,
        crate::ErrorCode::InvariantViolation | crate::ErrorCode::FatalWorkerFailure => {
            ErrorCode::Internal
        }
        _ => ErrorCode::InvalidCall,
    };
    let output = failed_output(backend, &plan, &predecessors, code)?;
    batch.complete(BatchResult {
        output,
        media: Vec::new(),
    });
    Ok(())
}

/// Reserve, step and commit one batch. `None` on a rank without a text
/// runtime.
fn run(parts: Parts<'_>, plan: &BatchPlan) -> std::result::Result<Option<Launched>, Failure> {
    let started = Instant::now();
    let (Some(runtime), Some(storage)) = (parts.text, parts.storage) else {
        return Ok(None);
    };
    let mut batch = TokenBatch::open(
        runtime,
        storage,
        parts.requests,
        parts.info,
        parts.rank,
        plan,
        started,
    )?;
    let result = batch.reserve().map_err(Failure::from).and_then(|()| {
        if batch.calls.iter().any(|state| state.active) {
            let stepped = batch.step()?;
            batch.finish(&stepped)?;
            Ok(Some(stepped))
        } else {
            batch.finish_skipped()?;
            Ok(None)
        }
    });
    match result {
        Ok(stepped) => {
            let stats = statistics(&batch, stepped.as_ref());
            Ok(Some(Launched {
                outputs: batch.calls.into_iter().map(|state| state.output).collect(),
                buffer: batch.buffer,
                started,
                stats,
            }))
        }
        Err(failure) if failure.committed => Err(failure),
        Err(failure) => {
            if let Err(cleanup) = batch.discard() {
                // Storage the cleanup could not retire stays reserved; the
                // rank cannot reuse it safely.
                return Err(Failure {
                    error: cleanup,
                    committed: true,
                });
            }
            Err(failure)
        }
    }
}

impl TokenBatch<'_> {
    /// Commit a batch whose calls were all skipped: only their false flags
    /// were published.
    fn finish_skipped(&mut self) -> std::result::Result<(), Failure> {
        let empty = Stepped {
            executed: crate::runner::text::Executed {
                mode: GraphMode::Eager,
                live_tokens: 0,
                padded_tokens: 0,
            },
            sampling: None,
            tasks: 0,
            rows: 0,
            drafts: 0,
        };
        self.finish(&empty)
    }
}

fn statistics(batch: &TokenBatch<'_>, stepped: Option<&Stepped>) -> ForwardStats {
    let mut stats = ForwardStats::default();
    let elapsed = batch.started.elapsed().as_micros() as u64;
    stats.component_us.insert("text_execute".into(), elapsed);
    let Some(stepped) = stepped else {
        return stats;
    };
    let executed = stepped.executed;
    let mode = executed.mode.as_str();
    *stats.mode_counts.entry(mode.into()).or_default() += 1;
    *stats.mode_tokens.entry(mode.into()).or_default() += executed.live_tokens as u64;
    *stats.mode_us.entry(mode.into()).or_default() += elapsed;
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
        stats.cuda_graph_unpadded_tokens += executed.live_tokens as u64;
        stats.cuda_graph_padded_tokens += executed.padded_tokens as u64;
    }
    if batch.mode == ForwardMode::Verify {
        stats.spec_verify_rows += stepped.rows as u64;
        stats.spec_verify_draft_tokens += stepped.drafts;
    }
    stats
}

/// Every call fails with `code`, reporting its accepted progress.
fn failed_output(
    backend: &WorkerBackend,
    plan: &BatchPlan,
    predecessors: &[Option<uniserve_core::CallId>],
    code: ErrorCode,
) -> Result<BatchOutput> {
    let requests = backend.requests();
    let completions = plan
        .calls
        .iter()
        .zip(predecessors)
        .map(|(call, predecessor)| {
            let key = call.request_key;
            // A failed first call has no accepted parent; a stale epoch never
            // reports a replacement request's progress.
            let progress = requests
                .peek(key.request_id.0)
                .filter(|request| request.key() == key && predecessor.is_some())
                .map(|request| request.progress())
                .transpose()?
                .unwrap_or_default();
            let mut output = request_output(key, call.call_id, call.code, progress)?;
            output.status = CallStatus::Error;
            output.error_code = Some(code);
            Ok(output)
        })
        .collect::<Result<_>>()?;
    Ok(BatchOutput {
        batch_id: plan.batch_id,
        completions,
        products: Vec::new(),
        worker_exec_us: None,
        forward_stats: None,
    })
}

/// Materialize a launched batch once its output fences completed.
pub(crate) fn poll(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    launched: &mut Launched,
) -> std::result::Result<Option<BatchResult>, WorkerError> {
    {
        let mut buffer = lock(&launched.buffer);
        if buffer.ready_at()?.is_none() {
            return Ok(None);
        }
        buffer.readback()?;
    }
    let mut completions = Vec::with_capacity(launched.outputs.len());
    for output in &mut launched.outputs {
        let key = output.output.request_key;
        let parent = backend
            .requests()
            .peek(key.request_id.0)
            .filter(|request| request.key() == key)
            .map(|request| request.progress())
            .transpose()?
            .unwrap_or_default();
        let mut buffer = lock(&launched.buffer);
        if let Err(error) = output.resolve(parent, &mut buffer) {
            // A malformed capture fails its call only.
            tracing::error!(
                target: "uniserve_worker.execution",
                "text result materialization failed: {error}"
            );
            output.output.status = CallStatus::Error;
            output.output.error_code = Some(ErrorCode::ComputeError);
            output.progress = parent;
            buffer.discard(output.row);
            completions.push(output.output.clone());
            continue;
        }
        completions.push(output.result()?.clone());
    }
    for output in &launched.outputs {
        if output.resolved() {
            output.accept(backend.requests_mut())?;
        } else {
            output.cancel(backend.requests_mut())?;
        }
    }
    let mut stats = std::mem::take(&mut launched.stats);
    let elapsed = launched.started.elapsed().as_micros() as u64;
    stats
        .component_us
        .insert("text_materialize".into(), elapsed);
    Ok(Some(BatchResult {
        output: BatchOutput {
            batch_id: plan.batch_id,
            completions,
            products: Vec::new(),
            worker_exec_us: Some(elapsed),
            forward_stats: Some(stats),
        },
        media: Vec::new(),
    }))
}

/// Cancel a launched batch's unresolved calls; its storage retires through
/// the fences it recorded.
pub(crate) fn abandon(
    backend: &mut WorkerBackend,
    launched: Launched,
) -> std::result::Result<(), WorkerError> {
    for output in &launched.outputs {
        output.cancel(backend.requests_mut())?;
    }
    lock(&launched.buffer).abandon();
    Ok(())
}

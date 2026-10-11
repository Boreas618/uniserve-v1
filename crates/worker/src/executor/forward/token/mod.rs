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
//! 4. **Materialization** ([`super::launch::materialize`]): once the
//!    lease's device fences complete, decode each call's result and accept
//!    its progress.

mod batch;
mod calls;
mod commit;
mod context;
pub(crate) mod start;
mod step;
pub(crate) mod warmup;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferId, CallKind, ForwardMode, ForwardStats, RequestKey, Wake,
};

use super::super::backend::{Parts, WorkerBackend, WorkerBatch};
use super::launch::{Launched, StreamOrder, error_code, failed_output};
use crate::cuda_graph::GraphMode;
use crate::error::WorkerError;
use crate::executor::BatchResult;
use crate::executor::lifecycle::RequestStorage;
use crate::request::decode_state::DecodeState;
use crate::runner::TokenSelection;
use crate::runner::text::{
    Executed, RowSource, Staged, StepContext, StepRequest, TextRunner, TokenRow,
};
use crate::storage::{BlockTables, KVCacheManager, OutputPool, UnitColumn};
use crate::transfer::Retirement;
use crate::{Error, Result};

/// The KV storage a text step borrows.
pub(crate) struct StepParts<'a> {
    pub(crate) tables: &'a BlockTables,
    pub(crate) cache: &'a mut KVCacheManager<Arc<Retirement>>,
}

use batch::TokenBatch;
use commit::Failure;
use step::Stepped;

/// Fields of a call's completion section: valid, active, token, accepted.
const COMPLETION_FIELDS: usize = 4;

/// The text runtime of a rank: one runner per lane binding, or per expert
/// microbatch of it, the slot continuation state they share, and their
/// output storage.
pub(crate) struct TextRuntime {
    runners: Vec<TextRunner>,
    /// Each lane binding's runners, one per expert microbatch; the first
    /// runs on the lane's stream, the others on its forks.
    groups: Vec<Vec<usize>>,
    /// The NCCL communicators of the runners' bindings, destroyed after
    /// their graphs.
    collectives: Vec<crate::parallel::StreamCollectives>,
    /// The unit reset column of each runner's stream.
    columns: Vec<UnitColumn>,
    /// The group serving each component's call kind.
    routes: HashMap<(String, CallKind), usize>,
    state: DecodeState,
    outputs: OutputPool,
    /// Global vocabulary of sampling.
    vocab: usize,
    /// Orders slot state across runner streams: recorded after each text
    /// batch's last write, waited on by the next batch on another stream.
    order: StreamOrder,
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

    /// The order of this runtime's batches across streams; other runtimes
    /// reading what text batches write (their KV) wait on it.
    pub(crate) fn order(&self) -> &StreamOrder {
        &self.order
    }

    /// The group of `runner` and its microbatch index in it.
    pub(crate) fn locate(&self, runner: usize) -> Result<(usize, usize)> {
        self.groups
            .iter()
            .enumerate()
            .find_map(|(group, peers)| {
                peers
                    .iter()
                    .position(|&peer| peer == runner)
                    .map(|peer| (group, peer))
            })
            .ok_or_else(|| Error::Invariant("a text runner belongs to no group".into()))
    }

    /// The warmup and capture plan every source of an expert union must
    /// share: each group's microbatches and graph shapes.
    pub(crate) fn capture_plan(&self) -> serde_json::Value {
        let groups: Vec<serde_json::Value> = self
            .groups
            .iter()
            .map(|group| {
                let shapes = self.runners[group[0]].shapes();
                serde_json::json!({
                    "microbatches": group.len(),
                    "decode": shapes.decode(),
                    "prefill": shapes
                        .prefill()
                        .iter()
                        .map(|shape| (shape.row_bucket, shape.token_bucket, shape.outputs))
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        serde_json::Value::Array(groups)
    }

    /// Run one invocation per microbatch runner of `group` (`None` for an
    /// empty microbatch) as one numerical step.
    ///
    /// Without experts the group has one runner and its invocation runs on
    /// the caller's thread. With experts, the step is an expert step: every
    /// microbatch stream forks from the group's lane stream before inputs
    /// are staged; the ranks agree on the step's capacity (or, during
    /// warmup, it is `forced` or derived and announced); then the
    /// microbatches run in rotation, empty ones joining, and the lane stream
    /// waits on every microbatch stream. Microbatches replay their graphs
    /// only once the union joins by replay
    /// ([`super::experts::ExpertRuntime::replays`]).
    pub(crate) fn invoke(
        &mut self,
        experts: Option<&super::experts::ExpertRuntime>,
        group: usize,
        requests: &[Option<StepRequest<'_>>],
        context: &mut StepParts<'_>,
        forced: Option<usize>,
    ) -> Result<Vec<Option<Executed>>> {
        let peers = self.groups[group].clone();
        if requests.len() != peers.len() {
            return Err(Error::Invariant(
                "a text step has one invocation per microbatch".into(),
            ));
        }
        let state = &self.state;
        let mut step = |runner: &TextRunner, request: &StepRequest<'_>| {
            runner.stage(
                request,
                &mut StepContext {
                    tables: context.tables,
                    cache: context.cache,
                    state,
                },
            )
        };
        let Some(experts) = experts else {
            let [runner] = peers[..] else {
                return Err(Error::Invariant(
                    "microbatch runners require expert execution".into(),
                ));
            };
            let request = requests[0]
                .as_ref()
                .ok_or(Error::State("a text step has an invocation"))?;
            let staged = step(&self.runners[runner], request)?;
            let launch = self.runners[runner].launch(&staged, None, true)?;
            launch.call.invoke()?;
            let executed = self.runners[runner].finish(staged, None, launch.mode)?;
            return Ok(vec![Some(executed)]);
        };

        let lane = self.runners[peers[0]]
            .stream()
            .ok_or(Error::State("expert steps run on a CUDA stream"))?
            .clone();
        experts.fork(&lane)?;
        let mut staged = Vec::with_capacity(peers.len());
        for (&runner, request) in peers.iter().zip(requests) {
            staged.push(
                request
                    .as_ref()
                    .map(|request| step(&self.runners[runner], request))
                    .transpose()?,
            );
        }
        let tokens = staged
            .iter()
            .flatten()
            .map(Staged::expert_tokens)
            .max()
            .unwrap_or(0);
        let capacity = experts.capacity(tokens, group as i64, forced)?;
        let replay = experts.replays();
        let mut modes = Vec::with_capacity(peers.len());
        let mut calls = Vec::with_capacity(peers.len());
        for (&runner, staged) in peers.iter().zip(&staged) {
            match staged {
                Some(staged) => {
                    let launch = self.runners[runner].launch(staged, Some(capacity), replay)?;
                    modes.push(Some(launch.mode));
                    calls.push(Some(super::experts::MicrobatchForward {
                        call: launch.call,
                        yields: launch.mode != GraphMode::Replay,
                    }));
                }
                None => {
                    modes.push(None);
                    calls.push(None);
                }
            }
        }
        {
            let _range = crate::profiling::range(c"uniserve.expert.step", None);
            experts.run(&lane, calls, capacity)?;
        }
        let mut executed = Vec::with_capacity(peers.len());
        for ((&runner, staged), mode) in peers.iter().zip(staged).zip(modes) {
            executed.push(match (staged, mode) {
                (Some(staged), Some(mode)) => {
                    Some(self.runners[runner].finish(staged, Some(capacity), mode)?)
                }
                _ => None,
            });
        }
        Ok(executed)
    }

    /// The order of this runtime's batches, for other runtimes that also
    /// write what text batches read (guidance prefixes, latent images'
    /// K/V).
    pub(crate) fn order_mut(&mut self) -> &mut StreamOrder {
        &mut self.order
    }

    /// The runner on the lane stream of the group serving `component`'s
    /// prefills, and the group.
    fn prefill_lane(&self, component: &str) -> Result<(usize, usize)> {
        let group = self
            .route(component, CallKind::Forward(ForwardMode::Prefill))
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "component {component:?} has no text prefill binding"
                ))
            })?;
        Ok((self.groups[group][0], group))
    }

    /// The device, stream and payload words of work that reads or writes
    /// `component`'s K/V beside its text batches: its prefill lane's.
    pub(crate) fn placement(
        &self,
        component: &str,
    ) -> Result<(Option<i32>, Option<Stream>, usize)> {
        let (lane, _) = self.prefill_lane(component)?;
        let runner = &self.runners[lane];
        let stream = runner.stream().cloned();
        Ok((runner.device(), stream, 0))
    }

    /// The unit reset column of `component`'s prefill lane stream (see
    /// [`Self::placement`]), for work that assigns K/V units beside its
    /// text batches.
    pub(crate) fn unit_column(&self, component: &str) -> Result<&UnitColumn> {
        let (lane, _) = self.prefill_lane(component)?;
        Ok(&self.columns[lane])
    }

    /// Write the K/V of token prefixes, each from position zero of its
    /// slot, with `component`'s prefill lane runner: guidance prefixes of
    /// image trajectories. The prefixes are evaluated causally and only
    /// their K/V is kept; slot continuation state is unchanged. The caller
    /// installed the slots' block tables and retains their extents. On a
    /// rank of an expert union the evaluation is an expert step whose other
    /// microbatches join.
    pub(crate) fn prefill_cache(
        &mut self,
        experts: Option<&super::experts::ExpertRuntime>,
        component: &str,
        prefixes: &[(u32, &[u32])],
        tables: &BlockTables,
        cache: &mut KVCacheManager<Arc<Retirement>>,
    ) -> Result<Executed> {
        let (lane, group) = self.prefill_lane(component)?;
        let rows: Vec<TokenRow> = prefixes
            .iter()
            .map(|&(slot, tokens)| {
                Ok(TokenRow {
                    slot,
                    source: RowSource::Host {
                        tokens: tokens.to_vec(),
                        position: 0,
                    },
                    prefix: 0,
                    query: u32::try_from(tokens.len())
                        .map_err(|_| Error::Invalid("prefix exceeds the token range".into()))?,
                    selection: TokenSelection::Cache,
                    gated: false,
                    force_finish: false,
                    causal: true,
                })
            })
            .collect::<Result<_>>()?;
        let stream = self.runners[lane].stream().cloned();
        self.order.wait(stream.as_ref())?;

        // The lane's runner evaluates every prefix.
        let mut requests: Vec<Option<StepRequest<'_>>> = vec![Some(StepRequest {
            decode: false,
            rows: &rows,
            sampling: None,
            prompt: &[],
            embeddings: &[],
        })];
        requests.resize_with(self.groups[group].len(), || None);
        let executed = self
            .invoke(
                experts,
                group,
                &requests,
                &mut StepParts { tables, cache },
                None,
            )?
            .into_iter()
            .next()
            .flatten()
            .ok_or_else(|| Error::Invariant("the prefix evaluation did not run".into()))?;
        self.order.record(stream.as_ref())?;
        Ok(executed)
    }

    pub(crate) fn set_wake(&self, _wake: Option<Wake>) {}

    /// Release graphs, steps and output storage after the lanes drained.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        for runner in &mut self.runners {
            result = result.and(runner.close(aborted));
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
}

impl RequestStorage for TextRuntime {
    fn admit_slots(&mut self, slots: &[usize]) -> Result<()> {
        // Reset on the first runner's stream; other streams order after it
        // before their next batch.
        let stream = self.runners.first().and_then(TextRunner::stream);
        self.order.wait(stream)?;
        self.state.reset(slots, stream)?;
        self.order.record(stream)
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
            batch.launch(launched);
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

/// Reserve, step and commit one batch. `None` on a rank without a text
/// runtime.
fn run(parts: Parts<'_>, plan: &BatchPlan) -> std::result::Result<Option<Launched>, Failure> {
    let started = Instant::now();
    let (Some(runtime), Some(storage)) = (parts.text, parts.storage) else {
        return Ok(None);
    };
    let mut batch = TokenBatch::open(
        runtime,
        parts.image,
        storage,
        parts.requests,
        parts.experts,
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
                timer: "text_materialize",
                host: None,
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
            invocations: Vec::new(),
            executed: crate::runner::text::Executed {
                mode: GraphMode::Eager,
                live_tokens: 0,
                padded_tokens: 0,
            },
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

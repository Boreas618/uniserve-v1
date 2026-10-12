//! The causal LM runners of a rank and the batches they serve: prefill,
//! decode and speculative-verify calls of causal language models.
//!
//! The control thread prepares a token batch and applies its results; the
//! lane's execution thread issues its stream work, which the batch records
//! as jobs ([`forward`] prepares the next one, [`complete`] applies its
//! completion). A batch is one job, unless host logic needs a result
//! before it can go on. Its stages:
//!
//! 1. **Reservation** ([`super::batch`]): bind every call's pending output and the
//!    batch's output lease; install its block tables, reset newly assigned
//!    units and retain the KV extents it reads and writes until its device
//!    work retires; bind its scalar products (token relays, transition and
//!    completion flags); lease its predicates. A call whose host predicate
//!    is false is skipped (`Predicated`) and only publishes false flags.
//! 2. **Step** ([`super::step`]): one input row per active call, the sampling tasks
//!    and prompt scoring, and one forward on the call kind's runner.
//! 3. **Capture and commit** ([`super::commit`]): advance each call's progress, copy
//!    completion and logprob words to the output lease, relay sampled tokens
//!    and transitions into their products, then commit products, slot state
//!    and pending calls together. A failure before the commit point retires
//!    everything the batch reserved and reports every call as failed; a
//!    failure after it is fatal.
//! 4. **Materialization** ([`crate::executor::forward::launch::materialize`]):
//!    once the
//!    lease's device fences complete, decode each call's result and accept
//!    its progress.
//!
//! Two results end a job before the batch's last one: the bytes of
//! host-gated predicates, which decide the calls stage 1 reserves for
//! (the opening job reads them), and the image denoiser's conditioning of
//! latent images, a forward that still issues itself and so closes the job
//! that holds the reservation, before the step. A graph captured after its
//! eager warmup needs no job of its own: the capture is an operation of the
//! job, whose completion carries the graph.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use uniserve_cuda::Stream;
use uniserve_ipc::{BufferId, CallKind, ForwardMode, ForwardStats, RequestKey, Wake};

use crate::cuda_graph::GraphMode;
use crate::error::WorkerError;
use crate::executor::BatchResult;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::forward::Stage;
use crate::executor::forward::launch::{Launched, StreamOrder, error_code, failed_output};
use crate::executor::lifecycle::RequestStorage;
use crate::lane::{Completion, ExpertStep, ExpertVariant, Job};
use crate::parallel::{ExpertUnion, MicrobatchForward};
use crate::profiling::kernels::KernelCalls;
use crate::request::decode_state::DecodeState;
use crate::runner::TokenSelection;
use crate::runner::causal_lm::{
    CausalLMRunner, Executed, PromptChunks, RowSource, Staged, StepContext, StepRequest, TokenRow,
};
use crate::storage::{BlockTables, KVCacheManager, OutputPool, UnitColumn};
use crate::transfer::Retirement;
use crate::{Error, Result};

/// The KV storage a text step borrows.
pub(crate) struct StepParts<'a> {
    pub(crate) tables: &'a BlockTables,
    pub(crate) cache: &'a mut KVCacheManager<Arc<Retirement>>,
}

use super::batch::{Batch, Opening, Owners, TokenBatch};
use super::commit::{Commit, Committed, Failure};
use super::step::{Rows, condition_latents};

/// One step's forwards as a job holds them, awaiting its completion.
pub(crate) struct Step {
    group: usize,
    /// Each microbatch's staged forward; none for an empty microbatch.
    staged: Vec<Option<Staged>>,
    /// How each populated microbatch's forward runs at every capacity the
    /// step can be agreed at; one entry of no capacity without experts.
    modes: Vec<(Option<usize>, Vec<Option<GraphMode>>)>,
}

/// A token batch between two of its jobs.
pub(crate) enum Pending {
    /// The job on the lane reads the batch's host-gated predicates.
    Predicates(Opening),
    /// The predicates are known: the batch reserves its storage next.
    Opened(Opening),
    /// The job on the lane ends with the image denoiser's conditioning of
    /// the batch's latent images.
    Conditioning(Box<TokenBatch>, Rows),
    /// The latent images are conditioned: the batch steps next.
    Conditioned(Box<TokenBatch>, Rows),
    /// The job on the lane is the batch's last.
    Committing(Box<TokenBatch>, Commit),
    /// A job of the batch failed before its commit point: the batch
    /// retires what it reserved next.
    Failing(Box<TokenBatch>, Failure),
    /// The job on the lane retires what the failed batch reserved.
    Discarding(Failure),
    /// The batch failed and holds nothing: its calls are reported next.
    Failed(Failure),
}

/// Fields of a call's completion section: valid, active, token, accepted.
pub(super) const COMPLETION_FIELDS: usize = 4;

/// The causal LM runners of a rank: one runner per lane binding, or per
/// expert microbatch of it, the slot continuation state they share, and
/// their output storage.
pub(crate) struct CausalLMRunners {
    pub(super) runners: Vec<CausalLMRunner>,
    /// Each runner's kernel-table label (component and module path) and its
    /// device name.
    pub(super) labels: Vec<(String, String)>,
    /// Each lane binding's runners, one per expert microbatch; the first
    /// runs on the lane's stream, the others on its forks.
    pub(super) groups: Vec<Vec<usize>>,
    /// The NCCL communicators of the runners' streams, destroyed after
    /// their graphs.
    pub(super) collectives: Vec<crate::parallel::StreamCollectives>,
    /// The unit reset column of each runner's stream.
    pub(super) columns: Vec<UnitColumn>,
    /// The group serving each component's call kind.
    pub(super) routes: HashMap<(String, CallKind), usize>,
    pub(super) state: DecodeState,
    pub(super) outputs: OutputPool,
    /// Global vocabulary of sampling.
    pub(super) vocab: usize,
    /// Orders slot state across runner streams: recorded after each text
    /// batch's last write, waited on by the next batch on another stream.
    pub(super) order: StreamOrder,
}

impl CausalLMRunners {
    pub(super) fn route(&self, component: &str, kind: CallKind) -> Option<usize> {
        self.routes.get(&(component.to_owned(), kind)).copied()
    }

    /// Seal every runner's graphs.
    pub(crate) fn seal(&mut self) {
        for runner in &mut self.runners {
            runner.seal();
        }
    }

    /// The order of these runners' batches across streams; other runtimes
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

    /// Record one invocation per microbatch runner of `group` (`None` for
    /// an empty microbatch) in `job` as one numerical step.
    ///
    /// Without experts the group has one runner, and its forward's
    /// operations follow its input writes in the job. With experts, the
    /// step is an expert step: every microbatch stream forks from the
    /// group's lane stream before inputs are staged, and one operation
    /// holds the rest. The lane's execution thread agrees on the step's
    /// capacity with the union's ranks (or, during warmup, runs it at
    /// `forced` or the smallest capacity holding its tokens, and announces
    /// it), then runs the microbatches' forwards for that capacity in
    /// rotation, empty ones joining, with the lane stream waiting on every
    /// microbatch stream. The job carries the forward of every capacity
    /// the agreement can choose, and once what follows a forward at any of
    /// them. Microbatches launch their graphs only once
    /// the union joins by launching its own
    /// ([`crate::parallel::ExpertUnion::replays`]).
    pub(crate) fn forward(
        &self,
        experts: Option<&Arc<ExpertUnion>>,
        group: usize,
        requests: &[Option<StepRequest<'_>>],
        context: &mut StepParts<'_>,
        forced: Option<usize>,
        job: &Job<'_>,
    ) -> Result<Step> {
        let peers = &self.groups[group];
        if requests.len() != peers.len() {
            return Err(Error::Invariant(
                "a text step has one invocation per microbatch".into(),
            ));
        }
        let state = &self.state;
        let mut stage = |runner: &CausalLMRunner, request: &StepRequest<'_>| {
            runner.stage(
                request,
                &mut StepContext {
                    tables: context.tables,
                    cache: context.cache,
                    state,
                },
                job,
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
            let staged = stage(&self.runners[runner], request)?;
            let then = Job::new();
            let mode = self.runners[runner].forward(&staged, None, true, job, &then)?;
            self.runners[runner].select(&staged, job)?;
            job.append(then);
            return Ok(Step {
                group,
                staged: vec![Some(staged)],
                modes: vec![(None, vec![Some(mode)])],
            });
        };

        let lane = self.runners[peers[0]]
            .stream()
            .ok_or(Error::State("expert steps run on a CUDA stream"))?
            .clone();
        experts.fork(job, &lane)?;
        let mut staged = Vec::with_capacity(peers.len());
        for (&runner, request) in peers.iter().zip(requests) {
            staged.push(
                request
                    .as_ref()
                    .map(|request| stage(&self.runners[runner], request))
                    .transpose()?,
            );
        }
        let tokens = staged
            .iter()
            .flatten()
            .map(Staged::expert_tokens)
            .max()
            .unwrap_or(0);

        // The capacities the step can run at: the one a warmup step
        // announces, or every capacity an agreement can choose for a rank
        // that sends `tokens` tokens.
        let holding = experts.parallel().capacities_for(tokens);
        let capacities: Vec<usize> = if experts.warming() {
            match forced {
                Some(capacity) => vec![capacity],
                None => holding.take(1).collect(),
            }
        } else {
            holding.collect()
        };
        if capacities.is_empty() {
            return Err(Error::Invalid(format!(
                "{tokens} tokens exceed the expert exchange's tokens per rank"
            )));
        }
        let replay = experts.replays();
        let mut after = Vec::with_capacity(peers.len());
        for (&runner, staged) in peers.iter().zip(&staged) {
            after.push(match staged {
                Some(staged) => {
                    let selection = Job::new();
                    self.runners[runner].select(staged, &selection)?;
                    Some(selection)
                }
                None => None,
            });
        }
        let mut variants = Vec::with_capacity(capacities.len());
        let mut modes = Vec::with_capacity(capacities.len());
        for capacity in capacities {
            let then = Job::new();
            let mut forwards = Vec::with_capacity(peers.len());
            let mut ran = Vec::with_capacity(peers.len());
            for (&runner, staged) in peers.iter().zip(&staged) {
                let Some(staged) = staged else {
                    forwards.push(None);
                    ran.push(None);
                    continue;
                };
                let forward = Job::new();
                let mode = self.runners[runner].forward(
                    staged,
                    Some(capacity),
                    replay,
                    &forward,
                    &then,
                )?;
                forwards.push(Some(MicrobatchForward {
                    job: forward,
                    // An eager forward enters Python for the model's
                    // forward, which passes the host turn at its expert
                    // dispatch points.
                    yields: mode != GraphMode::Replay,
                }));
                ran.push(Some(mode));
            }
            variants.push(ExpertVariant {
                capacity,
                forwards,
                then,
            });
            modes.push((Some(capacity), ran));
        }
        job.expert_step(ExpertStep {
            union: Arc::clone(experts),
            lane,
            tokens,
            kind: group as i64,
            forced,
            variants,
            after,
        });
        Ok(Step {
            group,
            staged,
            modes,
        })
    }

    /// Apply the completion of the job that ran `step`: every populated
    /// microbatch's runner stores the graph its forward captured and
    /// counts how it ran, at the capacity the step was agreed at.
    pub(crate) fn complete(
        &mut self,
        step: Step,
        completion: &mut Completion,
    ) -> Result<Vec<Option<Executed>>> {
        let capacity = match step.modes.first() {
            Some((None, _)) => None,
            _ => Some(completion.capacity()?),
        };
        let modes = step
            .modes
            .into_iter()
            .find(|(variant, _)| *variant == capacity)
            .ok_or_else(|| Error::Invariant("an expert step ran at an unplanned capacity".into()))?
            .1;
        let peers = self.groups[step.group].clone();
        let mut executed = Vec::with_capacity(peers.len());
        for ((&runner, staged), mode) in peers.iter().zip(step.staged).zip(modes) {
            executed.push(match (staged, mode) {
                (Some(staged), Some(mode)) => {
                    Some(self.runners[runner].complete(staged, capacity, mode, completion)?)
                }
                _ => None,
            });
        }
        Ok(executed)
    }

    /// A call kind `group` serves, which names the lane its jobs run on.
    pub(crate) fn kind(&self, group: usize) -> Result<CallKind> {
        self.routes
            .iter()
            .find(|(_, routed)| **routed == group)
            .map(|((_, kind), _)| *kind)
            .ok_or_else(|| Error::Invariant("a text runner group serves no call kind".into()))
    }

    /// The order of these runners' batches, for other runtimes that also
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
    ///
    /// The evaluation is recorded into `job`, the job of the image
    /// denoiser's forward that needs the prefixes, behind what that forward
    /// recorded before. [`Self::complete_prefill`] applies the job's
    /// completion to the returned step.
    pub(crate) fn prefill_cache(
        &mut self,
        experts: Option<&Arc<ExpertUnion>>,
        component: &str,
        prefixes: &[(u32, &[u32])],
        tables: &BlockTables,
        cache: &mut KVCacheManager<Arc<Retirement>>,
        job: &Job<'_>,
    ) -> Result<Step> {
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
        self.order.wait(job.on(stream.as_ref()))?;

        // The lane's runner evaluates every prefix.
        let prompt = PromptChunks::default();
        let mut requests: Vec<Option<StepRequest<'_>>> = vec![Some(StepRequest {
            mode: ForwardMode::Prefill,
            rows: &rows,
            finish_ids: &[],
            sampling: None,
            prompt: &prompt,
            embeddings: &[],
        })];
        requests.resize_with(self.groups[group].len(), || None);
        let step = self.forward(
            experts,
            group,
            &requests,
            &mut StepParts { tables, cache },
            None,
            job,
        )?;
        self.order.record(job.on(stream.as_ref()))?;
        Ok(step)
    }

    /// Apply the completion of the job that performed a prefix evaluation
    /// ([`Self::prefill_cache`]) to its `step`.
    pub(crate) fn complete_prefill(
        &mut self,
        step: Step,
        completion: &mut Completion,
    ) -> Result<Executed> {
        self.complete(step, completion)?
            .into_iter()
            .next()
            .flatten()
            .ok_or_else(|| Error::Invariant("the prefix evaluation did not run".into()))
    }

    pub(crate) fn set_wake(&self, _wake: Option<Wake>) {}

    /// Each runner's kernel-table label, device name and the call that
    /// lists the kernels its operators selected.
    pub(crate) fn kernel_calls(&self) -> Vec<KernelCalls> {
        self.runners
            .iter()
            .zip(&self.labels)
            .map(|(runner, (label, device))| {
                let call = runner.methods.call(&runner.functions.kernels, Vec::new());
                (label.clone(), device.clone(), vec![call])
            })
            .collect()
    }

    /// Release graphs, the Python runners' execution contexts and output
    /// storage after the lanes drained, newest runner first. An aborted
    /// close keeps the contexts: device work may still address their
    /// workspaces.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        for runner in self.runners.iter_mut().rev() {
            result = result.and(runner.close(aborted));
            if !aborted {
                let closed = runner
                    .methods
                    .call(&runner.functions.close, Vec::new())
                    .invoke()
                    .map(drop);
                result = result.and(closed);
            }
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

impl RequestStorage for CausalLMRunners {
    fn admit_slots(&mut self, slots: &[usize]) -> Result<()> {
        // Reset on the first runner's stream; other streams order after it
        // before their next batch.
        let stream = self.runners.first().and_then(CausalLMRunner::stream);
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

/// The next job of a token batch on its lane, or `None` when the batch
/// advanced without one: it failed and its calls are reported.
pub(crate) fn forward<'a>(
    backend: &'a mut WorkerBackend,
    batch: &'a mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<Option<Job<'a>>, WorkerError> {
    let experts = backend.experts().cloned();
    let state = match batch.token_parts().1.take() {
        Some(Pending::Failed(failure)) => return fail(backend, batch, failure).map(|()| None),
        state => state,
    };
    let (plan, pending, issued) = batch.token_parts();

    let parts = backend.parts();
    let (Some(runtime), Some(storage)) = (parts.text, parts.storage) else {
        return Err(crate::executor::forward::unsupported(kind, "text"));
    };
    let mut owners = Owners {
        runtime,
        image: parts.image,
        storage,
        requests: parts.requests,
        plan,
        experts,
    };
    let job = Job::new();
    let (job, next) = match state {
        None => match Opening::open(&mut owners, parts.info, parts.rank, &job) {
            Ok(opening) if opening.reads_predicates() => (Some(job), Pending::Predicates(opening)),
            Ok(opening) => reserve(owners, opening, job, issued),
            // The fences the opening recorded are still due.
            Err(error) if job.is_empty() => (None, Pending::Failed(error.into())),
            Err(error) => (Some(job), Pending::Discarding(error.into())),
        },
        Some(Pending::Opened(opening)) => reserve(owners, opening, job, issued),
        Some(Pending::Conditioned(batch, rows)) => {
            let mut batch = Batch::resume(owners, *batch, job);
            let committing = batch
                .step(rows)
                .map_err(Failure::from)
                .and_then(|stepped| batch.finish(Some(stepped)));
            commit(batch, committing)
        }
        Some(Pending::Failing(batch, failure)) => {
            let batch = Batch::resume(owners, *batch, job);
            discard(batch, failure)
        }
        Some(
            state @ (Pending::Predicates(_)
            | Pending::Conditioning(..)
            | Pending::Committing(..)
            | Pending::Discarding(_)
            | Pending::Failed(_)),
        ) => {
            // A job of the batch is on the lane: its completion comes
            // before the next job.
            *pending = Some(state);
            return Err(WorkerError::classify(
                Error::Invariant(
                    "a token batch's job was prepared before its last completed".into(),
                ),
                Some("token batch"),
            ));
        }
    };
    *pending = Some(next);
    Ok(job)
}

/// Apply the completion of the token batch's job on the lane.
pub(crate) fn complete(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    completion: Completion,
) -> std::result::Result<(), WorkerError> {
    let experts = backend.experts().cloned();
    let (_, pending, issued) = batch.token_parts();
    let state = pending.take();
    let conditioned = issued.take();
    let next = match (state, completion.check()) {
        // A predicate that could not be read fails the batch, which has
        // reserved nothing yet.
        (Some(Pending::Predicates(_)), Err(failure)) => Pending::Failed(failure.error.into()),
        (Some(Pending::Predicates(mut opening)), Ok(mut completion)) => {
            match opening.predicates(&mut completion) {
                Ok(()) => Pending::Opened(opening),
                Err(error) => Pending::Failed(error.into()),
            }
        }
        (Some(Pending::Conditioning(state, rows)), completed) => {
            let conditioned =
                conditioned.unwrap_or(Err(Error::State("the latent conditioning did not run")));
            let completed = completed.map(drop).map_err(|failure| failure.error);
            match completed.and(conditioned) {
                Ok(()) => Pending::Conditioned(state, rows),
                Err(error) => Pending::Failing(state, error.into()),
            }
        }
        (Some(Pending::Committing(state, commit)), completed) => {
            let (plan, _, _) = batch.token_parts();
            let parts = backend.parts();
            let (Some(runtime), Some(storage)) = (parts.text, parts.storage) else {
                return Err(WorkerError::classify(
                    Error::State("the text runners left a batch in flight"),
                    Some("token batch"),
                ));
            };
            let owners = Owners {
                runtime,
                image: parts.image,
                storage,
                requests: parts.requests,
                plan,
                experts,
            };
            let mut token_batch = Batch::resume(owners, *state, Job::new());
            let committed = match completed {
                Ok(mut completion) => token_batch.complete(commit, &mut completion),
                Err(failure) => Err(Failure {
                    committed: Batch::committed(&commit, failure.operation),
                    error: failure.error,
                }),
            };
            match committed {
                Ok(committed) => {
                    let stats = statistics(&token_batch, committed.as_ref());
                    let (_, state, _) = token_batch.park();
                    let launched = state.launch(stats);
                    batch.launch(launched);
                    batch.set_stage(Stage::Done);
                    return Ok(());
                }
                // Products and slot state may already be visible; the rank
                // cannot roll them back.
                Err(failure) if failure.committed => Pending::Failed(failure),
                Err(failure) => {
                    let (_, state, _) = token_batch.park();
                    Pending::Failing(Box::new(state), failure)
                }
            }
        }
        // Storage the cleanup could not retire stays reserved; the rank
        // cannot reuse it safely.
        (Some(Pending::Discarding(_)), Err(failure)) => Pending::Failed(Failure {
            error: failure.error,
            committed: true,
        }),
        (Some(Pending::Discarding(failure)), Ok(_)) => Pending::Failed(failure),
        (state, _) => {
            *batch.token_parts().1 = state;
            return Err(WorkerError::classify(
                Error::Invariant("a token batch has no job on its lane".into()),
                Some("token batch"),
            ));
        }
    };
    *batch.token_parts().1 = Some(next);
    Ok(())
}

/// Lease the opened batch's output storage, reserve for its active calls
/// and record its step and commit, or its job up to the image denoiser's
/// conditioning of its latent images.
fn reserve<'a>(
    owners: Owners<'a>,
    opening: Opening,
    job: Job<'a>,
    issued: &'a mut Option<Result<()>>,
) -> (Option<Job<'a>>, Pending) {
    let mut batch = match Batch::lease(owners, opening, job) {
        Ok(batch) => batch,
        // Nothing is reserved yet, and the job was dropped with only the
        // batch's order in it.
        Err(error) => return (None, Pending::Failed(error.into())),
    };
    let rows = batch.reserve().map_err(Failure::from).and_then(|()| {
        if batch.calls.iter().any(|state| state.active) {
            batch.rows().map(Some).map_err(Failure::from)
        } else {
            Ok(None)
        }
    });
    match rows {
        Ok(Some(mut rows)) if !rows.conditions.is_empty() => {
            // The image denoiser issues its own stream work: it ends this
            // job, and the step follows in the next.
            let conditions = std::mem::take(&mut rows.conditions);
            let (job, state, owners) = batch.park();
            let stream = state.stream.clone();
            let Owners {
                runtime,
                mut image,
                storage,
                ..
            } = owners;
            job.forward(move || {
                *issued = Some(condition_latents(
                    runtime,
                    &mut image,
                    storage,
                    &conditions,
                    stream.as_ref(),
                ));
            });
            (Some(job), Pending::Conditioning(Box::new(state), rows))
        }
        Ok(Some(rows)) => {
            let committing = batch
                .step(rows)
                .map_err(Failure::from)
                .and_then(|stepped| batch.finish(Some(stepped)));
            commit(batch, committing)
        }
        Ok(None) => {
            let committing = batch.finish(None);
            commit(batch, committing)
        }
        Err(failure) => discard(batch, failure),
    }
}

/// The batch's last job, or its retirement where recording the job failed.
fn commit<'a>(
    batch: Batch<'a>,
    committing: std::result::Result<Commit, Failure>,
) -> (Option<Job<'a>>, Pending) {
    match committing {
        Ok(commit) => {
            let (job, state, _) = batch.park();
            (Some(job), Pending::Committing(Box::new(state), commit))
        }
        Err(failure) => discard(batch, failure),
    }
}

/// Retire what a failed batch reserved: the job that records its fences, or
/// nothing where it holds no stream work.
fn discard<'a>(mut batch: Batch<'a>, failure: Failure) -> (Option<Job<'a>>, Pending) {
    if failure.committed {
        return (None, Pending::Failed(failure));
    }
    if let Err(cleanup) = batch.discard() {
        // Storage the cleanup could not retire stays reserved; the rank
        // cannot reuse it safely.
        return (
            None,
            Pending::Failed(Failure {
                error: cleanup,
                committed: true,
            }),
        );
    }
    let (job, _, _) = batch.park();
    if job.is_empty() {
        (None, Pending::Failed(failure))
    } else {
        (Some(job), Pending::Discarding(failure))
    }
}

/// Report a failed batch: a failure past the commit point stops the rank,
/// any other completes the batch with its calls failed.
fn fail(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    failure: Failure,
) -> std::result::Result<(), WorkerError> {
    let plan = Arc::clone(batch.shared_plan());
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

impl TokenBatch {
    /// The committed batch as launched work awaiting materialization.
    fn launch(self, stats: ForwardStats) -> Launched {
        Launched {
            outputs: self.calls.into_iter().map(|state| state.output).collect(),
            buffer: self.buffer,
            started: self.started,
            stats,
            timer: "text_materialize",
            host: None,
        }
    }
}

fn statistics(batch: &Batch<'_>, committed: Option<&Committed>) -> ForwardStats {
    let mut stats = ForwardStats::default();
    let elapsed = batch.started.elapsed().as_micros() as u64;
    stats.component_us.insert("text_execute".into(), elapsed);
    let Some(committed) = committed else {
        return stats;
    };
    let executed = committed.executed;
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
        stats.spec_verify_rows += committed.rows as u64;
        stats.spec_verify_draft_tokens += committed.drafts;
    }
    stats
}

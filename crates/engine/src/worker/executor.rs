//! Submission of explicitly targeted calls to worker groups.
//!
//! `WorkerExecutor` is the `Executor` the scheduler drives over a set of
//! `WorkerGroup`s. It splits each logical `ExecutionBatch` into one
//! `WorkerSubmission` per target worker, delivers each request's `Start` with
//! its first call on a worker and every other lifecycle command to the workers
//! that hold what it releases, and joins the workers' reports back into
//! `BatchResult`s: call completions are published as they arrive, command
//! receipts once every worker has finished the batch.
//!
//! Each destination has independent bounded capacity. Every worker has its own
//! FIFO of submissions whose head waits until the products it depends on are
//! published. A blocked queue does not hold back other workers, and each
//! worker receives its batches, and therefore a request's commands, in
//! submission order.
//!
//! The executor also holds the engine's copy of every cross-worker product
//! export until the scheduler frees the buffer or its request finishes
//! without retaining it, and on a `WorkerFailure` computes the calls,
//! requests, and buffers the failure invalidates.

mod codecs;
mod failure;
mod results;

use crate::executor::WorkerResult;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Duration;

use super::{BatchSubmitError, IncarnationRegistry, WorkerGroup};
use crate::CommandWaker;
use crate::executor::{
    BatchResult, CommandOutcome, ExecutionBatch, Executor, ExecutorInfo, ExecutorSubmitError,
    RankBuffer, RequestPlacement, TransferConfig, UnroutableInputs, WorkerFailure, WorkerGroupId,
};
use anyhow::Context;
use ipc::{
    BatchCommand, BufferAllocation, BufferId, CacheTransfer, Call, NewRequest, RequestKey,
    TensorExport, TensorRef,
};

pub(crate) use codecs::video_unit_codecs;

/// One worker's outstanding submission; returned calls leave this record.
struct PendingGroup {
    /// Whether the `WorkerGroup` accepted the submission and still owes its
    /// report. False while the submission waits in
    /// `WorkerExecutor::worker_submissions`, and cleared by failure
    /// reconciliation when the group will send no report.
    submitted: bool,
    /// Calls whose completion this worker has not yet returned, with each
    /// call's kind, which the completion is checked against when it is
    /// joined with its call.
    calls: HashMap<(RequestKey, ipc::CallId), ipc::CallKind>,
}

/// Dispatch and receipts for one scheduler batch.
struct PendingBatch {
    /// Workers, by index into `WorkerExecutor::workers`, that have not yet
    /// finished their part of the batch. The batch is terminal when empty.
    workers: HashMap<usize, PendingGroup>,
    /// The batch's lifecycle commands without `Start` admissions, so a
    /// position here is a `CommandResult::command_index`.
    commands: Vec<BatchCommand>,
    /// Outcomes other than `Applied`, keyed by position in `commands`.
    command_outcomes: HashMap<u32, CommandOutcome>,
}

impl PendingBatch {
    /// Records `outcome` for every command that targets one of `requests`.
    ///
    /// The first recorded outcome is kept, except that `Failed` replaces any
    /// other: a failed release keeps ownership even when another endpoint
    /// retires it.
    fn set_command_outcome(&mut self, requests: &HashSet<RequestKey>, outcome: CommandOutcome) {
        for (index, command) in self.commands.iter().enumerate() {
            if requests.contains(&command.request_key()) {
                let current = self.command_outcomes.entry(index as u32).or_insert(outcome);
                if outcome == CommandOutcome::Failed {
                    *current = outcome;
                }
            }
        }
    }
}

/// One worker's part of a logical batch, queued until it can be dispatched.
#[derive(Clone)]
struct WorkerSubmission {
    /// The calls routed to this worker, with its commands and the external
    /// exports it reads.
    batch: ExecutionBatch,
    /// Products this worker reads that another worker, or a component of this
    /// worker that does not share the reader's storage, produces. The
    /// submission stays at its queue head until each has an export in
    /// `transfer_products` or `cache_transfers`.
    dependencies: Vec<BufferId>,
}

impl WorkerSubmission {
    /// Builds the one submission a worker receives for this batch.
    ///
    /// A batch carries one call kind to one component, so a worker receives
    /// it once, under one identity, and answers it once. Product inputs the
    /// worker does not yet hold travel as dependencies and hold the submission
    /// in its queue until they resolve. Inputs, cache inputs, and dependencies
    /// that none of `calls` reads are dropped.
    fn build(
        batch_id: u64,
        calls: Vec<(Call, RequestPlacement)>,
        commands: Vec<BatchCommand>,
        inputs: Vec<TensorExport>,
        cache_inputs: Vec<CacheTransfer>,
        page_offloads: Option<crate::executor::PageOffloads>,
        dependencies: Vec<BufferId>,
    ) -> Self {
        let consumed = calls
            .iter()
            .flat_map(|(call, _)| call.input_buffers())
            .collect::<HashSet<_>>();
        let input_transfers = inputs
            .into_iter()
            .filter(|payload| consumed.contains(&payload.product.buffer_id()))
            .collect();
        let dependencies = dependencies
            .into_iter()
            .filter(|product| consumed.contains(product))
            .collect();
        let mut batch = ExecutionBatch::new(batch_id, calls, commands, input_transfers);
        batch.cache_inputs = cache_inputs
            .into_iter()
            .filter(|export| consumed.contains(&export.source))
            .collect();
        batch.page_offloads = page_offloads;
        Self {
            batch,
            dependencies,
        }
    }
}

/// What became of one queued submission offered to its worker.
enum Dispatch {
    /// The group accepted the submission.
    Submitted,
    /// The group had no free slot; nothing was recorded.
    Deferred,
    /// The group refused inputs that no configured edge carries to the rank
    /// reading them; nothing was recorded. The failure names the requests
    /// whose calls read those inputs.
    Refused(Box<WorkerFailure>),
}

/// The worker and component whose call produces a buffer.
#[derive(Clone)]
struct BufferRoute {
    worker_index: usize,
    component: String,
    /// The buffer is one of its producing call's declared `outputs`, which
    /// a component publishes from its output ranks
    /// (`ComponentConfig::publishes_on_every_rank`). Every member rank
    /// holds its own copy of the call's other results: its device scalars,
    /// encoder features, image, latents and prefix cache.
    declared_output: bool,
}

/// Dispatch targeted calls and track their independently completed results.
///
/// Worker indices used throughout are positions in `workers`.
pub struct WorkerExecutor {
    /// Worker identity to its index in `workers`.
    routing: HashMap<WorkerGroupId, usize>,
    workers: Vec<(WorkerGroupId, WorkerGroup)>,
    /// The worker and rank of every live rank incarnation, which the groups
    /// keep current as they replace ranks.
    incarnations: IncarnationRegistry,
    executor_info: ExecutorInfo,
    /// Sum of the workers' queue depths; `submit` refuses a batch with
    /// `WouldBlock` while this many logical batches are pending.
    depth: usize,
    /// Command-channel wake, polled beside the workers' progress descriptors.
    command_wake: crate::handle::WakeSignal,
    /// Logical batches that have not yet published their terminal result.
    pending: BTreeMap<u64, PendingBatch>,
    /// Results `poll` returns in order.
    ready: VecDeque<BatchResult>,
    /// Admission descriptor of every started request until its `Finish`
    /// completes; a worker receives it as a `Start` with the request's first
    /// call there.
    admissions: HashMap<RequestKey, NewRequest>,
    /// (worker, request) pairs whose `Start` the worker has accepted.
    admitted_workers: HashSet<(usize, RequestKey)>,
    /// Producing worker and component of each output buffer of a submitted
    /// call, until the buffer is freed or its request finishes without
    /// retaining it.
    buffer_routes: HashMap<BufferId, BufferRoute>,
    /// Persistent buffer addresses are local to the buffer pool of one rank of
    /// a physical Worker, keyed by worker index, rank, and buffer.
    buffer_allocations: HashMap<(usize, u32, BufferId), BufferAllocation>,
    /// Every worker that produces or reads a buffer: the targets of its `Free`.
    buffer_workers: HashMap<BufferId, HashSet<usize>>,
    /// Tensor exports returned by producing workers, merged across the
    /// ranks that publish parts of one product. A `channel` locator carries
    /// the product's bytes, which the engine therefore holds until the buffer
    /// is freed or its request finishes without retaining it.
    transfer_products: HashMap<BufferId, TensorExport>,
    /// Cache exports returned by producing workers.
    cache_transfers: HashMap<BufferId, CacheTransfer>,
    /// Per-worker FIFO of submissions the `WorkerGroup` has not yet accepted.
    worker_submissions: Vec<VecDeque<WorkerSubmission>>,
    /// Last collective sequence each worker accepted. It must strictly
    /// increase per worker: a multi-rank worker treats a batch with calls
    /// whose sequence does not advance as a fatal invariant violation.
    worker_collective_seqs: Vec<u64>,
    /// A command wake was drained; the next `poll` returns `Ok(None)` so the
    /// scheduler reads its commands before waiting again.
    command_wake_pending: bool,
    /// A failure reconciled inside `submit`. `submit` refuses further batches
    /// with `WouldBlock` until `poll` returns it.
    pending_failure: Option<WorkerFailure>,
    closed: bool,
}

/// Returns the spans a destination's ranks read a persistent dependency at.
///
/// A routed destination may reserve its own spans, which take precedence.
/// Without one, a rank takes the span the producer's rank of that index held;
/// the runtime's shared pool places a span at one offset on every rank, so a
/// destination rank the producer's pool did not cover takes the offset the
/// producer's ranks share. Empty when neither side holds a span.
fn dependency_spans(
    own: Vec<RankBuffer>,
    produced: Vec<RankBuffer>,
    ranks: u32,
) -> Vec<RankBuffer> {
    if !own.is_empty() {
        return own;
    }
    let Some(first) = produced.first().copied() else {
        return Vec::new();
    };
    (0..ranks)
        .map(|rank| {
            produced
                .iter()
                .find(|span| span.rank == rank)
                .copied()
                .unwrap_or(RankBuffer { rank, ..first })
        })
        .collect()
}

impl WorkerExecutor {
    /// Binds launched worker groups into one executor.
    ///
    /// Fails when the command wake cannot be created, `workers` is empty, a
    /// worker identity repeats, a group was initialized with transfer
    /// bindings other than `transfer`, the workers expose different outputs for
    /// a replicated component, the media routing or video codec pairing is
    /// refused, a transfer edge names an unbound worker or rank, repeats a
    /// physical rank pair, or uses a mechanism that cannot serve the edge's
    /// endpoints, or no runtime capacity view can be derived from the workers.
    /// As a side effect every group learns the deployment-wide media routing.
    pub fn try_new(
        mut workers: Vec<(WorkerGroupId, WorkerGroup)>,
        transfer: TransferConfig,
    ) -> anyhow::Result<Self> {
        let command_wake = crate::handle::WakeSignal::new()?;
        anyhow::ensure!(
            !workers.is_empty(),
            "WorkerExecutor needs at least one worker"
        );

        let mut routing = HashMap::new();
        for (index, (id, executor)) in workers.iter().enumerate() {
            anyhow::ensure!(
                executor.transfer_config() == &transfer,
                "WorkerGroup {id} was initialized with different transfer bindings"
            );
            anyhow::ensure!(
                routing.insert(id.clone(), index).is_none(),
                "worker identity {id} is repeated"
            );
        }
        let mut executor_info = ExecutorInfo::from_workers(
            workers
                .iter()
                .map(|worker| (worker.0.clone(), worker.1.info().clone()))
                .collect(),
        )?;
        // Each rank reports the buffer bytes it backs.
        executor_info.rank_buffer_bytes = workers
            .iter()
            .map(|(id, worker)| {
                let bytes = (0..)
                    .map_while(|rank| worker.worker_info(rank))
                    .map(|info| info.buffer_pool_bytes)
                    .collect();
                (id.clone(), bytes)
            })
            .collect();
        // Each rank reports its own prefix cache, with its host page pools.
        executor_info.rank_caches = workers
            .iter()
            .map(|(id, worker)| {
                let caches = (0..)
                    .map_while(|rank| worker.worker_info(rank))
                    .map(|info| (info.endpoint.host.clone(), info.prefix_cache.clone()))
                    .collect();
                (id.clone(), caches)
            })
            .collect();
        // Each video denoising rank reports its own host latent pool.
        executor_info.rank_latents = workers
            .iter()
            .map(|(id, worker)| {
                let pools = (0..)
                    .map_while(|rank| worker.worker_info(rank))
                    .map(|info| {
                        (
                            info.endpoint.host.clone(),
                            info.latent_pool.as_deref().cloned(),
                        )
                    })
                    .collect();
                (id.clone(), pools)
            })
            .collect();
        // Each rank executing video encoder calls reports its host's encoder
        // pool.
        executor_info.rank_encoders = workers
            .iter()
            .map(|(id, worker)| {
                let pools = (0..)
                    .map_while(|rank| worker.worker_info(rank))
                    .map(|info| (info.endpoint.host.clone(), info.encoder_pool.clone()))
                    .collect();
                (id.clone(), pools)
            })
            .collect();
        // The video graph may span workers, so each group learns the whole
        // deployment's routing to name the ranks that read its products.
        let media_routing = executor_info.media_routing()?;
        let placements = workers
            .iter()
            .map(|(id, worker)| {
                let (ranks, components) = worker.placement();
                (id, ranks, components)
            })
            .collect::<Vec<_>>();
        executor_info.video_codecs = video_unit_codecs(&media_routing, &placements)?;
        // The scheduler plans component capacity, rounds and its transfer
        // window from what each group was launched with, not from its ranks'
        // reports.
        executor_info.placements = workers
            .iter()
            .map(|(id, worker)| {
                let (ranks, components) = worker.placement();
                let placement = crate::executor::PoolPlacement {
                    queue_depth: worker.queue_depth(),
                    ranks: ranks.len(),
                    components: components.clone(),
                };
                (id.clone(), placement)
            })
            .collect();
        // Every group registers its live incarnations in one table, so a
        // publication's source resolves to its worker and rank whichever
        // group produced it.
        let incarnations = IncarnationRegistry::default();
        for (_, worker) in &mut workers {
            worker.set_media_routing(media_routing.clone());
            worker.share_incarnations(incarnations.clone());
        }

        // Every edge expands to the physical (source rank, destination rank)
        // pairs it names; an omitted rank names all ranks of its worker. A
        // pair may be bound by only one edge.
        let mut physical_edges = HashSet::new();
        for edge in &transfer.edges {
            anyhow::ensure!(
                routing.contains_key(&edge.source_worker)
                    && routing.contains_key(&edge.destination_worker),
                "transfer edge names an unbound WorkerGroup"
            );
            let source = &workers[routing[&edge.source_worker]].1;
            let destination = &workers[routing[&edge.destination_worker]].1;
            for (rank, worker) in [
                (edge.source_rank, source),
                (edge.destination_rank, destination),
            ] {
                anyhow::ensure!(
                    rank.is_none_or(|rank| worker.worker_info(rank as usize).is_some()),
                    "transfer edge names an unbound rank"
                );
            }
            for source_rank in 0..source.world_size() as u32 {
                if edge.source_rank.is_some_and(|rank| rank != source_rank) {
                    continue;
                }
                let source_on_cuda = source.rank_on_cuda(source_rank as usize);
                let source = source.worker_info(source_rank as usize).with_context(|| {
                    format!("transfer edge source has no rank {source_rank} in its world")
                })?;
                for destination_rank in 0..destination.world_size() as u32 {
                    if edge
                        .destination_rank
                        .is_some_and(|rank| rank != destination_rank)
                    {
                        continue;
                    }
                    let destination_on_cuda = destination.rank_on_cuda(destination_rank as usize);
                    let destination = destination
                        .worker_info(destination_rank as usize)
                        .with_context(|| {
                            format!(
                                "transfer edge destination has no rank {destination_rank} in its world"
                            )
                        })?;
                    anyhow::ensure!(
                        physical_edges
                            .insert((source.endpoint.clone(), destination.endpoint.clone())),
                        "physical transfer edge has conflicting bindings"
                    );
                    // An edge carries device products on one mechanism and
                    // host products on another. The launch initialized each
                    // rank with the mechanisms its edges name
                    // (`TransferConfig::rank_backends`).
                    for mechanism in edge.mechanisms() {
                        // Each mechanism decides for itself how far it reaches.
                        // An edge it cannot serve would fail on its first
                        // export, so it is refused here by name instead.
                        let (source_host, destination_host) =
                            (&source.endpoint.host, &destination.endpoint.host);
                        let crosses_hosts = source_host != destination_host;
                        match mechanism {
                            crate::executor::TransferBackend::Local => anyhow::ensure!(
                                source.endpoint.address_space == destination.endpoint.address_space,
                                "local transfer requires a shared address space"
                            ),
                            crate::executor::TransferBackend::CudaVmm => {
                                anyhow::ensure!(
                                    source_on_cuda && destination_on_cuda,
                                    "CUDA VMM requires CUDA devices on both endpoints"
                                );
                                // A device handle crosses hosts only where the
                                // producing device exports a fabric handle and the
                                // consuming device can import one.
                                if crosses_hosts {
                                    let unreachable = if source.fabric_handles {
                                        destination_host
                                    } else {
                                        source_host
                                    };
                                    anyhow::ensure!(
                                        source.fabric_handles && destination.fabric_handles,
                                        "transfer edge from worker {} on host {source_host} to \
                                     worker {} on host {destination_host} crosses hosts, and \
                                     the device on host {unreachable} exports a process \
                                     descriptor the other host cannot import",
                                        edge.source_worker.0,
                                        edge.destination_worker.0,
                                    );
                                }
                            }
                            // Shared storage names a segment in one host's namespace.
                            crate::executor::TransferBackend::PosixShm => anyhow::ensure!(
                                !crosses_hosts,
                                "transfer edge from worker {} on host {source_host} to worker {} \
                             on host {destination_host} crosses hosts over shared storage, \
                             which names a segment in one host's namespace",
                                edge.source_worker.0,
                                edge.destination_worker.0,
                            ),
                            // A product on the rank channel reaches wherever the
                            // channel does, which is every host of the group.
                            crate::executor::TransferBackend::Channel => {}
                        }
                    }
                }
            }
        }
        // The derived runtime view is discarded; the call only refuses
        // workers from which it cannot be derived.
        executor_info.runtime_info()?;

        let depth = workers.iter().map(|worker| worker.1.queue_depth()).sum();
        let worker_count = workers.len();
        Ok(Self {
            routing,
            workers,
            incarnations,
            executor_info,
            depth,
            command_wake,
            pending: BTreeMap::new(),
            ready: VecDeque::new(),
            admissions: HashMap::new(),
            admitted_workers: HashSet::new(),
            buffer_routes: HashMap::new(),
            buffer_allocations: HashMap::new(),
            buffer_workers: HashMap::new(),
            transfer_products: HashMap::new(),
            cache_transfers: HashMap::new(),
            worker_submissions: (0..worker_count).map(|_| VecDeque::new()).collect(),
            worker_collective_seqs: vec![0; worker_count],
            command_wake_pending: false,
            pending_failure: None,
            closed: false,
        })
    }

    /// Refreshes physical discovery after replacement; other group counters remain intact.
    fn refresh_worker(&mut self, index: usize) {
        self.executor_info.workers[index].1 = self.workers[index].1.info().clone();
    }

    /// The command wake descriptor followed by every worker's progress
    /// descriptors, for `park_descriptors`.
    fn progress_fds(&self) -> Vec<libc::pollfd> {
        std::iter::once(libc::pollfd {
            fd: self.command_wake.descriptor(),
            events: libc::POLLIN,
            revents: 0,
        })
        .chain(
            self.workers
                .iter()
                .flat_map(|(_, worker)| worker.progress_fds()),
        )
        .collect()
    }
}

impl WorkerExecutor {
    /// Records the admission descriptor of each request a batch starts.
    ///
    /// A request handed from a prefill group to a decode group is admitted
    /// again with its progress: the position it reached and the tokens it
    /// generated, which the groups that admit it later receive. Fails when a
    /// request is admitted again, before its `Finish` completes, with a
    /// descriptor that differs from the recorded one otherwise; descriptors
    /// recorded earlier in the same call stay recorded.
    fn cache_admissions(&mut self, admissions: &[NewRequest]) -> anyhow::Result<()> {
        for admission in admissions {
            if let Some(existing) = self.admissions.get_mut(&admission.request_key) {
                let progressed = |descriptor: &NewRequest| {
                    let mut descriptor = descriptor.clone();
                    if let Some(generation) = &mut descriptor.generation {
                        generation.initial_position = 0;
                        generation.generated_token_ids.clear();
                    }
                    descriptor
                };
                anyhow::ensure!(
                    progressed(existing) == progressed(admission),
                    "request {:?} was readmitted with a different descriptor before drop",
                    admission.request_key
                );
                existing.clone_from(admission);
            } else {
                self.admissions
                    .insert(admission.request_key, admission.clone());
            }
        }
        Ok(())
    }

    /// Returns the `Start` descriptors `worker_index` needs before `calls`.
    ///
    /// One descriptor per request in `calls` that the worker has not yet
    /// accepted a `Start` for, carrying the placement's worker-local request
    /// row when it names one and the admission's canonical row otherwise.
    /// Fails when a request has no recorded admission or its calls name
    /// different rows.
    fn admissions_for(
        &self,
        worker_index: usize,
        calls: &[(Call, RequestPlacement)],
    ) -> anyhow::Result<Vec<NewRequest>> {
        let mut admissions = Vec::new();
        let mut rows = HashMap::new();
        for (call, placement) in calls {
            if self
                .admitted_workers
                .contains(&(worker_index, call.request_key))
            {
                continue;
            }
            let admission = self.admissions.get(&call.request_key).ok_or_else(|| {
                anyhow::anyhow!(
                    "worker {worker_index} has no admission for request {:?}",
                    call.request_key
                )
            })?;
            let row = placement
                .request_pool_idx
                .unwrap_or(admission.request_pool_idx);
            if let Some(existing) = rows.insert(call.request_key, row) {
                anyhow::ensure!(
                    existing == row,
                    "worker {worker_index} received conflicting request rows for {:?}",
                    call.request_key
                );
                continue;
            }
            let mut admission = admission.clone();
            admission.request_pool_idx = row;
            admissions.push(admission);
        }
        Ok(admissions)
    }

    /// Includes every admitted or queued owner of this request's calls and products.
    fn request_workers(&self, request: RequestKey) -> Vec<usize> {
        let mut owners = self
            .admitted_workers
            .iter()
            .filter_map(|(worker, key)| (*key == request).then_some(*worker))
            .chain(
                self.pending
                    .values()
                    .flat_map(|pending| pending.workers.iter())
                    .filter_map(|(worker_index, worker)| {
                        worker
                            .calls
                            .keys()
                            .any(|(key, _)| *key == request)
                            .then_some(*worker_index)
                    }),
            )
            .chain(
                self.buffer_workers
                    .iter()
                    .filter(|(buffer, _)| buffer.owner == request)
                    .flat_map(|(_, workers)| workers.iter().copied()),
            )
            .collect::<Vec<_>>();
        owners.sort_unstable();
        owners.dedup();
        owners
    }

    /// Routes buffer release to its storage owners and request closure to every physical owner.
    ///
    /// Returns sorted worker indices. `Start` has no target of its own: a
    /// worker receives it with the request's first call (`admissions_for`).
    fn command_workers(&self, command: &BatchCommand) -> Vec<usize> {
        match command {
            BatchCommand::Start { .. } => Vec::new(),
            BatchCommand::Free { buffer } => {
                let mut owners = self
                    .buffer_workers
                    .get(buffer)
                    .map(|workers| workers.iter().copied().collect::<Vec<_>>())
                    .unwrap_or_default();
                owners.sort_unstable();
                owners
            }
            BatchCommand::Finish { request_key, .. } => self.request_workers(*request_key),
        }
    }

    /// Keep resident references direct when producer and consumer execute in
    /// the same address spaces. Different multi-rank components require published
    /// layouts, including when both components belong to this WorkerGroup.
    ///
    /// The caller has established that both run on the producer's worker.
    /// Returns true for the producing component itself when each of its ranks
    /// holds its own copy of the buffer, or for two components on the same
    /// single rank; false otherwise, including when either component of a
    /// pair is not loaded on the worker.
    fn shares_product_storage(&self, producer: &BufferRoute, consumer_entry: &str) -> bool {
        let (_, entries) = self.workers[producer.worker_index].1.placement();
        let source = entries.get(&producer.component);
        let destination = entries.get(consumer_entry);
        if producer.component == consumer_entry {
            // A consumer call on the component that produced the product runs on
            // the same ranks, and each rank reads the copy it holds when every
            // rank holds one. Every rank holds a call's own results beside its
            // declared outputs. A declared output is written by every rank of a
            // distributed component, whose round deals the same media units in
            // the same order to the same ranks, and of a sequence-parallel one,
            // whose ranks each write their shard; a tensor-parallel or pipelined
            // component publishes it from its output ranks alone, so its other
            // ranks read the product's export. Without a description of
            // the component, its ranks are taken to hold their own copies.
            return !producer.declared_output
                || source.is_none_or(|source| source.publishes_on_every_rank());
        }
        source
            .zip(destination)
            .is_some_and(|(source, destination)| {
                source.ranks.len() == 1 && source.ranks == destination.ranks
            })
    }

    /// Submits or queues one worker-local batch while preserving collective order.
    ///
    /// Prepends the `Start` of every request among its calls that the worker
    /// has not yet admitted, attaches the exports of the submission's
    /// dependencies, which the caller has checked are published, and lowers
    /// the batch to the wire. Returns `Deferred` when the group refuses it
    /// with `WouldBlock` (not ready, or no free slot), and `Refused` when the
    /// group cannot bind inputs to the ranks reading them; in both cases
    /// nothing is recorded and the caller requeues the submission.
    /// `Submitted` records the submission, its collective sequence, and the
    /// admissions it carried. Errors come from resolving admissions, from
    /// building or validating the wire batch, or from the group; a group that
    /// replaced its ranks reports a `WorkerFailure`.
    fn submit_worker(
        &mut self,
        worker_index: usize,
        submission: &WorkerSubmission,
    ) -> anyhow::Result<Dispatch> {
        let batch = &submission.batch;
        let admissions = self.admissions_for(worker_index, &batch.requests)?;
        let request_keys = admissions
            .iter()
            .map(|admission| admission.request_key)
            .collect::<Vec<_>>();
        let commands = admissions
            .into_iter()
            .map(|request| BatchCommand::Start {
                request: Box::new(request),
            })
            .chain(batch.commands.iter().cloned())
            .collect();
        let mut inputs = batch.input_transfers.clone();
        let mut cache_inputs = batch.cache_inputs.clone();
        for dependency in &submission.dependencies {
            if let Some(export) = self.cache_transfers.get(dependency) {
                anyhow::ensure!(
                    !cache_inputs.iter().any(|input| input.source == *dependency),
                    "cache input is supplied more than once"
                );
                cache_inputs.push(export.clone());
                continue;
            }
            anyhow::ensure!(
                !inputs
                    .iter()
                    .any(|payload| payload.product.buffer_id() == *dependency),
                "transferred input payload is supplied more than once"
            );
            let payload = self.transfer_products.get(dependency).ok_or_else(|| {
                anyhow::anyhow!(
                    "ready transferred input {:?} lost its transfer descriptor",
                    dependency
                )
            })?;
            inputs.push(payload.clone());
        }

        // The sequence is committed only once the group accepts the batch, so
        // a `WouldBlock` retry carries the same value.
        let collective_seq = self.worker_collective_seqs[worker_index]
            .checked_add(1)
            .context("collective sequence space exhausted")?;
        let (mut wire, mut rank_buffers) = ExecutionBatch {
            id: batch.id,
            requests: batch.requests.clone(),
            commands,
            input_transfers: inputs,
            cache_inputs,
            page_offloads: batch.page_offloads.clone(),
        }
        .into_protocol(collective_seq)?;

        // A transferred persistent input needs its destination-local span.
        for dependency in &submission.dependencies {
            if !self.transfer_products.contains_key(dependency)
                || !wire
                    .calls
                    .iter()
                    .flat_map(|call| call.buffer_inputs())
                    .any(|tensor| tensor.buffer_id() == *dependency)
            {
                continue;
            }
            // Spans are per rank, so each rank of the destination takes its
            // own (`dependency_spans`).
            let held = |worker: usize| {
                let mut spans = self
                    .buffer_allocations
                    .iter()
                    .filter(|((owner, _, buffer), _)| *owner == worker && buffer == dependency)
                    .map(|((_, rank, _), allocation)| RankBuffer {
                        rank: *rank,
                        allocation: *allocation,
                    })
                    .collect::<Vec<_>>();
                spans.sort_unstable_by_key(|span| span.rank);
                spans
            };
            let produced = self
                .buffer_routes
                .get(dependency)
                .map(|route| held(route.worker_index))
                .unwrap_or_default();
            let ranks = (0..)
                .map_while(|rank| self.workers[worker_index].1.worker_info(rank))
                .count();
            let spans = dependency_spans(held(worker_index), produced, ranks as u32);
            anyhow::ensure!(
                !spans.is_empty(),
                "persistent dependency {:?} has no scheduler params",
                dependency
            );
            for span in spans {
                if !rank_buffers.iter().any(|existing| {
                    existing.rank == span.rank
                        && existing.allocation.buffer == span.allocation.buffer
                }) {
                    rank_buffers.push(span);
                }
            }
        }

        // Drop every location whose source endpoint is not a live
        // incarnation of a bound rank.
        let tensors = wire
            .input_products
            .iter_mut()
            .map(|payload| &mut payload.value)
            .flat_map(|handle| handle.tensors_mut())
            .chain(
                wire.cache_inputs
                    .iter_mut()
                    .flat_map(|export| export.tensors_mut()),
            );
        for tensor in tensors {
            tensor
                .locations
                .retain(|location| self.incarnations.position(&location.source).is_some());
        }

        // Routing stays on the engine: the group names each call's reading
        // ranks from the workers its request was routed to.
        let readers = batch
            .requests
            .iter()
            .filter_map(|(call, placement)| {
                let readers = placement.readers.clone()?;
                Some(((call.request_key, call.call_id), readers))
            })
            .collect();
        match self.workers[worker_index]
            .1
            .submit_batch(wire, &readers, &rank_buffers)
        {
            Ok(()) => {}
            Err(BatchSubmitError::WouldBlock(_)) => return Ok(Dispatch::Deferred),
            Err(BatchSubmitError::Failed(error)) => {
                // Only an input refusal leaves the group intact; every other
                // failure propagates unchanged.
                let refusal = error.downcast::<UnroutableInputs>()?;
                let failure = self.refused_inputs(worker_index, batch, refusal)?;
                return Ok(Dispatch::Refused(Box::new(failure)));
            }
        }
        self.worker_collective_seqs[worker_index] = collective_seq;
        let pending = self
            .pending
            .get_mut(&batch.id)
            .context("submitted batch is not pending")?;
        let worker = pending
            .workers
            .get_mut(&worker_index)
            .context("submitted worker is not pending")?;
        anyhow::ensure!(!worker.submitted, "worker batch was submitted twice");
        worker.submitted = true;
        self.admitted_workers
            .extend(request_keys.into_iter().map(|key| (worker_index, key)));
        Ok(Dispatch::Submitted)
    }

    /// The failure of the requests whose calls read inputs a worker refused.
    ///
    /// A product with no location on an edge to its consumer is a property of
    /// that product and that edge, such as a device product that fell back to
    /// host bytes crossing an edge that carries only a device mechanism. It
    /// leaves both groups and the batch's other calls intact, so the
    /// failure invalidates no endpoint or buffer and names only the requests
    /// reading the refused products. Reconciliation retires those requests'
    /// calls from the requeued submission, whose remaining calls and
    /// commands dispatch later.
    ///
    /// # Errors
    ///
    /// Fails when no call of `batch` reads a refused product: nothing could
    /// then be retired, and the requeued submission would be refused again.
    fn refused_inputs(
        &self,
        worker_index: usize,
        batch: &ExecutionBatch,
        refusal: UnroutableInputs,
    ) -> anyhow::Result<WorkerFailure> {
        let requests = batch
            .requests
            .iter()
            .filter(|(call, _)| {
                call.input_buffers()
                    .any(|buffer| refusal.buffers.contains(&buffer))
            })
            .map(|(call, _)| call.request_key)
            .collect::<HashSet<_>>();
        anyhow::ensure!(
            !requests.is_empty(),
            "refused input products are read by no call: {refusal}"
        );
        Ok(WorkerFailure {
            worker_id: self.workers[worker_index].0.clone(),
            endpoints: Vec::new(),
            requests: requests.into_iter().collect(),
            refusals: HashMap::new(),
            retired: Vec::new(),
            buffers: Vec::new(),
            execution: None,
            message: refusal.to_string(),
        })
    }

    /// Dispatches each worker's queued batches in order, as far as they are ready.
    ///
    /// Batch ids are dispatched in order, so a rank sees a strictly increasing
    /// subsequence and cooperative ranks launch their collectives in the same
    /// order. A queue therefore holds at its head: a submission whose product
    /// dependencies are not yet published cannot have their locators bound, and
    /// dispatching the submission behind it would hand the rank a batch out of
    /// order, which a rank that launches strictly in channel order refuses.
    fn dispatch_ready(&mut self) -> anyhow::Result<()> {
        loop {
            let mut progressed = false;
            for worker_index in 0..self.workers.len() {
                let Some(head) = self.worker_submissions[worker_index].front() else {
                    continue;
                };
                let ready = head.dependencies.iter().all(|buffer| {
                    self.transfer_products.contains_key(buffer)
                        || self.cache_transfers.contains_key(buffer)
                });
                if !ready {
                    continue;
                }
                let submission = self.worker_submissions[worker_index]
                    .pop_front()
                    .ok_or_else(|| anyhow::anyhow!("ready worker submission disappeared"))?;
                match self.submit_worker(worker_index, &submission)? {
                    Dispatch::Submitted => progressed = true,
                    Dispatch::Deferred => {
                        self.worker_submissions[worker_index].push_front(submission);
                    }
                    Dispatch::Refused(failure) => {
                        // The refused submission keeps its place, so the
                        // reconciliation of this failure retires the named
                        // requests' calls from it.
                        self.worker_submissions[worker_index].push_front(submission);
                        return Err((*failure).into());
                    }
                }
            }
            if !progressed {
                return Ok(());
            }
        }
    }

    /// Advances worker command execution until no immediate progress remains.
    ///
    /// Latches a command wake, dispatches ready submissions, drains every
    /// worker's available reports without blocking, and dispatches again for
    /// the dependencies those reports published.
    fn pump(&mut self) -> anyhow::Result<()> {
        self.command_wake_pending |= self.command_wake.drain()?;
        self.dispatch_ready()?;
        for worker_index in 0..self.workers.len() {
            loop {
                let polled = self.workers[worker_index].1.poll_batch(Duration::ZERO);
                // Taken before the poll error propagates, so a recovery or
                // close made while polling reaches `executor_info` either way.
                if self.workers[worker_index].1.take_readiness_change() {
                    self.refresh_worker(worker_index);
                }
                let report = polled?;
                let Some(report) = report else {
                    break;
                };
                self.route_result(worker_index, report)?;
            }
        }
        self.dispatch_ready()
    }
}

impl WorkerExecutor {
    /// Returns the command-channel waker.
    pub fn command_waker(&self) -> CommandWaker {
        self.command_wake.waker()
    }
}

impl Executor for WorkerExecutor {
    fn has_capacity(&self, worker: &WorkerGroupId) -> bool {
        // Submissions already queued for the worker claim its free group slots.
        self.routing.get(worker).is_some_and(|&index| {
            self.workers[index].1.available_slots() > self.worker_submissions[index].len()
        })
    }

    fn command_has_capacity(&self, command: &BatchCommand) -> bool {
        self.command_workers(command)
            .into_iter()
            .all(|index| self.has_capacity(&self.workers[index].0))
    }

    fn is_ready(&self, worker: &WorkerGroupId) -> bool {
        self.routing
            .get(worker)
            .is_some_and(|&index| self.workers[index].1.is_ready())
    }

    fn is_closed(&self, worker: &WorkerGroupId) -> bool {
        self.routing
            .get(worker)
            .is_none_or(|&index| self.workers[index].1.is_closed())
    }

    /// Returns the worker metadata.
    fn info(&self) -> &ExecutorInfo {
        &self.executor_info
    }

    /// Partitions a logical batch across owning workers and registers aggregate completion state.
    ///
    /// Returns `WouldBlock` while a reconciled failure awaits `poll`, while
    /// `depth` logical batches are pending, or when a target worker has no
    /// free slot. A `WorkerFailure` raised while dispatching is reconciled and
    /// held for the next `poll`, and the batch counts as submitted: its fate
    /// arrives through that failure and the batch's results. A reconciliation
    /// error is returned as `Failed`. Any other error while partitioning or
    /// dispatching drops the batch's pending record, but state registered
    /// before the error, such as the admissions it carries, the routes,
    /// holders, and allocations of its calls, and submissions already queued,
    /// is not rolled back.
    fn submit(&mut self, mut batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        if self.closed {
            return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                "Executor is closed"
            )));
        }
        if self.pending_failure.is_some() {
            return Err(ExecutorSubmitError::WouldBlock(Box::new(batch)));
        }
        if self.pending.len() >= self.depth {
            return Err(ExecutorSubmitError::WouldBlock(Box::new(batch)));
        }
        batch.validate().map_err(ExecutorSubmitError::Failed)?;
        let mut targets = HashSet::new();
        for (_, placement) in &batch.requests {
            let Some(&index) = self.routing.get(&placement.worker) else {
                return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                    "unknown target worker {}",
                    placement.worker
                )));
            };
            targets.insert(index);
        }
        for command in &batch.commands {
            targets.extend(self.command_workers(command));
        }
        if targets
            .iter()
            .any(|&index| !self.has_capacity(&self.workers[index].0))
        {
            return Err(ExecutorSubmitError::WouldBlock(Box::new(batch)));
        }
        let batch_id = batch.id;
        if self.pending.contains_key(&batch_id) {
            return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                "worker router already has batch {batch_id} in flight"
            )));
        }
        let admissions = batch.admissions().cloned().collect::<Vec<_>>();
        self.cache_admissions(&admissions)
            .map_err(ExecutorSubmitError::Failed)?;

        let submit_result = (|| -> anyhow::Result<()> {
            // Each per-worker vector is indexed by worker index.
            let mut worker_ops = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<(Call, RequestPlacement)>>>();
            let mut worker_inputs = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<TensorExport>>>();
            let mut worker_cache_inputs: Vec<Vec<CacheTransfer>> =
                (0..self.workers.len()).map(|_| Vec::new()).collect();
            let mut worker_dependencies = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<BufferId>>>();
            let mut call_routes = HashMap::with_capacity(batch.requests.len());
            let mut input_routes = HashMap::new();

            // Route each call to its worker, check the worker can run it, and
            // register the producer and holders of its outputs and the
            // worker-local buffer allocations its placement names.
            for (call, placement) in &batch.requests {
                let variant = call.kind;
                let call_worker =
                    self.routing
                        .get(&placement.worker)
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!("call targets unknown worker {}", placement.worker)
                        })?;
                anyhow::ensure!(
                    self.workers[call_worker]
                        .1
                        .info()
                        .supported_calls
                        .contains(&variant),
                    "target worker does not support call {variant:?}"
                );
                let (_, entries) = self.workers[call_worker].1.placement();
                anyhow::ensure!(
                    entries.is_empty() || entries.contains_key(&call.component),
                    "call targets unloaded component {}",
                    call.component
                );
                let declared = call
                    .outputs
                    .iter()
                    .map(TensorRef::buffer_id)
                    .collect::<HashSet<_>>();
                for output in call.output_buffers() {
                    let route = BufferRoute {
                        worker_index: call_worker,
                        component: call.component.clone(),
                        declared_output: declared.contains(&output),
                    };
                    if let Some(existing) = self.buffer_routes.get(&output) {
                        anyhow::ensure!(
                            existing.worker_index == route.worker_index
                                && existing.component == route.component,
                            "product identity was routed to conflicting workers"
                        );
                    } else {
                        self.buffer_routes.insert(output, route);
                    }
                    // Every physical product needs release routing, including
                    // paged latents, prefix cache, and request-relay values that have no
                    // buffer allocation.
                    self.buffer_workers
                        .entry(output)
                        .or_default()
                        .insert(call_worker);
                }
                for params in &placement.buffers {
                    if let Some(existing) = self.buffer_allocations.insert(
                        (call_worker, params.rank, params.allocation.buffer),
                        params.allocation,
                    ) {
                        anyhow::ensure!(
                            existing == params.allocation,
                            "buffer identity was assigned conflicting rank-local allocations"
                        );
                    }
                }
                for output in call.buffer_outputs() {
                    anyhow::ensure!(
                        placement
                            .buffers
                            .iter()
                            .any(|params| params.allocation.buffer == output.buffer_id()),
                        "persistent product {:?} has no scheduler params",
                        output
                    );
                }
                call_routes.insert((call.request_key, call.call_id), call_worker);
                for input in call.input_buffers() {
                    input_routes
                        .entry(input)
                        .or_insert_with(HashSet::new)
                        .insert(call_worker);
                }
            }

            // External exports go to every worker with a call reading them.
            for payload in &batch.input_transfers {
                let worker_indices =
                    input_routes
                        .get(&payload.product.buffer_id())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "WorkerExecutor cannot route undeclared input product {:?}",
                                payload.product
                            )
                        })?;
                for &worker_index in worker_indices {
                    worker_inputs[worker_index].push(payload.clone());
                }
            }
            for export in &batch.cache_inputs {
                let workers = input_routes
                    .get(&export.source)
                    .context("cannot route undeclared cache input")?;
                for &worker_index in workers {
                    worker_cache_inputs[worker_index].push(export.clone());
                }
            }

            // Resolve every input: supplied externally, read in place from a
            // producer sharing the reader's storage, or a dependency the
            // worker's submission waits on until its producer publishes it.
            for (call, _) in &batch.requests {
                let consumer_worker = call_routes[&(call.request_key, call.call_id)];
                for input in call.input_buffers() {
                    self.buffer_workers
                        .entry(input)
                        .or_default()
                        .insert(consumer_worker);
                    let Some(producer) = self.buffer_routes.get(&input) else {
                        anyhow::ensure!(
                            worker_inputs[consumer_worker]
                                .iter()
                                .any(|payload| payload.product.buffer_id() == input)
                                || worker_cache_inputs[consumer_worker]
                                    .iter()
                                    .any(|export| export.source == input),
                            "input {:?} has no registered producer or supplied export",
                            input
                        );
                        continue;
                    };
                    if producer.worker_index == consumer_worker
                        && self.shares_product_storage(producer, &call.component)
                    {
                        continue;
                    }
                    anyhow::ensure!(
                        !worker_inputs[consumer_worker]
                            .iter()
                            .any(|existing| existing.product.buffer_id() == input)
                            && !worker_cache_inputs[consumer_worker]
                                .iter()
                                .any(|export| export.source == input),
                        "transfer descriptor must be supplied by its producing worker"
                    );
                    if !worker_dependencies[consumer_worker].contains(&input) {
                        worker_dependencies[consumer_worker].push(input);
                    }
                }
            }

            // A batch's page offloads belong to the one worker group that
            // runs all of its calls (`ExecutionBatch::page_offloads`).
            let mut page_offloads = (0..self.workers.len())
                .map(|_| None)
                .collect::<Vec<Option<crate::executor::PageOffloads>>>();
            if let Some(writes) = batch.page_offloads.take() {
                let worker = *self
                    .routing
                    .get(&writes.worker)
                    .context("page offloads name an unknown worker group")?;
                anyhow::ensure!(
                    batch
                        .requests
                        .iter()
                        .all(|(call, _)| call_routes[&(call.request_key, call.call_id)] == worker),
                    "a batch carrying page offloads spans several workers"
                );
                page_offloads[worker] = Some(writes);
            }

            for (call, placement) in batch.requests {
                let worker = call_routes[&(call.request_key, call.call_id)];
                worker_ops[worker].push((call, placement));
            }

            // `Start` reaches a worker with the request's first call there;
            // other commands go to every worker holding what they release.
            let mut worker_commands = (0..self.workers.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<BatchCommand>>>();
            for command in &batch.commands {
                if matches!(command, BatchCommand::Start { .. }) {
                    continue;
                }
                let targets = self.command_workers(command);
                // A rejected admission can retire before any endpoint has seen it.
                // Free is idempotent once every registered location has acknowledged
                // physical retirement; live locations stay indexed until that point.
                if targets.is_empty()
                    && matches!(
                        command,
                        BatchCommand::Finish { .. } | BatchCommand::Free { .. }
                    )
                {
                    continue;
                }

                anyhow::ensure!(
                    !targets.is_empty(),
                    "worker router has no destination for command {command:?}"
                );
                for worker_index in targets {
                    worker_commands[worker_index].push(command.clone());
                }
            }

            // Queue one submission per worker that has calls, commands or
            // page offloads, and register the batch before dispatching any of
            // them.
            let mut pending_workers = HashMap::new();
            for (worker_index, (((calls, commands), input_products), dependencies)) in worker_ops
                .into_iter()
                .zip(worker_commands)
                .zip(worker_inputs)
                .zip(worker_dependencies)
                .enumerate()
            {
                if calls.is_empty() && commands.is_empty() && page_offloads[worker_index].is_none()
                {
                    anyhow::ensure!(
                        dependencies.is_empty(),
                        "product dependency has no consuming worker submission"
                    );
                    continue;
                }
                pending_workers.insert(
                    worker_index,
                    PendingGroup {
                        submitted: false,
                        calls: calls
                            .iter()
                            .map(|(call, _)| ((call.request_key, call.call_id), call.kind))
                            .collect(),
                    },
                );
                // Keep call-owned resources together while dependencies wait.
                // Rank indices and collective order are derived at physical submission.
                self.worker_submissions[worker_index].push_back(WorkerSubmission::build(
                    batch_id,
                    calls,
                    commands,
                    input_products,
                    std::mem::take(&mut worker_cache_inputs[worker_index]),
                    std::mem::take(&mut page_offloads[worker_index]),
                    dependencies,
                ));
            }
            self.pending.insert(
                batch_id,
                PendingBatch {
                    workers: pending_workers,
                    commands: batch
                        .commands
                        .iter()
                        .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                        .cloned()
                        .collect(),
                    command_outcomes: HashMap::new(),
                },
            );
            self.dispatch_ready()?;

            // A batch with no worker part, such as one whose commands have no
            // remaining holder, completes at once.
            if self
                .pending
                .get(&batch_id)
                .is_some_and(|pending| pending.workers.is_empty())
            {
                self.publish_result(WorkerResult {
                    batch_id,
                    done: true,
                    results: Vec::new(),
                    products: Vec::new(),
                    worker_exec_us: None,
                    forward_stats: None,
                })?;
            }

            // A freed buffer loses its producer route and exports now, so
            // a later call that reads it is refused unless its batch supplies
            // an export. Its holders and allocations stay indexed until the
            // `Free` completes, and remain if the release fails.
            for command in &batch.commands {
                if let BatchCommand::Free { buffer } = command {
                    self.transfer_products
                        .retain(|identity, _| *identity != *buffer);
                    self.buffer_routes
                        .retain(|identity, _| *identity != *buffer);
                    self.cache_transfers.remove(buffer);
                }
            }
            Ok(())
        })();
        match submit_result {
            Ok(()) => Ok(()),
            Err(error) if error.is::<WorkerFailure>() => {
                self.pending_failure = Some(
                    self.reconcile_worker_failure(error)
                        .map_err(ExecutorSubmitError::Failed)?,
                );
                Ok(())
            }
            Err(error) => {
                self.pending.remove(&batch_id);
                Err(ExecutorSubmitError::Failed(error))
            }
        }
    }

    /// Drives worker progress and returns the next completed logical batch.
    ///
    /// A failure held by `submit` is returned first. A drained command wake
    /// makes the call return `Ok(None)` at once, before or after waiting, so
    /// the scheduler can read its commands. A worker failure found while
    /// pumping is reconciled and returned as a `WorkerFailure`; any other
    /// error is returned unchanged.
    fn poll(&mut self, timeout: Duration) -> anyhow::Result<Option<BatchResult>> {
        if let Some(loss) = self.pending_failure.take() {
            return Err(loss.into());
        }
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        if let Err(error) = self.pump() {
            return Err(self.reconcile_worker_failure(error)?.into());
        }
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        if let Some(result) = self.ready.pop_front() {
            return Ok(Some(result));
        }
        if timeout.is_zero() {
            return Ok(None);
        }
        crate::worker::park_descriptors(&self.progress_fds(), timeout)?;
        if let Err(error) = self.pump() {
            return Err(self.reconcile_worker_failure(error)?.into());
        }
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        Ok(self.ready.pop_front())
    }

    /// Closes every worker group and refuses later submissions.
    ///
    /// Groups can share expert communicators, so each must receive shutdown
    /// before another waits for collective retirement. Every group is joined
    /// even when one fails; the first error is returned.
    fn close(&mut self) -> anyhow::Result<()> {
        self.closed = true;
        std::thread::scope(|scope| {
            let closing = self
                .workers
                .iter_mut()
                .map(|worker| scope.spawn(|| worker.1.close()))
                .collect::<Vec<_>>();
            let mut first_error = None;
            for handle in closing {
                let result = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("worker shutdown thread panicked"))
                    .and_then(|result| result);
                if let Err(error) = result
                    && first_error.is_none()
                {
                    first_error = Some(error);
                }
            }
            first_error.map_or(Ok(()), Err)
        })
    }

    /// Runs the worker hook on every group, the groups together
    /// (`WorkerGroup::hook`), and returns every rank's result, group by
    /// group.
    fn hook(&mut self, name: &str, payload: &str) -> anyhow::Result<Vec<String>> {
        std::thread::scope(|scope| {
            let running = self
                .workers
                .iter_mut()
                .map(|worker| scope.spawn(|| worker.1.hook(name, payload)))
                .collect::<Vec<_>>();
            let mut results = Vec::new();
            for handle in running {
                results.extend(
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("worker hook thread panicked"))??,
                );
            }
            Ok(results)
        })
    }

    fn publish_world(&mut self) -> anyhow::Result<()> {
        match self.workers.first() {
            Some((_, group)) => group.claims().publish(),
            None => Ok(()),
        }
    }

    fn commit_experts(
        &mut self,
        order: &BTreeMap<String, Vec<u32>>,
    ) -> anyhow::Result<Vec<String>> {
        std::thread::scope(|scope| {
            let running = self
                .workers
                .iter_mut()
                .map(|(_, group)| scope.spawn(|| group.commit_experts(order)))
                .collect::<Vec<_>>();
            let mut results = Vec::new();
            for handle in running {
                results.extend(
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("expert commit thread panicked"))??,
                );
            }
            Ok(results)
        })
    }

    /// Retires `worlds` through the deployment's claims, which every group
    /// shares.
    fn retire(&mut self, worlds: &[u64]) -> anyhow::Result<()> {
        match self.workers.first() {
            Some((_, group)) => group.claims().retire(worlds),
            None => Ok(()),
        }
    }

    /// Retains weights and completed rollback cache through the deployment's
    /// claims, which every group shares.
    fn keep_weights(
        &mut self,
        worlds: &[u64],
        to_host: bool,
        cache_keys: &[Vec<u8>],
    ) -> anyhow::Result<Vec<String>> {
        match self.workers.first() {
            Some((_, group)) => group.claims().keep_weights(worlds, to_host, cache_keys),
            None => Ok(Vec::new()),
        }
    }

    /// Throttles the deployment's transfers on its nodes through the claims
    /// every group shares.
    fn throttle(&mut self, halve: bool) -> anyhow::Result<Vec<String>> {
        match self.workers.first() {
            Some((_, group)) => group.claims().throttle(halve),
            None => Ok(Vec::new()),
        }
    }

    /// Reads the memory ledgers of the deployment's nodes through the
    /// claims every group shares.
    fn ledgers(&mut self) -> anyhow::Result<Vec<String>> {
        let Some((_, group)) = self.workers.first() else {
            return Ok(Vec::new());
        };
        group
            .claims()
            .ledgers()?
            .into_iter()
            .map(|(host, ledger)| {
                let mut read: serde_json::Value = serde_json::from_str(&ledger)?;
                read["host"] = serde_json::Value::String(host);
                Ok(read.to_string())
            })
            .collect()
    }

    fn storage_nodes(&self) -> BTreeMap<String, String> {
        self.workers
            .first()
            .map(|(_, group)| group.claims().endpoints().clone())
            .unwrap_or_default()
    }

    /// Runs the worker hook on every group, the groups together, and
    /// returns each group's identity with its ranks' results.
    fn hook_groups(
        &mut self,
        name: &str,
        payload: &str,
    ) -> anyhow::Result<Vec<(WorkerGroupId, Vec<String>)>> {
        std::thread::scope(|scope| {
            let running = self
                .workers
                .iter_mut()
                .map(|worker| {
                    let id = worker.0.clone();
                    scope.spawn(move || worker.1.hook(name, payload).map(|results| (id, results)))
                })
                .collect::<Vec<_>>();
            running
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("worker hook thread panicked"))?
                })
                .collect()
        })
    }

    /// Commits every group's world together (`WorkerGroup::commit`), and
    /// returns every rank's report, group by group.
    fn commit(&mut self) -> anyhow::Result<Vec<String>> {
        std::thread::scope(|scope| {
            let committing = self
                .workers
                .iter_mut()
                .map(|worker| scope.spawn(|| worker.1.commit()))
                .collect::<Vec<_>>();
            let mut results = Vec::new();
            for handle in committing {
                results.extend(
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("worker commit thread panicked"))??,
                );
            }
            Ok(results)
        })
    }
}

#[cfg(test)]
mod dependency_span_tests {
    use super::dependency_spans;
    use crate::executor::RankBuffer;
    use ipc::RequestId;
    use ipc::{BufferAllocation, BufferId, CallId, RequestKey};

    fn span(rank: u32, offset: u64) -> RankBuffer {
        RankBuffer {
            rank,
            allocation: BufferAllocation {
                buffer: BufferId {
                    owner: RequestKey::new(1, RequestId(1), 1),
                    producer_call_id: CallId::new(1, 0),
                    output_index: 0,
                    generation: 1,
                },
                offset,
                bytes: 256,
            },
        }
    }

    #[test]
    fn a_destination_that_reserved_its_own_spans_reads_them() {
        let own = vec![span(0, 512), span(1, 1024)];
        let produced = vec![span(0, 0)];

        assert_eq!(dependency_spans(own.clone(), produced, 2), own);
    }

    #[test]
    fn a_destination_rank_the_producers_pool_did_not_cover_takes_the_shared_offset() {
        // The producer ran on one rank; the destination has three.
        let spans = dependency_spans(Vec::new(), vec![span(0, 768)], 3);

        assert_eq!(spans, vec![span(0, 768), span(1, 768), span(2, 768)]);
    }

    #[test]
    fn a_destination_rank_the_producer_covered_takes_that_ranks_span() {
        let produced = vec![span(0, 0), span(1, 256)];

        assert_eq!(
            dependency_spans(Vec::new(), produced, 3),
            vec![span(0, 0), span(1, 256), span(2, 0)]
        );
    }

    #[test]
    fn a_dependency_no_pool_placed_has_no_span() {
        assert!(dependency_spans(Vec::new(), Vec::new(), 2).is_empty());
    }
}

//! Logical execution batches, physical execution, and worker contracts.
//!
//! A serving scheduler submits its batches through the worker crate's
//! `Batches` (`Submissions::over_batches`). [`Worker`] is the same boundary
//! for a simulated pool: `SimWorker` (`crate::sim`) is its implementation,
//! and a scheduler test plugs it in through `SchedulerHandle::with_worker`.
//!
//! Besides the trait, this module owns the values that cross that boundary:
//! [`WorkerInfo`], which merges the pools' reported capabilities, beside
//! their layout, into the single capacity view the scheduler plans against;
//! the result types a poll returns; and [`ConnectorConfig`], the directed
//! transfer edges a deployment names between its worker groups, which each
//! replica's engine configuration keeps for its own groups (the daemon's
//! `replica_configs`).

pub use proto::config::{BlockConfig, BlockDistribution, SequenceParallel, WorkerGroupId};

#[cfg(test)]
use ipc::VideoDenoiserInfo;
use ipc::{ForwardMode, MediaCall};
use std::collections::{BTreeMap, BTreeSet, HashMap};
#[cfg(feature = "testing")]
use std::sync::Arc;
#[cfg(feature = "testing")]
use std::time::Duration;

use anyhow::Context as _;

#[cfg(feature = "testing")]
use ipc::BatchCommand;
use ipc::{CallId, CallKind, MainProcessInfo, RequestKey};

mod batch;
mod codecs;
pub(crate) mod inventory;

pub use batch::{ExecutionBatch, PageOffloads, RankBuffer, RequestDispatch};
use ipc::execution::PoolLayout;

/// Concrete pool information reported through the execution boundary.
#[derive(Debug, Clone, Default)]
pub struct WorkerInfo {
    /// Physical pool identities and their reported capabilities.
    pub workers: Vec<(WorkerGroupId, MainProcessInfo)>,
    /// The longest sequence the model was built for: the attention
    /// descriptor's static bound on the keys one row reads. Zero when not
    /// stated.
    pub max_model_len: u32,
    /// For each pool, how it is placed. A pool absent from the map is placed
    /// with no components and a queue depth of one.
    ///
    /// `from_inventory` and `SimWorker` fill this; `from_workers` leaves it
    /// empty.
    pub layouts: BTreeMap<WorkerGroupId, PoolLayout>,
    /// For each worker that decodes video, the encoder replicas whose rank
    /// dealing keeps every media unit on the host that decoded it. A decoder
    /// absent from the map places no host constraint on its encoder.
    ///
    /// `from_inventory` fills this from the layout; `from_workers` and
    /// `SimWorker` leave it empty. Scheduler admission restricts a video
    /// encoding call's candidates to the set of the worker chosen to decode.
    pub video_codecs: BTreeMap<WorkerGroupId, BTreeSet<WorkerGroupId>>,
    /// For each pool, the persistent buffer bytes each of its ranks backs,
    /// in rank order, as the ranks reported them. Each rank has its own
    /// buffer pool, sized to the products of the components it executes and
    /// of the components that read them, so the engine places a pool's
    /// media buffers rank by rank.
    ///
    /// `from_inventory` fills this from its ranks; a pool absent from it, as
    /// `from_workers` and `SimWorker` leave every pool, has its placed ranks
    /// (`PoolLayout::ranks`, one when it has no layout), each backing
    /// its reported `buffer_pool_bytes`.
    pub rank_buffer_bytes: BTreeMap<WorkerGroupId, Vec<u64>>,
    /// For each pool, each rank's prefix cache as the rank reported it, with
    /// the host page pools it maps, and the rank's host, in rank order.
    ///
    /// `from_inventory` fills this from its ranks; `from_workers` and
    /// `SimWorker` leave it empty, which serves no pool through the host KV
    /// cache.
    pub rank_caches: BTreeMap<WorkerGroupId, Vec<(String, Option<ipc::PrefixCacheInfo>)>>,
    /// For each pool, each rank's host latent pool as the rank reported it,
    /// and the rank's host, in rank order: the video denoising ranks report
    /// one, every other rank none (`ipc::MainProcessInfo::latent_pool`).
    ///
    /// `from_inventory` fills this from its ranks; `from_workers` leaves it
    /// empty, which stores no preempted trajectory, and a
    /// simulated worker states its own (`SimEngine::set_latent_pools`).
    pub rank_latents: BTreeMap<WorkerGroupId, Vec<(String, Option<ipc::LatentPoolInfo>)>>,
    /// For each pool, each rank's host encoder pool as the rank reported it,
    /// and the rank's host, in rank order: the ranks that execute video
    /// encoder calls report one, every other rank none
    /// (`ipc::MainProcessInfo::encoder_pool`).
    ///
    /// `from_inventory` fills this from its ranks; `from_workers` leaves it
    /// empty, which stores no encoder output, and a simulated
    /// worker states its own (`SimEngine::set_encoder_pools`).
    pub rank_encoders: BTreeMap<WorkerGroupId, RankEncoderPools>,
}

/// Each rank's host and its host encoder pool, as it reported it, in rank
/// order (`WorkerInfo::rank_encoders`).
pub type RankEncoderPools = Vec<(String, Option<Box<ipc::EncoderPoolInfo>>)>;

impl WorkerInfo {
    /// Constructs capability information for multiple pools.
    ///
    /// Each rank validated the report it built, so the records are taken as
    /// they are. Fails when `pools` is empty, repeats a pool id, or holds
    /// replicas of one component that expose different numerical outputs.
    pub fn from_workers(pools: Vec<(WorkerGroupId, MainProcessInfo)>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !pools.is_empty(),
            "worker info must contain at least one pool"
        );
        let mut ids = std::collections::HashSet::new();
        let mut components: BTreeMap<&str, &ipc::ComponentInfo> = BTreeMap::new();
        for (id, info) in &pools {
            anyhow::ensure!(ids.insert(id), "worker info repeats pool id {id}");
            for component in &info.components {
                if let Some(other) = components.get(component.name.as_str()) {
                    anyhow::ensure!(
                        other.outputs == component.outputs,
                        "replicas of component {} expose different numerical outputs",
                        component.name
                    );
                } else {
                    components.insert(&component.name, component);
                }
            }
        }
        Ok(Self {
            workers: pools,
            ..Self::default()
        })
    }

    /// Returns the configuration the engine placed component `name` with on
    /// `worker`, or `None` when the pool holds no such component.
    pub fn component_config(&self, worker: &WorkerGroupId, name: &str) -> Option<&BlockConfig> {
        self.layouts.get(worker)?.components.get(name)
    }

    /// The persistent buffer bytes each rank of `worker` backs, in rank
    /// order; empty for a pool this info does not hold.
    pub fn rank_buffer_bytes(&self, worker: &WorkerGroupId) -> Vec<u64> {
        if let Some(bytes) = self.rank_buffer_bytes.get(worker) {
            return bytes.clone();
        }
        let ranks = self
            .layouts
            .get(worker)
            .map_or(1, |layout| layout.ranks.max(1));
        self.workers
            .iter()
            .find(|(id, _)| id == worker)
            .map(|(_, info)| vec![info.buffer_pool_bytes; ranks])
            .unwrap_or_default()
    }

    /// The ranks `worker` holds; one for a pool this info does not hold.
    pub fn rank_count(&self, worker: &WorkerGroupId) -> usize {
        self.rank_buffer_bytes(worker).len().max(1)
    }

    /// Resolves which component serves each media call across every worker.
    ///
    /// The video graph may span workers: a model worker serves decoding and a
    /// host worker serves encoding and muxing. Each call is served by one
    /// component, which several workers may replicate, and a deployment that
    /// muxes serves the whole video graph between its workers.
    pub fn media_routing(&self) -> anyhow::Result<BTreeMap<MediaCall, String>> {
        // The first worker seen serving a call is kept only to name it if
        // another worker routes that call to a different component.
        let mut routing: BTreeMap<MediaCall, (String, &WorkerGroupId)> = BTreeMap::new();
        for (id, info) in &self.workers {
            for (call, component) in &info.media_components {
                if let Some((other, owner)) = routing.get(call) {
                    anyhow::ensure!(
                        other == component,
                        "media call {call:?} is served by component {other} of worker {owner} \
                         and by component {component} of worker {id}"
                    );
                    continue;
                }
                routing.insert(*call, (component.clone(), id));
            }
        }
        if routing.contains_key(&MediaCall::Muxing) {
            let missing = crate::graph::video_calls(false)
                .into_iter()
                .filter(|call| !routing.contains_key(call))
                .map(|call| format!("{call:?}"))
                .collect::<Vec<_>>();
            anyhow::ensure!(
                missing.is_empty(),
                "a deployment that muxes video serves no component for {}",
                missing.join(", ")
            );
        }
        Ok(routing
            .into_iter()
            .map(|(call, (component, _))| (call, component))
            .collect())
    }

    /// Returns the media units one call of `call` covers at once: the ranks
    /// of a distributed serving component times its `units_per_rank`, or one
    /// for any other component, on the narrowest worker serving the call.
    ///
    /// A latent encoding round is dealt to the latent encoder's ranks that a
    /// request's prompt does not take (`Scheduler::latent_round`), so while
    /// the prompt is encoded a round covers only the ranks the latent
    /// encoder does not share with the text encoder of its worker. Those
    /// ranks count; a latent encoder that shares every rank encodes after
    /// the prompt, on all of them.
    ///
    /// Returns `None` when no worker serves `call`, or the width does not
    /// fit in `u32`.
    pub fn media_units(&self, call: MediaCall) -> Option<u32> {
        self.workers
            .iter()
            .filter_map(|(worker, info)| {
                let component = info.media_components.get(&call)?;
                let config = self.component_config(worker, component)?;
                if config.distribution.is_none() {
                    return Some(1);
                }

                let prompt = info
                    .media_components
                    .get(&MediaCall::TextEncoding)
                    .filter(|_| call == MediaCall::LatentEncoding)
                    .and_then(|name| self.component_config(worker, name));
                let unshared = config
                    .ranks
                    .iter()
                    .filter(|rank| prompt.is_none_or(|prompt| !prompt.ranks.contains(rank)))
                    .count();
                let ranks = if unshared == 0 {
                    config.ranks.len()
                } else {
                    unshared
                };
                u32::try_from(ranks * config.units_per_rank.max(1)).ok()
            })
            .min()
    }

    /// Derives the runtime's immutable capacity view from concrete pools.
    /// The returned value is not part of worker identity and is never
    /// reported as a physical worker.
    ///
    /// A single pool's record is returned unchanged. With several pools, the
    /// record is cloned from the first pool that serves a cache forward call (or
    /// the first pool when none does), so fields the merge does not set, such
    /// as the endpoint, come from that pool. Call support and
    /// media routing are unions, and most batch, slot, and storage limits
    /// take the smallest nonzero value any pool reports.
    /// The exceptions are the prefix cache, whose layout the cache pools must share
    /// and whose block count is their smallest and per-token footprint their
    /// largest; request slots of a deployment that muxes video,
    /// the narrowest per-component sum of replica slots; and the video
    /// denoiser and latent page geometry, taken from the first denoising
    /// pool and absent or zero without one.
    ///
    /// # Errors
    ///
    /// Fails when there are no pools. With several pools, it also fails when
    /// media routing conflicts or is incomplete, when a deployment that muxes
    /// video reports no video denoiser, when pools describe the denoiser
    /// differently, when cache pools disagree on cache layout or a routed cache pool
    /// has no cache, when summed capacities overflow `u32`, or when the merged
    /// record fails `MainProcessInfo::validate`.
    pub fn runtime_info(&self) -> anyhow::Result<MainProcessInfo> {
        anyhow::ensure!(!self.workers.is_empty(), "worker exposes no physical pools");
        if self.workers.len() == 1 {
            let info = self.workers[0].1.clone();
            check_video_tasks(&info)?;
            return Ok(info);
        }

        // Route-specific capacities contribute only when a pool implements the
        // corresponding call family. `routed` picks the first such pool, and
        // `cache_indices` holds the index of the first pool serving each cache
        // forward mode (prefill, decode, verify), deduplicated and in pool
        // order.
        let routed = |variant: CallKind| {
            self.workers
                .iter()
                .find(|(_, info)| info.supported_calls.contains(&variant))
                .map(|(_, info)| info)
        };
        let mut cache_indices = [
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
            CallKind::Forward(ForwardMode::Verify),
        ]
        .into_iter()
        .filter_map(|variant| {
            self.workers
                .iter()
                .position(|(_, info)| info.supported_calls.contains(&variant))
        })
        .collect::<Vec<_>>();
        cache_indices.sort_unstable();
        cache_indices.dedup();

        // Fields not merged below keep the seed pool's values.
        let seed_index = cache_indices.first().copied().unwrap_or(0);
        let mut merged = self.workers[seed_index].1.clone();
        merged.media_components = self.media_routing()?;
        let denoising = routed(CallKind::Media(MediaCall::Denoising));
        // The denoising worker owns the denoiser: its description is the
        // deployment's.
        merged.video_denoiser = denoising.and_then(|info| info.video_denoiser.clone());
        // A deployment that assembles video denoises over the fixed schedule
        // the denoising worker describes.
        anyhow::ensure!(
            !merged.media_components.contains_key(&MediaCall::Muxing)
                || merged.video_denoiser.is_some(),
            "a deployment that muxes video reports no video denoiser"
        );
        // Every cache stage must agree on its groups' retention policies and page
        // shapes. Capacity is the narrowest pool because a request may
        // traverse all routed cache stages with one set of unit ids, and the
        // per-unit footprint is the largest any of them reports
        // (`PrefixCacheInfo::merge`).
        if let Some(first_index) = cache_indices.first().copied() {
            let mut prefix_cache = self.workers[first_index]
                .1
                .prefix_cache
                .clone()
                .context("worker routes cache work to a pool without a prefix cache")?;
            for index in cache_indices.iter().copied().skip(1) {
                let other = self.workers[index]
                    .1
                    .prefix_cache
                    .as_ref()
                    .context("worker routes cache work to a pool without a prefix cache")?;
                prefix_cache = prefix_cache
                    .merge(other)
                    .context("worker cache pools expose incompatible cache layouts")?;
            }
            merged.prefix_cache = Some(prefix_cache);
        } else {
            merged.prefix_cache = None;
        }

        // A call is supported when any pool serves it. Unless noted at the
        // field, the limits take the smallest nonzero value, and are zero
        // only when every pool reports zero.
        merged.supported_calls = CallKind::ALL
            .into_iter()
            .filter(|variant| routed(*variant).is_some())
            .collect();
        merged.max_batch_calls = self
            .workers
            .iter()
            .map(|(_, info)| info.max_batch_calls)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.max_batch_tokens = self
            .workers
            .iter()
            .map(|(_, info)| info.max_batch_tokens)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.request_slots = if merged.media_components.contains_key(&MediaCall::Muxing) {
            // A replicated media route owns an independent request-row bank in
            // every group. End-to-end concurrency is the narrowest sum
            // of replica capacities among the graph's components. Each
            // component is keyed to the first call it serves; its replicas are
            // the pools that support that call and bind the component.
            let mut components = BTreeMap::new();
            for (call, component) in &merged.media_components {
                components.entry(component.as_str()).or_insert(*call);
            }
            components
                .into_iter()
                .map(|(component, call)| {
                    self.workers
                        .iter()
                        .filter(|(_, info)| {
                            info.supported_calls.contains(&CallKind::Media(call))
                                && info
                                    .components
                                    .iter()
                                    .any(|binding| binding.name == component)
                        })
                        .map(|(_, info)| info.request_slots)
                        .try_fold(0u32, |total, slots| total.checked_add(slots))
                        .context("media replica request capacity exceeds protocol range")
                })
                .collect::<anyhow::Result<Vec<_>>>()?
                .into_iter()
                .min()
                .unwrap_or(0)
        } else {
            self.workers
                .iter()
                .map(|(_, info)| info.request_slots)
                .filter(|limit| *limit > 0)
                .min()
                .unwrap_or(0)
        };
        merged.max_unresolved_calls = self
            .workers
            .iter()
            .map(|(_, info)| info.max_unresolved_calls)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        // A transfer call may be placed on any pool, so the narrowest
        // transport binds the engine's window of reserved transfers.
        merged.transfer_tickets = self
            .workers
            .iter()
            .map(|(_, info)| info.transfer_tickets)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        // Latent page geometry is the denoising pool's, zero without one.
        let flow = routed(CallKind::Media(MediaCall::Denoising));
        merged.latent_page_units = flow.map_or(0, |info| info.latent_page_units);
        merged.latent_pages = flow.map_or(0, |info| info.latent_pages);
        // The denoising pool's image shapes are those its runtime prepared,
        // and the widest diffusion LM step of any worker is warmed down to
        // one row.
        merged.image_warmups = flow.map_or_else(Vec::new, |info| info.image_warmups.clone());
        merged.captured_images = flow.and_then(|info| info.captured_images.clone());
        merged.image_capacity = flow.and_then(|info| info.image_capacity);
        merged.captured_latents = routed(CallKind::Media(MediaCall::LatentEncoding))
            .and_then(|info| info.captured_latents.clone());
        merged.image_unconditional_token_ids =
            flow.and_then(|info| info.image_unconditional_token_ids.clone());
        // Each image encoder's output is the one the worker that serves its
        // encoding states.
        merged.image_encoders.clear();
        for (call, name) in [
            (MediaCall::VisionEncoding, crate::encodings::VISION_FEATURES),
            (MediaCall::LatentEncoding, crate::encodings::LATENT_FEATURES),
        ] {
            if let Some(output) =
                routed(CallKind::Media(call)).and_then(|info| info.image_encoders.get(name))
            {
                merged.image_encoders.insert(name.to_owned(), *output);
            }
        }
        merged.diffusion_lm_step_rows = self
            .workers
            .iter()
            .map(|(_, info)| info.diffusion_lm_step_rows)
            .max()
            .unwrap_or(0);
        merged.buffer_pool_bytes = self
            .workers
            .iter()
            .map(|(_, info)| info.buffer_pool_bytes)
            .filter(|capacity| *capacity > 0)
            .min()
            .unwrap_or(0);
        merged.encoder_cache_entries = self
            .workers
            .iter()
            .map(|(_, info)| info.encoder_cache_entries)
            .filter(|capacity| *capacity > 0)
            .min()
            .unwrap_or(0);
        merged.encoder_entry_bytes = self
            .workers
            .iter()
            .map(|(_, info)| info.encoder_entry_bytes)
            .filter(|capacity| *capacity > 0)
            .min()
            .unwrap_or(0);

        merged.validate()?;
        check_video_tasks(&merged)?;
        Ok(merged)
    }
}

/// Checks that the deployment places a component for every call its video
/// denoiser's requests make.
///
/// A conditioned request reads its media on a host component and encodes its
/// conditions before denoising, so a deployment whose denoiser takes
/// conditions must place the media reader and both condition encoders as
/// well as the generation calls.
fn check_video_tasks(info: &MainProcessInfo) -> anyhow::Result<()> {
    let Some(denoiser) = info.video_denoiser.as_ref() else {
        return Ok(());
    };
    let missing = crate::graph::video_calls(denoiser.takes_conditions)
        .into_iter()
        .filter(|call| !info.media_components.contains_key(call))
        .map(|call| format!("{call:?}"))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        missing.is_empty(),
        "the deployment's denoiser {} but places no component for {}",
        if denoiser.takes_conditions {
            "takes conditions"
        } else {
            "generates"
        },
        missing.join(", ")
    );
    Ok(())
}

/// One call result returned from a simulated batch.
#[cfg(feature = "testing")]
#[derive(Debug, Clone)]
pub struct CallResult {
    /// Validated completion values; media storage is carried by `media` below.
    pub output: crate::sim::CallOutput,
    /// Claimed immutable output storage, or its request-local acquisition error.
    pub media: Result<Option<Arc<ipc::SharedMedia>>, ipc::SharedMediaError>,
}

/// Distinguishes applied control from physical retirement and unacknowledged failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOutcome {
    /// Every target pool acknowledged the command.
    Applied,
    /// The targeted request was tainted by a worker failure, so the command
    /// settles without an acknowledgment. The scheduler treats it as settled,
    /// like `Applied`.
    Retired,
    /// The command was not acknowledged and its physical ownership remains;
    /// the scheduler queues a failed `Finish` or `Free` again.
    Failed,
}

/// Terminal outcome of one logical lifecycle command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    /// Position among lifecycle commands, excluding request admissions.
    pub command_index: u32,
    /// Request targeted by the command.
    pub request_key: RequestKey,
    /// A failed command retains physical ownership until a subsequent release succeeds.
    pub outcome: CommandOutcome,
}

/// One independently ready subset of a logical batch, as a simulated pool
/// reports it.
#[cfg(feature = "testing")]
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Logical batch identity assigned at submission.
    pub batch_id: u64,
    /// Call completions this join reports as ready.
    pub results: Vec<CallResult>,
    /// Receipts for the batch's lifecycle commands; empty unless `done`.
    pub command_results: Vec<CommandResult>,
    /// Whether this join completes the batch.
    pub done: bool,
    /// Per-worker execution durations in microseconds.
    pub worker_exec_us: Vec<u64>,
    /// Per-worker model-forward statistics.
    pub forward_stats: Vec<ipc::ForwardStats>,
}

/// Dynamic error returned while polling or administering a worker.
///
/// Callers report worker failures rather than match them: the scheduler
/// fails the affected calls and logs the error chain. The failures it acts
/// on travel inside as typed errors and are recovered by downcasting: an
/// worker's loss (`WorkerFailure`) and a rank's execution error
/// (`ExecuteError`).
pub type WorkerError = anyhow::Error;

/// Backpressure and terminal failures returned by logical batch submission.
#[derive(Debug, thiserror::Error)]
pub enum WorkerSubmitError {
    #[error("worker queue is full")]
    /// Returns ownership of a batch rejected by bounded queue capacity.
    WouldBlock(Box<ExecutionBatch>),
    #[error(transparent)]
    /// Reports a terminal submission failure.
    Failed(#[from] anyhow::Error),
}

/// A worker-reported execution error classified for scheduler failure policy.
///
/// The typed taxonomy and execution context travel together so failure policy
/// and diagnostics use the same call identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("worker execute error: {message}")]
pub struct ExecuteError {
    /// Batch whose response carried this error. Composite workers use it to
    /// join the same terminal outcome across ranks before returning the
    /// failure to the scheduler.
    pub batch_id: Option<u64>,
    /// Whether the error invalidates the worker.
    pub fatal: bool,
    /// The failure's class, when the rank classified it.
    pub code: Option<ipc::ErrorCode>,
    /// Human-readable failure description.
    pub message: String,
    /// Execution phase in which the failure occurred, when known.
    pub phase: Option<String>,
    /// Call kind of the failed batch or numerical call, when the failure
    /// belongs to a batch with calls (see `MainProcessResponseError::route`).
    pub route: Option<String>,
    /// Calls affected by the worker failure.
    pub calls: Vec<ipc::ErrorCallIdentity>,
}

impl ExecuteError {
    /// The classification of a batch some of whose calls computed
    /// non-finite logits (`ErrorCode::NumericalFault`), which fails an epoch
    /// on trial as a whole; `None` for a batch whose failed calls failed
    /// otherwise, whose failure the scheduler reconciles by their requests
    /// alone.
    #[cfg(feature = "testing")]
    pub fn non_finite<'a>(
        batch_id: u64,
        outputs: impl IntoIterator<Item = &'a crate::sim::CallOutput>,
    ) -> Option<Self> {
        let code = ipc::ErrorCode::NumericalFault;
        outputs
            .into_iter()
            .any(|output| {
                output.status == ipc::CallStatus::Error && output.error_code == Some(code)
            })
            .then(|| Self {
                batch_id: Some(batch_id),
                fatal: false,
                code: Some(code),
                message: format!("calls of batch {batch_id} computed non-finite logits"),
                phase: None,
                route: None,
                calls: Vec::new(),
            })
    }
}

/// Terminal work abandoned by one Worker, with the exact invalidated ownership.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct WorkerFailure {
    /// Worker group reporting the failure.
    pub worker_id: WorkerGroupId,
    /// Invalidated incarnations; empty when the group retains its allocations.
    pub endpoints: Vec<ipc::MainProcessEndpoint>,
    /// Request epochs affected by the failed work or invalidated allocations.
    pub requests: Vec<RequestKey>,
    /// The requests among `requests` that a worker refused, each with the
    /// reason its input-error completion states (`CallOutput::error_message`).
    /// The scheduler rejects these with their reason and fails the others
    /// with `message`.
    pub refusals: HashMap<RequestKey, String>,
    /// Accepted logical calls that will no longer produce a device completion.
    pub retired: Vec<(u64, RequestKey, CallId)>,
    /// Buffers whose published locations no longer cover their complete logical value.
    pub buffers: Vec<ipc::BufferId>,
    /// Classified execution failure, if the ranks returned one before retirement.
    pub execution: Option<ExecuteError>,
    /// Human-readable failure description.
    pub message: String,
}

/// The asynchronous, pipelined boundary a simulated pool implements.
///
/// One submitted batch may come back from `poll` as several
/// [`BatchResult`]s: each carries completions not reported by an earlier one,
/// and the last has `done` set and carries the lifecycle command receipts.
/// A failure may instead surface as an error from `poll`.
#[cfg(feature = "testing")]
pub trait Worker: Send {
    /// Returns the worker's concrete pool capabilities.
    fn info(&self) -> &WorkerInfo;
    /// Whether every worker of group `worker` is ready to accept execution.
    fn is_ready(&self, worker: &WorkerGroupId) -> bool;
    /// Whether group `worker` executes nothing further: it is closed, or
    /// could not be replaced. A group whose replacement is starting is not
    /// ready but not closed either. The default suits a worker that
    /// replaces no group in the background: a group that is not ready is
    /// closed.
    fn is_closed(&self, worker: &WorkerGroupId) -> bool {
        !self.is_ready(worker)
    }
    /// Whether group `worker` has an unclaimed destination submission slot.
    fn has_capacity(&self, worker: &WorkerGroupId) -> bool;
    /// Whether every physical owner of a lifecycle command can accept its submission.
    fn command_has_capacity(&self, command: &BatchCommand) -> bool;
    /// Submits one logical batch without blocking for capacity.
    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), WorkerSubmitError>;
    /// Waits up to `timeout` for one partial or terminal batch result.
    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchResult>, WorkerError>;
    /// Closes the worker and its physical workers.
    fn close(&mut self) -> Result<(), WorkerError>;
    /// Commits the workers' epoch: each worker takes the predecessor
    /// epoch's storage it mapped read-only. Returns what each worker
    /// reported. The default suits a worker whose workers succeed no
    /// epoch.
    fn commit(&mut self) -> Result<Vec<String>, WorkerError> {
        Ok(Vec::new())
    }
    /// Publishes this epoch's immutable pages after its commit decision.
    /// Workers without shared storage have nothing to publish.
    fn publish_epoch(&mut self) -> Result<(), WorkerError> {
        Ok(())
    }
    /// Ends the storage of every epoch of `epochs` on the nodes of the
    /// worker's workers. The default suits a worker that retains no
    /// storage.
    fn retire(&mut self, epochs: &[u64]) -> Result<(), WorkerError> {
        let _ = epochs;
        Ok(())
    }
    /// Keeps the weights alone of every epoch of `epochs` on the nodes of
    /// the worker's workers, in host storage when `to_host`, and returns
    /// each node's report. The default suits a worker that retains no
    /// storage, which has no node to report.
    fn keep_weights(
        &mut self,
        epochs: &[u64],
        to_host: bool,
        cache_keys: &[Vec<u8>],
    ) -> Result<Vec<String>, WorkerError> {
        let _ = (epochs, to_host, cache_keys);
        Ok(Vec::new())
    }
    /// The storage ledgers of the nodes of the worker's workers: for each
    /// node, JSON of `{"host": ..., "ledgers": [...], "host_storage": {...}}`.
    /// The default suits a worker that retains no storage, whose ledgers
    /// are empty.
    fn ledgers(&mut self) -> Result<Vec<String>, WorkerError> {
        Ok(Vec::new())
    }

    /// Storage Server endpoints for the deployment nodes. Local numerical
    /// workers retain no storage outside their process.
    fn storage_nodes(&self) -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::new()
    }
    /// Performs the storage `operation` on every worker while none runs a
    /// batch, and returns each worker group's identity with its ranks'
    /// results, in rank order. The default refuses: the worker has no
    /// workers to operate on.
    fn operate_groups(
        &mut self,
        operation: &ipc::execution::EpochOperation,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, WorkerError> {
        anyhow::bail!("this worker performs no storage operation ({operation:?})")
    }
    /// Maps the host KV cache's page pools on every worker's ranks, each
    /// holding `tokens` tokens of every cache group (`uniserve.host_cache`),
    /// while none runs a batch, and returns each worker group's identity
    /// with its ranks' pools (JSON, `null` for a rank that copies no pages),
    /// in rank order. The default refuses: the worker has no workers.
    fn map_host_cache(
        &mut self,
        tokens: u64,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, WorkerError> {
        anyhow::bail!("this worker maps no host KV cache ({tokens} tokens)")
    }
    /// Performs the storage `operation` on every worker while none runs a
    /// batch, and returns each worker's result. The default refuses: the
    /// worker has no workers to operate on.
    fn operate(
        &mut self,
        operation: &ipc::execution::EpochOperation,
    ) -> Result<Vec<String>, WorkerError> {
        anyhow::bail!("this worker performs no storage operation ({operation:?})")
    }

    /// Commit prepared expert allocations.
    fn commit_experts(
        &mut self,
        order: &BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<String>, WorkerError> {
        let _ = order;
        anyhow::bail!("this worker holds no expert weights")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A worker serving `routes`; the one that denoises describes the
    /// deployment's fixed schedule.
    fn media_info(routes: &[(MediaCall, &str)]) -> MainProcessInfo {
        let denoises = routes.iter().any(|(call, _)| *call == MediaCall::Denoising);
        MainProcessInfo {
            media_components: routes
                .iter()
                .map(|(call, component)| (*call, (*component).to_owned()))
                .collect(),
            supported_calls: routes
                .iter()
                .map(|(call, _)| CallKind::Media(*call))
                .collect(),
            video_denoiser: denoises.then(|| VideoDenoiserInfo {
                takes_conditions: false,
                denoise_steps: 4,
                description: b"{}".to_vec(),
            }),
            ..MainProcessInfo::default()
        }
    }

    fn media_pool(
        id: &str,
        routes: &[(MediaCall, &str)],
        request_slots: u32,
    ) -> (WorkerGroupId, MainProcessInfo) {
        let mut info = media_info(routes);
        info.request_slots = request_slots;
        info.components = routes
            .iter()
            .map(|(_, component)| *component)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|name| ipc::ComponentInfo {
                name: name.to_owned(),
                outputs: Vec::new(),
            })
            .collect();
        (WorkerGroupId(id.to_owned()), info)
    }

    fn media_replica(id: &str, request_slots: u32) -> (WorkerGroupId, MainProcessInfo) {
        media_pool(
            id,
            &[
                (MediaCall::TextEncoding, "text_encoder"),
                (MediaCall::LatentPreparation, "denoiser"),
                (MediaCall::Denoising, "denoiser"),
                (MediaCall::VideoDecoding, "video_decoder"),
                (MediaCall::VideoEncoding, "video_codec"),
                (MediaCall::AudioDecoding, "audio_decoder"),
                (MediaCall::AudioEncoding, "muxer"),
                (MediaCall::Muxing, "muxer"),
            ],
            request_slots,
        )
    }

    #[test]
    fn replicated_media_routes_add_independent_request_capacity() -> anyhow::Result<()> {
        // Each pool serves the whole video graph, so every component's
        // replica capacity is the sum over both pools.
        let info = WorkerInfo::from_workers(vec![
            media_replica("replica-0", 2),
            media_replica("replica-1", 3),
        ])?;

        assert_eq!(info.runtime_info()?.request_slots, 5);
        Ok(())
    }

    #[test]
    fn shared_media_stages_bound_aggregate_replica_capacity() -> anyhow::Result<()> {
        let flow = [
            (MediaCall::LatentPreparation, "denoiser"),
            (MediaCall::Denoising, "denoiser"),
            (MediaCall::VideoDecoding, "video_decoder"),
            (MediaCall::AudioDecoding, "audio_decoder"),
        ];
        let info = WorkerInfo::from_workers(vec![
            media_pool("text", &[(MediaCall::TextEncoding, "text_encoder")], 8),
            media_pool("flow-0", &flow, 2),
            media_pool("flow-1", &flow, 2),
            media_pool(
                "host",
                &[
                    (MediaCall::VideoEncoding, "video_codec"),
                    (MediaCall::AudioEncoding, "muxer"),
                    (MediaCall::Muxing, "muxer"),
                ],
                16,
            ),
        ])?;

        // The two flow banks contribute four routes. The wider shared text
        // and host stages must not multiply that end-to-end capacity.
        assert_eq!(info.runtime_info()?.request_slots, 4);
        Ok(())
    }

    #[test]
    fn media_routing_is_the_union_over_workers() {
        // A model worker decodes and a host worker encodes and muxes; the
        // deployment's routing names every video call once.
        let model = media_info(&[
            (MediaCall::TextEncoding, "text_encoder"),
            (MediaCall::LatentPreparation, "denoiser"),
            (MediaCall::Denoising, "denoiser"),
            (MediaCall::VideoDecoding, "video_decoder"),
            (MediaCall::AudioDecoding, "audio_decoder"),
        ]);
        let host = media_info(&[
            (MediaCall::VideoEncoding, "video_codec"),
            (MediaCall::AudioEncoding, "muxer"),
            (MediaCall::Muxing, "muxer"),
        ]);
        let info = WorkerInfo {
            workers: vec![
                (WorkerGroupId("model".to_owned()), model),
                (WorkerGroupId("host".to_owned()), host),
            ],
            ..WorkerInfo::default()
        };
        let routing = info.media_routing().expect("the union is complete");
        assert_eq!(routing.len(), 8);
        assert_eq!(routing[&MediaCall::VideoEncoding], "video_codec");
        assert_eq!(routing[&MediaCall::VideoDecoding], "video_decoder");
    }

    /// A distributed component's round covers its ranks times its units per
    /// rank, an undistributed one's a single unit, and the narrowest replica
    /// binds; a call no worker serves has no round units. A latent encoding
    /// round leaves out the ranks its worker's text encoder takes.
    #[test]
    fn media_units_count_the_units_one_call_covers() {
        let parallel = |ranks: Vec<usize>| BlockConfig::parallel(ranks);
        let temporal = |ranks: Vec<usize>, units_per_rank: usize| BlockConfig {
            distribution: Some(proto::config::BlockDistribution::TemporalUnits),
            units_per_rank,
            ..parallel(ranks)
        };
        // A pool placing each named component as configured.
        let pool = |id: &str, routes: &[(MediaCall, &str)], configs: Vec<(&str, BlockConfig)>| {
            let (id, info) = media_pool(id, routes, 0);
            let layout = PoolLayout {
                queue_depth: 1,
                ranks: 8,
                components: configs
                    .into_iter()
                    .map(|(name, config)| (name.to_owned(), config))
                    .collect(),
            };
            ((id.clone(), info), (id, layout))
        };
        let encoder = |id: &str, config: BlockConfig| {
            pool(
                id,
                &[(MediaCall::LatentEncoding, "latent_encoder")],
                vec![("latent_encoder", config)],
            )
        };
        let round_units = |pools: Vec<(
            (WorkerGroupId, MainProcessInfo),
            (WorkerGroupId, PoolLayout),
        )>| {
            let (workers, layouts) = pools.into_iter().unzip();
            WorkerInfo {
                workers,
                layouts,
                ..WorkerInfo::default()
            }
            .media_units(MediaCall::LatentEncoding)
        };

        assert_eq!(
            round_units(vec![encoder("a", temporal((0..4).collect(), 1))]),
            Some(4)
        );
        assert_eq!(
            round_units(vec![encoder("a", temporal((0..4).collect(), 2))]),
            Some(8)
        );
        assert_eq!(
            round_units(vec![
                encoder("a", temporal((0..4).collect(), 1)),
                encoder("b", temporal((0..2).collect(), 1)),
            ]),
            Some(2)
        );
        assert_eq!(
            round_units(vec![encoder("a", parallel((0..4).collect()))]),
            Some(1)
        );

        // A latent encoding round covers the ranks the prompt's encoding
        // leaves free: those the latent encoder does not share with its
        // worker's text encoder, or all of them when it shares every one.
        let beside = |text: Vec<usize>| {
            round_units(vec![pool(
                "a",
                &[
                    (MediaCall::LatentEncoding, "latent_encoder"),
                    (MediaCall::TextEncoding, "text_encoder"),
                ],
                vec![
                    ("latent_encoder", temporal(vec![4, 5, 6, 7, 0, 1, 2, 3], 1)),
                    ("text_encoder", parallel(text)),
                ],
            )])
        };
        assert_eq!(beside(vec![0, 1, 2, 3]), Some(4));
        assert_eq!(beside(vec![0]), Some(7));
        assert_eq!(beside((0..8).collect()), Some(8));
        assert_eq!(
            round_units(vec![pool(
                "a",
                &[(MediaCall::Denoising, "denoiser")],
                Vec::new()
            )]),
            None
        );
    }

    /// A deployment whose denoiser serves a conditioned task must place the
    /// media reader and both condition encoders; one serving `t2va` alone
    /// needs only the generation calls.
    #[test]
    fn a_denoiser_that_takes_conditions_requires_its_condition_components() {
        let generation = [
            (MediaCall::TextEncoding, "text_encoder"),
            (MediaCall::LatentPreparation, "denoiser"),
            (MediaCall::Denoising, "denoiser"),
            (MediaCall::VideoDecoding, "video_decoder"),
            (MediaCall::AudioDecoding, "audio_decoder"),
            (MediaCall::VideoEncoding, "video_codec"),
            (MediaCall::AudioEncoding, "muxer"),
            (MediaCall::Muxing, "muxer"),
        ];
        let serving = |takes_conditions: bool, routes: &[(MediaCall, &str)]| {
            let mut info = media_info(routes);
            info.video_denoiser = Some(VideoDenoiserInfo {
                takes_conditions,
                denoise_steps: 4,
                description: b"{}".to_vec(),
            });
            WorkerInfo {
                workers: vec![(WorkerGroupId("model".to_owned()), info)],
                ..WorkerInfo::default()
            }
        };

        assert!(serving(false, &generation).runtime_info().is_ok());
        let message = serving(true, &generation)
            .runtime_info()
            .expect_err("a conditioned request reads and encodes its conditions")
            .to_string();
        assert!(
            message.contains("takes conditions") && message.contains("MediaReading"),
            "{message}"
        );

        let conditioned = [
            (MediaCall::MediaReading, "media_reader"),
            (MediaCall::VisionEncoding, "text_encoder"),
            (MediaCall::LatentEncoding, "latent_encoder"),
        ];
        let routes = [generation.as_slice(), conditioned.as_slice()].concat();
        assert!(serving(true, &routes).runtime_info().is_ok());
    }

    #[test]
    fn media_routing_refuses_an_incomplete_video_graph_and_a_repeated_call() {
        let model = media_info(&[(MediaCall::VideoDecoding, "video_decoder")]);
        let host = media_info(&[(MediaCall::Muxing, "muxer")]);
        let info = WorkerInfo {
            workers: vec![
                (WorkerGroupId("model".to_owned()), model.clone()),
                (WorkerGroupId("host".to_owned()), host),
            ],
            ..WorkerInfo::default()
        };
        let message = info
            .media_routing()
            .expect_err("muxing needs the graph")
            .to_string();
        assert!(message.contains("VideoEncoding"), "{message}");

        let other = media_info(&[(MediaCall::VideoDecoding, "decoder")]);
        let info = WorkerInfo {
            workers: vec![
                (WorkerGroupId("model".to_owned()), model),
                (WorkerGroupId("other".to_owned()), other),
            ],
            ..WorkerInfo::default()
        };
        let message = info
            .media_routing()
            .expect_err("one owner per call")
            .to_string();
        assert!(message.contains("VideoDecoding"), "{message}");
    }
}

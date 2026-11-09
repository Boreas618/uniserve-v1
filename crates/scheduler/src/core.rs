//! The deployment's composition: its worker groups, its schedulers and their
//! handles.
//!
//! A deployment runs one scheduler per DP rank. [`SchedulerHandle`] owns a
//! scheduler's thread and is the front door request producers submit and
//! control requests through.
//!
//! [`WorkerGroupConfig`] is the static deployment description: each group has
//! an ordered list of physical [`WorkerRank`]s and named components whose
//! `ranks` index into that list. A scheduler reads the engine configuration
//! (`proto::config::EngineConfig`) as its process resolved it, the process's
//! launch, and the mechanism's limits for its ranks ([`SchedulerConfig::of`]). A main process builds one scheduler per
//! rank it holds with `SchedulerHandle::over_batches_telling`, over the
//! devices of its own process; `SchedulerHandle::with_worker` takes the CPU
//! simulation backend instead.

use ipc::MediaCall;
#[cfg(any(test, feature = "testing"))]
use proto::config::DEFAULT_BLOCK;
use std::collections::BTreeMap;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

#[cfg(feature = "testing")]
use crate::DEFAULT_MAX_RUNNING_REQUESTS;
use crate::guidance_prefix_rows;
use crate::handle::{CommandSender, EventRx, SubmitError};
#[cfg(feature = "testing")]
use crate::worker::Worker;
use crate::worker::WorkerGroupId;
use crate::{CommandWaker, GenerationLimits, Request};
use crate::{Scheduler, SpecialTokenIds};
use crate::{SchedulerConfig, SchedulerStats};
use anyhow::Context as _;
use ipc::MainProcessInfo;
use ipc::RequestId;
use proto::config::EngineArgs;
#[cfg(feature = "testing")]
use proto::config::EngineConfig;
#[cfg(test)]
use proto::config::{BlockConfig, BlockDistribution};
pub use proto::config::{WorkerGroupConfig, WorkerGroupRole, WorkerRank};
use worker::batch::Batches;

/// A storage scope no other deployment names.
pub(crate) fn unnamed_deployment() -> String {
    format!("deployment-{}", uuid::Uuid::new_v4().simple())
}

/// The arguments of an engine process of `model` on the CPU simulation backend: one pool of one
/// rank on this host, named as its worker group, the simulator's generation limits and end of
/// text, and every other section's default. Pair them with [`SchedulerHandle::with_worker`] and
/// [`SchedulerConfig::default`].
#[cfg(feature = "testing")]
#[expect(
    clippy::expect_used,
    reason = "the default is a positive constant and the simulated group has one rank"
)]
pub fn sim_args(model: impl Into<String>) -> EngineArgs {
    use std::num::NonZeroUsize;
    let group = WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cpu",
        1,
        2,
        WorkerGroupConfig::single_component(DEFAULT_BLOCK, 1),
    );
    let mut args = EngineArgs::new(EngineConfig {
        executable: proto::BUILD.to_owned(),
        model: proto::config::ModelConfig {
            model: model.into(),
            max_model_len: 0,
            dtype: crate::ModelDtype::BFloat16,
            quantization: Default::default(),
            generation_limits: crate::sim::SimEngine::generation_limits(),
            bos: 0,
            // `SimEngine` fabricates this end of text after `text_len` tokens; the
            // scheduler must recognize it to finish a simulated request.
            eos: vec![crate::sim::FAKE_EOS_TOKEN],
            end_of_image: 0,
            canvas_sampling: None,
            prefill_outputs: true,
        },
        parallel: proto::config::ParallelConfig {
            tensor_parallel_size: NonZeroUsize::MIN,
            data_parallel_size: NonZeroUsize::MIN,
            engine_host: "localhost".to_owned(),
            worker_groups: vec![group.clone()],
        },
        deployment: unnamed_deployment(),
        max_running_requests: NonZeroUsize::new(DEFAULT_MAX_RUNNING_REQUESTS)
            .expect("a positive default"),
        max_num_batched_tokens: None,
        attention_backend: String::new(),
        prefill_attention_backend: None,
        decode_attention_backend: None,
        moe_runner_backend: None,
        device: Default::default(),
        scheduler: Default::default(),
        media: Default::default(),
        mem_fraction_static: 0.9,
        load: Default::default(),
        kv_cache: Default::default(),
        moe: Default::default(),
        cuda_graph: Default::default(),
        worker: Default::default(),
        speculative: Default::default(),
        video: Default::default(),
        profiling: Default::default(),
    });
    // The simulation keeps no host tier: it has no Storage Server to keep one in.
    args.engine_config.kv_cache.host_tier = false;
    args.host = "localhost".to_owned();
    args.group = group.id.0.clone();
    args.world_size = u32::try_from(group.ranks.len()).expect("one rank");
    args.queue_depth = 2;
    args
}

/// The control tokens a scheduler's requests are finished and bounded by: the model's.
fn control_tokens(model: &proto::config::ModelConfig) -> SpecialTokenIds {
    SpecialTokenIds {
        bos: model.bos,
        eos: model.eos.clone(),
        end_of_image: model.end_of_image,
    }
}

/// The host KV cache's size `kv_cache` states (`kv_cache_config.host_ratio` and `host_gib`).
fn host_cache(
    kv_cache: &proto::config::KvCacheConfig,
) -> anyhow::Result<crate::kv_cache::HostCacheSize> {
    Ok(match (kv_cache.host_ratio, kv_cache.host_gib) {
        // A deployment that states neither keeps no host tier of its own.
        (None, None) => crate::kv_cache::HostCacheSize::default(),
        (ratio, gib) => crate::kv_cache::HostCacheSize::new(ratio, gib.map(|gib| gib.get() << 30))?,
    })
}

/// One DP rank's scheduler, running on a thread of its own, and the front
/// door through which its requests are submitted and controlled.
///
/// The thread owns the [`Scheduler`] and its worker; the handle reaches
/// it only through the scheduler's command channel, and keeps what the
/// scheduler reported at start: the worker capabilities, the media units of
/// each routed media call, the generation limits and the live statistics.
/// The dead latch records a fatal worker or worker failure, after which
/// the handle refuses submissions. Dropping the handle shuts the scheduler
/// down and joins its thread.
pub struct SchedulerHandle {
    commands: CommandSender,
    info: MainProcessInfo,
    /// Media units one call of each routed media call covers at once
    /// (`WorkerInfo::media_units`).
    media_units: BTreeMap<MediaCall, u32>,
    stats: Arc<SchedulerStats>,
    generation_limits: GenerationLimits,
    /// The worker group whose KV cache the scheduler's pool allocates, by
    /// which a successor epoch matches the scheduler's prefix cache index;
    /// `None` without a prefix cache.
    cache_group: Option<String>,
    /// The latent seed of what the scheduler's denoising computes, which
    /// keys the trajectories it stores (`Scheduler::latent_seed`).
    latent_seed: u64,
    /// The served model identifier.
    model: String,
    /// Set after `Scheduler::run` returns a fatal exit (worker or worker
    /// failure). A scheduler thread panic does not set it.
    dead: Arc<AtomicBool>,
    /// The first error of that fatal exit (`Scheduler::failure_handle`).
    failure: Arc<std::sync::OnceLock<String>>,
    /// Whether the scheduler started on trial, as a successor epoch's do;
    /// its commit ends the trial.
    trial: bool,
    /// The device-side submission path of the worker's own crate. A handle
    /// built over a simulated worker has none; the shadow pass and the
    /// in-process submission both need it.
    batches: Option<Arc<dyn Batches>>,
    /// Request ids, shared with the warm-up's view of this scheduler.
    next_id: Arc<AtomicU64>,
    /// Set once a policy instance actually serves (§7.3); see `wait_serving`.
    serving: tokio::sync::watch::Receiver<bool>,
    /// The generation each of this scheduler's batches is stamped with: the
    /// number the control thread's switchover arm last wrote (§7.3). One cell
    /// per rank, shared with the scheduler thread and the in-process submission
    /// path, which reads it once per step.
    generation: Arc<std::sync::atomic::AtomicU64>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl SchedulerHandle {
    /// [`Self::over_batches_telling`] with nothing told of a step's progress.
    ///
    /// # Errors
    ///
    /// Fails as [`Self::over_batches_telling`] does.
    #[cfg(feature = "testing")]
    pub fn over_batches(
        args: &EngineArgs,
        limits: SchedulerConfig,
        batches: Arc<dyn Batches>,
    ) -> anyhow::Result<Self> {
        Self::over_batches_telling(args, limits, batches, CommandWaker::noop())
    }

    /// Starts one scheduler thread over the worker crate's batch path — the
    /// in-process submission path of §2.4 — and returns its handle, with
    /// `progress` told whenever a step produced events or an answer: what the
    /// thread reading them waits on.
    ///
    /// The scheduler reads the process's arguments `args`: its epoch, its host
    /// and its pool, and the engine configuration as the process resolved it
    /// (the model and what was read of its checkpoint, the KV cache, the worker groups' roles). It
    /// takes `limits`, the mechanism's bounds for the ranks it drives
    /// ([`SchedulerConfig::of`]). The pool's capabilities come from what the
    /// worker reports (`Batches::inventory`), and the batches go to `batches`
    /// itself. The scheduler decides nothing until a policy serves it: the
    /// daemon's first `Precopy` and `Switchover` install one (§7.3), and until
    /// then a step forms no batch.
    ///
    /// Fails when the worker
    /// reports no pool, pools that disagree about a component, or not the
    /// launch's pool, when a pool serves fewer request rows than the running
    /// requests need, or when the scheduler's thread cannot be started.
    pub fn over_batches_telling(
        args: &EngineArgs,
        limits: SchedulerConfig,
        batches: Arc<dyn Batches>,
        progress: CommandWaker,
    ) -> anyhow::Result<Self> {
        let engine = &args.engine_config;
        let info = crate::worker::WorkerInfo::from_inventory(batches.inventory())?;
        Self::check_pool(args, &info)?;
        let runtime = info.runtime_info()?;
        Self::check_rows(&runtime, limits.max_running_requests)?;
        let submissions = crate::submissions_over(Arc::clone(&batches))?;
        let waker = submissions.waker();
        let scheduler = Scheduler::new(
            submissions,
            control_tokens(&engine.model),
            limits,
            engine.model.dtype,
            engine.model.generation_limits.clone(),
        )?;
        let mut scheduler = Self::configured(scheduler, args)?;
        scheduler.set_progress(progress);
        let mut handle = Self::spawn(scheduler, engine.model.model.clone(), waker)?;
        handle.batches = Some(batches);
        // The serving path's warm-up is the serving path's: it runs once the
        // epoch's first policy serves, on a thread of its own, and startup does
        // not wait for it.
        crate::warmup::after_serving(&handle, args);
        Ok(handle)
    }

    /// Runs a scheduler over a simulated worker, as
    /// [`Self::over_batches_telling`] reads `args` and `limits`.
    ///
    /// The command waker is a no-op, so a command does not interrupt the
    /// scheduler's idle park, and the worker's pools are not compared with
    /// the launch's.
    #[cfg(feature = "testing")]
    pub fn with_worker(
        args: &EngineArgs,
        limits: SchedulerConfig,
        worker: Box<dyn Worker>,
    ) -> anyhow::Result<Self> {
        let engine = &args.engine_config;
        let scheduler = Scheduler::new(
            crate::submissions(worker),
            control_tokens(&engine.model),
            limits,
            engine.model.dtype,
            engine.model.generation_limits.clone(),
        )?;
        let scheduler = Self::configured(scheduler, args)?;
        Self::spawn(scheduler, engine.model.model.clone(), CommandWaker::noop())
    }

    /// Refuses a token runtime whose workers hold fewer request rows than
    /// `max_running_requests` running requests need.
    ///
    /// A token runtime serves those requests only while its workers hold that
    /// many rows beyond the guidance prefix's; the scheduler would otherwise
    /// lower the limit. Workers without a prefix cache fit their request pools
    /// to device storage, and the scheduler bounds their requests by the rows
    /// they report.
    fn check_rows(info: &MainProcessInfo, max_running_requests: usize) -> anyhow::Result<()> {
        let request_rows = info.request_slots as usize;
        let needed_rows = max_running_requests + guidance_prefix_rows(info);
        anyhow::ensure!(
            !info.uses_cache() || request_rows >= needed_rows,
            "the workers hold {request_rows} request rows, but {max_running_requests} running \
             requests need {needed_rows}"
        );
        Ok(())
    }

    /// Refuses a worker that does not serve the launch's pool: the process's
    /// own group (`EngineArgs::group`), on its world's ranks.
    ///
    /// The pool is the process, named as the launch names it: the cache segments a successor
    /// adopts, the storage the process claims and the epoch's records all key it by that name.
    /// A scheduler started for a pool the worker does not serve, or for a
    /// rank count it does not serve, would place calls on ranks it does not
    /// have; the launch and the worker are two statements of one process,
    /// so they are compared before anything runs.
    fn check_pool(args: &EngineArgs, info: &crate::worker::WorkerInfo) -> anyhow::Result<()> {
        let group = WorkerGroupId(args.group.clone());
        anyhow::ensure!(
            info.workers.iter().any(|(served, _)| *served == group),
            "the launch names the pool {}, which the worker does not serve",
            group.0
        );
        // `from_inventory` states every pool's rank count; a worker
        // that reports none places the pool on one rank.
        let served = info.layouts.get(&group).map_or(1, |layout| layout.ranks);
        anyhow::ensure!(
            u32::try_from(served).is_ok_and(|served| served == args.world_size),
            "the launch places {} ranks of pool {}, but the worker serves {served}",
            args.world_size,
            group.0
        );
        Ok(())
    }

    /// Applies the configuration's per-epoch settings to a freshly built
    /// scheduler: its epoch identity, its prefix reuse and state interval, its
    /// prefill groups, and the host storages its ranks map.
    fn configured(mut scheduler: Scheduler, args: &EngineArgs) -> anyhow::Result<Scheduler> {
        let engine = &args.engine_config;
        let kv_cache = &engine.kv_cache;
        scheduler.set_epoch(args.epoch);
        scheduler.set_prefix_reuse(kv_cache.prefix_caching);
        scheduler.set_state_interval(
            kv_cache
                .state_interval
                .map_or(crate::kv_cache::DEFAULT_STATE_INTERVAL, |interval| {
                    interval.get() as usize
                }),
        );
        scheduler.set_prefill_groups(
            engine
                .parallel
                .worker_groups
                .iter()
                .filter(|group| group.role == WorkerGroupRole::Prefill)
                .map(|group| group.id.clone()),
        );
        // A scheduler with a host tier keeps its storages under the
        // deployment's name, where its claim is; one without keeps them in a
        // scope of its own, so two engine processes of one deployment on a node
        // never contest one scope.
        let host_tier = kv_cache.prefix_caching && kv_cache.host_tier;
        let scope = if host_tier {
            engine.deployment.clone()
        } else {
            unnamed_deployment()
        };
        if host_tier {
            scheduler
                .enable_kv_offload(
                    &scope,
                    args.epoch,
                    &args.host,
                    kv_cache
                        .boundary_interval
                        .map_or(crate::kv_cache::DEFAULT_BOUNDARY_INTERVAL, |interval| {
                            interval.get() as usize
                        }),
                    host_cache(kv_cache)?,
                    kv_cache.mooncake.is_some(),
                )
                .context("sizing the epoch's host KV cache")?;
        }
        // Denoising ranks that map host latent pools report them: a
        // preempted request's trajectory (a video's, or an image's) is
        // stored there.
        // Both pools are the deployment's, mapped by its ranks under its claim, so the stores reach
        // them under the deployment's name whether or not this scheduler keeps a host KV tier.
        scheduler.enable_trajectory_store(
            &storage::socket::default_socket(),
            &engine.deployment,
            args.epoch,
            &args.host,
        );
        // Ranks executing video encoders that map host encoder pools report
        // them: a request's encoder outputs are stored and found there.
        scheduler.enable_encoder_store(
            &storage::socket::default_socket(),
            &engine.deployment,
            args.epoch,
            &args.host,
        );
        Ok(scheduler)
    }

    /// Runs `scheduler`, which serves `model`, on a thread of its own.
    /// `waker` is fired after every command is enqueued; it is the
    /// worker's command waker for a worker that parks between steps, or
    /// the no-op waker for one the scheduler polls.
    ///
    /// # Errors
    ///
    /// Fails when the worker's media routing is inconsistent or the thread
    /// cannot be started.
    pub fn spawn(
        scheduler: Scheduler,
        model: impl Into<String>,
        waker: CommandWaker,
    ) -> anyhow::Result<Self> {
        // Requests size their media work by these widths before they are
        // submitted.
        let media_units = scheduler.media_units()?;
        let info = scheduler.info().clone();
        let generation_limits = scheduler.generation_limits().clone();
        let cache_group = scheduler.cache_group();
        let latent_seed = scheduler.latent_seed();
        let stats = scheduler.stats_handle();
        let failure = scheduler.failure_handle();
        // Taken before the scheduler moves onto its thread.
        let serving = scheduler.serving_receiver();
        let generation = scheduler.generation_handle();
        let trial = scheduler.on_trial();

        let (sender, receiver) = crossbeam_channel::unbounded();
        let dead = Arc::new(AtomicBool::new(false));
        let thread = {
            let dead = Arc::clone(&dead);
            std::thread::Builder::new()
                .name("scheduler".into())
                .spawn(move || {
                    if scheduler.run(receiver) {
                        dead.store(true, Ordering::SeqCst);
                    }
                })
                .context("failed to spawn the scheduler thread")?
        };
        Ok(Self {
            commands: CommandSender::with_waker(sender, waker),
            info,
            media_units,
            stats,
            generation_limits,
            cache_group,
            latent_seed,
            model: model.into(),
            dead,
            failure,
            trial,
            batches: None,
            next_id: Arc::new(AtomicU64::new(1)),
            serving,
            generation,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Everything the serving path's warm-up submits through, so a thread of
    /// the engine's can warm this rank without holding the handle it owns.
    pub(crate) fn warmup(&self) -> crate::warmup::Warm {
        crate::warmup::Warm {
            info: self.info.clone(),
            generation_limits: self.generation_limits.clone(),
            next_id: Arc::clone(&self.next_id),
            dead: Arc::clone(&self.dead),
            failure: Arc::clone(&self.failure),
            commands: self.commands.clone(),
        }
    }

    /// The generation this scheduler's batches carry: the one the epoch last
    /// switched to and has not purged (§7.3).
    ///
    /// The cell is shared with the scheduler thread, which reads it once per
    /// step; the control thread's switchover arm writes it while it holds the
    /// generations' lock, so a step never mixes two generations.
    pub fn generation(&self) -> Arc<std::sync::atomic::AtomicU64> {
        Arc::clone(&self.generation)
    }

    /// Records the serving generation, from the control thread's switchover arm.
    pub fn set_generation(&self, generation: u64) {
        self.generation
            .store(generation, std::sync::atomic::Ordering::Relaxed);
    }

    /// The scheduler's serving signal, for a thread that waits on it without
    /// holding the handle ([`crate::warmup::after_serving`]).
    pub fn serving(&self) -> tokio::sync::watch::Receiver<bool> {
        self.serving.clone()
    }

    /// The policy slot of this scheduler, as the control thread holds it (§7.3):
    /// the program serving and the one a precopy prepared.
    ///
    /// The slot lives on the scheduler thread, which alone instantiates, adopts
    /// and reverts a policy; the handle asks that thread.
    pub fn policy_mut(&self) -> crate::handle::PolicySlotHandle {
        crate::handle::PolicySlotHandle::new(self.commands.clone())
    }

    /// One of §7.3's steps 3 to 6, applied at this scheduler's next boundary
    /// where no batch and no submission is in flight, and answered with the step
    /// it applied at.
    pub fn stage(
        &self,
        stage: crate::Stage,
    ) -> crossbeam_channel::Receiver<Result<crate::handle::StageApplied, String>> {
        self.commands.stage(stage)
    }

    /// §7.3 step 2's shadow pass for the model generation `generation` (front's
    /// two axes: the model's generation, not a launch's): the following step's
    /// batches also run in shadow, against the resident data and without
    /// writing it.
    ///
    /// Answered once one step's batches have been shadowed, or with the reason
    /// the pass could not run.
    pub fn shadow(&self, generation: u64) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.shadow(generation)
    }

    /// One of the daemon's request commands, applied to the request it names,
    /// answered when it was enqueued.
    ///
    /// A command whose bookkeeping is the connection's — a consumer's position,
    /// a takeover — is answered with that reason: the engine's routing holds the
    /// request's stream, not a scheduler.
    pub fn request_command(
        &self,
        command: proto::request::wire::Command,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        let (answer, done) = crossbeam_channel::bounded(1);
        let _ = answer.send(self.commands.request_command(command));
        done
    }

    /// What this scheduler has done and what its policy has cost (§12), answered
    /// by the scheduler thread.
    pub fn counters(&self) -> crossbeam_channel::Receiver<crate::handle::Counters> {
        self.commands.counters()
    }

    /// Returns the worker-reported runtime capabilities.
    ///
    /// With several WorkerGroups this is the worker's merged view
    /// (`WorkerInfo::runtime_info`), not any single group's report.
    pub fn info(&self) -> &MainProcessInfo {
        &self.info
    }

    /// Returns the media units one call of `call` covers at once: the ranks
    /// of a distributed serving component times its units per rank, or one;
    /// `None` when no worker serves `call`.
    pub fn media_units(&self, call: MediaCall) -> Option<u32> {
        self.media_units.get(&call).copied()
    }

    /// Returns the generation limits resolved against worker capabilities.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.generation_limits.clone()
    }

    /// Returns the worker group whose KV cache the scheduler's pool
    /// allocates, which names its prefix cache index for a successor epoch;
    /// `None` without a prefix cache.
    pub fn cache_group(&self) -> Option<String> {
        self.cache_group.clone()
    }

    /// Returns the latent seed of what the scheduler's denoising computes,
    /// which keys the trajectories it stores at preemption; zero when it
    /// stores none.
    pub fn latent_seed(&self) -> u64 {
        self.latent_seed
    }

    /// Promotes the scheduler, whose epoch succeeds another: its workers
    /// adopt the predecessor's cache flags of the units they inherit, and
    /// its pool imports the predecessor's published pages, from which
    /// requests reuse their prefixes. Blocks until it is done.
    ///
    /// # Errors
    ///
    /// Fails when the epoch's own cache, grown into the predecessor's storage
    /// freed at once when it `grows`, cannot hold the requests the
    /// predecessor hands over (which `recompute` from their tokens or not),
    /// a worker refuses or dies, or the scheduler has stopped.
    pub fn promote(
        &self,
        parked: crate::ParkedCache,
        recompute: bool,
        grows: bool,
    ) -> anyhow::Result<()> {
        self.commands
            .promote(parked, recompute, grows)
            .recv()
            .map_err(|_| anyhow::anyhow!("the scheduler stopped while it promoted"))?
            .map_err(|error| anyhow::anyhow!(error))
    }

    /// Returns live scheduler statistics shared with the scheduler thread.
    pub fn stats(&self) -> &Arc<SchedulerStats> {
        &self.stats
    }

    /// Returns the served model identifier.
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Returns whether the scheduler stopped on a terminal worker or
    /// worker failure: from its first fatal error on, before its thread
    /// ends the requests it holds.
    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::SeqCst) || self.failure.get().is_some()
    }

    /// The first error of the scheduler's fatal stop, once it stops.
    pub fn failure(&self) -> Option<String> {
        self.failure.get().cloned()
    }

    /// Whether the scheduler started on trial, as a successor epoch's do.
    pub fn starts_on_trial(&self) -> bool {
        self.trial
    }

    /// Submits a request and returns its bounded event stream.
    ///
    /// The returned receiver cancels the request when dropped before its
    /// terminal event.
    ///
    /// # Errors
    ///
    /// Fails with `SubmitError::Dead` once the dead latch is set and with
    /// `SubmitError::Closed` when the command channel no longer accepts
    /// commands.
    pub fn submit(&self, request: impl Into<Request>) -> Result<EventRx, SubmitError> {
        if self.is_dead() {
            return Err(SubmitError::Dead);
        }
        self.commands.submit(request.into())
    }

    /// Submits requests in one command and returns their event streams in
    /// submission order: the scheduler queues every one of them before it
    /// forms its next batch, as it would requests from concurrent clients
    /// that all arrived before that batch. Separate `submit` calls give no
    /// such guarantee.
    ///
    /// # Errors
    ///
    /// As [`Self::submit`].
    pub fn submit_all(&self, requests: Vec<Request>) -> Result<Vec<EventRx>, SubmitError> {
        if self.is_dead() {
            return Err(SubmitError::Dead);
        }
        self.commands.submit_all(requests)
    }

    /// Cancels a request.
    pub fn cancel(&self, id: RequestId) {
        self.commands.cancel(id);
    }

    /// Finishes a request successfully, as when its decoder matched a stop
    /// string.
    pub fn stop(&self, id: RequestId) {
        self.commands.stop(id);
    }

    /// Acknowledges that the first `output_token_count` public tokens of a
    /// request with stop strings complete none of them.
    pub fn acknowledge_at(&self, id: RequestId, output_token_count: usize) {
        self.commands.acknowledge_at(id, output_token_count);
    }

    /// Aborts a request immediately without applying client cancellation
    /// semantics.
    pub fn abort(&self, id: RequestId) {
        self.commands.abort(id);
    }

    /// Parks the scheduler: every token request it holds that resumes from
    /// its generated tokens (`GenerationRequest::is_migratable`) ends as
    /// migratable, a queued one at once, a running one once its calls in
    /// flight drain, after every token it generated; the other requests
    /// finish; then it executes nothing more until resumed. The receiver
    /// gets its prefix cache index and what its migrating requests held
    /// once it parked. When it `lend`s, the epoch that succeeds it maps its
    /// cache: it lends that epoch the segments no request in flight
    /// continues from (`Scheduler::park`).
    pub fn park(
        &self,
        lend: bool,
    ) -> crossbeam_channel::Receiver<Result<crate::ParkedCache, String>> {
        self.commands.park(lend)
    }

    /// Reports the KV cache each of the scheduler's worker groups holds now,
    /// which an epoch succeeding this one maps; the receiver gets it.
    pub fn describe_cache(
        &self,
    ) -> crossbeam_channel::Receiver<Result<Vec<crate::GroupCache>, String>> {
        self.commands.describe_cache()
    }

    /// Accepted cache lengths of requests that can move to a successor.
    pub fn describe_requests(
        &self,
    ) -> crossbeam_channel::Receiver<Vec<proto::request::wire::RequestPrefix>> {
        self.commands.describe_requests()
    }

    /// §7.3 step 2: prepares `program`'s policy on this scheduler's state,
    /// without putting it in; the answer arrives when the blocking thread that
    /// loaded it is done.
    pub fn precopy_policy(
        &self,
        program: crate::PolicyProgram,
        source: std::sync::Arc<dyn crate::PolicySource + Send + Sync>,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.precopy_policy(program, source)
    }

    /// §7.3 step 3: the prepared policy answers from this scheduler's next
    /// batch boundary.
    pub fn switchover_policy(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.switchover_policy()
    }

    /// §7.3 step 5: probation ended; the replaced policy stays until the next
    /// precopy.
    pub fn commit_policy(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.commit_policy()
    }

    /// §7.3 step 4: the previous policy answers again, on the state this
    /// scheduler holds when the stage applies.
    pub fn revert_policy(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.revert_policy()
    }

    /// The step this scheduler reached and §12's cost counters for its policy,
    /// or a timeout when the scheduler does not answer.
    pub fn progress(&self) -> crossbeam_channel::Receiver<crate::Progress> {
        self.commands.progress()
    }

    /// Asks this scheduler for the policy failure it recorded since the last
    /// call, if it recorded one (§7.3).
    ///
    /// The answer is polled, not waited for: the caller of the stats tick is
    /// the engine's control thread, and a scheduler that does not answer must
    /// not hold it.
    pub fn take_policy_report(&self) -> crossbeam_channel::Receiver<Option<crate::PolicyReverted>> {
        self.commands.take_policy_report()
    }

    /// The state the epoch's policy is instantiated with, as this scheduler
    /// holds it and with `ranks` as the control thread knows them (§7.3);
    /// `None` when the scheduler does not answer in time.
    ///
    /// The state is the mechanism's, not a serving policy's, so an epoch's
    /// first precopy instantiates its policy on it before any policy serves.
    ///
    /// The control thread instantiates its route instance on it and reverts
    /// there after a `route` failure, so a failure costs one boundary, not the
    /// epoch.
    pub fn policy_state(
        &self,
        ranks: Vec<scheduling::guest::view::RankView>,
    ) -> Option<crate::TrialState> {
        // A scheduler answers at its next command boundary, and one whose worker has no wake
        // reaches it when its park ends, which is `IDLE_LIVENESS_POLL` away at most.
        const POLICY_STATE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
        self.commands
            .policy_state(ranks)
            .recv_timeout(POLICY_STATE_WAIT)
            .ok()
            .flatten()
    }

    /// Publishes the committed epoch's immutable pages on every claimed node.
    pub fn publish_epoch(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.publish_epoch()
    }

    /// Pin or publish a scheduler's pages for a shared transfer round.
    pub fn cache_warm(
        &self,
        operation: storage::manifest::cache::Operation,
        together: Arc<crate::HookBarrier>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.cache_warm(operation, together)
    }

    /// Ends the storage of `epochs` on the nodes of the scheduler's workers,
    /// which the deployment's claims cover; the receiver gets whether it did.
    pub fn retire(&self, epochs: Vec<u64>) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.retire(epochs)
    }

    /// Keeps the weights alone of `epochs` on the nodes of the scheduler's
    /// workers, which the deployment's claims cover, in host storage when
    /// `to_host`; the receiver gets each node's report.
    pub fn keep_weights(
        &self,
        epochs: Vec<u64>,
        to_host: bool,
        cache_keys: Vec<Vec<u8>>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.keep_weights(epochs, to_host, cache_keys)
    }

    /// Reads the storage ledgers of the nodes of the scheduler's workers,
    /// which the deployment's claims cover; the receiver gets each node's
    /// as JSON.
    pub fn ledgers(&self) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.ledgers()
    }

    /// Halves, or restores when not `halve`, the byte budgets of the links
    /// the deployment's transfers use on the nodes of the scheduler's
    /// workers, which the deployment's claims cover; the receiver gets the
    /// hosts it throttled.
    pub fn throttle(
        &self,
        halve: bool,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.throttle(halve)
    }

    /// Resumes a parked scheduler; the receiver gets whether it did. When
    /// it `reclaim`s, the epoch it lent cache units to writes none of them
    /// any more, and it takes them back.
    pub fn resume(&self, reclaim: bool) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.resume(reclaim)
    }

    /// Commits the scheduler's epoch, which succeeded another: its workers
    /// take the predecessor's storage they mapped read-only. The receiver gets
    /// what each worker reported.
    pub fn commit(&self) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.commit()
    }

    /// Revokes every cached prefix page; requests in flight keep theirs.
    pub fn flush_cache(&self) {
        self.commands.flush_cache();
    }

    /// Runs the worker hook `name` with the JSON `payload` on every worker
    /// once the batches in flight complete, together with the other
    /// schedulers of `together` (`HookBarrier`); the receiver gets each
    /// worker's result, or why the hook failed.
    pub fn hook(
        &self,
        name: String,
        payload: String,
        together: Arc<crate::HookBarrier>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.hook(name, payload, together)
    }

    /// Takes other scheduling limits and policy from the next step on; the
    /// receiver gets whether the scheduler did.
    pub fn configure(
        &self,
        limits: SchedulerConfig,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        self.commands.configure(limits)
    }

    /// Rewrites the weights in place from the checkpoint at `model_path`,
    /// the requests in flight continuing as `in_flight` says; the receiver
    /// gets what each worker reported, or why the update failed.
    pub fn update_weights(
        &self,
        model_path: String,
        in_flight: crate::WeightsInFlight,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.update_weights(model_path, in_flight)
    }

    /// Commit background expert copies and preserve the order on rank recovery.
    pub fn commit_experts(
        &self,
        order: BTreeMap<String, Vec<u32>>,
        together: Arc<crate::HookBarrier>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.commit_experts(order, together)
    }

    /// Gives back or restores KV cache storage; the receiver gets what each
    /// worker reported, or why the change failed.
    pub fn resize_cache(
        &self,
        change: crate::handle::CacheChange,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        self.commands.resize_cache(change)
    }

    /// Shuts down the scheduler, tears down its worker, and joins its
    /// thread. Idempotent.
    ///
    /// Blocks until the scheduler thread exits; on shutdown `Scheduler::run`
    /// first finishes queued and running requests with `Aborted` and closes
    /// the worker. A poisoned thread lock or a panicked scheduler thread is
    /// logged, not propagated.
    pub fn shutdown(&self) {
        self.commands.shutdown();
        let thread = match self.thread.lock() {
            Ok(mut guard) => guard.take(),
            Err(error) => {
                tracing::warn!("scheduler thread lock poisoned during shutdown");
                error.into_inner().take()
            }
        };
        if let Some(thread) = thread
            && thread.join().is_err()
        {
            tracing::warn!("scheduler thread panicked during shutdown");
        }
    }

    /// Shuts every scheduler of `schedulers` down together: each is asked
    /// before any is joined. Schedulers whose workers share an expert world
    /// must close those workers at once, since a closing expert rank waits
    /// for every peer to leave the union.
    pub fn shutdown_all<'a, I>(schedulers: I)
    where
        I: IntoIterator<Item = &'a SchedulerHandle>,
        I::IntoIter: Clone,
    {
        let schedulers = schedulers.into_iter();
        for scheduler in schedulers.clone() {
            scheduler.commands.shutdown();
        }
        for scheduler in schedulers {
            scheduler.shutdown();
        }
    }
}

impl Drop for SchedulerHandle {
    /// Runs `SchedulerHandle::shutdown`, blocking until the scheduler thread
    /// exits.
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host takes a contiguous block of the ranks, the first hosts take the
    /// remainder, and a rank's device is its local index on its own host.
    #[test]
    fn a_rank_layout_shares_ranks_across_the_named_hosts() {
        let components = WorkerGroupConfig::single_component("model", 1);

        let one =
            WorkerGroupConfig::placed(&["compute-0".to_owned()], "cpu", 2, 1, components.clone());
        assert!(
            one.ranks.iter().all(|rank| rank.host == "compute-0"),
            "a rank was placed on a host the rank layout does not name: {:?}",
            one.ranks
        );

        // Three ranks over two hosts: the first host takes the remainder, so
        // every rank is placed and each host's block stays contiguous.
        let two = WorkerGroupConfig::placed(
            &["compute-0".to_owned(), "compute-1".to_owned()],
            "cuda",
            3,
            1,
            components,
        );
        let hosts: Vec<_> = two.ranks.iter().map(|rank| rank.host.as_str()).collect();
        assert_eq!(hosts, ["compute-0", "compute-0", "compute-1"]);
        // A device name is the rank's place on its own host, so each host
        // starts at its first device.
        let devices: Vec<_> = two.ranks.iter().map(|rank| rank.device.as_str()).collect();
        assert_eq!(devices, ["cuda:0", "cuda:1", "cuda:0"]);
    }

    /// Entries covering the shapes a rank layout must validate: one component
    /// parallel over every rank, one dividing its output into units, and one
    /// placed alone.
    fn mixed_components(rank_count: usize) -> BTreeMap<String, BlockConfig> {
        let members: Vec<_> = (0..rank_count).collect();
        let mut components = WorkerGroupConfig::single_component("parallel", rank_count);
        components.insert(
            "divided".into(),
            BlockConfig {
                ranks: members,
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
                sequence_parallel: Default::default(),
                distribution: Some(BlockDistribution::TemporalUnits),
                units_per_rank: 1,
                weights: Default::default(),
            },
        );
        components.insert("alone".into(), BlockConfig::parallel(vec![0]));
        components
    }

    /// Groups of different sizes validate independently, and a group whose
    /// component spans more ranks than the group holds is refused.
    #[test]
    fn static_components_validate_their_own_parallel_members() -> anyhow::Result<()> {
        let mut workers = Vec::new();
        for (name, degree) in [("first", 1), ("second", 4), ("third", 2)] {
            let mut worker = WorkerGroupConfig::placed(
                &["localhost".to_owned()],
                "cuda",
                degree,
                2,
                WorkerGroupConfig::single_component(name, degree),
            );
            worker.id = WorkerGroupId(name.into());
            workers.push(worker);
        }
        WorkerGroupConfig::validate_all(&workers)?;

        workers[1].ranks.pop();
        assert!(WorkerGroupConfig::validate_all(&workers).is_err());
        Ok(())
    }

    #[test]
    fn the_shorthand_blocks_ranks_across_the_hosts_it_is_given() {
        // Eight devices over two four-device hosts, the engine's host first as
        // the CLI lists them. A deployment that places its muxer on rank zero
        // needs the lowest ranks on the first host, and each host must number
        // its own devices from zero.
        let hosts = ["rank-0".to_owned(), "rank-1".to_owned()];
        let worker = WorkerGroupConfig::placed(&hosts, "cuda", 8, 2, mixed_components(8));

        let rank_layout: Vec<_> = worker
            .ranks
            .iter()
            .map(|rank| (rank.host.as_str(), rank.device.as_str()))
            .collect();
        assert_eq!(
            rank_layout,
            vec![
                ("rank-0", "cuda:0"),
                ("rank-0", "cuda:1"),
                ("rank-0", "cuda:2"),
                ("rank-0", "cuda:3"),
                ("rank-1", "cuda:0"),
                ("rank-1", "cuda:1"),
                ("rank-1", "cuda:2"),
                ("rank-1", "cuda:3"),
            ]
        );
        assert!(worker.validate().is_ok());

        // A component the model spans over the group keeps all eight ranks,
        // and one it places alone stays on rank zero, which is the engine's
        // host.
        for entry in ["parallel", "divided"] {
            assert_eq!(
                worker.blocks[entry].ranks.len(),
                8,
                "{entry} spans the worker group"
            );
        }
        assert_eq!(worker.blocks["alone"].ranks, vec![0]);
    }

    /// The two-rank groups of four schedulers over two four-device hosts
    /// fill the engine's host first, and each group keeps its ranks on one
    /// host.
    #[test]
    fn the_data_parallel_shorthand_places_engine_cores_in_host_blocks() {
        let hosts = ["rank-0".to_owned(), "rank-1".to_owned()];
        let groups = WorkerGroupConfig::data_parallel(
            &hosts,
            "cuda",
            4,
            2,
            2,
            WorkerGroupConfig::single_component("model", 2),
        );

        let rank_layout: Vec<_> = groups
            .iter()
            .map(|worker| {
                (
                    worker.id.to_string(),
                    worker
                        .ranks
                        .iter()
                        .map(|rank| format!("{}/{}", rank.host, rank.device))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(
            rank_layout,
            vec![
                (
                    "model-0".into(),
                    vec!["rank-0/cuda:0".into(), "rank-0/cuda:1".into()]
                ),
                (
                    "model-1".into(),
                    vec!["rank-0/cuda:2".into(), "rank-0/cuda:3".into()]
                ),
                (
                    "model-2".into(),
                    vec!["rank-1/cuda:0".into(), "rank-1/cuda:1".into()]
                ),
                (
                    "model-3".into(),
                    vec!["rank-1/cuda:2".into(), "rank-1/cuda:3".into()]
                ),
            ]
        );
        assert!(WorkerGroupConfig::validate_all(&groups).is_ok());

        // One scheduler keeps the single-group identity.
        let single = WorkerGroupConfig::data_parallel(
            &hosts,
            "cuda",
            1,
            2,
            2,
            WorkerGroupConfig::single_component("model", 2),
        );
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].id.to_string(), "model");
    }

    #[test]
    fn a_rank_count_that_does_not_divide_its_hosts_still_places_every_rank() {
        // The first hosts take the remainder, which keeps each host's block
        // contiguous and rank zero on the engine's host.
        let hosts = ["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let worker = WorkerGroupConfig::placed(
            &hosts,
            "cuda",
            8,
            2,
            WorkerGroupConfig::single_component(DEFAULT_BLOCK, 8),
        );

        let hosts: Vec<_> = worker.ranks.iter().map(|rank| rank.host.as_str()).collect();
        assert_eq!(hosts, vec!["a", "a", "a", "b", "b", "b", "c", "c"]);
        assert_eq!(worker.ranks.len(), 8);
    }

    /// Reordering a component's ranks is a valid rank layout; repeating a rank
    /// is not.
    #[test]
    fn entry_geometry_uses_unique_ordered_rank_members() {
        let mut worker =
            WorkerGroupConfig::placed(&["localhost".to_owned()], "cuda", 4, 2, mixed_components(4));
        assert!(worker.validate().is_ok());

        worker.blocks.get_mut("parallel").unwrap().ranks.swap(0, 3);
        assert!(worker.validate().is_ok());

        worker.blocks.get_mut("parallel").unwrap().ranks[1] = 3;
        assert!(worker.validate().is_err());
    }

    #[test]
    fn static_bindings_reject_duplicate_identities_and_allow_component_replicas() {
        let worker = WorkerGroupConfig::placed(
            &["localhost".to_owned()],
            "cuda",
            1,
            2,
            WorkerGroupConfig::single_component(DEFAULT_BLOCK, 1),
        );
        let mut other = worker.clone();
        assert!(WorkerGroupConfig::validate_all(&[worker.clone(), other.clone()]).is_err());

        // Distinct identities on the same device validate, whether the groups
        // replicate one component name or hold different ones.
        other.id = WorkerGroupId("other".into());
        assert!(WorkerGroupConfig::validate_all(&[worker.clone(), other.clone()]).is_ok());

        let entry = other.blocks.remove(DEFAULT_BLOCK).unwrap();
        other.blocks.insert("divided".into(), entry);
        assert!(WorkerGroupConfig::validate_all(&[worker, other]).is_ok());
    }
}

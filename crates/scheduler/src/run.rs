//! Engine-loop construction, event parking, and owner-thread execution.
//!
//! Construction clamps the requested scheduler configuration and model
//! generation limits to the capacities the loaded workers report, then builds
//! the single-owner `Scheduler`. `Scheduler::run` is the owner-thread loop:
//! drain commands, take one nonblocking scheduling step, and park in the
//! worker only when the step made no progress.

use std::sync::Arc;

use super::*;
use ipc::{ForwardMode, MediaCall};

/// Applies the loaded workers' served features and capacities to the model's
/// sizing facts.
///
/// The workers state which generation features are served: a feature is
/// available when every call kind it needs is in the aggregate
/// `supported_calls`, whatever the model description lists. Numeric limits
/// are clamped to the reported batch-token, encoder-entry, buffer-pool, and
/// encoder-cache capacities.
fn resolve_generation_limits(
    mut limits: crate::GenerationLimits,
    info: &MainProcessInfo,
) -> crate::GenerationLimits {
    let supports = |kind| info.supported_calls.contains(&kind);
    let mut available = crate::GenerationFeatures::empty();
    if supports(CallKind::Forward(ForwardMode::Prefill))
        && supports(CallKind::Forward(ForwardMode::Decode))
    {
        available.insert(crate::GenerationFeatures::UNDERSTANDING);
    }
    if supports(CallKind::Media(MediaCall::VisionEncoding)) {
        available.insert(crate::GenerationFeatures::VISION_ENCODE);
    }
    if supports(CallKind::Media(MediaCall::LatentEncoding)) {
        available.insert(crate::GenerationFeatures::LATENT_ENCODE);
    }
    // A canvas denoises over a prompt the same worker's prefill cached.
    if supports(CallKind::Forward(ForwardMode::Prefill))
        && supports(CallKind::Forward(ForwardMode::TokenDenoising))
    {
        available.insert(crate::GenerationFeatures::TOKEN_DENOISING);
    }
    limits.features = available;

    limits.max_vit_grid_tokens = limits.max_vit_grid_tokens.min(info.max_batch_tokens);
    // One encoder product must fit both an encoder-cache entry and the buffer
    // pool.
    let feature_bytes = info.encoder_entry_bytes.min(info.buffer_pool_bytes);
    limits.max_latent_feature_bytes = limits.max_latent_feature_bytes.min(feature_bytes);
    limits.max_vision_feature_bytes = limits.max_vision_feature_bytes.min(feature_bytes);
    limits.encoder_cache_entries = limits.encoder_cache_entries.min(info.encoder_cache_entries);
    // A request's candidates are what the workers' samplers compute.
    limits.max_logprobs = limits.max_logprobs.min(info.max_logprobs);
    // Encoder caching requires a buffer pool.
    if info.buffer_pool_bytes == 0 {
        limits.encoder_cache_entries = 0;
    }
    limits
}

impl Scheduler {
    /// Constructs an engine loop from explicit scheduler and model capabilities.
    ///
    /// Worker limits clamp scheduler concurrency and resource capacities.
    ///
    /// Fails when the worker exposes an invalid aggregate capacity view.
    pub fn new(
        submissions: super::submission::Submissions,
        special_tokens: SpecialTokenIds,
        mut config: SchedulerConfig,
        generation_limits: crate::GenerationLimits,
    ) -> anyhow::Result<Self> {
        let info = submissions.info().runtime_info()?;
        let generation_limits = resolve_generation_limits(generation_limits, &info);
        let media_storage = submissions
            .info()
            .workers
            .iter()
            .filter(|(_, worker)| {
                worker
                    .supported_calls
                    .iter()
                    .any(|kind| matches!(kind, CallKind::Media(_)))
            })
            .map(|(id, worker)| {
                // The ranks back the results of the components the engine
                // placed on them, not a component list a rank reports.
                let components = submissions
                    .info()
                    .layouts
                    .get(id)
                    .map(|layout| layout.components.clone())
                    .unwrap_or_default();
                (
                    id.clone(),
                    MediaStorage::new(
                        worker.request_slots as usize,
                        &submissions.info().rank_buffer_bytes(id),
                        &components,
                        &info.media_components,
                    ),
                )
            })
            .collect();

        // Queue and batch limits cannot exceed the physical worker envelope.
        let max_batch_calls = info.max_batch_calls as usize;
        let max_batch_tokens = info.max_batch_tokens as usize;

        config.max_queued_requests = config
            .max_queued_requests
            .map(|bound| bound.max(proto::config::positive(1)));

        // The request pool also holds the rows the runtime keeps outside
        // `max_running_requests` (`guidance_prefix_rows`).
        // `SchedulerHandle::over_batches` refuses a token runtime whose
        // ranks hold too few, so this clamp lowers the limit only for
        // ranks that fit their request pools to device storage.
        let request_pool_capacity = info.request_slots as usize;
        let main_request_capacity = request_pool_capacity.max(1);

        // A stated bound the worker cannot hold is lowered here, never refused, when nothing
        // checks a decision against it: these three size a batch. A policy's own limits are the
        // mechanism's check and are refused instead, before weights
        // (the engine's `tier_refusal`). Each clamp names the option, the figure the
        // deployment stated and the figure the rank holds - a lowered bound whose numbers are
        // missing from the log is the same defect one severity down.
        let requested_seqs = config.max_running_requests.max(1);
        if requested_seqs > main_request_capacity {
            tracing::warn!(
                option = "max_running_requests",
                stated = requested_seqs,
                captured = main_request_capacity,
                request_rows = info.request_slots,
                "the workers' request rows lower the running-request limit"
            );
        }
        config.max_running_requests = requested_seqs.min(main_request_capacity);
        if config.max_num_batched_tokens.max(1) > max_batch_tokens {
            tracing::warn!(
                option = "max_num_batched_tokens",
                stated = config.max_num_batched_tokens,
                captured = max_batch_tokens,
                "the worker's token capacity lowers the token budget"
            );
        }
        config.max_num_batched_tokens = config.max_num_batched_tokens.max(1).min(max_batch_tokens);
        if max_batch_calls > 0 && config.max_batch > proto::config::positive(max_batch_calls) {
            tracing::warn!(
                option = "max_batch",
                stated = config.max_batch,
                captured = max_batch_calls,
                "the catalogue's row bound lowers the batch's call count"
            );
        }
        if max_batch_calls > 0 {
            config.max_batch = config
                .max_batch
                .min(proto::config::positive(max_batch_calls.max(1)));
        }

        // A canvas step and a block commit each spend a whole canvas of one
        // scheduling step's token budget, so the deployment's canvas must
        // fit it.
        if let Some(canvas) = config.canvas {
            if let Some(parameter) = canvas.invalid_parameter() {
                anyhow::bail!("block-diffusion sampling parameter `{parameter}` is out of range");
            }
            anyhow::ensure!(
                canvas.canvas_length as usize <= config.max_num_batched_tokens,
                "a {}-token generation canvas exceeds the {}-token step budget",
                canvas.canvas_length,
                config.max_num_batched_tokens
            );
        }

        // Storage and scheduler statistics share the resolved worker capacities.
        let cache = PrefixCacheCoordinator::from_main_process_info(&info);
        let stats = Arc::new(SchedulerStats::default());
        stats.prefix_cache.num_units.store(
            cache
                .as_ref()
                .map_or(0, |cache| cache.pool().usable_units()),
            Ordering::Relaxed,
        );

        // Runtime state remains single-owner; workers receive immutable batch
        // inputs assembled from these queues and request records.
        Ok(Self {
            worker: submissions,
            inflight: inflight::Inflight::new(),
            dispatch: dispatch::Dispatch::default(),
            storage: allocation::requests::Storage {
                cache,
                encoder_cache: crate::kv_cache::EncoderCacheManager::new(
                    generation_limits.encoder_cache_entries as usize,
                ),
                reserved_encoder_entries: 0,
                request_pool: RequestPool::new(info.request_slots as usize),
                buffer_pool: BufferPool::new(info.buffer_pool_bytes),
                media_storage,
                encoder_buffers: HashMap::new(),
                pending_buffer_frees: HashMap::new(),
            },
            info,
            generation_limits,
            special_tokens,
            requests: request::Requests::default(),
            waiting: VecDeque::new(),
            speculation_meter: Default::default(),
            retiring_requests: HashMap::new(),
            rows_held: 0,
            engine_id: 1,
            epoch: 0,
            next_product_generation: 1,
            next_request_epoch: 1,
            running: Vec::new(),
            output: output::OutputSender::default(),
            policy: None,
            policy_report: None,
            shadow_pass: None,
            generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            step_generation: 0,
            starvation_judged: scheduling::guest::Nanos(0),
            policy_precopy: None,
            policy_prepared: None,
            progress: crate::CommandWaker::noop(),
            policy_stage: None,
            recorder: mechanism::Recorder::default(),
            pages: std::collections::BTreeMap::new(),
            pages_seeded: false,
            watchdog: mechanism::Watchdog::default(),
            step: 0,
            config,
            lifecycle: Lifecycle::Serving,
            kv_offload: None,
            state_interval: crate::kv_cache::DEFAULT_STATE_INTERVAL,
            trajectories: None,
            encoder_outputs: None,
            hand_offs: None,
            draining: false,
            awaiting_index: None,
            held_for_index: Vec::new(),
            held: false,
            resizing: None,
            starved: Vec::new(),
            asleep: false,
            warming: super::warming::Warming::default(),
            trial: false,
            failure: Arc::default(),
            stats,
        })
    }

    /// The generation the epoch serves, which the batches of a step carry: the
    /// number the control thread's switchover arm last wrote here (§7.3).
    ///
    /// Read once per step, never per batch, so a switchover between two batches
    /// of one step cannot split them across generations.
    pub(super) fn serving_generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The serving generation's cell, for the handle that writes it.
    pub(crate) fn generation_handle(&self) -> Arc<std::sync::atomic::AtomicU64> {
        Arc::clone(&self.generation)
    }

    /// Returns the effective scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Takes `limits`' scheduling limits and policy from the next step on;
    /// the canvas stays. The limits must lie within the batch the workers
    /// were started for: the running requests within their request rows,
    /// a step's tokens and calls within their batch. A change of policy
    /// reorders the waiting queue by it, as if each request had been queued
    /// under it.
    ///
    /// # Errors
    ///
    /// Names the first limit beyond the workers' batch, or the canvas the
    /// step budget no longer holds; the scheduler keeps its limits.
    pub(super) fn reconfigure(&mut self, limits: SchedulerConfig) -> Result<(), String> {
        let info = &self.info;
        let running = (info.request_slots as usize).max(1);
        let beyond = |name: &str, value: usize, capacity: usize| {
            format!(
                "scheduler_config.{name} = {value} exceeds the {capacity} the workers were \
                 started for; an epoch change starts workers for it"
            )
        };
        if limits.max_running_requests > running {
            return Err(beyond(
                "max_running_requests",
                limits.max_running_requests,
                running,
            ));
        }
        let tokens = info.max_batch_tokens as usize;
        if limits.max_num_batched_tokens > tokens {
            return Err(beyond(
                "max_num_batched_tokens",
                limits.max_num_batched_tokens,
                tokens,
            ));
        }
        let calls = info.max_batch_calls as usize;
        if calls > 0 && limits.max_batch > proto::config::positive(calls) {
            return Err(beyond("max_batch", limits.max_batch.get(), calls));
        }
        if let Some(canvas) = self.config.canvas
            && canvas.canvas_length as usize > limits.max_num_batched_tokens.max(1)
        {
            return Err(format!(
                "a {}-token generation canvas exceeds the {}-token step budget",
                canvas.canvas_length, limits.max_num_batched_tokens
            ));
        }
        // The queue keeps no order of its own: the waiting requests are a
        // membership set, and each step's admission order is its output's
        // `admit` list (design §4.4). A new `policy` therefore reorders
        // nothing — the next output does.
        let graph_tiers = std::mem::take(&mut self.config.graph_tiers);
        self.config = SchedulerConfig {
            policy: Default::default(),
            max_batch: limits.max_batch.max(proto::config::positive(1)),
            max_num_batched_tokens: limits.max_num_batched_tokens.max(1),
            max_running_requests: limits.max_running_requests.max(1),
            chunked_prefill_size: limits
                .chunked_prefill_size
                .max(std::num::NonZeroUsize::new(1)),
            max_queued_requests: limits
                .max_queued_requests
                .map(|bound| bound.max(proto::config::positive(1))),
            canvas: self.config.canvas,
            // The draft limit is the epoch's; the refill policy is a
            // scheduling choice.
            draft_tokens: self.config.draft_tokens,
            graph_tiers,
        };
        tracing::info!(config = ?self.config, "the scheduler took new limits");
        Ok(())
    }

    /// The worker groups that prefill only (`WorkerGroupRole::Prefill`),
    /// whose requests hand their KV cache to a decode group.
    pub(crate) fn set_prefill_groups(
        &mut self,
        groups: impl IntoIterator<Item = crate::WorkerGroupId>,
    ) {
        self.dispatch.prefill_groups = groups.into_iter().collect();
    }

    /// The worker groups that decode only (`WorkerGroupRole::Decode`), which
    /// take requests by their hand-off alone.
    pub(crate) fn set_decode_groups(
        &mut self,
        groups: impl IntoIterator<Item = crate::WorkerGroupId>,
    ) {
        self.dispatch.decode_groups = groups.into_iter().collect();
    }

    /// Hand requests off from the prefill groups to the decode groups through
    /// this host's Storage Server, at `socket`, as a head of `epoch` of
    /// `deployment`, on host `host` (`crate::hand_off`).
    ///
    /// # Errors
    ///
    /// Fails when a Storage Server cannot be reached or a rank reports no
    /// retained cache planes.
    pub(crate) fn enable_hand_off(
        &mut self,
        socket: &std::path::Path,
        deployment: &str,
        epoch: u64,
        host: &str,
    ) -> anyhow::Result<()> {
        let ranks = self.worker.info().rank_caches.clone();
        self.hand_offs = Some(crate::hand_off::HandOffs::connect(
            socket, deployment, epoch, host, &ranks,
        )?);
        Ok(())
    }

    /// Sets the epoch that owns new requests' private cache continuations.
    pub(crate) fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    /// Enables or disables reusable prefix caching.
    pub fn set_prefix_reuse(&mut self, on: bool) {
        if let Some(cache) = self.storage.cache.as_mut() {
            cache.set_prefix_enabled(on);
        }
    }

    /// Checkpoints the state groups of decoding requests at the page
    /// boundaries that are multiples of `tokens`; a prompt call checkpoints
    /// the last boundary it reaches whatever the interval.
    pub(crate) fn set_state_interval(&mut self, tokens: usize) {
        self.state_interval = tokens.max(1);
    }

    /// Copy prompt pages of the worker groups whose ranks all map a page
    /// pool for every cache group through the host KV cache, as a head of
    /// `epoch` of `deployment` on `host` (what an epoch on trial commits is
    /// its own until its commit): ranks on `host` through its Storage Server's
    /// socket, the others through their servers' TCP addresses. Sliding
    /// windows are offloaded every `boundary_interval` pages of a long
    /// prompt, and `kv_storage` states whether a KV storage keeps every
    /// page the pools commit (`kv_cache_config.mooncake`). The epoch's host
    /// KV cache holds one token count on every rank, of `size`
    /// (`crate::kv_cache::HostCacheDemand`): its device tokens times the
    /// ratio, or what the stated bytes of each host hold, which every rank
    /// maps its pools at (`uniserve.host_cache`). Admission reads the
    /// published pages of the Storage Servers at `peers`, other hosts'
    /// endpoints (`kv_cache_config.peers`). A failure to reach this host's
    /// server leaves the host KV cache off.
    ///
    /// # Errors
    ///
    /// Fails when another host's server cannot size the cache, or a rank
    /// cannot map its pools.
    #[expect(
        clippy::too_many_arguments,
        reason = "the deployment, its epoch and host, and the host KV cache's settings"
    )]
    pub(crate) fn enable_kv_offload(
        &mut self,
        deployment: &str,
        epoch: u64,
        host: &str,
        boundary_interval: usize,
        size: crate::kv_cache::HostCacheSize,
        kv_storage: bool,
        peers: Vec<String>,
    ) -> anyhow::Result<()> {
        let socket = storage::default_socket();
        let mut kv_offload = match crate::kv_cache::KvOffloadManager::connect(
            &socket,
            deployment,
            epoch,
            boundary_interval,
            kv_storage,
        ) {
            Ok(kv_offload) => kv_offload,
            Err(error) => {
                tracing::warn!(%error, "the host KV cache is off: the storage server is unreachable");
                return Ok(());
            }
        };
        kv_offload.set_peers(peers);
        let mut rank_caches = self.worker.info().rank_caches.clone();
        let Some(demand) = crate::kv_cache::HostCacheDemand::of(&rank_caches, size.ratio) else {
            return Ok(());
        };
        // The epoch asks for its device tokens times the ratio, or, when
        // the deployment states each host's bytes, for whatever those hold;
        // each host lowers the count for the whole epoch to what its budget
        // holds over the epoch's ranks there.
        let asked = match size.host_bytes {
            Some(_) => u64::MAX,
            None => demand.tokens,
        };
        let mut tokens = asked;
        for (rank_host, (endpoint, token_bytes)) in demand.hosts {
            let held = kv_offload.host_cache_tokens(
                rank_host == host,
                endpoint.as_deref(),
                asked,
                token_bytes,
                size.host_bytes,
            )?;
            tokens = tokens.min(held);
        }
        tracing::info!(asked, tokens, "sized the epoch's host KV cache");
        for (worker, results) in self.worker.map_host_cache(tokens)? {
            let Some(ranks) = rank_caches.get_mut(&worker) else {
                continue;
            };
            for ((_, cache), result) in ranks.iter_mut().zip(results) {
                let pools: Option<Vec<ipc::HostPoolInfo>> = serde_json::from_str(&result)?;
                if let (Some(cache), Some(pools)) = (cache.as_mut(), pools) {
                    cache.page_pools = pools;
                }
            }
        }
        let mut served = false;
        for (worker, ranks) in &rank_caches {
            let reports: Option<Vec<crate::kv_cache::RankCache<'_>>> = ranks
                .iter()
                .map(|(rank_host, cache)| {
                    cache.as_ref().map(|cache| crate::kv_cache::RankCache {
                        local: rank_host == host,
                        cache,
                    })
                })
                .collect();
            if let Some(reports) = reports {
                served |= kv_offload.add_worker(worker, &reports);
            }
        }
        if served {
            self.kv_offload = Some(kv_offload);
        }
        Ok(())
    }

    /// Stores the trajectories of preempted requests (a video request's,
    /// and the image a token request is generating) in the host latent
    /// pools of the denoiser's worker group, reached through the Storage
    /// Server at `socket` for the ranks on `host` (the engine's) and over
    /// the TCP address another host's rank reports, when every rank that
    /// executes the denoiser reports one (`WorkerInfo::rank_latents`), as
    /// a head of `deployment` in `epoch`, whose trajectories the servers
    /// keep its own while it is on trial. The denoiser is the component
    /// that serves denoising calls, or the protocol's default component of
    /// a model that routes none. Without them a preempted request continues
    /// from noise wherever it resumes.
    pub fn enable_trajectory_store(
        &mut self,
        socket: &std::path::Path,
        deployment: &str,
        epoch: u64,
        host: &str,
    ) {
        let component = self
            .info
            .media_components
            .get(&MediaCall::Denoising)
            .cloned()
            .unwrap_or_else(|| DEFAULT_BLOCK.to_owned());
        let info = self.worker.info();
        let mut denoisers = Vec::new();
        for (worker, ranks) in &info.rank_latents {
            // The ranks that hold trajectories: those the denoiser's component
            // config assigns, or every rank of a group placed without one.
            let holders: Vec<usize> = info
                .component_config(worker, &component)
                .map_or_else(|| (0..ranks.len()).collect(), |config| config.ranks.clone());
            let reports: Option<Vec<(u32, crate::trajectories::RankLatents<'_>)>> = holders
                .iter()
                .map(|&rank| {
                    let (rank_host, pool) = ranks.get(rank)?;
                    Some((
                        rank as u32,
                        crate::trajectories::RankLatents {
                            host: rank_host,
                            pool: pool.as_ref()?,
                        },
                    ))
                })
                .collect();
            match reports {
                Some(reports) if !reports.is_empty() => denoisers.push((worker.clone(), reports)),
                _ => {}
            }
        }
        if denoisers.is_empty() {
            return;
        }
        let mut store =
            match crate::trajectories::TrajectoryStore::connect(socket, deployment, epoch, host) {
                Ok(store) => store,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "preempted trajectories are not stored: the storage server is unreachable"
                    );
                    return;
                }
            };
        let mut served = false;
        let mut identities = Vec::new();
        for (worker, reports) in &denoisers {
            identities.extend(reports.iter().map(|(_, rank)| rank.pool.identity.clone()));
            served |= store.add_worker(worker, reports);
        }
        if served {
            store.reseed(crate::trajectories::latent_seed(
                identities.iter().map(String::as_str),
            ));
            self.trajectories = Some(store);
        }
    }

    /// Stores the outputs of the video encoders in the host encoder pools of
    /// their worker groups, reached through the Storage Server at `socket`
    /// for the ranks on `host` (the engine's) and over the TCP address
    /// another host's rank reports, for each group whose ranks executing an
    /// encoder report the same pool (`WorkerInfo::rank_encoders`), as a
    /// head of `deployment` in `epoch`. Without them every request's
    /// encoders compute its outputs.
    pub fn enable_encoder_store(
        &mut self,
        socket: &std::path::Path,
        deployment: &str,
        epoch: u64,
        host: &str,
    ) {
        let info = self.worker.info();
        let groups: Vec<(
            crate::WorkerGroupId,
            Vec<(u32, crate::encoder_outputs::RankEncoders<'_>)>,
        )> = info
            .rank_encoders
            .iter()
            .map(|(worker, ranks)| {
                let reports = ranks
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, (rank_host, pool))| {
                        Some((
                            rank as u32,
                            crate::encoder_outputs::RankEncoders {
                                local: rank_host == host,
                                pool: pool.as_ref()?,
                            },
                        ))
                    })
                    .collect::<Vec<_>>();
                (worker.clone(), reports)
            })
            .filter(|(_, reports)| !reports.is_empty())
            .collect();
        if groups.is_empty() {
            return;
        }
        let mut store =
            match crate::encoder_outputs::EncoderStore::connect(socket, deployment, epoch) {
                Ok(store) => store,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "video encoder outputs are not stored: the storage server is unreachable"
                    );
                    return;
                }
            };
        let mut served = false;
        for (worker, reports) in &groups {
            served |= store.add_worker(worker, reports);
        }
        if served {
            self.encoder_outputs = Some(store);
        }
    }

    /// The latent seed of what the scheduler's denoising computes, which
    /// keys the trajectories it stores; zero when it stores none.
    pub fn latent_seed(&self) -> u64 {
        self.trajectories
            .as_ref()
            .map_or(0, crate::trajectories::TrajectoryStore::seed)
    }

    /// Returns the worker's aggregate worker capabilities.
    pub(crate) fn info(&self) -> &MainProcessInfo {
        &self.info
    }

    /// Returns the first worker group with a prefix cache, whose units the
    /// pool allocates and whose identity names its index for a successor
    /// epoch.
    pub(crate) fn cache_group(&self) -> Option<String> {
        self.worker
            .info()
            .workers
            .iter()
            .find(|(_, info)| info.prefix_cache.is_some())
            .map(|(group, _)| group.to_string())
    }

    /// The worker groups that keep a paged cache, each with its worker's
    /// endpoint: the groups a prefix cache index names.
    pub(crate) fn cache_workers(&self) -> Vec<(String, ipc::MainProcessEndpoint)> {
        self.worker
            .info()
            .workers
            .iter()
            .filter(|(_, info)| info.prefix_cache.is_some())
            .map(|(group, info)| (group.to_string(), info.endpoint.clone()))
            .collect()
    }

    /// Returns the model limits supported by the loaded execution workers.
    pub fn generation_limits(&self) -> &crate::GenerationLimits {
        &self.generation_limits
    }

    /// Returns the media units one call of each routed media call covers at
    /// once (`WorkerInfo::media_units`).
    ///
    /// # Errors
    ///
    /// Fails when the worker's media routing is inconsistent.
    pub(crate) fn media_units(
        &self,
    ) -> anyhow::Result<std::collections::BTreeMap<ipc::MediaCall, u32>> {
        let info = self.worker.info();
        Ok(info
            .media_routing()?
            .into_keys()
            .filter_map(|call| info.media_units(call).map(|units| (call, units)))
            .collect())
    }

    /// Returns a shared handle to scheduler counters.
    pub fn stats_handle(&self) -> Arc<SchedulerStats> {
        Arc::clone(&self.stats)
    }

    /// Puts the scheduler's epoch on trial until its commit
    /// (`Scheduler::trial`): a successor epoch's schedulers start on trial.
    pub fn set_trial(&mut self, trial: bool) {
        self.trial = trial;
    }

    /// Whether the scheduler's epoch is on trial.
    pub(crate) fn on_trial(&self) -> bool {
        self.trial
    }

    /// The first error of the scheduler's fatal stop, once it stopped on
    /// one; shared with whoever reports it.
    pub fn failure_handle(&self) -> Arc<std::sync::OnceLock<String>> {
        Arc::clone(&self.failure)
    }

    /// Returns shared access to an admitted video request's state.
    pub(super) fn media_state(&self, id: RequestId) -> Option<&VideoState> {
        self.requests.video(&id)
    }

    /// Returns mutable access to an admitted video request's state.
    pub(super) fn media_state_mut(&mut self, id: RequestId) -> Option<&mut VideoState> {
        self.requests.video_mut(&id)
    }

    /// Returns the admitted video requests' identifiers, in id order.
    pub(super) fn media_ids(&self) -> Vec<RequestId> {
        self.requests.running_video().map(|(id, _, _)| id).collect()
    }

    /// Returns the number of running requests.
    pub(super) fn running_request_count(&self) -> usize {
        self.running.len()
    }

    /// Returns the number of requests awaiting admission, the ones held for
    /// their predecessor's index included.
    pub(super) fn pending_request_count(&self) -> usize {
        self.waiting.len().saturating_add(self.held_for_index.len())
    }

    /// All drivers consume the same command ingress before scheduling work.
    ///
    /// Applies every queued command without blocking. Answers whether the
    /// engine must stop — a `Shutdown` command arrived or every sender has
    /// disconnected — and whether any command was applied, which is progress a
    /// tick made even when it formed no batch: a submission that only queues a
    /// request still changed the epoch a step is planned from.
    pub(super) fn take_commands_from(&mut self, commands: &Receiver<Command>) -> (bool, bool) {
        let mut handled = false;
        loop {
            match commands.try_recv() {
                Ok(command) => {
                    handled = true;
                    if self.handle_command(command) {
                        return (true, handled);
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => return (false, handled),
                Err(crossbeam_channel::TryRecvError::Disconnected) => return (true, handled),
            }
        }
    }

    /// Runs the step loop on the calling thread, which owns every piece of the
    /// scheduler's state from here on:
    ///
    /// 1. take what arrived: new requests, cancellations and the control
    ///    thread's other commands, and one completed batch;
    /// 2. `update` with the completed batch: advance its requests, publish
    ///    their pages, emit their events;
    /// 3. `schedule`: assemble the batches that fit now;
    /// 4. submit them to the executors, and block until something arrives.
    ///
    /// Returns `true` if the engine died (a device or a scheduler invariant
    /// failed) rather than shutting down on a `Shutdown` command or a closed
    /// command channel. Either way, every waiting or running request is
    /// finished with `Aborted`, and every receiver still open can read its
    /// complete event stream through its terminal event.
    pub fn run(mut self, rx: Receiver<Command>) -> bool {
        loop {
            let (stop, handled) = self.take_commands_from(&rx);
            if stop {
                break;
            }

            if handled {
                self.progress.wake();
            }
            let progressed = self.step() || handled;
            // An update's shadow pass is answered once the step that ran it has
            // ended (§7.3 step 2): the answer never claims a comparison the
            // step has not finished.
            self.settle_shadow();
            if !self.failed() && !progressed {
                self.wait(None);
            }
            if self.failed() {
                let _ = self.failure.set("the worker or a worker failed".to_owned());
                tracing::error!(
                    failure = self.failure.get().map_or("", String::as_str),
                    "engine fatal: stopping the control loop"
                );
                if self.trial {
                    self.hand_back_requests();
                }
                self.abort_all_requests();
                return true;
            }
        }
        // Finish everything still queued or running with `Aborted` before
        // closing the worker, so every open consumer receives a terminal
        // event rather than only a closed channel. The pages published so
        // far reach the host KV cache first.
        self.flush_offloads();
        self.abort_all_requests();
        false
    }

    /// Offload to the host KV cache every page reserved for it that is not
    /// there yet, before the workers close: let the batches in flight
    /// complete, which carry offloads and settle them, then carry each worker
    /// group's remaining reserved offloads in a batch of page offloads alone,
    /// and wait for their commits. The scheduler admits and schedules nothing more.
    /// Gives up after `PAGE_FLUSH_TIMEOUT` or when the engine fails, leaving
    /// the pages still on their way out to the host KV cache.
    pub(super) fn flush_offloads(&mut self) {
        if self.kv_offload.is_none() || self.failed() {
            return;
        }
        let draining = std::mem::replace(&mut self.draining, true);
        self.flush_pages();
        self.draining = draining;
    }

    /// The loop of [`Self::flush_offloads`], while it drains.
    fn flush_pages(&mut self) {
        let deadline = Instant::now() + PAGE_FLUSH_TIMEOUT;
        loop {
            let idle = self.inflight.pending_batches.is_empty()
                && self.inflight.pending_submissions.is_empty();
            let Some(kv_offload) = self.kv_offload.as_mut() else {
                return;
            };
            kv_offload.poll();
            if idle {
                for worker in kv_offload.queued_workers() {
                    let mut batch = self.finish_generation_batch(
                        ExecutionBatch::new(0, Vec::new(), Vec::new()),
                        Instant::now(),
                    );
                    let Some(kv_offload) = self.kv_offload.as_mut() else {
                        return;
                    };
                    let pages = kv_offload.take_offloads(&worker, batch.id);
                    batch.page_offloads = Some(crate::worker::PageOffloads {
                        worker,
                        pages,
                        latents: Vec::new(),
                    });
                    self.inflight.pending_submissions.push_back(batch);
                }
            }
            if !self
                .kv_offload
                .as_ref()
                .is_some_and(crate::kv_cache::KvOffloadManager::offloading)
            {
                return;
            }
            if self.failed() || Instant::now() >= deadline {
                tracing::warn!(
                    "published pages did not all reach the host KV cache before the workers close"
                );
                return;
            }
            if !self.step() {
                self.wait(Some(deadline));
            }
        }
    }

    /// Performs the storage `operation` on every worker once the batches in flight complete,
    /// admitting and scheduling nothing meanwhile, and returns each rank's typed reply.
    pub(super) fn run_operation<O: ipc::memory::Operation>(
        &mut self,
        operation: O,
    ) -> Result<Vec<O::Reply>, String> {
        let operation: EpochOperation = operation.into();
        let draining = std::mem::replace(&mut self.draining, true);
        let drained = self.drain_batches();
        self.draining = draining;
        drained?;
        let answers = self.operate_workers(&operation)?;
        ipc::memory::replies::<O>(&answers)
            .map_err(|error| format!("a worker's reply to {operation:?}: {error}"))
    }

    /// Drain all participating schedulers before an operation, and keep them
    /// drained until every participant completes it or reports its failure.
    pub(super) fn run_together(
        &mut self,
        together: &crate::HookBarrier,
        operation: impl FnOnce(&mut Self) -> Result<Vec<String>, String>,
    ) -> Result<Vec<String>, String> {
        let draining = std::mem::replace(&mut self.draining, true);
        let ran = self
            .drain_batches()
            .inspect_err(|error| together.fail(error))
            .and_then(|()| together.meet())
            .and_then(|()| {
                let results = operation(self).inspect_err(|error| together.fail(error));
                // Every scheduler ran the hook, or failed it, before any
                // schedules again; a party's failure here releases the
                // others, whose hooks ran.
                together.meet()?;
                results
            });
        self.draining = draining;
        ran
    }

    /// Completes the batches in flight, scheduling nothing new while the
    /// caller set `draining`.
    ///
    /// The drain is unbounded in what it may wait for and bounded only in
    /// silence: a batch the worker never answers, and never finishes,
    /// would otherwise hold the hook the drain serves for ever. Whenever a
    /// batch or a submission leaves the pending set the drain has made
    /// progress and its silence clock restarts, so a scheduler that keeps
    /// receiving results is never cut off, however long the drain takes
    /// under load. A drain that goes `DRAIN_SILENCE` without one abandons the
    /// batches and submissions it still holds, naming them.
    pub(super) fn drain_batches(&mut self) -> Result<(), String> {
        let mut silent_since = Instant::now();
        let mut held = self.drain_pending();
        while held.0 != 0 || held.1 != 0 {
            if self.failed() {
                return Err("the engine failed before the hook ran".to_owned());
            }
            if !self.step() {
                self.wait(Some(silent_since + DRAIN_SILENCE));
            }
            // Comparing the two sizes is enough because a drain sets
            // `draining`: nothing enters either pending set while it runs, so
            // the sizes can only fall. A step that scheduled a batch during a
            // drain could leave the sizes unchanged while the set changed
            // underneath them, so a change that lets a drain schedule must
            // re-read what the sets hold rather than how much they hold.
            let pending = self.drain_pending();
            if pending != held {
                held = pending;
                silent_since = Instant::now();
                continue;
            }
            if silent_since.elapsed() >= DRAIN_SILENCE {
                return Err(abandoned_drain(&self.inflight, held));
            }
        }
        Ok(())
    }

    /// What a drain still waits on: the batches the worker holds, and the
    /// batches it has not accepted.
    fn drain_pending(&self) -> (usize, usize) {
        (
            self.inflight.pending_batches.len(),
            self.inflight.pending_submissions.len(),
        )
    }

    /// Performs the storage `operation` on every worker, its batches complete.
    fn operate_workers(&mut self, operation: &EpochOperation) -> Result<Vec<String>, String> {
        let results = self
            .worker
            .operate(operation)
            .map_err(|error| format!("{error:#}"))?;
        tracing::info!(
            ?operation,
            workers = results.len(),
            "performed the storage operation"
        );
        Ok(results)
    }

    /// Blocks until something arrives: a command, a completed batch, a reader
    /// that freed room in an event channel, or an answer of a Storage Server.
    ///
    /// The wait has no tick. It ends without an arrival only at a moment the
    /// scheduler itself acts at: `bound`, which a caller that waits for
    /// something in particular states (a drain's silence, a flush's limit),
    /// the starvation bound of a running request, the end of the wait for a
    /// predecessor's index, or the host KV cache's own patience.
    pub(super) fn wait(&mut self, bound: Option<Instant>) {
        let _span = tracing::trace_span!("scheduler.wait").entered();
        let deadline = [
            bound,
            self.starvation_deadline(),
            self.awaiting_index
                .filter(|_| !self.held_for_index.is_empty()),
            self.kv_offload
                .as_ref()
                .and_then(crate::kv_cache::KvOffloadManager::deadline),
        ]
        .into_iter()
        .flatten()
        .min();
        // The Storage Servers answer on connections of their own: the host KV
        // cache's, the host latent and encoder pools' and the transfers'.
        let connections: Vec<std::os::fd::BorrowedFd<'_>> = self
            .kv_offload
            .iter()
            .flat_map(crate::kv_cache::KvOffloadManager::descriptors)
            .chain(
                self.trajectories
                    .iter()
                    .flat_map(crate::trajectories::TrajectoryStore::descriptors),
            )
            .chain(
                self.encoder_outputs
                    .iter()
                    .flat_map(crate::encoder_outputs::EncoderStore::descriptors),
            )
            .chain(
                self.hand_offs
                    .iter()
                    .flat_map(crate::hand_off::HandOffs::descriptors),
            )
            .collect();
        self.worker.wait(deadline, &connections);
    }

    /// When the first running request that the last step found within its
    /// starvation bound passes it, which is when a step must judge it.
    ///
    /// A request the last step already found past its bound was judged by it
    /// (it waits for a transfer, or it ended), so only later moments count,
    /// and never one less than a millisecond away: a step that could not
    /// judge (a full queue, a held scheduler) does not spin on it.
    fn starvation_deadline(&self) -> Option<Instant> {
        let timeout = self.policy.as_ref()?.limits().timeout;
        let now = crate::clock::now_monotonic_us().saturating_mul(1_000);
        let passes = self
            .watchdog
            .advancing()
            .map(|(_, since)| since.0.saturating_add(timeout.0))
            .filter(|passes| *passes > self.starvation_judged.0)
            .min()?;
        let remaining = Duration::from_nanos(passes.saturating_sub(now));
        Some(Instant::now() + remaining.max(Duration::from_millis(1)))
    }

    /// States what the step tells when it produced something (`progress`).
    pub(crate) fn set_progress(&mut self, progress: crate::CommandWaker) {
        self.progress = progress;
    }
}

/// Names what a drain that stopped progressing still holds: each batch with its
/// age and the requests it carries, and the submissions the worker never
/// accepted.
///
/// A batch the worker never answered and never finished is named by id, with
/// how long it has been in flight and which requests it carries, so a hook that
/// stopped can be diagnosed from its error alone.
fn abandoned_drain(inflight: &super::inflight::Inflight, held: (usize, usize)) -> String {
    let mut batches: Vec<_> = inflight.pending_batches.iter().collect();
    batches.sort_by_key(|(id, _)| **id);
    let batches: Vec<String> = batches
        .into_iter()
        .map(|(id, batch)| {
            let mut requests: Vec<_> = batch
                .calls
                .iter()
                .map(|(key, _)| key.request_id.0)
                .collect();
            requests.sort_unstable();
            requests.dedup();
            format!(
                "batch {id} ({:?} in flight, {} call(s), request(s) {requests:?})",
                batch.started.elapsed(),
                batch.calls.len()
            )
        })
        .collect();
    let queued: Vec<String> = inflight
        .pending_submissions
        .iter()
        .map(|batch| {
            let mut requests: Vec<_> = batch
                .requests
                .iter()
                .map(|(call, _)| call.request_key.request_id.0)
                .collect();
            requests.sort_unstable();
            requests.dedup();
            format!(
                "batch {} (never accepted, request(s) {requests:?})",
                batch.id
            )
        })
        .collect();
    format!(
        "the drain stopped progressing with {} batch(es) and {} submission(s): {}{}",
        held.0,
        held.1,
        batches.join(", "),
        if queued.is_empty() {
            String::new()
        } else {
            format!("; queued: {}", queued.join(", "))
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inflight::{Inflight, PendingBatch};
    use crate::{CallId, RequestKey};

    /// A call identity of `request`'s, as a batch registers it on acceptance.
    fn key(request: u64) -> RequestKey {
        RequestKey {
            engine_id: 1,
            request_id: RequestId(request),
            request_epoch: 0,
        }
    }

    #[test]
    fn a_drain_that_stopped_names_the_batch_and_the_requests_it_abandons() {
        let mut inflight = Inflight::new();
        inflight.pending_batches.insert(
            7,
            PendingBatch {
                numbered: Vec::new(),
                started: Instant::now(),
                calls: [
                    (key(3), CallId::new(7, 0)),
                    (key(3), CallId::new(7, 1)),
                    (key(9), CallId::new(7, 2)),
                ]
                .into_iter()
                .collect(),
                commands: Vec::new(),
                worker_exec_us: 0,
                prefill: true,
                tokens: 0,
                measured: None,
            },
        );

        let message = abandoned_drain(&inflight, (1, 0));

        assert!(
            message.contains("1 batch(es) and 0 submission(s)"),
            "{message}"
        );
        assert!(message.contains("batch 7"), "{message}");
        assert!(message.contains("3 call(s)"), "{message}");
        // The requests a batch carries are named by id, deduplicated and in
        // order, so the reader knows which work the abandoned batch held.
        assert!(message.contains("request(s) [3, 9]"), "{message}");
    }

    #[test]
    fn a_drain_that_stopped_names_the_submission_the_worker_never_took() {
        let mut inflight = Inflight::new();
        inflight
            .pending_submissions
            .push_back(ExecutionBatch::new(11, Vec::new(), Vec::new()));

        let message = abandoned_drain(&inflight, (0, 1));

        assert!(
            message.contains("0 batch(es) and 1 submission(s)"),
            "{message}"
        );
        assert!(message.contains("batch 11 (never accepted"), "{message}");
    }
}

//! Engine-loop construction, event parking, and owner-thread execution.
//!
//! Construction clamps the requested scheduler configuration and model
//! generation limits to the capacities the loaded workers report, then builds
//! the single-owner `Scheduler`. `Scheduler::run` is the owner-thread loop:
//! drain commands, take one nonblocking scheduling step, and park in the
//! executor only when the step made no progress.

use super::*;
use ipc::{ForwardMode, MediaCall};

/// Applies the loaded workers' served features and capacities to the model's
/// sizing facts.
///
/// The workers state which generation features are served: a feature is
/// available when every call kind it needs is in the aggregate
/// `supported_calls`, whatever the model description lists. Numeric limits
/// are clamped to the reported latent, batch-token, encoder-entry,
/// buffer-pool, and encoder-cache capacities.
fn resolve_generation_limits(
    mut limits: crate::GenerationLimits,
    info: &WorkerInfo,
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
    if supports(CallKind::Media(MediaCall::LatentPreparation))
        && supports(CallKind::Media(MediaCall::Denoising))
        && supports(CallKind::Media(MediaCall::ImageDecoding))
    {
        available.insert(crate::GenerationFeatures::IMAGE_GENERATION);
    }
    // A canvas denoises over a prompt the same worker's prefill cached.
    if supports(CallKind::Forward(ForwardMode::Prefill))
        && supports(CallKind::Forward(ForwardMode::TokenDenoising))
    {
        available.insert(crate::GenerationFeatures::TOKEN_DENOISING);
    }
    limits.features = available;

    limits.max_latent_units = limits.max_latent_units.min(info.latent_capacity_units());
    let latent_bound = limits.max_latent_units.min(u64::from(u32::MAX)) as u32;
    limits.max_vae_grid_tokens = limits.max_vae_grid_tokens.min(latent_bound);
    limits.max_vit_grid_tokens = limits.max_vit_grid_tokens.min(info.max_batch_tokens);
    // One encoder product must fit both an encoder-cache entry and the buffer
    // pool.
    let feature_bytes = info.encoder_entry_bytes.min(info.buffer_pool_bytes);
    limits.max_latent_feature_bytes = limits.max_latent_feature_bytes.min(feature_bytes);
    limits.max_vision_feature_bytes = limits.max_vision_feature_bytes.min(feature_bytes);
    limits.encoder_cache_entries = limits.encoder_cache_entries.min(info.encoder_cache_entries);
    // Encoder caching requires a buffer pool.
    if info.buffer_pool_bytes == 0 {
        limits.encoder_cache_entries = 0;
    }
    limits
}

/// Largest number of request-pool rows a runtime keeps outside
/// `max_running_requests` (`guidance_prefix_rows`). A launch sizes worker request pools
/// before the loaded capabilities decide the actual reserve.
pub(crate) const MAX_GUIDANCE_PREFIX_ROWS: usize = 2;

/// Request-pool rows the runtime keeps outside `max_running_requests`.
///
/// A cache runtime with denoising keeps a row for a running request's
/// text-unconditional guidance prefix, and another for its
/// image-unconditional prefix when the worker states one for its model
/// (`WorkerInfo::image_unconditional_token_ids`): each prefix
/// (`Scheduler::ensure_guidance_prefix`) allocates its own row from the same
/// request pool. Other runtimes keep none.
pub(crate) fn guidance_prefix_rows(info: &WorkerInfo) -> usize {
    let denoises = info.uses_cache()
        && info
            .supported_calls
            .contains(&CallKind::Media(MediaCall::Denoising));
    if !denoises {
        return 0;
    }
    1 + usize::from(info.image_unconditional_token_ids.is_some())
}

impl Scheduler {
    /// Constructs an engine loop from explicit scheduler and model capabilities.
    ///
    /// Worker limits clamp scheduler concurrency and resource capacities.
    ///
    /// Fails when the executor exposes an invalid aggregate capacity view.
    pub fn new(
        executor: Box<dyn Executor>,
        special_tokens: SpecialTokenIds,
        mut config: SchedulerConfig,
        model_dtype: crate::ModelDtype,
        generation_limits: crate::GenerationLimits,
    ) -> anyhow::Result<Self> {
        let info = executor.info().runtime_info()?;
        let generation_limits = resolve_generation_limits(generation_limits, &info);
        let latent_dtype = worker_float_dtype(Some(model_dtype));
        let media_storage = executor
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
                let components = executor
                    .info()
                    .placements
                    .get(id)
                    .map(|placement| placement.components.clone())
                    .unwrap_or_default();
                (
                    id.clone(),
                    MediaStorage::new(
                        worker.request_slots as usize,
                        &executor.info().rank_buffer_bytes(id),
                        &components,
                        &info.media_components,
                    ),
                )
            })
            .collect();

        // Queue and batch limits cannot exceed the physical executor envelope.
        let max_batch_calls = info.max_batch_calls as usize;
        let max_batch_tokens = info.max_batch_tokens as usize;
        // Each worker reports the transfer entries its transport holds in
        // flight; the merged record carries the narrowest.
        let transfer_capacity = info.transfer_tickets as usize;

        config.max_queued_requests = config.max_queued_requests.map(|bound| bound.max(1));

        // The request pool also holds the rows the runtime keeps outside
        // `max_running_requests` (`guidance_prefix_rows`). `SchedulerHandle::start_all` launches
        // workers with enough rows and refuses a token runtime whose workers
        // hold fewer, so this clamp lowers the limit only for a supplied
        // executor or for workers that fit their request pools to device
        // storage.
        let request_pool_capacity = info.request_slots as usize;
        let main_request_capacity = request_pool_capacity
            .saturating_sub(guidance_prefix_rows(&info))
            .max(1);

        let requested_seqs = config.max_running_requests.max(1);
        if requested_seqs > main_request_capacity {
            tracing::warn!(
                requested = requested_seqs,
                request_rows = info.request_slots,
                running = main_request_capacity,
                "the workers' request rows lower the running-request limit"
            );
        }
        config.max_running_requests = requested_seqs.min(main_request_capacity);
        config.max_num_batched_tokens = config.max_num_batched_tokens.max(1).min(max_batch_tokens);
        if max_batch_calls > 0 {
            config.max_batch = config.max_batch.min(max_batch_calls.max(1));
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
        let cache = PrefixCacheManager::from_worker_info(&info);
        let stats = Arc::new(SchedulerStats::default());
        stats.prefix_cache.num_units.store(
            cache.as_ref().map_or(0, PrefixCacheManager::usable_units),
            Ordering::Relaxed,
        );

        // Runtime state remains single-owner; executors receive immutable batch
        // inputs assembled from these queues and request records.
        Ok(Self {
            executor,
            inflight: inflight::Inflight::new(),
            placement: placement::Placement::default(),
            storage: allocation::Storage {
                cache,
                encoder_cache: crate::kv_cache::EncoderCacheManager::new(
                    generation_limits.encoder_cache_entries as usize,
                ),
                reserved_encoder_entries: 0,
                request_pool: RequestPool::new(info.request_slots as usize),
                latent_pool: LatentPool::new(info.latent_pages, info.latent_page_units),
                buffer_pool: BufferPool::new(info.buffer_pool_bytes),
                media_storage,
                encoder_buffers: HashMap::new(),
                pending_buffer_frees: HashMap::new(),
            },
            info,
            generation_limits,
            special_tokens,
            waiting: VecDeque::new(),
            waiting_media: VecDeque::new(),
            running: HashMap::new(),
            running_media: HashMap::new(),
            speculation_meter: Default::default(),
            retiring_requests: HashMap::new(),
            transfer_capacity,
            latent_dtype,
            engine_id: 1,
            next_product_generation: 1,
            next_request_epoch: 1,
            running_order: Vec::new(),
            output: output::OutputSender::default(),
            prefer_media: true,
            config,
            fatal: false,
            kv_offload: None,
            state_interval: crate::kv_cache::DEFAULT_STATE_INTERVAL,
            trajectories: None,
            closing: false,
            resizing: None,
            asleep: false,
            parking: None,
            parked: false,
            trial: false,
            failure: Arc::default(),
            stats,
        })
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
        let running = (info.request_slots as usize)
            .saturating_sub(guidance_prefix_rows(info))
            .max(1);
        let beyond = |name: &str, value: usize, capacity: usize| {
            format!(
                "scheduler_config.{name} = {value} exceeds the {capacity} the workers were \
                 started for; a world change starts workers for it"
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
        if calls > 0 && limits.max_batch > calls {
            return Err(beyond("max_batch", limits.max_batch, calls));
        }
        if let Some(canvas) = self.config.canvas
            && canvas.canvas_length as usize > limits.max_num_batched_tokens.max(1)
        {
            return Err(format!(
                "a {}-token generation canvas exceeds the {}-token step budget",
                canvas.canvas_length, limits.max_num_batched_tokens
            ));
        }
        if limits.policy != self.config.policy {
            let mut waiting = Vec::from(std::mem::take(&mut self.waiting));
            match limits.policy {
                SchedulingPolicy::Fcfs => {
                    waiting.sort_by(|a, b| a.queued_at.total_cmp(&b.queued_at))
                }
                SchedulingPolicy::Priority => waiting.sort_by(|a, b| {
                    a.req
                        .priority
                        .cmp(&b.req.priority)
                        .then(a.queued_at.total_cmp(&b.queued_at))
                }),
            }
            self.waiting = waiting.into();
        }
        self.config = SchedulerConfig {
            policy: limits.policy,
            max_batch: limits.max_batch.max(1),
            max_num_batched_tokens: limits.max_num_batched_tokens.max(1),
            max_running_requests: limits.max_running_requests.max(1),
            chunked_prefill_size: limits.chunked_prefill_size.max(1),
            max_queued_requests: limits.max_queued_requests.map(|bound| bound.max(1)),
            canvas: self.config.canvas,
            // The draft limit is the world's; the refill policy is a
            // scheduling choice.
            draft_tokens: self.config.draft_tokens,
            refill: limits.refill,
        };
        tracing::info!(config = ?self.config, "the scheduler took new limits");
        Ok(())
    }

    /// The worker groups that prefill only (`WorkerGroupRole::Prefill`),
    /// whose requests hand their KV cache to a decode group.
    pub fn set_prefill_groups(&mut self, groups: impl IntoIterator<Item = crate::WorkerGroupId>) {
        self.placement.prefill_groups = groups.into_iter().collect();
    }

    /// Enables or disables reusable prefix caching.
    pub fn set_prefix_reuse(&mut self, on: bool) {
        if let Some(cache) = self.storage.cache.as_mut() {
            cache.set_prefix_reuse(on);
        }
    }

    /// Checkpoints the state groups of decoding requests at the page
    /// boundaries that are multiples of `tokens`; a prompt call checkpoints
    /// the last boundary it reaches whatever the interval.
    pub fn set_state_interval(&mut self, tokens: usize) {
        self.state_interval = tokens.max(1);
    }

    /// Copy prompt pages of the worker groups whose ranks all map a page
    /// pool for every cache group through the host KV cache, as a head of
    /// `world` of `deployment` on `host` (what a world on trial commits is
    /// its own until its commit): ranks on `host` through its Storage Server's
    /// socket, the others through their servers' TCP addresses. Sliding
    /// windows are offloaded every `boundary_interval` pages of a long
    /// prompt, and `kv_storage` states whether a KV storage keeps every
    /// page the pools commit (`kv_cache_config.mooncake`). The world's host
    /// KV cache holds one token count on every rank, of `size`
    /// (`crate::kv_cache::HostCacheDemand`): its device tokens times the
    /// ratio, or what the stated bytes of each host hold, which every rank
    /// maps its pools at (`uniserve.host_cache`). A failure to reach this
    /// host's server leaves the host KV cache off.
    ///
    /// # Errors
    ///
    /// Fails when another host's server cannot size the cache, or a rank
    /// cannot map its pools.
    pub fn enable_kv_offload(
        &mut self,
        deployment: &str,
        world: u64,
        host: &str,
        boundary_interval: usize,
        size: crate::kv_cache::HostCacheSize,
        kv_storage: bool,
    ) -> anyhow::Result<()> {
        let socket = storage::socket::default_socket();
        let mut kv_offload = match crate::kv_cache::KvOffloadManager::connect(
            &socket,
            deployment,
            world,
            boundary_interval,
            kv_storage,
        ) {
            Ok(kv_offload) => kv_offload,
            Err(error) => {
                tracing::warn!(%error, "the host KV cache is off: the storage server is unreachable");
                return Ok(());
            }
        };
        let mut rank_caches = self.executor.info().rank_caches.clone();
        let Some(demand) = crate::kv_cache::HostCacheDemand::of(&rank_caches, size.ratio) else {
            return Ok(());
        };
        // The world asks for its device tokens times the ratio, or, when
        // the deployment states each host's bytes, for whatever those hold;
        // each host lowers the count for the whole world to what its budget
        // holds over the world's ranks there.
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
        tracing::info!(asked, tokens, "sized the world's host KV cache");
        let payload = serde_json::json!({ "tokens": tokens }).to_string();
        for (worker, results) in self.executor.hook_groups("uniserve.host_cache", &payload)? {
            let Some(ranks) = rank_caches.get_mut(&worker) else {
                continue;
            };
            for ((_, cache), result) in ranks.iter_mut().zip(results) {
                let pools: Option<Vec<ipc::PagePoolInfo>> = serde_json::from_str(&result)?;
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

    /// Stores the trajectories of preempted video requests in the host
    /// latent pools of the denoiser's worker group, reached through the
    /// Storage Server at `socket` for the ranks on `host` (the engine's)
    /// and over the TCP address another host's rank reports, when every
    /// rank that executes the denoiser reports one
    /// (`ExecutorInfo::rank_latents`), as a head of `deployment` in
    /// `world`, whose trajectories the servers keep its own while it is on
    /// trial. Without them a preempted request continues from noise
    /// wherever it resumes.
    pub fn enable_trajectory_store(
        &mut self,
        socket: &std::path::Path,
        deployment: &str,
        world: u64,
        host: &str,
    ) {
        let Some(component) = self
            .info
            .media_components
            .get(&MediaCall::Denoising)
            .cloned()
        else {
            return;
        };
        let info = self.executor.info();
        let mut denoisers = Vec::new();
        for (worker, ranks) in &info.rank_latents {
            // The ranks that hold trajectories: those the denoiser's component
            // config assigns, or every rank of a group launched without one.
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
                            local: rank_host == host,
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
        let mut store = match crate::trajectories::TrajectoryStore::connect(
            socket, deployment, world,
        ) {
            Ok(store) => store,
            Err(error) => {
                tracing::warn!(
                    %error,
                    "preempted video trajectories are not stored: the storage server is unreachable"
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

    /// The latent seed of what the scheduler's video denoising computes,
    /// which keys the trajectories it stores; zero when it stores none.
    pub fn latent_seed(&self) -> u64 {
        self.trajectories
            .as_ref()
            .map_or(0, crate::trajectories::TrajectoryStore::seed)
    }

    /// Returns the executor's aggregate worker capabilities.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Returns the first worker group with a prefix cache, whose units the
    /// pool allocates and whose identity names its index for a successor
    /// world.
    pub fn cache_group(&self) -> Option<String> {
        self.executor
            .info()
            .workers
            .iter()
            .find(|(_, info)| info.prefix_cache.is_some())
            .map(|(group, _)| group.to_string())
    }

    /// Returns the model limits supported by the loaded execution workers.
    pub fn generation_limits(&self) -> &crate::GenerationLimits {
        &self.generation_limits
    }

    /// Returns the media units one call of each routed media call covers at
    /// once (`ExecutorInfo::media_units`).
    ///
    /// # Errors
    ///
    /// Fails when the executor's media routing is inconsistent.
    pub fn media_units(&self) -> anyhow::Result<std::collections::BTreeMap<ipc::MediaCall, u32>> {
        let info = self.executor.info();
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

    /// Puts the scheduler's world on trial until its commit
    /// (`Scheduler::trial`): a successor world's schedulers start on trial.
    pub fn set_trial(&mut self, trial: bool) {
        self.trial = trial;
    }

    /// Whether the scheduler's world is on trial.
    pub fn on_trial(&self) -> bool {
        self.trial
    }

    /// The first error of the scheduler's fatal stop, once it stopped on
    /// one; shared with whoever reports it.
    pub fn failure_handle(&self) -> Arc<std::sync::OnceLock<String>> {
        Arc::clone(&self.failure)
    }

    /// Returns shared access to a media request state.
    pub(super) fn media_state(&self, id: RequestId) -> Option<&MediaFlowState> {
        self.running_media.get(&id)
    }

    /// Returns mutable access to a media request state.
    pub(super) fn media_state_mut(&mut self, id: RequestId) -> Option<&mut MediaFlowState> {
        self.running_media.get_mut(&id)
    }

    /// Returns the active media request identifiers.
    pub(super) fn media_ids(&self) -> Vec<RequestId> {
        self.running_media.keys().copied().collect()
    }

    /// Removes and returns a media request's state.
    pub(super) fn take_media_state(&mut self, id: RequestId) -> Option<MediaFlowState> {
        self.running_media.remove(&id)
    }

    /// Returns the number of running requests.
    pub(super) fn running_request_count(&self) -> usize {
        self.running.len().saturating_add(self.running_media.len())
    }

    /// Returns the number of pending requests.
    pub(super) fn pending_request_count(&self) -> usize {
        self.waiting.len().saturating_add(self.waiting_media.len())
    }

    /// All drivers consume the same command ingress before scheduling work.
    ///
    /// Applies every queued command without blocking. Returns `true` when the
    /// engine must stop: a `Shutdown` command arrived or every sender has
    /// disconnected.
    pub(super) fn drain_commands(&mut self, commands: &Receiver<Command>) -> bool {
        loop {
            match commands.try_recv() {
                Ok(command) => {
                    if self.handle_command(command) {
                        return true;
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => return false,
                Err(crossbeam_channel::TryRecvError::Disconnected) => return true,
            }
        }
    }

    /// Runs the owner-thread control loop, blocking only when fully idle.
    ///
    /// Returns `true` if the engine died (executor or worker failure) rather
    /// than shutting down on a `Shutdown` command or command-channel
    /// disconnect. Either way, every queued or running request is finished
    /// with `Aborted`, every receiver still open can read its complete event
    /// stream through its terminal event, and the executor is closed before
    /// returning.
    pub fn run(mut self, rx: Receiver<Command>) -> bool {
        loop {
            if self.drain_commands(&rx) {
                break;
            }

            let progressed = self.step_nonblocking();
            if !self.fatal && !progressed {
                self.park_for_progress();
            }
            if self.fatal {
                let _ = self
                    .failure
                    .set("the executor or a worker failed".to_owned());
                tracing::error!(
                    failure = self.failure.get().map_or("", String::as_str),
                    "engine fatal: stopping the control loop"
                );
                if self.trial {
                    self.hand_back_requests();
                }
                self.abort_all_requests();
                let _ = self.executor.close();
                return true;
            }
        }
        // Finish everything still queued or running with `Aborted` before
        // closing the executor, so every open consumer receives a terminal
        // event rather than only a closed channel. The pages published so
        // far reach the host KV cache first.
        self.flush_offloads();
        self.abort_all_requests();
        let _ = self.executor.close();
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
        if self.kv_offload.is_none() || self.fatal {
            return;
        }
        self.closing = true;
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
                        ExecutionBatch::new(0, Vec::new(), Vec::new(), Vec::new()),
                        Instant::now(),
                    );
                    let Some(kv_offload) = self.kv_offload.as_mut() else {
                        return;
                    };
                    let pages = kv_offload.take_offloads(&worker, batch.id);
                    batch.page_offloads = Some(crate::executor::PageOffloads {
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
            if self.fatal || Instant::now() >= deadline {
                tracing::warn!(
                    "published pages did not all reach the host KV cache before the workers close"
                );
                return;
            }
            if !self.step_nonblocking() {
                self.park_for_progress();
            }
        }
    }

    /// Runs the worker hook `name` with `payload` on every worker
    /// (`Executor::hook`) once the batches in flight complete, admitting
    /// and scheduling nothing meanwhile, and returns each worker's result.
    pub(super) fn run_hook(&mut self, name: &str, payload: &str) -> Result<Vec<String>, String> {
        let closing = std::mem::replace(&mut self.closing, true);
        let drained = self.drain_batches();
        self.closing = closing;
        drained?;
        self.hook_workers(name, payload)
    }

    /// Runs the worker hook `name` as [`Self::run_hook`] does, together
    /// with the engine's other schedulers (`HookBarrier`): it runs once
    /// every scheduler's batches in flight completed, and this scheduler
    /// schedules again only once every one ran it. Meanwhile its ranks go on
    /// joining the union's steps of schedulers still completing theirs.
    pub(super) fn run_hook_together(
        &mut self,
        name: &str,
        payload: &str,
        together: &crate::HookBarrier,
    ) -> Result<Vec<String>, String> {
        let closing = std::mem::replace(&mut self.closing, true);
        let ran = self
            .drain_batches()
            .inspect_err(|error| together.fail(error))
            .and_then(|()| together.meet())
            .and_then(|()| {
                let results = self.hook_workers(name, payload);
                // Every scheduler ran the hook, or failed it, before any
                // schedules again; a party's failure here releases the
                // others, whose hooks ran.
                let _ = together.meet();
                results
            });
        self.closing = closing;
        ran
    }

    /// Completes the batches in flight, scheduling nothing new while the
    /// caller set `closing`.
    fn drain_batches(&mut self) -> Result<(), String> {
        while !(self.inflight.pending_batches.is_empty()
            && self.inflight.pending_submissions.is_empty())
        {
            if self.fatal {
                return Err("the engine failed before the hook ran".to_owned());
            }
            if !self.step_nonblocking() {
                self.park_for_progress();
            }
        }
        Ok(())
    }

    /// Runs the worker hook on every worker, its batches complete.
    fn hook_workers(&mut self, name: &str, payload: &str) -> Result<Vec<String>, String> {
        let results = self
            .executor
            .hook(name, payload)
            .map_err(|error| format!("{error:#}"))?;
        tracing::info!(hook = name, workers = results.len(), "ran the worker hook");
        Ok(results)
    }

    /// Parks until a result, command, worker death, CPU continuation, or output-capacity wake.
    ///
    /// Waits in `Executor::poll` and applies at most one returned result.
    /// Commands and consumer reads fire the engine's `CommandWaker`. A real
    /// waker, such as `WorkerExecutor::command_waker`, ends the poll early.
    /// With the no-op waker, a command waits for the executor's own wake
    /// sources or the timeout. The timeout (`IDLE_LIVENESS_POLL`) is solely a
    /// liveness deadline.
    pub(super) fn park_for_progress(&mut self) {
        let _span = tracing::trace_span!("scheduler.park").entered();
        // Answers of the host KV cache and of the host latent pools arrive on
        // connections the executor does not wait on: while some are awaited,
        // the park is short.
        // So is the park of a closing scheduler, which awaits its last
        // page commits.
        let timeout = if self.closing
            || self
                .kv_offload
                .as_ref()
                .is_some_and(crate::kv_cache::KvOffloadManager::waiting)
            || self
                .trajectories
                .as_ref()
                .is_some_and(crate::trajectories::TrajectoryStore::waiting)
        {
            KV_OFFLOAD_POLL
        } else {
            IDLE_LIVENESS_POLL
        };
        match self.executor.poll(timeout) {
            Ok(Some(result)) => self.apply_result(result),
            Ok(None) => {}
            Err(error) => self.on_executor_error(error),
        }
        self.stats
            .general
            .in_flight
            .store(self.inflight.pending_batches.len(), Ordering::Relaxed);
        self.publish_cache_stats();
    }
}

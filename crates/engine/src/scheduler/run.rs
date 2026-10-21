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
            closing: false,
            stats,
        })
    }

    /// Returns the effective scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Enables or disables reusable prefix caching.
    pub fn set_prefix_reuse(&mut self, on: bool) {
        if let Some(cache) = self.storage.cache.as_mut() {
            cache.set_prefix_reuse(on);
        }
    }

    /// Copy prompt pages of the worker groups whose ranks all map a page
    /// pool for every cache group through the host KV cache, as a head of
    /// `deployment` on `host`: ranks on `host` through its Storage Server's
    /// socket, the others through their servers' TCP addresses. Sliding
    /// windows are offloaded every `boundary_interval` pages of a long
    /// prompt. A failure to reach this host's server leaves the host KV cache
    /// off.
    pub fn enable_kv_offload(&mut self, deployment: &str, host: &str, boundary_interval: usize) {
        let socket = storage::socket::default_socket();
        let mut kv_offload = match crate::kv_cache::KvOffloadManager::connect(
            &socket,
            deployment,
            boundary_interval,
        ) {
            Ok(kv_offload) => kv_offload,
            Err(error) => {
                tracing::warn!(%error, "the host KV cache is off: the storage server is unreachable");
                return;
            }
        };
        let mut served = false;
        for (worker, ranks) in &self.executor.info().rank_caches {
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
    }

    /// Returns the executor's aggregate worker capabilities.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Returns the prefix cache units of each worker group with a prefix
    /// cache, by group identity: what an engine standing by for this one
    /// adopts.
    pub fn cache_units(&self) -> BTreeMap<String, u32> {
        self.executor
            .info()
            .workers
            .iter()
            .filter_map(|(group, info)| {
                info.prefix_cache
                    .as_ref()
                    .map(|cache| (group.to_string(), cache.num_units))
            })
            .collect()
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
            if self.fatal {
                tracing::error!("engine fatal: executor/worker died; stopping the control loop");
                self.abort_all_requests();
                let _ = self.executor.close();
                return true;
            }

            if !progressed {
                self.park_for_progress();
                if self.fatal {
                    tracing::error!(
                        "engine fatal: executor/worker died during park; stopping the control loop"
                    );
                    self.abort_all_requests();
                    let _ = self.executor.close();
                    return true;
                }
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
                    batch.page_offloads = Some(crate::executor::PageOffloads { worker, pages });
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
        // Answers of the host KV cache arrive on a connection the executor
        // does not wait on: while some are awaited, the park is short.
        // So is the park of a closing scheduler, which awaits its last
        // page commits.
        let timeout = if self.closing
            || self
                .kv_offload
                .as_ref()
                .is_some_and(crate::kv_cache::KvOffloadManager::waiting)
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

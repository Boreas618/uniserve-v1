//! The worker's execution backend: request state, lifecycle commands,
//! batch validation, numerical dispatch and result delivery.
//!
//! [`WorkerBackend`] is the single [`Backend`] and [`ServiceBackend`] of a
//! rank. It owns the request pool, the request storage owners (through
//! [`RequestStorage`]) and the placed components with their bindings, and
//! dispatches each batch's calls to the capability runtime that serves them
//! ([`super::forward`]). Lifecycle commands (Start, Finish, Free), batch
//! admission, dependencies, retirement and result delivery are handled
//! here for every batch, with or without calls.

use std::collections::BTreeSet;
use std::sync::Arc;

use uniserve_core::CallId;
use uniserve_ipc::{
    Batch as BatchPlan, BatchCommand, BatchOutput, CallKind, ErrorCallIdentity, RequestKind,
    WorkerInfo, WorkerResponseError,
};

use super::forward;
use super::forward::batch::MediaBatch;
use super::forward::video::{Parts, VideoRuntime};
use super::lifecycle::{RequestStorage, Retirement};
use super::{Backend, BatchResult, Submission};
use crate::error::WorkerError;
use crate::ffi::Function;
use crate::model::registry::Components;
use crate::profiling::kernels::KernelRecords;
use crate::profiling::profiler::Profiler;
use crate::request::RequestPool;
use crate::service::ServiceBackend;
use crate::worker::RankStorage;
use crate::{Error, Result};

/// Serving-time kernel reporting: a cheap counter tells when any provider
/// choice changed, and only then are the bindings' records gathered.
pub struct KernelWatch {
    pub choices: Function,
    pub last: i64,
    pub records: KernelRecords,
    pub rank: usize,
    pub device: String,
}

/// The backend of one rank.
pub struct WorkerBackend {
    info: WorkerInfo,
    rank: usize,
    requests: RequestPool,
    /// The rank's storage owners: events, transports, products and KV.
    rank_storage: Option<RankStorage>,
    /// The video runtime of a video model's rank.
    video: Option<VideoRuntime>,
    storage: Vec<Box<dyn RequestStorage>>,
    components: Components,
    profiler: Option<Profiler>,
    kernels: Option<KernelWatch>,
}

/// A submitted batch with its lifecycle and numerical state.
pub struct WorkerBatch {
    plan: Arc<BatchPlan>,
    /// The accepted state call each call follows, by call.
    predecessors: Vec<Option<CallId>>,
    retirement: Retirement,
    /// A media batch's inputs, calls and completion.
    media: Option<MediaBatch>,
    output: Option<BatchResult>,
}

impl WorkerBatch {
    pub fn new(plan: Arc<BatchPlan>) -> Self {
        let retirement = Retirement::new(&plan);
        Self {
            plan,
            predecessors: Vec::new(),
            retirement,
            media: None,
            output: None,
        }
    }

    /// The batch's plan and media state, borrowed apart.
    pub(crate) fn media_parts(&mut self) -> (&BatchPlan, &mut Option<MediaBatch>) {
        (&self.plan, &mut self.media)
    }

    pub fn plan(&self) -> &BatchPlan {
        &self.plan
    }

    /// The batch's call kind; `None` for a lifecycle-only batch.
    pub fn route(&self) -> Option<CallKind> {
        self.plan.calls.first().map(|call| call.code)
    }

    pub fn predecessors(&self) -> &[Option<CallId>] {
        &self.predecessors
    }

    /// The calls a failure of this batch affects.
    fn identities(&self) -> Vec<ErrorCallIdentity> {
        self.plan
            .calls
            .iter()
            .map(|call| ErrorCallIdentity {
                request_key: call.request_key,
                call_id: call.call_id,
            })
            .collect()
    }

    /// Set the batch's result; capability runtimes call this once their
    /// outputs are materialized.
    pub fn complete(&mut self, result: BatchResult) {
        self.output = Some(result);
    }
}

impl WorkerBackend {
    pub fn new(info: WorkerInfo, rank: usize, components: Components) -> Result<Self> {
        Ok(Self {
            requests: RequestPool::new(info.request_slots as usize)?,
            info,
            rank,
            rank_storage: None,
            video: None,
            storage: Vec::new(),
            components,
            profiler: None,
            kernels: None,
        })
    }

    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    pub fn requests(&self) -> &RequestPool {
        &self.requests
    }

    pub fn requests_mut(&mut self) -> &mut RequestPool {
        &mut self.requests
    }

    pub fn components(&self) -> &Components {
        &self.components
    }

    pub fn components_mut(&mut self) -> &mut Components {
        &mut self.components
    }

    /// Register a storage owner taking part in request lifecycles.
    pub fn add_storage(&mut self, storage: Box<dyn RequestStorage>) {
        self.storage.push(storage);
    }

    /// Install the rank's storage owners.
    pub fn set_rank_storage(&mut self, storage: RankStorage) {
        self.rank_storage = Some(storage);
    }

    pub fn rank_storage(&self) -> Option<&RankStorage> {
        self.rank_storage.as_ref()
    }

    /// Install the video runtime of a video model's rank.
    pub(crate) fn set_video(&mut self, video: VideoRuntime) {
        self.video = Some(video);
    }

    pub(crate) fn video_mut(&mut self) -> Option<&mut VideoRuntime> {
        self.video.as_mut()
    }

    /// The parts a media batch executes with.
    pub(crate) fn media_parts(&mut self) -> Result<Parts<'_>> {
        Ok(Parts {
            requests: &mut self.requests,
            storage: self
                .rank_storage
                .as_ref()
                .ok_or_else(|| Error::Invariant("the rank has no storage owners".into()))?,
            video: self.video.as_mut().ok_or_else(|| {
                Error::Unsupported("video calls are served by the video runtime".into())
            })?,
            info: &self.info,
        })
    }

    /// Every storage owner taking part in request lifecycles, rank storage
    /// first.
    fn owners<'a>(
        rank_storage: &'a mut Option<RankStorage>,
        video: &'a mut Option<VideoRuntime>,
        storage: &'a mut [Box<dyn RequestStorage>],
    ) -> Vec<&'a mut dyn RequestStorage> {
        let mut owners: Vec<&mut dyn RequestStorage> = Vec::new();
        if let Some(rank) = rank_storage.as_mut() {
            owners.push(rank);
        }
        if let Some(video) = video.as_mut() {
            owners.push(video);
        }
        owners.extend(
            storage
                .iter_mut()
                .map(|owner| owner.as_mut() as &mut dyn RequestStorage),
        );
        owners
    }

    pub fn set_profiler(&mut self, profiler: Option<Profiler>) {
        self.profiler = profiler;
    }

    pub fn profiler_mut(&mut self) -> Option<&mut Profiler> {
        self.profiler.as_mut()
    }

    /// Report kernel selections that change while serving.
    pub fn watch_kernels(&mut self, watch: KernelWatch) {
        self.kernels = Some(watch);
    }

    /// Whether the component of a batch belongs to a video model.
    pub fn video_model(&self, batch: &WorkerBatch) -> bool {
        batch
            .plan
            .calls
            .first()
            .and_then(|call| self.components.get(&call.component))
            .is_some_and(|placed| {
                forward::decodes_video(
                    placed
                        .calls
                        .iter()
                        .flat_map(|call| call.capabilities.iter().copied()),
                )
            })
    }

    /// Install every Start, then follow each call's predecessor.
    fn admit_requests(&mut self, batch: &mut WorkerBatch) -> Result<()> {
        let mut admitted = Vec::new();
        let mut installed = Vec::new();
        let started = (|| -> Result<()> {
            for command in &batch.plan.commands {
                if let BatchCommand::Start { request } = command
                    && let Some(slot) = self.requests.start(request.as_ref().clone())?
                {
                    admitted.push(slot);
                    installed.push(request.as_ref().clone());
                }
            }
            Ok(())
        })();
        // Reset every admitted slot even when a later Start failed.
        if !admitted.is_empty() {
            for owner in Self::owners(&mut self.rank_storage, &mut self.video, &mut self.storage) {
                owner.admit_slots(&admitted)?;
            }
        }
        started?;
        if let Some(video) = &mut self.video {
            for request in &installed {
                video.admit(request)?;
            }
        }

        batch.predecessors = batch
            .plan
            .calls
            .iter()
            .map(|call| {
                self.requests
                    .predecessor(call.request_key, call.advances_state())
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// Refuse a batch outside this rank's advertised capacity.
    fn validate(&self, plan: &BatchPlan) -> Result<()> {
        let unsupported: BTreeSet<&str> = plan
            .calls
            .iter()
            .filter(|call| !self.info.supported_calls.contains(&call.code))
            .map(|call| call.code.as_str())
            .collect();
        if !unsupported.is_empty() {
            return Err(Error::Invalid(format!(
                "execution batch contains call kinds unsupported by this worker: {:?}",
                unsupported.into_iter().collect::<Vec<_>>()
            )));
        }
        if let Some(cache) = &self.info.kv_cache
            && plan
                .new_cache_units
                .iter()
                .flat_map(|allocation| &allocation.unit_ids)
                .any(|unit| unit.0 >= cache.num_units)
        {
            return Err(Error::Invalid(
                "KV allocation exceeds the fixed physical pool".into(),
            ));
        }
        if let Some(allocation) = plan
            .buffer_allocations
            .iter()
            .find(|allocation| allocation.offset + allocation.bytes > self.info.buffer_pool_bytes)
        {
            return Err(Error::Invalid(format!(
                "batch buffer allocation exceeds the worker buffer pool: offset={}, bytes={}, capacity={}, buffer={:?}",
                allocation.offset, allocation.bytes, self.info.buffer_pool_bytes, allocation.buffer,
            )));
        }
        if !self.info.components.is_empty() {
            for call in &plan.calls {
                if !self.info.components.iter().any(|component| {
                    component.name == call.component && component.config.ranks.contains(&self.rank)
                }) {
                    return Err(Error::Invalid(format!(
                        "call targets component {:?} outside this rank",
                        call.component
                    )));
                }
            }
        }
        if plan.calls.len() > self.info.max_batch_calls as usize {
            return Err(Error::Invalid(
                "execution batch exceeds the worker_config call limit".into(),
            ));
        }
        if plan
            .block_tables
            .iter()
            .map(|table| table.request_pool_idx)
            .chain(plan.forward.request_pool_indices.iter().copied())
            .any(|slot| slot > self.info.request_slots)
        {
            return Err(Error::Invalid(
                "execution batch exceeds request-slot capacity".into(),
            ));
        }
        Ok(())
    }

    fn report_new_kernels(&mut self) {
        let Some(watch) = &mut self.kernels else {
            return;
        };
        let Ok(choices) = watch.choices.call_as::<i64>(&[]) else {
            return;
        };
        if choices == watch.last {
            return;
        }
        watch.last = choices;
        let mut current = KernelRecords::default();
        for binding in self.components.bindings() {
            let label = binding_label(binding);
            let added = binding
                .kernels()
                .and_then(|records| current.add(&label, &binding.device.to_string(), &records));
            if let Err(error) = added {
                tracing::warn!(target: "uniserve_worker.profiling", "kernel report failed: {error}");
                return;
            }
        }
        if watch.records.merge(current) {
            watch.records.log("serving", watch.rank, &watch.device);
        }
    }

    /// Install (or clear) the service's completion wake on every storage
    /// owner.
    pub fn set_storage_wake(&self, wake: Option<uniserve_ipc::Wake>) {
        if let Some(rank) = &self.rank_storage {
            rank.set_wake(wake.clone());
        }
        if let Some(video) = &self.video {
            video.set_wake(wake.clone());
        }
        for owner in &self.storage {
            owner.set_wake(wake.clone());
        }
    }

    /// Close every storage owner, newest first, after the executor drained;
    /// returns the first failure. The video runtime closes before the rank
    /// storage its products live in.
    pub fn close_storage(&mut self, aborted: bool) -> Result<()> {
        let mut result = Ok(());
        while let Some(mut owner) = self.storage.pop() {
            result = result.and(owner.close(aborted));
        }
        if let Some(mut video) = self.video.take() {
            result = result.and(video.close(aborted));
        }
        if let Some(mut rank) = self.rank_storage.take() {
            result = result.and(rank.close(aborted));
        }
        result
    }

    /// Release request references after the executor drained its work.
    pub fn close(&mut self) {
        self.requests.close();
        if let Some(profiler) = &mut self.profiler {
            profiler.close();
        }
    }
}

/// A binding's kernel-table label: component, root path and call kinds.
pub fn binding_label(binding: &crate::model::bind::ComponentBinding) -> String {
    if binding.path.is_empty() {
        binding.component.clone()
    } else {
        format!("{}.{}", binding.component, binding.path)
    }
}

impl Backend for WorkerBackend {
    type Batch = WorkerBatch;
    type Output = BatchResult;
    type Error = WorkerError;

    fn error(&self, error: Error) -> WorkerError {
        WorkerError::classify(error, None)
    }

    fn classify(&self, error: WorkerError, batch: &WorkerBatch, _context: &str) -> WorkerError {
        let error = error.with_calls(batch.identities());
        match batch.route() {
            Some(kind) => error.with_route(kind.as_str()),
            None => error,
        }
    }

    fn note_cleanup(&self, error: &mut WorkerError, cleanup: WorkerError) {
        error.note(&cleanup);
    }

    fn admit(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(), WorkerError> {
        // Explicit revocation must not wait behind an allocation-dependent
        // batch; existing readers retain their leases.
        let mut owners = Self::owners(&mut self.rank_storage, &mut self.video, &mut self.storage);
        batch.retirement.admit(&mut owners)?;
        Ok(())
    }

    fn prepare(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(), WorkerError> {
        self.validate_kinds(batch)?;
        self.admit_requests(batch)?;
        self.validate(&batch.plan)?;
        for command in &batch.plan.commands {
            if let BatchCommand::Finish { request_key, .. } = command {
                self.requests.finish(*request_key)?;
            }
        }
        Ok(())
    }

    fn prepare_inputs(
        &mut self,
        batch: &mut WorkerBatch,
        _submission: &Arc<Submission>,
    ) -> std::result::Result<bool, WorkerError> {
        // Media batches import their products before executing; lifecycle
        // batches read no inputs.
        if !forward::media_route(batch.route(), self.video.is_some()) {
            return Ok(true);
        }
        let (plan, media) = batch.media_parts();
        let mut parts = self.media_parts()?;
        Ok(forward::video::prepare_inputs(&mut parts, plan, media)?)
    }

    fn await_inputs(
        &mut self,
        batch: &mut WorkerBatch,
        submission: &Arc<Submission>,
    ) -> std::result::Result<(), WorkerError> {
        forward::video::await_inputs(&batch.media, submission);
        Ok(())
    }

    fn execute(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(), WorkerError> {
        let Some(kind) = batch.route() else {
            batch.output = Some(BatchResult {
                output: BatchOutput {
                    batch_id: batch.plan.batch_id,
                    completions: Vec::new(),
                    products: Vec::new(),
                    worker_exec_us: None,
                    forward_stats: None,
                },
                media: Vec::new(),
            });
            return Ok(());
        };

        // Each executed batch with calls is one profiler step.
        let mut profiler = self.profiler.take();
        let executed = match &mut profiler {
            Some(profiler) => profiler.step(|| forward::execute(self, batch, kind)),
            None => forward::execute(self, batch, kind),
        };
        self.profiler = profiler;
        self.report_new_kernels();
        executed
    }

    fn begin_retirement(
        &mut self,
        batch: &mut WorkerBatch,
    ) -> std::result::Result<(), WorkerError> {
        let mut owners = Self::owners(&mut self.rank_storage, &mut self.video, &mut self.storage);
        batch.retirement.begin(&self.requests, &mut owners)?;
        Ok(())
    }

    fn poll(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(bool, bool), WorkerError> {
        let mut advanced = false;
        if batch.output.is_none()
            && let Some(media) = &mut batch.media
        {
            let parts = self.media_parts()?;
            let owners = parts.owners()?;
            if let Some(result) = media.poll(&batch.plan, owners, parts.requests)? {
                batch.output = Some(result);
                advanced = true;
            }
        }
        if batch.output.is_none() {
            return Ok((advanced, false));
        }
        let mut owners = Self::owners(&mut self.rank_storage, &mut self.video, &mut self.storage);
        let retired = batch.retirement.poll(&mut self.requests, &mut owners)?;
        Ok((retired || advanced, retired))
    }

    fn result(&mut self, batch: &mut WorkerBatch) -> std::result::Result<BatchResult, WorkerError> {
        let result = batch
            .output
            .as_mut()
            .ok_or_else(|| WorkerError::from(Error::State("batch output is not resolved")))?;
        Ok(BatchResult {
            output: result.output.clone(),
            media: std::mem::take(&mut result.media),
        })
    }

    fn close(&mut self, batch: &mut WorkerBatch) -> std::result::Result<(), WorkerError> {
        // Calls become pending only when a capability runtime commits them,
        // and that runtime cancels what it committed. Lifecycle commands
        // hold no resource beyond their retirement.
        if let Some(media) = &mut batch.media {
            let parts = self.media_parts()?;
            let owners = parts.owners()?;
            media.close(owners, parts.requests)?;
        }
        Ok(())
    }

    fn reap(&mut self) -> std::result::Result<(), WorkerError> {
        for owner in Self::owners(&mut self.rank_storage, &mut self.video, &mut self.storage) {
            owner.reap()?;
        }
        Ok(())
    }
}

impl WorkerBackend {
    fn validate_kinds(&self, batch: &WorkerBatch) -> Result<()> {
        let kinds: BTreeSet<CallKind> = batch.plan.calls.iter().map(|call| call.code).collect();
        if kinds.len() > 1 {
            return Err(Error::Invalid(
                "an execution batch carries calls of one kind".into(),
            ));
        }
        Ok(())
    }
}

impl ServiceBackend for WorkerBackend {
    fn batch(&self, plan: Arc<BatchPlan>) -> std::result::Result<WorkerBatch, WorkerError> {
        Ok(WorkerBatch::new(plan))
    }

    fn response_error(
        &self,
        kind: RequestKind,
        error: WorkerError,
    ) -> std::result::Result<WorkerResponseError, WorkerError> {
        let calls: Vec<String> = error
            .calls()
            .iter()
            .map(|call| format!("{}/{:?}", call.request_key.request_id.0, call.call_id))
            .collect();
        if error.code.traces() {
            tracing::error!(
                target: "uniserve_worker.profiling",
                "worker request {:?} failed: {error} [code={} calls={calls:?}]{}",
                kind.as_str(),
                error.code,
                error
                    .traceback()
                    .map(|traceback| format!("\n{traceback}"))
                    .unwrap_or_default(),
            );
        } else {
            tracing::warn!(
                target: "uniserve_worker.profiling",
                "worker request {:?} failed: {error} [code={} calls={calls:?}]",
                kind.as_str(),
                error.code,
            );
        }
        Ok(error.response())
    }

    fn has_open_requests(&self) -> std::result::Result<bool, WorkerError> {
        Ok(self.requests.has_open_requests()?)
    }

    fn awaiting_acknowledgment(&self) -> std::result::Result<bool, WorkerError> {
        Ok(self
            .rank_storage
            .iter()
            .map(|owner| owner as &dyn RequestStorage)
            .chain(self.storage.iter().map(|owner| owner.as_ref()))
            .any(|owner| owner.awaiting_acknowledgment()))
    }

    fn join_expert_step(&self, leaving: bool) -> std::result::Result<(bool, bool), WorkerError> {
        forward::experts::join_step(self, leaving)
    }
}

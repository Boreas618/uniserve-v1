//! The deployment's writers of the observation store: what its engines report, what its
//! requests did, what its HTTP layer served and which revision it runs.
//!
//! One [`Recorder`] belongs to a node's deployment and outlives its leaders: the node makes it
//! with the store and the revision it starts at, every leader hands it to the statistics sink of
//! each engine, to each request's [`Request`] recorder and to the HTTP layer, and the node moves
//! it to each revision that converges ([`Recorder::serve`]).
//!
//! Every quantity is one metric under the Prometheus conventions (`uniserve_` prefix, seconds and
//! bytes, counters ending in `_total`), labelled by where in the deployment it was measured
//! (`replica`, `scheduler`, `device`, ...). A row's revision is the configuration revision the
//! measurement ran under: an engine's report states the revision of the step values its
//! scheduler last read, and what the daemon itself measures carries the revision it serves.
//!
//! The statistics log (`logging_config.log_stats`) is fed by the same writers
//! ([`super::log::StatsLog`]) and reads no row back.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll, ready};
use std::time::{Duration, Instant, SystemTime};

use futures::{Stream, StreamExt as _};

use proto::admin::ManagedProcess;
use proto::report::{DomainSchedulerStats, ForwardStats, SchedulerStats};
use proto::request::wire::DeviceMemory;
use tracing::{info, warn};

use super::log::StatsLog;
use super::{Error, Fields, Kind, Labels, Store};
use proto::request::FinishReason;

use crate::api::output::{RequestOutput, RequestOutputStream, RequestUsage};

/// The deployment's writer of the observation store; clones share one recorder.
#[derive(Clone)]
pub(crate) struct Recorder {
    inner: Arc<Shared>,
}

struct Shared {
    store: Store,
    /// The revision the deployment serves: the one the last change converged to.
    revision: AtomicU64,
    /// What the statistics log line states.
    log: StatsLog,
}

impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recorder")
            .field("revision", &self.revision())
            .finish_non_exhaustive()
    }
}

/// Where in the deployment a scheduler's report was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Scheduler {
    pub(crate) replica: u32,
    pub(crate) scheduler: u32,
}

impl Scheduler {
    fn labels(self) -> Labels {
        [
            ("replica".to_owned(), self.replica.to_string()),
            ("scheduler".to_owned(), self.scheduler.to_string()),
        ]
        .into()
    }
}

/// How a request ended, as its `outcome` label and field state it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Outcome {
    /// It reached a stop condition: its end of sequence or a stop string.
    Stop,
    /// It reached its token limit.
    Length,
    /// Its output repeated past the repetition limit.
    Repetition,
    /// It found no room in the KV cache.
    CacheCapacity,
    /// Its client abandoned it, or it was cancelled.
    Abort,
    /// It failed.
    Error,
}

impl Outcome {
    /// The outcome a request that finished for `reason` ended with.
    pub(crate) fn of(reason: &FinishReason) -> Self {
        match reason {
            FinishReason::Completed | FinishReason::Eos | FinishReason::Stop => Self::Stop,
            FinishReason::MaxTokens => Self::Length,
            FinishReason::Cancelled | FinishReason::Aborted => Self::Abort,
            FinishReason::Repetition => Self::Repetition,
            FinishReason::CacheCapacity => Self::CacheCapacity,
            FinishReason::Error => Self::Error,
        }
    }

    /// The outcome's name in labels, fields and logs.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Repetition => "repetition",
            Self::CacheCapacity => "cache_capacity",
            Self::Abort => "abort",
            Self::Error => "error",
        }
    }
}

impl Recorder {
    /// The recorder of a deployment that writes `store` and starts at revision `revision`.
    pub(crate) fn new(store: Store, revision: u64) -> Self {
        Self {
            inner: Arc::new(Shared {
                store,
                revision: AtomicU64::new(revision),
                log: StatsLog::default(),
            }),
        }
    }

    /// The store the recorder writes.
    pub(crate) fn store(&self) -> &Store {
        &self.inner.store
    }

    /// The revision the deployment serves.
    pub(crate) fn revision(&self) -> u64 {
        self.inner.revision.load(Ordering::Relaxed)
    }

    /// What the statistics log line states.
    pub(crate) fn log(&self) -> &StatsLog {
        &self.inner.log
    }

    /// Records that `replicas` replicas of the deployment run revision `revision` from now on:
    /// what the daemon measures is attributed to it, and `uniserve_replica_revision` states it
    /// for each replica.
    pub(crate) fn serve(&self, revision: u64, replicas: usize) {
        self.inner.revision.store(revision, Ordering::Relaxed);
        for replica in 0..replicas {
            let labels = [("replica".to_owned(), replica.to_string())].into();
            self.set(
                "uniserve_replica_revision",
                &labels,
                revision,
                revision as f64,
            );
        }
    }

    fn add(&self, name: &str, labels: &Labels, revision: u64, delta: f64) {
        // A report's counters are its interval's deltas: an interval that moved none writes no
        // row, so a series starts at its first increase.
        if delta > 0.0 {
            written(name, self.inner.store.add(name, labels, revision, delta));
        }
    }

    fn set(&self, name: &str, labels: &Labels, revision: u64, value: f64) {
        written(name, self.inner.store.set(name, labels, revision, value));
    }

    fn observe(&self, name: &str, labels: &Labels, revision: u64, value: f64) {
        written(
            name,
            self.inner.store.observe(name, labels, revision, value),
        );
    }

    /// Records one scheduler's interval report under where it was measured. Its counters are the
    /// interval's deltas and add; its occupancy and lifetime maxima are gauges.
    pub(crate) fn scheduler(&self, at: Scheduler, stats: &SchedulerStats) {
        self.inner.log.scheduler(at, stats);

        let labels = at.labels();
        let revision = stats.revision;
        let seconds = |us: u64| us as f64 * 1e-6;

        for (name, value) in [
            (
                "uniserve_scheduler_running_requests",
                stats.num_running_reqs as f64,
            ),
            (
                "uniserve_scheduler_waiting_requests",
                stats.num_waiting_reqs as f64,
            ),
            ("uniserve_kv_cache_usage_ratio", stats.prefix_cache_usage),
            (
                "uniserve_scheduler_queue_wait_max_seconds",
                seconds(stats.max_queue_wait_us),
            ),
        ] {
            self.set(name, &labels, revision, value);
        }

        let cache = &stats.prefix_cache_stats;
        for (name, delta) in [
            (
                "uniserve_scheduler_admitted_requests_total",
                stats.num_admitted_reqs as f64,
            ),
            (
                "uniserve_scheduler_queue_wait_seconds_total",
                seconds(stats.queue_wait_us_total),
            ),
            (
                "uniserve_prefix_cache_queried_tokens_total",
                cache.base.queries as f64,
            ),
            (
                "uniserve_prefix_cache_hit_tokens_total",
                cache.base.hits as f64,
            ),
            (
                "uniserve_prefix_cache_host_hit_tokens_total",
                cache.host_hits as f64,
            ),
            (
                "uniserve_prefix_cache_peer_hit_tokens_total",
                cache.peer_hits as f64,
            ),
            ("uniserve_request_transfers_total", stats.transfers as f64),
            (
                "uniserve_request_transfer_failures_total",
                stats.failed_transfers as f64,
            ),
            (
                "uniserve_request_transferred_blocks_total",
                stats.transferred_blocks as f64,
            ),
            (
                "uniserve_scheduler_calls_without_blocks_total",
                stats.calls_without_blocks as f64,
            ),
            (
                "uniserve_scheduler_preempted_requests_total",
                stats.preempted_requests as f64,
            ),
            (
                "uniserve_kv_cache_evicted_blocks_total",
                stats.evicted_blocks as f64,
            ),
            (
                "uniserve_scheduler_given_up_draft_blocks_total",
                stats.given_up_draft_blocks as f64,
            ),
            (
                "uniserve_batch_exec_seconds_total",
                seconds(stats.worker_exec_us),
            ),
            (
                "uniserve_batch_roundtrip_seconds_total",
                seconds(stats.batch_roundtrip_us),
            ),
            ("uniserve_batches_total", stats.batch_count as f64),
            ("uniserve_batch_tokens_total", stats.batch_tokens as f64),
            (
                "uniserve_numerical_faults_total",
                stats.numerical_faults as f64,
            ),
        ] {
            self.add(name, &labels, revision, delta);
        }

        for domain in &stats.domain_stats {
            self.domain(&labels, revision, domain);
        }
        // A report states no worker counters when none of them moved in its interval.
        if let Some(worker) = &stats.worker_forward_stats {
            self.worker(&labels, revision, worker);
        }
    }

    /// One execution domain's accounting of a scheduler's report.
    fn domain(&self, labels: &Labels, revision: u64, domain: &DomainSchedulerStats) {
        let mut at = labels.clone();
        at.insert("domain".to_owned(), domain.domain.clone());

        self.set(
            "uniserve_scheduler_domain_active_credits",
            &at,
            revision,
            domain.active_credits as f64,
        );
        self.set(
            "uniserve_scheduler_domain_peak_credits",
            &at,
            revision,
            domain.peak_credits as f64,
        );
        for (name, delta) in [
            (
                "uniserve_scheduler_domain_backpressure_total",
                domain.backpressure_events,
            ),
            (
                "uniserve_scheduler_domain_reclaimed_credits_total",
                domain.reclaimed_credits,
            ),
            (
                "uniserve_scheduler_domain_completed_batches_total",
                domain.completed_batches,
            ),
        ] {
            self.add(name, &at, revision, delta as f64);
        }
        for (outcome, calls) in [
            ("launched", domain.launched_calls),
            ("completed", domain.completed_calls),
            ("predicated", domain.predicated_calls),
            ("error", domain.error_calls),
        ] {
            let labels = with(&at, "outcome", outcome);
            self.add(
                "uniserve_scheduler_domain_calls_total",
                &labels,
                revision,
                calls as f64,
            );
        }
        for (phase, us) in [
            ("queue", domain.queue_us),
            ("launch", domain.launch_us),
            ("device", domain.device_us),
            ("completion", domain.completion_us),
        ] {
            let labels = with(&at, "phase", phase);
            self.add(
                "uniserve_scheduler_domain_seconds_total",
                &labels,
                revision,
                us as f64 * 1e-6,
            );
        }
    }

    /// The worker counters of a scheduler's report.
    fn worker(&self, labels: &Labels, revision: u64, worker: &ForwardStats) {
        let seconds = |us: u64| us as f64 * 1e-6;

        for (name, delta) in [
            (
                "uniserve_attention_launches_total",
                worker.attention_launches as f64,
            ),
            (
                "uniserve_attention_seconds_total",
                seconds(worker.attention_us),
            ),
            (
                "uniserve_fence_records_lost_total",
                worker.fence_records_lost as f64,
            ),
            (
                "uniserve_graph_captures_total",
                worker.cuda_graph_captures as f64,
            ),
            (
                "uniserve_graph_replays_total",
                worker.cuda_graph_replays as f64,
            ),
            (
                "uniserve_graph_misses_total",
                worker.cuda_graph_misses as f64,
            ),
            (
                "uniserve_graph_fallbacks_total",
                worker.cuda_graph_fallbacks as f64,
            ),
            (
                "uniserve_graph_unpadded_tokens_total",
                worker.cuda_graph_unpadded_tokens as f64,
            ),
            (
                "uniserve_graph_padded_tokens_total",
                worker.cuda_graph_padded_tokens as f64,
            ),
            (
                "uniserve_decode_token_relay_hits_total",
                worker.text_decode_token_relay_hits as f64,
            ),
            (
                "uniserve_decode_token_relay_misses_total",
                worker.text_decode_token_relay_misses as f64,
            ),
            (
                "uniserve_decode_position_relay_hits_total",
                worker.text_decode_position_relay_hits as f64,
            ),
            (
                "uniserve_decode_position_relay_misses_total",
                worker.text_decode_position_relay_misses as f64,
            ),
            (
                "uniserve_spec_verify_rows_total",
                worker.spec_verify_rows as f64,
            ),
            (
                "uniserve_spec_verify_draft_tokens_total",
                worker.spec_verify_draft_tokens as f64,
            ),
            (
                "uniserve_spec_verify_accepted_tokens_total",
                worker.spec_verify_accepted_tokens as f64,
            ),
            (
                "uniserve_spec_verify_rejected_tokens_total",
                worker.spec_verify_rejected_tokens as f64,
            ),
            (
                "uniserve_spec_verify_committed_tokens_total",
                worker.spec_verify_committed_tokens as f64,
            ),
        ] {
            self.add(name, labels, revision, delta);
        }
        // Tokens committed per verification step that carried drafts, over the interval.
        if worker.spec_verify_rows > 0 {
            self.set(
                "uniserve_spec_decode_accepted_length",
                labels,
                revision,
                worker.spec_verify_committed_tokens as f64 / worker.spec_verify_rows as f64,
            );
        }

        let keyed = [
            ("uniserve_forwards_total", "mode", &worker.mode_counts, 1.0),
            (
                "uniserve_forward_tokens_total",
                "mode",
                &worker.mode_tokens,
                1.0,
            ),
            (
                "uniserve_forward_seconds_total",
                "mode",
                &worker.mode_us,
                1e-6,
            ),
            (
                "uniserve_graph_dispatches_total",
                "mode",
                &worker.cuda_graph_runtime_mode_counts,
                1.0,
            ),
            (
                "uniserve_attention_backend_launches_total",
                "backend",
                &worker.attention_backend_counts,
                1.0,
            ),
            (
                "uniserve_spec_verify_path_rows_total",
                "path",
                &worker.spec_verify_path_counts,
                1.0,
            ),
        ];
        for (name, key, values, unit) in keyed {
            for (value, &delta) in values {
                self.add(
                    name,
                    &with(labels, key, value),
                    revision,
                    delta as f64 * unit,
                );
            }
        }
        for (time, &us) in &worker.stream_us {
            let mut at = with(labels, "stream", &time.stream);
            at.insert("state".to_owned(), time.state.clone());
            self.add("uniserve_stream_seconds_total", &at, revision, seconds(us));
        }
        // The kernel inventory: what each operator launched under each descriptor key, which
        // `uniserve build-server build --keys` reads a deployment's key list from.
        for (launch, &count) in &worker.kernel_launches {
            let mut at = with(labels, "op", &launch.operator);
            at.insert("descriptor".to_owned(), launch.key.clone());
            self.add(
                "uniserve_kernel_launches_total",
                &at,
                revision,
                count as f64,
            );
        }
        for (call, &count) in &worker.blocking_calls {
            let mut at = with(labels, "thread", &call.thread);
            at.insert("call".to_owned(), call.call.clone());
            self.add("uniserve_blocking_calls_total", &at, revision, count as f64);
        }
        // The fence probes' step durations: module `step` is the whole step between its
        // markers, any other the step's summed durations of that module class's instances.
        for (step, histogram) in &worker.fence_seconds {
            let mut at = with(labels, "phase", &step.stage);
            at.insert("module".to_owned(), step.module.clone());
            written(
                "uniserve_step_device_seconds",
                self.inner
                    .store
                    .merge("uniserve_step_device_seconds", &at, revision, histogram),
            );
        }
    }

    /// Records one report of replica `replica`'s main process: each device's budget and what
    /// each of its pools has mapped, in bytes.
    pub(crate) fn memory(&self, replica: u32, devices: &[DeviceMemory]) {
        let revision = self.revision();
        for device in devices {
            let labels: Labels = [
                ("replica".to_owned(), replica.to_string()),
                ("device".to_owned(), device.device.to_string()),
            ]
            .into();
            self.set(
                "uniserve_memory_budget_bytes",
                &labels,
                revision,
                device.budget as f64,
            );
            for pool in &device.pools {
                self.set(
                    "uniserve_memory_mapped_bytes",
                    &with(&labels, "pool", &pool.pool),
                    revision,
                    pool.mapped as f64,
                );
            }
        }
    }

    /// Records one HTTP request a route of the client API served: its method, the status class
    /// of its response head and its matched route, and the seconds from its arrival to the end of
    /// its response body.
    pub(crate) fn http(&self, method: &str, status: &str, handler: &str, seconds: f64) {
        let revision = self.revision();
        let labels: Labels = [
            ("method".to_owned(), method.to_owned()),
            ("handler".to_owned(), handler.to_owned()),
        ]
        .into();
        self.add(
            "uniserve_http_requests_total",
            &with(&labels, "status", status),
            revision,
            1.0,
        );
        self.observe(
            "uniserve_http_request_duration_seconds",
            &labels,
            revision,
            seconds,
        );
    }

    /// Records a state an engine process of the deployment entered on `host`: a `process` event
    /// labelled by its host, worker group and rank, whose fields are its launch generation, pid,
    /// epoch and `state` (`starting`, `serving`, `parked`, `stopped` or `failed` with its
    /// `reason`).
    pub(crate) fn process(&self, host: &str, process: &ManagedProcess) {
        let labels = [
            ("host".to_owned(), host.to_owned()),
            ("group".to_owned(), process.group.clone()),
            ("rank".to_owned(), process.rank.to_string()),
        ]
        .into();
        let mut fields = match serde_json::to_value(process) {
            Ok(serde_json::Value::Object(fields)) => fields,
            _ => Fields::new(),
        };
        for label in ["group", "rank", "build"] {
            fields.remove(label);
        }
        self.inner
            .store
            .append(Kind::Process, labels, fields, self.revision());
    }
}

/// `labels` with `key` set to `value`.
fn with(labels: &Labels, key: &str, value: &str) -> Labels {
    let mut labels = labels.clone();
    labels.insert(key.to_owned(), value.to_owned());
    labels
}

/// Reports a write the store refused. Every name here is a valid metric of one instrument, so a
/// refusal is a value that is not finite, which is dropped rather than failing what measured it.
fn written(name: &str, result: Result<(), Error>) {
    if let Err(error) = result {
        warn!(metric = name, "an observation was not recorded: {error}");
    }
}

/// How many requests ended since the daemon started, and how many of them failed, which an
/// epoch's trial reads its failure rate from.
pub(crate) struct Outcomes {
    pub(crate) ended: AtomicU64,
    pub(crate) failed: AtomicU64,
}

/// Every request's outcome, as [`Request`] records it.
pub(crate) static OUTCOMES: Outcomes = Outcomes {
    ended: AtomicU64::new(0),
    failed: AtomicU64::new(0),
};

/// One request a scheduler accepted, recorded once when it ends: a `request` event and the
/// request metrics. A recorder dropped before the request ended records an abort.
///
/// The request is recorded under its outcome when it finished, as an abort when its output was
/// dropped before its terminal event (a client that disconnected), and as an error when it
/// failed, its stream yielded an error, or its stream ended without a terminal event. A rejected
/// request, which consumed no engine work, is not recorded. The token metrics count only a
/// request that finished.
pub(crate) struct Request {
    recorder: Recorder,
    /// Where it was served: its replica, and its scheduler once one accepted it.
    replica: u32,
    scheduler: Option<u32>,
    /// When it arrived, on the daemon's monotonic clock and on the wall clock.
    started: Instant,
    arrived: SystemTime,
    /// The request's id, logged in a summary line when it ends if request logging is on.
    log: Option<String>,
    /// The caller's session of the request (`prompt_cache_key`), which its event states.
    session: Option<String>,
    recorded: bool,
}

impl Request {
    /// The recorder of a request that arrived at `started`, served by `scheduler` of replica
    /// `replica`.
    pub(crate) fn new(
        recorder: Recorder,
        (replica, scheduler): (u32, Option<u32>),
        started: Instant,
        log: Option<String>,
    ) -> Self {
        let arrived = SystemTime::now()
            .checked_sub(started.elapsed())
            .unwrap_or_else(SystemTime::now);
        Self {
            recorder,
            replica,
            scheduler,
            started,
            arrived,
            log,
            session: None,
            recorded: false,
        }
    }

    /// The recorder of a request of the caller's session `session`.
    pub(crate) fn session(mut self, session: Option<String>) -> Self {
        self.session = session;
        self
    }

    /// Records the request as `outcome`, with the usage of a finished request, unless it was
    /// recorded already.
    ///
    /// The end-to-end latency is the time since the request started on this daemon's monotonic
    /// clock; the queue time is the engine's own measure, which a finished request's usage
    /// carries.
    pub(crate) fn record(&mut self, outcome: Outcome, usage: Option<&RequestUsage>) {
        if std::mem::replace(&mut self.recorded, true) {
            return;
        }
        // A request the client abandoned neither served nor failed.
        if outcome != Outcome::Abort {
            OUTCOMES.ended.fetch_add(1, Ordering::Relaxed);
            if outcome == Outcome::Error {
                OUTCOMES.failed.fetch_add(1, Ordering::Relaxed);
            }
        }

        let recorder = &self.recorder;
        let revision = recorder.revision();
        let latency = self.started.elapsed().as_secs_f64();
        let mut labels: Labels = [("replica".to_owned(), self.replica.to_string())].into();
        if let Some(scheduler) = self.scheduler {
            labels.insert("scheduler".to_owned(), scheduler.to_string());
        }

        recorder.add(
            "uniserve_requests_total",
            &with(&labels, "outcome", outcome.name()),
            revision,
            1.0,
        );
        recorder.observe(
            "uniserve_e2e_request_latency_seconds",
            &labels,
            revision,
            latency,
        );

        let mut fields = Fields::new();
        let arrived = self
            .arrived
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs_f64();
        fields.insert("arrival".to_owned(), arrived.into());
        fields.insert("outcome".to_owned(), outcome.name().into());
        fields.insert("latency".to_owned(), latency.into());
        if let Some(session) = &self.session {
            fields.insert("session".to_owned(), session.clone().into());
        }
        if let Some(usage) = usage {
            usage_metrics(recorder, &labels, revision, usage);
            recorder.inner.log.usage(
                u64::from(usage.prompt_tokens),
                u64::from(usage.output_tokens),
            );
            request_fields(&mut fields, usage);
        }
        recorder
            .inner
            .store
            .append(Kind::Request, labels, fields, revision);

        if let Some(request_id) = &self.log {
            info!(
                %request_id,
                outcome = outcome.name(),
                prompt_tokens = usage.map_or(0, |usage| usage.prompt_tokens),
                output_tokens = usage.map_or(0, |usage| usage.output_tokens),
                latency_s = latency,
                "request finished"
            );
        }
    }
}

impl crate::api::events::Stats for Request {
    fn started(&self) -> Instant {
        self.started
    }

    fn engine(&self) -> u32 {
        self.replica
    }

    fn observe(&mut self, output: &RequestOutput) {
        match output {
            RequestOutput::Finished { reason, usage, .. } => {
                self.record(Outcome::of(reason), Some(usage));
            }
            RequestOutput::Failed { .. } => self.record(Outcome::Error, None),
            RequestOutput::Rejected { .. } => self.recorded = true,
            _ => {}
        }
    }

    fn abandoned(&mut self) {
        self.record(Outcome::Abort, None);
    }
}

impl Drop for Request {
    /// Records a request whose output ended before its terminal event as aborted, so a
    /// handed-over, abandoned or dropped response is counted once.
    fn drop(&mut self) {
        self.record(Outcome::Abort, None);
    }
}

/// The output stream of one request a scheduler accepted, which records the request once
/// ([`Request`]) as its events pass through unchanged.
pub(crate) struct Recorded {
    inner: RequestOutputStream,
    request: Request,
}

impl Recorded {
    /// Wraps `inner`, the output of the request `request` records.
    pub(crate) fn new(inner: RequestOutputStream, request: Request) -> Self {
        Self { inner, request }
    }
}

impl Stream for Recorded {
    type Item = Result<RequestOutput, crate::api::output::OutputError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let next = ready!(self.inner.poll_next_unpin(cx));
        match &next {
            Some(Ok(event)) => crate::api::events::Stats::observe(&mut self.request, event),
            Some(Err(_)) | None => self.request.record(Outcome::Error, None),
        }
        Poll::Ready(next)
    }
}

/// Records a finished request's usage into the request metrics: its tokens, its time to first
/// token (for a request that generated tokens and produced visible output) and its queue time
/// (when the engine measured it).
fn usage_metrics(recorder: &Recorder, labels: &Labels, revision: u64, usage: &RequestUsage) {
    let prompt = f64::from(usage.prompt_tokens);
    let generated = f64::from(usage.output_tokens);

    recorder.add("uniserve_prompt_tokens_total", labels, revision, prompt);
    recorder.add(
        "uniserve_generation_tokens_total",
        labels,
        revision,
        generated,
    );
    recorder.observe("uniserve_request_prompt_tokens", labels, revision, prompt);
    recorder.observe(
        "uniserve_request_generation_tokens",
        labels,
        revision,
        generated,
    );
    if let Some(queue_us) = usage.timings.queue_us {
        recorder.observe(
            "uniserve_request_queue_time_seconds",
            labels,
            revision,
            queue_us as f64 * 1e-6,
        );
    }
    if usage.output_tokens > 0
        && let Some(first_us) = usage.timings.first_visible_output_us
    {
        recorder.observe(
            "uniserve_ttft_seconds",
            labels,
            revision,
            first_us as f64 * 1e-6,
        );
        if let Some(tpot) = tpot(usage, first_us) {
            recorder.observe("uniserve_tpot_seconds", labels, revision, tpot);
        }
    }
}

/// A finished request's time per output token after its first, in seconds: the rest of the
/// request over the tokens it generated after its first; `None` for a request of one token.
fn tpot(usage: &RequestUsage, first_us: u64) -> Option<f64> {
    (usage.output_tokens > 1).then(|| {
        let rest = usage.timings.total_us.saturating_sub(first_us) as f64 * 1e-6;
        rest / f64::from(usage.output_tokens - 1)
    })
}

/// The fields of a finished request's `request` event: its prompt, prefix hit and output
/// lengths in tokens, its time to first token and time per output token in seconds.
fn request_fields(fields: &mut Fields, usage: &RequestUsage) {
    fields.insert("prompt_tokens".to_owned(), usage.prompt_tokens.into());
    fields.insert("cached_tokens".to_owned(), usage.cached_tokens.into());
    fields.insert("output_tokens".to_owned(), usage.output_tokens.into());
    if let Some(queue_us) = usage.timings.queue_us {
        fields.insert("queue".to_owned(), (queue_us as f64 * 1e-6).into());
    }
    let Some(first_us) = usage.timings.first_visible_output_us else {
        return;
    };
    let ttft = first_us as f64 * 1e-6;
    fields.insert("ttft".to_owned(), ttft.into());
    if let Some(tpot) = tpot(usage, first_us) {
        fields.insert("tpot".to_owned(), tpot.into());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use proto::report::{DomainSchedulerStats, ForwardStats, SchedulerStats};
    use serde_json::Value;

    use super::*;
    use crate::observation::{Retention, System};

    fn recorder() -> Recorder {
        Recorder::new(
            Store::new(Arc::new(System), Retention::default()).unwrap(),
            7,
        )
    }

    async fn rows(store: &Store, sql: &str) -> Vec<Value> {
        let json = store.query(sql).await.unwrap().json().unwrap();
        serde_json::from_slice(&json).unwrap()
    }

    /// A scheduler's report becomes rows under its replica and scheduler and the revision its
    /// step values came from: counters add across reports, gauges keep the latest reading, and a
    /// counter no report moved has no row.
    #[tokio::test]
    async fn a_schedulers_reports_become_its_rows() {
        let recorder = recorder();
        let at = Scheduler {
            replica: 1,
            scheduler: 0,
        };
        let report = |running, admitted| SchedulerStats {
            revision: 3,
            num_running_reqs: running,
            num_admitted_reqs: admitted,
            ..SchedulerStats::default()
        };
        recorder.scheduler(at, &report(4, 2));
        recorder.scheduler(at, &report(6, 5));

        let store = recorder.store();
        let admitted = rows(
            store,
            "SELECT sum(value) AS admitted FROM stats \
             WHERE name = 'uniserve_scheduler_admitted_requests_total' \
             AND labels['replica'] = '1' AND labels['scheduler'] = '0' AND revision = 3",
        )
        .await;
        assert_eq!(admitted[0]["admitted"], 7.0);

        let running = rows(
            store,
            "SELECT value FROM stats WHERE name = 'uniserve_scheduler_running_requests' \
             ORDER BY time DESC LIMIT 1",
        )
        .await;
        assert_eq!(running[0]["value"], 6.0);

        let moved = rows(
            store,
            "SELECT count(*) AS n FROM stats WHERE name = 'uniserve_request_transfers_total'",
        )
        .await;
        assert_eq!(moved[0]["n"], 0);
    }

    /// A domain's calls and times are counted under their domain and kind, the fence probes'
    /// step durations are one histogram per phase and module class, in seconds, and the kernel
    /// inventory is one counter per operator and descriptor key.
    #[tokio::test]
    async fn breakdowns_are_labelled_by_their_keys() {
        let recorder = recorder();
        let mut fence = proto::histogram::Histogram::new();
        fence.observe(0.002);
        fence.observe(0.004);
        let stats = SchedulerStats {
            revision: 1,
            domain_stats: vec![DomainSchedulerStats {
                domain: "decode".to_owned(),
                launched_calls: 3,
                device_us: 1_500_000,
                ..DomainSchedulerStats::default()
            }],
            worker_forward_stats: Some(ForwardStats {
                fence_seconds: [(
                    proto::report::FenceStep {
                        stage: "decode".to_owned(),
                        module: "step".to_owned(),
                    },
                    fence,
                )]
                .into(),
                kernel_launches: [(
                    proto::report::KernelLaunch {
                        operator: "gemm".to_owned(),
                        key: "m128_n256".to_owned(),
                    },
                    96,
                )]
                .into(),
                ..ForwardStats::default()
            }),
            ..SchedulerStats::default()
        };
        recorder.scheduler(
            Scheduler {
                replica: 0,
                scheduler: 0,
            },
            &stats,
        );

        let store = recorder.store();
        let calls = rows(
            store,
            "SELECT value FROM stats WHERE name = 'uniserve_scheduler_domain_calls_total' \
             AND labels['domain'] = 'decode' AND labels['outcome'] = 'launched'",
        )
        .await;
        assert_eq!(calls[0]["value"], 3.0);
        let device = rows(
            store,
            "SELECT value FROM stats WHERE name = 'uniserve_scheduler_domain_seconds_total' \
             AND labels['phase'] = 'device'",
        )
        .await;
        assert_eq!(device[0]["value"], 1.5);
        let step = rows(
            store,
            "SELECT histogram FROM stats WHERE name = 'uniserve_step_device_seconds' \
             AND labels['phase'] = 'decode' AND labels['module'] = 'step' AND revision = 1",
        )
        .await;
        assert_eq!(step[0]["histogram"]["count"], 2);
        assert_eq!(step[0]["histogram"]["max"], 0.004);
        let kernels = rows(
            store,
            "SELECT labels['op'] AS op, labels['descriptor'] AS descriptor, sum(value) AS launches \
             FROM stats WHERE name = 'uniserve_kernel_launches_total' GROUP BY 1, 2",
        )
        .await;
        assert_eq!(kernels.len(), 1);
        assert_eq!(
            (
                &kernels[0]["op"],
                &kernels[0]["descriptor"],
                &kernels[0]["launches"]
            ),
            (&"gemm".into(), &"m128_n256".into(), &96.0.into())
        );
    }

    /// A finished request is one `request` event with its lengths and latencies, and one count
    /// under its outcome; a request whose recorder is dropped before it ended counts as an abort
    /// with no token metrics.
    #[tokio::test]
    async fn a_request_is_recorded_once_when_it_ends() {
        let recorder = recorder();
        let usage = RequestUsage {
            prompt_tokens: 12,
            output_tokens: 5,
            cached_tokens: 8,
            timings: crate::api::output::RuntimeTimings {
                compile_us: 0,
                queue_us: Some(1_000),
                first_visible_output_us: Some(20_000),
                total_us: 60_000,
            },
        };
        let mut finished = Request::new(recorder.clone(), (0, Some(1)), Instant::now(), None)
            .session(Some("conversation-7".to_owned()));
        finished.record(Outcome::Length, Some(&usage));
        finished.record(Outcome::Error, None);
        drop(finished);
        drop(Request::new(
            recorder.clone(),
            (0, None),
            Instant::now(),
            None,
        ));

        let store = recorder.store();
        let events = rows(
            store,
            "SELECT fields, revision FROM events WHERE kind = 'request' ORDER BY time",
        )
        .await;
        assert_eq!(events.len(), 2);
        let fields = &events[0]["fields"];
        assert_eq!(fields["outcome"], "length");
        assert_eq!(fields["prompt_tokens"], "12");
        assert_eq!(fields["cached_tokens"], "8");
        assert_eq!(fields["ttft"], "0.02");
        assert_eq!(fields["tpot"], "0.01");
        assert_eq!(fields["session"], "conversation-7");
        assert_eq!(events[0]["revision"], 7);
        assert_eq!(events[1]["fields"]["outcome"], "abort");

        let outcomes = rows(
            store,
            "SELECT labels['outcome'] AS outcome, sum(value) AS n FROM stats \
             WHERE name = 'uniserve_requests_total' GROUP BY 1 ORDER BY 1",
        )
        .await;
        assert_eq!(outcomes.len(), 2);
        assert_eq!(
            (&outcomes[0]["outcome"], &outcomes[0]["n"]),
            (&"abort".into(), &1.0.into())
        );
        assert_eq!(
            (&outcomes[1]["outcome"], &outcomes[1]["n"]),
            (&"length".into(), &1.0.into())
        );
        let tokens = rows(
            store,
            "SELECT sum(value) AS n FROM stats WHERE name = 'uniserve_generation_tokens_total'",
        )
        .await;
        assert_eq!(tokens[0]["n"], 5.0);
        let latencies = rows(
            store,
            "SELECT name, histogram FROM stats \
             WHERE name IN ('uniserve_ttft_seconds', 'uniserve_tpot_seconds') ORDER BY name",
        )
        .await;
        assert_eq!(latencies.len(), 2);
        assert_eq!(latencies[0]["name"], "uniserve_tpot_seconds");
        assert_eq!(latencies[0]["histogram"]["max"], 0.01);
        assert_eq!(latencies[1]["histogram"]["max"], 0.02);
    }

    /// The revision a deployment serves is each replica's `uniserve_replica_revision`, and what
    /// the daemon measures after the switch carries it.
    #[tokio::test]
    async fn the_served_revision_is_a_gauge_per_replica() {
        let recorder = recorder();
        recorder.serve(9, 2);
        recorder.http("POST", "2xx", "/v1/completions", 0.25);

        let store = recorder.store();
        let revisions = rows(
            store,
            "SELECT labels['replica'] AS replica, value FROM stats \
             WHERE name = 'uniserve_replica_revision' ORDER BY 1",
        )
        .await;
        assert_eq!(revisions.len(), 2);
        assert!(revisions.iter().all(|row| row["value"] == 9.0));
        let http = rows(
            store,
            "SELECT revision FROM stats WHERE name = 'uniserve_http_requests_total'",
        )
        .await;
        assert_eq!(http[0]["revision"], 9);
    }
}

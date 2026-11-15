//! The admin API (`/admin`): the deployment's status, its event log, and the
//! operations that change it while it serves.
//!
//! The endpoints follow SGLang's and vLLM's (`get_server_info`,
//! `pause_generation`, `continue_generation`, `flush_cache`). A write
//! operation runs on a task of its own, so it completes, or fails and says
//! so in the event log, whether or not the client that asked for it stays
//! connected; one runs at a time, and another is refused with `409`. An
//! external manager attaches by reading `status` and `events` and detaches by
//! stopping; the daemon keeps no state of it.
//!
//! The API is served apart from the client API, on two Unix sockets
//! ([`serve`]), and access is their file permissions:
//!
//! - the operator's listener serves every route ([`router`]);
//! - the harness's listener serves the verbs of an update ([`harness_router`]):
//!   apply, for package selections and the options the operator lists
//!   (`admin_config.harness_options`), commit, rollback, and reading the
//!   status, the events and the update's report. The operator's operations
//!   (weight update, sleep, wake, LoRA, rebalance, pause, flush, profiling,
//!   shutdown, daemon restart) are not routes of it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::api::openai::ApiError;
use crate::leader::{Leader, LeaderStatus, UpdateError};
use crate::updates::change::{Applied, ApplyRequest, InFlight};
use crate::updates::update::Phase;

/// Events the log keeps; older ones leave it.
const EVENT_CAPACITY: usize = 1024;

/// The longest an events request waits for a new event.
const MAX_WAIT: Duration = Duration::from_secs(60);

/// How often the admin samples the leader's status for changes.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// One entry of the event log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct AdminEvent {
    /// Its sequence number, from 1, increasing by one.
    pub seq: u64,
    /// Seconds since the Unix epoch.
    pub time: f64,
    /// The event's category: `operation`, `engine`, `deployment` or
    /// `config`.
    pub kind: String,
    pub message: String,
    /// What the event carries beyond its message, always present: `{}` when
    /// it carries nothing. An update's phase is `phase` here, beside the
    /// update's number and the epoch it moved, so a reader of the log reads
    /// the phase from the data and the category from `kind`.
    #[serde(default)]
    pub data: serde_json::Map<String, serde_json::Value>,
}

/// A write operation: running, or the last one that ended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Operation {
    pub id: u64,
    /// `config`, `commit`, `rollback`, `update_weights`, `rebalance_experts`,
    /// `pause_generation`, `continue_generation`, `flush_cache`, `sleep`,
    /// `wake_up`, `load_lora_adapter`, `unload_lora_adapter`, `profile`,
    /// `restart` or `shutdown`.
    pub kind: String,
    /// `running`, `succeeded` or `failed`.
    pub state: String,
    pub started: f64,
    pub ended: Option<f64>,
    /// What it did, once it succeeded.
    pub result: Option<serde_json::Value>,
    /// Why it failed.
    pub error: Option<String>,
}

/// The admin's event log and operations' record as a hot restart hands them
/// to the successor.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AdminHandoff {
    events: Vec<AdminEvent>,
    next_event: u64,
    last: Option<Operation>,
    next_operation: u64,
}

/// The admin's state: the event log and the operations.
pub(crate) struct Admin {
    events: Mutex<VecDeque<AdminEvent>>,
    next_event: AtomicU64,
    appended: Notify,
    running: Mutex<Option<Operation>>,
    last: Mutex<Option<Operation>>,
    next_operation: AtomicU64,
    /// Cancelled to stop serving: a shutdown.
    shutdown: CancellationToken,
    /// Whether the shutdown releases the deployment's storage at the host's
    /// Storage Server, once its engine is down.
    retire: Arc<AtomicBool>,
    /// Set once a hot restart handed the deployment to a successor, which
    /// takes the operations from then on.
    handoff_sent: AtomicBool,
    /// The node's epoch records: the same instance the leader keeps, so the
    /// admin reports and moves the records the updater itself does.
    epochs: Arc<crate::epoch::Epochs>,
    /// The daemon's readiness record once it serves, which `GET /admin/status` reports.
    readiness: Mutex<Option<service::Readiness>>,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

impl Admin {
    /// An admin over `epochs`, the records the node's leader keeps: there is
    /// one `Epochs` per node and the admin holds the same instance, so
    /// `GET /admin/config` and the update operations read what the updater
    /// itself moves.
    pub(crate) fn with_epochs(
        shutdown: CancellationToken,
        retire: Arc<AtomicBool>,
        epochs: Arc<crate::epoch::Epochs>,
    ) -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(VecDeque::with_capacity(EVENT_CAPACITY)),
            next_event: AtomicU64::new(1),
            appended: Notify::new(),
            running: Mutex::new(None),
            last: Mutex::new(None),
            next_operation: AtomicU64::new(1),
            shutdown,
            retire,
            handoff_sent: AtomicBool::new(false),
            epochs,
            readiness: Mutex::new(None),
        })
    }

    /// The daemon's readiness record, which `GET /admin/status` reports from now on.
    pub(crate) fn set_readiness(&self, readiness: service::Readiness) {
        *service::lock(&self.readiness) = Some(readiness);
    }

    /// The node's epoch records, which the admin reports and the generation
    /// operations move.
    pub(crate) fn epochs(&self) -> &crate::epoch::Epochs {
        &self.epochs
    }

    /// The event log and the operations' record, for the successor of a hot
    /// restart to go on with ([`Admin::resume`]).
    pub(crate) fn hand_over(&self) -> AdminHandoff {
        self.handoff_sent.store(true, Ordering::Release);
        AdminHandoff {
            events: service::lock(&self.events).iter().cloned().collect(),
            next_event: self.next_event.load(Ordering::Relaxed),
            last: service::lock(&self.last).clone(),
            next_operation: self.next_operation.load(Ordering::Relaxed),
        }
    }

    /// An admin that goes on with the event log and the operations' record
    /// of the daemon a hot restart replaced, over `epochs`, the records the
    /// successor's leader keeps: the successor's `GET /admin/config` reports
    /// them and the processes its daemon supervises, as [`Admin::with_epochs`]
    /// states.
    pub(crate) fn resume(
        shutdown: CancellationToken,
        retire: Arc<AtomicBool>,
        handoff: AdminHandoff,
        epochs: Arc<crate::epoch::Epochs>,
    ) -> Arc<Self> {
        let admin = Self::with_epochs(shutdown, retire, epochs);
        *service::lock(&admin.events) = handoff.events.into_iter().collect();
        admin
            .next_event
            .store(handoff.next_event, Ordering::Relaxed);
        *service::lock(&admin.last) = handoff.last;
        admin
            .next_operation
            .store(handoff.next_operation, Ordering::Relaxed);
        admin
    }

    /// Appends an event.
    pub(crate) fn record(&self, kind: &'static str, message: impl Into<String>) {
        let mut events = service::lock(&self.events);
        if events.len() == EVENT_CAPACITY {
            events.pop_front();
        }
        events.push_back(AdminEvent {
            seq: self.next_event.fetch_add(1, Ordering::Relaxed),
            time: now(),
            kind: kind.to_owned(),
            message: message.into(),
            data: serde_json::Map::new(),
        });
        drop(events);
        self.appended.notify_waiters();
    }

    /// Appends an update's phase event: a `deployment` event whose `data` is
    /// the update as it stood when it entered the phase (its number, form,
    /// phase, epoch and failure).
    pub(crate) fn update_event(&self, transition: crate::updates::Transition) {
        let phase = transition.phase.name();
        let data = match transition.report {
            serde_json::Value::Object(data) => data,
            _ => serde_json::Map::new(),
        };
        let mut events = service::lock(&self.events);
        if events.len() == EVENT_CAPACITY {
            events.pop_front();
        }
        events.push_back(AdminEvent {
            seq: self.next_event.fetch_add(1, Ordering::Relaxed),
            time: now(),
            kind: "deployment".to_owned(),
            message: format!("update {}: {phase}", transition.update),
            data,
        });
        drop(events);
        self.appended.notify_waiters();
    }

    /// The events after `after`, waiting up to `wait` for one when there
    /// is none, and the sequence number to read from next. `gap` is whether
    /// events after `after` already left the log.
    async fn events_after(&self, after: u64, wait: Duration) -> (Vec<AdminEvent>, u64, bool) {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let notified = self.appended.notified();
            {
                let events = service::lock(&self.events);
                let gap = events.front().is_some_and(|first| first.seq > after + 1);
                let found: Vec<AdminEvent> = events
                    .iter()
                    .filter(|event| event.seq > after)
                    .cloned()
                    .collect();
                let next = events.back().map_or(after, |last| last.seq.max(after));
                if !found.is_empty() || tokio::time::Instant::now() >= deadline {
                    return (found, next, gap);
                }
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                continue;
            }
        }
    }

    /// Starts operation `kind`; refused, with the running one, while
    /// another runs.
    fn begin(&self, kind: &'static str) -> Result<Operation, Box<Operation>> {
        let mut running = service::lock(&self.running);
        if let Some(other) = running.as_ref() {
            return Err(Box::new(other.clone()));
        }
        if self.handoff_sent.load(Ordering::Acquire) {
            return Err(Box::new(Operation {
                id: 0,
                kind: "restart".to_owned(),
                state: "succeeded".to_owned(),
                started: now(),
                ended: Some(now()),
                result: None,
                error: Some("the deployment was handed to a successor daemon".to_owned()),
            }));
        }
        let operation = Operation {
            id: self.next_operation.fetch_add(1, Ordering::Relaxed),
            kind: kind.to_owned(),
            state: "running".to_owned(),
            started: now(),
            ended: None,
            result: None,
            error: None,
        };
        *running = Some(operation.clone());
        drop(running);
        self.record("operation", format!("{kind} {} started", operation.id));
        Ok(operation)
    }

    /// Ends `operation`, the running one, with `result`.
    fn end(
        &self,
        mut operation: Operation,
        result: Result<serde_json::Value, String>,
    ) -> Operation {
        *service::lock(&self.running) = None;
        operation.ended = Some(now());
        match result {
            Ok(result) => {
                operation.state = "succeeded".to_owned();
                operation.result = Some(result);
                self.record(
                    "operation",
                    format!("{} {} succeeded", operation.kind, operation.id),
                );
            }
            Err(error) => {
                operation.state = "failed".to_owned();
                self.record(
                    "operation",
                    format!("{} {} failed: {error}", operation.kind, operation.id),
                );
                operation.error = Some(error);
            }
        }
        *service::lock(&self.last) = Some(operation.clone());
        operation
    }

    /// Records the leader's changes of state, sampled every second, and
    /// every phase its updates enter, until the server stops.
    pub(crate) fn watch(self: &Arc<Self>, leader: &Arc<Leader>) {
        if let Some(mut transitions) = leader.updates.transitions() {
            let admin = Arc::clone(self);
            tokio::spawn(async move {
                while let Some(transition) = transitions.recv().await {
                    admin.update_event(transition);
                }
            });
        }
        let admin = Arc::clone(self);
        let leader = Arc::downgrade(leader);
        tokio::spawn(async move {
            let mut previous: Option<LeaderStatus> = None;
            loop {
                let Some(status) = leader.upgrade().map(|leader| leader.status()) else {
                    return;
                };
                if let Some(previous) = &previous {
                    if previous.engine != status.engine {
                        let reason = status
                            .reason
                            .as_deref()
                            .map(|reason| format!(": {reason}"))
                            .unwrap_or_default();
                        admin.record(
                            "engine",
                            format!(
                                "the engine went from {} to {}{reason}",
                                previous.engine, status.engine
                            ),
                        );
                    }
                    if previous.paused != status.paused {
                        admin.record(
                            "deployment",
                            if status.paused {
                                "new requests wait"
                            } else {
                                "new requests are served"
                            },
                        );
                    }
                }
                previous = Some(status);
                tokio::select! {
                    () = admin.shutdown.cancelled() => return,
                    () = tokio::time::sleep(SAMPLE_INTERVAL) => {}
                }
            }
        });
    }
}

/// The state of every admin route.
#[derive(Clone)]
pub(crate) struct AdminState {
    pub leader: Arc<Leader>,
    pub admin: Arc<Admin>,
    pub jobs: Arc<crate::api::openai::video_jobs::VideoJobs>,
    /// What restarts the daemon in place; `None` where it cannot.
    pub restarter: Option<Arc<crate::restart::Restarter>>,
}

/// The file mode of the operator's admin socket: its owner alone.
const ADMIN_MODE: u32 = 0o600;

/// The file mode of the harness socket: its owner and its group, which is
/// how the operator grants a harness access.
const HARNESS_MODE: u32 = 0o660;

/// The operator's and the harness's admin sockets, bound when the daemon starts and served once
/// its engine serves ([`serve`]).
pub(crate) struct Listeners {
    admin: super::listener::Listener,
    harness: super::listener::Listener,
    paths: (std::path::PathBuf, std::path::PathBuf),
}

impl Listeners {
    /// Binds the admin sockets `config` states, or `admin.sock` and `harness.sock` in the
    /// daemon's instance directory `instance`, each taking its path over from whoever holds it
    /// (a hot restart's predecessor): the operator's with mode `0600`, the harness's with
    /// `0660`.
    ///
    /// # Errors
    ///
    /// Fails when a socket cannot be bound at its path.
    pub(crate) fn bind(
        config: &crate::options::AdminConfig,
        instance: &std::path::Path,
    ) -> anyhow::Result<Self> {
        use anyhow::Context as _;

        let paths = (
            config.admin_socket(instance),
            config.harness_socket(instance),
        );
        let bind = |path: &std::path::Path, mode, name| {
            super::listener::Listener::bind_over(path, mode)
                .with_context(|| format!("binding the {name} socket {}", path.display()))
        };
        Ok(Self {
            admin: bind(&paths.0, ADMIN_MODE, "admin")?,
            harness: bind(&paths.1, HARNESS_MODE, "harness")?,
            paths,
        })
    }

    /// The operator's and the harness's endpoints, as the readiness record states them.
    pub(crate) fn endpoints(&self) -> (Option<service::Endpoint>, Option<service::Endpoint>) {
        (
            Some(service::Endpoint::unix(&self.paths.0)),
            Some(service::Endpoint::unix(&self.paths.1)),
        )
    }
}

/// Serves the admin API of `state` on the operator's and the harness's sockets until `shutdown`
/// is cancelled; the two servers run on tasks of their own.
pub(crate) fn serve(listeners: Listeners, state: AdminState, shutdown: &CancellationToken) {
    serve_routes(
        listeners,
        router(state.clone()),
        harness_router(state),
        shutdown,
    );
}

/// Serves `admin` on the operator's socket and `harness` on the harness's until `shutdown` is
/// cancelled, each on a task of its own.
pub(crate) fn serve_routes(
    listeners: Listeners,
    admin: Router,
    harness: Router,
    shutdown: &CancellationToken,
) {
    let Listeners {
        admin: admin_listener,
        harness: harness_listener,
        paths,
    } = listeners;
    for (listener, path, routes, name) in [
        (admin_listener, paths.0, admin, "admin"),
        (harness_listener, paths.1, harness, "harness"),
    ] {
        tracing::info!(socket = %path.display(), "serving the {name} API");
        let stopped = shutdown.clone();
        tokio::spawn(async move {
            let served = axum::serve(listener, routes)
                .with_graceful_shutdown(async move { stopped.cancelled().await })
                .await;
            if let Err(error) = served {
                tracing::warn!(%error, "the {name} listener ended");
            }
        });
    }
}

/// The harness's routes: the verbs of an update and what it reads to decide
/// one. Apply is checked against what a harness may set.
pub(crate) fn harness_router(state: AdminState) -> Router {
    Router::new()
        .route("/admin/status", get(status))
        .route("/admin/events", get(events))
        .route(
            "/admin/config",
            get(config).put(|State(state): State<AdminState>, body| {
                apply_config(Caller::Harness, state, body)
            }),
        )
        .route("/admin/config/commit", post(commit))
        .route("/admin/config/rollback", post(rollback))
        .with_state(state)
}

/// The operator's routes: every admin route.
pub(crate) fn router(state: AdminState) -> Router {
    Router::new()
        .route("/admin/status", get(status))
        .route("/admin/events", get(events))
        .route(
            "/admin/config",
            get(config).put(|State(state): State<AdminState>, body| {
                apply_config(Caller::Operator, state, body)
            }),
        )
        .route("/admin/config/commit", post(commit))
        .route("/admin/config/rollback", post(rollback))
        .route("/admin/pause_generation", post(pause_generation))
        .route("/admin/continue_generation", post(continue_generation))
        .route("/admin/flush_cache", post(flush_cache))
        .route("/admin/shutdown", post(shutdown))
        .route("/admin/restart", post(restart))
        .route("/admin/update_weights", post(update_weights))
        .route("/admin/rebalance_experts", post(rebalance_experts))
        .route("/admin/sleep", post(sleep))
        .route("/admin/wake_up", post(wake_up))
        .route("/admin/load_lora_adapter", post(load_lora_adapter))
        .route("/admin/unload_lora_adapter", post(unload_lora_adapter))
        .route(
            "/admin/profile",
            get(profile_status).post(start_profile).delete(stop_profile),
        )
        .with_state(state)
}

/// `GET /admin/status`: what the deployment serves, the running and the
/// last operation, and the next event's sequence number.
async fn status(State(state): State<AdminState>) -> Json<serde_json::Value> {
    let admin = &state.admin;
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "deployment": state.leader.status(),
        "operation": service::lock(&admin.running).clone(),
        "last_operation": service::lock(&admin.last).clone(),
        "next_event": admin.next_event.load(Ordering::Relaxed),
        "readiness": service::lock(&admin.readiness).clone(),
    }))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EventsQuery {
    /// The last sequence number the caller read; 0 for every event kept.
    #[serde(default)]
    after: u64,
    /// Seconds to wait for an event when none is newer than `after`.
    #[serde(default)]
    wait: f64,
}

/// `GET /admin/events?after=N&wait=S`: the events after `N`, waiting up to
/// `S` seconds (at most 60) for one. `gap` says that events after `N` left
/// the log, and the caller should read the status again.
async fn events(State(state): State<AdminState>, Query(query): Query<EventsQuery>) -> Response {
    if !(query.wait.is_finite() && query.wait >= 0.0) {
        return ApiError::invalid_request("wait must be a number of seconds", Some("wait"))
            .into_response();
    }
    let wait = Duration::from_secs_f64(query.wait).min(MAX_WAIT);
    let (events, next, gap) = state.admin.events_after(query.after, wait).await;
    Json(json!({ "events": events, "next": next, "gap": gap })).into_response()
}

fn yes() -> bool {
    true
}

/// Runs operation `kind` on a task of its own and responds with it: once it
/// ends when `wait`, else at once (`202`).
async fn run<F>(admin: &Arc<Admin>, kind: &'static str, wait: bool, work: F) -> Response
where
    F: std::future::Future<Output = Result<serde_json::Value, (StatusCode, String)>>
        + Send
        + 'static,
{
    let operation = match admin.begin(kind) {
        Ok(operation) => operation,
        Err(other) => {
            return ApiError::conflict(format!(
                "operation {} ({}) is running",
                other.id, other.kind
            ))
            .into_response();
        }
    };
    let ended = tokio::spawn({
        let admin = Arc::clone(admin);
        let operation = operation.clone();
        async move {
            match work.await {
                Ok(result) => (StatusCode::OK, admin.end(operation, Ok(result))),
                Err((status, error)) => (status, admin.end(operation, Err(error))),
            }
        }
    });
    if !wait {
        return (StatusCode::ACCEPTED, Json(operation)).into_response();
    }
    match ended.await {
        Ok((status, operation)) => (status, Json(operation)).into_response(),
        Err(error) => {
            ApiError::server_error(format!("the operation's task ended: {error}")).into_response()
        }
    }
}

/// The body of `PUT /admin/config`: an [`ApplyRequest`], and whether the
/// response waits for the change.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigBody {
    #[serde(default)]
    replica: Option<usize>,
    /// The whole target configuration as TOML; the current one when unset.
    #[serde(default)]
    config: Option<String>,
    #[serde(default)]
    set: Vec<String>,
    #[serde(default)]
    in_flight: InFlight,
    #[serde(default)]
    dry_run: bool,
    /// Apply a configuration a rollback marked unavailable.
    #[serde(default)]
    force: bool,
    /// Seconds from now by which the update must be committed, or the
    /// daemon rolls it back.
    #[serde(default)]
    deadline_s: Option<f64>,
    #[serde(default = "yes")]
    wait: bool,
}

/// `GET /admin/config`: the configuration, its digest, the two slots'
/// epochs, the change in progress, the trial, the configurations a
/// rollback marked unavailable and every item of the GPUs' quotas
/// (`Leader::config_status`).
async fn config(State(state): State<AdminState>) -> Json<serde_json::Value> {
    let mut body = state.leader.config_status().await;
    // The epochs and the processes this daemon manages sit beside the
    // configuration and its epochs: `EpochRecord`s are the daemon's record
    // of what it executes, not the engine's (`crate::epoch`).
    if let Some(object) = body.as_object_mut() {
        let report = state.admin.epochs().report();
        object.insert("epochs".to_owned(), report["epochs"].clone());
        object.insert("processes".to_owned(), report["processes"].clone());
    }
    Json(body)
}

/// Who calls an admin route: the operator on its socket, which serves every route and option,
/// or a harness on the harness socket, which may set the package selections and the options
/// `admin_config.harness_options` lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Caller {
    Operator,
    Harness,
}

/// `PUT /admin/config` for `caller`: changes the deployment to a configuration
/// (`Leader::apply`), in place, by a hot restart of the daemon, or as an
/// update on trial. An empty body applies the current configuration and waits.
///
/// The `config` operation ends once the change is applied or its update
/// started, so a commit or a rollback can follow at once. With `wait`, the
/// response comes when the update reached its trial or ended before it.
async fn apply_config(caller: Caller, state: AdminState, body: axum::body::Bytes) -> Response {
    let body = if body.is_empty() {
        ConfigBody {
            wait: true,
            ..ConfigBody::default()
        }
    } else {
        match serde_json::from_slice::<ConfigBody>(&body) {
            Ok(body) => body,
            Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
        }
    };
    apply(&state, body, caller).await
}

/// Applies `body` for `caller`.
async fn apply(state: &AdminState, body: ConfigBody, caller: Caller) -> Response {
    let leader = Arc::clone(&state.leader);
    let admin = Arc::clone(&state.admin);
    let jobs = Arc::clone(&state.jobs);
    let restarter = state.restarter.clone();
    let wait = body.wait;
    let request = ApplyRequest {
        replica: body.replica,
        config: body.config,
        set: body.set,
        in_flight: body.in_flight,
        dry_run: body.dry_run,
        force: body.force,
        deadline_s: body.deadline_s,
    };
    if caller == Caller::Harness
        && let Err(refusal) = leader.harness_may_apply(&request)
    {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({ "error": { "message": refusal, "type": "forbidden" } })),
        )
            .into_response();
    }
    let mut phases = leader.updates.watch();
    let applying = Arc::clone(&leader);
    let operation = admin.begin("config");
    let operation = match operation {
        Ok(operation) => operation,
        Err(other) => {
            return ApiError::conflict(format!(
                "operation {} ({}) is running",
                other.id, other.kind
            ))
            .into_response();
        }
    };
    let number = operation.id;
    let ended = tokio::spawn({
        let admin = Arc::clone(&admin);
        async move {
            // The change's future holds the model files it resolves.
            let applied = match Box::pin(applying.apply(request, number)).await {
                Ok(Applied::Daemon {
                    config,
                    executable,
                    then,
                }) => match restarter {
                    None => Err((
                        StatusCode::BAD_REQUEST,
                        "the change is of the daemon's own options, and this daemon does not \
                         restart in place"
                            .to_owned(),
                    )),
                    Some(restarter) => {
                        let update = then.is_some();
                        restarter
                            .restart(executable, Some(*config), then, &applying, &admin, &jobs)
                            .await
                            .map(|pid| {
                                // The successor applies the rest of the change
                                // in an operation of its own.
                                json!({ "change": "daemon", "pid": pid, "then": update.then_some("update") })
                            })
                            .map_err(|error| {
                                (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
                            })
                    }
                },
                applied => applied_result(applied),
            };
            match applied {
                Ok(result) => (StatusCode::OK, admin.end(operation, Ok(result))),
                Err((status, error)) => (status, admin.end(operation, Err(error))),
            }
        }
    });
    if !wait {
        let running = service::lock(&admin.running).clone();
        return (StatusCode::ACCEPTED, Json(running)).into_response();
    }
    let (status, mut operation) = match ended.await {
        Ok(ended) => ended,
        Err(error) => {
            return ApiError::server_error(format!("the operation's task ended: {error}"))
                .into_response();
        }
    };
    let started = operation
        .result
        .as_ref()
        .is_some_and(|result| result.get("update").is_some());
    if !started {
        return (status, Json(operation)).into_response();
    }

    // The update runs on by itself: the response states it as it stands when
    // its trial begins or it ends before one.
    loop {
        phases.borrow_and_update();
        let moving = matches!(
            leader.updates.phase_of(number),
            Some(Phase::Preparing | Phase::Switching)
        );
        if !moving || phases.changed().await.is_err() {
            break;
        }
    }
    let update = leader.updates.status().unwrap_or_default();
    let status = if matches!(
        leader.updates.phase_of(number),
        Some(Phase::Trial | Phase::Committed)
    ) {
        StatusCode::OK
    } else {
        StatusCode::CONFLICT
    };
    // The operation is the one that started the update; its result states
    // the update as it stands now.
    if let Some(result) = operation.result.as_mut() {
        result["update"] = update;
    }
    (status, Json(operation)).into_response()
}

/// What `PUT /admin/config` states for a change that needs no hot restart of
/// the daemon, or why it failed.
fn applied_result(
    applied: Result<Applied, UpdateError>,
) -> Result<serde_json::Value, (StatusCode, String)> {
    match applied {
        Ok(Applied::Unchanged) => Ok(json!({ "change": "none" })),
        Ok(Applied::InPlace(report)) => Ok(json!({ "change": "in_place", "in_place": report })),
        Ok(Applied::DryRun(report)) => Ok(json!({ "change": "dry_run", "dry_run": report })),
        Ok(Applied::Update(report)) => {
            Ok(json!({ "change": report["change"].clone(), "update": report }))
        }
        Ok(Applied::Daemon { .. }) => Err((
            StatusCode::CONFLICT,
            "the daemon's own options changed again after its hot restart".to_owned(),
        )),
        Err(error) => Err((
            update_status(&error),
            format!("{:#}", anyhow::Error::from(error)),
        )),
    }
}

/// Applies `request`, the rest of a change whose daemon options the hot
/// restart that started this daemon applied, as an operation of its own
/// (`config`), which `GET /admin/status` reports.
pub(crate) fn apply_after_restart(
    admin: &Arc<Admin>,
    leader: &Arc<crate::leader::Leader>,
    request: ApplyRequest,
) {
    let operation = match admin.begin("config") {
        Ok(operation) => operation,
        Err(other) => {
            admin.record(
                "operation",
                format!(
                    "the change after the hot restart did not start: operation {} ({}) runs",
                    other.id, other.kind
                ),
            );
            return;
        }
    };
    let admin = Arc::clone(admin);
    let leader = Arc::clone(leader);
    tokio::spawn(async move {
        let applied = Box::pin(leader.apply(request, operation.id)).await;
        let result = applied_result(applied).map_err(|(_, error)| error);
        admin.end(operation, result);
    });
}

/// `POST /admin/config/commit`: commits the update in flight
/// (`Leader::commit`).
async fn commit(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "commit", true, async move {
        leader.commit().await.map_err(|error| {
            (
                update_status(&error),
                format!("{:#}", anyhow::Error::from(error)),
            )
        })
    })
    .await
}

/// The body of `POST /admin/config/rollback`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RollbackBody {
    /// Why the update is rolled back, which the unavailable mark records.
    #[serde(default)]
    reason: Option<String>,
}

/// `POST /admin/config/rollback`: rolls the update in flight back
/// (`Leader::rollback`).
async fn rollback(State(state): State<AdminState>, body: axum::body::Bytes) -> Response {
    let body = if body.is_empty() {
        RollbackBody::default()
    } else {
        match serde_json::from_slice::<RollbackBody>(&body) {
            Ok(body) => body,
            Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
        }
    };
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "rollback", true, async move {
        let reason = body
            .reason
            .unwrap_or_else(|| "the trial was rolled back by decision".to_owned());
        leader.rollback(reason).await.map_err(|error| {
            (
                update_status(&error),
                format!("{:#}", anyhow::Error::from(error)),
            )
        })
    })
    .await
}

/// The status of a failed update.
fn update_status(error: &UpdateError) -> StatusCode {
    match error {
        UpdateError::InProgress => StatusCode::CONFLICT,
        UpdateError::Refused(_) => StatusCode::BAD_REQUEST,
        UpdateError::NoEngine => StatusCode::SERVICE_UNAVAILABLE,
        UpdateError::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// `POST /admin/pause_generation`: new requests wait; those in flight go on.
async fn pause_generation(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "pause_generation", true, async move {
        let already = leader.pause();
        Ok(json!({ "already_paused": already }))
    })
    .await
}

/// `POST /admin/continue_generation`: new requests reach the engine again.
async fn continue_generation(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "continue_generation", true, async move {
        leader.resume();
        Ok(json!({}))
    })
    .await
}

/// `POST /admin/flush_cache`: the engine revokes every cached prefix page.
async fn flush_cache(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "flush_cache", true, async move {
        leader
            .flush_cache()
            .map(|()| json!({}))
            .map_err(|error| (update_status(&error), error.to_string()))
    })
    .await
}

/// The body of `POST /admin/update_weights`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateWeightsBody {
    /// Another checkpoint of the served model (a local directory or a
    /// Hugging Face repository), or a plugin's weight source by its URL.
    model_path: crate::leader::WeightSource,
    /// How the requests in flight continue: `keep` (the default), `recompute`
    /// or `abort`.
    #[serde(default)]
    in_flight: crate::leader::WeightsInFlight,
}

/// `POST /admin/update_weights`: the engine rewrites its weights in place
/// from another checkpoint of the served model (SGLang's
/// `update_weights_from_disk`).
async fn update_weights(State(state): State<AdminState>, body: axum::body::Bytes) -> Response {
    let body: UpdateWeightsBody = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
    };
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "update_weights", true, async move {
        leader
            .update_weights(&body.model_path, body.in_flight)
            .await
            .map(|results| {
                json!({
                    "model_path": body.model_path,
                    "in_flight": body.in_flight,
                    "results": results.iter().map(weight_report).collect::<Vec<_>>(),
                })
            })
            .map_err(|error| {
                (
                    update_status(&error),
                    format!("{:#}", anyhow::Error::from(error)),
                )
            })
    })
    .await
}

/// Redistribute experts within each selected replica's current epoch.
async fn rebalance_experts(State(state): State<AdminState>, body: axum::body::Bytes) -> Response {
    let request: crate::experts::Rebalance = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
    };
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "rebalance_experts", true, async move {
        leader
            .rebalance_experts(request)
            .await
            .map_err(|error| (update_status(&error), format!("{error:#}")))
    })
    .await
}

/// `POST /admin/sleep`: the engine gives back its KV cache and CUDA graph
/// storage and keeps its weights (vLLM's `sleep`); new requests wait until
/// `/admin/wake_up`.
async fn sleep(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "sleep", true, async move {
        leader
            .sleep()
            .await
            .map(|()| json!({ "asleep": true }))
            .map_err(|error| {
                (
                    update_status(&error),
                    format!("{:#}", anyhow::Error::from(error)),
                )
            })
    })
    .await
}

/// `POST /admin/wake_up`: the engine restores what `/admin/sleep` gave back,
/// and the requests that waited are served.
async fn wake_up(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "wake_up", true, async move {
        leader
            .wake()
            .await
            .map(|()| json!({ "asleep": false }))
            .map_err(|error| {
                (
                    update_status(&error),
                    format!("{:#}", anyhow::Error::from(error)),
                )
            })
    })
    .await
}

/// One rank's report of a change of its weights, as the admin API answers it.
fn weight_report(report: &proto::request::wire::WeightReport) -> serde_json::Value {
    json!({
        "group": report.group,
        "rank": report.rank,
        "identity": report.identity,
        "identities": report.identities,
        "moved": report.moved,
    })
}

/// The request's JSON object body.
fn json_object(body: &[u8]) -> Result<serde_json::Value, Box<Response>> {
    match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(value) if value.is_object() => Ok(value),
        Ok(_) => Err(Box::new(
            ApiError::invalid_request("the body is not a JSON object", None).into_response(),
        )),
        Err(error) => Err(Box::new(
            ApiError::json_parse_error(error.to_string()).into_response(),
        )),
    }
}

/// `POST /admin/load_lora_adapter`: the workers load a LoRA adapter
/// (`{"lora_name": NAME, ...}`, vLLM's endpoint of that name) into a free
/// slot of the plugin's adapter pool, and requests naming it as their model
/// compute with it.
async fn load_lora_adapter(State(state): State<AdminState>, body: axum::body::Bytes) -> Response {
    let body = match json_object(&body) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "load_lora_adapter", true, async move {
        leader
            .load_adapter(body)
            .await
            .map(|results| json!({ "results": results }))
            .map_err(|error| {
                (
                    update_status(&error),
                    format!("{:#}", anyhow::Error::from(error)),
                )
            })
    })
    .await
}

/// `POST /admin/unload_lora_adapter`: the workers unload the LoRA adapter
/// `{"lora_name": NAME}` names.
async fn unload_lora_adapter(State(state): State<AdminState>, body: axum::body::Bytes) -> Response {
    let body = match json_object(&body) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "unload_lora_adapter", true, async move {
        leader
            .unload_adapter(body)
            .await
            .map(|results| json!({ "results": results }))
            .map_err(|error| {
                (
                    update_status(&error),
                    format!("{:#}", anyhow::Error::from(error)),
                )
            })
    })
    .await
}

/// `POST /admin/profile`: starts a profiling session
/// (`proto::profile::SessionRequest`) on every worker of the executing epoch
/// at the engine's next step boundary (`crate::profiling`), and responds with
/// it: its identity, each worker's file and collectors. One runs at a time:
/// another is refused (`400`) until it ends.
async fn start_profile(State(state): State<AdminState>, body: axum::body::Bytes) -> Response {
    let request: proto::profile::SessionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
    };
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "profile", true, async move {
        leader.start_profile(request).await.map_err(|error| {
            (
                update_status(&error),
                format!("{:#}", anyhow::Error::from(error)),
            )
        })
    })
    .await
}

/// `GET /admin/profile`: the running session, or the last that ended: its
/// state, the steps and bytes its workers recorded, the step time it costs
/// and what its quality-of-service guard did; `null` before any.
async fn profile_status(State(state): State<AdminState>) -> Json<serde_json::Value> {
    Json(json!({ "session": state.leader.profile_status() }))
}

/// `DELETE /admin/profile`: ends the running session at the engine's next
/// step boundary, every worker writing its file, and responds with it.
async fn stop_profile(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, "profile", true, async move {
        leader.stop_profile("operator").await.map_err(|error| {
            (
                update_status(&error),
                format!("{:#}", anyhow::Error::from(error)),
            )
        })
    })
    .await
}

/// The body of `POST /admin/restart`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RestartBody {
    /// The `uniserve` executable of the build to restart as; this daemon's
    /// own when unset.
    #[serde(default)]
    executable: Option<String>,
    #[serde(default = "yes")]
    wait: bool,
}

/// `POST /admin/restart`: hands the deployment to a successor daemon, which
/// serves from then on while this one finishes the requests it took
/// (`crate::restart`). Responds once the successor serves, with its process
/// id.
async fn restart(State(state): State<AdminState>, body: axum::body::Bytes) -> Response {
    let Some(restarter) = state.restarter.clone() else {
        return ApiError::invalid_request("this daemon does not restart in place", None)
            .into_response();
    };
    let body = if body.is_empty() {
        RestartBody {
            executable: None,
            wait: true,
        }
    } else {
        match serde_json::from_slice::<RestartBody>(&body) {
            Ok(body) => body,
            Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
        }
    };
    let leader = Arc::clone(&state.leader);
    let admin = Arc::clone(&state.admin);
    let jobs = Arc::clone(&state.jobs);
    run(&state.admin, "restart", body.wait, async move {
        restarter
            .restart(body.executable, None, None, &leader, &admin, &jobs)
            .await
            .map(|pid| json!({ "pid": pid }))
            .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")))
    })
    .await
}

/// The body of `POST /admin/shutdown`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShutdownBody {
    /// Release the deployment's weights and caches at the host's Storage
    /// Server once its engine is down (the default); otherwise they stay
    /// for a restart of the same deployment.
    #[serde(default = "yes")]
    retire: bool,
}

/// `POST /admin/shutdown`: brings the deployment down as SIGTERM does,
/// finishing what is in flight within `server_config.shutdown_timeout`, and
/// releases its storage unless `retire` is false. Responds at once (`202`).
async fn shutdown(State(state): State<AdminState>, body: axum::body::Bytes) -> Response {
    let body = if body.is_empty() {
        ShutdownBody { retire: true }
    } else {
        match serde_json::from_slice::<ShutdownBody>(&body) {
            Ok(body) => body,
            Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
        }
    };
    let admin = Arc::clone(&state.admin);
    run(&state.admin, "shutdown", false, async move {
        admin.retire.store(body.retire, Ordering::Release);
        admin.record(
            "deployment",
            if body.retire {
                "shutting down; the deployment's storage is released after"
            } else {
                "shutting down; the deployment's storage stays retained"
            },
        );
        admin.shutdown.cancel();
        Ok(json!({ "retire": body.retire }))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn events_are_read_after_a_sequence_number_and_waited_for() {
        let admin = Admin::with_epochs(
            CancellationToken::new(),
            Arc::new(AtomicBool::new(false)),
            Arc::new(crate::epoch::Epochs::new()),
        );
        admin.record("deployment", "first");
        admin.record("deployment", "second");
        let (events, next, gap) = admin.events_after(1, Duration::ZERO).await;
        assert_eq!(
            events
                .iter()
                .map(|event| event.message.as_str())
                .collect::<Vec<_>>(),
            vec!["second"]
        );
        assert_eq!((next, gap), (2, false));

        // A reader past the last event waits for the next one.
        let waiting = tokio::spawn({
            let admin = Arc::clone(&admin);
            async move { admin.events_after(2, Duration::from_secs(5)).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        admin.record("engine", "third");
        let (events, next, _) = waiting.await.unwrap();
        assert_eq!(events[0].message, "third");
        assert_eq!(next, 3);
    }

    #[tokio::test]
    async fn a_reader_behind_the_log_is_told_of_the_gap() {
        let admin = Admin::with_epochs(
            CancellationToken::new(),
            Arc::new(AtomicBool::new(false)),
            Arc::new(crate::epoch::Epochs::new()),
        );
        for index in 0..EVENT_CAPACITY + 2 {
            admin.record("deployment", format!("event {index}"));
        }
        let (events, _, gap) = admin.events_after(0, Duration::ZERO).await;
        assert!(gap);
        assert_eq!(events.len(), EVENT_CAPACITY);
    }

    /// A hot restart's successor reports the epochs and processes of the
    /// records its leader keeps, not records of its own.
    #[test]
    fn a_resumed_admin_reports_its_leader_s_records() {
        let epochs = Arc::new(crate::epoch::Epochs::new());
        epochs.set_processes(vec![crate::epoch::ManagedProcess {
            group: crate::engine::control::ENGINE_GROUP.to_owned(),
            rank: 0,
            pid: Some(42),
            epoch: Some(1),
            build: String::new(),
            generation: 1,
            state: crate::epoch::ProcessState::Serving,
        }]);
        let predecessor = Admin::with_epochs(
            CancellationToken::new(),
            Arc::new(AtomicBool::new(false)),
            Arc::new(crate::epoch::Epochs::new()),
        );
        let admin = Admin::resume(
            CancellationToken::new(),
            Arc::new(AtomicBool::new(false)),
            predecessor.hand_over(),
            Arc::clone(&epochs),
        );
        let report = admin.epochs().report();
        assert_eq!(report["processes"][0]["pid"], serde_json::json!(42));
    }

    #[tokio::test]
    async fn one_operation_runs_at_a_time() {
        let admin = Admin::with_epochs(
            CancellationToken::new(),
            Arc::new(AtomicBool::new(false)),
            Arc::new(crate::epoch::Epochs::new()),
        );
        let first = admin.begin("update").unwrap();
        let refused = admin.begin("flush_cache").unwrap_err();
        assert_eq!(refused.id, first.id);
        let ended = admin.end(first, Err("it failed".to_owned()));
        assert_eq!(
            (ended.state.as_str(), ended.error.as_deref()),
            ("failed", Some("it failed"))
        );
        assert!(admin.begin("flush_cache").is_ok());
    }
}

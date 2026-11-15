//! The admin API (`/admin`): the node's status, its event log, and the operations that change
//! its deployment while it serves. The requests and responses are the interface crate's schema
//! (`proto::admin`), which clients read as the daemon serves it.
//!
//! The endpoints follow SGLang's and vLLM's (`get_server_info`,
//! `pause_generation`, `continue_generation`, `flush_cache`). A write
//! operation runs on a task of its own, so it completes, or fails and says
//! so in the event log, whether or not the client that asked for it stays
//! connected; one runs at a time, and another is refused with `409`. An
//! external manager attaches by reading `status` and `events` and detaches by
//! stopping; the daemon keeps no state of it.
//!
//! The event log and the operations' record are the node's ([`Journal`]): every deployment
//! the node serves writes them, and a hot restart hands them to the successor, so a reader
//! follows one log across the node's deployments and restarts.
//!
//! The API is served apart from the client API, on two Unix sockets
//! ([`serve_routes`]), and access is their file permissions:
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
use proto::admin::{
    ApplyRequest, ConfigChange, Event, EventBody, EventsPage, EventsQuery, Following, NodeState,
    Operation, OperationKind, OperationResponse, OperationState, Paused, Phase, Restart, Restarted,
    RollbackRequest, Shutdown, Status, UpdateReport,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::api::openai::ApiError;
use crate::leader::{Leader, UpdateError};
use crate::updates::change::Applied;

/// The node's event log and operations' record as a hot restart hands them to the successor.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct JournalHandoff {
    events: Vec<Event>,
    next_event: u64,
    last: Option<Operation>,
    next_operation: u64,
}

/// Why an operation does not start.
#[derive(Debug)]
pub(crate) enum Refusal {
    /// Another operation runs.
    Running(Operation),
    /// A hot restart handed the node to a successor, which takes the operations from then on.
    HandedOver,
}

impl Refusal {
    /// The `409` the refusal answers.
    pub(crate) fn response(&self) -> Response {
        let message = match self {
            Self::Running(other) => format!("operation {} ({}) is running", other.id, other.kind),
            Self::HandedOver => "the node was handed to a successor daemon".to_owned(),
        };
        ApiError::conflict(message).into_response()
    }
}

/// The node's journal: its event log and its operations' record. The node keeps one for its
/// whole life, every deployment it serves writes it, and a hot restart hands it on.
pub(crate) struct Journal {
    /// What the event log keeps and how long a reader waits (`admin_config.event_log`).
    log: crate::options::EventLog,
    events: Mutex<VecDeque<Event>>,
    next_event: AtomicU64,
    appended: Notify,
    running: Mutex<Option<Operation>>,
    last: Mutex<Option<Operation>>,
    next_operation: AtomicU64,
    /// Set once a hot restart handed the node to a successor, which takes the operations from
    /// then on.
    handoff_sent: AtomicBool,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

impl Journal {
    /// An empty journal under `log`.
    pub(crate) fn new(log: crate::options::EventLog) -> Arc<Self> {
        Arc::new(Self {
            log,
            events: Mutex::new(VecDeque::with_capacity(log.capacity.get())),
            next_event: AtomicU64::new(1),
            appended: Notify::new(),
            running: Mutex::new(None),
            last: Mutex::new(None),
            next_operation: AtomicU64::new(1),
            handoff_sent: AtomicBool::new(false),
        })
    }

    /// A journal that goes on with the event log and the operations' record of the daemon a hot
    /// restart replaced.
    pub(crate) fn resume(handoff: JournalHandoff, log: crate::options::EventLog) -> Arc<Self> {
        let journal = Self::new(log);
        *service::lock(&journal.events) = handoff.events.into_iter().collect();
        journal
            .next_event
            .store(handoff.next_event, Ordering::Relaxed);
        *service::lock(&journal.last) = handoff.last;
        journal
            .next_operation
            .store(handoff.next_operation, Ordering::Relaxed);
        journal
    }

    /// The event log and the operations' record, for the successor of a hot restart to go on
    /// with ([`Journal::resume`]). No operation starts here from then on.
    pub(crate) fn hand_over(&self) -> JournalHandoff {
        self.handoff_sent.store(true, Ordering::Release);
        JournalHandoff {
            events: service::lock(&self.events).iter().cloned().collect(),
            next_event: self.next_event.load(Ordering::Relaxed),
            last: service::lock(&self.last).clone(),
            next_operation: self.next_operation.load(Ordering::Relaxed),
        }
    }

    /// Appends an event of category and data `body`, and wakes the readers that wait for one.
    pub(crate) fn record(&self, body: EventBody, message: impl Into<String>) {
        let mut events = service::lock(&self.events);
        if events.len() == self.log.capacity.get() {
            events.pop_front();
        }
        events.push_back(Event {
            seq: self.next_event.fetch_add(1, Ordering::Relaxed),
            time: now(),
            message: message.into(),
            body,
        });
        drop(events);
        self.appended.notify_waiters();
    }

    /// The node's status when it serves what `node` states: the running and the last operation
    /// and the next event's sequence number beside it.
    pub(crate) fn status(&self, node: NodeState) -> Status {
        Status {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            node,
            operation: service::lock(&self.running).clone(),
            last_operation: service::lock(&self.last).clone(),
            next_event: self.next_event.load(Ordering::Relaxed),
        }
    }

    /// The events after `after`, waiting up to `wait` for one when there is none, and the
    /// sequence number to read from next. `gap` is whether events after `after` already left
    /// the log.
    async fn events_after(&self, after: u64, wait: Duration) -> EventsPage {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let notified = self.appended.notified();
            {
                let events = service::lock(&self.events);
                let gap = events.front().is_some_and(|first| first.seq > after + 1);
                let found: Vec<Event> = events
                    .iter()
                    .filter(|event| event.seq > after)
                    .cloned()
                    .collect();
                let next = events.back().map_or(after, |last| last.seq.max(after));
                if !found.is_empty() || tokio::time::Instant::now() >= deadline {
                    return EventsPage {
                        events: found,
                        next,
                        gap,
                    };
                }
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                continue;
            }
        }
    }

    /// Starts an operation of `kind`; refused while another runs or once the node was handed
    /// over.
    pub(crate) fn begin(&self, kind: OperationKind) -> Result<Operation, Refusal> {
        let mut running = service::lock(&self.running);
        if let Some(other) = running.as_ref() {
            return Err(Refusal::Running(other.clone()));
        }
        if self.handoff_sent.load(Ordering::Acquire) {
            return Err(Refusal::HandedOver);
        }
        let operation = Operation {
            id: self.next_operation.fetch_add(1, Ordering::Relaxed),
            kind,
            state: OperationState::Running,
            started: now(),
            ended: None,
            error: None,
        };
        *running = Some(operation.clone());
        drop(running);
        self.record(
            EventBody::Operation(operation.clone()),
            format!("{kind} {} started", operation.id),
        );
        Ok(operation)
    }

    /// Ends `operation`, the running one: succeeded, or failed with the error.
    pub(crate) fn end(&self, mut operation: Operation, ended: Result<(), String>) -> Operation {
        *service::lock(&self.running) = None;
        operation.ended = Some(now());
        let message = match ended {
            Ok(()) => {
                operation.state = OperationState::Succeeded;
                format!("{} {} succeeded", operation.kind, operation.id)
            }
            Err(error) => {
                operation.state = OperationState::Failed;
                let message = format!("{} {} failed: {error}", operation.kind, operation.id);
                operation.error = Some(error);
                message
            }
        };
        *service::lock(&self.last) = Some(operation.clone());
        self.record(EventBody::Operation(operation.clone()), message);
        operation
    }
}

/// A deployment's admin state: the node's journal it writes, and what ends the deployment.
pub(crate) struct Admin {
    /// The node's journal.
    pub(crate) journal: Arc<Journal>,
    /// Cancelled to stop serving: a shutdown.
    shutdown: CancellationToken,
    /// Whether the shutdown releases the deployment's storage at the host's
    /// Storage Server, once its engine is down.
    retire: Arc<AtomicBool>,
    /// The daemon's readiness record once it serves, which `GET /admin/status` reports.
    readiness: Mutex<Option<service::Readiness>>,
}

impl Admin {
    /// The admin state of a deployment that writes the node's `journal` and ends at `shutdown`.
    pub(crate) fn new(
        shutdown: CancellationToken,
        retire: Arc<AtomicBool>,
        journal: Arc<Journal>,
    ) -> Arc<Self> {
        Arc::new(Self {
            journal,
            shutdown,
            retire,
            readiness: Mutex::new(None),
        })
    }

    /// The daemon's readiness record, which `GET /admin/status` reports from now on.
    pub(crate) fn set_readiness(&self, readiness: service::Readiness) {
        *service::lock(&self.readiness) = Some(readiness);
    }

    /// Appends an event of the deployment's (`deployment`).
    pub(crate) fn record(&self, message: impl Into<String>) {
        self.journal.record(EventBody::Deployment, message);
    }

    /// Records the leader's changes of state, as the leader publishes them, and every phase its
    /// updates enter, until the server stops.
    pub(crate) fn watch(self: &Arc<Self>, leader: &Arc<Leader>) {
        if let Some(mut transitions) = leader.updates.transitions() {
            let journal = Arc::clone(&self.journal);
            tokio::spawn(async move {
                while let Some(transition) = transitions.recv().await {
                    let message = format!("update {}: {}", transition.update, transition.phase);
                    journal.record(EventBody::Update(Box::new(transition.report)), message);
                }
            });
        }
        // The changes are taken before the status they follow is read, so none is missed.
        let mut changes = leader.status_changes();
        let mut previous = leader.status();
        let admin = Arc::clone(self);
        let leader = Arc::downgrade(leader);
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = admin.shutdown.cancelled() => return,
                    () = changes.changed() => {}
                }
                let Some(status) = leader.upgrade().map(|leader| leader.status()) else {
                    return;
                };
                if previous.engine != status.engine {
                    let reason = status
                        .reason
                        .as_deref()
                        .map(|reason| format!(": {reason}"))
                        .unwrap_or_default();
                    admin.journal.record(
                        EventBody::Engine {
                            from: previous.engine,
                            to: status.engine,
                            reason: status.reason.clone(),
                        },
                        format!(
                            "the engine went from {} to {}{reason}",
                            previous.engine, status.engine
                        ),
                    );
                }
                if previous.paused != status.paused {
                    admin.record(if status.paused {
                        "new requests wait"
                    } else {
                        "new requests are served"
                    });
                }
                previous = status;
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
    /// Binds the admin sockets `admin.sock` and `harness.sock` in the daemon's instance
    /// directory `instance`, each taking its path over from whoever holds it (a hot restart's
    /// predecessor): the operator's with mode `0600`, the harness's with `0660`.
    ///
    /// # Errors
    ///
    /// Fails when a socket cannot be bound at its path.
    pub(crate) fn bind(instance: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context as _;

        let paths = (instance.join("admin.sock"), instance.join("harness.sock"));
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
        .route("/admin/events", get(deployment_events))
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
        .route("/admin/events", get(deployment_events))
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

/// `GET /admin/status` while the deployment serves: the deployment's status, the node's
/// operations and the next event's sequence number. The node answers the route itself while no
/// deployment serves (`super::node`).
async fn status(State(state): State<AdminState>) -> Json<Status> {
    let readiness = service::lock(&state.admin.readiness).clone().map(Box::new);
    Json(state.admin.journal.status(NodeState::Serving {
        deployment: Box::new(state.leader.status()),
        readiness,
    }))
}

/// `GET /admin/events?after=N&wait=S`: the events after `N`, waiting up to
/// `S` seconds (at most `admin_config.event_log.longest_wait`) for one. `gap` says that events
/// after `N` left the log, and the caller should read the status again.
pub(crate) async fn events(journal: &Journal, query: EventsQuery) -> Response {
    if !(query.wait.is_finite() && query.wait >= 0.0) {
        return ApiError::invalid_request(
            "wait must be a number of seconds",
            Some("wait".to_owned()),
        )
        .into_response();
    }
    let wait = Duration::from_secs_f64(query.wait).min(journal.log.longest_wait);
    Json(journal.events_after(query.after, wait).await).into_response()
}

/// `GET /admin/events` while the deployment serves: the node's journal, which the deployment
/// writes.
async fn deployment_events(
    State(state): State<AdminState>,
    Query(query): Query<EventsQuery>,
) -> Response {
    events(&state.admin.journal, query).await
}

/// A request's JSON body as `T`: `T`'s default for an empty body, else the body, which must
/// read as `T`.
pub(crate) fn body<T: DeserializeOwned + Default>(bytes: &[u8]) -> Result<T, Box<Response>> {
    if bytes.is_empty() {
        return Ok(T::default());
    }
    serde_json::from_slice(bytes)
        .map_err(|error| Box::new(ApiError::json_parse_error(error.to_string()).into_response()))
}

/// Runs an operation of `kind` on a task of its own and responds with it and its result: once
/// it ends when `wait`, else at once (`202`).
async fn run<R, F>(admin: &Arc<Admin>, kind: OperationKind, wait: bool, work: F) -> Response
where
    R: Serialize + Send + 'static,
    F: std::future::Future<Output = Result<R, (StatusCode, String)>> + Send + 'static,
{
    let journal = &admin.journal;
    let operation = match journal.begin(kind) {
        Ok(operation) => operation,
        Err(refusal) => return refusal.response(),
    };
    let ended = tokio::spawn({
        let journal = Arc::clone(journal);
        let operation = operation.clone();
        async move {
            match work.await {
                Ok(result) => (
                    StatusCode::OK,
                    OperationResponse {
                        operation: journal.end(operation, Ok(())),
                        result: Some(result),
                    },
                ),
                Err((status, error)) => (
                    status,
                    OperationResponse {
                        operation: journal.end(operation, Err(error)),
                        result: None,
                    },
                ),
            }
        }
    });
    if !wait {
        let running = OperationResponse::<R> {
            operation,
            result: None,
        };
        return (StatusCode::ACCEPTED, Json(running)).into_response();
    }
    match ended.await {
        Ok((status, response)) => (status, Json(response)).into_response(),
        Err(error) => {
            ApiError::server_error(format!("the operation's task ended: {error}")).into_response()
        }
    }
}

/// `GET /admin/config`: the configuration, its digest, the replicas' epochs, the update in
/// flight or the last one, the configurations a rollback marked unavailable, every item of the
/// GPUs' quotas, the epochs' records and the processes the daemon manages
/// (`Leader::config_status`).
async fn config(State(state): State<AdminState>) -> Response {
    match state.leader.config_status().await {
        Ok(status) => Json(status).into_response(),
        Err(error) => ApiError::server_error(format!("{error:#}")).into_response(),
    }
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
async fn apply_config(caller: Caller, state: AdminState, bytes: axum::body::Bytes) -> Response {
    let request: ApplyRequest = match body(&bytes) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    apply(&state, request, caller).await
}

/// Applies `request` for `caller`.
async fn apply(state: &AdminState, request: ApplyRequest, caller: Caller) -> Response {
    let leader = Arc::clone(&state.leader);
    let journal = Arc::clone(&state.admin.journal);
    let admin = Arc::clone(&state.admin);
    let jobs = Arc::clone(&state.jobs);
    let restarter = state.restarter.clone();
    let wait = request.wait;
    if caller == Caller::Harness
        && let Err(refusal) = leader.harness_may_apply(&request)
    {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "error": { "message": refusal, "type": "forbidden" } })),
        )
            .into_response();
    }
    let mut phases = leader.updates.watch();
    let applying = Arc::clone(&leader);
    let operation = match journal.begin(OperationKind::Config) {
        Ok(operation) => operation,
        Err(refusal) => return refusal.response(),
    };
    let number = operation.id;
    let ended = tokio::spawn({
        let journal = Arc::clone(&journal);
        let operation = operation.clone();
        async move {
            // The change's future holds the model files it resolves.
            let applied = match Box::pin(applying.apply(request, number)).await {
                Ok(Applied::Daemon { config, then }) => match restarter {
                    None => Err((
                        StatusCode::BAD_REQUEST,
                        "the change is of the daemon's own options, and this daemon does not \
                         restart in place"
                            .to_owned(),
                    )),
                    Some(restarter) => {
                        // The successor applies the rest of the change in an operation of its
                        // own.
                        let then_follows = then.is_some().then_some(Following::Update);
                        restarter
                            .restart(Some(*config), then, &applying, &admin, &jobs)
                            .await
                            .map(|pid| ConfigChange::Daemon {
                                pid,
                                then: then_follows,
                            })
                            .map_err(|error| {
                                (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
                            })
                    }
                },
                applied => applied_result(applied),
            };
            match applied {
                Ok(change) => (
                    StatusCode::OK,
                    OperationResponse {
                        operation: journal.end(operation, Ok(())),
                        result: Some(change),
                    },
                ),
                Err((status, error)) => (
                    status,
                    OperationResponse {
                        operation: journal.end(operation, Err(error)),
                        result: None,
                    },
                ),
            }
        }
    });
    if !wait {
        let running = OperationResponse::<ConfigChange> {
            operation,
            result: None,
        };
        return (StatusCode::ACCEPTED, Json(running)).into_response();
    }
    let (status, mut response) = match ended.await {
        Ok(ended) => ended,
        Err(error) => {
            return ApiError::server_error(format!("the operation's task ended: {error}"))
                .into_response();
        }
    };
    let Some(ConfigChange::Generation { update } | ConfigChange::Epoch { update }) =
        response.result.as_mut()
    else {
        return (status, Json(response)).into_response();
    };

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
    let status = if matches!(
        leader.updates.phase_of(number),
        Some(Phase::Trial | Phase::Committed)
    ) {
        StatusCode::OK
    } else {
        StatusCode::CONFLICT
    };
    // The operation is the one that started the update; its result states the update as it
    // stands now: the one in flight, or the one that ended last, which is this one.
    if let Some(standing) = leader.updates.status() {
        *update = standing;
    }
    (status, Json(response)).into_response()
}

/// What `PUT /admin/config` states for a change that needs no hot restart of
/// the daemon, or why it failed.
fn applied_result(
    applied: Result<Applied, UpdateError>,
) -> Result<ConfigChange, (StatusCode, String)> {
    match applied {
        Ok(Applied::Unchanged) => Ok(ConfigChange::None),
        Ok(Applied::InPlace(in_place)) => Ok(ConfigChange::InPlace { in_place }),
        Ok(Applied::DryRun(dry_run)) => Ok(ConfigChange::DryRun { dry_run }),
        Ok(Applied::Update(update)) => Ok(started(*update)),
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

/// The change an update that started states, by its form.
fn started(update: UpdateReport) -> ConfigChange {
    match update.change {
        proto::admin::ChangeForm::Generation => ConfigChange::Generation { update },
        proto::admin::ChangeForm::Epoch => ConfigChange::Epoch { update },
    }
}

/// Applies `request`, the rest of a change whose daemon options the hot
/// restart that started this daemon applied, as an operation of its own
/// (`config`), which `GET /admin/status` and the event log report.
pub(crate) fn apply_after_restart(
    admin: &Arc<Admin>,
    leader: &Arc<crate::leader::Leader>,
    request: ApplyRequest,
) {
    let journal = Arc::clone(&admin.journal);
    let operation = match journal.begin(OperationKind::Config) {
        Ok(operation) => operation,
        Err(refusal) => {
            admin.record(format!(
                "the change after the hot restart did not start: {}",
                match refusal {
                    Refusal::Running(other) =>
                        format!("operation {} ({}) runs", other.id, other.kind),
                    Refusal::HandedOver => "the node was handed to a successor".to_owned(),
                }
            ));
            return;
        }
    };
    let leader = Arc::clone(leader);
    tokio::spawn(async move {
        let applied = Box::pin(leader.apply(request, operation.id)).await;
        let ended = applied_result(applied)
            .map(drop)
            .map_err(|(_, error)| error);
        journal.end(operation, ended);
    });
}

/// `POST /admin/config/commit`: commits the update in flight
/// (`Leader::commit`).
async fn commit(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, OperationKind::Commit, true, async move {
        leader.commit().await.map_err(|error| {
            (
                update_status(&error),
                format!("{:#}", anyhow::Error::from(error)),
            )
        })
    })
    .await
}

/// `POST /admin/config/rollback`: rolls the update in flight back
/// (`Leader::rollback`).
async fn rollback(State(state): State<AdminState>, bytes: axum::body::Bytes) -> Response {
    let request: RollbackRequest = match body(&bytes) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let leader = Arc::clone(&state.leader);
    run(&state.admin, OperationKind::Rollback, true, async move {
        let reason = request
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
    run(
        &state.admin,
        OperationKind::PauseGeneration,
        true,
        async move {
            Ok(Paused {
                already_paused: leader.pause(),
            })
        },
    )
    .await
}

/// `POST /admin/continue_generation`: new requests reach the engine again.
async fn continue_generation(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(
        &state.admin,
        OperationKind::ContinueGeneration,
        true,
        async move {
            leader.resume();
            Ok(())
        },
    )
    .await
}

/// `POST /admin/flush_cache`: the engine revokes every cached prefix page.
async fn flush_cache(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, OperationKind::FlushCache, true, async move {
        leader
            .flush_cache()
            .map_err(|error| (update_status(&error), error.to_string()))
    })
    .await
}

/// The body of `POST /admin/update_weights`.
#[derive(Debug, serde::Deserialize)]
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
    run(
        &state.admin,
        OperationKind::UpdateWeights,
        true,
        async move {
            leader
                .update_weights(&body.model_path, body.in_flight)
                .await
                .map(|results| {
                    serde_json::json!({
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
        },
    )
    .await
}

/// Redistribute experts within each selected replica's current epoch.
async fn rebalance_experts(State(state): State<AdminState>, body: axum::body::Bytes) -> Response {
    let request: crate::experts::Rebalance = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
    };
    let leader = Arc::clone(&state.leader);
    run(
        &state.admin,
        OperationKind::RebalanceExperts,
        true,
        async move {
            leader
                .rebalance_experts(request)
                .await
                .map_err(|error| (update_status(&error), format!("{error:#}")))
        },
    )
    .await
}

/// `POST /admin/sleep`: the engine gives back its KV cache and CUDA graph
/// storage and keeps its weights (vLLM's `sleep`); new requests wait until
/// `/admin/wake_up`.
async fn sleep(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, OperationKind::Sleep, true, async move {
        leader
            .sleep()
            .await
            .map(|()| serde_json::json!({ "asleep": true }))
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
    run(&state.admin, OperationKind::WakeUp, true, async move {
        leader
            .wake()
            .await
            .map(|()| serde_json::json!({ "asleep": false }))
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
    serde_json::json!({
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
    run(
        &state.admin,
        OperationKind::LoadLoraAdapter,
        true,
        async move {
            leader
                .load_adapter(body)
                .await
                .map(|results| serde_json::json!({ "results": results }))
                .map_err(|error| {
                    (
                        update_status(&error),
                        format!("{:#}", anyhow::Error::from(error)),
                    )
                })
        },
    )
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
    run(
        &state.admin,
        OperationKind::UnloadLoraAdapter,
        true,
        async move {
            leader
                .unload_adapter(body)
                .await
                .map(|results| serde_json::json!({ "results": results }))
                .map_err(|error| {
                    (
                        update_status(&error),
                        format!("{:#}", anyhow::Error::from(error)),
                    )
                })
        },
    )
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
    run(&state.admin, OperationKind::Profile, true, async move {
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
    Json(serde_json::json!({ "session": state.leader.profile_status() }))
}

/// `DELETE /admin/profile`: ends the running session at the engine's next
/// step boundary, every worker writing its file, and responds with it.
async fn stop_profile(State(state): State<AdminState>) -> Response {
    let leader = Arc::clone(&state.leader);
    run(&state.admin, OperationKind::Profile, true, async move {
        leader.stop_profile("operator").await.map_err(|error| {
            (
                update_status(&error),
                format!("{:#}", anyhow::Error::from(error)),
            )
        })
    })
    .await
}

/// `POST /admin/restart`: hands the deployment to a successor daemon, which
/// serves from then on while this one finishes the requests it took
/// (`crate::restart`). Responds once the successor serves, with its process
/// id.
async fn restart(State(state): State<AdminState>, bytes: axum::body::Bytes) -> Response {
    let Some(restarter) = state.restarter.clone() else {
        return ApiError::invalid_request("this daemon does not restart in place", None)
            .into_response();
    };
    let request: Restart = match body(&bytes) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let leader = Arc::clone(&state.leader);
    let admin = Arc::clone(&state.admin);
    let jobs = Arc::clone(&state.jobs);
    run(
        &state.admin,
        OperationKind::Restart,
        request.wait,
        async move {
            restarter
                .restart(None, None, &leader, &admin, &jobs)
                .await
                .map(|pid| Restarted { pid })
                .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")))
        },
    )
    .await
}

/// `POST /admin/shutdown`: brings the deployment down as SIGTERM does,
/// finishing what is in flight within `server_config.shutdown_timeout`, and
/// releases its storage unless `retire` is false. Responds at once (`202`).
async fn shutdown(State(state): State<AdminState>, bytes: axum::body::Bytes) -> Response {
    let request: Shutdown = match body(&bytes) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let admin = Arc::clone(&state.admin);
    run(&state.admin, OperationKind::Shutdown, false, async move {
        admin.retire.store(request.retire, Ordering::Release);
        admin.record(if request.retire {
            "shutting down; the deployment's storage is released after"
        } else {
            "shutting down; the deployment's storage stays retained"
        });
        admin.shutdown.cancel();
        Ok(request)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A journal under the default event log.
    fn journal() -> Arc<Journal> {
        Journal::new(crate::options::EventLog::default())
    }

    #[tokio::test]
    async fn events_are_read_after_a_sequence_number_and_waited_for() {
        let journal = journal();
        journal.record(EventBody::Deployment, "first");
        journal.record(EventBody::Deployment, "second");
        let page = journal.events_after(1, Duration::ZERO).await;
        assert_eq!(
            page.events
                .iter()
                .map(|event| event.message.as_str())
                .collect::<Vec<_>>(),
            vec!["second"]
        );
        assert_eq!((page.next, page.gap), (2, false));

        // A reader past the last event waits for the next one.
        let waiting = tokio::spawn({
            let journal = Arc::clone(&journal);
            async move { journal.events_after(2, Duration::from_secs(5)).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        journal.record(EventBody::Deployment, "third");
        let page = waiting.await.unwrap();
        assert_eq!(page.events[0].message, "third");
        assert_eq!(page.next, 3);
    }

    #[tokio::test]
    async fn a_reader_behind_the_log_is_told_of_the_gap() {
        let journal = journal();
        for index in 0..crate::options::EVENT_LOG_CAPACITY.get() + 2 {
            journal.record(EventBody::Deployment, format!("event {index}"));
        }
        let page = journal.events_after(0, Duration::ZERO).await;
        assert!(page.gap);
        assert_eq!(page.events.len(), crate::options::EVENT_LOG_CAPACITY.get());
    }

    /// One operation runs at a time; its start and end are events that carry it, and a journal
    /// handed to a successor starts none.
    #[tokio::test]
    async fn one_operation_runs_at_a_time_and_its_events_carry_it() {
        let journal = journal();
        let first = journal.begin(OperationKind::Config).unwrap();
        let refused = journal.begin(OperationKind::FlushCache).unwrap_err();
        assert!(matches!(refused, Refusal::Running(other) if other.id == first.id));
        let ended = journal.end(first, Err("it failed".to_owned()));
        assert_eq!(
            (ended.state, ended.error.as_deref()),
            (OperationState::Failed, Some("it failed"))
        );
        let page = journal.events_after(0, Duration::ZERO).await;
        assert_eq!(
            page.events.last().map(|event| &event.body),
            Some(&EventBody::Operation(ended))
        );

        let successor = Journal::resume(journal.hand_over(), crate::options::EventLog::default());
        assert!(matches!(
            journal.begin(OperationKind::FlushCache),
            Err(Refusal::HandedOver)
        ));
        let next = successor.begin(OperationKind::FlushCache).unwrap();
        assert_eq!(next.id, 2);
    }
}

//! The HTTP server: router assembly, the listener, and graceful shutdown.
//!
//! [`serve`] creates this daemon's instance directory, binds every listener
//! the node has (the leader's client API, admin sockets and join listener),
//! reports them in its readiness record, starts this node's daemon and, on the
//! leader, the [`Leader`], and serves the router of the tasks the engine
//! serves until shutdown. Each API module builds its own routes over the engine
//! (`crate::api::openai`); this module registers them by task, as vLLM registers
//! its routers by `supported_tasks`, and wraps them in the middleware of
//! `middleware`.

pub(crate) mod admin;
pub(crate) mod conn;
pub(crate) mod handoff;
pub(crate) mod listener;
mod middleware;
mod node;
pub(crate) mod served;
pub(crate) mod stream;

pub use node::node;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::metrics::METRICS;
use crate::supervision::Daemon;
use anyhow::{Context as _, Result};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::middleware::{Next, from_fn};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;
use thiserror_ext::AsReport as _;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use crate::api::openai::video_jobs::VideoJobs;
use crate::api::openai::{self, ApiError, chat_completions, videos};
use crate::http::listener::Listener;
use crate::http::middleware::ConcurrencyLimit;
use crate::leader::{Leader, Task};
use crate::options::{MediaConfig, ServerArgs, ServerConfig, Stated};

/// HTTP body limit of every route but the video submission routes, in bytes.
pub(crate) const BODY_LIMIT: usize = 64 << 20;

/// The HTTP body limit of the video submission routes, in bytes.
///
/// A `data:` URI carries its media base64 encoded, four bytes per three, so
/// a body holding `media.max_request_bytes` of media plus the request's own
/// fields fits; it never falls below the server-wide body limit.
pub(crate) fn video_body_limit(media: &MediaConfig) -> usize {
    let encoded = media.max_request_bytes.div_ceil(3) * 4 + (1 << 20);
    usize::try_from(encoded)
        .unwrap_or(usize::MAX)
        .max(BODY_LIMIT)
}

const OPENMETRICS_CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// The log targets of the HTTP stack this module serves and fetches with, whose level
/// `--log-level-http` sets: the server (`axum` over `hyper`, `tower_http`'s trace layer) and the
/// client the API fetches media with (`reqwest` over `hyper` and, for HTTP/2, `h2`). Each is a
/// target prefix, so a library's submodules follow it. The list belongs with the dependencies
/// this module declares: a library the module adds is added here.
pub const LOG_TARGETS: &[&str] = &["axum", "hyper", "hyper_util", "h2", "tower_http", "reqwest"];

/// What a daemon reports of itself ([`service::Readiness`]): to the descriptor its caller passed
/// (`--ready-fd`), once its listeners are bound and once it serves, and on its status route.
pub(crate) struct Ready {
    /// The caller's descriptor, until the daemon serves.
    fd: Option<service::ReadyFd>,
    instance: std::path::PathBuf,
    endpoints: service::Endpoints,
}

impl Ready {
    /// The reports of a daemon whose instance directory is `instance`, to `fd` when its caller
    /// passed one.
    pub(crate) fn new(fd: Option<service::ReadyFd>, instance: &std::path::Path) -> Self {
        Self {
            fd,
            instance: instance.to_owned(),
            endpoints: service::Endpoints::default(),
        }
    }

    /// The endpoints the daemon reports from now on.
    pub(crate) fn endpoints(&mut self) -> &mut service::Endpoints {
        &mut self.endpoints
    }

    /// The record of `state`.
    pub(crate) fn record(&self, state: service::State) -> service::Readiness {
        service::Readiness::of_process(
            state,
            Some(self.instance.clone()),
            self.endpoints.clone(),
            (),
        )
    }

    /// Reports `state`; the descriptor closes once the daemon serves. A caller that is gone
    /// loses nothing the daemon needs.
    pub(crate) fn report(&mut self, state: service::State) {
        let record = self.record(state);
        info!(state = ?state, endpoints = ?record.endpoints, instance = %self.instance.display(), "readiness");
        if let Some(fd) = &mut self.fd
            && let Err(error) = fd.report(&record)
        {
            warn!(%error, "the caller did not take the readiness record");
        }
        if state == service::State::Serving {
            self.fd = None;
        }
    }

    /// Adds the endpoints of the host's Storage Server the deployment uses, as the server
    /// published them, once a claim started it.
    pub(crate) fn add_storage(&mut self) {
        let running = storage::default_socket()
            .and_then(|socket| service::host::running::<storage::Settings>(&socket));
        match running {
            Ok(Some(record)) => {
                self.endpoints.storage = record.endpoints.storage;
                self.endpoints.peer = record.endpoints.peer;
            }
            Ok(None) => {}
            Err(error) => warn!(%error, "the host's Storage Server's record is unreadable"),
        }
    }
}

/// The instance directory of a daemon of this process, in the user's runtime directory.
///
/// # Errors
///
/// Fails when the user has no runtime directory or the directory cannot be created.
pub(crate) fn create_instance() -> Result<service::Instance> {
    let runtime =
        service::RuntimeDirectory::of_user().context("finding the user's runtime directory")?;
    service::Instance::create(&runtime, "daemon")
        .context("creating the daemon's instance directory")
}

/// Runs this node of the deployment until `shutdown` is cancelled.
///
/// The daemon first creates its instance directory and binds every listener it has: on the
/// leader the client API and the two admin sockets, and the join listener when other nodes may
/// join (`dist_init_addr`, port 0 allowed). It reports them in its readiness record (`ready`,
/// [`service::State::Listening`]), so a caller learns the endpoints before the slow start.
/// Every node then starts its daemon, which supervises the node's worker processes; the
/// leader's waits until every other node's daemon joined. A node other than the leader then
/// runs only its daemon, until `shutdown` or until the leader's connection closes.
///
/// The leader serves the HTTP API once the engine has started: connections that arrive earlier
/// wait in the listeners' backlog, and the readiness record that states
/// [`service::State::Serving`] follows the engine's start. Cancellation before the engine has
/// started stops the start and returns `Ok(())`. Once it has started, the server shuts down as
/// `serve_on` describes, and the engine is shut down on every path, which finishes any request
/// still running as aborted, so a stream that outlived the shutdown budget ends then. The daemon
/// stops last, ending whatever process the engine left, and the instance directory goes with
/// it unless a hot restart's successor took it over.
///
/// # Errors
///
/// Fails when the configuration is invalid, the instance directory or a listener cannot be
/// made, the daemon cannot start or fails, the engine cannot start, the server fails, or the
/// engine shutdown fails.
pub async fn serve(
    config: ServerArgs,
    stated: Stated,
    families: processing::Registry,
    devices: crate::supervision::DeviceProbe,
    ready: Option<service::ReadyFd>,
    shutdown: CancellationToken,
) -> Result<()> {
    config
        .validate(&stated)
        .context("invalid serving configuration")?;

    let instance = create_instance()?;
    let mut ready = Ready::new(ready, instance.path());
    // Every listener is bound before the slow start, and reported.
    let join = match &config.dist_init_addr {
        Some(endpoint) if config.leads() => Some(Daemon::bind_join(endpoint)?),
        _ => None,
    };
    let bound = if config.leads() {
        let listener = Listener::bind(&config.admin_config)
            .await
            .context("failed to bind the daemon's listener")?;
        let admins = admin::Listeners::bind(&config.admin_config, instance.path())?;
        let endpoints = ready.endpoints();
        endpoints.client = Some(listener.endpoint()?);
        (endpoints.admin, endpoints.harness) = admins.endpoints();
        Some((listener, admins))
    } else {
        None
    };
    let mut options = config.daemon_options(devices, instance.path());
    if let Some((_, endpoint)) = &join {
        ready.endpoints().join = Some(endpoint.clone());
        options.dist_init_addr = Some(endpoint.clone());
    }
    ready.report(service::State::Listening);

    // The leader's start waits for the other nodes to join, which blocks.
    let join = join.map(|(listener, _)| listener);
    let starting = tokio::task::spawn_blocking(move || Daemon::start(options, join));
    let daemon = tokio::select! {
        daemon = starting => daemon
            .context("the daemon's start did not complete")?
            .context("failed to start this node's daemon")?,
        () = shutdown.cancelled() => return Ok(()),
    };
    // Set by `POST /admin/shutdown`: the deployment's storage is released
    // once its processes are down.
    let retire = Arc::new(AtomicBool::new(false));
    let address = daemon.address().map(str::to_owned);
    // The epoch records the admin reports and an update writes are one.
    let epoch_records = Arc::new(crate::epoch::Epochs::new());
    let updates = daemon.updater(
        Arc::clone(&epoch_records),
        config.served_model_name().to_owned(),
    );
    let daemon = Arc::new(std::sync::Mutex::new(Some(daemon)));
    let (Some(address), Some((listener, admins))) = (address, bound) else {
        ready.report(service::State::Serving);
        info!(node_rank = config.node_rank, host = %config.host_identity, "serving this node's processes for the leader");
        while ::service::lock(&daemon)
            .as_ref()
            .is_some_and(Daemon::is_running)
        {
            tokio::select! {
                () = shutdown.cancelled() => break,
                () = tokio::time::sleep(DAEMON_POLL_INTERVAL) => {}
            }
        }
        return stop_daemon(daemon).await;
    };
    let Some((updates, awake)) = updates else {
        anyhow::bail!("the leader's daemon commands no engine");
    };
    Daemon::advance_updates(Arc::clone(&updates), awake)
        .context("starting the updater's thread")?;
    let engine = tokio::select! {
        engine = Leader::start(&families, &config, &stated, address, instance.path(), updates) => {
            engine.map(Arc::new)
        }
        () = shutdown.cancelled() => return stop_daemon(daemon).await,
    };
    let engine = match engine {
        Ok(engine) => engine,
        Err(error) => {
            if let Err(stopped) = stop_daemon(daemon).await {
                warn!("the daemon did not stop after the engine failed to start: {stopped:#}");
            }
            return Err(error);
        }
    };
    let admin = admin::Admin::with_epochs(
        shutdown.clone(),
        Arc::clone(&retire),
        Arc::clone(&epoch_records),
    );
    admin.record(
        "deployment",
        format!("serving {}", engine.served_model_name()),
    );
    admin.watch(&engine);
    ready.add_storage();
    admin.set_readiness(ready.record(service::State::Serving));
    ready.report(service::State::Serving);
    let served = async {
        let jobs = Arc::new(VideoJobs::new(
            config.video_job_config.clone(),
            shutdown.child_token(),
        ));
        serve_leader(
            &config,
            &stated,
            Arc::clone(&engine),
            admin,
            listener,
            admins,
            instance.path(),
            jobs,
            Arc::clone(&daemon),
            &shutdown,
        )
        .await
    }
    .await;
    let handoff_sent = served.as_ref().is_ok_and(|handoff_sent| *handoff_sent);
    // A successor that took the deployment over owns the instance directory from then on.
    if handoff_sent {
        instance.hand_over();
    }
    finish(
        &config,
        &engine,
        daemon,
        retire,
        handoff_sent,
        served.map(drop),
    )
    .await
}

/// Serves the leader `engine` on `listener` and the admin API on `admins` with `admin` and
/// `jobs` until `shutdown` is cancelled, or until a hot restart's successor serves and this
/// daemon finished the requests it took; returns whether the deployment was handed over.
/// `daemon` is this node's daemon's service and `instance` its instance directory, which a hot
/// restart hands over.
///
/// # Errors
///
/// As [`serve_on`].
#[allow(clippy::too_many_arguments)]
pub(crate) async fn serve_leader(
    config: &ServerArgs,
    stated: &Stated,
    engine: Arc<Leader>,
    admin: Arc<admin::Admin>,
    mut listener: Listener,
    admins: admin::Listeners,
    instance: &std::path::Path,
    jobs: Arc<VideoJobs>,
    daemon: Arc<std::sync::Mutex<Option<Daemon>>>,
    shutdown: &CancellationToken,
) -> Result<bool> {
    let accepting = shutdown.child_token();
    let restarter = Arc::new(crate::restart::Restarter::new(
        daemon,
        &mut listener,
        instance,
        stated.clone(),
        accepting.clone(),
    )?);
    let handoff_sent = Arc::clone(&restarter.handoff_sent);
    let restarting = restarter.restarting();
    let mut decided = restarter.decide();
    serve_on(
        &engine,
        &admin,
        listener,
        admins,
        config,
        shutdown,
        &accepting,
        jobs,
        Some(restarter),
    )
    .await?;
    // A restart in flight speaks for the deployment. Exiting here would close the handoff
    // channel under the successor while it is still adopting the engines, and the successor
    // would report that close as its own failure (`bdf651928`).
    // Only a restart that is still running can be waited for: the channel starts
    // at `None`, so its state alone cannot tell "nothing asked" from "in flight".
    if restarting.load(std::sync::atomic::Ordering::Acquire) && decided.borrow().is_none() {
        let waited = tokio::time::timeout(
            crate::restart::RESTART_GRACE,
            decided.wait_for(Option::is_some),
        )
        .await;
        if waited.is_err() {
            tracing::warn!(
                "a restart was still in flight when this daemon stopped serving; its successor \
                 may have been left holding a handoff nobody responds to"
            );
        }
    }
    Ok(handoff_sent.load(Ordering::Acquire))
}

/// Ends a leader's serving: shuts its engine down, stops the daemon and,
/// when `retire` was set, releases the deployment's storage; or, when the
/// deployment was handed to a hot restart's successor, lets go of the
/// engine and leaves the service and the storage to the successor.
/// `served` is how serving ended.
///
/// # Errors
///
/// Returns `served`'s failure, or the first failure of the steps that
/// follow.
pub(crate) async fn finish(
    config: &ServerArgs,
    engine: &Arc<Leader>,
    daemon: Arc<std::sync::Mutex<Option<Daemon>>>,
    retire: Arc<AtomicBool>,
    handoff_sent: bool,
    served: Result<()>,
) -> Result<()> {
    if handoff_sent {
        info!("the successor daemon serves the deployment; this daemon exits");
        return served.and(engine.detach().await);
    }
    let stopped = engine.shutdown().await;
    let served = served.and(stopped).and(stop_daemon(daemon).await);
    if retire.load(Ordering::Acquire) {
        let deployment = config.served_model_name().to_owned();
        let retired = tokio::task::spawn_blocking(move || retire_deployment(&deployment))
            .await
            .context("the deployment's release did not complete")?;
        return served.and(retired);
    }
    served
}

/// Stops this node's daemon, ending every process it supervises.
async fn stop_daemon(daemon: Arc<std::sync::Mutex<Option<Daemon>>>) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        let daemon = ::service::lock(&daemon).take();
        match daemon {
            Some(mut daemon) => daemon.stop(),
            None => Ok(()),
        }
    })
    .await
    .context("the daemon's stop did not complete")?
}

/// Releases `deployment`'s weights and caches at this host's Storage
/// Server, whose claim this daemon held until it stopped.
fn retire_deployment(deployment: &str) -> Result<()> {
    let socket = storage::default_socket().context("finding this host's Storage Server socket")?;
    let mut operator = storage::Client::connect(&socket, storage::Role::Operator)
        .with_context(|| format!("connecting to the storage server at {}", socket.display()))?;
    let released = operator
        .retire(deployment.to_owned())
        .with_context(|| format!("retiring deployment {deployment}"))?;
    info!(deployment, released, "released the deployment's storage");
    Ok(())
}

/// How often a node other than the leader checks that its daemon still
/// serves.
const DAEMON_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Serves `engine` on `listener` until `shutdown` is cancelled and the
/// server has drained, or its budget has run out; or until `accepting` is
/// cancelled, as a hot restart's successor serves, and the requests and
/// video jobs this daemon took have ended, however long they take.
///
/// Shutdown has one budget, `server_config.shutdown_timeout`, measured from
/// the cancellation. The server stops accepting connections and waits for
/// the requests in flight, streamed responses included; the asynchronous
/// video jobs stop at once, since a restarted server holds none of them, and
/// their tasks end. When the budget runs out first, the server stops waiting.
/// The caller shuts the engine down.
///
/// # Errors
///
/// Fails when the listener cannot report its address or the server fails.
#[allow(clippy::too_many_arguments)]
async fn serve_on(
    engine: &Arc<Leader>,
    admin: &Arc<admin::Admin>,
    listener: Listener,
    admins: admin::Listeners,
    config: &ServerArgs,
    shutdown: &CancellationToken,
    accepting: &CancellationToken,
    jobs: Arc<VideoJobs>,
    restarter: Option<Arc<crate::restart::Restarter>>,
) -> Result<()> {
    let server = &config.server_config;
    let bind_address = listener.local_addr()?;
    let app = router(
        Arc::clone(engine),
        Arc::clone(&jobs),
        server,
        video_body_limit(&config.media_config),
        restarter.clone(),
    );
    // The admin API has listeners of its own, apart from the client API.
    admin::serve(
        admins,
        admin::AdminState {
            leader: Arc::clone(engine),
            admin: Arc::clone(admin),
            jobs: Arc::clone(&jobs),
            restarter: restarter.clone(),
        },
        shutdown,
    );
    info!(%bind_address, model = engine.served_model_name(), "starting HTTP server");

    // The daemon admits requests until its handoff's pause starts: from then
    // on the deployment's next request is the successor's, and each connection
    // ends at its next request boundary. The registry
    // exists once the router is built, and a daemon that cannot hand over (no
    // restarter) admits for as long as it serves.
    let admitting = restarter
        .as_ref()
        .and_then(|restarter| restarter.handoff())
        .map_or_else(handoff::Admissions::default, |handoff| handoff.admissions());

    // The responses the daemon this one replaced handed over go on here: each
    // connection writes the bytes the predecessor did not write and then goes
    // on with the resumed body.
    for stream in crate::restart::streams::take_adopted() {
        let app = app.clone();
        let admitting = admitting.clone();
        tokio::spawn(async move {
            let request_id = stream.request_id.clone();
            let frozen_at_ns = stream.frozen_at_ns;
            let adopted = request_id.clone();
            let ended = request_id.clone();
            // The adoption is recorded here and nowhere else: this process
            // serves the response, and the moment it records is the
            // one before its first frame of its own. The record of the
            // response's end names the response, not the connection this
            // process may go on serving.
            let error = conn::adopt(
                stream.descriptor,
                stream.http,
                stream.body,
                app,
                admitting.clone(),
                move || {
                    crate::restart::streams::adopted(&adopted, frozen_at_ns);
                },
                move || {
                    crate::restart::streams::finished(&ended);
                },
            )
            .await
            .err();
            if let Some(error) = error {
                tracing::warn!(%request_id, %error, "a connection of the handoff ended");
            }
        });
    }

    let serving = conn::serve(listener, app, accepting.clone(), admitting);
    let drain = async {
        serving.await.context("HTTP server failed")?;
        jobs.wait().await;
        Ok(())
    };
    tokio::pin!(drain);

    // The server returns before shutdown only when it fails; the budget
    // starts when the cancellation is observed.
    tokio::select! {
        result = &mut drain => return result,
        () = shutdown.cancelled() => {}
    }
    match tokio::time::timeout(server.shutdown_timeout, drain).await {
        Ok(result) => result,
        Err(_) => {
            warn!(
                timeout = ?server.shutdown_timeout,
                "graceful shutdown deadline elapsed; the engine shutdown aborts the requests still in flight"
            );
            Ok(())
        }
    }
}

/// Assembles the router of the tasks `engine` serves.
///
/// Every engine serves the operational routes `/health` and `/metrics`,
/// which take no API key and are not counted in HTTP metrics, and `/version`,
/// which takes the key but is not counted. `GET /v1/models` and the routes of
/// each served task (`Leader::supported_tasks`) take the key and are counted.
/// The generation routes (chat completions, image generation and synchronous
/// video) share the in-flight limit. A path no
/// route serves gets FastAPI's `404 {"detail": "Not Found"}`.
///
/// Layers apply from the outside in: request ID (which also sets the
/// `X-Request-Id` response header when enabled), tracing, body limit, API
/// key, HTTP metrics, the concurrency limit, and the request timeout, so a
/// shed or timed-out request is counted in HTTP metrics and carries its
/// request ID.
pub(crate) fn router(
    engine: Arc<Leader>,
    jobs: Arc<VideoJobs>,
    server: &ServerConfig,
    video_body_limit: usize,
    restarter: Option<Arc<crate::restart::Restarter>>,
) -> Router {
    let handoff = Arc::new(handoff::Handoff::new());
    if let Some(restarter) = restarter.as_ref() {
        restarter.attach_handoff(Arc::clone(&handoff));
    }
    let timeout = |router: Router| match server.request_timeout {
        Some(timeout) => router.layer(from_fn(move |request: Request, next: Next| {
            middleware::enforce_timeout(timeout, request, next)
        })),
        None => router,
    };
    let limit = server.max_concurrent_requests.map(ConcurrencyLimit::new);
    let limited = |router: Router, overloaded: fn(String) -> Response| match &limit {
        Some(limit) => {
            let limit = limit.clone();
            router.route_layer(from_fn(move |request: Request, next: Next| {
                limit.clone().admit(overloaded, request, next)
            }))
        }
        None => router,
    };

    let served = served::Served::shared(Arc::clone(&engine), Arc::clone(&handoff));
    let mut generation = Router::new();
    let mut api = openai::models_router(Arc::clone(&served));
    for task in engine.supported_tasks() {
        match task {
            Task::Generate => {
                generation = generation.merge(chat_completions::router(Arc::clone(&served)));
            }
            Task::VideoGeneration => {
                generation = generation.merge(videos::generation_router(
                    Arc::clone(&served),
                    video_body_limit,
                ));
                api = api.merge(videos::job_router(
                    Arc::clone(&served),
                    Arc::clone(&jobs),
                    video_body_limit,
                ));
            }
        }
    }
    let generation = limited(timeout(generation), |message| {
        ApiError::service_unavailable("server_overloaded", message).into_response()
    });
    let api = timeout(api)
        .merge(generation)
        .layer(from_fn(middleware::track_http_metrics));

    let mut authenticated = Router::new().route("/version", get(version)).merge(api);
    if let Some(key) = &server.api_key {
        let key: Arc<str> = Arc::from(key.as_str());
        authenticated = authenticated.layer(from_fn(move |request: Request, next: Next| {
            middleware::require_api_key(Arc::clone(&key), request, next)
        }));
    }

    let request_id_headers = server.request_id_headers;
    Router::new()
        .route("/health", get(health))
        .with_state(engine)
        .route("/metrics", get(scrape_metrics))
        .merge(authenticated)
        .fallback(not_found)
        // Sets the limit that axum's body extractors (`Json`, `Multipart`,
        // `Bytes`) apply; the video submission routes set their own.
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .layer(TraceLayer::new_for_http())
        .layer(from_fn(move |request: Request, next: Next| {
            middleware::resolve_request_id(request_id_headers, request, next)
        }))
}

/// Responds `200 OK` while any scheduler is alive (`Leader::is_healthy`),
/// then `503 Service Unavailable`.
async fn health(State(engine): State<Arc<Leader>>) -> StatusCode {
    if engine.is_healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Renders the process-global metrics registry in OpenMetrics text format.
async fn scrape_metrics() -> Response {
    match METRICS.render() {
        Ok(body) => ([(CONTENT_TYPE, OPENMETRICS_CONTENT_TYPE)], body).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to render metrics: {}", error.as_report()),
        )
            .into_response(),
    }
}

/// Returns the UniServe distribution version, the Cargo workspace version
/// fixed at compile time.
async fn version() -> Json<serde_json::Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

/// Responds to a path no route serves as FastAPI does.
async fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "detail": "Not Found" })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::time::{Duration, Instant};

    use axum::body::Body;
    use axum::http::Request;
    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpStream;
    use tower::ServiceExt as _;

    use super::*;
    use crate::test_support::{
        SERVED_MODEL, admin_router, daemon_routers, engine_of, harness_router, post_json,
        read_response, router, router_with, send, served_engine, text_fake,
    };

    /// A `GET` request to `path`.
    fn get_request(path: &str) -> Request<Body> {
        Request::builder().uri(path).body(Body::empty()).unwrap()
    }

    /// A `DELETE` request to `path`.
    fn delete_request(path: &str) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(path)
            .body(Body::empty())
            .unwrap()
    }

    /// A chat completion of `tokens` tokens, streamed when `stream` is set.
    fn chat_body(tokens: u32, stream: bool) -> Value {
        json!({
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "prompt"}],
            "max_completion_tokens": tokens,
            "temperature": 0,
            "allowed_token_ids": [u32::from(b'a')],
            "stream": stream,
        })
    }

    /// The admin API reports the deployment, holds new requests while
    /// generation is paused, and logs what it did.
    #[tokio::test]
    async fn the_admin_api_pauses_generation_and_logs_it() {
        let app = daemon_routers(served_engine(crate::test_support::CHAT));
        let (status, _, body) = send(&app, get_request("/admin/status")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["deployment"]["deployment"], json!(SERVED_MODEL));
        assert_eq!(body["deployment"]["engine"], json!("serving"));
        assert_eq!(body["deployment"]["paused"], json!(false));
        assert_eq!(body["deployment"]["held"], json!(0));
        assert_eq!(body["deployment"]["standby"], json!([]));
        assert_eq!(body["deployment"]["update"], json!(null));

        let (status, _, body) = send(&app, post_json("/admin/pause_generation", &json!({}))).await;
        assert_eq!(
            (status, &body["state"]),
            (StatusCode::OK, &json!("succeeded"))
        );
        let (_, _, body) = send(&app, get_request("/admin/status")).await;
        assert_eq!(body["deployment"]["paused"], json!(true));

        // A chat completion waits while generation is paused, and is served
        // once it continues.
        let waiting = tokio::spawn({
            let app = app.clone();
            async move {
                send(
                    &app,
                    post_json("/v1/chat/completions", &chat_body(2, false)),
                )
                .await
            }
        });
        // The status counts the request the daemon holds.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let (_, _, body) = send(&app, get_request("/admin/status")).await;
            if body["deployment"]["held"] == json!(1) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the held request is not reported: {body}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!waiting.is_finished(), "a paused deployment holds requests");
        let (status, _, _) = send(&app, post_json("/admin/continue_generation", &json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _, _) = waiting.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        let (_, _, body) = send(&app, get_request("/admin/status")).await;
        assert_eq!(body["deployment"]["held"], json!(0));

        // Each operation started and succeeded in the log.
        let (_, _, body) = send(&app, get_request("/admin/events?after=0")).await;
        let messages: Vec<&str> = body["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|event| event["message"].as_str().unwrap())
            .collect();
        assert_eq!(
            messages,
            vec![
                "pause_generation 1 started",
                "pause_generation 1 succeeded",
                "continue_generation 2 started",
                "continue_generation 2 succeeded",
            ]
        );
        // An update this leader cannot run (it has no engine process of its
        // own) fails and is logged as failed.
        let request = Request::builder()
            .method("PUT")
            .uri("/admin/config")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({ "set": ["worker_config.watchdog_timeout=301"] }).to_string(),
            ))
            .unwrap();
        let (status, _, body) = send(&app, request).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert_eq!(body["state"], json!("failed"));
        // Its configuration and epochs read back.
        let (status, _, body) = send(&app, get_request("/admin/config")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body["config"]
                .as_str()
                .is_some_and(|config| !config.is_empty())
        );
        assert!(body["update"].is_null());
    }

    /// A path of `method` and `uri` with an empty JSON body.
    fn request(method: &str, uri: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap()
    }

    /// The admin API is not served with the client API; the operator's
    /// listener serves every admin route, and the harness's the verbs of an
    /// update and what it reads to decide one, never the operator's
    /// operations.
    #[tokio::test]
    async fn each_listener_serves_its_own_routes() {
        let update_routes = [
            ("GET", "/admin/status"),
            ("GET", "/admin/events"),
            ("GET", "/admin/config"),
            ("PUT", "/admin/config"),
            ("POST", "/admin/config/commit"),
            ("POST", "/admin/config/rollback"),
        ];
        let operator_routes = [
            ("POST", "/admin/pause_generation"),
            ("POST", "/admin/continue_generation"),
            ("POST", "/admin/flush_cache"),
            ("POST", "/admin/shutdown"),
            ("POST", "/admin/restart"),
            ("POST", "/admin/update_weights"),
            ("POST", "/admin/rebalance_experts"),
            ("POST", "/admin/sleep"),
            ("POST", "/admin/wake_up"),
            ("POST", "/admin/load_lora_adapter"),
            ("POST", "/admin/unload_lora_adapter"),
            ("GET", "/admin/profile"),
        ];

        // Whether a router serves a route: an unserved path is its 404, and
        // an unserved method of a served path its 405.
        async fn serves(app: &Router, method: &str, uri: &str) -> bool {
            let response = app.clone().oneshot(request(method, uri)).await.unwrap();
            !matches!(
                response.status(),
                StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED
            )
        }

        let client = router(served_engine(crate::test_support::CHAT));
        let admin = admin_router(served_engine(crate::test_support::CHAT));
        let harness = harness_router(served_engine(crate::test_support::CHAT));
        for (method, uri) in update_routes {
            assert!(
                !serves(&client, method, uri).await,
                "{method} {uri} on the client API"
            );
            assert!(
                serves(&admin, method, uri).await,
                "{method} {uri} on the admin listener"
            );
            assert!(
                serves(&harness, method, uri).await,
                "{method} {uri} on the harness listener"
            );
        }
        for (method, uri) in operator_routes {
            assert!(
                !serves(&client, method, uri).await,
                "{method} {uri} on the client API"
            );
            assert!(
                serves(&admin, method, uri).await,
                "{method} {uri} on the admin listener"
            );
            assert!(
                !serves(&harness, method, uri).await,
                "{method} {uri} on the harness listener"
            );
        }
        // The client API is not an admin route of either listener.
        assert!(!serves(&admin, "POST", "/v1/chat/completions").await);
        assert!(!serves(&harness, "GET", "/v1/models").await);
    }

    /// A harness applies package selections and the options the operator
    /// lists for it; another option, a whole configuration and `force` are
    /// refused before anything is resolved. The operator's listener takes
    /// the same request.
    #[tokio::test]
    async fn a_harness_sets_only_what_the_operator_allows() {
        let put = |body: Value| {
            Request::builder()
                .method("PUT")
                .uri("/admin/config")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        let harness = harness_router(served_engine(crate::test_support::CHAT));
        for refused in [
            json!({ "set": ["worker_config.watchdog_timeout=301"] }),
            json!({ "set": ["admin_config.harness_options=[\"dtype\"]"] }),
            json!({ "config": "attention_backend = \"trtllm\"\n" }),
            json!({ "set": ["attention_backend=\"trtllm\""], "force": true }),
            json!({ "set": ["attention_backend=\"trtllm\""], "replica": 0 }),
        ] {
            let (status, _, body) = send(&harness, put(refused.clone())).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{refused}: {body}");
        }
        // A package selection passes the check: this leader starts no engine
        // of its own, so the apply itself has nothing to move.
        let (status, _, body) = send(
            &harness,
            put(json!({ "set": ["attention_backend=\"trtllm\""], "deadline_s": 600 })),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");

        let admin = admin_router(served_engine(crate::test_support::CHAT));
        let (status, _, body) = send(
            &admin,
            put(json!({ "set": ["worker_config.watchdog_timeout=301"] })),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");

        // A deadline is a positive number of seconds.
        let (status, _, body) = send(
            &admin,
            put(json!({ "set": ["attention_backend=\"trtllm\""], "deadline_s": -1 })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }

    /// A sleep the engine does not take fails by name, and the deployment
    /// holds new requests from the sleep on, as it does while the engine
    /// sleeps; a wake the engine does not take fails the same way, and the
    /// requests keep waiting. The test engine takes neither: it holds no
    /// session with a Storage Server whose KV cache segments a sleep gives
    /// back and a wake creates again.
    #[tokio::test]
    async fn a_sleep_the_engine_does_not_take_fails_and_holds_requests() {
        let app = daemon_routers(served_engine(crate::test_support::CHAT));
        let (status, _, body) = send(&app, post_json("/admin/sleep", &json!({}))).await;
        assert_eq!(
            (status, &body["state"]),
            (StatusCode::INTERNAL_SERVER_ERROR, &json!("failed")),
            "{body}"
        );
        let error = body["error"].as_str().unwrap_or_default();
        assert!(error.contains("the engine did not sleep"), "{body}");
        let (_, _, body) = send(&app, get_request("/admin/status")).await;
        assert_eq!(
            (&body["deployment"]["asleep"], &body["deployment"]["paused"]),
            (&json!(true), &json!(true))
        );

        let waiting = tokio::spawn({
            let app = app.clone();
            async move {
                send(
                    &app,
                    post_json("/v1/chat/completions", &chat_body(2, false)),
                )
                .await
            }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !waiting.is_finished(),
            "a deployment put to sleep holds requests"
        );
        let (status, _, body) = send(&app, post_json("/admin/wake_up", &json!({}))).await;
        assert_eq!(
            (status, &body["state"]),
            (StatusCode::INTERNAL_SERVER_ERROR, &json!("failed")),
            "{body}"
        );
        let error = body["error"].as_str().unwrap_or_default();
        assert!(error.contains("the engine did not wake"), "{body}");
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !waiting.is_finished(),
            "the requests wait while the engine did not wake"
        );
        waiting.abort();
    }

    /// A profiling session runs on every worker until the operator ends it,
    /// one at a time; its status reports each worker's collectors and, once
    /// it ended, why and each worker's file. The test engine holds no rank:
    /// no worker runs a collector or writes a file.
    #[tokio::test]
    async fn a_profiling_session_runs_until_the_operator_ends_it() {
        let app = daemon_routers(served_engine(crate::test_support::CHAT));
        let (status, _, body) = send(&app, get_request("/admin/profile")).await;
        assert_eq!((status, &body["session"]), (StatusCode::OK, &json!(null)));

        let output = tempfile::tempdir().expect("a profile directory");
        let request = json!({
            "collectors": ["timeline", "pm_sampling"],
            "stages": ["decode"],
            "output_dir": output.path(),
        });
        let (status, _, body) = send(&app, post_json("/admin/profile", &request)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let session = &body["result"];
        assert_eq!(session["state"], json!("running"), "{body}");
        assert_eq!(session["workers"], json!([]));
        assert_eq!(session["collectors"], json!([]));

        // One session at a time.
        let (status, _, body) = send(&app, post_json("/admin/profile", &request)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let (_, _, body) = send(&app, get_request("/admin/profile")).await;
        assert_eq!(body["session"]["state"], json!("running"));

        let (status, _, body) = send(&app, delete_request("/admin/profile")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let ended = &body["result"];
        assert_eq!(
            (&ended["state"], &ended["end_reason"]),
            (&json!("ended"), &json!("operator")),
            "{body}"
        );
        assert_eq!(ended["files"], json!([]));
        let (status, _, _) = send(&app, delete_request("/admin/profile")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// A session of a window of seconds ends by itself once they passed; a
    /// request that runs no collector or names an unknown option is refused.
    #[tokio::test]
    async fn a_session_ends_after_its_seconds_and_a_faulty_request_is_refused() {
        let app = daemon_routers(served_engine(crate::test_support::CHAT));
        let output = tempfile::tempdir().expect("a profile directory");
        let (status, _, body) = send(
            &app,
            post_json(
                "/admin/profile",
                &json!({"collectors": ["pm_sampling"], "seconds": 0.5, "output_dir": output.path()}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let (_, _, body) = send(&app, get_request("/admin/profile")).await;
            if body["session"]["state"] == json!("ended") {
                assert_eq!(body["session"]["end_reason"], json!("seconds"));
                break;
            }
            assert!(tokio::time::Instant::now() < deadline, "{body}");
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        for faulty in [
            json!({"collectors": [], "output_dir": output.path()}),
            json!({"collectors": ["timeline"], "output_dir": output.path(), "window": 3}),
        ] {
            let (status, _, body) = send(&app, post_json("/admin/profile", &faulty)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        }
    }

    /// A LoRA adapter's checkpoint: one tensor in a safetensors file of a
    /// directory of its own, its header padded to a multiple of eight bytes as
    /// the format states.
    fn adapter_checkpoint() -> tempfile::TempDir {
        let directory = tempfile::tempdir().expect("an adapter directory");
        let mut header =
            r#"{"lora_A.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_owned();
        header.push_str(&" ".repeat(header.len().next_multiple_of(8) - header.len()));

        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&1.0_f32.to_le_bytes());
        std::fs::write(directory.path().join("adapter.safetensors"), file)
            .expect("an adapter checkpoint");
        directory
    }

    /// A loaded LoRA adapter is listed under its name until it is unloaded; a
    /// name is loaded once, and the served model's name is no adapter's. A
    /// request that names a loaded adapter is refused rather than answered by
    /// the base model, since no kernel package computes with an adapter
    /// (`docs/admin.md`, "LoRA adapters"), and one that names an unloaded
    /// adapter names no model. The test engine holds no rank, which loads it.
    #[tokio::test]
    async fn a_loaded_adapter_is_served_under_its_name_until_unloaded() {
        let app = daemon_routers(served_engine(crate::test_support::CHAT));
        let checkpoint = adapter_checkpoint();
        let adapter = json!({"lora_name": "tuned", "lora_path": checkpoint.path()});
        let (status, _, body) = send(&app, post_json("/admin/load_lora_adapter", &adapter)).await;
        assert_eq!(
            (status, &body["state"]),
            (StatusCode::OK, &json!("succeeded")),
            "{body}"
        );
        let (status, _, _) = send(&app, post_json("/admin/load_lora_adapter", &adapter)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _, _) = send(
            &app,
            post_json(
                "/admin/load_lora_adapter",
                &json!({"lora_name": SERVED_MODEL}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (_, _, body) = send(&app, get_request("/v1/models")).await;
        let names: Vec<&str> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|model| model["id"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec![SERVED_MODEL, "tuned"]);
        let mut request = chat_body(2, false);
        request["model"] = json!("tuned");
        let (status, _, body) = send(&app, post_json("/v1/chat/completions", &request)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(
            body["error"]["type"],
            json!("invalid_request_error"),
            "{body}"
        );

        let (status, _, body) = send(
            &app,
            post_json("/admin/unload_lora_adapter", &json!({"lora_name": "tuned"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let (status, _, _) = send(&app, post_json("/v1/chat/completions", &request)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _, _) = send(
            &app,
            post_json("/admin/unload_lora_adapter", &json!({"lora_name": "tuned"})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// A weight update names its checkpoint, and one whose files cannot be
    /// resolved is refused before the engine is asked.
    #[tokio::test]
    async fn a_weight_update_from_an_unknown_checkpoint_is_refused() {
        let app = daemon_routers(served_engine(crate::test_support::CHAT));
        let (status, _, _) = send(
            &app,
            post_json("/admin/update_weights", &json!({"path": "/nowhere"})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _, body) = send(
            &app,
            post_json(
                "/admin/update_weights",
                &json!({"model_path": "/nonexistent/checkpoint"}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["state"], json!("failed"), "{body}");
        let (_, _, body) = send(&app, get_request("/admin/status")).await;
        assert_eq!(body["deployment"]["paused"], json!(false));
    }

    /// A token model serves chat completions and no video route; a video
    /// model serves the video routes and no chat completions. A route the
    /// model does not serve gets FastAPI's 404.
    #[tokio::test]
    async fn routes_follow_the_served_tasks() {
        let not_found = json!({"detail": "Not Found"});

        let chat = router(served_engine(crate::test_support::CHAT));
        let (status, _, _) = send(
            &chat,
            post_json("/v1/chat/completions", &chat_body(2, false)),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        for request in [
            post_json("/v1/videos", &json!({})),
            get_request("/v1/capabilities"),
            post_json("/v1/images/generations", &json!({"prompt": "draw"})),
        ] {
            let (status, _, body) = send(&chat, request).await;
            assert_eq!((status, body), (StatusCode::NOT_FOUND, not_found.clone()));
        }

        let video = router(served_engine(crate::test_support::VIDEO));
        let (status, _, _) = send(&video, get_request("/v1/capabilities")).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _, body) = send(
            &video,
            post_json("/v1/chat/completions", &chat_body(2, false)),
        )
        .await;
        assert_eq!((status, body), (StatusCode::NOT_FOUND, not_found));
    }

    /// With an API key, every route but `/health` and `/metrics` requires it,
    /// and a refusal still names its request.
    #[tokio::test]
    async fn the_api_key_guards_every_route_but_health_and_metrics() {
        let router = router_with(
            served_engine(crate::test_support::CHAT),
            &ServerConfig {
                api_key: Some("secret".to_owned()),
                request_id_headers: true,
                ..ServerConfig::default()
            },
        );

        for path in ["/health", "/metrics"] {
            let response = router.clone().oneshot(get_request(path)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
        }
        for path in ["/version", "/v1/models"] {
            let (status, headers, body) = send(&router, get_request(path)).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{path}");
            assert_eq!(body["error"]["code"], "invalid_api_key");
            assert!(headers.contains_key("x-request-id"));
        }

        let mut authorized = get_request("/v1/models");
        authorized
            .headers_mut()
            .insert("authorization", "Bearer secret".parse().unwrap());
        let (status, _, body) = send(&router, authorized).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"][0]["id"], SERVED_MODEL);
    }

    /// A request whose answer is not ready within the request timeout gets
    /// 504 `request_timeout` once the timeout has passed. A chat completion
    /// answered as one document returns from its handler with its head held
    /// and generates afterwards, so the timeout bounds that generation too.
    ///
    /// The engine holds every batch it launches, so no answer exists until
    /// the test releases it; the clock stands still from the handler's return
    /// on and moves only when nothing else can run.
    #[tokio::test]
    async fn a_request_past_the_timeout_gets_504() {
        let (executors, made) = std::sync::mpsc::channel();
        let engine = engine_of(
            crate::test_support::CHAT,
            testkit::args(SERVED_MODEL),
            testkit::Fixture::new(move |fake| {
                let fake = text_fake(fake);
                fake.hold();
                let _ = executors.send(fake.clone());
                fake
            }),
        );
        // Shorter than a fixture's stall, so the timeout is the first wait
        // the paused clock reaches.
        let timeout = Duration::from_secs(10);
        let router = router_with(
            engine,
            &ServerConfig {
                request_timeout: Some(timeout),
                ..ServerConfig::default()
            },
        );

        let sent = tokio::time::Instant::now();
        let response = router
            .oneshot(post_json("/v1/chat/completions", &chat_body(2, false)))
            .await
            .unwrap();
        tokio::time::pause();
        let (status, _, body) = read_response(response, "the chat completion").await;

        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
        assert_eq!(body["error"]["code"], "request_timeout");
        assert!(tokio::time::Instant::now() >= sent + timeout);
        for executor in made.try_iter() {
            executor.resume();
        }
    }

    /// At the in-flight limit, a generation request is shed with a retryable
    /// 503 until the streamed response that holds the permit is dropped.
    #[tokio::test]
    async fn a_full_server_sheds_requests_until_a_stream_ends() {
        let router = router_with(
            served_engine(crate::test_support::CHAT),
            &ServerConfig {
                max_concurrent_requests: NonZeroU64::new(1),
                ..ServerConfig::default()
            },
        );
        let streaming = router
            .clone()
            .oneshot(post_json("/v1/chat/completions", &chat_body(2, true)))
            .await
            .unwrap();
        assert_eq!(streaming.status(), StatusCode::OK);

        let (status, headers, body) = send(
            &router,
            post_json("/v1/chat/completions", &chat_body(2, false)),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(headers["retry-after"], "1");
        assert_eq!(body["error"]["code"], "server_overloaded");
        // The limit sheds generation only.
        let (status, _, _) = send(&router, get_request("/v1/models")).await;
        assert_eq!(status, StatusCode::OK);

        drop(streaming);
        let (status, _, _) = send(
            &router,
            post_json("/v1/chat/completions", &chat_body(2, false)),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    /// HTTP metrics label a request by its route template and method, and
    /// count an extension method under `other`.
    #[tokio::test]
    async fn http_metrics_bound_the_method_label() {
        let router = router(served_engine(crate::test_support::CHAT));
        let request = Request::builder()
            .method("PURGE")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let response = router.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        drop(response);

        let metrics = METRICS.render().unwrap();
        assert!(
            metrics
                .lines()
                .any(|line| line.starts_with("http_requests_total{")
                    && line.contains(r#"method="other""#)
                    && line.contains(r#"status="4xx""#)
                    && line.contains(r#"handler="/v1/models""#)),
            "{metrics}"
        );
    }

    /// Binds a loopback listener on an ephemeral port and returns it with
    /// its address.
    async fn loopback() -> (Listener, String) {
        let listener = Listener::bind(&crate::options::AdminConfig {
            host: "127.0.0.1".to_owned(),
            port: 0,
            ..crate::options::AdminConfig::default()
        })
        .await
        .unwrap();
        let address = listener.local_addr().unwrap();
        (listener, address)
    }

    /// The serving options of a test server whose admin sockets are in
    /// `directory`. The default places them in the host's runtime directory,
    /// at a path every server of this process and a daemon on the host
    /// share, and binding one there would take it from them.
    fn serving_args(directory: &std::path::Path) -> ServerArgs {
        let mut config = ServerArgs::default();
        let socket = |name: &str| Some(directory.join(name).to_str().unwrap().to_owned());
        config.admin_config.admin_uds = socket("admin.sock");
        config.admin_config.harness_uds = socket("harness.sock");
        config
    }

    /// Serves `engine` on `listener` under `config` until `shutdown` is
    /// cancelled, as a daemon that cannot hand over does.
    fn spawn_server(
        engine: &Arc<Leader>,
        listener: Listener,
        config: ServerArgs,
        shutdown: &CancellationToken,
    ) -> tokio::task::JoinHandle<Result<()>> {
        let engine = Arc::clone(engine);
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            let admin = admin::Admin::new(shutdown.clone(), Arc::new(AtomicBool::new(false)));
            let jobs = Arc::new(VideoJobs::new(
                config.video_job_config.clone(),
                shutdown.child_token(),
            ));
            // The test's admin sockets are named, so they lie in its directory.
            let instance = std::path::Path::new(config.admin_config.admin_uds.as_deref().unwrap())
                .parent()
                .unwrap()
                .to_owned();
            let admins = admin::Listeners::bind(&config.admin_config, &instance)?;
            serve_on(
                &engine, &admin, listener, admins, &config, &shutdown, &shutdown, jobs, None,
            )
            .await
        })
    }

    /// Shutdown lets a stream already in flight finish, stops accepting
    /// connections, and returns once drained.
    #[tokio::test]
    async fn shutdown_drains_requests_in_flight_and_refuses_new_connections() {
        let engine = Arc::new(served_engine(crate::test_support::CHAT));
        let (listener, address) = loopback().await;
        let sockets = tempfile::tempdir().unwrap();
        let shutdown = CancellationToken::new();
        let server = spawn_server(&engine, listener, serving_args(sockets.path()), &shutdown);

        let response = reqwest::Client::new()
            .post(format!("http://{address}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(chat_body(16, true).to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        shutdown.cancel();

        let body = response.text().await.unwrap();
        assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(TcpStream::connect(&address).await.is_err());
        engine.shutdown().await.unwrap();
    }

    /// A request that never completes holds the drain only until the
    /// shutdown budget runs out: the server waits for it for the whole
    /// budget, measured from the cancellation, and returns once the budget
    /// has passed.
    ///
    /// The client withholds the request's body, so the request stays in
    /// flight; the clock is paused and moves only when nothing else can run.
    #[tokio::test(start_paused = true)]
    async fn shutdown_stops_waiting_when_its_budget_runs_out() {
        let engine = Arc::new(served_engine(crate::test_support::CHAT));
        let (listener, address) = loopback().await;
        let sockets = tempfile::tempdir().unwrap();
        let budget = Duration::from_secs(30);
        let mut config = serving_args(sockets.path());
        config.server_config.shutdown_timeout = budget;
        let shutdown = CancellationToken::new();
        let server = spawn_server(&engine, listener, config, &shutdown);

        // The server responds `100 Continue` once the handler waits for the
        // body, which this client never sends, so the request is in flight.
        let mut connection = TcpStream::connect(&address).await.unwrap();
        connection
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\nhost: test\r\n\
                  content-type: application/json\r\ncontent-length: 64\r\n\
                  expect: 100-continue\r\n\r\n",
            )
            .await
            .unwrap();
        let mut head = [0_u8; 25];
        connection.read_exact(&mut head).await.unwrap();
        assert!(head.starts_with(b"HTTP/1.1 100 Continue"));

        shutdown.cancel();
        let cancelled = tokio::time::Instant::now();
        tokio::time::sleep(budget - Duration::from_millis(1)).await;
        assert!(
            !server.is_finished(),
            "the server stopped waiting before its budget ran out"
        );
        tokio::time::timeout(budget, server)
            .await
            .expect("the server stops waiting once its budget ran out")
            .unwrap()
            .unwrap();
        assert!(tokio::time::Instant::now() >= cancelled + budget);
        engine.shutdown().await.unwrap();
    }

    /// The video routes' fixture: a router over the test video family.
    fn video_router() -> Router {
        router(served_engine(crate::test_support::VIDEO))
    }

    /// A five-second video request of the served model.
    fn video_request() -> serde_json::Value {
        serde_json::json!({
            "model": crate::test_support::SERVED_MODEL,
            "prompt": "A river",
            "size": "64x64",
            "seconds": 5,
        })
    }

    /// An engine rejection reaches an asynchronous job with the error code the
    /// synchronous route reports the same rejection with.
    #[tokio::test]
    async fn a_rejected_job_reports_the_synchronous_error_code() {
        let router = video_router();

        // The simulator has no video media stages, so the engine rejects
        // the request as invalid on both routes.
        let (sync_status, _, sync_body) =
            send(&router, post_json("/v1/videos/sync", &video_request())).await;
        assert_eq!(sync_status, StatusCode::BAD_REQUEST);

        let (status, _, job) = send(&router, post_json("/v1/videos", &video_request())).await;
        assert_eq!(status, StatusCode::OK);
        let uri = format!("/v1/videos/{}", job["id"].as_str().unwrap());
        let failed = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let request = axum::extract::Request::builder()
                    .uri(&uri)
                    .body(Body::empty())
                    .unwrap();
                let (_, _, job) = send(&router, request).await;
                if job["status"] == "failed" {
                    return job;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        assert_eq!(failed["error"]["code"], sync_body["error"]["code"]);
    }

    /// Both submission routes refuse a model the server does not serve.
    #[tokio::test]
    async fn the_video_routes_refuse_another_model() {
        let router = video_router();
        let mut request = video_request();
        request["model"] = serde_json::json!("another-model");
        for path in ["/v1/videos/sync", "/v1/videos"] {
            let (status, _, body) = send(&router, post_json(path, &request)).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
            assert_eq!(body["error"]["code"], "model_not_found", "{path}");
        }
    }

    /// The capabilities state the model's own part beside the request fields every video model
    /// takes.
    #[tokio::test]
    async fn capabilities_state_the_models_part() {
        let router = video_router();
        let request = axum::extract::Request::builder()
            .uri("/v1/capabilities")
            .body(Body::empty())
            .unwrap();
        let (status, _, body) = send(&router, request).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["video"]["model"]["sizes"],
            serde_json::json!(["64x64"])
        );
        assert_eq!(
            body["video"]["request_fields"],
            serde_json::json!(crate::api::protocol::request_field_names())
        );
    }
}

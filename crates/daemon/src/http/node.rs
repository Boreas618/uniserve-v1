//! A daemon that serves no deployment until asked (`uniserve serve` without
//! a model).
//!
//! The daemon's listeners start at once: the client API's address, and the
//! admin API's two Unix sockets beside it (`super::admin`). The first
//! `PUT /admin/config` on the operator's socket brings a deployment up: its
//! daemon, its first epoch and its routes, which the node then serves on the
//! same listeners, so that every later `PUT /admin/config` is the
//! deployment's. The deployment's `POST /admin/shutdown` brings it down and,
//! by default, releases its storage at the host's Storage Server; the node
//! then serves no deployment again and takes the next one. A node serves one
//! single-node deployment at a time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use anyhow::{Context as _, Result};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{Next, from_fn};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt as _;
use tracing::{info, warn};

use super::admin::Admin;
use super::listener::Listener;
use super::{middleware, retire_deployment, router, video_body_limit};
use crate::leader::Leader;
use crate::options::{AdminConfig, ServerArgs, ServerConfig, resolve_text};
use crate::supervision::Daemon;
use frontend::openai::ApiError;
use frontend::openai::video_jobs::VideoJobs;

/// How often a deployment that is brought down checks for requests in
/// flight.
const DRAIN_POLL: Duration = Duration::from_millis(50);

/// What the node serves.
#[derive(Clone)]
enum Serving {
    Idle,
    /// A deployment is starting.
    Starting(String),
    /// A deployment serves through its routes: the client API's, the
    /// operator's admin routes and the harness's.
    Epoch {
        name: String,
        client: Router,
        admin: Router,
        harness: Router,
    },
    /// A deployment is being brought down.
    Stopping(String),
}

/// The node's state.
struct Node {
    serving: Mutex<Serving>,
    /// Why the last deployment failed to start or to stop.
    last_error: Mutex<Option<String>>,
    /// The token that brings the current deployment down.
    current: Mutex<Option<CancellationToken>>,
    server: ServerConfig,
    /// The host's Build Server, which a deployment that configures none resolves its modules
    /// at.
    build_config: build_server::command::Settings,
    /// The model families the executable serves.
    families: processing::Registry,
    /// Reads this host's idle devices, for each deployment's daemon.
    devices: crate::supervision::DeviceProbe,
}

fn locked<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Serves the client API on `admin`'s address and the admin API on its two
/// Unix sockets, and the deployments the first `PUT /admin/config` brings up,
/// until `shutdown` is cancelled; the deployment serving then is brought down
/// and its storage released.
///
/// # Errors
///
/// Fails when a listener cannot be bound or the server fails.
pub async fn node(
    admin: AdminConfig,
    server: ServerConfig,
    build_config: build_server::command::Settings,
    families: processing::Registry,
    devices: crate::supervision::DeviceProbe,
    shutdown: CancellationToken,
) -> Result<()> {
    let listener = Listener::bind(&admin)
        .await
        .context("failed to bind the admin API's listener")?;
    let address = listener.local_addr()?;
    let node = Arc::new(Node {
        serving: Mutex::new(Serving::Idle),
        last_error: Mutex::new(None),
        current: Mutex::new(None),
        server: server.clone(),
        build_config,
        families,
        devices,
    });
    // The operator's socket takes the node's own verbs and, once a
    // deployment serves, its admin routes; the harness's socket only a
    // serving deployment's harness routes.
    for (path, mode, routes) in [
        (
            admin.admin_socket(),
            0o600,
            Router::new()
                .route("/admin/config", put(deploy).get(idle_config))
                .route("/admin/node", get(node_status))
                .fallback(dispatch_admin)
                .with_state(Arc::clone(&node)),
        ),
        (
            admin.harness_socket(),
            0o660,
            Router::new()
                .fallback(dispatch_harness)
                .with_state(Arc::clone(&node)),
        ),
    ] {
        let socket = Listener::bind_over(&path, mode)
            .with_context(|| format!("failed to bind the admin socket {}", path.display()))?;
        let stopped = shutdown.clone();
        tokio::spawn(async move {
            let served = axum::serve(socket, routes)
                .with_graceful_shutdown(async move { stopped.cancelled().await })
                .await;
            if let Err(error) = served {
                warn!(%error, "an admin listener ended");
            }
        });
    }
    let mut app = Router::new()
        .fallback(dispatch)
        .with_state(Arc::clone(&node));
    if let Some(key) = &server.api_key {
        let key: Arc<str> = Arc::from(key.as_str());
        // A deployment's own routes check the key themselves.
        app = app.route_layer(from_fn(move |request: Request, next: Next| {
            middleware::require_api_key(Arc::clone(&key), request, next)
        }));
    }
    info!(%address, "the node serves no deployment yet");
    // The connection writer is what takes a response's body out of its extensions (`conn`), so a
    // deployment's routes respond with nothing through any other server.
    let served = super::conn::serve(
        listener,
        app,
        shutdown.clone(),
        super::handoff::Admissions::default(),
    )
    .await
    .context("the node's HTTP server failed");
    // The deployment serving now comes down with the node.
    let current = locked(&node.current).take();
    if let Some(token) = current {
        token.cancel();
        while !matches!(*locked(&node.serving), Serving::Idle) {
            tokio::time::sleep(DRAIN_POLL).await;
        }
    }
    served
}

/// The body of the first `PUT /admin/config`, as a deployment's own reads
/// it (`docs/admin.md`).
#[derive(Debug, Deserialize)]
struct DeployBody {
    /// The deployment's configuration, as a `--config` TOML file states it.
    #[serde(default)]
    config: String,
    /// `PATH=VALUE` assignments over it, as `--set` states them.
    #[serde(default)]
    set: Vec<String>,
    /// Validate the configuration and start nothing.
    #[serde(default)]
    dry_run: bool,
}

/// `GET /admin/config` of a node that serves no deployment: no
/// configuration and empty slots.
async fn idle_config(State(node): State<Arc<Node>>) -> Response {
    let routes = match &*locked(&node.serving) {
        Serving::Epoch { admin, .. } => Some(admin.clone()),
        _ => None,
    };
    if let Some(routes) = routes {
        let request = Request::builder()
            .uri("/admin/config")
            .body(axum::body::Body::empty())
            .unwrap_or_default();
        return match routes.oneshot(request).await {
            Ok(response) => response,
            Err(never) => match never {},
        };
    }
    Json(json!({
        "config": null,
        "digest": null,
        "slots": [null, null],
        "update": null,
        "trial": null,
        "unavailable": [],
        "last": null,
    }))
    .into_response()
}

/// The first `PUT /admin/config`: brings a deployment up; responds once it
/// starts (`202`), with its name. A node that serves a deployment hands
/// the request to the deployment's own admin API.
async fn deploy(State(node): State<Arc<Node>>, request: Request) -> Response {
    let routes = match &*locked(&node.serving) {
        Serving::Epoch { admin, .. } => Some(admin.clone()),
        _ => None,
    };
    if let Some(routes) = routes {
        return match routes.oneshot(request).await {
            Ok(response) => response,
            Err(never) => match never {},
        };
    }
    let body = match axum::body::to_bytes(request.into_body(), usize::MAX).await {
        Ok(body) => body,
        Err(error) => return ApiError::invalid_request(error.to_string(), None).into_response(),
    };
    let body: DeployBody = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(error) => return ApiError::json_parse_error(error.to_string()).into_response(),
    };
    let (mut config, stated) = match resolve_text(&body.config, &body.set) {
        Ok(resolved) => resolved,
        Err(error) => {
            return ApiError::invalid_request(format!("{error:#}"), Some("config")).into_response();
        }
    };
    if config.nnodes.get() != 1 {
        return ApiError::invalid_request(
            "a node serves a deployment of one node; a deployment of several starts with \
             `uniserve serve` on each node",
            Some("config"),
        )
        .into_response();
    }
    // The node's server serves the deployment's routes.
    config.server_config = node.server.clone();
    if config.build_config == build_server::command::Settings::default() {
        config.build_config = node.build_config.clone();
    }
    if let Err(error) = config.validate(&stated) {
        return ApiError::invalid_request(format!("{error:#}"), Some("config")).into_response();
    }
    if config.model.is_none() {
        return ApiError::invalid_request("the configuration names no model", Some("config"))
            .into_response();
    }
    let name = config.served_model_name().to_owned();
    if body.dry_run {
        return Json(
            json!({ "change": "dry_run", "dry_run": { "deployment": name, "epoch": true } }),
        )
        .into_response();
    }
    {
        let mut serving = locked(&node.serving);
        if !matches!(*serving, Serving::Idle) {
            return ApiError::conflict("the node serves a deployment; shut it down first")
                .into_response();
        }
        *serving = Serving::Starting(name.clone());
    }
    let token = CancellationToken::new();
    *locked(&node.current) = Some(token.clone());
    *locked(&node.last_error) = None;
    tokio::spawn(run(Arc::clone(&node), config, stated, token));
    (StatusCode::ACCEPTED, Json(json!({ "deployment": name }))).into_response()
}

/// Brings the deployment of `config` up, serves it until `token` is
/// cancelled (by its `POST /admin/shutdown` or the node's shutdown), and
/// brings it down.
async fn run(
    node: Arc<Node>,
    config: ServerArgs,
    stated: crate::options::Stated,
    token: CancellationToken,
) {
    let name = config.served_model_name().to_owned();
    let served = serve_deployment(&node, &config, &stated, &token).await;
    if let Err(error) = &served {
        warn!(deployment = name, "the deployment ended: {error:#}");
        *locked(&node.last_error) = Some(format!("{error:#}"));
    }
    *locked(&node.current) = None;
    *locked(&node.serving) = Serving::Idle;
    info!(deployment = name, "the node serves no deployment");
}

/// The deployment's whole life on the node.
async fn serve_deployment(
    node: &Arc<Node>,
    config: &ServerArgs,
    stated: &crate::options::Stated,
    token: &CancellationToken,
) -> Result<()> {
    let name = config.served_model_name().to_owned();
    let options = config.daemon_options(node.devices);
    let daemon = tokio::task::spawn_blocking(move || Daemon::start(options))
        .await
        .context("the daemon's start did not complete")?
        .context("failed to start the deployment's daemon")?;
    let address = daemon
        .address()
        .context("a deployment of one node leads")?
        .to_owned();
    let retire = Arc::new(AtomicBool::new(false));
    let epoch_records = Arc::new(crate::epoch::Epochs::new());
    let (updates, awake) = daemon
        .updater(Arc::clone(&epoch_records), name.clone())
        .context("the leader's daemon commands no engine")?;
    Daemon::advance_updates(Arc::clone(&updates), awake)
        .context("starting the updater's thread")?;
    let served = async {
        let engine = tokio::select! {
            engine = Leader::start(&node.families, config, stated, address, updates) => {
                Arc::new(engine?)
            }
            () = token.cancelled() => return Ok(()),
        };
        let admin = Admin::with_epochs(
            token.clone(),
            Arc::clone(&retire),
            Arc::clone(&epoch_records),
        );
        admin.record("deployment", format!("serving {name}"));
        admin.watch(&engine);
        let jobs = Arc::new(VideoJobs::new(
            config.video_job_config.clone(),
            token.child_token(),
        ));
        let client = router(
            Arc::clone(&engine),
            Arc::clone(&jobs),
            &node.server,
            video_body_limit(&config.media_config),
            None,
        );
        let state = super::admin::AdminState {
            leader: Arc::clone(&engine),
            admin,
            jobs: Arc::clone(&jobs),
            restarter: None,
        };
        *locked(&node.serving) = Serving::Epoch {
            name: name.clone(),
            client,
            admin: super::admin::router(state.clone()),
            harness: super::admin::harness_router(state),
        };
        info!(deployment = name, "the node serves the deployment");
        token.cancelled().await;

        // New connections find no deployment; requests in flight finish
        // within the shutdown budget.
        *locked(&node.serving) = Serving::Stopping(name.clone());
        let drained = tokio::time::timeout(node.server.shutdown_timeout, async {
            while engine.status().in_flight > 0 {
                tokio::time::sleep(DRAIN_POLL).await;
            }
        })
        .await;
        if drained.is_err() {
            warn!(
                deployment = name,
                "the shutdown budget ran out; requests in flight abort"
            );
        }
        jobs.wait().await;
        engine.shutdown().await
    }
    .await;
    let stopped = tokio::task::spawn_blocking(move || {
        let mut daemon = daemon;
        daemon.stop()
    })
    .await
    .context("the daemon's stop did not complete")?;
    let mut result = served.and(stopped);
    if retire.load(Ordering::Acquire) {
        let deployment = name.clone();
        let retired = tokio::task::spawn_blocking(move || retire_deployment(&deployment))
            .await
            .context("the deployment's release did not complete")
            .and_then(|retired| retired);
        result = result.and(retired);
    }
    result
}

/// `GET /admin/node`: what the node serves.
async fn node_status(State(node): State<Arc<Node>>) -> Json<serde_json::Value> {
    let (state, deployment) = match &*locked(&node.serving) {
        Serving::Idle => ("idle", None),
        Serving::Starting(name) => ("starting", Some(name.clone())),
        Serving::Epoch { name, .. } => ("serving", Some(name.clone())),
        Serving::Stopping(name) => ("stopping", Some(name.clone())),
    };
    Json(json!({
        "state": state,
        "deployment": deployment,
        "last_error": locked(&node.last_error).clone(),
    }))
}

/// The client API: the serving deployment's routes, or `503` while none
/// serves.
async fn dispatch(State(node): State<Arc<Node>>, request: Request) -> Response {
    let routes = match &*locked(&node.serving) {
        Serving::Epoch { client, .. } => Some(client.clone()),
        _ => None,
    };
    serve_routes(routes, request).await
}

/// Every other admin path on the operator's socket: the serving deployment's
/// admin routes.
async fn dispatch_admin(State(node): State<Arc<Node>>, request: Request) -> Response {
    let routes = match &*locked(&node.serving) {
        Serving::Epoch { admin, .. } => Some(admin.clone()),
        _ => None,
    };
    serve_routes(routes, request).await
}

/// The harness's socket: the serving deployment's harness routes.
async fn dispatch_harness(State(node): State<Arc<Node>>, request: Request) -> Response {
    let routes = match &*locked(&node.serving) {
        Serving::Epoch { harness, .. } => Some(harness.clone()),
        _ => None,
    };
    serve_routes(routes, request).await
}

/// `request` through `routes`, or `503` while no deployment serves.
async fn serve_routes(routes: Option<Router>, request: Request) -> Response {
    match routes {
        Some(routes) => match routes.oneshot(request).await {
            Ok(response) => response,
            Err(never) => match never {},
        },
        None => ApiError::service_unavailable(
            "no_deployment",
            "the node serves no deployment; PUT /admin/config brings one up",
        )
        .into_response(),
    }
}

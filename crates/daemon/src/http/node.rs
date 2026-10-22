//! A node that serves no deployment until asked (`uniserve node`).
//!
//! The node's HTTP server starts at once and answers the admin API. `POST
//! /admin/deploy` brings a deployment up on the node: its daemon, its engine
//! and its routes, which the node then serves on the same address, admin
//! routes included. The deployment's `POST /admin/shutdown` brings it down
//! and, by default, releases its memory at the host's Storage Server; the
//! node then serves no deployment again and takes the next one. A node
//! serves one single-node deployment at a time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use anyhow::{Context as _, Result};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{Next, from_fn};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
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
use crate::openai::ApiError;
use crate::openai::video_jobs::VideoJobs;
use crate::options::{ServeConfig, ServerOptions, resolve_text};
use crate::supervision::Daemon;

/// How often a deployment that is brought down checks for requests in
/// flight.
const DRAIN_POLL: Duration = Duration::from_millis(50);

/// What the node serves.
#[derive(Clone)]
enum Serving {
    Idle,
    /// A deployment is starting.
    Starting(String),
    /// A deployment serves through its routes.
    Deployment(String, Router),
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
    server: ServerOptions,
}

fn locked<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Serves the admin API on `server`'s address, and the deployments `POST
/// /admin/deploy` brings up, until `shutdown` is cancelled; the deployment
/// serving then is brought down and its memory released.
///
/// # Errors
///
/// Fails when the listener cannot be bound or the server fails.
pub async fn node(server: ServerOptions, shutdown: CancellationToken) -> Result<()> {
    let listener = Listener::bind(&server)
        .await
        .context("failed to bind listener for the node")?;
    let address = listener.local_addr()?;
    let node = Arc::new(Node {
        serving: Mutex::new(Serving::Idle),
        last_error: Mutex::new(None),
        current: Mutex::new(None),
        server: server.clone(),
    });
    let mut app = Router::new()
        .route("/admin/deploy", post(deploy))
        .route("/admin/node", get(node_status))
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
    let served = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown.clone().cancelled_owned())
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

/// The body of `POST /admin/deploy`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeployBody {
    /// The deployment's configuration, as a `--config` TOML file states it.
    #[serde(default)]
    config: String,
    /// `PATH=VALUE` assignments over it, as `--set` states them.
    #[serde(default)]
    set: Vec<String>,
}

/// `POST /admin/deploy`: brings a deployment up; answers once it starts
/// (`202`), with its name. `409` while another deployment serves.
async fn deploy(State(node): State<Arc<Node>>, Json(body): Json<DeployBody>) -> Response {
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
    if let Err(error) = config.validate(&stated) {
        return ApiError::invalid_request(format!("{error:#}"), Some("config")).into_response();
    }
    let name = config.served_model_name().to_owned();
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
    config: ServeConfig,
    stated: crate::options::Stated,
    token: CancellationToken,
) {
    let name = config.served_model_name().to_owned();
    let outcome = serve_deployment(&node, &config, &stated, &token).await;
    if let Err(error) = &outcome {
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
    config: &ServeConfig,
    stated: &crate::options::Stated,
    token: &CancellationToken,
) -> Result<()> {
    let name = config.served_model_name().to_owned();
    let options = config.daemon_options();
    let daemon = tokio::task::spawn_blocking(move || Daemon::start(options))
        .await
        .context("the daemon's start did not complete")?
        .context("failed to start the deployment's daemon")?;
    let address = daemon
        .address()
        .context("a deployment of one node leads")?
        .to_owned();
    let retire = Arc::new(AtomicBool::new(false));
    let served = async {
        let engine = tokio::select! {
            engine = Leader::start(config, stated, address) => Arc::new(engine?),
            () = token.cancelled() => return Ok(()),
        };
        let admin = Admin::new(token.clone(), Arc::clone(&retire));
        admin.record("deployment", format!("serving {name}"));
        admin.watch(&engine);
        let jobs = Arc::new(VideoJobs::new(
            config.video_job_config.clone(),
            token.child_token(),
        ));
        let routes = router(
            Arc::clone(&engine),
            Arc::clone(&jobs),
            &node.server,
            video_body_limit(&config.media_config),
            admin,
            None,
        );
        *locked(&node.serving) = Serving::Deployment(name.clone(), routes);
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
    let mut outcome = served.and(stopped);
    if retire.load(Ordering::Acquire) {
        let deployment = name.clone();
        let retired = tokio::task::spawn_blocking(move || retire_deployment(&deployment))
            .await
            .context("the deployment's release did not complete")
            .and_then(|retired| retired);
        outcome = outcome.and(retired);
    }
    outcome
}

/// `GET /admin/node`: what the node serves.
async fn node_status(State(node): State<Arc<Node>>) -> Json<serde_json::Value> {
    let (state, deployment) = match &*locked(&node.serving) {
        Serving::Idle => ("idle", None),
        Serving::Starting(name) => ("starting", Some(name.clone())),
        Serving::Deployment(name, _) => ("serving", Some(name.clone())),
        Serving::Stopping(name) => ("stopping", Some(name.clone())),
    };
    Json(json!({
        "state": state,
        "deployment": deployment,
        "last_error": locked(&node.last_error).clone(),
    }))
}

/// Every other path: the serving deployment's routes, or `503` while none
/// serves.
async fn dispatch(State(node): State<Arc<Node>>, request: Request) -> Response {
    let routes = match &*locked(&node.serving) {
        Serving::Deployment(_, routes) => Some(routes.clone()),
        _ => None,
    };
    match routes {
        Some(routes) => match routes.oneshot(request).await {
            Ok(response) => response,
            Err(never) => match never {},
        },
        None => ApiError::service_unavailable(
            "no_deployment",
            "the node serves no deployment; POST /admin/deploy brings one up",
        )
        .into_response(),
    }
}

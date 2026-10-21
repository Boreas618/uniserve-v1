//! The HTTP server: router assembly, the listener, and graceful shutdown.
//!
//! [`serve`] starts this node's daemon and, on the leader, the [`Engine`],
//! binds the configured TCP or Unix-domain listener, and serves the router
//! of the tasks the engine serves until shutdown. Each API module builds its own routes over the engine
//! (`crate::openai`); this module registers them by task, as vLLM registers
//! its routers by `supported_tasks`, and wraps them in the middleware of
//! `middleware`.

mod listener;
mod middleware;

use std::sync::Arc;
use std::time::Duration;

use crate::metrics::METRICS;
use crate::supervision::Daemon;
use anyhow::{Context as _, Result};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::middleware::{Next, from_fn};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use thiserror_ext::AsReport as _;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use crate::frontend::{Engine, Task};
use crate::http::listener::Listener;
use crate::http::middleware::ConcurrencyLimit;
use crate::openai::video_jobs::VideoJobs;
use crate::openai::{self, ApiError, chat_completions, images, videos};
use crate::options::{ServeConfig, ServerOptions, Stated};
use crate::serving::systemone::SystemOneError;

pub(crate) use middleware::RequestId;

/// HTTP body limit of every route but the video submission routes, in bytes.
pub(crate) const BODY_LIMIT: usize = 64 << 20;

const OPENMETRICS_CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// Runs this node of the deployment until `shutdown` is cancelled.
///
/// Every node starts its daemon first, which supervises the node's worker
/// processes; the leader's waits until every other node's daemon joined. A
/// node other than the leader then runs only its daemon, until `shutdown`
/// or until the leader's connection closes.
///
/// The leader serves the HTTP API. Its listener is bound once the engine has
/// started, so the port accepts connections only when the server can answer
/// them. Cancellation before the engine has started stops the start and
/// returns `Ok(())` without binding a listener. Once it has started, the
/// server shuts down as [`serve_on`] describes, and the engine is shut down
/// on every path, which finishes any request still running as aborted, so a
/// stream that outlived the shutdown budget ends then. The daemon stops
/// last, ending whatever process the engine left.
///
/// # Errors
///
/// Fails when the configuration is invalid, the daemon cannot start or
/// fails, the engine cannot start, the listener cannot be bound or report
/// its address, the server fails, or the engine shutdown fails.
pub async fn serve(config: ServeConfig, stated: Stated, shutdown: CancellationToken) -> Result<()> {
    config
        .validate(&stated)
        .context("invalid serving configuration")?;

    // The leader's start waits for the other nodes to join, which blocks.
    let options = config.daemon_options();
    let starting = tokio::task::spawn_blocking(move || Daemon::start(options));
    let daemon = tokio::select! {
        daemon = starting => daemon
            .context("the daemon's start did not complete")?
            .context("failed to start this node's daemon")?,
        () = shutdown.cancelled() => return Ok(()),
    };
    let served = match daemon.address() {
        Some(address) => lead(&config, &stated, address.to_owned(), &shutdown).await,
        None => {
            info!(node_rank = config.node_rank, host = %config.host_identity, "serving this node's processes for the leader");
            while daemon.is_running() {
                tokio::select! {
                    () = shutdown.cancelled() => break,
                    () = tokio::time::sleep(DAEMON_POLL_INTERVAL) => {}
                }
            }
            Ok(())
        }
    };
    let stopped = tokio::task::spawn_blocking(move || {
        let mut daemon = daemon;
        daemon.stop()
    })
    .await
    .context("the daemon's stop did not complete")?;
    served.and(stopped)
}

/// How often a node other than the leader checks that its daemon still
/// serves.
const DAEMON_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Serves the API on the leader, whose daemon's process socket is `daemon`,
/// until `shutdown` is cancelled.
async fn lead(
    config: &ServeConfig,
    stated: &Stated,
    daemon: String,
    shutdown: &CancellationToken,
) -> Result<()> {
    let engine = tokio::select! {
        engine = Engine::start(config, stated, daemon) => Arc::new(engine?),
        () = shutdown.cancelled() => return Ok(()),
    };
    let served = async {
        let listener = Listener::bind(&config.server_config)
            .await
            .context("failed to bind listener for HTTP server")?;
        serve_on(&engine, listener, config, shutdown).await
    }
    .await;
    let stopped = engine.shutdown().await;
    served.and(stopped)
}

/// Serves `engine` on `listener` until `shutdown` is cancelled and the
/// server has drained, or its budget has run out.
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
async fn serve_on(
    engine: &Arc<Engine>,
    listener: Listener,
    config: &ServeConfig,
    shutdown: &CancellationToken,
) -> Result<()> {
    let server = &config.server_config;
    let bind_address = listener.local_addr()?;
    let jobs = Arc::new(VideoJobs::new(
        config.video_job_config.clone(),
        shutdown.child_token(),
    ));
    let app = router(
        Arc::clone(engine),
        Arc::clone(&jobs),
        server,
        config.media_config.body_limit(),
    );
    info!(%bind_address, model = engine.served_model_name(), "starting HTTP server");

    let serving =
        axum::serve(listener, app).with_graceful_shutdown(shutdown.clone().cancelled_owned());
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
/// Every engine answers the operational routes `/health` and `/metrics`,
/// which take no API key and are not counted in HTTP metrics, and `/version`,
/// which takes the key but is not counted. `GET /v1/models` and the routes of
/// each served task (`Engine::supported_tasks`) take the key and are counted.
/// The generation routes (chat completions, image generation, synchronous
/// video, and the System One readout) share the in-flight limit. A path no
/// route serves answers FastAPI's `404 {"detail": "Not Found"}`.
///
/// Layers apply from the outside in: request ID (which also sets the
/// `X-Request-Id` response header when enabled), tracing, body limit, API
/// key, HTTP metrics, the concurrency limit, and the request timeout, so a
/// shed or timed-out request is counted in HTTP metrics and carries its
/// request ID.
pub(crate) fn router(
    engine: Arc<Engine>,
    jobs: Arc<VideoJobs>,
    server: &ServerOptions,
    video_body_limit: usize,
) -> Router {
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

    let mut generation = Router::new();
    let mut readout = None;
    let mut api = openai::models_router(Arc::clone(&engine));
    for task in engine.supported_tasks() {
        match task {
            Task::Generate => {
                generation = generation.merge(chat_completions::router(Arc::clone(&engine)));
            }
            Task::ImageGeneration => {
                generation = generation.merge(images::router(Arc::clone(&engine)));
            }
            Task::VideoGeneration => {
                generation = generation.merge(videos::generation_router(
                    Arc::clone(&engine),
                    video_body_limit,
                ));
                api = api.merge(videos::job_router(
                    Arc::clone(&engine),
                    Arc::clone(&jobs),
                    video_body_limit,
                ));
            }
            Task::Readout => {
                readout = Some(
                    Router::new()
                        .route("/v1/systemone", post(systemone))
                        .with_state(Arc::clone(&engine)),
                );
            }
        }
    }
    let generation = limited(timeout(generation), |message| {
        ApiError::service_unavailable("server_overloaded", message).into_response()
    });
    // The readout answers a shed request in its own FastAPI error shape.
    let readout = readout.map(|readout| {
        limited(timeout(readout), |message| {
            SystemOneError::Overloaded(message).into_response()
        })
    });
    let api = timeout(api)
        .merge(generation)
        .merge(readout.unwrap_or_default())
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

/// Answers `200 OK` while any scheduler is alive (`Engine::is_healthy`),
/// then `503 Service Unavailable`.
async fn health(State(engine): State<Arc<Engine>>) -> StatusCode {
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

/// Answers one System One request (`POST /v1/systemone`).
///
/// The body is read as bytes so that malformed JSON is reported in the
/// endpoint's own `422` shape.
async fn systemone(State(engine): State<Arc<Engine>>, body: Bytes) -> Response {
    match engine.systemone(&body).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

/// Answers a path no route serves as FastAPI does.
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
    use crate::profile::ModelParameters;
    use crate::test_support::{SERVED_MODEL, post_json, router, router_with, send, sim_engine};

    /// A `GET` request to `path`.
    fn get_request(path: &str) -> Request<Body> {
        Request::builder().uri(path).body(Body::empty()).unwrap()
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

    /// A token model serves chat completions and no video route; a video
    /// model serves the video routes and no chat completions. A route the
    /// model does not serve answers FastAPI's 404.
    #[tokio::test]
    async fn routes_follow_the_served_tasks() {
        let not_found = json!({"detail": "Not Found"});

        let chat = router(sim_engine(ModelParameters::Qwen3));
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

        let video = router(sim_engine(ModelParameters::MiniMaxH3));
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
            sim_engine(ModelParameters::Qwen3),
            &ServerOptions {
                api_key: Some("secret".to_owned()),
                request_id_headers: true,
                ..ServerOptions::default()
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

    /// A request whose response head is not ready within the timeout answers
    /// 504; a zero timeout expires before any generation can answer.
    #[tokio::test]
    async fn a_request_past_the_timeout_answers_504() {
        let router = router_with(
            sim_engine(ModelParameters::Qwen3),
            &ServerOptions {
                request_timeout: Some(Duration::ZERO),
                ..ServerOptions::default()
            },
        );

        let (status, _, body) = send(
            &router,
            post_json("/v1/chat/completions", &chat_body(2, false)),
        )
        .await;

        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(body["error"]["code"], "request_timeout");
    }

    /// At the in-flight limit, a generation request is shed with a retryable
    /// 503 until the streamed response that holds the permit is dropped.
    #[tokio::test]
    async fn a_full_server_sheds_requests_until_a_stream_ends() {
        let router = router_with(
            sim_engine(ModelParameters::Qwen3),
            &ServerOptions {
                max_concurrent_requests: NonZeroU64::new(1),
                ..ServerOptions::default()
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
        let router = router(sim_engine(ModelParameters::Qwen3));
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
        let listener = Listener::bind(&ServerOptions {
            host: "127.0.0.1".to_owned(),
            port: 0,
            ..ServerOptions::default()
        })
        .await
        .unwrap();
        let address = listener.local_addr().unwrap();
        (listener, address)
    }

    /// Shutdown lets a stream already in flight finish, stops accepting
    /// connections, and returns once drained.
    #[tokio::test]
    async fn shutdown_drains_requests_in_flight_and_refuses_new_connections() {
        let engine = Arc::new(sim_engine(ModelParameters::Qwen3));
        let (listener, address) = loopback().await;
        let shutdown = CancellationToken::new();
        let server = tokio::spawn({
            let engine = Arc::clone(&engine);
            let shutdown = shutdown.clone();
            async move { serve_on(&engine, listener, &ServeConfig::default(), &shutdown).await }
        });

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
    /// shutdown budget runs out.
    #[tokio::test]
    async fn shutdown_stops_waiting_when_its_budget_runs_out() {
        let engine = Arc::new(sim_engine(ModelParameters::Qwen3));
        let (listener, address) = loopback().await;
        let budget = Duration::from_millis(200);
        let mut config = ServeConfig::default();
        config.server_config.shutdown_timeout = budget;
        let shutdown = CancellationToken::new();
        let server = tokio::spawn({
            let engine = Arc::clone(&engine);
            let shutdown = shutdown.clone();
            async move { serve_on(&engine, listener, &config, &shutdown).await }
        });

        // The server answers `100 Continue` once the handler waits for the
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

        let cancelled = Instant::now();
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(cancelled.elapsed() >= budget);
        engine.shutdown().await.unwrap();
    }
}

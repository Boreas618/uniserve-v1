//! The HTTP server: router assembly, the listeners, and the node that serves them.
//!
//! [`serve`] creates this daemon's instance directory, binds every listener
//! the node has (the leader's client API, admin socket and join listener),
//! reports them in its readiness record, and serves as the node (`node`): the
//! deployment of the package it was started with, changed by each
//! `PUT /admin/policy`. Each API module builds its own routes over the engine
//! (`crate::api::openai`); this module registers them by task, as vLLM registers
//! its routers by `supported_tasks`, and wraps them in the middleware of
//! `middleware`.

pub(crate) mod admin;
pub(crate) mod conn;
pub(crate) mod handoff;
pub(crate) mod listener;
mod middleware;
pub(crate) mod node;
pub(crate) mod served;
pub(crate) mod stream;

use std::sync::Arc;

use crate::supervision::Daemon;
use anyhow::{Context as _, Result};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::StatusCode;
use axum::middleware::{Next, from_fn};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use crate::api::openai::video_jobs::VideoJobs;
use crate::api::openai::{self, ApiError, chat_completions, videos};
use crate::config::{Config, MediaConfig, ServerConfig};
use crate::http::listener::Listener;
use crate::http::middleware::ConcurrencyLimit;
use crate::leader::{Leader, Task};

/// The HTTP body limit of the video submission routes, in bytes.
///
/// A `data:` URI carries its media base64 encoded, four bytes per three, so
/// the limit is the server's body limit (`server_config.max_request_body`,
/// which holds the request's own fields) plus `media.max_request_bytes` of
/// encoded media, when the deployment states a request total.
pub(crate) fn video_body_limit(media: &MediaConfig, server: &ServerConfig) -> usize {
    let encoded = media.max_request_bytes.map_or(0, |total| {
        usize::try_from(total.div_ceil(3) * 4).unwrap_or(usize::MAX)
    });
    server.max_request_body.saturating_add(encoded)
}

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

    /// Adds the endpoints of the Storage Server of `deployment` on this node, as the server
    /// published them, once a claim started it.
    pub(crate) fn add_storage(&mut self, deployment: &str) {
        let running = service::RuntimeDirectory::of_user()
            .and_then(|runtime| storage::socket(&runtime, deployment))
            .and_then(|socket| service::host::running::<storage::Settings>(&socket));
        match running {
            Ok(Some(record)) => {
                self.endpoints.storage = record.endpoints.storage;
                self.endpoints.peer = record.endpoints.peer;
            }
            Ok(None) => {}
            Err(error) => warn!(%error, "the deployment's Storage Server's record is unreadable"),
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

/// Runs this node of the deployment `deployment`, whose package `package` is, until `shutdown`
/// is cancelled, or with its deployment.
///
/// The node finds its place in the package's `hosts` by its host name. The leader first creates
/// its instance directory and binds every listener it has: the client API, the admin socket
/// and, for a deployment of several nodes, the join listener (`admin_config.join`). It
/// reports them in its readiness record (`ready`, [`service::State::Listening`]), so a caller
/// learns the endpoints before the slow start, and then serves as its node ([`node::run`]): it
/// applies the package as the deployment's start, from nothing or from the deployment's current
/// revision (`crate::policy::revision`), and serves the deployment. The record that states
/// [`service::State::Serving`] follows the deployment's start.
///
/// Another node runs only its daemon, which joins the leader's and supervises the node's
/// processes, until `shutdown` or until the leader's connection closes.
///
/// # Errors
///
/// Fails when the package is invalid or does not name this host, the instance directory, the
/// state directory or a listener cannot be made, the daemon cannot start or fails, the
/// deployment cannot start or fails to come down, or the server fails.
pub async fn serve(
    package: crate::policy::Package,
    deployment: String,
    ready: Option<service::ReadyFd>,
    shutdown: CancellationToken,
) -> Result<()> {
    package
        .config
        .validate(&package.stated)
        .context("invalid serving configuration")?;
    let identity = package.config.identity()?;
    if !identity.leads() {
        return member(package.config, deployment, identity, ready, shutdown).await;
    }

    let revisions =
        crate::policy::revision::Revisions::open(crate::policy::revision::directory(&deployment)?)?;
    revisions.abandon()?;
    let revisions = Arc::new(revisions);
    let instance = create_instance()?;
    let mut ready = Ready::new(ready, instance.path());
    // Every listener is bound before the slow start, and reported.
    let listener = Listener::bind(&package.config.admin_config)
        .await
        .context("failed to bind the daemon's listener")?;
    let admin = admin::Listener::bind(instance.path())?;
    let join = identity.join.as_ref().map(Daemon::bind_join).transpose()?;
    {
        let endpoints = ready.endpoints();
        endpoints.client = Some(listener.endpoint()?);
        endpoints.admin = Some(admin.endpoint());
        endpoints.join = join.as_ref().map(|(_, endpoint)| endpoint.clone());
    }
    ready.report(service::State::Listening);

    let start = node::Start {
        options: package.config.clone(),
        deployment,
        recorder: crate::observation::record::Recorder::new(
            crate::observation::Store::new(
                Arc::new(crate::observation::System),
                crate::observation::Retention::default(),
            )
            .context("creating the observation store")?,
            revisions.current()?.unwrap_or(0),
        ),
        revisions,
        instance,
        ready,
        shutdown,
    };
    let first = node::First::Package {
        package: Box::new(package),
        join,
    };
    node::run(start, listener, admin, first).await.map(drop)
}

/// Runs a node other than the leader: its daemon joins the leader's and supervises this node's
/// processes until `shutdown`, or until the leader's connection closes and the service ends.
async fn member(
    config: Config,
    deployment: String,
    identity: crate::config::Identity,
    ready: Option<service::ReadyFd>,
    shutdown: CancellationToken,
) -> Result<()> {
    let instance = create_instance()?;
    let mut ready = Ready::new(ready, instance.path());
    ready.report(service::State::Listening);
    let runtime =
        service::RuntimeDirectory::of_user().context("finding the user's runtime directory")?;
    let options = config.daemon_options(&identity, instance.path(), runtime, &deployment);
    let starting = tokio::task::spawn_blocking(move || Daemon::start(options, None));
    let daemon = tokio::select! {
        daemon = starting => daemon
            .context("the daemon's start did not complete")?
            .context("failed to start this node's daemon")?,
        () = shutdown.cancelled() => return Ok(()),
    };
    ready.report(service::State::Serving);
    info!(node_rank = identity.rank, host = %identity.host, "serving this node's processes for the leader");
    let stopper = daemon.stopper();
    let mut ended = tokio::task::spawn_blocking(move || daemon.wait());
    tokio::select! {
        ended = &mut ended => return ended.context("the daemon's service did not complete")?,
        () = shutdown.cancelled() => stopper.stop(),
    }
    ended
        .await
        .context("the daemon's service did not complete")?
}

/// How long the Storage Server of a deployment that is retired may take to stop: it ends its
/// children and closes its connections.
const STORAGE_SERVER_STOP: std::time::Duration = std::time::Duration::from_secs(60);

/// Releases `deployment`'s weights and caches on this node: its Storage Server, whose claim
/// this daemon held until it stopped, holds nothing but the deployment's, so it stops, and its
/// storage is released with it.
pub(crate) fn retire_deployment(deployment: &str) -> Result<()> {
    let runtime =
        service::RuntimeDirectory::of_user().context("finding the user's runtime directory")?;
    let socket = storage::socket(&runtime, deployment)
        .with_context(|| format!("finding the Storage Server socket of {deployment}"))?;
    service::host::stop(&socket, STORAGE_SERVER_STOP)
        .with_context(|| format!("stopping the Storage Server of {deployment}"))?;
    info!(deployment, "released the deployment's storage");
    Ok(())
}

/// Assembles the router of the tasks `engine` serves, whose handoffable responses register with
/// `handoff`, the node's registry of the responses a hot restart hands over.
///
/// Every engine serves the operational route `/health`, which takes no API key
/// and is not recorded, and `/version`, which takes the key but is not
/// recorded. `GET /v1/models` and the routes of
/// each served task (`Leader::supported_tasks`) take the key and are counted.
/// The generation routes (chat completions, image generation and synchronous
/// video) share the in-flight limit. A path no
/// route serves gets the API's error document, a 404 whose code is `route_not_found`.
///
/// Layers apply from the outside in: request ID (which also sets the
/// `X-Request-Id` response header when enabled), tracing, body limit, API
/// key, the HTTP record (`middleware::track_http_metrics`), the concurrency
/// limit, and the request timeout, so a shed or timed-out request is recorded
/// and carries its request ID.
pub(crate) fn router(
    engine: Arc<Leader>,
    jobs: Arc<VideoJobs>,
    server: &ServerConfig,
    video_body_limit: usize,
    handoff: Arc<handoff::Handoff>,
) -> Router {
    // Every route is recorded into the deployment's own observation store.
    let recorder = engine.recorder().clone();
    let timeout = |router: Router| match server.request_timeout {
        Some(timeout) => router.layer(from_fn(move |request: Request, next: Next| {
            middleware::enforce_timeout(timeout, request, next)
        })),
        None => router,
    };
    let limit = server
        .max_concurrent_requests
        .map(|limit| ConcurrencyLimit::new(limit, server.retry_after));
    let limited = |router: Router, overloaded: fn(String) -> Response| match &limit {
        Some(limit) => {
            let limit = limit.clone();
            router.route_layer(from_fn(move |request: Request, next: Next| {
                limit.clone().admit(overloaded, request, next)
            }))
        }
        None => router,
    };

    let served = served::Served::shared(
        Arc::clone(&engine),
        Arc::clone(&handoff),
        server.stream_keep_alive,
    );
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
        .layer(from_fn(move |request: Request, next: Next| {
            middleware::track_http_metrics(recorder.clone(), request, next)
        }));

    let mut authenticated = Router::new().route("/version", get(version)).merge(api);
    if let Some(key) = &server.api_key {
        let key: Arc<str> = Arc::from(key.as_str());
        authenticated = authenticated.layer(from_fn(move |request: Request, next: Next| {
            middleware::require_api_key(Arc::clone(&key), request, next)
        }));
    }

    let request_id_headers = server.request_id_headers;
    let max_request_id = server.max_request_id;
    Router::new()
        .route("/health", get(health))
        .with_state(engine)
        .merge(authenticated)
        .fallback(not_found)
        // Sets the limit that axum's body extractors (`Json`, `Multipart`,
        // `Bytes`) apply; the video submission routes set their own.
        .layer(DefaultBodyLimit::max(server.max_request_body))
        .layer(TraceLayer::new_for_http())
        .layer(from_fn(move |request: Request, next: Next| {
            middleware::resolve_request_id(request_id_headers, max_request_id, request, next)
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

/// Returns the UniServe distribution version, the Cargo workspace version
/// fixed at compile time.
async fn version() -> Json<serde_json::Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

/// Responds to a request no route serves with the API's own error document: a 404 whose code is
/// `route_not_found`, naming the method and path.
async fn not_found(method: axum::http::Method, uri: axum::http::Uri) -> Response {
    ApiError::not_found(
        "route_not_found",
        format!("no route serves {method} {}", uri.path()),
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
        SERVED_MODEL, engine_of, post_json, router, router_with, send, served_engine, text_fake,
    };

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

    /// Takes in place on `leader`'s first replica the change `change` makes to its
    /// configuration, as an apply of a package that states it does once it found no update
    /// moves it ([`crate::leader::Replica::take`]). The test engine serves no launch of its own,
    /// so the change commits to none.
    async fn take(
        leader: &Leader,
        change: impl FnOnce(&mut Config),
    ) -> Result<(), crate::leader::UpdateError> {
        let replica = Arc::clone(&leader.replicas()[0]);
        let current = service::lock(&replica.config).clone();
        let mut target = current.clone();
        change(&mut target);
        let changes = proto::reload::changes(Some(&current), &target);
        let engine = Config::default()
            .engine_config()
            .expect("the configuration resolves");
        Box::pin(replica.take(&current, target, &changes, engine, None, 2))
            .await
            .map(drop)
    }

    /// `paused` holds the new requests in the daemon from the change on, and the requests it
    /// holds are served once a change clears it.
    #[tokio::test]
    async fn paused_holds_new_requests_until_a_change_clears_it() {
        let leader = Arc::new(served_engine(crate::test_support::CHAT));
        let app = router(Arc::clone(&leader));
        take(&leader, |config| config.paused = true).await.unwrap();

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
        let deadline = Instant::now() + Duration::from_secs(10);
        while leader.held() != 1 {
            assert!(Instant::now() < deadline, "the held request is not counted");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!waiting.is_finished(), "a paused deployment holds requests");

        take(&leader, |config| config.paused = false).await.unwrap();
        let (status, _, _) = waiting.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(leader.held(), 0);
    }

    /// `asleep` holds new requests from the change on; an engine that does not take the sleep
    /// fails the change by name, and one that does not wake fails it the same way while the
    /// requests keep waiting. The test engine takes neither: it holds no session with a Storage
    /// Server whose KV cache segments a sleep gives back and a wake creates again.
    #[tokio::test]
    async fn asleep_holds_requests_and_an_engine_that_does_not_sleep_fails_by_name() {
        let leader = Arc::new(served_engine(crate::test_support::CHAT));
        let app = router(Arc::clone(&leader));
        let error = take(&leader, |config| config.asleep = true)
            .await
            .unwrap_err();
        assert!(
            format!("{:#}", anyhow::Error::from(error)).contains("the engine did not sleep"),
            "the refusal names the sleep"
        );
        assert!(leader.paused(), "a deployment put to sleep holds requests");

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

        // The failed change left the replica's configuration awake: a change that sleeps and a
        // change that wakes again are both changes of it.
        let replica = Arc::clone(&leader.replicas()[0]);
        let current = service::lock(&replica.config).clone();
        let mut asleep = current.clone();
        asleep.asleep = true;
        *service::lock(&replica.config) = asleep;
        let error = take(&leader, |config| config.asleep = false)
            .await
            .unwrap_err();
        assert!(
            format!("{:#}", anyhow::Error::from(error)).contains("the engine did not wake"),
            "the refusal names the wake"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !waiting.is_finished(),
            "the requests wait while the engine did not wake"
        );
        waiting.abort();
    }

    /// A change of `served_model_name` renames the API from the next request on: a request
    /// naming the former name is refused as naming no model, and the models listed are the new
    /// name's.
    #[tokio::test]
    async fn the_api_is_renamed_from_the_next_request_on() {
        let leader = Arc::new(served_engine(crate::test_support::CHAT));
        let app = router(Arc::clone(&leader));
        leader.rename("renamed");

        let (status, _, _) = send(
            &app,
            post_json("/v1/chat/completions", &chat_body(2, false)),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let mut request = chat_body(2, false);
        request["model"] = json!("renamed");
        let (status, _, body) = send(&app, post_json("/v1/chat/completions", &request)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let (_, _, body) = send(&app, get_request("/v1/models")).await;
        assert_eq!(body["data"][0]["id"], json!("renamed"), "{body}");
    }

    /// A profiling session runs on every worker while the configuration states it, and a
    /// change that states none ends it; each state it enters is a `log` event of source
    /// `profile`. The test engine holds no rank: no worker runs a collector or writes a file.
    #[tokio::test]
    async fn a_profiling_session_runs_while_the_configuration_states_it() {
        let leader = Arc::new(served_engine(crate::test_support::CHAT));
        let output = tempfile::tempdir().expect("a profile directory");
        let session = proto::profile::SessionRequest {
            collectors: vec![
                proto::profile::Collector::Timeline,
                proto::profile::Collector::PmSampling,
            ],
            stages: vec!["decode".to_owned()],
            output_dir: output.path().to_owned(),
            ..proto::profile::SessionRequest::default()
        };
        take(&leader, |config| {
            config.profiling_config.session = Some(session);
        })
        .await
        .unwrap();
        take(&leader, |config| config.profiling_config.session = None)
            .await
            .unwrap();

        let rows = leader
            .recorder()
            .store()
            .query(
                "SELECT fields['state'] AS state, fields['end_reason'] AS reason FROM events \
                 WHERE kind = 'log' AND labels['source'] = 'profile' ORDER BY time",
            )
            .await
            .unwrap()
            .json()
            .unwrap();
        let rows: Value = serde_json::from_slice(&rows).unwrap();
        assert_eq!(
            rows,
            json!([
                {"state": "running", "reason": null},
                {"state": "ended", "reason": "changed"},
            ])
        );
    }

    /// A session of a window of seconds ends by itself once they passed.
    #[tokio::test]
    async fn a_session_ends_after_its_seconds() {
        let leader = Arc::new(served_engine(crate::test_support::CHAT));
        let output = tempfile::tempdir().expect("a profile directory");
        take(&leader, |config| {
            config.profiling_config.session = Some(proto::profile::SessionRequest {
                collectors: vec![proto::profile::Collector::PmSampling],
                seconds: Some(0.5),
                output_dir: output.path().to_owned(),
                ..proto::profile::SessionRequest::default()
            });
        })
        .await
        .unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let rows = leader
                .recorder()
                .store()
                .query(
                    "SELECT fields['end_reason'] AS reason FROM events WHERE kind = 'log' \
                     AND labels['source'] = 'profile' AND fields['state'] = 'ended'",
                )
                .await
                .unwrap()
                .json()
                .unwrap();
            let rows: Value = serde_json::from_slice(&rows).unwrap();
            if rows != json!([]) {
                assert_eq!(rows, json!([{"reason": "seconds"}]));
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the session runs on"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// A LoRA adapter's checkpoint: one tensor in a safetensors file of a
    /// directory of its own, under the PEFT file name the engine reads
    /// (`adapter_model.safetensors`), its header padded to a multiple of eight
    /// bytes as the format states.
    fn adapter_checkpoint() -> tempfile::TempDir {
        let directory = tempfile::tempdir().expect("an adapter directory");
        let mut header =
            r#"{"lora_A.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_owned();
        header.push_str(&" ".repeat(header.len().next_multiple_of(8) - header.len()));

        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&1.0_f32.to_le_bytes());
        std::fs::write(directory.path().join("adapter_model.safetensors"), file)
            .expect("an adapter checkpoint");
        directory
    }

    /// An adapter of `adapters` is listed under its name until a change removes it. A request
    /// that names a held adapter is refused rather than answered by the base model, since no
    /// kernel package computes with an adapter (`docs/admin.md`, "LoRA adapters"), and one that
    /// names a removed adapter names no model. The test engine holds no rank, which loads it.
    #[tokio::test]
    async fn held_adapters_are_served_under_their_names_until_removed() {
        let leader = Arc::new(served_engine(crate::test_support::CHAT));
        let app = router(Arc::clone(&leader));
        let checkpoint = adapter_checkpoint();
        let path = checkpoint.path().to_str().unwrap().to_owned();
        let adapters = std::collections::BTreeMap::from([("tuned".to_owned(), path)]);
        leader.hold_adapters(&adapters).await.unwrap();
        // The same set again changes nothing.
        leader.hold_adapters(&adapters).await.unwrap();

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

        leader
            .hold_adapters(&std::collections::BTreeMap::new())
            .await
            .unwrap();
        let (status, _, _) = send(&app, post_json("/v1/chat/completions", &request)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    /// A new request is processed by the processor that serves when it arrives: once an epoch of
    /// model files that process requests otherwise replaced it, the same request renders with
    /// the replacement's chat template (here one that repeats the message), and a rollback's
    /// return to the former processor renders it as before.
    #[tokio::test]
    async fn a_request_is_processed_by_the_processor_that_serves_when_it_arrives() {
        let leader = Arc::new(served_engine(crate::test_support::CHAT));
        let app = router(Arc::clone(&leader));
        let prompt_tokens = || async {
            let (status, _, body) = send(
                &app,
                post_json("/v1/chat/completions", &chat_body(2, false)),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
            body["usage"]["prompt_tokens"].as_u64().unwrap()
        };
        let before = prompt_tokens().await;

        let limits = leader.replicas()[0]
            .hold()
            .client()
            .expect("the replica serves")
            .info()
            .limits
            .clone();
        let doubled = scheduling::processing::Bind {
            family: crate::test_support::CHAT.to_owned(),
            files: crate::api::test_support::chat_files(
                "ChatForCausalLM",
                "{{ messages[0].content }}{{ messages[0].content }}",
            ),
            ..scheduling::processing::Bind::default()
        };
        let binding = crate::api::test_support::program()
            .bind(doubled, crate::api::test_support::configured_tokenizer())
            .unwrap();
        let repeating = crate::api::input::Processor::new(
            crate::test_support::test_model(crate::test_support::CHAT),
            crate::api::test_support::configured_tokenizer(),
            binding,
            None,
            limits,
        )
        .unwrap();
        let former = leader.replicas.process_with(Arc::new(repeating));
        assert_eq!(prompt_tokens().await, 2 * before);

        leader.replicas.process_with(former);
        assert_eq!(prompt_tokens().await, before);
    }

    /// A configuration whose adapter takes the served model's name, or whose profiling session
    /// runs no collector, is refused as a start refuses it.
    #[test]
    fn a_faulty_adapter_or_session_is_refused() {
        let mut named = Config {
            model: Some("org/model".to_owned()),
            ..Config::default()
        };
        named
            .adapters
            .insert("org/model".to_owned(), "/adapters/tuned".to_owned());
        let error = named
            .validate(&crate::config::Stated::default())
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("adapters.org/model"),
            "{error:#}"
        );

        let session = Config {
            model: Some("org/model".to_owned()),
            profiling_config: proto::config::ProfilingConfig {
                session: Some(proto::profile::SessionRequest::default()),
                ..proto::config::ProfilingConfig::default()
            },
            ..Config::default()
        };
        let error = session
            .validate(&crate::config::Stated::default())
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("profiling_config.session"),
            "{error:#}"
        );
    }

    /// A token model serves chat completions and no video route; a video
    /// model serves the video routes and no chat completions. A route the
    /// model does not serve gets the API's 404, `route_not_found`.
    #[tokio::test]
    async fn routes_follow_the_served_tasks() {
        let not_found = |body: &serde_json::Value| body["error"]["code"] == "route_not_found";

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
            assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
            assert!(not_found(&body), "{body}");
        }

        let video = router(served_engine(crate::test_support::VIDEO));
        let (status, _, _) = send(&video, get_request("/v1/capabilities")).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _, body) = send(
            &video,
            post_json("/v1/chat/completions", &chat_body(2, false)),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        assert!(not_found(&body), "{body}");
    }

    /// With an API key, every route but `/health` requires it, and a refusal
    /// still names its request.
    #[tokio::test]
    async fn the_api_key_guards_every_route_but_health() {
        let router = router_with(
            served_engine(crate::test_support::CHAT),
            &ServerConfig {
                api_key: Some("secret".to_owned()),
                request_id_headers: true,
                ..ServerConfig::default()
            },
        );

        let response = router
            .clone()
            .oneshot(get_request("/health"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
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
        // The timeout is the first wait the paused clock reaches.
        let timeout = Duration::from_secs(10);
        let router = router_with(
            engine,
            &ServerConfig {
                request_timeout: Some(timeout),
                ..ServerConfig::default()
            },
        );

        let sent = tokio::time::Instant::now();
        tokio::time::pause();
        let (status, _, body) = send(
            &router,
            post_json("/v1/chat/completions", &chat_body(2, false)),
        )
        .await;

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

    /// The HTTP record labels a request by its route template and method, and
    /// counts an extension method under `other`.
    #[tokio::test]
    async fn the_http_record_bounds_the_method_label() {
        let engine = Arc::new(served_engine(crate::test_support::CHAT));
        let router = router(Arc::clone(&engine));
        let request = Request::builder()
            .method("PURGE")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let response = router.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        drop(response);

        let rows = engine
            .recorder()
            .store()
            .query(
                "SELECT sum(value) AS n FROM stats WHERE name = 'uniserve_http_requests_total' \
                 AND labels['method'] = 'other' AND labels['status'] = '4xx' \
                 AND labels['handler'] = '/v1/models'",
            )
            .await
            .unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&rows.json().unwrap()).unwrap();
        assert!(rows[0]["n"].as_f64().unwrap() >= 1.0, "{rows}");
    }

    /// Binds a loopback listener on an ephemeral port and returns it with
    /// its address.
    async fn loopback() -> (Listener, String) {
        loopback_at("127.0.0.1").await
    }

    /// Binds a listener on an ephemeral port of the loopback address `host` and returns it with
    /// its address.
    async fn loopback_at(host: &str) -> (Listener, String) {
        let listener = Listener::bind(&crate::config::AdminConfig {
            host: host.to_owned(),
            port: 0,
            ..crate::config::AdminConfig::default()
        })
        .await
        .unwrap();
        let address = listener.endpoint().unwrap().to_string();
        (listener, address)
    }

    /// Serves `engine`, already started, on `listener` under `config` as a node started for its
    /// deployment, whose instance directory lies in the runtime directory `runtime`, until
    /// `shutdown` is cancelled; the engine comes down with the node.
    fn spawn_server(
        engine: &Arc<Leader>,
        listener: Listener,
        config: Config,
        runtime: &std::path::Path,
        shutdown: &CancellationToken,
    ) -> tokio::task::JoinHandle<Result<()>> {
        let engine = Arc::clone(engine);
        let shutdown = shutdown.clone();
        let runtime = service::RuntimeDirectory::within(runtime).unwrap();
        tokio::spawn(async move {
            let instance = service::Instance::create(&runtime, "daemon")?;
            let admin = admin::Listener::bind(instance.path())?;
            let ready = Ready::new(None, instance.path());
            let token = shutdown.clone();
            let jobs = Arc::new(VideoJobs::new(
                config.video_job_config.clone(),
                token.child_token(),
            ));
            let recorder = engine.recorder().clone();
            let brought = node::Brought {
                config: config.clone(),
                stated: crate::config::Stated::default(),
                leader: engine,
                jobs,
                adopted: Vec::new(),
                daemon: None,
                token,
                then: None,
            };
            let state = tempfile::tempdir()?;
            let revisions = Arc::new(crate::policy::revision::Revisions::open(
                state.path().join("sim"),
            )?);
            let start = node::Start {
                options: config,
                deployment: "sim".to_owned(),
                revisions,
                recorder,
                instance,
                ready,
                shutdown,
            };
            node::run(
                start,
                listener,
                admin,
                node::First::Serving(Box::new(brought)),
            )
            .await
            .map(drop)
        })
    }

    /// Shutdown lets a stream already in flight finish, stops accepting
    /// connections, and returns once drained: by then no descriptor of the client listener is
    /// open, so a connection is refused whatever else still holds the node.
    ///
    /// The server listens on a loopback address of its own, which no other test binds.
    #[tokio::test]
    async fn shutdown_drains_requests_in_flight_and_refuses_new_connections() {
        let engine = Arc::new(served_engine(crate::test_support::CHAT));
        let (listener, address) = loopback_at("127.84.0.1").await;
        let runtime = tempfile::tempdir().unwrap();
        let shutdown = CancellationToken::new();
        let server = spawn_server(
            &engine,
            listener,
            Config::default(),
            runtime.path(),
            &shutdown,
        );

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
        let runtime = tempfile::tempdir().unwrap();
        let budget = Duration::from_secs(30);
        let mut config = Config::default();
        config.server_config.shutdown_timeout = budget;
        let shutdown = CancellationToken::new();
        let server = spawn_server(&engine, listener, config, runtime.path(), &shutdown);

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

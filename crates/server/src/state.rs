//! Shared application state and graceful engine shutdown coordination.
//!
//! `build_state` returns one `Arc<AppState>`. Under `http::serve` it is shared
//! by the router, in-flight handlers and middleware, and the detached
//! video-job tasks, and its strong count is the shutdown signal:
//! `AppState::shutdown` waits until it is the last owner before shutting down
//! the `Engine`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::Engine;
use tokio::time::{Duration, Instant, sleep_until};
use tracing::warn;

/// How often `AppState::shutdown` re-checks whether it holds the last strong
/// reference.
const SHUTDOWN_REFCOUNT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Shared router state: the single-model `Engine`, the asynchronous video job
/// store, and the HTTP policies the router and middleware apply.
pub struct AppState {
    engine: Engine,
    /// Process-local video jobs and retained artifacts used by the video
    /// routes; `shutdown` cancels every job in it.
    pub(crate) videos: Arc<crate::video_jobs::VideoJobs>,
    enable_request_id_headers: bool,
    api_key: Option<String>,
    request_timeout: Option<Duration>,
    max_concurrent_requests: Option<u64>,
    /// HTTP body limit of the video submission routes, in bytes.
    video_body_limit: usize,
    /// In-flight requests counted by `http::middleware::track_server_load`,
    /// which counts only requests to the routes listed in its
    /// `TRACKED_HANDLERS`.
    server_load: AtomicU64,
}

impl AppState {
    /// Creates application state with default HTTP policies.
    pub fn new(engine: Engine) -> Self {
        Self {
            engine,
            videos: Arc::new(crate::video_jobs::VideoJobs::new(
                crate::options::VideoJobOptions::default(),
            )),
            enable_request_id_headers: false,
            api_key: None,
            request_timeout: None,
            max_concurrent_requests: None,
            video_body_limit: crate::http::BODY_LIMIT,
            server_load: AtomicU64::new(0),
        }
    }

    /// Configures request-ID response headers.
    pub fn with_request_id_headers(mut self, enabled: bool) -> Self {
        self.enable_request_id_headers = enabled;
        self
    }

    /// Configures bearer-token authentication.
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    /// Configures the HTTP timeout until a response head is returned.
    pub fn with_request_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Configures the maximum number of concurrent chat completion, image
    /// generation, System One and synchronous video requests.
    pub fn with_max_concurrent_requests(mut self, limit: Option<u64>) -> Self {
        self.max_concurrent_requests = limit;
        self
    }

    /// Configures the retention and memory budget of asynchronous video jobs.
    pub fn with_video_jobs(mut self, settings: crate::options::VideoJobOptions) -> Self {
        self.videos = Arc::new(crate::video_jobs::VideoJobs::new(settings));
        self
    }

    /// Sets the HTTP body limit of the video submission routes, whose bodies
    /// may carry condition media inline (`MediaOptions::body_limit`).
    pub fn with_video_body_limit(mut self, bytes: usize) -> Self {
        self.video_body_limit = bytes;
        self
    }

    /// Returns the HTTP body limit of the video submission routes.
    pub fn video_body_limit(&self) -> usize {
        self.video_body_limit
    }

    /// Returns whether response request-ID headers are enabled.
    pub fn enable_request_id_headers(&self) -> bool {
        self.enable_request_id_headers
    }

    /// Returns the configured API key, if authentication is enabled.
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// Returns the configured request timeout.
    pub fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
    }

    /// Returns the configured concurrent-request limit.
    pub fn max_concurrent_requests(&self) -> Option<u64> {
        self.max_concurrent_requests
    }

    /// Returns the externally served model name.
    pub fn served_model_name(&self) -> &str {
        self.engine.served_model_name()
    }

    /// Returns the engine serving the model.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Returns the current number of active HTTP requests.
    pub fn server_load(&self) -> u64 {
        self.server_load.load(Ordering::Relaxed)
    }

    /// Increments the active HTTP request count.
    pub fn increment_server_load(&self) {
        self.server_load.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements the active HTTP request count.
    ///
    /// The counter wraps below zero, so each call must pair with an earlier
    /// `increment_server_load`; the load middleware pairs them through a drop
    /// guard.
    pub fn decrement_server_load(&self) {
        self.server_load.fetch_sub(1, Ordering::Relaxed);
    }

    /// Waits until request-owned references are dropped, then shuts down the
    /// engine.
    ///
    /// Every video job is cancelled first: each detached generation task in
    /// the video routes holds an `Arc<AppState>` until the task ends, and
    /// cancellation ends a running task, dropping its request stream.
    /// The state is then polled until this call owns the last strong
    /// reference, when no request stream is left, and `Engine::shutdown`
    /// shuts down the engine cores. If `deadline` passes first, the engine
    /// shutdown is skipped with a warning and the call still returns `Ok`.
    ///
    /// # Errors
    ///
    /// Returns the engine's shutdown error.
    pub async fn shutdown(mut self: Arc<Self>, deadline: Instant) -> anyhow::Result<()> {
        self.videos.cancel_all();

        loop {
            match Arc::try_unwrap(self) {
                Ok(state) => {
                    state.engine.shutdown().await?;
                    return Ok(());
                }
                Err(state) => self = state,
            }
            let ref_count = Arc::strong_count(&self);
            let now = Instant::now();
            if now >= deadline {
                warn!(
                    ref_count,
                    "shutdown deadline elapsed before app state became idle; skipping engine shutdown"
                );
                return Ok(());
            }
            sleep_until(std::cmp::min(
                deadline,
                now + SHUTDOWN_REFCOUNT_POLL_INTERVAL,
            ))
            .await;
        }
    }
}

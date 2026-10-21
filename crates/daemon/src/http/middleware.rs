//! HTTP middleware: request identification, the API key, HTTP metrics, the
//! concurrency limit of the generation routes, and the request timeout.
//!
//! `http::router` installs them as `axum::middleware` function layers. HTTP
//! metrics and the concurrency limit must observe the whole response,
//! including a streamed body that outlives the handler, so both wrap the
//! response body in [`GuardedBody`] and finish their accounting when that
//! body is dropped.

use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use crate::metrics::{HttpHandlerLabels, HttpRequestLabels, METRICS};
use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{MatchedPath, Request};
use axum::http::header::{HeaderName, RETRY_AFTER};
use axum::http::{HeaderValue, Method};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body::{Frame, SizeHint};
use tokio::sync::Semaphore;
use uuid::Uuid;

use crate::openai::ApiError;

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Maximum accepted length, in bytes, of a client-supplied request ID.
///
/// A generated ID (a `uuid4` in simple form) is 32 characters and legitimate
/// correlation IDs are short. The cap bounds how much client-controlled data
/// enters the request ID, logs, and the echoed response header.
const MAX_REQUEST_ID_LEN: usize = 128;

/// `Retry-After` hint, in seconds, of a request the concurrency limit sheds.
const RETRY_AFTER_SECONDS: HeaderValue = HeaderValue::from_static("1");

/// The request's base ID, inserted into the request extensions by
/// [`resolve_request_id`].
///
/// Route handlers prefix it per API (`chatcmpl-`, `img-`, `vid-`) and name
/// the request by the result in responses and logs. As in vLLM, the ID need
/// not be unique: the engine identifies each request by its own identifier.
#[derive(Clone, Debug)]
pub(crate) struct RequestId(pub(crate) String);

/// Resolves the request's ID and, when `echo` is set, returns it in the
/// `X-Request-Id` response header.
///
/// The client's `X-Request-Id` is used when it is acceptable (see
/// `is_acceptable_request_id`); otherwise a fresh ID is generated. The ID is
/// inserted as [`RequestId`] before the inner service runs. This is the
/// outermost layer, so every response carries the header when enabled,
/// including an API-key refusal and a shed request.
pub(crate) async fn resolve_request_id(echo: bool, mut request: Request, next: Next) -> Response {
    let id = request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|value| value.to_str().ok())
        .filter(|id| is_acceptable_request_id(id))
        .map_or_else(generate_request_id, ToOwned::to_owned);
    request.extensions_mut().insert(RequestId(id.clone()));

    let mut response = next.run(request).await;

    // An accepted or generated ID is visible ASCII without whitespace, which
    // is always a valid header value.
    if echo && let Ok(value) = HeaderValue::from_str(&id) {
        response.headers_mut().insert(X_REQUEST_ID, value);
    }
    response
}

/// Returns whether a client-supplied request ID (already visible ASCII) is
/// used as the request's ID.
///
/// Rejects empty and over-long values and any value containing whitespace or
/// a control character, which would otherwise allow log injection.
fn is_acceptable_request_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_REQUEST_ID_LEN
        && id.chars().all(|c| !c.is_control() && !c.is_whitespace())
}

/// Generates a fresh request ID: a full (untruncated) `uuid4` in simple form,
/// so IDs stay collision-resistant under high request volume.
fn generate_request_id() -> String {
    Uuid::new_v4().simple().to_string()
}

/// Refuses a request without `Authorization: Bearer <key>` for `key` with
/// `401 Unauthorized`.
///
/// The scheme may be spelled `Bearer` or `bearer`, and whitespace around the
/// token is ignored. A missing header, a non-visible-ASCII header value, or
/// any other scheme is refused.
pub(crate) async fn require_api_key(key: Arc<str>, request: Request, next: Next) -> Response {
    let token = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            value
                .strip_prefix("Bearer ")
                .or_else(|| value.strip_prefix("bearer "))
        });
    if token.is_some_and(|token| constant_time_eq(token.trim().as_bytes(), key.as_bytes())) {
        next.run(request).await
    } else {
        ApiError::Unauthorized.into_response()
    }
}

/// Compares byte strings without exiting early on the first differing byte.
///
/// Inputs of different lengths return `false` immediately, so timing can
/// reveal the expected key's length but not its contents.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut difference = 0_u8;
    for (left, right) in left.iter().zip(right.iter()) {
        difference |= left ^ right;
    }
    difference == 0
}

/// Bounds the time until the inner service returns a response head,
/// answering `504 Gateway Timeout` when it expires.
///
/// The response body is not covered: a streamed chat completion and a video
/// content download are bounded only until their headers are returned, not
/// while their bodies are sent. Handlers that generate before responding,
/// such as non-streaming chat completions and `/v1/videos/sync`, are bounded
/// through generation. On expiry the handler future is dropped, which cancels
/// its engine request.
pub(crate) async fn enforce_timeout(timeout: Duration, request: Request, next: Next) -> Response {
    match tokio::time::timeout(timeout, next.run(request)).await {
        Ok(response) => response,
        Err(_) => ApiError::timeout(format!(
            "Request exceeded the {}s server timeout.",
            timeout.as_secs()
        ))
        .into_response(),
    }
}

/// The in-flight limit shared by every generation route.
///
/// A request holds one permit from admission until its response body is
/// dropped, so a streamed chat completion holds its permit for the whole
/// stream, as SGLang's gateway holds its token in the response body. A
/// request that finds every permit taken is shed at once with
/// `503 Service Unavailable` and a `Retry-After` hint rather than queued, so a
/// fixed-capacity engine browns out instead of accepting unbounded work.
#[derive(Clone)]
pub(crate) struct ConcurrencyLimit {
    permits: Arc<Semaphore>,
    limit: u64,
}

impl ConcurrencyLimit {
    /// A limit of `limit` concurrent generation requests; a limit above the
    /// semaphore's capacity is that capacity, which no server reaches.
    pub(crate) fn new(limit: NonZeroU64) -> Self {
        let permits = usize::try_from(limit.get())
            .unwrap_or(usize::MAX)
            .min(Semaphore::MAX_PERMITS);
        Self {
            permits: Arc::new(Semaphore::new(permits)),
            limit: limit.get(),
        }
    }

    /// Admits `request` under a permit, or answers it with the response
    /// `overloaded` renders from the shedding message, in the error shape of
    /// the route.
    pub(crate) async fn admit(
        self,
        overloaded: fn(String) -> Response,
        request: Request,
        next: Next,
    ) -> Response {
        let Ok(permit) = Arc::clone(&self.permits).try_acquire_owned() else {
            let message = format!(
                "Server overloaded: {} concurrent requests already in flight. \
                 Retry after a short delay.",
                self.limit
            );
            let mut response = overloaded(message);
            response
                .headers_mut()
                .insert(RETRY_AFTER, RETRY_AFTER_SECONDS);
            return response;
        };
        guard_body(next.run(request).await, permit)
    }
}

/// Records API-server HTTP metrics with Python-compatible
/// (`PrometheusFastApiInstrumentator` style) family names and labels.
///
/// The `handler` label is the matched route template and the `method` label
/// the request method, `other` for a method outside the standard set, which
/// keeps the label set bounded as vLLM does. The `status` label is the status
/// class of the response head, so a stream that fails after its head was sent
/// still counts under that class. The duration is measured until the response
/// body is dropped: an SSE handler returns its body at once, so observing at
/// handler return would record a near-zero duration.
pub(crate) async fn track_http_metrics(request: Request, next: Next) -> Response {
    let handler = request
        .extensions()
        .get::<MatchedPath>()
        .map_or("none", MatchedPath::as_str)
        .to_owned();
    let method = method_label(request.method());
    let started_at = Instant::now();

    let response = next.run(request).await;
    let status = status_group(response.status().as_u16());
    guard_body(
        response,
        MetricsGuard {
            started_at,
            method,
            handler,
            status,
        },
    )
}

/// Records HTTP metrics when dropped, i.e. when the response body has been
/// fully sent (or aborted). Carries the labels and start time captured at
/// request entry.
struct MetricsGuard {
    started_at: Instant,
    method: &'static str,
    handler: String,
    status: &'static str,
}

impl Drop for MetricsGuard {
    /// Records request count and end-to-end body delivery latency.
    fn drop(&mut self) {
        let elapsed = self.started_at.elapsed().as_secs_f64();
        let metrics = &METRICS.api_server;

        metrics
            .http_requests
            .get_or_create(&HttpRequestLabels {
                method: self.method.to_owned(),
                status: self.status,
                handler: self.handler.clone(),
            })
            .inc();

        metrics
            .http_request_duration_seconds
            .get_or_create(&HttpHandlerLabels {
                method: self.method.to_owned(),
                handler: self.handler.clone(),
            })
            .observe(elapsed);

        metrics.http_request_duration_highr_seconds.observe(elapsed);
    }
}

/// Returns the metrics label of an HTTP method: its name for a standard
/// method, `other` for any extension method.
fn method_label(method: &Method) -> &'static str {
    match *method {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::DELETE => "DELETE",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        Method::PATCH => "PATCH",
        Method::CONNECT => "CONNECT",
        Method::TRACE => "TRACE",
        _ => "other",
    }
}

/// Returns the metrics class for an HTTP status.
fn status_group(status: u16) -> &'static str {
    match status / 100 {
        1 => "1xx",
        2 => "2xx",
        3 => "3xx",
        4 => "4xx",
        5 => "5xx",
        _ => "unknown",
    }
}

/// Moves `guard` into `response`'s body, so it drops with the body.
fn guard_body<G: Send + Unpin + 'static>(response: Response, guard: G) -> Response {
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(GuardedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

/// A response body that keeps request accounting alive until it is dropped.
///
/// The body forwards every frame and hint to `inner` unchanged; `_guard` exists
/// only for its `Drop`, which runs when the server finishes sending the body or
/// drops it because the connection closed. For an SSE response that is the end
/// of the stream, not the moment the handler returned.
struct GuardedBody<G> {
    inner: Body,
    _guard: G,
}

impl<G: Unpin> HttpBody for GuardedBody<G> {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use crate::options::ServerOptions;
    use crate::profile::ModelParameters;
    use crate::test_support::{SERVED_MODEL, post_json, router_with, send, sim_engine};

    /// Sends a two-token chat completion with `request_id` as its
    /// `X-Request-Id` and returns the response header and the body's
    /// completion `id`.
    async fn chat(request_id: Option<&str>) -> (String, String) {
        let router = router_with(
            sim_engine(ModelParameters::Qwen3),
            &ServerOptions {
                request_id_headers: true,
                ..ServerOptions::default()
            },
        );
        let mut request = post_json(
            "/v1/chat/completions",
            &serde_json::json!({
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": "prompt"}],
                "max_completion_tokens": 2,
                "temperature": 0,
                "allowed_token_ids": [u32::from(b'a')],
            }),
        );
        if let Some(request_id) = request_id {
            request
                .headers_mut()
                .insert("x-request-id", request_id.parse().unwrap());
        }

        let (status, headers, body) = send(&router, request).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let header = headers["x-request-id"].to_str().unwrap().to_owned();
        (header, body["id"].as_str().unwrap().to_owned())
    }

    /// The response header names the request the server ran: the completion
    /// `id` is the header value behind the `chatcmpl-` prefix.
    #[tokio::test]
    async fn the_response_header_carries_the_request_id() {
        let (header, id) = chat(None).await;
        assert_eq!(id, format!("chatcmpl-{header}"));

        let (header, id) = chat(Some("client/42")).await;
        assert_eq!(header, "client/42");
        assert_eq!(id, "chatcmpl-client/42");
    }

    /// A client ID the server refuses (empty, containing whitespace, or longer
    /// than 128 bytes) is replaced by one generated ID, used for both the
    /// header and the request.
    #[tokio::test]
    async fn a_refused_client_id_is_replaced_by_one_generated_id() {
        for refused in [String::new(), "has space".to_owned(), "a".repeat(129)] {
            let (header, id) = chat(Some(&refused)).await;
            assert_ne!(header, refused);
            assert_eq!(id, format!("chatcmpl-{header}"));
        }
    }
}

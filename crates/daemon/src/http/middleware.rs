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

use crate::metrics::{HttpHandlerLabels, HttpRequestLabels, Metrics};
use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{MatchedPath, Request};
use axum::http::header::{HeaderName, RETRY_AFTER};
use axum::http::{HeaderValue, Method};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body::{Frame, SizeHint};
use tokio::sync::Semaphore;
use uuid::Uuid;

use crate::api::openai::{ApiError, RequestId};
use crate::http::stream::HandedBody;

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Resolves the request's ID and, when `echo` is set, returns it in the
/// `X-Request-Id` response header.
///
/// The client's `X-Request-Id` is used when it is acceptable (see
/// `is_acceptable_request_id`, with `server_config.max_request_id` as
/// `max_length`); otherwise a fresh ID is generated. The ID is inserted as
/// [`RequestId`] before the inner service runs. This is the outermost layer,
/// so every response carries the header when enabled, including an API-key
/// refusal and a shed request.
pub(crate) async fn resolve_request_id(
    echo: bool,
    max_length: usize,
    mut request: Request,
    next: Next,
) -> Response {
    let id = request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|value| value.to_str().ok())
        .filter(|id| is_acceptable_request_id(id, max_length))
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
/// Rejects empty values, values longer than `max_length` bytes and any value
/// containing whitespace or a control character, which would otherwise allow
/// log injection.
fn is_acceptable_request_id(id: &str, max_length: usize) -> bool {
    !id.is_empty()
        && id.len() <= max_length
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
/// The authentication scheme is matched without regard to case, as RFC 9110
/// §11.1 states, and the credentials follow it after whitespace. A missing
/// header, a non-visible-ASCII header value, or any other scheme is refused.
pub(crate) async fn require_api_key(key: Arc<str>, request: Request, next: Next) -> Response {
    let token = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().split_once(char::is_whitespace))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, token)| token.trim());
    if token.is_some_and(|token| constant_time_eq(token.as_bytes(), key.as_bytes())) {
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

/// Bounds the time until the response head is sent, answering
/// `504 Gateway Timeout` when it expires.
///
/// One deadline, `timeout` after the request arrives, bounds the inner
/// service and then, for a response answered as one document (a non-streaming
/// chat completion), the wait for that document: such a response returns from
/// its handler with its head held while it generates, and the connection sends
/// the head once the document exists (`HandedBody::bound_head`). A handler
/// that generates before it responds, such as `/v1/videos/sync`, is bounded
/// through generation by the first part. The response body is not covered: a
/// streamed chat completion and a video content download are bounded only
/// until their headers are returned, not while their bodies are sent. On
/// expiry the handler future is dropped, or the held response's driver is told
/// to end, which cancels its engine request.
pub(crate) async fn enforce_timeout(timeout: Duration, request: Request, next: Next) -> Response {
    let deadline = tokio::time::Instant::now() + timeout;
    match tokio::time::timeout_at(deadline, next.run(request)).await {
        Ok(response) => {
            if let Some(body) = response.extensions().get::<HandedBody>() {
                body.bound_head(deadline, timed_out(timeout));
            }
            response
        }
        Err(_) => timed_out(timeout).into_response(),
    }
}

/// The failure of a request whose head was not sent within `timeout`.
fn timed_out(timeout: Duration) -> ApiError {
    ApiError::timeout(format!(
        "Request exceeded the {}s server timeout.",
        timeout.as_secs()
    ))
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
    /// The `Retry-After` a shed request is answered with
    /// (`server_config.retry_after`), in whole seconds.
    retry_after: HeaderValue,
}

impl ConcurrencyLimit {
    /// A limit of `limit` concurrent generation requests, shedding with a
    /// `Retry-After` of `retry_after`; a limit above the semaphore's capacity
    /// is that capacity, which no server reaches.
    pub(crate) fn new(limit: NonZeroU64, retry_after: Duration) -> Self {
        let permits = usize::try_from(limit.get())
            .unwrap_or(usize::MAX)
            .min(Semaphore::MAX_PERMITS);
        Self {
            permits: Arc::new(Semaphore::new(permits)),
            limit: limit.get(),
            retry_after: HeaderValue::from(retry_after.as_secs()),
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
                .insert(RETRY_AFTER, self.retry_after.clone());
            return response;
        };
        guard_body(next.run(request).await, permit)
    }
}

/// Records the API server's HTTP metrics (`metrics::api_server`).
///
/// It wraps the API's routes alone (the router's fallback is outside it), so
/// every request it sees matched a route's path: the `handler` label is that
/// route's template, a method the route does not serve included, and the
/// `method` label the request method, `other` for a method outside the
/// standard set, which keeps the label set bounded. The `status` label is the status class of the
/// response head, so a stream that fails after its head was sent still counts
/// under that class. The duration is measured until the response body is
/// dropped: an SSE handler returns its body at once, so observing at handler
/// return would record a near-zero duration.
pub(crate) async fn track_http_metrics(
    metrics: Arc<Metrics>,
    request: Request,
    next: Next,
) -> Response {
    #[expect(
        clippy::expect_used,
        reason = "the layer wraps the API's routes, and the router names the route of every \
                  request it hands one"
    )]
    let handler = request
        .extensions()
        .get::<MatchedPath>()
        .expect("a matched route")
        .as_str()
        .to_owned();
    let method = method_label(request.method());
    let started_at = Instant::now();

    let response = next.run(request).await;
    let status = status_group(response.status().as_u16());
    guard_body(
        response,
        MetricsGuard {
            metrics,
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
    /// The deployment's registry the request is recorded into.
    metrics: Arc<Metrics>,
    started_at: Instant,
    method: &'static str,
    handler: String,
    status: &'static str,
}

impl Drop for MetricsGuard {
    /// Records request count and end-to-end body delivery latency.
    fn drop(&mut self) {
        let elapsed = self.started_at.elapsed().as_secs_f64();
        let metrics = &self.metrics.api_server;

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
    use crate::config::ServerConfig;
    use crate::test_support::{SERVED_MODEL, post_json, router_with, send, served_engine};
    use axum::http::StatusCode;

    /// Sends a two-token chat completion with `request_id` as its
    /// `X-Request-Id` and returns the response header and the body's
    /// completion `id`.
    async fn chat(request_id: Option<&str>) -> (String, String) {
        let router = router_with(
            served_engine(crate::test_support::CHAT),
            &ServerConfig {
                request_id_headers: true,
                ..ServerConfig::default()
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

    /// The API key's authentication scheme is matched whatever its case, and
    /// any other scheme, or another key, is refused.
    #[tokio::test]
    async fn the_bearer_scheme_is_matched_whatever_its_case() {
        let router = router_with(
            served_engine(crate::test_support::CHAT),
            &ServerConfig {
                api_key: Some("secret".to_owned()),
                ..ServerConfig::default()
            },
        );
        for (authorization, status) in [
            ("Bearer secret", StatusCode::OK),
            ("bearer secret", StatusCode::OK),
            ("BEARER   secret", StatusCode::OK),
            ("Basic secret", StatusCode::UNAUTHORIZED),
            ("Bearer other", StatusCode::UNAUTHORIZED),
            ("Bearersecret", StatusCode::UNAUTHORIZED),
        ] {
            let request = axum::http::Request::builder()
                .uri("/v1/models")
                .header("authorization", authorization)
                .body(axum::body::Body::empty())
                .unwrap();
            let (answered, _, body) = send(&router, request).await;
            assert_eq!(answered, status, "{authorization:?}: {body}");
        }
    }

    /// A client ID the server refuses (empty, containing whitespace, or longer
    /// than `server_config.max_request_id`) is replaced by one generated ID,
    /// used for both the header and the request.
    #[tokio::test]
    async fn a_refused_client_id_is_replaced_by_one_generated_id() {
        let longest = ServerConfig::default().max_request_id;
        for refused in [
            String::new(),
            "has space".to_owned(),
            "a".repeat(longest + 1),
        ] {
            let (header, id) = chat(Some(&refused)).await;
            assert_ne!(header, refused);
            assert_eq!(id, format!("chatcmpl-{header}"));
        }
    }
}

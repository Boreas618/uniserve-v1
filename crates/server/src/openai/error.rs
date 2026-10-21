//! OpenAI-compatible error categories and HTTP response conversion.
//!
//! [`ApiError`] is the error vocabulary of the OpenAI surface, as vLLM's
//! Rust frontend defines it. Each variant fixes an HTTP status
//! ([`ApiError::status_code`]) and an OpenAI error body with a stable `type`
//! and `code` ([`ApiError::to_error_response`]). The derived constructors
//! (`ApiError::invalid_request`, ...) and macros (`invalid_request!`,
//! `bail_server_error!`, ...) build each variant; serving failures convert
//! through `From<ServeError>` and engine admission rejections through
//! [`ApiError::rejected`].

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::header::RETRY_AFTER;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use thiserror_ext::{Construct, Macro};

use crate::openai::{ErrorDetail, ErrorResponse};
use models::video::plan::VideoInputError;

use crate::serving::ServeError;
use crate::serving::video::artifact::VideoStreamError;

/// `Retry-After` hint, in seconds, of a `503 Service Unavailable`.
const RETRY_AFTER_SECONDS: HeaderValue = HeaderValue::from_static("1");

/// Error categories of the OpenAI-compatible API.
#[derive(Debug, Clone, PartialEq, Eq, Construct, Macro)]
#[thiserror_ext(macro(vis = pub(crate), path = "crate::openai::error"))]
pub enum ApiError {
    /// The request is valid JSON but asks for unsupported or invalid
    /// behavior.
    InvalidRequest {
        /// Human-readable validation failure.
        message: String,
        /// Request parameter associated with the failure, when known.
        param: Option<&'static str>,
    },
    /// The request body could not be parsed as the endpoint's JSON schema.
    JsonParseError {
        /// Parser failure message.
        message: String,
    },
    /// The request carries no valid API key.
    Unauthorized,
    /// The requested model name does not match the served model.
    ModelNotFound {
        /// Requested model name.
        model: String,
    },
    /// The request names a resource that does not exist, such as an unknown
    /// or expired video job.
    NotFound {
        /// Stable machine-readable error code, such as `video_not_found`.
        code: &'static str,
        /// Human-readable description.
        message: String,
    },
    /// A valid call conflicts with the current runtime state, for example a
    /// request for the content of a video job that has none (queued,
    /// running, or failed).
    Conflict {
        /// Human-readable conflict description.
        message: String,
    },
    /// The request body has a content type the endpoint does not take.
    UnsupportedMediaType {
        /// The content types the endpoint takes.
        message: String,
    },
    /// The server could not produce a result, for example because an
    /// internal invariant, the engine, or output processing failed, or
    /// because generation ended without a usable output.
    ServerError {
        /// Human-readable internal failure.
        message: String,
    },
    /// The server cannot take the request now, but the same request may
    /// succeed later: the engine's waiting queue or the server's in-flight
    /// limit is full.
    ServiceUnavailable {
        /// Stable machine-readable error code, such as `server_overloaded`.
        code: &'static str,
        /// Human-readable admission failure.
        message: String,
    },
    /// The server did not answer within its request timeout.
    Timeout {
        /// Human-readable description of the expired timeout.
        message: String,
    },
}

/// Result type of the OpenAI-compatible API.
pub type Result<T> = std::result::Result<T, ApiError>;

impl ApiError {
    /// Maps an engine admission rejection to its caller-facing category.
    ///
    /// `Invalid` becomes a 400 invalid request that fails the same way on
    /// retry; `Overloaded` becomes a 503 that the same request may pass once
    /// the engine's waiting queue drains.
    pub fn rejected(kind: engine::RejectionKind, message: impl Into<String>) -> Self {
        match kind {
            engine::RejectionKind::Invalid => Self::invalid_request(message, None),
            engine::RejectionKind::Overloaded => {
                Self::service_unavailable("server_overloaded", message)
            }
        }
    }

    /// Returns the HTTP status associated with this error category.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidRequest { .. } | Self::JsonParseError { .. } => StatusCode::BAD_REQUEST,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::ModelNotFound { .. } | Self::NotFound { .. } => StatusCode::NOT_FOUND,
            Self::Conflict { .. } => StatusCode::CONFLICT,
            Self::UnsupportedMediaType { .. } => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::ServerError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::ServiceUnavailable { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Self::Timeout { .. } => StatusCode::GATEWAY_TIMEOUT,
        }
    }

    /// Returns the stable machine-readable `code` of this error, as the error
    /// body reports it.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidRequest { .. } => "invalid_request_error",
            Self::JsonParseError { .. } => "json_parse_error",
            Self::Unauthorized => "invalid_api_key",
            Self::ModelNotFound { .. } => "model_not_found",
            Self::NotFound { code, .. } | Self::ServiceUnavailable { code, .. } => code,
            Self::Conflict { .. } => "runtime_state_conflict",
            Self::UnsupportedMediaType { .. } => "unsupported_media_type",
            Self::ServerError { .. } => "server_error",
            Self::Timeout { .. } => "request_timeout",
        }
    }

    /// Converts this error into the standard OpenAI-compatible JSON error
    /// payload, whose `code` is [`ApiError::code`].
    pub fn to_error_response(&self) -> ErrorResponse {
        let (message, error_type, param) = match self {
            Self::InvalidRequest { message, param } => {
                (message.clone(), "invalid_request_error", *param)
            }
            Self::Unauthorized => (
                "Missing or invalid API key. Provide `Authorization: Bearer <key>`.".to_owned(),
                "invalid_request_error",
                None,
            ),
            Self::ModelNotFound { model } => (
                format!("The model `{model}` does not exist."),
                "invalid_request_error",
                Some("model"),
            ),
            Self::JsonParseError { message }
            | Self::NotFound { message, .. }
            | Self::UnsupportedMediaType { message } => {
                (message.clone(), "invalid_request_error", None)
            }
            Self::Conflict { message } => (message.clone(), "conflict_error", None),
            Self::ServerError { message } => (message.clone(), "server_error", None),
            Self::ServiceUnavailable { message, .. } => {
                (message.clone(), "service_unavailable_error", None)
            }
            Self::Timeout { message } => (message.clone(), "timeout", None),
        };

        ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: error_type.to_string(),
                param: param.map(str::to_string),
                code: Some(self.code().to_string()),
            },
        }
    }
}

impl IntoResponse for ApiError {
    /// Converts the error into an HTTP response; a `503` carries a
    /// `Retry-After` hint.
    fn into_response(self) -> Response {
        let mut response = (self.status_code(), Json(self.to_error_response())).into_response();
        if let Self::ServiceUnavailable { .. } = self {
            response
                .headers_mut()
                .insert(RETRY_AFTER, RETRY_AFTER_SECONDS);
        }
        response
    }
}

impl From<JsonRejection> for ApiError {
    /// A body that is not the endpoint's JSON is a JSON parse error, whatever
    /// the cause: a missing JSON content type, malformed JSON, a schema
    /// mismatch, or a body over the size limit.
    fn from(rejection: JsonRejection) -> Self {
        Self::json_parse_error(rejection.body_text())
    }
}

impl From<ServeError> for ApiError {
    /// Maps one serving error into the OpenAI error vocabulary.
    ///
    /// Unsupported features or sampling controls, context limits, input
    /// images that cannot be fetched or decoded, refused video requests, and
    /// tokenization failures become invalid requests, a refused video request
    /// naming its field; engine, video-processing and output-processing
    /// failures become server errors. An input-image or tokenization failure's message
    /// appends its cause (the failing image and rule, or the violated field
    /// or limit), which `ServeError::ImageInput` and `ServeError::Tokenize`
    /// carry as their error source rather than in their own `Display`.
    fn from(error: ServeError) -> Self {
        match error {
            ServeError::UnsupportedSamplingControl {
                control, reason, ..
            } => Self::invalid_request(
                format!("`{control}` is not supported by this model: {reason}."),
                Some(control),
            ),
            error @ (ServeError::UnsupportedFeature { .. }
            | ServeError::ContextLengthExceeded { .. }) => {
                Self::invalid_request(error.to_string(), None)
            }
            ServeError::ImageInput { ref source, .. } => {
                Self::invalid_request(format!("{error}: {source}"), None)
            }
            ServeError::Tokenize { ref source, .. } => {
                Self::invalid_request(format!("{error}: {source}"), None)
            }
            ServeError::Video(error) => match &error {
                VideoInputError::Invalid { field, .. } => {
                    Self::invalid_request(error.to_string(), Some(field.param()))
                }
                VideoInputError::Internal(message) => {
                    Self::server_error(format!("video condition processing failed: {message}"))
                }
            },
            ServeError::Submit(source) => {
                Self::server_error(format!("engine runtime error: {source}"))
            }
            ServeError::OutputProcessing { source, .. } => {
                Self::server_error(format!("output processing error: {source}"))
            }
        }
    }
}

impl From<VideoStreamError> for ApiError {
    /// An engine rejection keeps its category ([`ApiError::rejected`]); any
    /// other video stream that ended without an artifact is a server error.
    fn from(error: VideoStreamError) -> Self {
        match error {
            VideoStreamError::Rejected { kind, message } => Self::rejected(kind, message),
            error => Self::server_error(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::RejectionKind;

    #[test]
    fn an_overloaded_engine_is_retryable_and_an_invalid_request_is_not() {
        let overloaded =
            ApiError::rejected(RejectionKind::Overloaded, "queue full").into_response();
        assert_eq!(overloaded.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(overloaded.headers()[RETRY_AFTER], "1");

        let invalid = ApiError::rejected(RejectionKind::Invalid, "unsupported");
        assert_eq!(invalid.status_code(), StatusCode::BAD_REQUEST);
    }

    /// A preprocessing rejection must tell the caller which limit or field it
    /// violated, not only that the request was refused.
    #[test]
    fn a_tokenize_rejection_reports_its_cause() {
        let cause = "video duration must be finite, positive, and at most 15 seconds";
        let error = ApiError::from(ServeError::Tokenize {
            request_id: String::from("vid-1"),
            source: crate::serving::TokenizeError::Invalid(cause.to_string()),
        });

        assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
        let message = error.to_error_response().error.message;
        assert!(message.contains("vid-1"), "{message}");
        assert!(message.contains(cause), "{message}");
    }

    /// A refused video request is an invalid request that names the field at
    /// fault in its message, with the condition list as the parameter of a
    /// condition; a failure of the server's own video processing is a server
    /// error.
    #[test]
    fn video_rejections_map_to_openai_categories() {
        let rejected = ApiError::from(ServeError::from(VideoInputError::condition(
            2,
            "is not an image",
        )));
        assert_eq!(
            rejected,
            ApiError::invalid_request("conditions[2]: is not an image", Some("conditions"))
        );

        let rejected = ApiError::from(ServeError::from(VideoInputError::invalid(
            models::video::plan::RequestField::TargetDuration,
            "is required",
        )));
        assert_eq!(
            rejected,
            ApiError::invalid_request(
                "target.duration_seconds: is required",
                Some("target.duration_seconds")
            )
        );

        let failed = ApiError::from(ServeError::from(VideoInputError::internal(
            "ffprobe did not start",
        )));
        assert_eq!(failed.code(), "server_error");
    }

    /// An input image failure names the request and the image's position,
    /// then the rule the reference violates.
    #[test]
    fn an_image_input_rejection_reports_the_image_and_its_cause() {
        let error = ApiError::from(ServeError::ImageInput {
            request_id: String::from("chat-1"),
            index: 1,
            source: crate::serving::media::MediaError::Status { status: 404 },
        });

        assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
        let message = error.to_error_response().error.message;
        assert_eq!(
            message,
            "request `chat-1` has an invalid input image 2: the URL returned HTTP status 404"
        );
    }
}

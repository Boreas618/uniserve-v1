//! Worker request and response envelopes.
//!
//! One [`WorkerRequest`] or [`WorkerResponse`] is the message of one
//! [`Frame`]. Each envelope's optional `message_id` is the
//! request-response correlation identity: a rank answers a request under the
//! identity it carried, and a transport that matches a response to its request
//! reads the identity from the decoded envelope, taking `None` as zero. It is
//! unrelated to the planned calls' [`CallId`] values inside a batch. The one
//! message a rank sends unasked, [`WorkerResponse::Stalled`], answers no
//! request and carries no identity.

use super::*;

/// Kind discriminator carried by each worker request frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    /// Queries worker identity and capabilities.
    Info,
    /// Submits a batch.
    Submit,
    /// Requests orderly worker shutdown.
    Close,
}

impl RequestKind {
    /// Returns the stable name of this request kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Submit => "submit",
            Self::Close => "close",
        }
    }
}

/// Head-to-worker request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerRequest {
    /// Queries worker identity and capabilities.
    Info {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
    /// Submits a batch.
    Submit {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// Fully lowered batch.
        batch: Box<Batch>,
    },
    /// Requests orderly worker shutdown.
    Close {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
}

impl WorkerRequest {
    /// Returns this envelope's request-kind discriminator.
    pub const fn kind(&self) -> RequestKind {
        match self {
            Self::Info { .. } => RequestKind::Info,
            Self::Submit { .. } => RequestKind::Submit,
            Self::Close { .. } => RequestKind::Close,
        }
    }

    /// Returns the optional request-response correlation identity.
    pub const fn message_id(&self) -> Option<u64> {
        match self {
            Self::Info { message_id }
            | Self::Submit { message_id, .. }
            | Self::Close { message_id } => *message_id,
        }
    }

    /// Replaces the request-response correlation identity (`message_id`).
    pub fn set_message_id(&mut self, value: Option<u64>) {
        match self {
            Self::Info { message_id }
            | Self::Submit { message_id, .. }
            | Self::Close { message_id } => *message_id = value,
        }
    }

    /// Returns the batch carried by this request, if any.
    pub const fn batch(&self) -> Option<&Batch> {
        match self {
            Self::Submit { batch, .. } => Some(batch),
            _ => None,
        }
    }

    /// Constructs a worker-capability request.
    pub fn info() -> Self {
        Self::Info { message_id: None }
    }

    /// Constructs a batch-submission request.
    pub fn submit(batch: Batch) -> Self {
        Self::Submit {
            message_id: None,
            batch: Box::new(batch),
        }
    }

    /// Constructs a worker-shutdown request.
    pub fn close() -> Self {
        Self::Close { message_id: None }
    }
}

/// Kind discriminator carried by each worker response frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseKind {
    /// Worker capability response.
    Info,
    /// Partial or terminal batch result.
    Result,
    /// Successful control acknowledgement.
    Ok,
    /// Structured worker failure.
    Error,
    /// A lane whose job has not completed within the rank's stall bound.
    Stalled,
}

/// Request and call identity attached to a worker error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorCallIdentity {
    /// Request owning the failed call.
    pub request_key: RequestKey,
    /// Failed call identity.
    pub call_id: CallId,
}

/// Structured worker failure with optional call identities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerResponseError {
    /// Human-readable failure description.
    pub message: String,
    /// The failure's class, when the rank classified it.
    pub code: Option<ErrorCode>,
    /// Whether the worker is unsafe to serve further requests.
    pub fatal: bool,
    /// Optional worker phase that failed.
    pub phase: Option<String>,
    /// Call kind of the failed batch (for example `prefill`), or the forward
    /// mode of the failed numerical call within it; absent when the failure
    /// belongs to no batch with calls, such as an undecodable request, a
    /// submission refused at admission, or a lifecycle-only batch.
    pub route: Option<String>,
    /// Calls affected by the failure.
    pub calls: Vec<ErrorCallIdentity>,
}

/// A job one of a rank's lanes has performed for at least the rank's stall
/// bound, half the head's watchdog timeout
/// ([`LaunchDescriptor::watchdog_timeout_ms`](crate::LaunchDescriptor)),
/// as the rank's control thread observes it: the job was submitted and its
/// completion has not arrived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stall {
    /// The lane performing the job: its configured lane id, or `lane-0` for
    /// the full-device lane of a rank without configured lanes. `None` when
    /// the rank's Python thread performs it, as it does every job of a call
    /// kind no lane serves (every job of a rank without a CUDA device).
    pub lane: Option<String>,
    /// The batch the job belongs to: the one whose forward, input
    /// admission or output wait it is. `None` for a job of no batch, such
    /// as a lifecycle command's work.
    pub batch_id: Option<u64>,
    /// Call kind of that batch (for example `prefill`).
    pub route: Option<String>,
    /// Import path of the runner method the job calls; `None` for a job
    /// that calls none, such as a graph launch.
    pub method: Option<String>,
    /// The calls of that batch.
    pub calls: Vec<ErrorCallIdentity>,
    /// How long the job has run, in milliseconds: since its submission, or
    /// since the lane completed the job before it, whichever is later.
    pub running_ms: u64,
}

/// Worker-to-head response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerResponse {
    /// Returns worker identity and capabilities.
    Info {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// Loaded worker capabilities.
        info: WorkerInfo,
    },
    /// Returns partial or terminal batch progress.
    Result {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// Physical batch result.
        result: BatchOutput,
    },
    /// Acknowledges a successful control request.
    Ok {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
    /// Returns a structured worker failure.
    Error {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// Structured failure details.
        error: WorkerResponseError,
    },
    /// Reports, unasked, a lane that stays inside one job. The rank's
    /// control thread sends it once, and goes on serving the channel; the
    /// head fails the rank on it.
    Stalled {
        /// The lane, the job's batch and how long the job has run.
        stall: Stall,
    },
}

impl WorkerResponse {
    /// Returns this envelope's response-kind discriminator.
    pub const fn kind(&self) -> ResponseKind {
        match self {
            Self::Info { .. } => ResponseKind::Info,
            Self::Result { .. } => ResponseKind::Result,
            Self::Ok { .. } => ResponseKind::Ok,
            Self::Error { .. } => ResponseKind::Error,
            Self::Stalled { .. } => ResponseKind::Stalled,
        }
    }

    /// Returns the optional request-response correlation identity; `None`
    /// for a stall report, which answers no request.
    pub const fn message_id(&self) -> Option<u64> {
        match self {
            Self::Info { message_id, .. }
            | Self::Result { message_id, .. }
            | Self::Ok { message_id }
            | Self::Error { message_id, .. } => *message_id,
            Self::Stalled { .. } => None,
        }
    }

    /// Replaces the request-response correlation identity (`message_id`) of
    /// a response; a stall report has none to replace.
    pub fn set_message_id(&mut self, value: Option<u64>) {
        match self {
            Self::Info { message_id, .. }
            | Self::Result { message_id, .. }
            | Self::Ok { message_id }
            | Self::Error { message_id, .. } => *message_id = value,
            Self::Stalled { .. } => {}
        }
    }

    /// Returns the batch result carried by this response, if any.
    pub const fn output(&self) -> Option<&BatchOutput> {
        match self {
            Self::Result { result, .. } => Some(result),
            _ => None,
        }
    }

    /// Constructs a worker-capability response.
    pub fn info(info: WorkerInfo) -> Self {
        Self::Info {
            message_id: None,
            info,
        }
    }

    /// Constructs a batch-result response.
    pub fn result(result: BatchOutput) -> Self {
        Self::Result {
            message_id: None,
            result,
        }
    }

    /// Constructs an acknowledgment response.
    pub fn ok() -> Self {
        Self::Ok { message_id: None }
    }

    /// Constructs a structured worker-error response.
    pub fn error(error: WorkerResponseError) -> Self {
        Self::Error {
            message_id: None,
            error,
        }
    }
}

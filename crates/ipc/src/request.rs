//! Worker request and response envelopes.
//!
//! One [`MainProcessRequest`] or [`MainProcessResponse`] is the message of one
//! [`Frame`]. Each envelope's optional `message_id` is the
//! request-response correlation identity: a rank responds to a request under the
//! identity it carried, and a transport that matches a response to its request
//! reads the identity from the decoded envelope, taking `None` as zero. The one
//! message a rank sends unasked, [`MainProcessResponse::Stalled`], responds to no
//! request and carries no identity.
//!
//! Between the engine processes of an epoch these frames carry the jobs of a
//! worker group whose ranks are on several hosts: its batches and shadow
//! batches under the numbers the scheduler node gave them, the buffer pool's
//! completions the ranks' device block tables are fed by, and the requests the
//! ranks stamp. A request or a completion of the buffer pool travels as the
//! record of its own encoding (`storage::runtime`), the one statement of its
//! layout. Every [`MainProcessResponse::Result`] states the last numbered job
//! the responding process's ranks of the group have ended, so the group's
//! `completed` is the smallest over all of its ranks.

use storage::runtime::{Completion, GroupId, Request, RequestId};

use super::*;

/// Kind discriminator carried by each worker request frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    /// Queries worker identity and capabilities.
    Info,
    /// Submits a batch.
    Submit,
    /// Submits a shadow batch.
    Shadow,
    /// Maps a completion of the buffer pool into the ranks' device block tables.
    Map,
    /// Applies requests to the ranks' device block tables and stamps them.
    Stamp,
    /// Requests orderly worker shutdown.
    Close,
    /// Promotes a standby worker to serving.
    Promote,
    /// Runs a named worker hook.
    Hook,
}

impl RequestKind {}

/// Engine-to-worker request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MainProcessRequest {
    /// Asks for the capacity of the worker group of the process's first rank. Responds with
    /// [`MainProcessResponse::Info`].
    Info {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
    /// Submits a batch to the process's ranks of its group. Responds with
    /// [`MainProcessResponse::Result`]: the batch's output and the `Finish` of
    /// each copy the ranks ran with it.
    Submit {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// The batch's number among the group's jobs that address its device
        /// block tables, which the scheduler node gave it.
        sequence: u64,
        /// Whether the process's ranks only take the number: another rank of
        /// the group runs the batch alone.
        pass: bool,
        /// The batch.
        batch: Box<crate::batch::Batch>,
    },
    /// Submits a shadow batch: the paired forwards of the batch's generation
    /// against the serving one, which a group's ranks run together as they run
    /// its serving batches. Responds as [`Self::Submit`] is.
    Shadow {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// The shadow batch's number among the group's jobs, as for a batch.
        sequence: u64,
        /// The shadow batch.
        batch: Box<crate::batch::Batch>,
    },
    /// Maps a completion of the buffer pool, with the request it answers, into
    /// the device block tables of the process's ranks of `group`, under
    /// `slot`. Responds with an empty [`MainProcessResponse::Result`].
    Map {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        group: GroupId,
        /// The slot under which the request's entries are kept in the group.
        slot: u32,
        /// The request's record (`storage::runtime::Request::encode`), under
        /// the request's number.
        request: Vec<u8>,
        /// The completion's record (`storage::runtime::Completion::encode`).
        completion: Vec<u8>,
    },
    /// Applies requests to the device block tables of the process's ranks of
    /// `group` and stamps them. Responds with a [`MainProcessResponse::Result`]
    /// that returns the requests, each with the group's stamp as the process's
    /// ranks state it.
    Stamp {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        group: GroupId,
        /// The requests' records (`storage::runtime::Request::encode`); their
        /// numbers are not read.
        requests: Vec<Vec<u8>>,
    },
    /// Requests orderly worker shutdown.
    Close {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
    /// Promotes a parked standby worker to serving: it takes write access
    /// to the exclusive storages its predecessor held, once the predecessor
    /// is out, and makes their mappings writable at unchanged addresses.
    /// Responds with `Ok` once the worker serves.
    Promote {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
    /// Runs the worker hook `name` (`uniserve.hooks`), which a plugin
    /// implements, with `payload`, a JSON value. The engine sends it while
    /// the worker runs no forward. Responds with
    /// [`MainProcessResponse::Hooked`] and the hook's JSON result.
    Hook {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        name: String,
        payload: String,
    },
}

impl MainProcessRequest {
    /// Returns this envelope's request-kind discriminator.
    pub const fn kind(&self) -> RequestKind {
        match self {
            Self::Info { .. } => RequestKind::Info,
            Self::Submit { .. } => RequestKind::Submit,
            Self::Shadow { .. } => RequestKind::Shadow,
            Self::Map { .. } => RequestKind::Map,
            Self::Stamp { .. } => RequestKind::Stamp,
            Self::Close { .. } => RequestKind::Close,
            Self::Promote { .. } => RequestKind::Promote,
            Self::Hook { .. } => RequestKind::Hook,
        }
    }

    /// Replaces the request-response correlation identity (`message_id`).
    pub fn set_message_id(&mut self, value: Option<u64>) {
        match self {
            Self::Info { message_id }
            | Self::Submit { message_id, .. }
            | Self::Shadow { message_id, .. }
            | Self::Map { message_id, .. }
            | Self::Stamp { message_id, .. }
            | Self::Close { message_id }
            | Self::Promote { message_id }
            | Self::Hook { message_id, .. } => *message_id = value,
        }
    }

    /// Returns the batch carried by this request, if any.
    pub const fn batch(&self) -> Option<&crate::batch::Batch> {
        match self {
            Self::Submit { batch, .. } | Self::Shadow { batch, .. } => Some(batch),
            _ => None,
        }
    }

    /// Constructs a worker-capability request.
    pub fn info() -> Self {
        Self::Info { message_id: None }
    }

    /// Constructs a batch-submission request of the batch numbered
    /// `sequence`, which the process's ranks only number when `pass` is set.
    pub fn submit(sequence: u64, pass: bool, batch: crate::batch::Batch) -> Self {
        Self::Submit {
            message_id: None,
            sequence,
            pass,
            batch: Box::new(batch),
        }
    }

    /// Constructs a shadow-batch request of the shadow batch numbered
    /// `sequence`.
    pub fn shadow(sequence: u64, batch: crate::batch::Batch) -> Self {
        Self::Shadow {
            message_id: None,
            sequence,
            batch: Box::new(batch),
        }
    }

    /// Constructs the request that maps `completion`, the answer to
    /// `request` numbered `id`, into the device block tables of `group` under
    /// `slot`.
    pub fn map(
        group: GroupId,
        slot: u32,
        id: RequestId,
        request: &Request,
        completion: &Completion,
    ) -> Self {
        let (mut encoded, mut answer) = (Vec::new(), Vec::new());
        request.encode(id, &mut encoded);
        completion.encode(id, &mut answer);

        Self::Map {
            message_id: None,
            group,
            slot,
            request: encoded,
            completion: answer,
        }
    }

    /// Constructs the request that has the ranks of `group` apply and stamp
    /// `requests`.
    pub fn stamp(group: GroupId, requests: &[Request]) -> Self {
        Self::Stamp {
            message_id: None,
            group,
            requests: records(requests),
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
    /// A job of the rank that has not completed within the rank's stall
    /// bound.
    Stalled,
    /// A worker hook's result.
    Hooked,
}

/// Structured worker failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MainProcessResponseError {
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
}

/// A job the rank's builder thread has performed for at least the rank's
/// stall bound, as the rank's control thread observes it: the job was
/// submitted and its completion has not arrived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stall {
    /// The batch the job belongs to: the one whose forward, input
    /// admission or output wait it is. `None` for a job of no batch, such
    /// as a lifecycle command's work.
    pub batch_id: Option<u64>,
    /// Call kind of that batch (for example `prefill`).
    pub route: Option<String>,
    /// Import path of the library method the job calls; `None` for a job
    /// that calls none, such as a copy or a wait.
    pub method: Option<String>,
    /// How long the job has run, in milliseconds: since its submission, or
    /// since the builder thread completed the job before it, whichever is
    /// later.
    pub running_ms: u64,
}

/// Worker-to-engine response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MainProcessResponse {
    /// States what the process's ranks of a worker group serve: the capacity its first rank
    /// stated once its executors were ready, which the scheduler node plans the group against.
    Info {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// The group's capacity.
        capacity: crate::Capacity,
    },
    /// Responds to a job of a worker group: a batch, a shadow batch, a
    /// completion mapped, requests stamped.
    Result {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// A batch's output; empty for every other job.
        result: crate::batch::BatchOutput,
        /// The requests the job returns for the buffer pool's submission
        /// queue, as records (`storage::runtime::Request::encode`) whose
        /// numbers are not read: a stamp's requests, each stamped; the `Finish`
        /// of each copy a batch ran.
        requests: Vec<Vec<u8>>,
        /// The last numbered job of the group every rank of the responding
        /// process has ended, with every earlier one ended.
        ended: u64,
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
        error: MainProcessResponseError,
    },
    /// Responds to a [`MainProcessRequest::Hook`] with the hook's JSON result.
    Hooked {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        result: String,
    },
    /// Reports, unasked, a job the rank's builder thread stays inside. The
    /// rank's control thread sends it once, and goes on serving the
    /// channel; the engine fails the rank on it.
    Stalled {
        /// The job's batch, its method and how long it has run.
        stall: Stall,
    },
}

impl MainProcessResponse {
    /// Returns this envelope's response-kind discriminator.
    pub const fn kind(&self) -> ResponseKind {
        match self {
            Self::Info { .. } => ResponseKind::Info,
            Self::Result { .. } => ResponseKind::Result,
            Self::Ok { .. } => ResponseKind::Ok,
            Self::Error { .. } => ResponseKind::Error,
            Self::Stalled { .. } => ResponseKind::Stalled,
            Self::Hooked { .. } => ResponseKind::Hooked,
        }
    }

    /// Replaces the request-response correlation identity (`message_id`) of
    /// a response; a stall report has none to replace.
    pub fn set_message_id(&mut self, value: Option<u64>) {
        match self {
            Self::Info { message_id, .. }
            | Self::Result { message_id, .. }
            | Self::Ok { message_id }
            | Self::Error { message_id, .. }
            | Self::Hooked { message_id, .. } => *message_id = value,
            Self::Stalled { .. } => {}
        }
    }

    /// Returns the batch result carried by this response, if any.
    pub const fn output(&self) -> Option<&crate::batch::BatchOutput> {
        match self {
            Self::Result { result, .. } => Some(result),
            _ => None,
        }
    }

    /// Constructs the response that states a group's capacity.
    pub fn info(capacity: crate::Capacity) -> Self {
        Self::Info {
            message_id: None,
            capacity,
        }
    }

    /// Constructs the response to a job of a worker group that returns
    /// `result` and `requests`, from a process whose ranks of the group have
    /// ended every numbered job up to `ended`.
    pub fn result(result: crate::batch::BatchOutput, requests: &[Request], ended: u64) -> Self {
        Self::Result {
            message_id: None,
            result,
            requests: records(requests),
            ended,
        }
    }

    /// Constructs an acknowledgment response.
    pub fn ok() -> Self {
        Self::Ok { message_id: None }
    }

    /// Constructs a structured worker-error response.
    pub fn error(error: MainProcessResponseError) -> Self {
        Self::Error {
            message_id: None,
            error,
        }
    }
}

/// The records of `requests`, each under the number 0, which no reader of a
/// [`MainProcessRequest::Stamp`] or a [`MainProcessResponse::Result`] reads.
fn records(requests: &[Request]) -> Vec<Vec<u8>> {
    requests
        .iter()
        .map(|request| {
            let mut record = Vec::new();
            request.encode(RequestId(0), &mut record);
            record
        })
        .collect()
}

/// The requests of `records`, as [`MainProcessRequest::Stamp`] and
/// [`MainProcessResponse::Result`] carry them.
///
/// # Errors
///
/// A record that is not a request of this version of the buffer pool's
/// interface.
pub fn decode_requests(records: &[Vec<u8>]) -> storage::Result<Vec<Request>> {
    records
        .iter()
        .map(|record| Request::decode(record).map(|(_, request)| request))
        .collect()
}

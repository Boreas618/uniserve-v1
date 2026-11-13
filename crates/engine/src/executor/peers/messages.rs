//! Worker request and response envelopes.
//!
//! One [`MainProcessRequest`] or [`MainProcessResponse`] is the message of one
//! frame of the peer link ([`super::link`]). Each envelope's optional `message_id` is the
//! request-response correlation identity: a rank responds to a request under the
//! identity it carried, and a transport that matches a response to its request
//! reads the identity from the decoded envelope, taking `None` as zero. The one
//! message a rank sends unasked, [`MainProcessResponse::Stalled`], responds to no
//! request and carries no identity.
//!
//! Between the engine processes of an epoch these frames carry:
//!
//! - **The session.** The scheduler node relays its session with its host's buffer pool once,
//!   at the start ([`MainProcessRequest::Session`]): the layout it opened and the answer, whose
//!   numbers every completion, block table and copy names. The peer answers once its ranks are
//!   open, bound, spawned and warm ([`MainProcessResponse::Placed`]): where their units lie, the
//!   segments they created, and what they serve.
//! - **The jobs of a worker group** whose ranks are on several hosts: its batches and shadow
//!   batches under the numbers the scheduler node gave them, the buffer pool's completions the
//!   ranks' device block tables are fed by, and the requests the ranks stamp. A request or a
//!   completion of the buffer pool travels as the record of its own encoding
//!   (`storage::runtime`), the one statement of its layout. Every response to a job states the
//!   last numbered job the responding process's ranks of the group have ended, so the group's
//!   `completed` is the smallest over all of its ranks.
//! - **A generation's switch, commit and rollback**, in the order of every group's jobs on
//!   every rank ([`MainProcessRequest::Switch`]).
//! - **The data plane's operations** on the peer's ranks ([`MainProcessRequest::Data`],
//!   [`DataOperation`]), each answered with the JSON of what the operation returns.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use storage::protocol::UnitRun;
use storage::runtime::{Completion, GroupId, Open, Opened, Request, RequestId, Segments};

use crate::batch::{Batch, BatchOutput, Capacity, ErrorCode};

/// Engine-to-worker request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MainProcessRequest {
    /// Relays the scheduler node's session with its host's buffer pool, once, after `Open` was
    /// answered: the peer's ranks map the segments `opened` states, create the rest, and take
    /// their parts of the host regions at their own host's Storage Server. Responds with
    /// [`MainProcessResponse::Placed`] once the peer's ranks are open, bound, spawned and warm.
    Session {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// The layout the scheduler node opened the session with.
        open: Box<Open>,
        /// The buffer pool's answer: the numbers of the regions and their segment tables.
        opened: Box<Opened>,
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
        batch: Box<Batch>,
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
        batch: Box<Batch>,
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
    /// Switches every rank of the process to the prepared generation `generation`, between the
    /// jobs of every group handed before and after it. Responds with an empty
    /// [`MainProcessResponse::Result`].
    Switch {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        generation: u64,
    },
    /// Commits the trial of generation `generation` on every rank of the process, behind the
    /// jobs handed before it. Responds as [`Self::Switch`] is.
    Commit {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        generation: u64,
    },
    /// Rolls generation `generation` back on every rank of the process, between the jobs of
    /// every group handed before and after it. Responds as [`Self::Switch`] is.
    Rollback {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        generation: u64,
    },
    /// Runs one operation of the data plane on the process's ranks. Responds with
    /// [`MainProcessResponse::Operated`], or with [`MainProcessResponse::Error`] when the
    /// operation failed.
    Data {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        operation: DataOperation,
    },
    /// Requests orderly worker shutdown.
    Close {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
    },
}

impl MainProcessRequest {
    /// Replaces the request-response correlation identity (`message_id`).
    pub(crate) fn set_message_id(&mut self, value: Option<u64>) {
        match self {
            Self::Session { message_id, .. }
            | Self::Submit { message_id, .. }
            | Self::Shadow { message_id, .. }
            | Self::Map { message_id, .. }
            | Self::Stamp { message_id, .. }
            | Self::Switch { message_id, .. }
            | Self::Commit { message_id, .. }
            | Self::Rollback { message_id, .. }
            | Self::Data { message_id, .. }
            | Self::Close { message_id } => *message_id = value,
        }
    }

    /// Constructs a batch-submission request of the batch numbered
    /// `sequence`, which the process's ranks only number when `pass` is set.
    pub(crate) fn submit(sequence: u64, pass: bool, batch: Batch) -> Self {
        Self::Submit {
            message_id: None,
            sequence,
            pass,
            batch: Box::new(batch),
        }
    }

    /// Constructs a shadow-batch request of the shadow batch numbered
    /// `sequence`.
    pub(crate) fn shadow(sequence: u64, batch: Batch) -> Self {
        Self::Shadow {
            message_id: None,
            sequence,
            batch: Box::new(batch),
        }
    }

    /// Constructs the request that maps `completion`, the answer to
    /// `request` numbered `id`, into the device block tables of `group` under
    /// `slot`.
    pub(crate) fn map(
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
    pub(crate) fn stamp(group: GroupId, requests: &[Request]) -> Self {
        Self::Stamp {
            message_id: None,
            group,
            requests: records(requests),
        }
    }

    /// Constructs the request that relays the session `open` opened and `opened` answered.
    pub(crate) fn session(open: &Open, opened: &Opened) -> Self {
        Self::Session {
            message_id: None,
            open: Box::new(open.clone()),
            opened: Box::new(opened.clone()),
        }
    }

    /// Constructs the request that runs `operation` on the process's ranks.
    pub(crate) fn data(operation: DataOperation) -> Self {
        Self::Data {
            message_id: None,
            operation,
        }
    }

    /// Constructs a worker-shutdown request.
    pub(crate) fn close() -> Self {
        Self::Close { message_id: None }
    }
}

/// One operation of the data plane (`crate::process::data_plane::DataPlane`) on a peer's
/// ranks, with the arguments of the method that runs it there. The answer
/// ([`MainProcessResponse::Operated`]) is the JSON of the method's value.
///
/// A weight load's replacement stays in the process that loaded it: the peer keeps its own
/// from [`Self::Pair`] to the [`Self::Stage`] or [`Self::Discard`] that ends it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataOperation {
    /// Releases the KV planes' segments from segment `from` on, on every rank of `groups`.
    /// Answers the CUDA ordinals of the devices of the ranks that gave segments back.
    ReleaseSegments { groups: Vec<GroupId>, from: u32 },
    /// Creates again the first `count` segments the ranks of `groups` gave back, or all of them.
    /// Answers each such rank's segment table.
    CreateSegments {
        groups: Vec<GroupId>,
        count: Option<u32>,
    },
    /// Answers the segment table of every rank of `groups`.
    SegmentTables { groups: Vec<GroupId> },
    /// Answers how many of the segments the ranks of `groups` gave back every such rank has room
    /// for; none when no such rank splits its planes.
    FittingSegments { groups: Vec<GroupId> },
    /// Answers the room of the device of every rank of `groups` whose planes are split.
    DeviceRooms { groups: Vec<GroupId> },
    /// Answers the room every device has for a load.
    Room,
    /// Answers the bytes the graph pools map on the device that maps the fewest; none for a
    /// process of no device.
    GraphBytes,
    /// Suspends the graph pools and releases every segment. Answers every rank's segment table.
    Sleep,
    /// Restores the graph pools and creates every segment again. Answers every rank's segment
    /// table.
    Wake,
    /// Suspends the graph pools.
    SuspendGraphs,
    /// Restores the graph pools.
    RestoreGraphs,
    /// Has the ranks' caches and states launch from the generation their graphs serve.
    SetKernels,
    /// Reserves what the ranks' processes hold beside their pools in their GPUs' quotas.
    ReserveServing,
    /// Pairs a load of `checkpoint` with the expert order `experts` with the weights that serve,
    /// and keeps the replacement. Answers the most device bytes a rank needs beside what it
    /// serves.
    Pair {
        checkpoint: String,
        experts: BTreeMap<String, Vec<u32>>,
        rearranged: bool,
    },
    /// Publishes what the kept replacement changes.
    Load,
    /// Keeps the loaded replacement for the boundary that puts it in place.
    Stage,
    /// Ends the kept replacement: what its load published goes back.
    Discard,
    /// Puts the staged replacement in place. Answers each rank's report.
    UpdateWeights,
    /// Puts the redistribution of the routed experts made for `order` in place. Answers each
    /// rank's report.
    CommitExperts { order: BTreeMap<String, Vec<u32>> },
    /// Answers the names of the weights every rank serves, in rank order and each rank's load
    /// order: what the identity of the deployment's weights is the digest of.
    ServingOrder,
    /// Answers once the process's own preparation of generation `generation` ended, or the
    /// preparation's failure: no shadow batch of the generation reaches its ranks before.
    Prepared { generation: u64 },
    /// Answers the operators the process's ranks have no comparison for in the generation update
    /// in flight, each with the reason: those their preparation added and those the shadow
    /// batches that reached them could not compare. Asked once the shadow pass ended.
    Uncompared,
    /// Runs the operation the daemon names `name` with the JSON `payload` beside the batches:
    /// an expert redistribution's stages, a LoRA adapter's load and unload, a profiling
    /// session's changes, the fence probes' switch. Answers each rank's result.
    Hook { name: String, payload: String },
}

/// Kind discriminator carried by each worker response frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseKind {
    /// What a peer's ranks hold once its start relayed the session.
    Placed,
    /// Partial or terminal batch result.
    Result,
    /// The value of a data plane's operation.
    Operated,
    /// Structured worker failure.
    Error,
    /// A job of the rank that has not completed within the rank's stall
    /// bound.
    Stalled,
}

/// Structured worker failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct MainProcessResponseError {
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
pub(crate) struct Stall {
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
#[expect(
    clippy::large_enum_variant,
    reason = "a batch's output travels by value once per batch, in the frame it is decoded from"
)]
pub(crate) enum MainProcessResponse {
    /// Answers the relayed session ([`MainProcessRequest::Session`]) once the peer's ranks are
    /// open, bound, spawned and warm: what the scheduler node grants, places and schedules them by.
    Placed {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// The TCP address of the peer's host's Storage Server (`StorageLaunch::tcp`), at
        /// which its ranks' units are retained.
        endpoint: String,
        /// Per KV region by name, the segments the peer's ranks created, as `(segment, first
        /// unit, units)`: every rank of a region created the same.
        created: Vec<(String, Segments)>,
        /// Per region, device memory and part (the rank's index in its worker group), where a
        /// rank's part of the region's units lies (`Place`), which the scheduler node states for
        /// it (`PlaceRemote`).
        places: Vec<(String, String, u32, Vec<UnitRun>)>,
        /// What the peer's ranks serve, per worker group they are ranks of.
        capacities: Vec<(GroupId, Capacity)>,
    },
    /// Responds to a job of a worker group: a batch, a shadow batch, a
    /// completion mapped, requests stamped.
    Result {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// A batch's output; empty for every other job.
        result: BatchOutput,
        /// The requests the job returns for the buffer pool's submission
        /// queue, as records (`storage::runtime::Request::encode`) whose
        /// numbers are not read: a stamp's requests, each stamped; the `Finish`
        /// of each copy a batch ran.
        requests: Vec<Vec<u8>>,
        /// The last numbered job of the group every rank of the responding
        /// process has ended, with every earlier one ended.
        ended: u64,
    },
    /// Answers a data plane's operation ([`MainProcessRequest::Data`]).
    Operated {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// The JSON of the operation's value.
        reply: String,
    },
    /// Returns a structured worker failure.
    Error {
        /// Optional request-response correlation identity.
        message_id: Option<u64>,
        /// Structured failure details.
        error: MainProcessResponseError,
        /// The `Finish { ok: false }` of each copy a failed batch had started, stamped with the
        /// batch, as records (`storage::runtime::Request::encode`) whose numbers are not read;
        /// empty for any other failure.
        requests: Vec<Vec<u8>>,
        /// The last numbered job of the failed job's group every rank of the responding
        /// process has ended, with every earlier one ended; zero for a job of no group.
        ended: u64,
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
    pub(crate) const fn kind(&self) -> ResponseKind {
        match self {
            Self::Placed { .. } => ResponseKind::Placed,
            Self::Result { .. } => ResponseKind::Result,
            Self::Operated { .. } => ResponseKind::Operated,
            Self::Error { .. } => ResponseKind::Error,
            Self::Stalled { .. } => ResponseKind::Stalled,
        }
    }

    /// Replaces the request-response correlation identity (`message_id`) of
    /// a response; a stall report has none to replace.
    pub(crate) fn set_message_id(&mut self, value: Option<u64>) {
        match self {
            Self::Placed { message_id, .. }
            | Self::Result { message_id, .. }
            | Self::Operated { message_id, .. }
            | Self::Error { message_id, .. } => *message_id = value,
            Self::Stalled { .. } => {}
        }
    }

    /// Constructs the response to a job of a worker group that returns
    /// `result` and `requests`, from a process whose ranks of the group have
    /// ended every numbered job up to `ended`.
    pub(crate) fn result(result: BatchOutput, requests: &[Request], ended: u64) -> Self {
        Self::Result {
            message_id: None,
            result,
            requests: records(requests),
            ended,
        }
    }

    /// Constructs the response to request `id` that failed for `message`, returning `finished`,
    /// the `Finish { ok: false }` of each copy a failed batch had started, from a process whose
    /// ranks of the job's group have ended every numbered job up to `ended`.
    pub(crate) fn failed(id: u64, message: String, finished: &[Request], ended: u64) -> Self {
        Self::Error {
            message_id: Some(id),
            error: MainProcessResponseError {
                message,
                code: None,
                fatal: false,
                phase: None,
                route: None,
            },
            requests: records(finished),
            ended,
        }
    }
}

/// The records of `requests`, each under the number 0, which no reader of a
/// [`MainProcessRequest::Stamp`], a [`MainProcessResponse::Result`] or a
/// [`MainProcessResponse::Error`] reads.
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

/// The requests of `records`, as [`MainProcessRequest::Stamp`], [`MainProcessResponse::Result`]
/// and [`MainProcessResponse::Error`] carry them.
///
/// # Errors
///
/// A record that is not a request of this version of the buffer pool's
/// interface.
pub(crate) fn decode_requests(records: &[Vec<u8>]) -> storage::Result<Vec<Request>> {
    records
        .iter()
        .map(|record| Request::decode(record).map(|(_, request)| request))
        .collect()
}

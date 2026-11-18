//! The frames of the peer link and the requests and responses they carry.
//!
//! Every frame the scheduler node sends a peer is a [`ToPeer`]: one [`MainProcessRequest`] under
//! the number the scheduler node's link gave it. Every frame a peer sends back is a [`FromPeer`]:
//! one [`MainProcessResponse`] under the number of the request it answers. The number is the
//! frame's, not the request's: the link that sends a request numbers it, and the request and the
//! response themselves state none.
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
//!   [`DataOperation`]), each answered with what the operation returns, in the link's own
//!   encoding ([`MainProcessResponse::Operated`]); among them the operations the daemon commands
//!   on every rank beside the batches ([`RankOperation`]), answered with their typed reports
//!   ([`RankReports`]).

use std::collections::BTreeMap;
use std::path::PathBuf;

use proto::profile::{Collector, RankRelease, RankStart, RankStop, SessionStart};
use serde::{Deserialize, Serialize};
use storage::runtime::{Completion, GroupId, Open, Opened, Request, RequestId, Segments};

use crate::batch::{Batch, BatchOutput, Capacity, ErrorCode};
use crate::worker::capture::Key;

/// Engine-to-worker request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MainProcessRequest {
    /// Relays the scheduler node's session with its host's buffer pool, once, after `Open` was
    /// answered: the peer's ranks map the segments `opened` states, create the rest, and take
    /// their parts of the host regions at their own host's Storage Server. Responds with
    /// [`MainProcessResponse::Placed`] once the peer's ranks are open, bound, spawned and warm.
    Session {
        /// The layout the scheduler node opened the session with.
        open: Box<Open>,
        /// The buffer pool's answer: the numbers of the regions and their segment tables.
        opened: Box<Opened>,
    },
    /// Submits a batch to the process's ranks of its group. Responds with
    /// [`MainProcessResponse::Result`]: the batch's output and the `Finish` of
    /// each copy the ranks ran with it.
    Submit {
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
        /// The shadow batch's number among the group's jobs, as for a batch.
        sequence: u64,
        /// The shadow batch.
        batch: Box<Batch>,
    },
    /// Maps a completion of the buffer pool, with the request it answers, into
    /// the device block tables of the process's ranks of `group`, under
    /// `slot`. Responds with an empty [`MainProcessResponse::Result`].
    Map {
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
        group: GroupId,
        /// The requests' records (`storage::runtime::Request::encode`); their
        /// numbers are not read.
        requests: Vec<Vec<u8>>,
    },
    /// Switches every rank of the process to the prepared generation `generation`, between the
    /// jobs of every group handed before and after it. Responds with an empty
    /// [`MainProcessResponse::Result`].
    Switch { generation: u64 },
    /// Switches `buckets` to the prepared generation `generation` ahead of the others on every
    /// rank of the process, between the jobs of every group handed before and after it.
    /// Responds as [`Self::Switch`] is.
    SwitchBuckets { generation: u64, buckets: Vec<Key> },
    /// Responds with an empty [`MainProcessResponse::Result`] once every rank of the process
    /// installed the outcome of every bucket of `buckets` (every bucket the generation examines,
    /// when `None`) for the prepared generation `generation`, and with `Error` once one of them
    /// will not. It takes no place in any group's order, and its answer waits on none.
    Ready {
        generation: u64,
        buckets: Option<Vec<Key>>,
    },
    /// Commits the trial of generation `generation` on every rank of the process, behind the
    /// jobs handed before it. Responds as [`Self::Switch`] is.
    Commit { generation: u64 },
    /// Rolls generation `generation` back on every rank of the process, between the jobs of
    /// every group handed before and after it. Responds as [`Self::Switch`] is.
    Rollback { generation: u64 },
    /// Runs one operation of the data plane on the process's ranks. Responds with
    /// [`MainProcessResponse::Operated`], or with [`MainProcessResponse::Error`] when the
    /// operation failed.
    Data { operation: DataOperation },
    /// Requests orderly worker shutdown.
    Close,
}

impl MainProcessRequest {
    /// Constructs a batch-submission request of the batch numbered
    /// `sequence`, which the process's ranks only number when `pass` is set.
    pub(crate) fn submit(sequence: u64, pass: bool, batch: Batch) -> Self {
        Self::Submit {
            sequence,
            pass,
            batch: Box::new(batch),
        }
    }

    /// Constructs a shadow-batch request of the shadow batch numbered
    /// `sequence`.
    pub(crate) fn shadow(sequence: u64, batch: Batch) -> Self {
        Self::Shadow {
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
            group,
            requests: records(requests),
        }
    }

    /// Constructs the request that relays the session `open` opened and `opened` answered.
    pub(crate) fn session(open: &Open, opened: &Opened) -> Self {
        Self::Session {
            open: Box::new(open.clone()),
            opened: Box::new(opened.clone()),
        }
    }

    /// Constructs the request that runs `operation` on the process's ranks.
    pub(crate) fn data(operation: DataOperation) -> Self {
        Self::Data { operation }
    }

    /// Constructs a worker-shutdown request.
    pub(crate) fn close() -> Self {
        Self::Close
    }
}

/// A frame of the peer link from the scheduler node to a peer: `request`, numbered `id` by the
/// scheduler node's link, which the answer states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ToPeer {
    pub(crate) id: u64,
    pub(crate) request: MainProcessRequest,
}

/// A frame of the peer link from a peer to the scheduler node: `response`, the answer to the
/// request the scheduler node numbered `id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct FromPeer {
    pub(crate) id: u64,
    pub(crate) response: MainProcessResponse,
}

/// One operation of the data plane (`crate::process::data_plane::DataPlane`) on a peer's
/// ranks, with the arguments of the method that runs it there. The answer
/// ([`MainProcessResponse::Operated`]) is the method's value in the link's encoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataOperation {
    /// Releases the segments of device region `region` (the buffer pool's number: a KV region or
    /// a state region) from segment `from` on, on every rank of `groups`. Answers the CUDA
    /// ordinals of the devices of the ranks that gave segments back.
    ReleaseSegments {
        groups: Vec<GroupId>,
        region: u8,
        from: u32,
    },
    /// Creates again the first `count` segments of device region `region` the ranks of `groups`
    /// gave back, or all of them. Answers each such rank's segment table of the region.
    CreateSegments {
        groups: Vec<GroupId>,
        region: u8,
        count: Option<u32>,
    },
    /// Answers the segment table of device region `region` of every rank of `groups`.
    SegmentTables { groups: Vec<GroupId>, region: u8 },
    /// Answers how many of the segments of device region `region` the ranks of `groups` gave
    /// back every such rank has room for; none when no such rank splits its memory of it.
    FittingSegments { groups: Vec<GroupId>, region: u8 },
    /// Answers the room of the device of every rank of `groups` whose memory of device region
    /// `region` is split.
    DeviceRooms { groups: Vec<GroupId>, region: u8 },
    /// Answers how many of the segments of device region `region` the ranks of `groups` hold
    /// every such rank holds beside its warm process in its GPU's quota; none when no such rank
    /// splits its memory of it.
    WarmSegments { groups: Vec<GroupId>, region: u8 },
    /// Suspends the graph pools and releases every segment. Answers every rank's segment table.
    Sleep,
    /// Suspends the graph pools.
    SuspendGraphs,
    /// Restores the graph pools.
    RestoreGraphs,
    /// Has the ranks' caches and states launch from the generation their graphs serve.
    SetKernels,
    /// Reserves what the ranks' processes hold beside their pools in their GPUs' quotas.
    ReserveServing,
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
    /// Answers what a generation update that replaces `operators` asks of every rank's device
    /// of the peer (`crate::process::binding::UpdateRoom`), its runners measured under
    /// `options`, the generation's graph options where they differ from the serving ones.
    UpdateRoom {
        operators: Vec<String>,
        options: Option<proto::config::GraphOptions>,
    },
    /// Answers the bytes each GPU's quota limit of the peer falls by under `fractions`, every
    /// rank's share of its GPU by the rank's number (`WeightBinding::lowered_limits`).
    LoweredLimits {
        fractions: std::collections::BTreeMap<u32, f64>,
    },
    /// Takes `fractions` as every rank's share of its GPU (`WeightBinding::set_fractions`).
    SetFractions {
        fractions: std::collections::BTreeMap<u32, f64>,
    },
    /// Runs an operation the daemon commands on every rank beside the batches. Answers the
    /// peer's ranks' [`RankReports`].
    Rank(RankOperation),
}

/// An operation the daemon commands on every rank of the engine beside the batches (the request
/// interface's data-plane engine commands): the main process runs it on its own ranks and relays
/// it to every peer's ([`DataOperation::Rank`]). None changes what a batch reads: an expert
/// redistribution copies into new storage, and its change of the served weights is a commit at a
/// boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankOperation {
    /// Prepares the copies that put each named routed-expert layer's experts in its order.
    /// Answers [`RankReports::ExpertAllocations`].
    PrepareExperts { orders: BTreeMap<String, Vec<u32>> },
    /// Runs the prepared copies beside the batches. Answers [`RankReports::None`] once they
    /// completed.
    StartExperts,
    /// Abandons the prepared or copying redistribution. Answers [`RankReports::None`].
    AbortExperts,
    /// Loads the LoRA adapter at `path` into `slot`. Answers [`RankReports::Adapters`].
    LoadAdapter { slot: u32, path: PathBuf },
    /// Releases `slot`'s adapter. Answers [`RankReports::Adapters`].
    UnloadAdapter { slot: u32 },
    /// The step values of a configuration revision (`proto::config::Tuning`), which the
    /// scheduler node relays to every peer before its own schedulers read them: the peer's
    /// control thread takes what it reads itself (its watchdog's bound, its report period, its
    /// comparison bound, its NVTX ranges). Answers [`RankReports::None`].
    Tune(Box<proto::config::Tuning>),
    /// Starts a profiling session. Answers [`RankReports::ProfileStarts`].
    StartProfile(Box<SessionStart>),
    /// Ends profiling session `session` for `reason`. Answers [`RankReports::ProfileStops`].
    StopProfile { session: u64, reason: String },
    /// Turns `collector` of profiling session `session` off for `reason`. Answers
    /// [`RankReports::ProfileReleases`].
    ReleaseProfile {
        session: u64,
        collector: Collector,
        reason: String,
    },
    /// Switches the fence probes on or off, the module classes `off` names off whatever
    /// `enabled` says, and the kernel probes as `kernels` says while `enabled` holds. Answers
    /// [`RankReports::Fences`].
    SwitchFences {
        enabled: bool,
        off: Vec<String>,
        kernels: bool,
    },
    /// Reads the spans the process keeps in its rings. Answers [`RankReports::Spans`].
    ReadSpans,
}

/// What the ranks report of a [`RankOperation`], every rank's in rank order, the main process's
/// first and then each peer's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RankReports {
    /// The operation reports nothing.
    None,
    ExpertAllocations(Vec<ExpertAllocation>),
    Adapters(Vec<AdapterReport>),
    ProfileStarts(Vec<RankStart>),
    ProfileStops(Vec<RankStop>),
    ProfileReleases(Vec<RankRelease>),
    /// Each process's switch of its fence probes.
    Fences(Vec<FencesSwitched>),
    /// Each process's spans, as it named them.
    Spans(Vec<Vec<service::span::Span>>),
}

impl RankReports {
    /// Appends another process's reports of the same operation after these.
    ///
    /// # Errors
    ///
    /// Reports of another kind: the processes ran different operations.
    pub(crate) fn append(&mut self, other: Self) -> Result<(), String> {
        match (self, other) {
            (Self::None, Self::None) => {}
            (Self::ExpertAllocations(ours), Self::ExpertAllocations(theirs)) => ours.extend(theirs),
            (Self::Adapters(ours), Self::Adapters(theirs)) => ours.extend(theirs),
            (Self::ProfileStarts(ours), Self::ProfileStarts(theirs)) => ours.extend(theirs),
            (Self::ProfileStops(ours), Self::ProfileStops(theirs)) => ours.extend(theirs),
            (Self::ProfileReleases(ours), Self::ProfileReleases(theirs)) => ours.extend(theirs),
            (Self::Fences(ours), Self::Fences(theirs)) => ours.extend(theirs),
            (Self::Spans(ours), Self::Spans(theirs)) => ours.extend(theirs),
            (ours, theirs) => {
                return Err(format!(
                    "a process reported {theirs:?} beside {ours:?}: they ran different operations"
                ));
            }
        }
        Ok(())
    }
}

/// One rank's prepared allocation for a new expert order: the orders it took, the GPU it
/// allocated on (its UUID) and the bytes it reserves there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ExpertAllocation {
    pub(crate) orders: BTreeMap<String, Vec<u32>>,
    pub(crate) gpu: [u8; 16],
    pub(crate) bytes: u64,
}

/// One rank's adapter slot: the tensors a load mapped there; `None` for a release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AdapterReport {
    pub(crate) slot: u32,
    pub(crate) tensors: Option<u32>,
}

/// One process's switch of its fence probes: whether its graphs hold probes at all, the switch
/// it applied, and the probed classes it switched off.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FencesSwitched {
    pub(crate) probes: bool,
    pub(crate) enabled: bool,
    pub(crate) off: Vec<String>,
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
        /// The TCP address of the peer's host's Storage Server (`StorageLaunch::tcp`), at
        /// which its ranks' units are retained.
        endpoint: String,
        /// Per KV region by name, the segments the peer's ranks created, as `(segment, first
        /// unit, units)`: every rank of a region created the same.
        created: Vec<(String, Segments)>,
        /// Per region, device memory and part (the rank's index in its worker group), where a
        /// rank's part of the region's units lies (`Place`), which the scheduler node states for
        /// it (`PlaceRemote`).
        places: Vec<crate::process::ranks::Place>,
        /// What the peer's ranks serve, per worker group they are ranks of.
        capacities: Vec<(GroupId, Capacity)>,
    },
    /// Responds to a job of a worker group: a batch, a shadow batch, a
    /// completion mapped, requests stamped.
    Result {
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
        /// The operation's value, in the link's own encoding (MessagePack).
        #[serde(with = "serde_bytes")]
        reply: Vec<u8>,
    },
    /// Returns a structured worker failure.
    Error {
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
}

impl MainProcessResponse {
    /// Returns this envelope's response-kind discriminator.
    pub(crate) const fn kind(&self) -> ResponseKind {
        match self {
            Self::Placed { .. } => ResponseKind::Placed,
            Self::Result { .. } => ResponseKind::Result,
            Self::Operated { .. } => ResponseKind::Operated,
            Self::Error { .. } => ResponseKind::Error,
        }
    }

    /// Constructs the response to a job of a worker group that returns
    /// `result` and `requests`, from a process whose ranks of the group have
    /// ended every numbered job up to `ended`.
    pub(crate) fn result(result: BatchOutput, requests: &[Request], ended: u64) -> Self {
        Self::Result {
            result,
            requests: records(requests),
            ended,
        }
    }

    /// Constructs the response to a request that failed for `message`, returning `finished`, the
    /// `Finish { ok: false }` of each copy a failed batch had started, from a process whose ranks
    /// of the job's group have ended every numbered job up to `ended`.
    pub(crate) fn failed(message: String, finished: &[Request], ended: u64) -> Self {
        Self::Error {
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

//! `Worker`: one thread per device, the only issuer of work on the device's streams.
//!
//! Every other thread of the engine reaches a device through its worker's [`Queue`]: the
//! scheduler submits a batch, maps a completion of the buffer pool or has a request stamped, the
//! control thread a generation to prepare or switch to, a memory owner a copy it decided on. The
//! worker thread takes the jobs one at a time in the order they were submitted and calls the
//! device's runners, which nothing else calls. A stream's order is therefore the order of the
//! jobs, and a job that must run with nothing in flight on the device only has to be a job.
//!
//! ## Jobs
//!
//! A [`Job`] is one of:
//!
//! - a **batch to execute**, through the runner that serves the batch's phase
//!   ([`Runner::execute`]), under the number the engine gave it among the group's jobs. An
//!   expert-union member with no tokens of its own serves its part of a peer's step as a batch of
//!   variant `ExpertJoin`, which the stepping peer submits through [`Worker::join`] and which
//!   takes no number: it addresses no device block table. A batch another rank of the group runs
//!   alone is **passed**: the rank takes its number and runs nothing. A batch whose work is the
//!   host's (a media request's reading, encoding and muxing) takes two turns: the first starts
//!   the work on its runner's thread pool ([`Runner::start`]) and the job is held aside until the
//!   work ended, the second writes what it produced and ends the batch ([`Runner::execute`]); the
//!   jobs behind it run in between, and its number stays open across them;
//! - a **shadow batch** of the generation being prepared: one paired forward
//!   ([`Graphs::shadow`](crate::runner::graphs::Graphs::shadow)) per operator the batch's bucket
//!   launches whose kernel the generation replaces. It reads the device block tables, so it takes a
//!   number;
//! - a completion of the buffer pool to **map** into the rank's device block tables, and
//!   requests to **stamp** ([`BlockTables`]);
//! - the **capture** of the runners' buckets (warm-up), largest first;
//! - a generation to **prepare**, to **switch** to, to **commit** and to **roll back**, each
//!   on every runner of the device;
//! - a change of the device's **profiling session**;
//! - a **wait** for stream events, which deferred storage release hangs on;
//! - a **control operation**: a closure the worker calls with its runners. This is how a
//!   memory owner or the engine runs what must happen at a batch boundary, with nothing in
//!   flight on the device: suspending and restoring graph storage, a weight update. Work a
//!   storage owner decides on another thread is issued this way too, so nothing but the worker
//!   issues on the device's streams.
//!
//! A capture, a preparation and a shadow batch take one queue turn per bucket or operator and
//! go back to the tail in between, so batches submitted meanwhile are executed between two of
//! their turns instead of behind all of them.
//!
//! ## The device block tables and the sequences
//!
//! The worker thread owns the rank's [`BlockTables`]. A completion the scheduler maps and a
//! request it has stamped are jobs, so a batch sees exactly the entries of the jobs before it:
//! "before the group's next batch launches" is the queue's order. Before each batch the worker
//! hands every runner the table's changes and starts the queued copies the copy budget admits;
//! when the batch has ended it applies the `Finish` of each copy it ran and returns it, stamped,
//! with the batch's output.
//!
//! ## Copies
//!
//! A copy the buffer pool's `Copy` queues starts with a later numbered batch that every rank of
//! the group runs, on the device's copy stream, and the batch's end covers it. The copy budget
//! bounds the bytes the copies started with one batch move together ([`COPY_BUDGET`]): a batch
//! starts the oldest queued copies, in the order they were queued, while their bytes stay within
//! the budget, and the first copy that would pass it waits, with every copy behind it, for a
//! later batch. A copy larger than the budget starts alone, as the only copy of its batch. The
//! bytes are those [`BlockTables::queued_copies`] counts over the group's ranks, and every rank
//! of a group has the same budget, so all start the same copies with the same batch and return
//! the same `Finish`es.
//!
//! Each copy a batch starts is run by a runner whose memory holds both runs of each of its pairs
//! ([`Runner::maps`]), whichever runner serves the batch: the batch's own when it does, otherwise
//! the first that does (`crate::runner`, "Which runner runs a copy"). The batch ends with the
//! copies: once its runner returned, the worker waits for the copies of every runner it handed
//! some to ([`Runner::wait_copies`]) before it applies their `Finish`. A copy is reported run
//! when its runner issued it, and failed when the runner refused it or no runner maps it.
//!
//! No copy runs outside a batch. A group whose worker holds copies not yet finished, and for
//! which a step formed no other batch, is sent a batch with no rows (`specs/scheduler-design.md`
//! §2.1): it takes its number, hands the runners the tables' changes and starts the copies the
//! budget admits like any batch, and runs no forward; its runner ends the copies and returns an
//! output with no calls. A batch whose runner fails after copies started with it reports each of
//! them failed: once the device has passed what the batch issued, the batch's failure carries
//! their `Finish { ok: false }`, stamped with the batch ([`Error::Copies`]).
//!
//! The rank counts the numbered jobs it has *launched* (the stamp a request gets when the rank
//! applies it) and the ones it has *ended*: the last number whose job, and every earlier one,
//! completed on the device. It states its ended sequence to the group's [`Sequences`] before the
//! job's submission completes, so the group's `completed` word has passed a batch by the time its
//! output is read. A numbered job that failed counts as ended only once the device has passed
//! everything the runners issued; while that is unknown, nothing after it counts as ended either.
//!
//! ## Completion and waiting
//!
//! A job's [`Submission`] completes when the worker has performed the job: for a batch, when
//! the runner returned its output, which it does once the device completed the batch. The
//! worker never polls: an idle worker blocks on its queue, and what must wait for the device
//! waits on a stream event ([`Graphs::wait`](crate::runner::graphs::Graphs::wait),
//! `platform::Event::synchronize`).
//!
//! ## Failure
//!
//! A failed job posts its error with the work that was in flight
//! ([`crate::worker::Error::InFlight`]), and the jobs behind it still run. The exception is a
//! device whose context reports a sticky error, which no later launch survives: the worker checks
//! the context after a failure and from then on fails every job with that error without issuing it.
//! The control thread makes the same check on its own schedule ([`Worker::sticky_error`],
//! [`watchdog`]), because a job stuck behind such an error never returns to make it.
//!
//! ## What execution owns
//!
//! A model computes on tensors and holds no runtime object. Everything else a forward needs is
//! the worker's and its runners':
//!
//! - the execution context a forward computes under, installed on the worker's thread around
//!   each model call (`ops::Context::enter`);
//! - the tensors a forward starts from: a runner's input tensors at fixed addresses, the weights
//!   as retained storage, and the cache planes and recurrent state as external tensors;
//! - the captured graphs, their buckets and the generations of kernels they launch;
//! - the token selection that ends a causal forward.
//!
//! ## Rules every module keeps
//!
//! - One worker thread issues all work of a device, and nothing but that thread touches the
//!   device's streams.
//! - No host callback is placed on a stream or in a graph: a replay's host values reach the
//!   device as input copies issued before the graph's launch.
//! - Whatever must wait for the device waits on a stream event, never by polling.
//! - Addresses a captured graph reads stay fixed for as long as the graph exists.
//! - The ranks of a group capture the same buckets in the same order.
//!
//! ## The parts
//!
//! | Module | Holds |
//! | --- | --- |
//! | [`queue`] | [`Queue`]: the jobs in submission order, the job in flight, the close |
//! | [`sequences`] | [`Sequences`]: the ended sequences of a group's ranks and its `completed` word |
//! | [`watchdog`] | [`Watchdog`](watchdog::Watchdog): when the control thread looks at the jobs in flight, and which it reports |
//! | [`completion`] | How an operation ended, the completion its waiters share, and the submission of a job to a worker or to every rank of a group |
//! | [`error`] | The one error of device work and its classification for the engine |
//! | [`stream`] | [`EventPool`](stream::events::EventPool), the event leases storage owners defer their releases behind |
//! | [`device`] | The devices a rank computes on, as an engine's launch names them |

pub mod completion;
pub(crate) mod device;
pub(crate) mod error;
pub(crate) mod queue;
pub(crate) mod sequences;
pub(crate) mod stream;
pub(crate) mod watchdog;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use ops::Generation;
use platform::cupti::blocking::{Calls, Watched};
use platform::{DeviceId, Event};
use service::lock;
use storage::runtime::{Completion, GroupId, Request, RequestId};

use crate::batch::{Batch, BatchId, BatchOutput, Phase, Variant};

use self::completion::{Outcome, Submission, Waker, add_stamp};
pub use self::error::{Error, Result};
use self::queue::Entry;
pub(crate) use self::queue::Queue;
pub use self::sequences::Sequences;
use crate::memory::tables::{BlockCopy, BlockTables};
use crate::profiling;
use crate::profiling::session::{Change, Session};
use crate::runner::Runner;
use crate::runner::buckets::Bucket;

/// The copy budget per batch the engine gives its workers: the bytes, summed over a group's
/// ranks, that the copies started with one batch may move together, 512 MiB.
///
/// The copies run on the device's copy stream beside the batch's forward, and the batch ends with
/// them. Host storage feeds a GB200 GPU at 209 GB/s (the Transfer Server's interference
/// measurement, `docs/configuration.md` under `transfer_slowdown`), so the budget is at most
/// 2.6 ms of copying on one device, an eighth of the 20.5 ms decode step of that measurement, in
/// which copies from host storage at a quarter of the link's bandwidth slowed the steps by 2% and
/// at a sixteenth by none.
pub const COPY_BUDGET: u64 = 512 << 20;

/// The copies one batch started: the request of each, in the order they were queued, with
/// whether it was issued; and the runners, by index, that were handed copies, whose copies the
/// batch's end waits for.
type Started = (Vec<(RequestId, bool)>, Vec<usize>);

/// A control operation: what the worker calls with its runners at a batch boundary, with
/// nothing in flight on the device. The runners are in the order the worker was given them,
/// so `RunnerId(index)` names `runners[index]`.
pub(crate) type Operation = Box<dyn FnOnce(&mut [Box<dyn Runner>]) -> Result<()> + Send>;

/// One job of a worker's queue.
pub(crate) enum Job {
    /// Execute one batch through the runner that serves its phase. `sequence` is the batch's
    /// number among the group's jobs that address its device block tables; an expert join has
    /// none.
    Execute {
        batch: Box<Batch>,
        sequence: Option<u64>,
    },
    /// End `batch`, numbered `sequence`, whose host work its runner started and which ended:
    /// the runner writes what the work produced, and the batch ends with the copies that started
    /// with it.
    Resume {
        batch: Box<Batch>,
        sequence: u64,
        /// The copies that started with the batch, and the runners that issued them.
        copies: Started,
        /// The batch's wait in the queue, and when its first turn began.
        queued: (Duration, Instant),
    },
    /// Take the number of a batch another rank of the group runs alone, and run nothing.
    Pass(u64),
    /// Run the paired forwards of one shadow batch: the generation `batch.generation` names,
    /// which is being prepared, against the serving one, for each operator the batch's bucket
    /// launches whose kernel changed. One paired forward per queue turn.
    Shadow {
        batch: Box<Batch>,
        /// The shadow batch's number, as a batch's.
        sequence: u64,
        /// The changed operators already compared: zero when the job is submitted.
        compared: usize,
    },
    /// Map `completion`, the buffer pool's answer to `request` numbered `id`, into the rank's
    /// device block tables under `slot` ([`BlockTables::map`]).
    Map {
        slot: u32,
        id: RequestId,
        request: Request,
        completion: Completion,
    },
    /// Apply requests to the rank's device block tables ([`BlockTables::apply`]) and return them
    /// stamped with the number of the last job the rank launched.
    Stamp(Vec<Request>),
    /// Capture every bucket the runners state and have no graph of, largest first: the warm-up.
    /// One bucket per queue turn.
    Capture,
    /// Prepare a generation beside the serving one on every runner: the graphs it changes are
    /// rebound or recaptured, one bucket per queue turn.
    Prepare(Arc<Generation>),
    /// Switch every runner to the prepared generation of this number, which then serves on
    /// trial.
    Switch(u64),
    /// Commit the trial of the generation of this number: the generation it replaced is
    /// released.
    Commit(u64),
    /// Roll the generation of this number back, on trial or prepared and never switched to: its
    /// graphs are released, and after a trial the generation it replaced serves again.
    Rollback(u64),
    /// Change the device's profiling session; the rank's response to the daemon's operation is
    /// sent to `response`.
    Profile {
        change: Change,
        response: Sender<serde_json::Value>,
    },
    /// Wait until the device has passed the latest record of each event.
    #[cfg_attr(not(test), expect(dead_code, reason = "only tests call it"))]
    Wait(Vec<Arc<Event>>),
    /// Call `operation` with the runners, once the device has passed everything they issued.
    Control {
        /// What the operation does, for a stall report and a failure.
        label: String,
        operation: Operation,
    },
}

impl Job {
    /// The batch the job executes, ends or shadows.
    fn batch(&self) -> Option<&Batch> {
        match self {
            Self::Execute { batch, .. }
            | Self::Resume { batch, .. }
            | Self::Shadow { batch, .. } => Some(batch),
            _ => None,
        }
    }

    /// Whether the job runs a runner's forwards, whose time the worker counts: a batch with
    /// no rows runs none, and a host batch's work is the host's.
    fn runs_forwards(&self) -> bool {
        match self {
            Self::Execute { batch, .. } => forwards(batch),
            Self::Shadow { .. } | Self::Capture | Self::Prepare(_) => true,
            _ => false,
        }
    }

    /// The number of the batch whose remainder the job is, which waits for its host work.
    fn waits(&self) -> Option<u64> {
        match self {
            Self::Resume { sequence, .. } => Some(*sequence),
            _ => None,
        }
    }
}

/// What the job is, as a stall report and a failure name the work in flight.
impl std::fmt::Display for Job {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execute { batch, .. } => write!(
                formatter,
                "executing batch {} of phase {:?} ({})",
                batch.id.0, batch.phase, batch.shape
            ),
            Self::Resume { batch, .. } => write!(
                formatter,
                "ending batch {} of phase {:?} ({}) after its host work",
                batch.id.0, batch.phase, batch.shape
            ),
            Self::Pass(sequence) => write!(
                formatter,
                "passing job {sequence}, which another rank runs alone"
            ),
            Self::Shadow { batch, .. } => write!(
                formatter,
                "shadowing generation {} on batch {} of phase {:?} ({})",
                batch.generation, batch.id.0, batch.phase, batch.shape
            ),
            Self::Map { id, .. } => write!(
                formatter,
                "mapping the completion of request {} into the block tables",
                id.0
            ),
            Self::Stamp(requests) => write!(
                formatter,
                "applying and stamping {} requests",
                requests.len()
            ),
            Self::Capture => formatter.write_str("warming up: capturing the runners' buckets"),
            Self::Prepare(next) => write!(formatter, "preparing generation {}", next.number()),
            Self::Switch(number) => write!(formatter, "switching to generation {number}"),
            Self::Commit(number) => write!(formatter, "committing generation {number}"),
            Self::Rollback(number) => write!(formatter, "rolling back generation {number}"),
            Self::Profile { .. } => formatter.write_str("changing the profiling session"),
            Self::Wait(events) => write!(
                formatter,
                "waiting for {} events of deferred storage release",
                events.len()
            ),
            Self::Control { label, .. } => formatter.write_str(label),
        }
    }
}

/// The rank a batch runs on alone, when its media rows name one. `None` for a batch every rank of
/// the group runs, a video decode round and its encode among them.
///
/// # Errors
///
/// Rows that name different ranks, or some a rank and some none.
pub(crate) fn alone(batch: &Batch) -> Result<Option<u32>> {
    let Some(first) = batch.media.first() else {
        return Ok(None);
    };

    match batch.media.iter().find(|row| row.rank != first.rank) {
        Some(other) => Err(Error::Invalid(format!(
            "batch {} names rank {:?} in one row and rank {:?} in another",
            batch.id.0, first.rank, other.rank
        ))),
        None => Ok(first.rank),
    }
}

/// Whether `batch` has no rows: it advances no request, and it is no expert join, whose forward
/// serves a peer's step. A scheduler sends one to a group whose worker holds copies not yet
/// finished when a step formed no other batch for the group (`specs/scheduler-design.md` §2.1):
/// it takes its number, hands the runners the tables' changes and starts the copies the copy
/// budget admits as any batch does, and runs no forward.
fn empty(batch: &Batch) -> bool {
    batch.requests.is_empty() && batch.shape.variant != Variant::ExpertJoin
}

/// Whether `batch` runs a forward on the device: it has rows, and its work is not the host's. Only
/// such a batch is a step of a profiling session and marks one.
fn forwards(batch: &Batch) -> bool {
    !empty(batch) && !batch.shape.variant.host()
}

/// How many of the oldest queued copies, which move `sizes` bytes each in the order they were
/// queued, one batch starts under `budget`: as many as fit in it together, and at least one, so
/// a copy larger than the budget starts alone. The first copy that does not fit waits with every
/// copy behind it, smaller ones included, so copies start in the order they were queued.
fn admitted(sizes: impl IntoIterator<Item = u64>, budget: u64) -> usize {
    let mut moved: u64 = 0;
    let mut count = 0;

    for bytes in sizes {
        let total = moved.saturating_add(bytes);
        if count > 0 && total > budget {
            break;
        }

        moved = total;
        count += 1;
    }
    count
}

/// What a worker has done since it started; monotonic sums the control thread reads while
/// the worker adds to them.
#[derive(Default)]
struct Counters {
    /// Batches executed.
    steps: AtomicU64,
    /// Nanoseconds the worker spent in jobs that run a runner's forwards.
    forward_ns: AtomicU64,
    /// Kernels the runners' serving forwards launched ([`Graphs::launches`]), summed over them.
    ///
    /// [`Graphs::launches`]: crate::runner::graphs::Graphs::launches
    launches: AtomicU64,
}

/// The worker of one device, as every other thread holds it.
pub struct Worker {
    device: Option<DeviceId>,
    /// The worker group the device is a rank of, and the rank's place among those that report to
    /// the group's [`Sequences`].
    group: GroupId,
    rank: usize,
    queue: Arc<Queue>,
    counters: Arc<Counters>,
    /// What the generation update in flight has no comparison for; see [`Worker::uncompared`].
    uncompared: Arc<Mutex<Vec<String>>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    /// Starts the worker of `device` over `runners`, whose graphs serve `kernels`, as rank
    /// `rank` of the worker group whose device block tables `tables` holds for this rank, and
    /// which states its ended sequence to `sequences`.
    ///
    /// `copy_budget` is the bytes, summed over the group's ranks, that the copies started with
    /// one batch may move together ([`COPY_BUDGET`]). Every worker of a group is given the
    /// same budget and tables given the same unit sizes, or the ranks would start different
    /// copies with one batch and the group's submission would fail.
    ///
    /// `device` is the CUDA device the runners issue on: the worker keeps its context current on
    /// the thread and checks it for a sticky error. `None` is a worker of runners that hold no
    /// device (host work, or graphs made by [`Graphs::meta`](crate::runner::graphs::Graphs::meta)),
    /// which records no profiling session.
    ///
    /// From this call on the runners and the tables are the worker thread's alone.
    ///
    /// # Errors
    ///
    /// Tables and sequences of different groups, a rank that does not report to the sequences,
    /// or a thread that cannot start.
    pub fn spawn(
        device: Option<DeviceId>,
        runners: Vec<Box<dyn Runner>>,
        kernels: Arc<Generation>,
        tables: BlockTables,
        sequences: Arc<Sequences>,
        rank: usize,
        copy_budget: u64,
    ) -> Result<Self> {
        let group = tables.group();
        if sequences.group() != group || rank >= sequences.ranks() {
            return Err(Error::Invalid(format!(
                "a worker of group {} as rank {rank} reports to the sequences of group {}, \
                 which has {} ranks",
                group.0,
                sequences.group().0,
                sequences.ranks()
            )));
        }

        let phases: Vec<Phase> = runners
            .iter()
            .flat_map(|runner| runner.phases().iter().copied())
            .collect();
        let queue = Queue::new(phases);
        let counters = Arc::new(Counters::default());
        let uncompared = Arc::new(Mutex::new(Vec::new()));

        let state = Thread {
            device,
            runners,
            queue: Arc::clone(&queue),
            kernels: BTreeMap::from([(kernels.number(), kernels)]),
            counters: Arc::clone(&counters),
            uncompared: Arc::clone(&uncompared),
            unmeasured: Vec::new(),
            sticky: None,
            session: None,
            blocking: Calls::default(),
            renewals: profiling::graph_renewals(),
            tables,
            copy_budget,
            sequences,
            rank,
            launched: 0,
            open: BTreeSet::new(),
        };
        let name = match device {
            Some(device) => format!("worker-{}", device.0),
            None => "worker-host".to_owned(),
        };
        let thread = std::thread::Builder::new()
            .name(name)
            .spawn(move || state.run())
            .map_err(|error| Error::transport_source("starting a worker thread", error))?;

        Ok(Self {
            device,
            group,
            rank,
            queue,
            counters,
            uncompared,
            thread: Some(thread),
        })
    }

    /// The CUDA device this worker issues on, if it has one.
    pub(crate) fn device(&self) -> Option<DeviceId> {
        self.device
    }

    /// The worker group the device is a rank of.
    pub(crate) fn group(&self) -> GroupId {
        self.group
    }

    /// The rank's place among those that report to the group's [`Sequences`].
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    /// The queue the worker takes its jobs from: what the control thread's watchdog reads, and
    /// what closes it.
    pub(crate) fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// The phases the runners serve, in the runners' order.
    pub(crate) fn phases(&self) -> &[Phase] {
        self.queue.phases()
    }

    /// Whether a runner of this worker serves `phase`.
    pub(crate) fn serves(&self, phase: Phase) -> bool {
        self.queue.phases().contains(&phase)
    }

    /// Submits `batch`, the group's job number `sequence`, for execution; the submission completes
    /// with the step's output and the stamped `Finish` of each copy that ran with it. When the
    /// batch fails after copies started with it, the failure carries their `Finish { ok: false }`
    /// ([`Error::finished`]). The group's jobs reach every rank in the order of their numbers
    /// ([`Route`]).
    ///
    /// # Errors
    ///
    /// As [`Queue::submit`]: a batch no runner serves is refused here.
    pub(crate) fn submit(&self, sequence: u64, batch: Batch) -> Result<Submission> {
        self.queue.submit(Job::Execute {
            batch: Box::new(batch),
            sequence: Some(sequence),
        })
    }

    /// Submits the group's job number `sequence`, a batch another rank runs alone: this rank
    /// takes the number and runs nothing.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub(crate) fn pass(&self, sequence: u64) -> Result<Submission> {
        self.queue.submit(Job::Pass(sequence))
    }

    /// Submits `batch`, a member's part in an expert union's step it has no tokens for (variant
    /// `ExpertJoin`), which another member asks for: it addresses no device block table, so it
    /// takes no number and starts no copy.
    ///
    /// # Errors
    ///
    /// A batch of another variant, or as [`Queue::submit`].
    pub(crate) fn join(&self, batch: Batch) -> Result<Submission> {
        if batch.shape.variant != Variant::ExpertJoin {
            return Err(Error::Invalid(format!(
                "batch {} of variant {} is no expert join: a batch that addresses the device \
                 block tables is numbered by its group",
                batch.id.0, batch.shape.variant
            )));
        }

        self.queue.submit(Job::Execute {
            batch: Box::new(batch),
            sequence: None,
        })
    }

    /// Submits the paired forwards of the shadow batch `batch`, the group's job number `sequence`
    /// ([`Job::Shadow`]).
    ///
    /// # Errors
    ///
    /// As [`Queue::submit`].
    pub(crate) fn shadow(&self, sequence: u64, batch: Batch) -> Result<Submission> {
        self.queue.submit(Job::Shadow {
            batch: Box::new(batch),
            sequence,
            compared: 0,
        })
    }

    /// Submits the map of `completion`, the buffer pool's answer to `request` numbered `id`, into
    /// the rank's device block tables under `slot` ([`BlockTables::map`]): every job submitted
    /// after it sees what it gives. The submission fails when the tables refuse the pair.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub(crate) fn map(
        &self,
        slot: u32,
        id: RequestId,
        request: Request,
        completion: Completion,
    ) -> Result<Submission> {
        self.queue.submit(Job::Map {
            slot,
            id,
            request,
            completion,
        })
    }

    /// Submits the application of `requests` to the rank's device block tables
    /// ([`BlockTables::apply`]); the submission completes with them, each with the group's
    /// stamp as this rank states it: the number of the last job it launched before it applied
    /// them.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub(crate) fn stamp(&self, requests: Vec<Request>) -> Result<Submission> {
        self.queue.submit(Job::Stamp(requests))
    }

    /// Submits the capture of every bucket the runners state ([`Job::Capture`]).
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub(crate) fn warm_up(&self) -> Result<Submission> {
        self.queue.submit(Job::Capture)
    }

    /// Submits the preparation of generation `next` ([`Job::Prepare`]). A preparation that
    /// fails leaves a generation to roll back ([`Worker::rollback`]).
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub(crate) fn prepare(&self, next: Arc<Generation>) -> Result<Submission> {
        self.queue.submit(Job::Prepare(next))
    }

    /// Submits the switch to the prepared generation `number`, which takes effect between the
    /// batches submitted before and after it.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub(crate) fn switch(&self, number: u64) -> Result<Submission> {
        self.queue.submit(Job::Switch(number))
    }

    /// Submits the commit of generation `number`, which serves on trial.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub(crate) fn commit(&self, number: u64) -> Result<Submission> {
        self.queue.submit(Job::Commit(number))
    }

    /// Submits the rollback of generation `number`, on trial or prepared. Rolling back a
    /// generation this worker holds nothing of succeeds and changes nothing.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub(crate) fn rollback(&self, number: u64) -> Result<Submission> {
        self.queue.submit(Job::Rollback(number))
    }

    /// Changes the device's profiling session between two batches and returns the rank's
    /// response to the daemon's operation. It waits for the worker, so it is a work thread's
    /// call and never the control thread's.
    ///
    /// # Errors
    ///
    /// When the session refuses the change, the worker has no device, or the queue is closed.
    pub(crate) fn profile(&self, change: Change) -> Result<serde_json::Value> {
        let (response, responded) = crossbeam_channel::bounded(1);
        let submission = self.queue.submit(Job::Profile { change, response })?;

        returned(&submission, &responded)
    }

    /// Submits a wait for `events`: the submission completes once the device has passed the
    /// latest record of each, which is when storage whose release hangs on them may be released.
    /// The events are recorded by jobs submitted before this one.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    #[cfg_attr(not(test), expect(dead_code, reason = "only tests call it"))]
    pub(crate) fn wait(&self, events: Vec<Arc<Event>>) -> Result<Submission> {
        self.queue.submit(Job::Wait(events))
    }

    /// Submits the control operation `operation`, named `label` in a stall report and a failure:
    /// the worker calls it between the jobs submitted before and after it, once the device
    /// has passed everything the runners issued.
    ///
    /// # Errors
    ///
    /// Once the queue is closed.
    pub(crate) fn control(
        &self,
        label: impl Into<String>,
        operation: Operation,
    ) -> Result<Submission> {
        self.queue.submit(Job::Control {
            label: label.into(),
            operation,
        })
    }

    /// Runs `operation` as a control operation ([`Worker::control`]), waits for it and returns
    /// its value. A work thread's call, never the control thread's.
    ///
    /// # Errors
    ///
    /// What the operation fails with, stated with its label, or the close of the queue.
    pub(crate) fn call<T: Send + 'static>(
        &self,
        label: impl Into<String>,
        operation: impl FnOnce(&mut [Box<dyn Runner>]) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (value, produced) = crossbeam_channel::bounded(1);
        let submission = self.control(
            label,
            Box::new(move |runners| {
                // A caller that stopped waiting has nothing to be told.
                let _ = value.send(operation(runners)?);
                Ok(())
            }),
        )?;

        returned(&submission, &produced)
    }

    /// When the worker took the job in flight and the batch the job belongs to; `None` between
    /// two jobs. The control thread's watchdog reads it ([`watchdog::Watchdog`]).
    pub(crate) fn in_flight(&self) -> Option<(Instant, Option<BatchId>)> {
        self.queue.in_flight()
    }

    /// What the worker is performing, in words: the job, and the bucket, generation or operator
    /// it is at. `None` between two jobs.
    pub(crate) fn performing(&self) -> Option<String> {
        self.queue.performing()
    }

    /// The sticky error of the device's context, if it has one: an illegal memory access or a
    /// device-side assert left the context unusable, and the engine must be replaced. The
    /// control thread checks while a job has been in flight for a second, because the jobs
    /// behind such an error never complete.
    ///
    /// # Errors
    ///
    /// When the context cannot be read.
    pub(crate) fn sticky_error(&self) -> Result<Option<platform::Error>> {
        match self.device {
            Some(device) => Ok(platform::PrimaryContext::retain(device)?.sticky_error()?),
            None => Ok(None),
        }
    }

    /// The batches executed since the worker started; the nanoseconds it spent in jobs that run a
    /// runner's forwards: batches, captures, preparations and paired forwards; and the kernels its
    /// runners' serving forwards launched, each replay's captured kernel nodes and each eager
    /// forward's launches ([`Graphs::launches`](crate::runner::graphs::Graphs::launches)). A
    /// runner's `execute` returns once the device completed the step, so the time is the device's
    /// as well as the host's.
    pub(crate) fn counters(&self) -> (u64, u64, u64) {
        (
            self.counters.steps.load(Ordering::Relaxed),
            self.counters.forward_ns.load(Ordering::Relaxed),
            self.counters.launches.load(Ordering::Relaxed),
        )
    }

    /// What the generation update in flight has no comparison for, each with the reason: the
    /// kernels the generation adds, which nothing serving computes, and the pairs its paired
    /// forwards could not compare. Empty states that every kernel the update replaced and a
    /// shadow batch reached was compared. It is cleared when the next preparation begins.
    pub(crate) fn uncompared(&self) -> Vec<String> {
        lock(&self.uncompared).clone()
    }

    /// Closes the queue and joins the thread: the job in flight ends, and the jobs still queued
    /// end as cancelled.
    ///
    /// # Errors
    ///
    /// When the worker thread panicked.
    pub(crate) fn close(mut self) -> Result<()> {
        self.join_thread()
    }

    fn join_thread(&mut self) -> Result<()> {
        self.queue.close();

        match self.thread.take() {
            Some(thread) => thread
                .join()
                .map_err(|_| Error::State("the worker thread panicked")),
            None => Ok(()),
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if self.join_thread().is_err() {
            tracing::error!(device = ?self.device, "a worker thread panicked");
        }
    }
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Worker")
            .field("device", &self.device)
            .field("group", &self.group)
            .field("rank", &self.rank)
            .field("queue", &self.queue)
            .finish_non_exhaustive()
    }
}

/// The value a job sent through `receiver`, once `submission` completed.
fn returned<T>(submission: &Submission, receiver: &Receiver<T>) -> Result<T> {
    match submission.wait() {
        Outcome::Success(_) => receiver
            .recv()
            .map_err(|_| Error::State("the worker's job completed without its value")),
        Outcome::Failed(fault) => Err(Error::Shared(fault)),
        Outcome::Cancelled => Err(Error::State("the worker closed before it ran the job")),
    }
}

/// The worker thread's state: the runners and the tables nothing else touches, and what the
/// thread keeps between two jobs.
struct Thread {
    device: Option<DeviceId>,
    /// The device's runners; `RunnerId(index)` names `runners[index]`.
    runners: Vec<Box<dyn Runner>>,
    queue: Arc<Queue>,
    /// The generations the runners' graphs hold, by number: the serving one, and the one being
    /// prepared or kept during a trial. Which of them serves is the graphs' to say
    /// ([`Graphs::generation`](crate::runner::graphs::Graphs::generation)); a shadow batch reads
    /// the two it compares here.
    kernels: BTreeMap<u64, Arc<Generation>>,
    counters: Arc<Counters>,
    uncompared: Arc<Mutex<Vec<String>>>,
    /// The kernels the generation being prepared adds, which nothing serving is compared with.
    unmeasured: Vec<String>,
    /// The context's sticky error, once one was seen: every launch after it fails, so the
    /// worker issues nothing more and fails each job with this error.
    sticky: Option<platform::Error>,
    /// The profiling session that records the device, while one does.
    session: Option<Session>,
    /// This thread's driver calls that waited for the device, when its last batch output left.
    blocking: Calls,
    /// The process's count of graph renewals the runners' executables were instantiated at.
    renewals: u64,
    /// The rank's device block tables.
    tables: BlockTables,
    /// The bytes, summed over the group's ranks, that the copies started with one batch may
    /// move together.
    copy_budget: u64,
    /// Where the rank states its ended sequence, as rank `rank`.
    sequences: Arc<Sequences>,
    rank: usize,
    /// The number of the last numbered job the rank launched: the stamp of a request it applies.
    launched: u64,
    /// The numbered jobs the rank launched that have not ended: a job of several turns ends with
    /// its last, after later jobs; a failed job ends once the device has passed it.
    open: BTreeSet<u64>,
}

impl Thread {
    /// The kernels the runners' serving forwards launched so far, summed over the runners.
    fn launches(&self) -> u64 {
        self.runners
            .iter()
            .map(|runner| runner.graphs().launches())
            .sum()
    }

    /// Performs the queue's jobs until it closes.
    fn run(mut self) {
        // The device's context stays current on this thread: a native module reads the device
        // from the runtime to find its per-device handles, and a thread with no current context
        // reads device zero.
        let context = self.device.map(platform::PrimaryContext::retain);
        let current = match &context {
            Some(Ok(context)) => Some(context.enter()),
            _ => None,
        };
        if let Some(Err(error)) = &context {
            tracing::error!(device = ?self.device, %error, "the device's context is not current");
        }
        if let Some(Err(error)) = &current {
            tracing::error!(device = ?self.device, %error, "the device's context is not current");
        }

        // The driver calls this thread waits in are counted while it serves: waiting for its own
        // device's work is its design, so the figure is a record and not a fault.
        let _watched = self.device.map(|_| Watched::enter());

        while let Some(Entry {
            job,
            submission,
            submitted,
        }) = self.queue.next()
        {
            if let Some(sticky) = &self.sticky {
                // Nothing is issued to an unusable context, and the job fails with the sticky
                // error instead of a launch error that would hide it.
                self.queue
                    .complete(submission, Err(Error::Cuda(sticky.clone())));
                continue;
            }

            let counted = job.runs_forwards();
            let launched = if counted { self.launches() } else { 0 };
            let started = Instant::now();
            let turn = self.turn(job, started.saturating_duration_since(submitted));
            if counted {
                let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.counters
                    .forward_ns
                    .fetch_add(elapsed, Ordering::Relaxed);
                self.counters
                    .launches
                    .fetch_add(self.launches().saturating_sub(launched), Ordering::Relaxed);
            }

            match turn {
                Ok(ControlFlow::Continue(job)) => {
                    let waits = job.waits();
                    let entry = Entry {
                        job,
                        submission,
                        submitted,
                    };

                    match waits {
                        Some(sequence) => self.queue.park(sequence, entry),
                        None => self.queue.requeue(entry),
                    }
                }
                Ok(ControlFlow::Break(returned)) => self.queue.complete(submission, Ok(returned)),
                Err(error) => {
                    tracing::warn!(
                        device = ?self.device,
                        %error,
                        in_flight = self.queue.performing().as_deref().unwrap_or_default(),
                        "a worker's job failed"
                    );
                    self.queue.complete(submission, Err(error));
                }
            }
        }
    }

    /// Performs one turn of `job`, which waited `queued` in the queue, and keeps the rank's
    /// sequences: a numbered job is launched with its first turn and ends with its last, or with
    /// its failure once the device has passed what the runners issued. Returns the job's value
    /// when it ended, or the job's remainder, which goes back to the tail of the queue.
    ///
    /// The ended sequence is stated before the job's submission completes, so the group's
    /// `completed` covers a batch before its output is read.
    fn turn(
        &mut self,
        job: Job,
        queued: Duration,
    ) -> Result<ControlFlow<(BatchOutput, Vec<Request>), Job>> {
        // The number of a numbered job, and whether this turn is its first.
        let (sequence, first) = match &job {
            Job::Execute { sequence, .. } => (*sequence, true),
            Job::Resume { sequence, .. } => (Some(*sequence), false),
            Job::Pass(sequence) => (Some(*sequence), true),
            Job::Shadow {
                sequence, compared, ..
            } => (Some(*sequence), *compared == 0),
            _ => (None, false),
        };
        if let Some(sequence) = sequence
            && first
        {
            self.launch(sequence);
        }

        let turn = self.perform(job, queued);
        if turn.is_err() {
            // A failure may be the first symptom of a sticky error.
            self.check_sticky();
        }

        if let Some(sequence) = sequence
            && !matches!(turn, Ok(ControlFlow::Continue(_)))
        {
            self.end(sequence, turn.is_ok());
        }
        turn
    }

    /// The rank launched the group's job number `sequence`.
    fn launch(&mut self, sequence: u64) {
        self.launched = self.launched.max(sequence);
        self.open.insert(sequence);
    }

    /// The group's job number `sequence` ended, `ran` when it succeeded: the rank states the
    /// last number it has ended with every earlier one ended. A failed job ends only once the
    /// device has passed everything the runners issued; while that is unknown it stays open, and
    /// nothing after it counts as ended.
    fn end(&mut self, sequence: u64, ran: bool) {
        if !ran && let Err(error) = self.settle() {
            tracing::error!(
                device = ?self.device,
                sequence,
                %error,
                "a failed job's device work cannot be awaited; the group's completed word stays \
                 below it"
            );
            return;
        }

        self.open.remove(&sequence);
        // Numbers start at 1: the one before the first open job is at least 0, which names none.
        let ended = match self.open.first() {
            Some(first) => first.saturating_sub(1),
            None => self.launched,
        };
        self.sequences.ended(self.rank, ended);
    }

    /// Performs one turn of `job`, which waited `queued` in the queue: the job's value when it
    /// ended, or the job's remainder, which goes back to the tail of the queue.
    fn perform(
        &mut self,
        job: Job,
        queued: Duration,
    ) -> Result<ControlFlow<(BatchOutput, Vec<Request>), Job>> {
        let done = ControlFlow::Break((BatchOutput::default(), Vec::new()));

        match job {
            Job::Execute { batch, sequence } => self.execute(batch, sequence, queued),
            Job::Resume {
                batch,
                copies,
                queued,
                ..
            } => {
                let index = self.runner_of(batch.phase)?;
                let output = self.runners[index].execute(&batch, &self.tables);
                Ok(ControlFlow::Break(
                    self.ended(&batch, output, copies, queued)?,
                ))
            }
            Job::Pass(_) => Ok(done),
            Job::Shadow {
                batch,
                sequence,
                compared,
            } => Ok(if self.shadow(&batch, compared)? {
                ControlFlow::Continue(Job::Shadow {
                    batch,
                    sequence,
                    compared: compared + 1,
                })
            } else {
                done
            }),
            Job::Map {
                slot,
                id,
                request,
                completion,
            } => {
                self.tables.map(slot, id, &request, &completion)?;
                Ok(done)
            }
            Job::Stamp(requests) => Ok(ControlFlow::Break((
                BatchOutput::default(),
                self.stamp(requests)?,
            ))),
            Job::Capture => Ok(if self.capture()? {
                ControlFlow::Continue(Job::Capture)
            } else {
                done
            }),
            Job::Prepare(next) => Ok(if self.prepare(&next)? {
                ControlFlow::Continue(Job::Prepare(next))
            } else {
                done
            }),
            Job::Switch(number) => {
                self.switch(number)?;
                Ok(done)
            }
            Job::Commit(number) => {
                self.commit(number)?;
                Ok(done)
            }
            Job::Rollback(number) => {
                self.rollback(number)?;
                Ok(done)
            }
            Job::Profile { change, response } => {
                let device = self.device.ok_or(Error::State(
                    "a worker with no device records no profiling session",
                ))?;

                // A session's collectors start and stop with nothing of the device in flight.
                self.settle()?;
                let changed = Session::change(&mut self.session, change, device)?;

                // A caller that stopped waiting has nothing to be told.
                let _ = response.send(changed);
                Ok(done)
            }
            Job::Wait(events) => {
                for event in &events {
                    event.synchronize()?;
                }
                Ok(done)
            }
            Job::Control { operation, .. } => {
                self.settle()?;
                operation(&mut self.runners)?;
                Ok(done)
            }
        }
    }

    /// Applies `requests` to the tables and stamps each with the number of the last job the rank
    /// launched, as this group's stamp.
    fn stamp(&mut self, mut requests: Vec<Request>) -> Result<Vec<Request>> {
        let group = self.tables.group();

        for request in &mut requests {
            self.tables.apply(request)?;
            add_stamp(request, group, self.launched);
        }
        Ok(requests)
    }

    /// Executes `batch`, numbered `sequence` unless it is an expert join, through the runner that
    /// serves its phase: the tables' changes go to every runner first, and the queued copies the
    /// copy budget admits start with a numbered batch every rank of the group runs. Returns the
    /// output and the stamped `Finish` of each copy that ran, or, when the runner started the
    /// batch's host work ([`Runner::start`]), the batch's remainder, which waits for that work.
    /// A batch with no rows and a host batch are no step of a profiling session and mark none:
    /// their runner runs no forward.
    ///
    /// # Errors
    ///
    /// What fails the batch. A runner's failure after copies started with the batch carries
    /// their `Finish { ok: false }` ([`Thread::failed`]).
    fn execute(
        &mut self,
        batch: Box<Batch>,
        sequence: Option<u64>,
        queued: Duration,
    ) -> Result<ControlFlow<(BatchOutput, Vec<Request>), Job>> {
        let index = self.runner_of(batch.phase)?;
        let step = batch.id.0;
        let queued = (queued, Instant::now());

        // A session's window is made of whole steps: its collectors are armed and disarmed
        // between two steps, with nothing of the device in flight.
        let stage = batch.shape.variant.stage();
        if forwards(&batch)
            && self
                .session
                .as_mut()
                .is_some_and(|session| session.begin(step, stage))
        {
            self.apply_session()?;
        }
        self.renew()?;

        self.write_tables()?;
        let copies = if sequence.is_some() && alone(&batch)?.is_none() {
            self.start_copies(index)
        } else {
            (Vec::new(), Vec::new())
        };

        // A numbered batch whose work its runner runs on the host leaves the worker while the
        // work runs, its number open; the work's end puts its remainder back in the queue.
        if let Some(sequence) = sequence
            && !empty(&batch)
        {
            let queue = Arc::clone(&self.queue);
            let resume: Waker = Arc::new(move || queue.resume(sequence));

            match self.runners[index].start(&batch, &self.tables, resume) {
                Ok(false) => {}
                Ok(true) => {
                    return Ok(ControlFlow::Continue(Job::Resume {
                        batch,
                        sequence,
                        copies,
                        queued,
                    }));
                }
                Err(error) => return Err(self.failed(error, copies.0)),
            }
        }

        // The step's range markers are launched by the graphs, around the forward itself.
        if forwards(&batch) {
            self.runners[index].graphs().mark_step(step, stage);
        }
        let output = self.runners[index].execute(&batch, &self.tables);
        Ok(ControlFlow::Break(
            self.ended(&batch, output, copies, queued)?,
        ))
    }

    /// Ends `batch`, whose runner returned `output`, with `copies`, the copies that started with
    /// it and the runners that issued them; the batch waited `queued.0` in the queue, and its
    /// first turn began at `queued.1`. Returns the output and the stamped `Finish` of each copy.
    ///
    /// # Errors
    ///
    /// The runner's failure, or a wait for the copies that failed, carrying the copies'
    /// `Finish { ok: false }` ([`Thread::failed`]).
    fn ended(
        &mut self,
        batch: &Batch,
        output: Result<BatchOutput>,
        (copies, issuers): Started,
        (queued, started): (Duration, Instant),
    ) -> Result<(BatchOutput, Vec<Request>)> {
        let step = batch.id.0;

        // The batch ends with its copies, whichever runners issued them.
        let output = output.and_then(|output| {
            for &index in &issuers {
                self.runners[index].wait_copies()?;
            }
            Ok(output)
        });
        let mut output = match output {
            Ok(output) => output,
            Err(error) => return Err(self.failed(error, copies)),
        };

        if forwards(batch)
            && self
                .session
                .as_mut()
                .is_some_and(|session| session.end(step))
        {
            self.apply_session()?;
        }

        // The batch ended, and the copies with it.
        let finished = self.finish(copies)?;

        // The runner states the step's device and host time; the wait in the queue is the
        // worker's to state.
        output.timing.queued_us = u64::try_from(queued.as_micros()).unwrap_or(u64::MAX);
        if let Some(device) = self.device {
            output.forward_stats =
                profiling::forward_stats(device, self.session.as_mut(), &mut self.blocking);
        }
        self.counters.steps.fetch_add(1, Ordering::Relaxed);

        // A media call is seconds of work a few times per request, so one line each is the
        // request's timeline; a token batch's would be the log.
        if !batch.media.is_empty() {
            tracing::info!(
                device = ?self.device,
                phase = ?batch.phase,
                batch = step,
                queued_ms = queued.as_millis() as u64,
                total_ms = started.elapsed().as_millis() as u64,
                "a media batch was executed"
            );
        }
        Ok((output, finished))
    }

    /// The `Finish` of each of `copies`, which started with the batch that just ended, each `ok`
    /// as stated: it was issued, and the batch was served. Each is applied to the tables, as
    /// every `Finish` is, and stamped with the last batch the rank launched, which covers it.
    fn finish(&mut self, copies: Vec<(RequestId, bool)>) -> Result<Vec<Request>> {
        let group = self.tables.group();
        let mut finished = Vec::with_capacity(copies.len());

        for (copy, ok) in copies {
            let mut finish = Request::Finish {
                copy,
                ok,
                after: Vec::new(),
            };
            self.tables.apply(&finish)?;
            add_stamp(&mut finish, group, self.launched);
            finished.push(finish);
        }
        Ok(finished)
    }

    /// The failure of a batch whose runner failed with `error` after `copies` started with it.
    ///
    /// Once the device has passed everything the runners issued, their copies included
    /// ([`Thread::settle`]), each copy is reported failed: its result is unknown, and the
    /// scheduler learns it with the batch instead of at its bound. The failure then carries their `Finish { ok: false }`, stamped
    /// with this batch ([`Error::Copies`]). While the device's progress is unknown it carries
    /// none: the batch stays open, the group's `completed` stays below it, and whatever later
    /// gives the copies up is covered by a stamp no lower than the batch's.
    fn failed(&mut self, error: Error, copies: Vec<(RequestId, bool)>) -> Error {
        if copies.is_empty() {
            return error;
        }
        if let Err(unsettled) = self.settle() {
            tracing::error!(
                device = ?self.device,
                %unsettled,
                copies = copies.len(),
                "a failed batch's device work cannot be awaited; the copies that started with it \
                 are not reported"
            );
            return error;
        }

        // Applying a `Finish` that gives a copy up cannot be refused.
        let given_up = copies.into_iter().map(|(copy, _)| (copy, false)).collect();
        match self.finish(given_up) {
            Ok(finished) => Error::Copies {
                error: Box::new(error),
                finished,
            },
            Err(_) => error,
        }
    }

    /// Hands every runner the tables' changes since the last batch, each change once.
    fn write_tables(&mut self) -> Result<()> {
        let changes = self.tables.take_changes();
        if changes.is_empty() {
            return Ok(());
        }

        for runner in &mut self.runners {
            runner.write_tables(&changes)?;
        }
        Ok(())
    }

    /// Starts, for the batch runner `index` runs next, the oldest copies no batch has started
    /// that the copy budget admits, each on the runner whose memory holds it
    /// ([`Thread::copier`]). Returns the copies with whether each was issued, and the runners
    /// that were handed some: a copy its runner cannot issue, or that no runner maps, is
    /// reported failed, and the batch still runs.
    fn start_copies(&mut self, index: usize) -> Started {
        // The queue's order and each copy's bytes over the group are the same on every rank, so
        // every rank starts the same copies with this batch.
        let admitted = admitted(
            self.tables.queued_copies().map(|(_, bytes)| bytes),
            self.copy_budget,
        );
        let copies: Vec<BlockCopy> = std::iter::from_fn(|| self.tables.start_copy())
            .take(admitted)
            .collect();

        // Each copy's request, its runner, and whether a runner can run it. A copy without runs
        // lies in other ranks' memory and asks nothing of this rank.
        let mut routes = Vec::with_capacity(copies.len());
        let mut handed: BTreeMap<usize, Vec<BlockCopy>> = BTreeMap::new();
        for copy in copies {
            let request = copy.request;
            if copy.runs.is_empty() {
                routes.push((request, None, true));
                continue;
            }

            let regions: BTreeSet<u8> = copy
                .runs
                .iter()
                .flat_map(|(from, to)| [from.0, to.0])
                .collect();
            match self.copier(index, &regions) {
                Some(runner) => {
                    routes.push((request, Some(runner), true));
                    handed.entry(runner).or_default().push(copy);
                }
                None => {
                    tracing::error!(
                        device = ?self.device,
                        request = request.0,
                        ?regions,
                        "no runner of the rank maps both runs of a copy's pairs; the copy is \
                         reported failed"
                    );
                    routes.push((request, None, false));
                }
            }
        }

        // Each runner issues its copies in one call, in the order they were queued.
        let mut issued = BTreeMap::new();
        for (&runner, copies) in &handed {
            let result = self.runners[runner].copy(copies);
            if let Err(error) = &result {
                tracing::warn!(
                    device = ?self.device,
                    runner,
                    %error,
                    copies = copies.len(),
                    "the copies a runner was handed were not issued and are reported failed"
                );
            }
            issued.insert(runner, result.is_ok());
        }

        let started = routes
            .into_iter()
            .map(|(request, runner, runs)| {
                let ok = runs && runner.is_none_or(|runner| issued.get(&runner) == Some(&true));
                (request, ok)
            })
            .collect();
        (started, handed.into_keys().collect())
    }

    /// The runner that runs a copy whose runs lie in `regions`, those of both runs of each of its
    /// pairs: runner `index`, which serves the batch, when it maps every one ([`Runner::maps`]),
    /// otherwise the first runner of the rank that does; `None` when no runner does.
    fn copier(&mut self, index: usize, regions: &BTreeSet<u8>) -> Option<usize> {
        let maps = |runner: &mut Box<dyn Runner>| regions.iter().all(|region| runner.maps(*region));

        if maps(&mut self.runners[index]) {
            return Some(index);
        }
        self.runners.iter_mut().position(maps)
    }

    /// One paired forward of the shadow batch `batch`, `compared` of whose changed operators
    /// were compared in earlier turns. Returns whether an operator is left for a later turn.
    fn shadow(&mut self, batch: &Batch, compared: usize) -> Result<bool> {
        let index = self.runner_of(batch.phase)?;
        if !self.kernels.contains_key(&batch.generation) {
            return Err(Error::Invariant(format!(
                "shadow batch {} names generation {}, which this worker is not preparing",
                batch.id.0, batch.generation
            )));
        }
        self.write_tables()?;
        let runner = &mut self.runners[index];

        // The inputs are prepared again for each paired forward, which issues them; a paired
        // forward writes scratch memory only, which the runner's shadow inputs state.
        let bucket = runner.prepare_shadow_inputs(batch, &self.tables)?;
        let operators = runner.graphs().replaced(&bucket);
        let Some(operator) = operators.get(compared).copied() else {
            return Ok(false);
        };

        self.queue.note(format!(
            "shadowing `{operator}` of generation {} in bucket {bucket} on batch {} (runner {})",
            batch.generation, batch.id.0, index
        ));
        let elements = runner.graphs().shadow(&**runner, &bucket, operator)?;
        tracing::info!(
            device = ?self.device,
            runner = index,
            operator,
            %bucket,
            generation = batch.generation,
            elements,
            "a replaced kernel was compared with the one it replaces"
        );

        self.publish_uncompared();
        Ok(compared + 1 < operators.len())
    }

    /// Captures the first bucket a runner states and has no graph of. Returns whether another is
    /// left for a later turn.
    fn capture(&mut self) -> Result<bool> {
        // The runners in their order, and each runner's buckets in its own: largest first.
        let mut uncaptured = Vec::new();
        for (index, runner) in self.runners.iter().enumerate() {
            let captured = runner.graphs().buckets();
            uncaptured.extend(
                runner
                    .buckets()
                    .into_iter()
                    .filter(|bucket| !captured.contains(bucket))
                    .map(|bucket| (index, bucket)),
            );
        }
        let Some(&(index, bucket)) = uncaptured.first() else {
            self.set_warming(false);
            return Ok(false);
        };
        let more = uncaptured.len() > 1;

        // Every runner's forwards are a warm-up's while a bucket is left, so the members of an
        // expert union capture in step; a batch served between two turns runs with it set.
        self.set_warming(true);

        // A capture is long device work with no batch to show for it: the line that it started,
        // with none that it finished, is the shape a stuck warm-up has.
        self.queue.note(format!(
            "warming up: capturing bucket {bucket} of runner {index}"
        ));
        tracing::info!(device = ?self.device, runner = index, %bucket, "a bucket's capture starts");

        let started = Instant::now();
        let runner = &self.runners[index];
        let captured = runner.graphs().capture(&**runner, &bucket);

        // The warm-up ends with its last capture, or with the capture that failed it.
        if captured.is_err() || !more {
            self.set_warming(false);
        }
        captured?;

        tracing::info!(
            device = ?self.device,
            runner = index,
            %bucket,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "the bucket's capture finished"
        );
        Ok(more)
    }

    /// States `warming` on every runner ([`Runner::set_warming`]).
    fn set_warming(&self, warming: bool) {
        for runner in &self.runners {
            runner.set_warming(warming);
        }
    }

    /// One turn of the preparation of `next`: the first begins it on every runner, and each
    /// later one prepares one bucket's graph. Returns whether a bucket is left for a later turn.
    fn prepare(&mut self, next: &Arc<Generation>) -> Result<bool> {
        let number = next.number();
        let began = self.kernels.contains_key(&number)
            && self
                .runners
                .iter()
                .all(|runner| runner.graphs().generation() != number);

        if began {
            if let Some((index, bucket)) = self.pending() {
                self.queue.note(format!(
                    "preparing generation {number}: bucket {bucket} of runner {index}"
                ));

                let runner = &self.runners[index];
                let done = runner.graphs().prepare_bucket(&**runner, &bucket)?;
                tracing::info!(
                    device = ?self.device,
                    runner = index,
                    %bucket,
                    generation = number,
                    ?done,
                    "a bucket's graph was prepared for a generation"
                );
            }
            return Ok(self.pending().is_some());
        }

        // The device prepares a generation on all of its runners or on none: a runner that
        // refuses it ends the preparation of the runners before it.
        for (index, runner) in self.runners.iter().enumerate() {
            let Err(error) = runner.graphs().prepare(&**runner, Arc::clone(next)) else {
                continue;
            };

            for earlier in &self.runners[..index] {
                if let Err(error) = earlier.graphs().rollback(number) {
                    tracing::warn!(%error, generation = number, "a refused preparation was not ended");
                }
            }
            return Err(error);
        }

        // The update's evidence starts empty, but for the kernels it adds: nothing serving
        // computes them, so no paired forward will compare them.
        let serving = self
            .runners
            .first()
            .and_then(|runner| self.kernels.get(&runner.graphs().generation()));
        self.unmeasured = serving.map_or_else(Vec::new, |serving| added_kernels(serving, next));
        self.kernels.insert(number, Arc::clone(next));
        self.publish_uncompared();

        Ok(self.pending().is_some())
    }

    /// Switches every runner to the prepared generation `number`.
    fn switch(&mut self, number: u64) -> Result<()> {
        // Checked for every runner before any switches, so the runners serve one generation.
        if let Some((index, bucket)) = self.pending() {
            return Err(Error::Invariant(format!(
                "generation {number} is not prepared: bucket {bucket} of runner {index} has no \
                 graph of it"
            )));
        }

        for runner in &self.runners {
            runner.graphs().switch(number)?;
        }
        Ok(())
    }

    /// Commits the trial of generation `number` on every runner, and drops the generations it
    /// replaced.
    fn commit(&mut self, number: u64) -> Result<()> {
        let committed = self.on_every_runner(|runner| runner.graphs().commit(number));

        if committed.is_ok() {
            self.kernels.retain(|held, _| *held >= number);
        }
        committed
    }

    /// Rolls generation `number` back on every runner.
    fn rollback(&mut self, number: u64) -> Result<()> {
        // A preparation that a runner refused left nothing of the generation here, and the
        // engine rolls a failed update back on every device alike.
        if !self.kernels.contains_key(&number) {
            return Ok(());
        }

        let rolled_back = self.on_every_runner(|runner| runner.graphs().rollback(number));

        // The generation's kernels go with its graphs, unless it is the one that still serves:
        // a rollback the graphs refused.
        let serves = self
            .runners
            .iter()
            .any(|runner| runner.graphs().generation() == number);
        if !serves {
            self.kernels.remove(&number);
            self.unmeasured.clear();
            self.publish_uncompared();
        }
        rolled_back
    }

    /// Calls `operation` on every runner, whatever an earlier one returned, so a failure on one
    /// does not leave the others holding what the operation releases. Returns the first error.
    fn on_every_runner(&self, operation: impl Fn(&dyn Runner) -> Result<()>) -> Result<()> {
        let mut first = Ok(());

        for runner in &self.runners {
            let result = operation(&**runner);
            if first.is_ok() {
                first = result;
            }
        }
        first
    }

    /// The first bucket the generation being prepared still needs a graph of, with its runner:
    /// the runners in their order and each runner's buckets largest first.
    fn pending(&self) -> Option<(usize, Bucket)> {
        self.runners.iter().enumerate().find_map(|(index, runner)| {
            let graphs = runner.graphs();

            runner
                .buckets()
                .into_iter()
                .find(|bucket| graphs.pending(bucket).is_some())
                .map(|bucket| (index, bucket))
        })
    }

    /// States what the update in flight has no comparison for, for [`Worker::uncompared`].
    fn publish_uncompared(&self) {
        let mut uncompared = self.unmeasured.clone();
        for runner in &self.runners {
            uncompared.extend(runner.graphs().uncompared());
        }

        *lock(&self.uncompared) = uncompared;
    }

    /// Waits for the device to pass everything the runners issued: the last forward of every
    /// runner's graphs and every copy a runner issued ([`Runner::wait_copies`]). What a job that
    /// needs nothing in flight on the device starts with.
    fn settle(&mut self) -> Result<()> {
        for runner in &mut self.runners {
            runner.graphs().wait()?;
            runner.wait_copies()?;
        }
        Ok(())
    }

    /// Changes the session's collectors as its window asks, with nothing of the device in
    /// flight.
    fn apply_session(&mut self) -> Result<()> {
        self.settle()?;

        if let Some(session) = self.session.as_mut() {
            session.apply();
        }
        Ok(())
    }

    /// Instantiates the runners' graph executables again when the process declared them stale
    /// since they were instantiated: CUPTI changed how it records a graph's launch, or storage
    /// the graphs address was mapped again.
    fn renew(&mut self) -> Result<()> {
        let renewals = profiling::graph_renewals();
        if renewals == self.renewals {
            return Ok(());
        }

        self.settle()?;
        for runner in &self.runners {
            runner.graphs().renew()?;
        }
        self.renewals = renewals;
        Ok(())
    }

    /// Reads the device's context for a sticky error and keeps the first one found.
    fn check_sticky(&mut self) {
        let Some(device) = self.device else {
            return;
        };

        let sticky =
            platform::PrimaryContext::retain(device).and_then(|context| context.sticky_error());
        if let Ok(Some(sticky)) = sticky {
            tracing::error!(
                device = device.0,
                %sticky,
                "the device's context has a sticky error; no further job is issued"
            );
            self.sticky = Some(sticky);
        }
    }

    /// The runner that serves `phase`.
    fn runner_of(&self, phase: Phase) -> Result<usize> {
        self.runners
            .iter()
            .position(|runner| runner.phases().contains(&phase))
            .ok_or_else(|| {
                Error::Invariant(format!("no runner of this worker serves phase {phase:?}"))
            })
    }
}

/// The kernels `next` holds under keys `serving` has none for, each as a statement of what the
/// update leaves unmeasured, in a stable order.
fn added_kernels(serving: &Generation, next: &Generation) -> Vec<String> {
    let mut added: Vec<String> = next
        .iter()
        .filter(|(operator, key, _)| !serving.contains(operator, key))
        .map(|(operator, key, _)| {
            if serving.keys(operator).next().is_none() {
                format!(
                    "operator `{operator}` is new in generation {}: the serving generation \
                     holds no kernel of it, so its `{key}` kernel is unmeasured",
                    next.number()
                )
            } else {
                format!(
                    "`{operator}` `{key}` is new in generation {}: the serving generation \
                     launches the operator under other keys and none of them computes this one, \
                     so this kernel is unmeasured",
                    next.number()
                )
            }
        })
        .collect();

    added.sort();
    added
}

/// Keeps `value`, and every resource it owns, until the process exits.
///
/// An aborted close, or a release whose device completion is unknown, cannot tell when device
/// work, a DMA engine or a peer rank stops addressing what `value` owns: releasing it could give
/// storage, a mapping or a stream that is still in use to another owner. The value is leaked
/// instead, and process exit returns its resources to the driver.
#[expect(
    clippy::mem_forget,
    reason = "resources whose last device access is unknown must outlive it"
)]
pub(crate) fn leak<T>(value: T) {
    std::mem::forget(value);
}

#[cfg(test)]
mod tests;

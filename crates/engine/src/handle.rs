//! The command channel of a scheduler thread and the event channels of its
//! requests.
//!
//! Frontends control the scheduler thread with [`Command`]s that its
//! [`SchedulerHandle`](crate::SchedulerHandle) sends on a crossbeam channel;
//! each successful send fires the channel's `CommandWaker`, which wakes a
//! parked scheduler unless it is the no-op waker.
//! Each accepted request receives a bounded event channel ([`EventTx`] and
//! [`EventRx`], [`EVENT_BUFFER_CAPACITY`] events). A scheduler that stops
//! while a request's channel is full hands the events it could not send to
//! the receiver, which yields them after the channel's buffered events.
//!
//! [`EventRx`] also carries the frontend decoder's stop-string decisions. The
//! scheduler holds a token request with stop strings at a bounded number of
//! resolved calls whose tokens await a decision; the decoder calls
//! [`EventRx::acknowledge_consumed_prefix`] after checking that the tokens
//! received so far complete no stop string, which reports them as
//! [`Command::Acknowledge`], or [`EventRx::cancel_at_consumed_prefix`] on a
//! match. Dropping an [`EventRx`] before a terminal event cancels the request.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::{Event, Request};
use ipc::RequestId;
use tokio::sync::mpsc;

use crate::lock;

/// Thread-safe notification used to wake an engine after command enqueue.
///
/// Callers run `wake` on their own thread: the command sender right after it
/// enqueues a command, and `EventRx` after it consumes an output event and
/// frees channel capacity. The engine's implementations (the `WakeSignal`
/// socket write and the simulator's progress channel send) do not block.
pub trait Wake: Send + Sync {
    /// Signals the waiting engine owner.
    fn wake(&self);
}

impl<F> Wake for F
where
    F: Fn() + Send + Sync,
{
    /// Invokes the closure as a wake notification.
    fn wake(&self) {
        self()
    }
}

/// The meeting point of the schedulers that run one worker hook together
/// ([`Command::Hook`]).
///
/// The schedulers of one engine serve ranks of one expert union, and a rank
/// joins the union's steps of other schedulers outside its own. A hook that
/// changes how a rank runs the union's steps (its captured joins, the
/// memory they replay from) must find no step of any scheduler in flight,
/// and every rank must have run it before any scheduler steps again. Each
/// scheduler meets the others twice: once its batches in flight completed,
/// and once it ran the hook.
///
/// A scheduler that cannot reach a meeting fails the barrier, which releases
/// the others with its reason instead of leaving them waiting.
pub struct HookBarrier {
    parties: usize,
    state: Mutex<HookMeeting>,
    met: std::sync::Condvar,
}

/// The barrier's current meeting: who arrived, its sequence number, and
/// why it failed.
struct HookMeeting {
    arrived: usize,
    meeting: u64,
    failed: Option<String>,
}

impl HookBarrier {
    /// A barrier for `parties` schedulers (at least one).
    pub fn new(parties: usize) -> Arc<Self> {
        Arc::new(Self {
            parties: parties.max(1),
            state: Mutex::new(HookMeeting {
                arrived: 0,
                meeting: 0,
                failed: None,
            }),
            met: std::sync::Condvar::new(),
        })
    }

    /// Waits until every party arrived at the current meeting.
    ///
    /// # Errors
    ///
    /// Why a party failed the barrier, before or while this one waited.
    pub fn meet(&self) -> Result<(), String> {
        let mut state = lock(&self.state);
        if let Some(reason) = &state.failed {
            return Err(reason.clone());
        }
        let meeting = state.meeting;
        state.arrived += 1;
        if state.arrived == self.parties {
            state.arrived = 0;
            state.meeting += 1;
            self.met.notify_all();
            return Ok(());
        }
        while state.meeting == meeting && state.failed.is_none() {
            state = self
                .met
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        match &state.failed {
            Some(reason) if state.meeting == meeting => Err(reason.clone()),
            _ => Ok(()),
        }
    }

    /// Releases every party waiting now or later with `reason`; the first
    /// reason stands.
    pub fn fail(&self, reason: &str) {
        let mut state = lock(&self.state);
        if state.failed.is_none() {
            state.failed = Some(reason.to_owned());
        }
        self.met.notify_all();
    }
}

/// Cloneable command wake backed by a live notifier or a no-op value.
#[derive(Clone, Default)]
pub enum CommandWaker {
    /// Performs no notification.
    #[default]
    Noop,
    /// Delegates notification to a shared [`Wake`] implementation.
    Live(Arc<dyn Wake>),
}

impl CommandWaker {
    /// Returns a waker that performs no notification.
    pub fn noop() -> Self {
        Self::Noop
    }

    /// Wraps a thread-safe wake closure.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self::Live(Arc::new(wake))
    }

    /// Signals the live notifier, or returns immediately for [`Self::Noop`].
    pub fn wake(&self) {
        if let Self::Live(waker) = self {
            waker.wake();
        }
    }

    /// Returns whether this waker performs no notification.
    pub fn is_noop(&self) -> bool {
        matches!(self, Self::Noop)
    }
}

impl std::fmt::Debug for CommandWaker {
    /// Formats the waker by variant without exposing the live closure.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CommandWaker")
            .field(&if self.is_noop() { "Noop" } else { "Live" })
            .finish()
    }
}

/// Events a closing sender hands to its receiver, shared by the two ends of
/// one request's event channel.
type Handoff = Arc<Mutex<VecDeque<Event>>>;

/// Pollable command ingress whose lifetime is independent of Worker membership.
///
/// A nonblocking Unix socket pair: the `CommandWaker` from [`Self::waker`]
/// owns the write end and writes a byte per wake (a full socket already holds
/// a pending notification), and the owner polls [`Self::descriptor`] for
/// readability and consumes the bytes with [`Self::drain`]. `WorkerExecutor`
/// owns one beside its worker groups and polls it with their progress
/// descriptors.
pub(crate) struct WakeSignal {
    reader: std::os::unix::net::UnixStream,
    waker: CommandWaker,
}

impl WakeSignal {
    pub(crate) fn new() -> std::io::Result<Self> {
        use std::io::Write as _;

        let (reader, writer) = std::os::unix::net::UnixStream::pair()?;
        reader.set_nonblocking(true)?;
        writer.set_nonblocking(true)?;
        let waker = CommandWaker::new(move || {
            loop {
                match (&writer).write(&[1]) {
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    // A full socket already has a pending notification. A closed
                    // reader means its control owner has stopped accepting work.
                    _ => break,
                }
            }
        });
        Ok(Self { reader, waker })
    }

    pub(crate) fn waker(&self) -> CommandWaker {
        self.waker.clone()
    }

    /// Raw read-end descriptor for `poll`; valid while this value lives.
    pub(crate) fn descriptor(&self) -> i32 {
        use std::os::fd::AsRawFd as _;
        self.reader.as_raw_fd()
    }

    /// Consumes latched notifications before the owner considers parking again.
    ///
    /// Returns whether at least one notification was pending. Reads until the
    /// socket is empty or its write end has closed, so any number of wakes
    /// coalesce into one `true`. Read errors other than `WouldBlock` and
    /// `Interrupted` propagate.
    pub(crate) fn drain(&mut self) -> std::io::Result<bool> {
        use std::io::Read as _;

        let mut received = false;
        let mut bytes = [0_u8; 256];
        loop {
            match self.reader.read(&mut bytes) {
                Ok(0) => return Ok(received),
                Ok(_) => received = true,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    return Ok(received);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

/// Maximum number of generation events buffered for one request consumer.
///
/// The scheduler sizes its per-request output journal
/// (`OUTPUT_JOURNAL_CAPACITY`) from this value.
pub const EVENT_BUFFER_CAPACITY: usize = 64;

/// Failure to publish an event into a request's bounded channel.
#[derive(Debug, thiserror::Error)]
pub enum EventSendError {
    #[error("generation event channel is full")]
    /// Carries the event rejected by a full bounded channel.
    Full(Box<Event>),
    #[error("generation event channel is closed")]
    /// Carries the event rejected after the receiver closed.
    Closed(Box<Event>),
}

/// Cause recorded when an event receiver closes before terminal completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamCancelCause {
    #[default]
    /// The consumer dropped or explicitly closed the event stream.
    DroppedStream,
    /// Incremental decoding matched a configured stop string.
    StopStringMatched,
}

/// Request-submission failures visible to engine clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SubmitError {
    #[error("engine command channel is closed")]
    /// The scheduler command channel is closed to submissions.
    Closed,
    #[error("engine is unavailable after a worker failure")]
    /// A terminal worker failure made the engine unavailable.
    Dead,
}

/// Bounded engine-to-caller event sender.
#[derive(Clone)]
pub struct EventTx {
    inner: mpsc::Sender<Event>,
    handoff: Handoff,
}

impl EventTx {
    /// Closes this sender and hands `events`, which did not fit into the
    /// channel, to the receiver.
    ///
    /// The receiver yields them in order once it has drained the channel and
    /// every sender has closed, so only a request's last publisher uses this.
    /// Nothing is kept when the receiver has already closed.
    pub(crate) fn close_with(self, events: VecDeque<Event>) {
        if events.is_empty() || self.inner.is_closed() {
            return;
        }
        // The events are in place before this sender drops, so a receiver
        // that observes the closed channel also observes them.
        lock(&self.handoff).extend(events);
    }

    /// Attempts to publish an event without waiting for channel capacity.
    ///
    /// Never blocks. The scheduler keeps an event rejected as `Full` in the
    /// request's output journal (`EventJournal`) and resends it, in order, on
    /// a later flush that finds the channel has room.
    pub fn send(&self, event: Event) -> Result<(), EventSendError> {
        self.inner.try_send(event).map_err(|error| match error {
            mpsc::error::TrySendError::Full(event) => EventSendError::Full(Box::new(event)),
            mpsc::error::TrySendError::Closed(event) => EventSendError::Closed(Box::new(event)),
        })
    }

    /// Returns the channel's currently available event slots.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns whether the receiving side has closed.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

/// Bounded engine-to-caller event receiver. Releasing one channel slot wakes
/// the scheduler so an output-capacity-stalled request becomes runnable without
/// polling.
///
/// After the channel closes and drains, the receiver yields the events its
/// sender handed over on closing (`EventTx::close_with`), and only then
/// reports the stream closed.
pub struct EventRx {
    inner: mpsc::Receiver<Event>,
    handoff: Handoff,
    waker: CommandWaker,
    cancellation: Option<EventCancellation>,
    /// Text tokens of the received `TextToken` and `TextTokens` events.
    text_tokens_received: usize,
    /// Text-token count of the last `Command::Acknowledge` sent.
    acknowledged_token_count: usize,
    on_finish: Option<Box<dyn FnOnce() + Send + 'static>>,
}

/// Control state that lets an [`EventRx`] act on its request.
///
/// The `handle` clone keeps the command channel open while the receiver
/// holds it; it is dropped on the terminal event, channel close, or first
/// cancellation.
struct EventCancellation {
    handle: CommandSender,
    request_id: RequestId,
    /// Whether the request's tokens await the frontend decoder's stop-string
    /// decisions: set by `submit` for token requests with stop strings.
    awaits_decoder: bool,
}

impl EventRx {
    /// Wraps a Tokio receiver without engine cancellation or wake integration.
    pub fn from_receiver(inner: mpsc::Receiver<Event>) -> Self {
        Self {
            inner,
            handoff: Handoff::default(),
            waker: CommandWaker::noop(),
            cancellation: None,
            text_tokens_received: 0,
            acknowledged_token_count: 0,
            on_finish: None,
        }
    }

    /// Registers a callback to run once on terminal completion or channel close.
    ///
    /// Dropping the receiver also runs a callback that has not yet run.
    pub fn set_on_finish(&mut self, on_finish: impl FnOnce() + Send + 'static) {
        self.on_finish = Some(Box::new(on_finish));
    }

    /// Receives the next event and advances the received token count.
    ///
    /// Returns `None` once the channel is closed and empty and every handed
    /// over event has been yielded, which also runs the completion callback.
    pub async fn recv(&mut self) -> Option<Event> {
        let event = match self.inner.recv().await {
            Some(event) => Some(event),
            None => self.take_handoff(),
        };
        match event.as_ref() {
            Some(event) => {
                self.observe(event);
                // The freed channel slot may unblock a request stalled on
                // output capacity.
                self.waker.wake();
            }
            None => self.finish(),
        }
        event
    }

    /// Receives the next event.
    pub async fn next(&mut self) -> Option<Event> {
        self.recv().await
    }

    /// Attempts to receive an event without waiting.
    ///
    /// Reports `Disconnected` only after every handed over event has been
    /// yielded.
    pub fn try_recv(&mut self) -> Result<Event, mpsc::error::TryRecvError> {
        let event = match self.inner.try_recv() {
            Err(mpsc::error::TryRecvError::Disconnected) => self
                .take_handoff()
                .ok_or(mpsc::error::TryRecvError::Disconnected),
            event => event,
        };
        if let Ok(event) = event.as_ref() {
            self.observe(event);
            self.waker.wake();
        }
        event
    }

    /// Removes the next event a closed sender handed over.
    fn take_handoff(&self) -> Option<Event> {
        lock(&self.handoff).pop_front()
    }

    /// Updates the received token count and completion state for a received
    /// event.
    fn observe(&mut self, event: &Event) {
        match event {
            Event::TextToken { .. } => {
                self.text_tokens_received = self.text_tokens_received.saturating_add(1);
            }
            Event::TextTokens { ids } => {
                self.text_tokens_received = self.text_tokens_received.saturating_add(ids.len());
            }
            event if event.is_terminal() => self.finish(),
            _ => {}
        }
    }

    /// Accepts every text token observed through this receiver as completing
    /// no stop string, so the scheduler resolves the decisions they await.
    ///
    /// Sends nothing for a request without stop strings, whose tokens await
    /// no decision, when no token arrived since the last acknowledgement, or
    /// when the receiver has no armed cancellation (built by `from_receiver`,
    /// finished, or already cancelled).
    pub fn acknowledge_consumed_prefix(&mut self) {
        if self.text_tokens_received <= self.acknowledged_token_count {
            return;
        }
        if let Some(cancellation) = self
            .cancellation
            .as_ref()
            .filter(|cancellation| cancellation.awaits_decoder)
        {
            let _ = cancellation.handle.send(Command::Acknowledge {
                request_id: cancellation.request_id,
                output_token_count: self.text_tokens_received,
            });
            self.acknowledged_token_count = self.text_tokens_received;
        }
    }

    /// Cancels the request.
    pub fn cancel(&mut self) {
        self.cancel_at_consumed_prefix(StreamCancelCause::DroppedStream);
    }

    /// Closes generation after the tokens the consumer has received.
    ///
    /// `DroppedStream` cancels the request; `StopStringMatched` finishes it
    /// successfully with `Stop`. Only the first cancellation of an armed
    /// receiver sends a command; see `acknowledge_consumed_prefix` for when
    /// it is disarmed.
    pub fn cancel_at_consumed_prefix(&mut self, cause: StreamCancelCause) {
        let Some(cancellation) = self.cancellation.take() else {
            return;
        };
        let command = match cause {
            StreamCancelCause::DroppedStream => Command::Cancel(cancellation.request_id),
            StreamCancelCause::StopStringMatched => Command::Stop(cancellation.request_id),
        };
        let _ = cancellation.handle.send(command);
    }

    /// Disarms cancellation and invokes the completion callback exactly once.
    fn finish(&mut self) {
        self.cancellation = None;
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }
}

impl Drop for EventRx {
    /// Cancels an unfinished request, runs an unrun completion callback, and
    /// wakes the scheduler so it observes the closed channel.
    fn drop(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            let _ = cancellation
                .handle
                .send(Command::Cancel(cancellation.request_id));
        }
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
        self.waker.wake();
    }
}

fn event_channel_with_waker(
    waker: CommandWaker,
    cancellation: Option<EventCancellation>,
) -> (EventTx, EventRx) {
    let (tx, rx) = mpsc::channel(EVENT_BUFFER_CAPACITY);
    let handoff = Handoff::default();
    (
        EventTx {
            inner: tx,
            handoff: Arc::clone(&handoff),
        },
        EventRx {
            inner: rx,
            handoff,
            waker,
            cancellation,
            text_tokens_received: 0,
            acknowledged_token_count: 0,
            on_finish: None,
        },
    )
}

/// A request and the destination of its public events.
pub struct Submission {
    /// Request admitted by the scheduler.
    pub request: Box<Request>,
    /// Destination for public request events.
    pub event_tx: EventTx,
}

/// Command sent from a frontend handler to the scheduler thread.
pub enum Command {
    /// Admits requests in order and binds their event channels. The
    /// scheduler applies the whole command before it forms another batch, so
    /// requests submitted together are all queued for the same batch.
    Submit(Vec<Submission>),
    /// Cancels a request on behalf of its consumer.
    Cancel(RequestId),
    /// Finishes a request with `Stop`: the frontend decoder matched one of
    /// its stop strings.
    Stop(RequestId),
    /// Frontend decoder found that this public token prefix of a request with
    /// stop strings completes none of them.
    Acknowledge {
        /// Request whose public output was consumed.
        request_id: RequestId,
        /// Number of public tokens consumed by the frontend.
        output_token_count: usize,
    },
    /// Aborts a request immediately at the server boundary.
    Abort(RequestId),
    /// Parks the scheduler (`Scheduler::park`): ends every token request
    /// that resumes from its generated tokens as migratable, lets the
    /// others finish, and answers with the prefix cache index once nothing
    /// runs; it executes nothing more until resumed.
    Park(crossbeam_channel::Sender<Result<crate::scheduler::ParkedCache, String>>),
    /// Resumes a parked scheduler, and answers whether it did.
    Resume(crossbeam_channel::Sender<Result<(), String>>),
    /// Promotes a successor scheduler (`Scheduler::promote`): imports its
    /// predecessor's prefix cache index, once its cache holds the requests
    /// the predecessor hands over (which `recompute` from their tokens or
    /// not, while it `grows` into the predecessor's memory at once or not),
    /// and answers with the outcome.
    Promote {
        cache: crate::scheduler::ParkedCache,
        recompute: bool,
        grows: bool,
        answer: crossbeam_channel::Sender<Result<(), String>>,
    },
    /// Commits a successor scheduler (`Scheduler::commit`), and answers
    /// with what each worker reported.
    Commit(crossbeam_channel::Sender<Result<Vec<String>, String>>),
    /// Publishes the committed world's immutable pages on every claimed node.
    PublishWorld(crossbeam_channel::Sender<Result<(), String>>),
    /// Ends the memory of other worlds on the nodes of the scheduler's
    /// workers, discarding the pages the scheduler's world does not hit when
    /// `discard_pages` (`Executor::retire`), and answers whether it did.
    Retire {
        worlds: Vec<u64>,
        discard_pages: bool,
        answer: crossbeam_channel::Sender<Result<(), String>>,
    },
    /// Reads the memory ledgers of the nodes of the scheduler's workers
    /// (`Executor::ledgers`), and answers with each node's.
    Ledgers(crossbeam_channel::Sender<Result<Vec<String>, String>>),
    /// Halves, or restores when not `halve`, the byte budgets of the links
    /// the deployment's transfers use on the nodes of the scheduler's
    /// workers (`Executor::throttle`), and answers with those nodes' hosts.
    Throttle {
        halve: bool,
        answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    },
    /// Keeps the weights alone of other worlds on the nodes of the
    /// scheduler's workers, in host memory when `to_host`
    /// (`Executor::keep_weights`), and answers with each node's report.
    KeepWeights {
        worlds: Vec<u64>,
        to_host: bool,
        answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    },
    /// Reports the KV cache each worker group holds now
    /// (`Scheduler::describe_cache`).
    DescribeCache(crossbeam_channel::Sender<Result<Vec<crate::scheduler::GroupCache>, String>>),
    /// Accepted prefixes of running requests that can move to another world.
    DescribeRequests(crossbeam_channel::Sender<Vec<proto::request::wire::RequestPrefix>>),
    CacheWarm {
        operation: storage::manifest::cache::Operation,
        together: Arc<HookBarrier>,
        answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    },
    /// Revokes every cached prefix page, so that later prompts compute
    /// their prefixes anew; requests in flight keep their pages.
    FlushCache,
    /// Runs the worker hook `name` with `payload` on every worker once the
    /// batches in flight complete, and answers with each worker's result.
    /// Every scheduler `together` names runs it at once: none runs it
    /// before all have no batch in flight, and none schedules again before
    /// all ran it.
    Hook {
        name: String,
        payload: String,
        together: Arc<HookBarrier>,
        answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    },
    /// Takes other scheduling limits and policy from the next step on
    /// (`Scheduler::reconfigure`), and answers whether it did.
    Configure {
        limits: Box<crate::scheduler::SchedulerConfig>,
        answer: crossbeam_channel::Sender<Result<(), String>>,
    },
    /// Gives back or restores KV cache memory (`Scheduler::resize_cache`),
    /// and answers with what each worker reported.
    ResizeCache {
        change: CacheChange,
        answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    },
    /// Rewrites the weights in place from the checkpoint at `model_path`
    /// (`Scheduler::update_weights`), the requests in flight continuing as
    /// `in_flight` says, and answers with what each worker reported.
    UpdateWeights {
        model_path: String,
        in_flight: WeightsInFlight,
        answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    },
    /// Commit prepared expert copies across a common step boundary.
    CommitExperts {
        order: std::collections::BTreeMap<String, Vec<u32>>,
        together: Arc<crate::HookBarrier>,
        answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    },
    /// Requests orderly engine shutdown.
    Shutdown,
}

/// How the requests in flight continue through an in-place weight update
/// (vLLM's `keep`, `abort` and SGLang's `in_place`, `retract`, `abort`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightsInFlight {
    /// Each continues on the new weights with the KV cache it reached; the
    /// pages it writes afterwards reach no shared tier.
    #[default]
    Keep,
    /// Each computes its state again with the new weights.
    Recompute,
    /// Each ends, aborted, as do the requests the scheduler queued.
    Abort,
}

/// Target rank resources by GPU UUID, including the concrete weight chunks
/// created beside retained source chunks.
pub type RankBudgets = std::collections::BTreeMap<[u8; 16], storage::manifest::weights::RankBudget>;

/// What a world succeeding this one on the same devices adds beside what it
/// shares with it, which a shrink makes room for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Successor {
    /// It loads its weights from the model's files (their keys differ), so
    /// it holds as many beside the serving world's.
    pub loads_weights: bool,
    /// It maps none of the serving world's KV cache, so copied or
    /// recomputed in-flight state needs units of its own.
    pub unmapped_cache: bool,
    /// Its graph policy captures graphs (`cuda_graph_config.policy` is not
    /// `off`), so its warmup takes a graph share where its model can be
    /// captured.
    pub captures_graphs: bool,
    /// Its target numerical capacity and concrete transfer allocations.
    pub rank_budgets: RankBudgets,
}

/// A change to the device memory of a scheduler's KV cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheChange {
    /// Gives back the tail of the cache until each device has the storage a
    /// world succeeding this one needs beside the storage they share, the
    /// weights it creates included; the requests that hold the tail run
    /// on, and are recomputed only once `preempt_after` passed, when given.
    Shrink {
        preempt_after: Option<std::time::Duration>,
        successor: Successor,
    },
    /// Creates the given-back tail again.
    Restore,
    /// Gives back the whole cache and the CUDA graph memory, keeping the
    /// weights, once no request runs; the scheduler admits nothing until
    /// it wakes.
    Sleep,
    /// Restores what a sleep gave back.
    Wake,
    /// Creates as many of the given-back segments as each device's memory
    /// ledger has room for.
    Grow,
    /// Plans a shrink without withdrawing anything: answers with each
    /// device's memory beside a world succeeding this one, or fails with
    /// why such a world does not fit.
    Plan(Successor),
    /// For a parked world whose successor maps none of its cache: gives
    /// back every segment past the first, once the idle cached pages they
    /// hold reached the host KV cache, so that the successor's cache grows
    /// into them. A resume creates them again.
    Evict,
    /// Gives back the tail of the cache until each device's storage ledger
    /// has room for `bytes` beside everything it books, with per-GPU values
    /// from `devices` overriding that amount; tail holders continue running.
    /// Fails, changing nothing, when even the segments the cache cannot
    /// give back leave too little room.
    MakeRoom {
        bytes: u64,
        devices: std::collections::BTreeMap<storage::manifest::Gpu, u64>,
    },
}

/// The cloneable sender of one scheduler's commands: the counterpart of the
/// receiver `Scheduler::run` and `Scheduler::step` drain. A
/// [`SchedulerHandle`](crate::SchedulerHandle) holds the sender of the
/// scheduler it runs.
///
/// The scheduler stops when every sender of its command channel has dropped,
/// which includes the clones held by live [`EventRx`] values that are still
/// armed for cancellation.
///
/// Every send fires the `CommandWaker` right after enqueuing, so when the
/// scheduler is parked on an event-driven executor it wakes immediately to
/// observe the command instead of waiting out the park's safety-net timeout.
/// With the no-op waker, a parked scheduler observes commands only when its
/// executor poll returns.
#[derive(Clone)]
pub struct CommandSender {
    tx: crossbeam_channel::Sender<Command>,
    waker: CommandWaker,
}

impl CommandSender {
    /// Constructs a sender with the no-op waker, for a scheduler that
    /// observes commands only through its timed executor poll.
    pub fn new(tx: crossbeam_channel::Sender<Command>) -> Self {
        Self::with_waker(tx, CommandWaker::noop())
    }

    /// Constructs a sender that fires `waker` after every enqueue, used when
    /// the scheduler drives an event-driven executor that parks between
    /// steps.
    pub fn with_waker(tx: crossbeam_channel::Sender<Command>, waker: CommandWaker) -> Self {
        Self { tx, waker }
    }

    /// Enqueues a command, then wakes the scheduler after a successful send.
    pub(crate) fn send(&self, cmd: Command) -> Result<(), crossbeam_channel::SendError<Command>> {
        let r = self.tx.send(cmd);
        // Wake only on a successful enqueue: if the channel is closed there is
        // no scheduler to wake, and the error is propagated to the caller.
        if r.is_ok() {
            self.waker.wake();
        }
        r
    }

    /// Submits a request and returns its bounded event stream.
    ///
    /// The returned receiver cancels the request when dropped before its
    /// terminal event. Fails with `SubmitError::Closed` when the command
    /// channel is closed; a sender never reports `SubmitError::Dead`, which
    /// `SchedulerHandle::submit` checks before calling it.
    pub fn submit(&self, request: impl Into<Request>) -> Result<EventRx, SubmitError> {
        let (submission, events) = self.bind(request.into());
        self.send(Command::Submit(vec![submission]))
            .map_err(|_| SubmitError::Closed)?;
        Ok(events)
    }

    /// Submits requests in one command and returns their event streams in
    /// submission order.
    ///
    /// The scheduler queues every one of them before it forms its next batch,
    /// as it would requests from concurrent clients that all arrived before
    /// that batch. Separate `submit` calls give no such guarantee: the
    /// scheduler may form batches between them. Each receiver behaves as one
    /// returned by `submit`, and the errors are the same.
    pub fn submit_all(
        &self,
        requests: impl IntoIterator<Item = Request>,
    ) -> Result<Vec<EventRx>, SubmitError> {
        let (submissions, streams) = requests
            .into_iter()
            .map(|request| self.bind(request))
            .unzip();
        self.send(Command::Submit(submissions))
            .map_err(|_| SubmitError::Closed)?;
        Ok(streams)
    }

    /// Pairs a request with a new event channel whose receiver cancels the
    /// request when dropped before its terminal event.
    fn bind(&self, request: Request) -> (Submission, EventRx) {
        let request_id = request.request_id();
        // Stop-string matching happens in the frontend decoder, which
        // acknowledges each token it has checked.
        let awaits_decoder = match &request {
            Request::Generation(request) => !request.stop_strings.is_empty(),
            Request::Diffusion(_) => false,
        };
        let (event_tx, event_rx) = event_channel_with_waker(
            self.waker.clone(),
            Some(EventCancellation {
                handle: self.clone(),
                request_id,
                awaits_decoder,
            }),
        );
        let submission = Submission {
            request: Box::new(request),
            event_tx,
        };
        (submission, event_rx)
    }

    /// Cancels a request.
    pub fn cancel(&self, id: RequestId) {
        let _ = self.send(Command::Cancel(id));
    }

    /// Finishes a request successfully, as when its decoder matched a stop
    /// string.
    pub fn stop(&self, id: RequestId) {
        let _ = self.send(Command::Stop(id));
    }

    /// Acknowledges that the first `output_token_count` public tokens of a
    /// request with stop strings complete none of them.
    pub fn acknowledge_at(&self, id: RequestId, output_token_count: usize) {
        let _ = self.send(Command::Acknowledge {
            request_id: id,
            output_token_count,
        });
    }

    /// Aborts a request immediately without applying client cancellation semantics.
    pub fn abort(&self, id: RequestId) {
        let _ = self.send(Command::Abort(id));
    }

    /// Parks the scheduler (`Command::Park`); the receiver gets its
    /// prefix cache index and what its migrating requests held once it
    /// parked, or why it did not.
    pub fn park(
        &self,
    ) -> crossbeam_channel::Receiver<Result<crate::scheduler::ParkedCache, String>> {
        let (answer, parked) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::Park(answer));
        parked
    }

    /// Resumes a parked scheduler (`Command::Resume`); the receiver gets
    /// whether it did.
    pub fn resume(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        let (answer, resumed) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::Resume(answer));
        resumed
    }

    /// Promotes a successor scheduler with what its predecessor parked
    /// with (`Command::Promote`); the receiver gets the outcome.
    pub fn promote(
        &self,
        cache: crate::scheduler::ParkedCache,
        recompute: bool,
        grows: bool,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        let (answer, promoted) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::Promote {
            cache,
            recompute,
            grows,
            answer,
        });
        promoted
    }

    /// Publishes the committed world's immutable pages on every claimed node.
    pub fn publish_world(&self) -> crossbeam_channel::Receiver<Result<(), String>> {
        let (answer, published) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::PublishWorld(answer));
        published
    }

    /// Ends the memory of `worlds` on the nodes of the scheduler's workers,
    /// discarding the pages the scheduler's world does not hit when
    /// `discard_pages` (`Command::Retire`); the receiver gets whether it did.
    pub fn retire(
        &self,
        worlds: Vec<u64>,
        discard_pages: bool,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        let (answer, retired) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::Retire {
            worlds,
            discard_pages,
            answer,
        });
        retired
    }

    /// Keeps the weights alone of `worlds` on the nodes of the scheduler's
    /// workers, in host memory when `to_host` (`Command::KeepWeights`); the
    /// receiver gets each node's report.
    pub fn keep_weights(
        &self,
        worlds: Vec<u64>,
        to_host: bool,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        let (answer, kept) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::KeepWeights {
            worlds,
            to_host,
            answer,
        });
        kept
    }

    /// Reads the memory ledgers of the nodes of the scheduler's workers
    /// (`Command::Ledgers`); the receiver gets each node's.
    pub fn ledgers(&self) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        let (answer, read) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::Ledgers(answer));
        read
    }

    /// Halves, or restores when not `halve`, the byte budgets of the links
    /// the deployment's transfers use (`Command::Throttle`); the receiver
    /// gets the hosts it throttled.
    pub fn throttle(
        &self,
        halve: bool,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        let (answer, throttled) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::Throttle { halve, answer });
        throttled
    }

    /// Reports the KV cache each worker group holds now
    /// (`Command::DescribeCache`).
    pub fn describe_cache(
        &self,
    ) -> crossbeam_channel::Receiver<Result<Vec<crate::scheduler::GroupCache>, String>> {
        let (answer, described) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::DescribeCache(answer));
        described
    }

    /// Read accepted cache lengths of requests that can move to a successor.
    pub fn describe_requests(
        &self,
    ) -> crossbeam_channel::Receiver<Vec<proto::request::wire::RequestPrefix>> {
        let (answer, described) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::DescribeRequests(answer));
        described
    }

    /// Execute one scheduler-owned cache warming operation at a batch boundary.
    pub fn cache_warm(
        &self,
        operation: storage::manifest::cache::Operation,
        together: Arc<HookBarrier>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        let (answer, response) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::CacheWarm {
            operation,
            together,
            answer,
        });
        response
    }

    /// Commits a successor scheduler's world (`Command::Commit`); the
    /// receiver gets what each worker reported.
    pub fn commit(&self) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        let (answer, committed) = crossbeam_channel::bounded(1);
        let _ = self.send(Command::Commit(answer));
        committed
    }

    /// Revokes every cached prefix page (`Command::FlushCache`).
    pub fn flush_cache(&self) {
        let _ = self.send(Command::FlushCache);
    }

    /// Runs the worker hook `name` (`Command::Hook`) together with the
    /// other schedulers of `together`; the receiver gets each worker's
    /// result, or why the hook failed.
    pub fn hook(
        &self,
        name: String,
        payload: String,
        together: Arc<HookBarrier>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        let (answer, receiver) = crossbeam_channel::bounded(1);
        if let Err(error) = self.send(Command::Hook {
            name,
            payload,
            together,
            answer,
        }) && let Command::Hook {
            answer, together, ..
        } = error.0
        {
            // The schedulers waiting for this one go on without it.
            together.fail("the scheduler stopped");
            let _ = answer.send(Err("the scheduler stopped".to_owned()));
        }
        receiver
    }

    /// Takes other scheduling limits and policy from the next step on
    /// (`Command::Configure`); the receiver gets whether the scheduler did.
    pub fn configure(
        &self,
        limits: crate::scheduler::SchedulerConfig,
    ) -> crossbeam_channel::Receiver<Result<(), String>> {
        let (answer, receiver) = crossbeam_channel::bounded(1);
        let limits = Box::new(limits);
        if let Err(error) = self.send(Command::Configure { limits, answer })
            && let Command::Configure { answer, .. } = error.0
        {
            let _ = answer.send(Err("the scheduler stopped".to_owned()));
        }
        receiver
    }

    /// Gives back or restores KV cache memory (`Command::ResizeCache`); the
    /// receiver gets what each worker reported, or why the change failed.
    pub fn resize_cache(
        &self,
        change: CacheChange,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        let (answer, receiver) = crossbeam_channel::bounded(1);
        if let Err(error) = self.send(Command::ResizeCache { change, answer })
            && let Command::ResizeCache { answer, .. } = error.0
        {
            let _ = answer.send(Err("the scheduler stopped".to_owned()));
        }
        receiver
    }

    /// Rewrites the weights in place from the checkpoint at `model_path`,
    /// the requests in flight continuing as `in_flight` says
    /// (`Command::UpdateWeights`); the receiver gets what each worker
    /// reported, or why the update failed.
    pub fn update_weights(
        &self,
        model_path: String,
        in_flight: WeightsInFlight,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        let (answer, receiver) = crossbeam_channel::bounded(1);
        if let Err(error) = self.send(Command::UpdateWeights {
            model_path,
            in_flight,
            answer,
        }) && let Command::UpdateWeights { answer, .. } = error.0
        {
            let _ = answer.send(Err("the scheduler stopped".to_owned()));
        }
        receiver
    }

    /// Requests orderly scheduler shutdown.
    pub fn shutdown(&self) {
        let _ = self.send(Command::Shutdown);
    }

    pub fn commit_experts(
        &self,
        order: std::collections::BTreeMap<String, Vec<u32>>,
        together: Arc<crate::HookBarrier>,
    ) -> crossbeam_channel::Receiver<Result<Vec<String>, String>> {
        let (answer, receiver) = crossbeam_channel::bounded(1);
        if let Err(error) = self.send(Command::CommitExperts {
            order,
            together,
            answer,
        }) && let Command::CommitExperts { answer, .. } = error.0
        {
            let _ = answer.send(Err("the scheduler stopped".to_owned()));
        }
        receiver
    }
}

#[cfg(test)]
mod tests {
    use crate::{GenerationConstraint, GenerationRequest, ImageGenerationConfig};
    use ipc::{ImageParams, RequestId, SamplingParams};

    use super::*;

    fn test_request(request_id: u64) -> GenerationRequest {
        let constraint = GenerationConstraint::UndOnly;
        let policy = ImageGenerationConfig::default();
        GenerationRequest {
            request_id: RequestId(request_id),
            prompt_token_ids: vec![1, 2, 3],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_images: 1,
            max_und_tokens: 32,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: policy,
            readout: Vec::new(),
            generated_token_ids: Vec::new(),
            adapter: 0,
            generated_images: Vec::new(),
        }
    }

    /// Request ids of a `Command::Submit`, in submission order.
    fn submitted_ids(command: Command) -> Vec<RequestId> {
        match command {
            Command::Submit(submissions) => submissions
                .iter()
                .map(|submission| submission.request.request_id())
                .collect(),
            _ => panic!("expected Submit command"),
        }
    }

    /// The event channel of a `Command::Submit` of one request.
    fn submitted_events(command: Command) -> EventTx {
        match command {
            Command::Submit(mut submissions) if submissions.len() == 1 => {
                submissions.pop().unwrap().event_tx
            }
            _ => panic!("expected a Submit command of one request"),
        }
    }

    /// `submit` enqueues a `Command::Submit` carrying the request, and the
    /// scheduler side receives exactly that request id.
    #[test]
    fn submit_enqueues_submit_command_with_request() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::new(tx);

        let _events = handle.submit(test_request(11)).unwrap();

        assert_eq!(submitted_ids(rx.recv().unwrap()), vec![RequestId(11)]);
    }

    /// `submit_all` hands the scheduler every request in one command, in
    /// submission order, so the scheduler queues them all before it forms
    /// its next batch.
    #[test]
    fn submit_all_enqueues_its_requests_in_one_command() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::new(tx);

        let events = handle
            .submit_all([test_request(21), test_request(22), test_request(23)].map(Request::from))
            .unwrap();

        assert_eq!(events.len(), 3);
        assert_eq!(
            submitted_ids(rx.recv().unwrap()),
            vec![RequestId(21), RequestId(22), RequestId(23)]
        );
        assert!(rx.try_recv().is_err());
    }

    /// Tokens of a request without stop strings await no decoder decision,
    /// so receiving and accepting them sends nothing; dropping the receiver
    /// before the terminal event cancels the request.
    #[test]
    fn dropping_event_receiver_cancels_the_request() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::new(tx);
        let mut events = handle.submit(test_request(12)).unwrap();
        let event_tx = submitted_events(rx.recv().unwrap());

        event_tx
            .send(Event::TextToken {
                id: 7,
                logprob: None,
            })
            .unwrap();
        assert!(matches!(
            events.try_recv(),
            Ok(Event::TextToken { id: 7, .. })
        ));
        events.acknowledge_consumed_prefix();
        drop(events);

        match rx.recv().unwrap() {
            Command::Cancel(request_id) => assert_eq!(request_id, RequestId(12)),
            _ => panic!("expected Cancel command"),
        }
        assert!(rx.try_recv().is_err());
    }

    /// The decoder of a request with stop strings reports each newly accepted
    /// token prefix once, and a matched stop string finishes the request with
    /// `Stop`.
    #[test]
    fn stop_string_request_reports_decoder_decisions() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::new(tx);
        let mut request = test_request(14);
        request.stop_strings = vec!["end".to_string()];
        let mut events = handle.submit(request).unwrap();
        let event_tx = submitted_events(rx.recv().unwrap());

        event_tx
            .send(Event::TextTokens { ids: vec![7, 8] })
            .unwrap();
        assert!(events.try_recv().is_ok());
        events.acknowledge_consumed_prefix();
        events.acknowledge_consumed_prefix();
        match rx.recv().unwrap() {
            Command::Acknowledge {
                request_id,
                output_token_count,
            } => {
                assert_eq!(request_id, RequestId(14));
                assert_eq!(output_token_count, 2);
            }
            _ => panic!("expected Acknowledge command"),
        }

        events.cancel_at_consumed_prefix(StreamCancelCause::StopStringMatched);
        match rx.recv().unwrap() {
            Command::Stop(request_id) => assert_eq!(request_id, RequestId(14)),
            _ => panic!("expected Stop command"),
        }
        drop(events);
        assert!(rx.try_recv().is_err());
    }

    /// Cancellation, stop, semantic acknowledgement, and server abort
    /// preserve their distinct command payloads.
    #[test]
    fn request_control_commands_preserve_their_semantics() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::new(tx);

        handle.cancel(RequestId(1));
        handle.stop(RequestId(4));
        handle.acknowledge_at(RequestId(3), 9);
        handle.abort(RequestId(2));

        match rx.recv().unwrap() {
            Command::Cancel(request_id) => assert_eq!(request_id, RequestId(1)),
            _ => panic!("expected Cancel command"),
        }
        match rx.recv().unwrap() {
            Command::Stop(request_id) => assert_eq!(request_id, RequestId(4)),
            _ => panic!("expected Stop command"),
        }
        match rx.recv().unwrap() {
            Command::Acknowledge {
                request_id,
                output_token_count,
            } => {
                assert_eq!(request_id, RequestId(3));
                assert_eq!(output_token_count, 9);
            }
            _ => panic!("expected Acknowledge command"),
        }
        match rx.recv().unwrap() {
            Command::Abort(id) => assert_eq!(id, RequestId(2)),
            _ => panic!("expected Abort command"),
        }
    }

    /// Events a closing sender hands over follow every event already buffered
    /// in the full channel, in order, and the stream reports closed only after
    /// the last of them.
    #[tokio::test]
    async fn receiver_yields_handed_over_events_after_the_channel_drains() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::new(tx);
        let mut events = handle.submit(test_request(13)).unwrap();
        let event_tx = submitted_events(rx.recv().unwrap());

        let token = |id| Event::TextToken { id, logprob: None };
        for id in 0..EVENT_BUFFER_CAPACITY as u32 {
            event_tx.send(token(id)).unwrap();
        }
        let capacity = EVENT_BUFFER_CAPACITY as u32;
        assert!(matches!(
            event_tx.send(token(capacity)),
            Err(EventSendError::Full(_))
        ));
        let terminal = Event::Finished {
            reason: crate::FinishReason::Aborted,
            stop_reason: None,
            prompt_tokens: 3,
            completion_tokens: EVENT_BUFFER_CAPACITY + 1,
            images: 0,
        };
        event_tx.close_with(VecDeque::from([token(capacity), terminal]));

        for expected in 0..=capacity {
            match events.recv().await {
                Some(Event::TextToken { id, .. }) => assert_eq!(id, expected),
                other => panic!("expected token {expected}, received {other:?}"),
            }
        }
        assert!(matches!(
            events.recv().await,
            Some(Event::Finished {
                reason: crate::FinishReason::Aborted,
                ..
            })
        ));
        assert!(events.recv().await.is_none());
    }

    /// After the scheduler side of the channel is dropped, `submit` reports an
    /// error rather than panicking or silently dropping the request.
    #[test]
    fn submit_after_receiver_dropped_returns_error() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::new(tx);
        drop(rx);

        let result = handle.submit(test_request(99));
        assert!(result.is_err());
    }

    /// The waker fires exactly once per successful enqueue, and not at all when
    /// the channel is closed (no scheduler to wake).
    #[test]
    fn waker_fires_on_successful_enqueue_only() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let wakes = Arc::new(AtomicUsize::new(0));
        let wakes_clone = Arc::clone(&wakes);
        let waker = CommandWaker::new(move || {
            wakes_clone.fetch_add(1, Ordering::SeqCst);
        });

        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::with_waker(tx, waker);

        handle.cancel(RequestId(1));
        handle.abort(RequestId(2));
        assert_eq!(wakes.load(Ordering::SeqCst), 2);

        // Closing the channel means there is no scheduler to wake.
        drop(rx);
        handle.cancel(RequestId(3));
        assert_eq!(wakes.load(Ordering::SeqCst), 2);
    }

    /// Cloned handles share the same underlying channel: a command sent through
    /// the clone is observed by the original's receiver.
    #[test]
    fn cloned_handle_shares_channel() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = CommandSender::new(tx);
        let clone = handle.clone();

        clone.shutdown();

        assert!(matches!(rx.recv().unwrap(), Command::Shutdown));
    }

    /// Schedulers meeting at a hook barrier pass each meeting only once
    /// every one arrived: no party passes the second meeting before the
    /// last has passed the first and arrived again.
    #[test]
    fn hook_barrier_parties_meet_each_time_all_arrive() {
        let together = HookBarrier::new(3);
        let passed = Arc::new(Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            for party in 0..3 {
                let together = Arc::clone(&together);
                let passed = Arc::clone(&passed);
                scope.spawn(move || {
                    for meeting in 0..2 {
                        together.meet().unwrap();
                        passed.lock().unwrap().push((meeting, party));
                    }
                });
            }
        });
        let passed = passed.lock().unwrap();
        assert_eq!(passed.len(), 6);
        // Every pass of the first meeting precedes every pass of the second.
        let last_first = passed.iter().rposition(|&(meeting, _)| meeting == 0);
        let first_second = passed.iter().position(|&(meeting, _)| meeting == 1);
        assert!(last_first < first_second);
    }

    /// A party that cannot reach a meeting releases the parties waiting
    /// there, and those that arrive later, with its reason.
    #[test]
    fn a_failed_hook_barrier_releases_its_parties_with_the_reason() {
        let together = HookBarrier::new(3);
        let waiting = {
            let together = Arc::clone(&together);
            std::thread::spawn(move || together.meet())
        };
        // The waiting party arrived; the barrier fails before the third does.
        while lock(&together.state).arrived == 0 {
            std::thread::yield_now();
        }
        together.fail("the engine failed before the hook ran");
        assert_eq!(
            waiting.join().unwrap(),
            Err("the engine failed before the hook ran".to_owned())
        );
        assert_eq!(
            together.meet(),
            Err("the engine failed before the hook ran".to_owned())
        );
    }

    /// A barrier of one party never waits.
    #[test]
    fn a_hook_barrier_of_one_party_passes_at_once() {
        let together = HookBarrier::new(1);
        assert_eq!(together.meet(), Ok(()));
        assert_eq!(together.meet(), Ok(()));
    }
}

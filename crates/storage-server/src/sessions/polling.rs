//! The buffer pool's thread: it serves every session's queue pairs.
//!
//! The buffer pools of the host's deployments, one per deployment, live on
//! one thread of their own, which serves their queue pairs in poll mode
//! (`specs/buffer-pool.md` §9, §10). Neither side interrupts the other: the
//! thread reads the submission queues and the `completed` registers in
//! shared memory, and backs off to a bounded sleep ([`SLEEP`]) only when
//! every queue is empty and nothing is deferred, so it makes no system call
//! for an engine and an engine's step path makes none for it. It moves no
//! bytes and waits on nothing inside a handler, and it uses one core.
//!
//! A buffer pool holds its blocks by `Rc` and is not `Send`: it is made,
//! used and dropped on this thread. The server's main loop (one `poll(2)`
//! thread) and this thread exchange closures, each run on the other's
//! thread with the other's state:
//!
//! - The main loop sends work ([`Thread::send`]) and waits for the result
//!   of what it answers a client with at once ([`Thread::call`]): a session
//!   opened or closed, an engine found gone, a task's progress and end, a
//!   peer's holds, the pools' listing, and at a restart in place the freeze
//!   and the hand-off. The thread takes work at the start of each turn, and
//!   a send wakes it from its sleep.
//! - The thread sends notices ([`Thread::notices`]): a task to start, with
//!   what it copies, and a task to stop. A notice wakes the main loop
//!   through a pipe in its poll set; a task runs on a child process, which
//!   the main loop owns. The helpers that talk to other hosts' Storage
//!   Servers for a task send their notices the same way ([`Notifier`]).
//!
//! Each turn, for each deployment: read each group's `completed` register
//! and advance the pool; take the requests of each submission queue in
//! order (a record that cannot be read is answered `Refused { cause:
//! Request }` where its number can be read, and dropped otherwise); tell the
//! pool the host's monotonic clock; write the completions, a completion
//! that finds no room staying pending in order (the scheduler keeps room for
//! everything outstanding, so that is the engine's fault and never a reason
//! to wait); write each region's three capacity words for each queue pair
//! when they changed; and send the tasks to start and to stop. A completion
//! longer than a ring holds is answered `Refused { cause: Request }`: only a
//! `List` of a cache too large for the ring comes to that, and it changed
//! nothing. A queue pair whose indices the engine broke is served no more.
//!
//! The thread acts on its inputs at one place per input, and there each
//! pool's [`Watch`] sees them: when the server's options say so, a trace of
//! every input and completion ([`Trace`], `specs/buffer-pool.md` §18), and
//! the pool's check of its tables after every input, whose failures are
//! logged and counted in the pool's listing while the pool goes on.
//!
//! A restart in place freezes the thread: it takes no request and reads no
//! `completed` word, so no work starts, while the tasks run to their end
//! and their completions are written. Once no task is outstanding the
//! thread's whole state, every pool's image with its queue pairs, passes to
//! the successor ([`Pools::hand_off`], [`Thread::start`]); the queue pairs
//! keep their indices, and requests submitted meanwhile wait in the rings.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use storage::QueuePair;
use storage::protocol::{BufferPoolListing, HostCopy, Key, Segment};
use storage::runtime::{
    Cause, Completion, GroupId, Open, Opened, Registers, Request, RequestId, Source, VERSION,
};

use crate::Server;
use crate::buffer_pool::trace::{Input, Trace, stated_input};
use crate::buffer_pool::{BufferPool, ClientId, FORMAT, QueueId, format1, format2, format3};

/// How long the thread sleeps when every queue is empty and nothing is
/// deferred: the longest a request of an idle engine waits to be taken.
pub(crate) const SLEEP: Duration = Duration::from_micros(200);

/// The longest a trace keeps what it recorded in its buffers, in
/// nanoseconds, while the thread is busy; it writes out whenever the thread
/// has nothing to do.
const TRACE_FLUSH: u64 = 100_000_000;

/// The bytes of each ring of a queue pair. The largest record a ring takes
/// is a little less; a scheduler keeps room for everything outstanding.
const RING_BYTES: usize = 4 << 20;

/// Work the main loop sends the thread, run with the thread's state.
pub(crate) type Work = Box<dyn FnOnce(&mut Pools) + Send>;

/// A notice the thread sends the main loop, run with the server.
pub(crate) type Notice = Box<dyn FnOnce(&mut Server) + Send>;

/// A run of units as a task names it: its region's name, its first unit and
/// its count.
pub(crate) type NamedRun = (String, u32, u32);

/// A host part as a task names it: the host, as the session's layout names
/// it (`storage::runtime::Group::parts`), and the part index.
pub(crate) type HostPart = (String, u32);

/// Memories as a task names them: device memories and main processes'
/// memories by name, and host parts.
pub(crate) type Memories = (Vec<String>, Vec<HostPart>);

/// One copy of a task, as the main loop runs it: the key it is for if any,
/// the runs it reads and writes, the memories that hold the content it
/// reads and those it writes into ([`BufferPool::work`]). A copy of a slot
/// row, whose units exist once, names no memory.
pub(crate) type Copied = (
    Option<Key>,
    Vec<NamedRun>,
    Vec<NamedRun>,
    Memories,
    Memories,
);

/// The three capacity words of a region for a queue pair: `available`,
/// `retiring`, `evictable` ([`BufferPool::capacity`]).
type Capacity = (u32, u32, u32);

/// What sends the main loop notices and wakes it: the buffer pool's
/// thread, and the helpers that talk to other hosts for its tasks.
#[derive(Clone)]
pub(crate) struct Notifier {
    notices: mpsc::Sender<Notice>,
    /// The write end of the main loop's wake pipe.
    wake: Arc<OwnedFd>,
}

impl Notifier {
    /// Sends the main loop a notice and wakes it.
    pub(crate) fn send(&self, notice: Notice) {
        if self.notices.send(notice).is_ok() {
            let _ = rustix::io::write(&*self.wake, b"n");
        }
    }
}

/// The main loop's end of the thread.
pub(crate) struct Thread {
    work: Option<mpsc::Sender<Work>>,
    notices: mpsc::Receiver<Notice>,
    /// The read end of the pipe a notice wakes the main loop through.
    wake: OwnedFd,
    notifier: Notifier,
    thread: Option<JoinHandle<()>>,
}

impl Thread {
    /// Starts the thread: with no pool, or going on with the state a
    /// predecessor handed over (`image`, the bytes of [`Pools::hand_off`] in
    /// the format stated beside them, with the queue pairs' memfds in its
    /// order). `limits` is how long a transfer and a put to the remote store
    /// may take, in nanoseconds.
    ///
    /// # Errors
    ///
    /// Fails when the pipe or the thread cannot be made, or the state
    /// handed over cannot be made again.
    pub(crate) fn start(
        limits: (u64, u64),
        image: Option<(Vec<u8>, u32, Vec<OwnedFd>)>,
    ) -> io::Result<Self> {
        use rustix::pipe::{PipeFlags, pipe_with};

        let (reader, writer) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK)?;
        let (work, jobs) = mpsc::channel::<Work>();
        let (notify, notices) = mpsc::channel::<Notice>();
        let notifier = Notifier {
            notices: notify,
            wake: Arc::new(writer),
        };
        let thread_notifier = notifier.clone();
        let (started, result) = mpsc::channel::<io::Result<()>>();
        let thread = std::thread::Builder::new()
            .name("buffer-pool".into())
            .spawn(move || {
                let pools = match image {
                    Some((bytes, format, descriptors)) => {
                        Pools::resume(&bytes, format, descriptors, limits)
                    }
                    None => Ok(BTreeMap::new()),
                };
                let deployments = match pools {
                    Ok(deployments) => {
                        let _ = started.send(Ok(()));
                        deployments
                    }
                    Err(error) => {
                        let _ = started.send(Err(error));
                        return;
                    }
                };
                let pools = Pools {
                    deployments,
                    frozen: false,
                    limits,
                    notifier: thread_notifier,
                    trace: None,
                    check: false,
                    flushed: 0,
                };
                serve(&jobs, pools);
            })?;
        result
            .recv()
            .map_err(|_| io::Error::other("the buffer pool's thread ended at its start"))??;
        Ok(Self {
            work: Some(work),
            notices,
            wake: reader,
            notifier,
            thread: Some(thread),
        })
    }

    /// What a helper sends the main loop its notices with.
    pub(crate) fn notifier(&self) -> Notifier {
        self.notifier.clone()
    }

    /// Runs `work` on the thread, at its next turn.
    pub(crate) fn send(&self, work: impl FnOnce(&mut Pools) + Send + 'static) {
        if let Some(sender) = &self.work {
            let _ = sender.send(Box::new(work));
        }
        if let Some(thread) = &self.thread {
            thread.thread().unpark();
        }
    }

    /// Runs `work` on the thread and returns its result; `None` when the
    /// thread has ended. The thread answers within one of its turns.
    pub(crate) fn call<T: Send + 'static>(
        &self,
        work: impl FnOnce(&mut Pools) -> T + Send + 'static,
    ) -> Option<T> {
        let (sender, receiver) = mpsc::channel();
        self.send(move |pools| {
            let _ = sender.send(work(pools));
        });
        receiver.recv().ok()
    }

    /// The descriptor that is readable when notices wait.
    pub(crate) fn wake(&self) -> BorrowedFd<'_> {
        self.wake.as_fd()
    }

    /// The notices that wait, in the order the thread sent them.
    pub(crate) fn notices(&self) -> Vec<Notice> {
        let mut buffer = [0_u8; 64];
        while matches!(rustix::io::read(&self.wake, &mut buffer), Ok(count) if count > 0) {}
        self.notices.try_iter().collect()
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        // The thread ends at its next turn once no work can arrive.
        self.work = None;
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}

/// The thread's loop: work first, then a turn, then the back-off.
fn serve(jobs: &mpsc::Receiver<Work>, mut pools: Pools) {
    loop {
        loop {
            match jobs.try_recv() {
                Ok(work) => work(&mut pools),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }
        if pools.turn() {
            std::hint::spin_loop();
        } else if pools.idle() {
            // Nothing but work from the main loop can make work, and its
            // send wakes the thread.
            std::thread::park();
        } else {
            std::thread::park_timeout(SLEEP);
        }
    }
}

/// The host's monotonic clock in nanoseconds, the clock of the tasks'
/// deadlines.
pub(crate) fn monotonic() -> u64 {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    (now.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(now.tv_nsec as u64)
}

/// The thread's state: the buffer pool of each deployment with its queue
/// pairs.
pub(crate) struct Pools {
    deployments: BTreeMap<String, Hosted>,
    /// A restart in place is under way: no request is taken and no
    /// `completed` word read.
    frozen: bool,
    /// How long a transfer and a put to the remote store may take.
    limits: (u64, u64),
    notifier: Notifier,
    /// The directory of the pools' traces, while the server records them.
    trace: Option<PathBuf>,
    /// Whether each pool checks its tables after every input.
    check: bool,
    /// When the traces last wrote out their buffers, on the host's
    /// monotonic clock.
    flushed: u64,
}

/// One deployment's buffer pool and the queue pairs of its sessions.
struct Hosted {
    pool: BufferPool,
    watch: Watch,
    queues: BTreeMap<QueueId, Queue>,
    /// The number the next session takes; numbers are never reused.
    next: ClientId,
    /// The source of each prefetch whose task has not started yet.
    sources: HashMap<(QueueId, RequestId), Source>,
    /// Whether the pool changed since its capacity words were written.
    dirty: bool,
}

/// The buffer pool's end of one scheduler's queue pair.
struct Queue {
    pair: QueuePair,
    /// Per group of the scheduler, its id, the register of its `completed`
    /// word and the value last read there.
    completed: Vec<(GroupId, usize, u64)>,
    /// Per region the scheduler uses, its number, the register of its first
    /// capacity word and the words last written there.
    capacity: Vec<(u8, usize, Option<Capacity>)>,
    /// Completions that found no room, as records, in order.
    pending: VecDeque<Vec<u8>>,
    /// The engine broke the ring's indices: the queue is served no more.
    broken: bool,
}

/// The thread's state as a restart in place hands it over: per deployment,
/// its pool `P` (borrowed to write it, owned to read it), the number of its
/// next session and its queue pairs; the queue pairs' memfds travel apart,
/// in this order.
#[derive(Serialize, Deserialize)]
struct Handed<P> {
    deployments: Vec<(String, P, ClientId, Vec<QueueHandoff>)>,
}

/// The pools of a handoff in the earlier image format `format`, whose
/// image of a pool is an `I`, each made as this build holds it.
///
/// # Errors
///
/// A handoff that is not one of that format, or a pool its image does not
/// state.
fn converted<I>(bytes: &[u8], format: u32) -> io::Result<Handed<BufferPool>>
where
    I: for<'de> Deserialize<'de>,
    BufferPool: TryFrom<I, Error = String>,
{
    let handed: Handed<I> = rmp_serde::from_slice(bytes).map_err(io::Error::other)?;
    let mut deployments = Vec::with_capacity(handed.deployments.len());
    for (deployment, image, next, queues) in handed.deployments {
        let pool = BufferPool::try_from(image).map_err(|error| {
            io::Error::other(format!(
                "the buffer pool of {deployment} handed over in format {format}: {error}"
            ))
        })?;
        deployments.push((deployment, pool, next, queues));
    }
    Ok(Handed { deployments })
}

/// A queue pair's state as a restart in place hands it over.
#[derive(Serialize, Deserialize)]
struct QueueHandoff {
    queue: QueueId,
    completed: Vec<(GroupId, u64)>,
    capacity: Vec<(u8, u64)>,
    pending: Vec<Vec<u8>>,
    broken: bool,
}

/// What sees one pool's inputs: its trace, and its check.
struct Watch {
    trace: Option<Trace>,
    /// Whether the pool's check runs (`--buffer-pool-check`).
    check: bool,
    /// The checks that found the pool inconsistent, and whether the last
    /// one did: a failure is logged when the pool turns inconsistent.
    failures: u64,
    failing: bool,
    /// When the last check ended and how long it took. The check costs time
    /// in proportion to the pool's blocks and owners, on the thread whose
    /// answers the schedulers' steps wait for, so it runs after an input
    /// only once the thread has run at least [`CHECK_SPACING`] times that
    /// long since: checking takes at most a tenth of the thread
    /// (`specs/buffer-pool.md` §18). The replay of the trace checks after
    /// every input.
    checked: Option<(Instant, Duration)>,
}

/// The time the thread runs between two checks, in multiples of the
/// previous check's duration.
const CHECK_SPACING: u32 = 9;

impl Watch {
    /// The watch of the pool of `deployment`: a trace in `trace`, from the
    /// pool's image on, and the check when `check` is set. A trace that
    /// cannot be started is logged and left out.
    fn new(deployment: &str, pool: &BufferPool, trace: Option<&Path>, check: bool) -> Self {
        let trace = trace.and_then(|directory| match Trace::start(directory, deployment, pool) {
            Ok(trace) => Some(trace),
            Err(error) => {
                tracing::error!(deployment, %error, "the buffer pool's trace cannot be started");
                None
            }
        });
        Self {
            trace,
            check,
            failures: 0,
            failing: false,
            checked: None,
        }
    }

    /// The pool acted on an input: the trace records it, and the check
    /// runs when it is due ([`Watch::checked`]). `input` is made only when
    /// either is on.
    fn acted(&mut self, deployment: &str, pool: &BufferPool, input: impl FnOnce() -> Input) {
        if self.trace.is_none() && !self.check {
            return;
        }
        let input = input();
        let recorded = self.trace.as_mut().map(|trace| trace.record(&input));
        let position = match recorded {
            Some(Ok(position)) => Some(position),
            Some(Err(error)) => {
                self.stop(deployment, &error);
                None
            }
            None => None,
        };
        if !self.check {
            return;
        }
        if let Some((ended, took)) = self.checked
            && ended.elapsed() < took * CHECK_SPACING
        {
            return;
        }
        let started = Instant::now();
        let found = pool.check();
        let ended = Instant::now();
        self.checked = Some((ended, ended - started));
        match found {
            Ok(()) => self.failing = false,
            Err(inconsistency) => {
                self.failures += 1;
                if !self.failing {
                    tracing::error!(
                        deployment,
                        input = %stated_input(pool, &input),
                        position = ?position,
                        %inconsistency,
                        "the buffer pool is inconsistent after an input; it goes on"
                    );
                }
                self.failing = true;
            }
        }
    }

    /// The pool was told the time: an input when a task passed its
    /// deadline, and otherwise the time the next input is recorded at.
    fn ticked(&mut self, deployment: &str, pool: &BufferPool, now: u64, late: bool) {
        if late {
            self.acted(deployment, pool, || Input::Tick(now));
        } else if let Some(trace) = &mut self.trace {
            trace.clock(now);
        }
    }

    /// The trace keeps the completions `pool` produced.
    fn completed(
        &mut self,
        deployment: &str,
        pool: &BufferPool,
        produced: &[(QueueId, RequestId, Completion)],
    ) {
        let written = self
            .trace
            .as_mut()
            .map(|trace| trace.completed(pool, produced));
        if let Some(Err(error)) = written {
            self.stop(deployment, &error);
        }
    }

    /// The trace keeps a task the pool handed to whoever runs it.
    fn started(&mut self, deployment: &str, task: (QueueId, RequestId)) {
        if let Some(Err(error)) = self
            .trace
            .as_mut()
            .map(|trace| trace.record(&Input::Start(task)))
        {
            self.stop(deployment, &error);
        }
    }

    /// The trace writes out what it buffers.
    fn flush(&mut self, deployment: &str) {
        if let Some(Err(error)) = self.trace.as_mut().map(Trace::flush) {
            self.stop(deployment, &error);
        }
    }

    /// The trace failed: it records no more, and the pool goes on.
    fn stop(&mut self, deployment: &str, error: &io::Error) {
        tracing::error!(deployment, %error, "the buffer pool's trace failed; it records no more");
        self.trace = None;
    }
}

impl Hosted {
    /// A new pool of `deployment`, watched as `trace` and `check` say.
    fn new(deployment: &str, limits: (u64, u64), trace: Option<&Path>, check: bool) -> Self {
        let pool = BufferPool::new(limits);
        Self {
            watch: Watch::new(deployment, &pool, trace, check),
            pool,
            queues: BTreeMap::new(),
            next: 1,
            sources: HashMap::new(),
            dirty: true,
        }
    }

    /// Takes the completions the pool produced into their queue pairs'
    /// pending records, each in its session's version, in order; the trace
    /// keeps them. Returns whether there were any.
    fn deliver(&mut self, deployment: &str) -> bool {
        let completions = self.pool.completions();
        self.watch.completed(deployment, &self.pool, &completions);
        let any = !completions.is_empty();
        for (queue, request, completion) in completions {
            // A closed session's queue pair is gone, and its completions with it.
            let Some(served) = self.queues.get_mut(&queue) else {
                continue;
            };
            let version = self.pool.version(queue.0).unwrap_or(VERSION);
            let mut record = Vec::new();
            completion.encode_as(version, request, &mut record);
            served.pending.push_back(record);
        }
        any
    }

    /// One turn for the deployment; returns whether the thread has work
    /// without sleeping: a request was taken or a `completed` word moved,
    /// or deferred work waits for one.
    fn turn(&mut self, deployment: &str, frozen: bool, now: u64) -> (bool, Vec<Notice>) {
        let mut busy = false;
        if !frozen {
            for (queue, served) in &mut self.queues {
                if served.broken {
                    continue;
                }
                for (group, register, last) in &mut served.completed {
                    let completed = served.pair.register(*register).load(Ordering::Acquire);
                    if completed != *last {
                        *last = completed;
                        busy = true;
                        self.pool.advance(queue.0, *group, completed);
                        let input = || Input::Completed(queue.0, *group, completed);
                        self.watch.acted(deployment, &self.pool, input);
                    }
                }
            }
            let mut record = Vec::new();
            for (queue, served) in &mut self.queues {
                // A session's records are of the version its `Open` stated.
                let version = self.pool.version(queue.0).unwrap_or(VERSION);
                while !served.broken {
                    match served.pair.next_request(&mut record) {
                        Ok(true) => {}
                        Ok(false) => break,
                        Err(error) => {
                            tracing::warn!(deployment, scheduler = queue.1, %error, "an engine broke its submission queue; it is served no more");
                            served.broken = true;
                            break;
                        }
                    }
                    busy = true;
                    match Request::decode_as(&record, version) {
                        Ok((id, request)) => {
                            if let Request::Prefetch { source, .. } = &request {
                                self.sources.insert((*queue, id), *source);
                            }
                            self.pool.take(*queue, id, request);
                        }
                        Err(error) => {
                            tracing::warn!(deployment, scheduler = queue.1, %error, "an engine submitted a record of no request");
                            if let Some(id) = storage::runtime::number(&record) {
                                let refused = Completion::Refused {
                                    cause: Cause::Request,
                                };
                                let mut answer = Vec::new();
                                refused.encode_as(version, id, &mut answer);
                                served.pending.push_back(answer);
                            }
                        }
                    }
                    let input = || Input::Request(*queue, record.clone());
                    self.watch.acted(deployment, &self.pool, input);
                }
            }
        }
        let late = self.pool.tick(now);
        self.watch.ticked(deployment, &self.pool, now, late);

        let delivered = self.deliver(deployment);
        self.dirty |= busy || delivered;
        for (queue, served) in &mut self.queues {
            let version = self.pool.version(queue.0).unwrap_or(VERSION);
            served.flush(deployment, *queue, version);
        }
        if self.dirty {
            self.dirty = false;
            for (queue, served) in &mut self.queues {
                for (region, register, last) in &mut served.capacity {
                    let words = self.pool.capacity(*queue, *region);
                    if *last != Some(words) {
                        let (available, retiring, evictable) = words;
                        for (offset, word) in
                            [available, retiring, evictable].into_iter().enumerate()
                        {
                            let register = served.pair.register(*register + offset);
                            register.store(u64::from(word), Ordering::Release);
                        }
                        *last = Some(words);
                    }
                }
            }
        }

        let mut notices = Vec::new();
        for task in self.pool.started() {
            self.watch.started(deployment, task);
            notices.push(self.start(deployment, task));
        }
        for task in self.pool.stopped() {
            let deployment = deployment.to_owned();
            let stop: Notice = Box::new(move |server: &mut Server| {
                server.stop_pool_task(&deployment, task);
            });
            notices.push(stop);
        }
        (busy || (!frozen && self.pool.deferred()), notices)
    }

    /// The notice that starts a task: the epoch of its session, what it
    /// copies, by the names of regions and memories, and for a prefetch
    /// its source.
    fn start(&mut self, deployment: &str, task: (QueueId, RequestId)) -> Notice {
        let source = self.sources.remove(&task);
        let pool = &self.pool;
        let epoch = pool.epoch(task.0.0).unwrap_or_default();
        let named = |runs: Vec<storage::runtime::Run>| -> Vec<NamedRun> {
            runs.into_iter()
                .map(|(region, first, count)| {
                    let name = pool.region_name(region).unwrap_or_default();
                    (name.to_owned(), first, count)
                })
                .collect()
        };
        // The memories and the host parts of a validity; none for a slot row,
        // whose units exist once.
        let memories = |bits: u64| -> Memories {
            if bits == u64::MAX {
                return (Vec::new(), Vec::new());
            }
            let set = (0..64).filter(|bit| bits & (1 << bit) != 0);
            let names = set.clone().filter_map(|bit| pool.memory_name(bit));
            let parts = set.filter_map(|bit| pool.host_part(bit));
            (
                names.map(str::to_owned).collect(),
                parts.map(|(host, part)| (host.to_owned(), part)).collect(),
            )
        };
        let copies: Vec<Copied> = pool
            .work(task)
            .into_iter()
            .map(|(key, reads, writes, from, to)| {
                (
                    key,
                    named(reads),
                    named(writes),
                    memories(from),
                    memories(to),
                )
            })
            .collect();
        let deployment = deployment.to_owned();
        Box::new(move |server: &mut Server| {
            server.start_pool_task((deployment, epoch), task, source, copies);
        })
    }
}

impl Queue {
    /// Writes the pending completions in order, as far as the completion
    /// queue has room; a refusal is written in the session's `version`.
    fn flush(&mut self, deployment: &str, queue: QueueId, version: u32) {
        while let Some(record) = self.pending.front() {
            if self.broken {
                return;
            }
            if record.len() > self.pair.max_record() {
                tracing::warn!(
                    deployment,
                    scheduler = queue.1,
                    bytes = record.len(),
                    "a completion is longer than its ring holds; its request is refused"
                );
                let Some(id) = storage::runtime::number(record) else {
                    self.pending.pop_front();
                    continue;
                };
                let mut refused = Vec::new();
                Completion::Refused {
                    cause: Cause::Request,
                }
                .encode_as(version, id, &mut refused);
                self.pending[0] = refused;
                continue;
            }
            match self.pair.complete(record) {
                Ok(true) => {
                    self.pending.pop_front();
                }
                Ok(false) => return,
                Err(error) => {
                    tracing::warn!(deployment, scheduler = queue.1, %error, "an engine broke its completion queue; it is served no more");
                    self.broken = true;
                }
            }
        }
    }
}

impl Pools {
    /// One turn over every deployment; returns whether the thread has work
    /// without sleeping.
    fn turn(&mut self) -> bool {
        let now = monotonic();
        let mut busy = false;
        let mut notices = Vec::new();
        for (deployment, hosted) in &mut self.deployments {
            let (working, sent) = hosted.turn(deployment, self.frozen, now);
            busy |= working;
            notices.extend(sent);
        }
        for notice in notices {
            self.notify(notice);
        }
        if !busy || now.saturating_sub(self.flushed) > TRACE_FLUSH {
            self.flushed = now;
            for (deployment, hosted) in &mut self.deployments {
                hosted.watch.flush(deployment);
            }
        }
        busy
    }

    /// From now on, records each pool's trace in `trace`, and checks each
    /// pool's tables after every input when `check` is set; a pool's trace
    /// starts again from its image.
    pub(crate) fn watch(&mut self, trace: Option<PathBuf>, check: bool) {
        for (deployment, hosted) in &mut self.deployments {
            hosted.watch = Watch::new(deployment, &hosted.pool, trace.as_deref(), check);
        }
        self.trace = trace;
        self.check = check;
    }

    /// Whether no session is open and no task outstanding: no queue to
    /// read and no deadline to keep.
    fn idle(&self) -> bool {
        self.deployments
            .values()
            .all(|hosted| hosted.queues.is_empty() && !hosted.pool.tasks())
    }

    /// Sends the main loop a notice and wakes it.
    fn notify(&self, notice: Notice) {
        self.notifier.send(notice);
    }

    /// How long a transfer and a put to the remote store may take from now
    /// on, in nanoseconds, in every pool.
    pub(crate) fn set_limits(&mut self, limits: (u64, u64)) {
        self.limits = limits;
        for (deployment, hosted) in &mut self.deployments {
            hosted.pool.set_limits(limits);
            let input = || Input::Limits(limits);
            hosted.watch.acted(deployment, &hosted.pool, input);
        }
    }

    /// See [`BufferPool::grant_segment`], for the region `region` of the
    /// pool of `deployment`.
    pub(crate) fn grant(
        &mut self,
        deployment: &str,
        region: &str,
        segment: Segment,
        first: u32,
        units: u32,
    ) -> Result<(), String> {
        let hosted = self
            .deployments
            .get_mut(deployment)
            .ok_or_else(|| format!("{deployment} has no buffer pool"))?;
        let number = hosted
            .pool
            .region_number(region)
            .ok_or_else(|| format!("no session of {deployment} stated a region {region}"))?;
        let granted = hosted.pool.grant_segment(number, segment, first, units);
        let input = || Input::Grant(region.to_owned(), segment, first, units);
        hosted.watch.acted(deployment, &hosted.pool, input);
        granted
    }

    /// See [`BufferPool::withdraw_segment`], for the region `region` of the
    /// pool of `deployment`.
    pub(crate) fn withdraw(
        &mut self,
        deployment: &str,
        region: &str,
        segment: Segment,
    ) -> Result<(), String> {
        let hosted = self
            .deployments
            .get_mut(deployment)
            .ok_or_else(|| format!("{deployment} has no buffer pool"))?;
        let number = hosted
            .pool
            .region_number(region)
            .ok_or_else(|| format!("no session of {deployment} stated a region {region}"))?;
        let withdrawn = hosted.pool.withdraw_segment(number, segment);
        let input = || Input::Withdraw(region.to_owned(), segment);
        hosted.watch.acted(deployment, &hosted.pool, input);
        withdrawn
    }

    /// Opens an engine's session with the pool of `deployment`
    /// ([`BufferPool::admit`]), one of the sessions its epoch may have
    /// (an engine and its standby each open one), and a queue pair per
    /// scheduler, whose registers hold the `completed` word of each of its
    /// groups, in the order the scheduler lists them, then the three
    /// capacity words of each region its groups and its classes name, by the
    /// region's number. Returns the session's number in the pool, the
    /// answer and the queue pairs' memfds.
    ///
    /// # Errors
    ///
    /// Refuses, opening nothing, a layout the pool refuses, or queue pairs
    /// that cannot be made.
    pub(crate) fn open(
        &mut self,
        deployment: &str,
        open: &Open,
    ) -> Result<(ClientId, Opened, Vec<OwnedFd>), String> {
        let (limits, trace, check) = (self.limits, self.trace.as_deref(), self.check);
        let hosted = self
            .deployments
            .entry(deployment.to_owned())
            .or_insert_with(|| Hosted::new(deployment, limits, trace, check));
        let client = hosted.next;
        let admitted = hosted.pool.admit(client, open);
        let input = || Input::Open(client, open.clone());
        hosted.watch.acted(deployment, &hosted.pool, input);
        let regions = admitted?;
        hosted.next += 1;
        hosted.dirty = true;

        // The regions every scheduler uses: its groups' and every class's host region.
        let number = |index: u8| regions.get(usize::from(index)).map(|(number, _)| *number);
        let hosts: BTreeSet<u8> = open
            .classes
            .iter()
            .filter_map(|class| class.host.and_then(|(region, _)| number(region)))
            .collect();
        let mut registers = Vec::with_capacity(open.schedulers.len());
        let mut descriptors = Vec::with_capacity(open.schedulers.len());
        let mut queues = Vec::with_capacity(open.schedulers.len());
        for (index, groups) in open.schedulers.iter().enumerate() {
            let mut used = hosts.clone();
            for group in open
                .groups
                .iter()
                .filter(|group| groups.contains(&group.id))
            {
                let named = group.regions.iter().flatten().chain([&group.slots]);
                used.extend(named.filter_map(|region| number(*region)));
            }
            let completed: Vec<u32> = (0..groups.len() as u32).collect();
            let capacity: Vec<(u8, u32)> = used
                .iter()
                .enumerate()
                .map(|(order, region)| (*region, groups.len() as u32 + 3 * order as u32))
                .collect();
            let made =
                QueuePair::create(RING_BYTES, groups.len() + 3 * used.len()).and_then(|pair| {
                    let descriptor = pair.descriptor().try_clone_to_owned()?;
                    Ok((pair, descriptor))
                });
            let (pair, descriptor) = match made {
                Ok(made) => made,
                Err(error) => {
                    hosted.pool.end(client);
                    hosted
                        .watch
                        .acted(deployment, &hosted.pool, || Input::Close(client));
                    return Err(format!("a queue pair cannot be made: {error}"));
                }
            };
            descriptors.push(descriptor);
            queues.push((
                (client, index as u16),
                Queue {
                    pair,
                    completed: groups
                        .iter()
                        .zip(&completed)
                        .map(|(group, register)| (*group, *register as usize, 0))
                        .collect(),
                    capacity: capacity
                        .iter()
                        .map(|(region, register)| (*region, *register as usize, None))
                        .collect(),
                    pending: VecDeque::new(),
                    broken: false,
                },
            ));
            registers.push(Registers {
                completed,
                capacity,
            });
        }
        hosted.queues.extend(queues);
        let segments = regions
            .iter()
            .map(|(number, _)| hosted.pool.segments_of(*number))
            .collect();
        let limits = regions
            .iter()
            .map(|(number, _)| hosted.pool.limit_of(*number))
            .collect();
        tracing::info!(
            deployment,
            epoch = open.epoch,
            trial = open.trial,
            session = client,
            schedulers = open.schedulers.len(),
            "an engine opened its session with the buffer pool"
        );
        Ok((
            client,
            Opened {
                regions,
                segments,
                limits,
                registers,
            },
            descriptors,
        ))
    }

    /// Ends the session `client` of the pool of `deployment` and drops its
    /// queue pairs ([`BufferPool::end`]). Returns whether it was open.
    pub(crate) fn close(&mut self, deployment: &str, client: ClientId) -> bool {
        let Some(hosted) = self.deployments.get_mut(deployment) else {
            return false;
        };
        let Some(epoch) = hosted.pool.epoch(client) else {
            return false;
        };
        // What the pool produced goes out first: the session's own
        // completions leave with its queue pairs, the others' are kept.
        hosted.deliver(deployment);
        hosted.pool.end(client);
        hosted
            .watch
            .acted(deployment, &hosted.pool, || Input::Close(client));
        hosted.queues.retain(|queue, _| queue.0 != client);
        hosted.sources.retain(|(queue, _), _| queue.0 != client);
        hosted.dirty = true;
        tracing::info!(
            deployment,
            epoch,
            session = client,
            "a session with the buffer pool closed"
        );
        true
    }

    /// The update of an epoch on trial is committed
    /// ([`BufferPool::commit_update`]).
    pub(crate) fn commit_update(&mut self, deployment: &str, epoch: u64) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.commit_update(epoch);
            hosted
                .watch
                .acted(deployment, &hosted.pool, || Input::Publish(epoch));
            hosted.dirty = true;
        }
    }

    /// The update of an epoch on trial is rolled back, its session closed
    /// ([`BufferPool::roll_back`]).
    pub(crate) fn roll_back(&mut self, deployment: &str, epoch: u64) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.roll_back(epoch);
            hosted
                .watch
                .acted(deployment, &hosted.pool, || Input::RollBack(epoch));
            hosted.dirty = true;
        }
    }

    /// Drops the pool of a deployment that is no longer served, with every
    /// session and queue pair; what its tasks wrote is gone with it.
    pub(crate) fn release(&mut self, deployment: &str) {
        // The trace keeps what the pool produced before it is gone.
        if let Some(mut hosted) = self.deployments.remove(deployment) {
            hosted.deliver(deployment);
        }
    }

    /// The next key of a task is done ([`BufferPool::done`]).
    pub(crate) fn done(&mut self, deployment: &str, task: (QueueId, RequestId), tokens: u16) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.done(task, tokens);
            hosted
                .watch
                .acted(deployment, &hosted.pool, || Input::Done(task, tokens));
            hosted.dirty = true;
        }
    }

    /// A task ended and touches its blocks no more ([`BufferPool::ended`]).
    pub(crate) fn ended(&mut self, deployment: &str, task: (QueueId, RequestId), ok: bool) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.ended(task, ok);
            hosted
                .watch
                .acted(deployment, &hosted.pool, || Input::Ended(task, ok));
            hosted.dirty = true;
        }
    }

    /// Holds the host copies of `keys` for `peer` ([`BufferPool::hold`]).
    /// Returns the copies held, by the name of their region, each with the
    /// host parts that hold it: none for a copy of groups that state no host
    /// parts, which every part of its region holds.
    pub(crate) fn hold(
        &mut self,
        deployment: &str,
        peer: u64,
        keys: &[Key],
    ) -> Vec<(HostCopy, Vec<HostPart>)> {
        let Some(hosted) = self.deployments.get_mut(deployment) else {
            return Vec::new();
        };
        hosted.dirty = true;
        let held = hosted.pool.hold(peer, keys);
        let input = || Input::Hold(peer, keys.to_vec());
        hosted.watch.acted(deployment, &hosted.pool, input);

        let pool = &hosted.pool;
        let parts = |key: &Key| -> Vec<HostPart> {
            let bits = pool.held_on(key);
            (0..64)
                .filter(|bit| bits & (1 << bit) != 0)
                .filter_map(|bit| pool.host_part(bit))
                .map(|(host, part)| (host.to_owned(), part))
                .collect()
        };
        held.into_iter()
            .map(|(key, (region, first, count), tokens)| {
                let copy = HostCopy {
                    key,
                    tokens,
                    region: pool.region_name(region).unwrap_or_default().to_owned(),
                    first,
                    count,
                };
                (copy, parts(&key))
            })
            .collect()
    }

    /// Ends the holds of `peer` in `deployment`'s pool. Returns whether it
    /// held anything there.
    pub(crate) fn release_peer(&mut self, deployment: &str, peer: u64) -> bool {
        let Some(hosted) = self.deployments.get_mut(deployment) else {
            return false;
        };
        hosted.dirty = true;
        let released = hosted.pool.release_peer(peer);
        let input = || Input::ReleasePeer(peer);
        hosted.watch.acted(deployment, &hosted.pool, input);
        released
    }

    /// The peer's connection closed: its holds in every pool end.
    pub(crate) fn peer_gone(&mut self, peer: u64) {
        for (deployment, hosted) in &mut self.deployments {
            hosted.dirty |= hosted.pool.release_peer(peer);
            let input = || Input::ReleasePeer(peer);
            hosted.watch.acted(deployment, &hosted.pool, input);
        }
    }

    /// Every pool as an operator's listing states it.
    pub(crate) fn listing(&self) -> Vec<BufferPoolListing> {
        let listed = |(deployment, hosted): (&String, &Hosted)| BufferPoolListing {
            check_failures: hosted.watch.failures,
            ..hosted.pool.listing(deployment)
        };
        self.deployments.iter().map(listed).collect()
    }

    /// A restart in place is under way: no request is taken and no
    /// `completed` word read from now on.
    pub(crate) fn freeze(&mut self) {
        self.frozen = true;
    }

    /// The restart in place under way is given up: requests are taken and
    /// `completed` words read again.
    pub(crate) fn thaw(&mut self) {
        self.frozen = false;
    }

    /// The versions of the runtime interface the open sessions speak, which
    /// a successor of a restart in place must speak.
    pub(crate) fn versions(&self) -> BTreeSet<u32> {
        let pools = self.deployments.values();
        pools.flat_map(|hosted| hosted.pool.versions()).collect()
    }

    /// Whether the thread is frozen and no task of any pool is outstanding:
    /// its state may pass to a successor.
    pub(crate) fn quiet(&self) -> bool {
        self.frozen && self.deployments.values().all(|hosted| !hosted.pool.tasks())
    }

    /// The thread's whole state for a successor: the bytes of every pool's
    /// image with its queue pairs, and the queue pairs' memfds in their
    /// order. The thread is frozen first, and holds nothing afterwards, so
    /// that it writes nothing the successor writes again.
    ///
    /// # Errors
    ///
    /// Fails when a pool cannot be written or a memfd duplicated.
    pub(crate) fn hand_off(&mut self) -> io::Result<(Vec<u8>, Vec<OwnedFd>)> {
        self.frozen = true;
        // The completions the pools produced go into the queue pairs'
        // pending records, which travel in order, and the traces keep them.
        for (deployment, hosted) in &mut self.deployments {
            hosted.deliver(deployment);
        }
        let mut descriptors = Vec::new();
        let mut deployments = Vec::with_capacity(self.deployments.len());
        for (deployment, hosted) in &self.deployments {
            let mut queues = Vec::with_capacity(hosted.queues.len());
            for (queue, served) in &hosted.queues {
                descriptors.push(served.pair.descriptor().try_clone_to_owned()?);
                queues.push(QueueHandoff {
                    queue: *queue,
                    completed: served
                        .completed
                        .iter()
                        .map(|(group, register, _)| (*group, *register as u64))
                        .collect(),
                    capacity: served
                        .capacity
                        .iter()
                        .map(|(region, register, _)| (*region, *register as u64))
                        .collect(),
                    pending: served.pending.iter().cloned().collect(),
                    broken: served.broken,
                });
            }
            deployments.push((deployment.clone(), &hosted.pool, hosted.next, queues));
        }
        let bytes = rmp_serde::to_vec(&Handed { deployments }).map_err(io::Error::other)?;
        self.deployments.clear();
        Ok((bytes, descriptors))
    }

    /// The pools a predecessor handed over in the image format `format`,
    /// this build's, format 3, format 2 or format 1, each checked, with their queue
    /// pairs, which go on from the indices the predecessor published.
    fn resume(
        bytes: &[u8],
        format: u32,
        descriptors: Vec<OwnedFd>,
        limits: (u64, u64),
    ) -> io::Result<BTreeMap<String, Hosted>> {
        let handed: Handed<BufferPool> = match format {
            FORMAT => rmp_serde::from_slice(bytes).map_err(io::Error::other)?,
            format3::FORMAT => converted::<format3::Image>(bytes, format)?,
            format2::FORMAT => converted::<format2::Image>(bytes, format)?,
            format1::FORMAT => converted::<format1::Image>(bytes, format)?,
            other => {
                return Err(io::Error::other(format!(
                    "the buffer pools are handed over in image format {other}; this build reads \
                     formats {}, {}, {} and {FORMAT}",
                    format1::FORMAT,
                    format2::FORMAT,
                    format3::FORMAT
                )));
            }
        };
        let mut descriptors = descriptors.into_iter();
        let mut deployments = BTreeMap::new();
        for (deployment, pool, next, queues) in handed.deployments {
            pool.check().map_err(|inconsistency| {
                io::Error::other(format!(
                    "the buffer pool of {deployment} handed over is inconsistent: {inconsistency}"
                ))
            })?;
            // The server's options start a watch of the pool once it runs.
            let mut hosted = Hosted::new(&deployment, limits, None, false);
            hosted.pool = pool;
            hosted.next = next;
            for queue in queues {
                let descriptor = descriptors.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "a queue pair's memfd is missing",
                    )
                })?;
                let pair = QueuePair::open(descriptor, false).map_err(io::Error::other)?;
                hosted.queues.insert(
                    queue.queue,
                    Queue {
                        pair,
                        completed: queue
                            .completed
                            .into_iter()
                            .map(|(group, register)| (group, register as usize, 0))
                            .collect(),
                        capacity: queue
                            .capacity
                            .into_iter()
                            .map(|(region, register)| (region, register as usize, None))
                            .collect(),
                        pending: queue.pending.into(),
                        broken: queue.broken,
                    },
                );
            }
            deployments.insert(deployment, hosted);
        }
        Ok(deployments)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, mpsc};
    use std::time::{Duration, Instant};

    use storage::QueuePair;
    use storage::protocol::Key;
    use storage::runtime::{
        Class, Completion, Group, GroupId, Open, OwnerId, Region, Request, RequestId, Tier, VERSION,
    };

    use super::{Notifier, Pools};

    /// The thread's state with the check on, serving the session of a
    /// layout of one group over four KV units and two slot rows, and the
    /// engine's end of its queue pair.
    fn served() -> (Pools, QueuePair) {
        served_with(4, true)
    }

    /// As [`served`], with `units` KV units, and the check on when `check`.
    fn served_with(units: u32, check: bool) -> (Pools, QueuePair) {
        let (_, wake) = rustix::pipe::pipe().unwrap();
        let notifier = Notifier {
            notices: mpsc::channel().0,
            wake: Arc::new(wake),
        };
        let mut pools = Pools {
            deployments: BTreeMap::new(),
            frozen: false,
            limits: (1_000, 1_000),
            notifier,
            trace: None,
            check,
            flushed: 0,
        };
        let region = |name: &str, units| Region {
            name: name.into(),
            tier: Tier::Device,
            space: name.into(),
            unit_bytes: 64,
            part_bytes: 64,
            reserved: 1,
            units,
            limit: units + 1,
            segments: Vec::new(),
            memory: String::new(),
        };
        let open = Open {
            version: VERSION,
            epoch: 0,
            trial: false,
            regions: vec![region("units", units), region("rows", 2)],
            classes: vec![Class {
                units: 1,
                host: None,
                state_in_slot: false,
                tensors: [0; 32],
                encoding: [0; 32],
            }],
            trajectory: 0,
            groups: vec![Group {
                id: GroupId(0),
                memory: "gpu0".into(),
                slots: 1,
                regions: vec![vec![0]],
                receives: Vec::new(),
                parts: Vec::new(),
            }],
            schedulers: vec![vec![GroupId(0)]],
            peers: Vec::new(),
        };
        let (_, _, descriptors) = pools.open("model", &open).unwrap();
        let engine = QueuePair::open(descriptors.into_iter().next().unwrap(), true).unwrap();
        (pools, engine)
    }

    fn submit(engine: &mut QueuePair, number: u64, request: &Request) {
        let mut record = Vec::new();
        request.encode(RequestId(number), &mut record);
        assert!(engine.submit(&record).unwrap());
    }

    fn failures(pools: &Pools) -> u64 {
        pools.listing()[0].check_failures
    }

    /// With the check on, a pool whose tables a fault made inconsistent is
    /// reported by the first check after an input, and goes on serving.
    #[test]
    fn a_pool_made_inconsistent_is_reported_by_the_next_check_and_goes_on() {
        let (mut pools, mut engine) = served();
        let attach = Request::Attach {
            owner: OwnerId(1),
            group: GroupId(0),
        };
        submit(&mut engine, 1, &attach);
        pools.turn();
        assert_eq!(failures(&pools), 0);

        // The bytes of the region's first unit join the free quota of its
        // memory space while the region holds them: two places hold them
        // (R10). That is no input, and nothing is reported yet.
        let pool = &mut pools.deployments.get_mut("model").unwrap().pool;
        pool.grant(0, 64, 64);
        pools.turn();
        assert_eq!(failures(&pools), 0);

        // The check of four units takes microseconds, so after a pause the
        // thread has run far longer than nine checks since the last one and
        // the next input is checked.
        std::thread::sleep(Duration::from_millis(5));
        let allocate = Request::Allocate {
            owner: OwnerId(1),
            class: 0,
            at: 0,
            count: 1,
            units: 0,
        };
        submit(&mut engine, 2, &allocate);
        pools.turn();
        assert_eq!(failures(&pools), 1);

        let mut answers = Vec::new();
        let mut record = Vec::new();
        while engine.next_completion(&mut record).unwrap() {
            answers.push(Completion::decode(&record).unwrap().1);
        }
        assert!(
            matches!(
                answers[..],
                [Completion::Attached { .. }, Completion::Allocated { .. }]
            ),
            "{answers:?}"
        );
    }

    /// Serves `requests` in turns of at most 64, numbering them on from
    /// `number` (a queue pair's numbers rise), and returns the time the
    /// turns took and the completions, in order.
    fn serve(
        pools: &mut Pools,
        engine: &mut QueuePair,
        number: &mut u64,
        requests: &[Request],
    ) -> (Duration, Vec<Completion>) {
        let mut took = Duration::ZERO;
        let mut answers = Vec::new();
        let mut record = Vec::new();
        for chunk in requests.chunks(64) {
            for request in chunk {
                *number += 1;
                submit(engine, *number, request);
            }
            let started = Instant::now();
            pools.turn();
            took += started.elapsed();
            while engine.next_completion(&mut record).unwrap() {
                answers.push(Completion::decode(&record).unwrap().1);
            }
        }
        (took, answers)
    }

    /// The live check costs time in proportion to the pool's blocks, on the
    /// thread whose answers the steps wait for, and takes at most a tenth of
    /// that thread: a pool of 3000 cached blocks, whose check takes
    /// hundreds of times as long as an allocation, serves allocations with
    /// the check on in about the time it takes with the check off (checked
    /// after every input, it would take tens of times as long).
    #[test]
    fn the_live_check_takes_at_most_a_tenth_of_the_thread() {
        const BLOCKS: u32 = 3000;
        let named = |index: u32| {
            let mut bytes = [0; 32];
            bytes[..4].copy_from_slice(&index.to_le_bytes());
            Key(bytes)
        };

        // One owner commits 3000 blocks; one unit stays free, which the
        // timed requests allocate and release in turn.
        let mut filling = vec![
            Request::Attach {
                owner: OwnerId(1),
                group: GroupId(0),
            },
            Request::Allocate {
                owner: OwnerId(1),
                class: 0,
                at: 0,
                count: BLOCKS,
                units: 0,
            },
        ];
        filling.extend((0..BLOCKS).map(|at| Request::Commit {
            owner: OwnerId(1),
            class: 0,
            at,
            key: named(at),
            tokens: 64,
            after: Vec::new(),
        }));
        let cycles: Vec<Request> = (0..20_000)
            .flat_map(|_| {
                [
                    Request::Allocate {
                        owner: OwnerId(1),
                        class: 0,
                        at: BLOCKS,
                        count: 1,
                        units: 0,
                    },
                    Request::Release {
                        owner: OwnerId(1),
                        class: 0,
                        at: BLOCKS,
                        count: 1,
                        after: Vec::new(),
                    },
                ]
            })
            .collect();

        let mut times = [Duration::MAX; 2];
        for _ in 0..3 {
            for (check, time) in [false, true].into_iter().zip(&mut times) {
                let (mut pools, mut engine) = served_with(BLOCKS + 1, check);
                let mut number = 0;
                let (_, filled) = serve(&mut pools, &mut engine, &mut number, &filling);
                let committed = filled
                    .iter()
                    .filter(|answer| matches!(answer, Completion::Committed))
                    .count();
                assert_eq!(committed, BLOCKS as usize, "{:?}", &filled[..2]);

                let (took, served) = serve(&mut pools, &mut engine, &mut number, &cycles);
                assert!(
                    served
                        .iter()
                        .step_by(2)
                        .all(|answer| matches!(answer, Completion::Allocated { .. })),
                    "{:?}",
                    &served[..2]
                );
                *time = (*time).min(took);
                assert_eq!(failures(&pools), 0);
            }
        }
        let [off, on] = times;
        assert!(
            on < off * 3 / 2,
            "40000 allocations and releases took {off:?} without the check and {on:?} with it"
        );
    }
}

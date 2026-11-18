//! The buffer pool's thread: it serves every session's queue pairs.
//!
//! The buffer pools of the host's deployments, one per deployment, live on
//! one thread of their own, which serves their queue pairs
//! (`specs/buffer-pool.md` §9, §10). The thread reads the submission queues
//! and the `completed` registers in shared memory and turns while any of
//! them gave it something; once a turn found nothing it sleeps on the
//! doorbells of its queue pairs (`storage::queue`), the wake of the main
//! loop's work and the earliest task deadline, having declared on each queue
//! pair that it waits for requests and `completed` words (and for room in
//! the completion queue while completions wait for it). An engine's step
//! therefore makes a system call for the thread only when the thread sleeps,
//! and the thread makes one for a scheduler only when that scheduler sleeps.
//! It moves no bytes and waits on nothing inside a handler.
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
//!   a send fires the thread's wake, which ends its sleep.
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
//! when they changed, ringing the scheduler's doorbell if it waits for them;
//! and send the tasks to start and to stop. A completion
//! longer than a ring holds is answered `Refused { cause: Request }`: only a
//! `List` of a cache too large for the ring comes to that, and it changed
//! nothing. A queue pair whose indices the engine broke is served no more.
//!
//! The thread acts on its inputs at one place per input, and there each
//! pool's [`Watch`] sees them: when the server's options say so, a trace of
//! every input and completion ([`Trace`], `specs/buffer-pool.md` §18), whose
//! replay checks the pool's tables after every input.
//!
//! A restart in place freezes the thread: it takes no request and reads no
//! `completed` word, so no work starts, while the tasks run to their end
//! and their completions are written. Once no task is outstanding the
//! thread's whole state, every pool's image with its queue pairs, passes to
//! the successor ([`Pools::hand_off`], [`Thread::start`]); the queue pairs
//! keep their indices, and requests submitted meanwhile wait in the rings.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;

use serde::{Deserialize, Serialize};
use service::{Wake, Waker};
use storage::protocol::{BufferPoolListing, HostCopy, Key, Segment};
use storage::runtime::{
    Cause, Completion, GroupId, Open, Opened, Registers, Request, RequestId, Source,
};
use storage::{Interest, QueueDescriptors, QueuePair, Signals};

use crate::Server;
use crate::buffer_pool::trace::{Input, Trace};
use crate::buffer_pool::{BufferPool, ClientId, Encoding, QueueId, Validity};

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
    /// Ends the thread's sleep once work was sent.
    awake: Wake,
    notices: mpsc::Receiver<Notice>,
    /// The read end of the pipe a notice wakes the main loop through.
    wake: OwnedFd,
    notifier: Notifier,
    thread: Option<JoinHandle<()>>,
}

impl Thread {
    /// Starts the thread: with no pool, or going on with the state a
    /// predecessor handed over (`image`, the bytes of [`Pools::hand_off`] in
    /// this build's image format, with the queue pairs' descriptors in its
    /// order). `limits` is how long a transfer and a put to the remote store
    /// may take, in nanoseconds.
    ///
    /// # Errors
    ///
    /// Fails when the pipe or the thread cannot be made, or the state
    /// handed over cannot be made again.
    pub(crate) fn start(
        limits: (u64, u64),
        image: Option<(Vec<u8>, Vec<OwnedFd>)>,
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
        let waker = Waker::new()?;
        let awake = waker.wake();
        let (started, result) = mpsc::channel::<io::Result<()>>();
        let thread = std::thread::Builder::new()
            .name("buffer-pool".into())
            .spawn(move || {
                let pools = match image {
                    Some((bytes, descriptors)) => Pools::resume(&bytes, descriptors, limits),
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
                };
                serve(&jobs, waker, pools);
            })?;
        result
            .recv()
            .map_err(|_| io::Error::other("the buffer pool's thread ended at its start"))??;
        Ok(Self {
            work: Some(work),
            awake,
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
        self.awake.wake();
    }

    /// Runs `work` on the thread and returns its result; `None` when the
    /// thread has ended. The thread answers within one of its turns.
    pub(crate) fn call<T: Send + 'static>(
        &self,
        work: impl FnOnce(&mut Pools) -> T + Send + 'static,
    ) -> io::Result<T> {
        let (sender, receiver) = mpsc::channel();
        self.send(move |pools| {
            let _ = sender.send(work(pools));
        });
        // The answer's channel closes unanswered only when the thread ended.
        receiver
            .recv()
            .map_err(|_| io::Error::other("the buffer pool's thread ended"))
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
        self.awake.wake();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The thread's loop: work first, then a turn; after a turn that found
/// nothing, the sleep ([`Pools::sleep`]), which `waker` ends when the main
/// loop sends work.
fn serve(jobs: &mpsc::Receiver<Work>, mut waker: Waker, mut pools: Pools) {
    loop {
        loop {
            match jobs.try_recv() {
                Ok(work) => work(&mut pools),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }
        if !pools.turn() {
            pools.sleep(&mut waker);
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
    /// What the engine had published to this end when the last turn looked
    /// at the pair, which the sleep compares with ([`QueuePair::arm`]).
    seen: Signals,
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
/// next session and its queue pairs; the queue pairs' descriptors
/// ([`QueueDescriptors`]) travel apart, in this order.
#[derive(Serialize, Deserialize)]
struct Handed<P> {
    deployments: Vec<(String, P, ClientId, Vec<QueueHandoff>)>,
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

/// What sees one pool's inputs: its trace, which the replay checks after every input
/// (`buffer_pool::trace::replay`).
struct Watch {
    trace: Option<Trace>,
}

impl Watch {
    /// The watch of the pool of `deployment`: a trace in `trace`, from the
    /// pool's image on. A trace that cannot be started is logged and left out.
    fn new(deployment: &str, pool: &BufferPool, trace: Option<&Path>) -> Self {
        let trace = trace.and_then(|directory| match Trace::start(directory, deployment, pool) {
            Ok(trace) => Some(trace),
            Err(error) => {
                tracing::error!(deployment, %error, "the buffer pool's trace cannot be started");
                None
            }
        });
        Self { trace }
    }

    /// The pool acted on an input: the trace records it. `input` is made only while the pool is
    /// traced.
    fn acted(&mut self, deployment: &str, input: impl FnOnce() -> Input) {
        let Some(trace) = &mut self.trace else {
            return;
        };
        if let Err(error) = trace.record(&input()) {
            self.stop(deployment, &error);
        }
    }

    /// The pool was told the time: an input when a task passed its
    /// deadline, and otherwise the time the next input is recorded at.
    fn ticked(&mut self, deployment: &str, now: u64, late: bool) {
        if late {
            self.acted(deployment, || Input::Tick(now));
        } else if let Some(trace) = &mut self.trace {
            trace.clock(now);
        }
    }

    /// The trace keeps the completions the pool produced.
    fn completed(&mut self, deployment: &str, produced: &[(QueueId, RequestId, Completion)]) {
        let written = self.trace.as_mut().map(|trace| trace.completed(produced));
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
    fn new(deployment: &str, limits: (u64, u64), trace: Option<&Path>) -> Self {
        let pool = BufferPool::new(limits);
        Self {
            watch: Watch::new(deployment, &pool, trace),
            pool,
            queues: BTreeMap::new(),
            next: 1,
            sources: HashMap::new(),
            dirty: true,
        }
    }

    /// Takes the completions the pool produced into their queue pairs'
    /// pending records, in order; the trace keeps them. Returns whether
    /// there were any.
    fn deliver(&mut self, deployment: &str) -> bool {
        let completions = self.pool.completions();
        self.watch.completed(deployment, &completions);
        let any = !completions.is_empty();
        for (queue, request, completion) in completions {
            // A closed session's queue pair is gone, and its completions with it.
            let Some(served) = self.queues.get_mut(&queue) else {
                continue;
            };
            let mut record = Vec::new();
            completion.encode(request, &mut record);
            served.pending.push_back(record);
        }
        any
    }

    /// One turn for the deployment; returns whether the thread has work
    /// without sleeping: a request was taken or a `completed` word moved.
    /// Deferred work waits for an input that comes with a wake: a
    /// `completed` word, which rings the doorbell, or a task's end, which
    /// is work from the main loop.
    fn turn(&mut self, deployment: &str, frozen: bool, now: u64) -> (bool, Vec<Notice>) {
        // What the engines published before this look: a publication after
        // it keeps the sleep that follows from starting.
        for served in self.queues.values_mut() {
            served.seen = served.pair.signals();
        }
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
                        self.watch.acted(deployment, input);
                    }
                }
            }
            let mut record = Vec::new();
            for (queue, served) in &mut self.queues {
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
                    match Request::decode(&record) {
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
                                refused.encode(id, &mut answer);
                                served.pending.push_back(answer);
                            }
                        }
                    }
                    let input = || Input::Request(*queue, record.clone());
                    self.watch.acted(deployment, input);
                }
            }
        }
        let late = self.pool.tick(now);
        self.watch.ticked(deployment, now, late);

        let delivered = self.deliver(deployment);
        self.dirty |= busy || delivered;
        for (queue, served) in &mut self.queues {
            served.flush(deployment, *queue);
        }
        if self.dirty {
            self.dirty = false;
            for (queue, served) in &mut self.queues {
                let mut written = false;
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
                        written = true;
                    }
                }
                // A scheduler asleep while a request waits for units wakes to
                // read them.
                if written {
                    served.pair.notify_registers();
                }
            }
        }

        let mut notices = Vec::new();
        for task in self.pool.started() {
            self.watch.started(deployment, task);
            notices.extend(self.start(deployment, task));
        }
        for task in self.pool.stopped() {
            let deployment = deployment.to_owned();
            let stop: Notice = Box::new(move |server: &mut Server| {
                server.stop_pool_task(&deployment, task);
            });
            notices.push(stop);
        }
        (busy, notices)
    }

    /// The notice that starts a task: the epoch of its session, what it
    /// copies, by the names of regions and memories, and for a prefetch
    /// its source. None for a task whose session closed before it started, which ended with it.
    fn start(&mut self, deployment: &str, task: (QueueId, RequestId)) -> Option<Notice> {
        let source = self.sources.remove(&task);
        let pool = &self.pool;
        let epoch = pool.epoch(task.0.0)?;
        if let Some(work) = pool.import_work(task) {
            let deployment = deployment.to_owned();
            return Some(Box::new(move |server: &mut Server| {
                server.start_import((deployment, epoch), task, work);
            }));
        }
        let named = |runs: Vec<storage::runtime::Run>| -> Vec<NamedRun> {
            runs.into_iter()
                .map(|(region, first, count)| (pool.region_name(region).to_owned(), first, count))
                .collect()
        };
        // The memories and the host parts of a validity; none for a slot row, whose units exist
        // once.
        let memories = |valid: Validity| -> Memories {
            let (names, parts) = pool.memory_names(valid);
            (
                names.into_iter().map(str::to_owned).collect(),
                parts
                    .into_iter()
                    .map(|(host, part)| (host.to_owned(), part))
                    .collect(),
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
                    memories(Validity::In(to)),
                )
            })
            .collect();
        let deployment = deployment.to_owned();
        Some(Box::new(move |server: &mut Server| {
            server.start_pool_task((deployment, epoch), task, source, copies);
        }))
    }
}

impl Queue {
    /// Writes the pending completions in order, as far as the completion
    /// queue has room.
    fn flush(&mut self, deployment: &str, queue: QueueId) {
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
                .encode(id, &mut refused);
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
        // A trace writes out as its buffers fill, and the rest once the thread has nothing to do,
        // so a turn that answers steps writes no more than a full buffer.
        if !busy {
            for (deployment, hosted) in &mut self.deployments {
                hosted.watch.flush(deployment);
            }
        }
        busy
    }

    /// Bounds the tasks every pool starts from now on by `limits` (`BufferPool::set_limits`),
    /// asked while no trace records, which would hold the pools' limits only from an image.
    pub(crate) fn limit(&mut self, limits: (u64, u64)) {
        self.limits = limits;
        for hosted in self.deployments.values_mut() {
            hosted.pool.set_limits(limits);
        }
    }

    /// From now on, records each pool's trace in `trace`; a pool's trace starts again from its
    /// image.
    pub(crate) fn watch(&mut self, trace: Option<PathBuf>) {
        for (deployment, hosted) in &mut self.deployments {
            hosted.watch = Watch::new(deployment, &hosted.pool, trace.as_deref());
        }
        self.trace = trace;
    }

    /// Sleeps after a turn that found nothing, until an engine publishes
    /// what the thread waits for on one of its queue pairs, the main loop
    /// sends work (`waker`), or the earliest task deadline of any pool
    /// passes.
    ///
    /// On every queue pair it serves the thread declares that it waits for
    /// requests and `completed` words (none while frozen, when it takes
    /// neither) and, while completions wait for room, for the room the
    /// scheduler makes ([`QueuePair::arm`]). An engine that published such a
    /// thing since the turn looked keeps the thread from sleeping: the
    /// declarations are withdrawn and the thread turns again. Once awake the
    /// thread withdraws every declaration, so that an engine makes no system
    /// call for it while it is busy, and drains the doorbells.
    fn sleep(&mut self, waker: &mut Waker) {
        let frozen = self.frozen;
        let queues = || {
            self.deployments
                .values()
                .flat_map(|hosted| hosted.queues.values())
                .filter(|served| !served.broken)
        };

        let mut armed = true;
        for served in queues() {
            let mut interest = Interest::empty();
            if !frozen {
                interest |= Interest::RECORDS | Interest::REGISTERS;
            }
            if !served.pending.is_empty() {
                interest |= Interest::ROOM;
            }
            if !served.pair.arm(interest, served.seen) {
                armed = false;
                break;
            }
        }

        if armed {
            let mut descriptors = vec![waker.poll_fd()];
            descriptors.extend(queues().map(|served| libc::pollfd {
                fd: served.pair.doorbell().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            }));
            // Until the earliest deadline, rounded up to whole milliseconds so
            // the poll does not return before it; without one, until a wake.
            let timeout = self.next_deadline().map_or(-1, |deadline| {
                let left = deadline.saturating_sub(monotonic());
                libc::c_int::try_from(left.div_ceil(1_000_000)).unwrap_or(libc::c_int::MAX)
            });
            // SAFETY: poll borrows this live array only for the call. A
            // failure (a signal) ends the sleep like a wake: the next turn
            // looks at everything.
            unsafe {
                libc::poll(
                    descriptors.as_mut_ptr(),
                    descriptors.len() as libc::nfds_t,
                    timeout,
                )
            };
        }

        for served in queues() {
            served.pair.disarm();
            served.pair.drain();
        }
        waker.drain();
    }

    /// The earliest deadline of a task of any pool, on the host's monotonic
    /// clock: when the next tick may stop one.
    fn next_deadline(&self) -> Option<u64> {
        self.deployments
            .values()
            .filter_map(|hosted| hosted.pool.next_deadline())
            .min()
    }

    /// Sends the main loop a notice and wakes it.
    fn notify(&self, notice: Notice) {
        self.notifier.send(notice);
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
        hosted.watch.acted(deployment, input);
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
        hosted.watch.acted(deployment, input);
        withdrawn
    }

    /// Opens an engine's session with the pool of `deployment`
    /// ([`BufferPool::admit`]), one of the sessions its epoch may have
    /// (an engine and its standby each open one), and a queue pair per
    /// scheduler, whose registers hold the `completed` word of each of its
    /// groups, in the order the scheduler lists them, then the three
    /// capacity words of each region its groups and its classes name, by the
    /// region's number. Returns the session's number in the pool, the
    /// answer and the queue pairs' descriptors ([`QueueDescriptors`]), in
    /// order.
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
        let (limits, trace) = (self.limits, self.trace.as_deref());
        let hosted = self
            .deployments
            .entry(deployment.to_owned())
            .or_insert_with(|| Hosted::new(deployment, limits, trace));
        let client = hosted.next;
        let admitted = hosted.pool.admit(client, open);
        let input = || Input::Open(client, open.clone());
        hosted.watch.acted(deployment, input);
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
        let mut descriptors = Vec::with_capacity(open.schedulers.len() * QueueDescriptors::COUNT);
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
                    let descriptors = pair.duplicate()?;
                    Ok((pair, descriptors))
                });
            let (pair, pair_descriptors) = match made {
                Ok(made) => made,
                Err(error) => {
                    hosted.pool.end(client);
                    hosted.watch.acted(deployment, || Input::Close(client));
                    return Err(format!("a queue pair cannot be made: {error}"));
                }
            };
            descriptors.extend(pair_descriptors.into_array());
            queues.push((
                (client, index as u16),
                Queue {
                    pair,
                    seen: Signals::default(),
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
                session: client,
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
        hosted.watch.acted(deployment, || Input::Close(client));
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
            hosted.watch.acted(deployment, || Input::Publish(epoch));
            hosted.dirty = true;
        }
    }

    /// The update of an epoch on trial is rolled back, its session closed
    /// ([`BufferPool::roll_back`]).
    pub(crate) fn roll_back(&mut self, deployment: &str, epoch: u64) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.roll_back(epoch);
            hosted.watch.acted(deployment, || Input::RollBack(epoch));
            hosted.dirty = true;
        }
    }

    /// The next key of a task is done ([`BufferPool::done`]).
    pub(crate) fn done(&mut self, deployment: &str, task: (QueueId, RequestId), tokens: u16) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.done(task, tokens);
            hosted.watch.acted(deployment, || Input::Done(task, tokens));
            hosted.dirty = true;
        }
    }

    /// A task ended and touches its blocks no more ([`BufferPool::ended`]).
    pub(crate) fn ended(&mut self, deployment: &str, task: (QueueId, RequestId), ok: bool) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.ended(task, ok);
            hosted.watch.acted(deployment, || Input::Ended(task, ok));
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
        hosted.watch.acted(deployment, input);

        let pool = &hosted.pool;
        let parts = |key: &Key| -> Vec<HostPart> {
            let held = pool.held_on(key).into_iter();
            held.map(|(host, part)| (host.to_owned(), part)).collect()
        };
        held.into_iter()
            .map(
                |(key, (region, first, count), tokens, (tensors, encoding))| {
                    let copy = HostCopy {
                        key,
                        tokens,
                        region: pool.region_name(region).to_owned(),
                        first,
                        count,
                        tensors,
                        encoding,
                    };
                    (copy, parts(&key))
                },
            )
            .collect()
    }

    /// What the peers hold of the keys of `deployment`'s prefetch `task`:
    /// per copy, in order, the encoding of the peer's host copy it reads
    /// ([`BufferPool::peer_copies`]). Returns the runs the leading copies
    /// write, by the names of their regions.
    pub(crate) fn peer_copies(
        &mut self,
        deployment: &str,
        task: (QueueId, RequestId),
        held: &[Option<Encoding>],
    ) -> Vec<NamedRun> {
        let Some(hosted) = self.deployments.get_mut(deployment) else {
            return Vec::new();
        };
        hosted.dirty = true;
        let runs = hosted.pool.peer_copies(task, held);
        let input = || Input::PeerCopies(task, held.to_vec());
        hosted.watch.acted(deployment, input);

        let pool = &hosted.pool;
        runs.into_iter()
            .map(|(region, first, count)| (pool.region_name(region).to_owned(), first, count))
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
        hosted.watch.acted(deployment, input);
        released
    }

    /// The peer's connection closed: its holds and its borrows in every pool end.
    pub(crate) fn peer_gone(&mut self, peer: u64) {
        for (deployment, hosted) in &mut self.deployments {
            hosted.dirty |= hosted.pool.release_peer(peer);
            let input = || Input::ReleasePeer(peer);
            hosted.watch.acted(deployment, input);
            hosted.dirty |= hosted.pool.release_borrows(peer, None);
            let input = || Input::ReleaseBorrows(peer, None);
            hosted.watch.acted(deployment, input);
        }
    }

    /// The blocks of the lend of `session`'s owner `owner` in `deployment`'s pool, held for
    /// `peer` ([`BufferPool::borrow`]).
    pub(crate) fn borrow(
        &mut self,
        deployment: &str,
        peer: u64,
        session: ClientId,
        owner: storage::runtime::OwnerId,
    ) -> Option<crate::buffer_pool::LentBlocks> {
        let hosted = self.deployments.get_mut(deployment)?;
        hosted.dirty = true;
        let lent = hosted.pool.borrow(peer, session, owner);
        hosted
            .watch
            .acted(deployment, || Input::Borrow(peer, session, owner));
        lent
    }

    /// The borrows of `peer` in `deployment`'s pool end: of one lend, or every one
    /// ([`BufferPool::release_borrows`]).
    pub(crate) fn release_borrows(
        &mut self,
        deployment: &str,
        peer: u64,
        lend: Option<(ClientId, storage::runtime::OwnerId)>,
    ) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.dirty |= hosted.pool.release_borrows(peer, lend);
            hosted
                .watch
                .acted(deployment, || Input::ReleaseBorrows(peer, lend));
        }
    }

    /// The lend of `session`'s owner `owner` in `deployment`'s pool ends
    /// ([`BufferPool::give_back`]).
    pub(crate) fn give_back(
        &mut self,
        deployment: &str,
        session: ClientId,
        owner: storage::runtime::OwnerId,
    ) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.dirty |= hosted.pool.give_back(session, owner);
            hosted
                .watch
                .acted(deployment, || Input::GiveBack(session, owner));
        }
    }

    /// What `deployment`'s pool holds on each tier ([`BufferPool::holdings`]); nothing for a
    /// deployment without a pool here.
    pub(crate) fn holdings(
        &self,
        deployment: &str,
    ) -> Vec<(storage::runtime::Tier, Vec<Key>, u64)> {
        self.deployments
            .get(deployment)
            .map(|hosted| hosted.pool.holdings())
            .unwrap_or_default()
    }

    /// Every pool as an operator's listing states it.
    pub(crate) fn listing(&self) -> Vec<BufferPoolListing> {
        let listed = |(deployment, hosted): (&String, &Hosted)| BufferPoolListing {
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

    /// Whether the thread is frozen and no task of any pool is outstanding:
    /// its state may pass to a successor.
    pub(crate) fn quiet(&self) -> bool {
        self.frozen && self.deployments.values().all(|hosted| !hosted.pool.tasks())
    }

    /// The thread's whole state for a successor: the bytes of every pool's
    /// image with its queue pairs, and the queue pairs' descriptors in their
    /// order. The thread is frozen first, and holds nothing afterwards, so
    /// that it writes nothing the successor writes again.
    ///
    /// # Errors
    ///
    /// Fails when a pool cannot be written or a descriptor duplicated.
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
                let pair = served.pair.duplicate().map_err(io::Error::other)?;
                descriptors.extend(pair.into_array());
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

    /// The pools a predecessor handed over in this build's image format,
    /// each checked, with their queue pairs, which go on from the indices
    /// the predecessor published.
    fn resume(
        bytes: &[u8],
        descriptors: Vec<OwnedFd>,
        limits: (u64, u64),
    ) -> io::Result<BTreeMap<String, Hosted>> {
        let handed: Handed<BufferPool> = rmp_serde::from_slice(bytes).map_err(io::Error::other)?;
        let mut descriptors = descriptors.into_iter();
        let mut deployments = BTreeMap::new();
        for (deployment, mut pool, next, queues) in handed.deployments {
            pool.check().map_err(|inconsistency| {
                io::Error::other(format!(
                    "the buffer pool of {deployment} handed over is inconsistent: {inconsistency}"
                ))
            })?;
            // The successor's settings bound the tasks that start from now on, and the trace its
            // options start once it runs records them in the pool's image.
            pool.set_limits(limits);
            // The server's options start a watch of the pool once it runs.
            let mut hosted = Hosted::new(&deployment, limits, None);
            hosted.pool = pool;
            hosted.next = next;
            for queue in queues {
                let handed = QueueDescriptors::take(&mut descriptors).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "a queue pair's descriptors are missing",
                    )
                })?;
                let pair = QueuePair::open(handed, false).map_err(io::Error::other)?;
                hosted.queues.insert(
                    queue.queue,
                    Queue {
                        pair,
                        seen: Signals::default(),
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

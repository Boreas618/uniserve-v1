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
//!   the main loop owns.
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
//! A restart in place freezes the thread: it takes no request and reads no
//! `completed` word, so no work starts, while the tasks run to their end
//! and their completions are written. Once no task is outstanding the
//! thread's whole state, every pool's image with its queue pairs, passes to
//! the successor ([`Pools::hand_off`], [`Thread::start`]); the queue pairs
//! keep their indices, and requests submitted meanwhile wait in the rings.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use storage::QueuePair;
use storage::protocol::{BufferPoolListing, HostCopy, Key};
use storage::runtime::{
    Cause, Completion, GroupId, Open, Opened, Registers, Request, RequestId, Source,
};

use crate::Server;
use crate::buffer_pool::{BufferPool, ClientId, QueueId};

/// How long the thread sleeps when every queue is empty and nothing is
/// deferred: the longest a request of an idle engine waits to be taken.
pub(crate) const SLEEP: Duration = Duration::from_micros(200);

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

/// One copy of a task, as the main loop runs it: the key it is for if any,
/// the runs it reads and writes, the device memories that hold the content
/// it reads, and the device memory it writes into when it writes one
/// ([`BufferPool::work`]).
pub(crate) type Copied = (
    Option<Key>,
    Vec<NamedRun>,
    Vec<NamedRun>,
    Vec<String>,
    Option<String>,
);

/// The three capacity words of a region for a queue pair: `available`,
/// `retiring`, `evictable` ([`BufferPool::capacity`]).
type Capacity = (u32, u32, u32);

/// The main loop's end of the thread.
pub(crate) struct Thread {
    work: Option<mpsc::Sender<Work>>,
    notices: mpsc::Receiver<Notice>,
    /// The read end of the pipe a notice wakes the main loop through.
    wake: OwnedFd,
    thread: Option<JoinHandle<()>>,
}

impl Thread {
    /// Starts the thread: with no pool, or going on with the state a
    /// predecessor handed over (`image`, the bytes of [`Pools::hand_off`],
    /// with the queue pairs' memfds in its order). `limits` is how long a
    /// transfer and a put to the remote store may take, in nanoseconds.
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
                    notices: notify,
                    wake: writer,
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
            thread: Some(thread),
        })
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
fn monotonic() -> u64 {
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
    notices: mpsc::Sender<Notice>,
    /// The write end of the main loop's wake pipe.
    wake: OwnedFd,
}

/// One deployment's buffer pool and the queue pairs of its sessions.
struct Hosted {
    pool: BufferPool,
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

/// A queue pair's state as a restart in place hands it over.
#[derive(Serialize, Deserialize)]
struct QueueHandoff {
    queue: QueueId,
    completed: Vec<(GroupId, u64)>,
    capacity: Vec<(u8, u64)>,
    pending: Vec<Vec<u8>>,
    broken: bool,
}

impl Hosted {
    fn new(limits: (u64, u64)) -> Self {
        Self {
            pool: BufferPool::new(limits),
            queues: BTreeMap::new(),
            next: 1,
            sources: HashMap::new(),
            dirty: true,
        }
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
                }
            }
        }
        self.pool.tick(now);

        let completions = self.pool.completions();
        self.dirty |= busy || !completions.is_empty();
        for (queue, request, completion) in completions {
            // A closed session's queue pair is gone, and its completions with it.
            let Some(served) = self.queues.get_mut(&queue) else {
                continue;
            };
            let mut record = Vec::new();
            completion.encode(request, &mut record);
            served.pending.push_back(record);
        }
        for (queue, served) in &mut self.queues {
            served.flush(deployment, *queue);
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

    /// The notice that starts a task: what it copies, by the names of
    /// regions and memories, and for a prefetch its source.
    fn start(&mut self, deployment: &str, task: (QueueId, RequestId)) -> Notice {
        let source = self.sources.remove(&task);
        let pool = &self.pool;
        let named = |runs: Vec<storage::runtime::Run>| -> Vec<NamedRun> {
            runs.into_iter()
                .map(|(region, first, count)| {
                    let name = pool.region_name(region).unwrap_or_default();
                    (name.to_owned(), first, count)
                })
                .collect()
        };
        // The memories of a validity; none for a block whose units exist once.
        let memories = |bits: u8| -> Vec<String> {
            if bits == u8::MAX {
                return Vec::new();
            }
            (0..8)
                .filter(|bit| bits & (1 << bit) != 0)
                .filter_map(|bit| pool.memory_name(bit).map(str::to_owned))
                .collect()
        };
        let copies: Vec<Copied> = pool
            .work(task)
            .into_iter()
            .map(|(key, reads, writes, from, to)| {
                let to = memories(to).into_iter().next();
                (key, named(reads), named(writes), memories(from), to)
            })
            .collect();
        let deployment = deployment.to_owned();
        Box::new(move |server: &mut Server| {
            server.start_pool_task(deployment, task, source, copies);
        })
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
        busy
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
        if self.notices.send(notice).is_ok() {
            let _ = rustix::io::write(&self.wake, b"n");
        }
    }

    /// Opens an engine epoch's session with the pool of `deployment`
    /// ([`BufferPool::admit`]) and a queue pair per scheduler, whose
    /// registers hold the `completed` word of each of its groups, in the
    /// order the scheduler lists them, then the three capacity words of each
    /// region its groups and its classes name, by the region's number.
    /// Returns the answer and the queue pairs' memfds.
    ///
    /// # Errors
    ///
    /// Refuses, opening nothing, a layout the pool refuses, an epoch whose
    /// session is open, or queue pairs that cannot be made.
    pub(crate) fn open(
        &mut self,
        deployment: &str,
        open: &Open,
    ) -> Result<(Opened, Vec<OwnedFd>), String> {
        let limits = self.limits;
        let hosted = self
            .deployments
            .entry(deployment.to_owned())
            .or_insert_with(|| Hosted::new(limits));
        if hosted.pool.session(open.epoch).is_some() {
            return Err(format!("the session of epoch {} is open", open.epoch));
        }
        let client = hosted.next;
        let regions = hosted.pool.admit(client, open)?;
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
        tracing::info!(
            deployment,
            epoch = open.epoch,
            trial = open.trial,
            session = client,
            schedulers = open.schedulers.len(),
            "an engine epoch opened its session with the buffer pool"
        );
        Ok((Opened { regions, registers }, descriptors))
    }

    /// Ends the session of `epoch` of `deployment` and drops its queue
    /// pairs ([`BufferPool::end`]). Returns whether it was open.
    pub(crate) fn close(&mut self, deployment: &str, epoch: u64) -> bool {
        let Some(hosted) = self.deployments.get_mut(deployment) else {
            return false;
        };
        let Some(client) = hosted.pool.session(epoch) else {
            return false;
        };
        hosted.pool.end(client);
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
            hosted.dirty = true;
        }
    }

    /// The update of an epoch on trial is rolled back, its session closed
    /// ([`BufferPool::roll_back`]).
    pub(crate) fn roll_back(&mut self, deployment: &str, epoch: u64) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.roll_back(epoch);
            hosted.dirty = true;
        }
    }

    /// Drops the pool of a deployment that is no longer served, with every
    /// session and queue pair; what its tasks wrote is gone with it.
    pub(crate) fn release(&mut self, deployment: &str) {
        self.deployments.remove(deployment);
    }

    /// The next key of a task is done ([`BufferPool::done`]).
    pub(crate) fn done(&mut self, deployment: &str, task: (QueueId, RequestId), tokens: u16) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.done(task, tokens);
            hosted.dirty = true;
        }
    }

    /// A task ended and touches its blocks no more ([`BufferPool::ended`]).
    pub(crate) fn ended(&mut self, deployment: &str, task: (QueueId, RequestId), ok: bool) {
        if let Some(hosted) = self.deployments.get_mut(deployment) {
            hosted.pool.ended(task, ok);
            hosted.dirty = true;
        }
    }

    /// Holds the host copies of `keys` for `peer` ([`BufferPool::hold`]).
    /// Returns the copies held, by the name of their region.
    pub(crate) fn hold(&mut self, deployment: &str, peer: u64, keys: &[Key]) -> Vec<HostCopy> {
        let Some(hosted) = self.deployments.get_mut(deployment) else {
            return Vec::new();
        };
        hosted.dirty = true;
        let held = hosted.pool.hold(peer, keys);
        held.into_iter()
            .map(|(key, (region, first, count))| HostCopy {
                key,
                region: hosted
                    .pool
                    .region_name(region)
                    .unwrap_or_default()
                    .to_owned(),
                first,
                count,
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
        hosted.pool.release_peer(peer)
    }

    /// The peer's connection closed: its holds in every pool end.
    pub(crate) fn peer_gone(&mut self, peer: u64) {
        for hosted in self.deployments.values_mut() {
            hosted.dirty |= hosted.pool.release_peer(peer);
        }
    }

    /// Every pool as an operator's listing states it.
    pub(crate) fn listing(&self) -> Vec<BufferPoolListing> {
        self.deployments
            .iter()
            .map(|(deployment, hosted)| hosted.pool.listing(deployment))
            .collect()
    }

    /// A restart in place is under way: no request is taken and no
    /// `completed` word read from now on.
    pub(crate) fn freeze(&mut self) {
        self.frozen = true;
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

    /// The pools a predecessor handed over, each checked, with their queue
    /// pairs, which go on from the indices the predecessor published.
    fn resume(
        bytes: &[u8],
        descriptors: Vec<OwnedFd>,
        limits: (u64, u64),
    ) -> io::Result<BTreeMap<String, Hosted>> {
        let handed: Handed<BufferPool> = rmp_serde::from_slice(bytes).map_err(io::Error::other)?;
        let mut descriptors = descriptors.into_iter();
        let mut deployments = BTreeMap::new();
        for (deployment, pool, next, queues) in handed.deployments {
            pool.check().map_err(|inconsistency| {
                io::Error::other(format!(
                    "the buffer pool of {deployment} handed over is inconsistent: {inconsistency}"
                ))
            })?;
            let mut hosted = Hosted::new(limits);
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

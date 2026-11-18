//! The copy threads that run copies: one thread per link and copy runner, its
//! queue of pieces by priority, and its link's byte budget.
//!
//! A task's moves become pieces, each queued on the copy thread of its link and of
//! the GPU whose copy engine runs it (or the CPU, between host buffers, or the
//! NIXL agent, for a READ from another host). A copy thread repeatedly takes one
//! chunk off the front piece of its highest priority that has one, waits until its
//! link's budget covers the chunk, and copies it. Every runner copies through one
//! loop ([`Copier`]): it submits a chunk, keeps its link's chunks in flight, and
//! counts a chunk copied once the host saw it complete.
//!
//! - **Chunks.** A chunk lasts the configured chunk time at the link's
//!   bandwidth, as the host's calibration states it, and at least the link's
//!   fixed cost of one copy, in whole pages ([`Sizing`]): a copy engine is never
//!   held by one copy much longer than that, far below a scheduler step of the
//!   epoch executing beside the transfers, and a task of a higher priority starts
//!   on the next chunk.
//! - **Budgets.** A link may move its budget's share of its bandwidth: a token
//!   bucket refilled at that rate, holding at most the chunks the thread keeps in
//!   flight, and charged before each chunk. The share follows [`Throttle`]'s
//!   policy: whole at first, halved by a head down to [`Throttle::FLOOR`] while its
//!   scheduler steps slow down, and grown back by [`Throttle::RESTORE_PER_SECOND`]
//!   of the bandwidth per second after its restoration. A chunk waits for its
//!   budget on a timed wait that a throttle change cuts short, since the change
//!   alters the refill rate, and so does the end of its task, which abandons it.
//! - **Completion.** A GPU copy thread keeps its link's chunks in flight on its
//!   own stream, so its copy engine never waits for the host, and counts a chunk
//!   copied once the blocking event recorded after it completed. A network copy
//!   thread keeps its link's NIXL READs posted and counts a chunk copied once its
//!   READ completed, as the host observes it: a READ passes through no CUDA
//!   stream. A task ends when its last chunk completed: complete, cancelled or
//!   failed. A cancelled or failed task's queued pieces leave at once, and it
//!   ends once its submitted chunks completed, since only then are its regions
//!   free of copies.
//! - **Links from other hosts.** The calibration states the rate of copies from
//!   another host into each end of this host, over the NVLink fabric and by READ;
//!   the first copy thread of such a link reports the link at that rate, so the
//!   listing names the links tasks copied over. A link the calibration does not
//!   state fails the chunks queued for it.
//! - **Threads and contexts.** A copy thread runs while its queue has
//!   chunks queued or in flight, and ends when it has none; the next piece
//!   queued for its link and runner starts another. A GPU copy thread holds its GPU
//!   ([`Hold`]) and creates its stream in the GPU's context, and every piece
//!   holds the GPUs whose storage or context its ends need, so a GPU's
//!   context is released once its last task's last chunk completed and its
//!   copy threads ended ([`super::contexts`]).
//!
//! A device fault (a failed copy submission or completion) leaves the CUDA
//! context unusable, so the process ends: the Storage Server fails every
//! task it held and starts a new Transfer Server. A failed READ (another
//! host or the network failed) fails its task alone.

use std::collections::{HashMap, HashSet, VecDeque};
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ::storage::protocol::{
    Link, LinkBandwidth, LinkEnd, Priority, TaskFailure, TaskStatus, Throttle,
};
use platform::{
    DeviceId, Event, EventFlags, Pitched, PitchedBatch, SourceOrder, Stream, memcpy_async,
};

use super::calibration::{Calibration, Cost, RemotePath, Sizing};
use super::contexts::{Contexts, Hold};
use super::mooncake;
use super::network::{Network, Reach, Source};
use super::nixl;
use super::storage::{Fabric, Mapping};
use super::storage_name;

/// What runs a copy thread's copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Runner {
    /// The copy engines of this GPU, on a stream of the copy thread's.
    Device(DeviceId),
    /// The CPU, between host buffers.
    Host,
    /// This process's NIXL agent or Transfer Engine, with READs from and
    /// WRITEs into another host through the network adapters.
    Network,
}

/// A copy thread: the link it moves bytes over, what runs its copies, and,
/// for a link from or to another host's storage, the path its copies take.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct ThreadKey {
    pub(super) link: Link,
    pub(super) runner: Runner,
    /// `None` for a link within the host.
    pub(super) path: Option<RemotePath>,
}

/// One end of a piece: an allocation the server gave, another host's
/// allocation imported on the copy thread's GPU, or another host's storage that a
/// READ reaches. Each holds the GPU whose context reaches it, declared after
/// what lives in that context, so that it is dropped last.
#[derive(Clone)]
pub(super) enum End {
    /// An allocation the server gave, and the hold on the GPU whose session
    /// maps or registers it: a device allocation's own GPU; for host
    /// storage, the GPU whose copy engines run the copy, and none when the
    /// CPU copies it or a READ writes it.
    Given {
        mapped: Arc<Mapping>,
        hold: Option<Hold>,
    },
    /// Another host's allocation of the fabric, imported on the GPU that
    /// runs the copy, which `_hold` holds.
    Fabric {
        import: Arc<Fabric>,
        _hold: Hold,
    },
    Remote(Arc<Source>),
}

impl End {
    /// The address of byte 0 as `runner`'s copies use it.
    fn address(&self, runner: Runner) -> Result<u64, String> {
        match (self, runner) {
            (End::Given { mapped, hold: None }, Runner::Host) => mapped.host_address(),
            (
                End::Given {
                    mapped,
                    hold: Some(hold),
                },
                Runner::Device(_),
            ) => {
                let session = hold.session()?;
                match mapped.end {
                    LinkEnd::Device { .. } => session.device_address(mapped),
                    LinkEnd::Host => session.host_address(mapped),
                    LinkEnd::Remote { .. } => {
                        Err("the server gives allocations of this host only".into())
                    }
                }
            }
            (End::Fabric { import, .. }, Runner::Device(_)) => import.address(),
            (End::Remote(_), _) | (_, Runner::Network) => {
                Err("another host's storage is read by a READ, not copied by address".into())
            }
            _ => Err("a copy's runner does not reach one of its ends".into()),
        }
    }
}

/// Bytes of one move still to copy, or one chunk of them.
#[derive(Clone)]
pub(super) struct Piece {
    pub(super) task: u64,
    pub(super) source: End,
    pub(super) source_offset: u64,
    pub(super) destination: End,
    pub(super) destination_offset: u64,
    /// The bytes of each row.
    pub(super) bytes: u64,
    /// Rows at the pitches, row `r` from `source_offset + r *
    /// source_pitch` to `destination_offset + r * destination_pitch`; one
    /// row is a plain copy.
    pub(super) rows: u64,
    pub(super) source_pitch: u64,
    pub(super) destination_pitch: u64,
}

impl Piece {
    /// The bytes the piece copies.
    fn copied(&self) -> u64 {
        self.bytes * self.rows
    }
}

/// A task as the copy threads run it.
struct TaskState {
    /// Pieces queued and not taken whole.
    pieces: usize,
    /// Chunks submitted, or waiting for their budget, and not completed.
    running: usize,
    copied: u64,
    /// How the task ends, once a cancellation or a failure decided it.
    end: Option<TaskStatus>,
    reported: Instant,
    /// When the task's pieces were queued, and how many there were: the
    /// task's time on the copy threads, which its end reports.
    queued: Instant,
    queued_pieces: usize,
    /// The links its pieces copy over, which a halving of its tasks names.
    links: HashSet<Link>,
}

/// A link's byte budget: a share of its bandwidth, and the token bucket
/// that holds the link to it.
struct Budget {
    share: f64,
    /// When a restoration started, and the share then.
    restoring: Option<(Instant, f64)>,
    /// Bytes the link may move before it waits; negative while a chunk
    /// charged ahead of its rate.
    tokens: f64,
    refilled: Instant,
}

impl Budget {
    fn new(now: Instant) -> Self {
        Self {
            share: 1.0,
            restoring: None,
            tokens: f64::INFINITY,
            refilled: now,
        }
    }

    fn share(&self, now: Instant) -> f64 {
        match self.restoring {
            Some((since, from)) => (from
                + Throttle::RESTORE_PER_SECOND * now.duration_since(since).as_secs_f64())
            .min(1.0),
            None => self.share,
        }
    }

    fn change(&mut self, throttle: Throttle, now: Instant) {
        let share = self.share(now);
        match throttle {
            Throttle::Halve => {
                self.share = (share / 2.0).max(Throttle::FLOOR);
                self.restoring = None;
            }
            Throttle::Restore => {
                self.share = share;
                if self.restoring.is_none() && share < 1.0 {
                    self.restoring = Some((now, share));
                }
            }
        }
    }

    /// Add the tokens the link earned since the last refill, at its share of
    /// `bandwidth` bytes per second, up to `capacity` bytes.
    fn refill(&mut self, bandwidth: f64, capacity: f64, now: Instant) {
        let rate = self.share(now) * bandwidth;
        let elapsed = now.duration_since(self.refilled).as_secs_f64();
        self.refilled = now;
        self.tokens = (self.tokens + rate * elapsed).min(capacity);
    }

    /// Charge `bytes` of a link of `bandwidth`, whose bucket holds `capacity` bytes.
    fn charge(&mut self, bytes: u64, bandwidth: f64, capacity: f64, now: Instant) {
        self.refill(bandwidth, capacity, now);
        self.tokens -= bytes as f64;
    }

    /// How long until the bytes charged are covered, at the link's share now.
    fn delay(&mut self, bandwidth: f64, capacity: f64, now: Instant) -> Duration {
        self.refill(bandwidth, capacity, now);
        let rate = self.share(now) * bandwidth;
        if self.tokens >= 0.0 || rate <= 0.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(-self.tokens / rate)
        }
    }

    /// Give back `bytes` a chunk charged and did not copy.
    fn refund(&mut self, bytes: u64) {
        self.tokens += bytes as f64;
    }
}

/// The pieces of one copy thread, by priority.
#[derive(Default)]
struct Queue {
    levels: [VecDeque<Piece>; Priority::COUNT],
}

/// What the copy threads share with the main thread.
struct State {
    tasks: HashMap<u64, TaskState>,
    queues: HashMap<ThreadKey, Queue>,
    /// The copy threads whose thread runs.
    running: HashSet<ThreadKey>,
    budgets: HashMap<Link, Budget>,
    /// Each report: the task, the bytes it copied, and its end once it
    /// ended.
    reports: Vec<(u64, u64, Option<TaskStatus>)>,
    /// Links from other hosts copy threads started on, at their calibrated
    /// bandwidths, not reported yet.
    remote_links: Vec<LinkBandwidth>,
    /// Links from other hosts reported so far.
    reported_links: HashSet<Link>,
}

/// What a copy thread does next.
enum Next {
    /// Copy this chunk.
    Chunk(Piece),
    /// Nothing is queued: complete the oldest chunk in flight.
    Busy,
    /// Nothing is queued or in flight: the thread ends.
    Idle,
}

impl State {
    /// One chunk of at most `chunk` bytes off the front piece of the
    /// highest priority `copy thread` has.
    fn take(&mut self, thread: &ThreadKey, chunk: u64) -> Option<Piece> {
        let queue = self.queues.get_mut(thread)?;
        let level = queue.levels.iter_mut().find(|level| !level.is_empty())?;
        let front = level.front_mut()?;
        let mut taken = front.clone();
        let whole = if front.rows > 1 {
            // Rows at a pitch are taken whole, as many as fit the chunk,
            // and at least one.
            let rows = (chunk / front.bytes).clamp(1, front.rows);
            taken.rows = rows;
            front.rows -= rows;
            front.source_offset += rows * front.source_pitch;
            front.destination_offset += rows * front.destination_pitch;
            front.rows == 0
        } else {
            let bytes = front.bytes.min(chunk);
            taken.bytes = bytes;
            front.bytes -= bytes;
            front.source_offset += bytes;
            front.destination_offset += bytes;
            front.bytes == 0
        };
        if whole {
            level.pop_front();
        }
        if let Some(task) = self.tasks.get_mut(&taken.task) {
            if whole {
                task.pieces -= 1;
            }
            task.running += 1;
        }
        Some(taken)
    }

    /// Take every queued piece of `task` out of its queue. The caller drops
    /// them once it released the lock: the last holds of a GPU close its
    /// session as they drop.
    fn purge(&mut self, task: u64) -> Vec<Piece> {
        let mut purged = Vec::new();
        for queue in self.queues.values_mut() {
            for level in &mut queue.levels {
                let (kept, removed) = std::mem::take(level)
                    .into_iter()
                    .partition(|piece| piece.task != task);
                *level = kept;
                purged.extend::<VecDeque<Piece>>(removed);
            }
        }
        if let Some(state) = self.tasks.get_mut(&task) {
            state.pieces = 0;
        }
        purged
    }

    /// Whether `task` ended or its end is decided: a chunk of it waiting for
    /// its budget is not copied.
    fn ended(&self, task: u64) -> bool {
        self.tasks
            .get(&task)
            .is_none_or(|state| state.end.is_some())
    }

    /// Report `task` ended when nothing of it is queued or running, or its
    /// progress when it last reported at least `progress` ago. Returns
    /// whether a report was made.
    fn report(&mut self, task: u64, now: Instant, progress: Duration) -> bool {
        let Some(state) = self.tasks.get_mut(&task) else {
            return false;
        };
        if state.pieces == 0 && state.running == 0 {
            let end = state.end.take().unwrap_or(TaskStatus::Complete);
            let copied = state.copied;
            tracing::info!(
                task,
                pieces = state.queued_pieces,
                bytes = copied,
                queued_ms = now.duration_since(state.queued).as_millis() as u64,
                "a transfer task left the threads"
            );
            self.tasks.remove(&task);
            self.reports.push((task, copied, Some(end)));
            return true;
        }
        if now.duration_since(state.reported) >= progress {
            state.reported = now;
            self.reports.push((task, state.copied, None));
            return true;
        }
        false
    }

    /// A chunk of `task` completed, copying `bytes`, or failed. Returns
    /// whether a report was made, and the pieces a failure purged.
    fn complete(
        &mut self,
        task: u64,
        bytes: u64,
        failure: Option<String>,
        now: Instant,
        progress: Duration,
    ) -> (bool, Vec<Piece>) {
        let Some(state) = self.tasks.get_mut(&task) else {
            return (false, Vec::new());
        };
        state.running -= 1;
        let purged = match failure {
            None => {
                state.copied += bytes;
                Vec::new()
            }
            Some(message) => {
                if state.end.is_none() {
                    state.end = Some(TaskStatus::Failed {
                        failure: TaskFailure::Copy,
                        message,
                    });
                }
                self.purge(task)
            }
        };
        (self.report(task, now, progress), purged)
    }

    /// A chunk of `task` that waited for its budget is not copied: its task
    /// ended meanwhile. Returns whether a report was made.
    fn abandon(&mut self, task: u64, now: Instant, progress: Duration) -> bool {
        if let Some(state) = self.tasks.get_mut(&task) {
            state.running -= 1;
        }
        self.report(task, now, progress)
    }
}

/// The state behind its lock, the main thread's wake-up, and what every copy
/// thread sizes its chunks by.
struct Shared {
    state: Mutex<State>,
    /// Signalled when a budget's rate changes or a task's end is decided, which
    /// a chunk waiting for its budget takes up at once.
    paced: Condvar,
    /// The write end of the main thread's wake pipe, written when a report
    /// is made.
    wake: OwnedFd,
    calibration: Arc<Calibration>,
    /// Each GPU's UUID, by the device that runs its copy threads.
    gpus: HashMap<DeviceId, [u8; 16]>,
    /// The time a chunk lasts at its link's bandwidth, in nanoseconds: the server may change it
    /// while threads run ([`CopyThreads::pace`]), and each thread sizes its next chunk by the
    /// value it reads then.
    chunk: AtomicU64,
    /// How often a running task reports its progress at most, in nanoseconds, read at each
    /// report.
    progress: AtomicU64,
    /// The host's page size, which chunks are whole multiples of.
    page: u64,
}

/// `duration` in whole nanoseconds, the longest a `u64` holds at most.
fn nanoseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        ::service::lock(&self.state)
    }

    /// The time a chunk lasts at its link's bandwidth now.
    fn chunk(&self) -> Duration {
        Duration::from_nanos(self.chunk.load(Ordering::Relaxed))
    }

    /// How often a running task reports its progress at most now.
    fn progress(&self) -> Duration {
        Duration::from_nanos(self.progress.load(Ordering::Relaxed))
    }

    fn wake_main(&self) {
        let _ = rustix::io::write(&self.wake, b"x");
    }

    /// The cost of `thread`'s link, as the host's calibration states it: a
    /// link within the host; or copies from another host along the thread's
    /// path, into the link's destination for a READ and through the runner's
    /// GPU over the fabric, which is reported the first time a thread starts
    /// on it.
    fn cost(&self, thread: &ThreadKey) -> Result<Cost, String> {
        let link = &thread.link;
        let Some(path) = thread.path else {
            return self.calibration.link(link).ok_or_else(|| {
                format!(
                    "the host's calibration states no link from {} to {}",
                    storage_name(&link.source),
                    storage_name(&link.destination)
                )
            });
        };
        let end = match (path, thread.runner) {
            // A WRITE's local end is its source, whose RDMA rate the READs into it measure.
            (RemotePath::Read | RemotePath::Mooncake, _)
                if matches!(link.destination, LinkEnd::Remote { .. }) =>
            {
                link.source.clone()
            }
            (RemotePath::Read | RemotePath::Mooncake, _) => link.destination.clone(),
            (RemotePath::Fabric, Runner::Device(device)) => {
                let gpu = self
                    .gpus
                    .get(&device)
                    .ok_or_else(|| format!("{device} is not calibrated"))?;
                LinkEnd::Device { gpu: *gpu }
            }
            (RemotePath::Fabric, Runner::Host | Runner::Network) => {
                return Err(
                    "a GPU of this host copies another host's storage of the fabric".into(),
                );
            }
        };
        let cost = self.calibration.remote(path, &end).ok_or_else(|| {
            format!(
                "the host's calibration states no rate of copies from another host ({path:?}) \
                 into {}",
                storage_name(&end)
            )
        })?;
        let mut state = self.lock();
        if state.reported_links.insert(link.clone()) {
            tracing::info!(
                ?link,
                gigabytes_per_second = cost.bytes_per_second as f64 / 1e9,
                "copies over a link from another host start"
            );
            state.remote_links.push(LinkBandwidth {
                link: link.clone(),
                bytes_per_second: cost.bytes_per_second,
            });
            drop(state);
            self.wake_main();
        }
        Ok(cost)
    }

    /// What the copy thread `thread` does next: copy a chunk of at most
    /// `chunk` bytes; complete a chunk in flight when `busy` and nothing is
    /// queued; or end. A thread that ends leaves under the lock that saw its
    /// queue empty, so a piece queued after it starts a new thread.
    fn next(&self, thread: &ThreadKey, chunk: u64, busy: bool) -> Next {
        let mut state = self.lock();
        if let Some(piece) = state.take(thread, chunk) {
            return Next::Chunk(piece);
        }
        if busy {
            return Next::Busy;
        }
        state.running.remove(thread);
        Next::Idle
    }

    /// Charge `piece`, a chunk, to `link`'s budget and wait until the budget
    /// covers it. The wait is cut short by a change of the budget's rate (a
    /// throttle) and by the end of the piece's task, which abandons the chunk
    /// and refunds its bytes. Returns whether the chunk is to be copied.
    fn pace(&self, link: &Link, piece: &Piece, sizing: &Sizing) -> bool {
        let bytes = piece.copied();
        let capacity = (sizing.depth as u64 * sizing.bytes) as f64;
        let mut state = self.lock();
        let charged = Instant::now();
        state
            .budgets
            .entry(link.clone())
            .or_insert_with(|| Budget::new(charged))
            .charge(bytes, sizing.bandwidth, capacity, charged);
        loop {
            let now = Instant::now();
            if state.ended(piece.task) {
                if let Some(budget) = state.budgets.get_mut(link) {
                    budget.refund(bytes);
                }
                let reported = state.abandon(piece.task, now, self.progress());
                drop(state);
                if reported {
                    self.wake_main();
                }
                return false;
            }
            let delay = state
                .budgets
                .get_mut(link)
                .map_or(Duration::ZERO, |budget| {
                    budget.delay(sizing.bandwidth, capacity, now)
                });
            if delay.is_zero() {
                return true;
            }
            state = ::service::wait_timeout(&self.paced, state, delay).0;
        }
    }

    /// `piece`, a chunk, completed or failed; the caller drops it after.
    fn complete(&self, piece: &Piece, failure: Option<String>) {
        let failed = failure.is_some();
        let (reported, purged) = self.lock().complete(
            piece.task,
            piece.copied(),
            failure,
            Instant::now(),
            self.progress(),
        );
        drop(purged);
        if failed {
            // The task's chunks waiting for their budget are abandoned.
            self.paced.notify_all();
        }
        if reported {
            self.wake_main();
        }
    }

    /// Fail every chunk queued for `thread` with `message`, until nothing is
    /// queued: the thread cannot copy. It ends when its queue is idle.
    fn refuse(&self, thread: &ThreadKey, message: &str) {
        while let Next::Chunk(piece) = self.next(thread, u64::MAX, false) {
            self.complete(&piece, Some(message.to_owned()));
        }
    }
}

/// How copies are cut and reported: the Transfer Server's options.
pub(super) struct Policy {
    /// The time a chunk lasts at its link's bandwidth.
    pub(super) chunk: Duration,
    /// How often a running task reports its progress at most.
    pub(super) progress: Duration,
}

/// The copy threads and their shared state, as the main thread drives them.
pub(super) struct CopyThreads {
    shared: Arc<Shared>,
    /// The NIXL agent network copy threads post their READs with.
    network: Arc<Network>,
    /// The GPUs' contexts, which GPU copy threads hold while their threads run.
    contexts: Arc<Contexts>,
}

impl CopyThreads {
    /// Copy threads that cut and report copies by `policy` at the rates of
    /// `calibration` on `gpus` (each one's device and UUID), and wake the main
    /// thread through `wake`.
    pub(super) fn new(
        wake: OwnedFd,
        policy: &Policy,
        (calibration, gpus): (Arc<Calibration>, HashMap<DeviceId, [u8; 16]>),
        network: Arc<Network>,
        contexts: Arc<Contexts>,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    tasks: HashMap::new(),
                    queues: HashMap::new(),
                    running: HashSet::new(),
                    budgets: HashMap::new(),
                    reports: Vec::new(),
                    remote_links: Vec::new(),
                    reported_links: HashSet::new(),
                }),
                paced: Condvar::new(),
                wake,
                calibration,
                gpus,
                chunk: AtomicU64::new(nanoseconds(policy.chunk)),
                progress: AtomicU64::new(nanoseconds(policy.progress)),
                page: rustix::param::page_size() as u64,
            }),
            network,
            contexts,
        }
    }

    /// Queue task `task`'s pieces, each for its copy thread, at `priority`,
    /// and start every copy thread that does not run.
    pub(super) fn submit(
        &mut self,
        task: u64,
        priority: Priority,
        pieces: Vec<(ThreadKey, Piece)>,
    ) {
        let mut started = Vec::new();
        let mut state = self.shared.lock();
        let now = Instant::now();
        state.tasks.insert(
            task,
            TaskState {
                pieces: pieces.len(),
                running: 0,
                copied: 0,
                end: None,
                reported: now,
                queued: now,
                queued_pieces: pieces.len(),
                links: pieces
                    .iter()
                    .map(|(thread, _)| thread.link.clone())
                    .collect(),
            },
        );
        for (thread, piece) in pieces {
            if state.running.insert(thread.clone()) {
                started.push(thread.clone());
            }
            state.queues.entry(thread).or_default().levels[priority.rank()].push_back(piece);
        }
        drop(state);
        for thread in started {
            self.spawn(thread);
        }
    }

    /// End task `task` at once as `end`, before it copied anything: it
    /// cannot run, or it was cancelled before its pieces were queued.
    pub(super) fn end(&mut self, task: u64, end: TaskStatus) {
        self.shared.lock().reports.push((task, 0, Some(end)));
        self.shared.wake_main();
    }

    /// Cancel task `task`: its queued pieces leave, its chunks waiting for
    /// their budget are abandoned, and it ends once its submitted chunks
    /// completed.
    pub(super) fn cancel(&mut self, task: u64) {
        let mut state = self.shared.lock();
        let Some(entry) = state.tasks.get_mut(&task) else {
            return;
        };
        if entry.end.is_none() {
            entry.end = Some(TaskStatus::Cancelled);
        }
        let purged = state.purge(task);
        let reported = state.report(task, Instant::now(), self.shared.progress());
        drop(state);
        drop(purged);
        self.shared.paced.notify_all();
        if reported {
            self.shared.wake_main();
        }
    }

    /// Cut every chunk from now on by `policy`: each thread sizes its next chunk by the new chunk
    /// time, and each task reports its progress by the new period from its next report.
    pub(super) fn pace(&self, policy: &Policy) {
        self.shared
            .chunk
            .store(nanoseconds(policy.chunk), Ordering::Relaxed);
        self.shared
            .progress
            .store(nanoseconds(policy.progress), Ordering::Relaxed);
    }

    /// Change the budgets of `links` and of every link a task of `tasks`
    /// copies over; a restoration that names neither restores every link
    /// whose budget is reduced.
    pub(super) fn throttle(&mut self, links: &[Link], tasks: &[u64], throttle: Throttle) {
        let now = Instant::now();
        let mut state = self.shared.lock();
        let mut changed: HashSet<Link> = links.iter().cloned().collect();
        for task in tasks {
            if let Some(entry) = state.tasks.get(task) {
                changed.extend(entry.links.iter().cloned());
            }
        }
        if changed.is_empty() && tasks.is_empty() && throttle == Throttle::Restore {
            changed = state
                .budgets
                .iter()
                .filter(|(_, budget)| budget.share(now) < 1.0)
                .map(|(link, _)| link.clone())
                .collect();
        }
        for link in changed {
            state
                .budgets
                .entry(link)
                .or_insert_with(|| Budget::new(now))
                .change(throttle, now);
        }
        drop(state);
        // Chunks waiting for their budget wait by the new rate.
        self.shared.paced.notify_all();
    }

    /// The reports made since the last call.
    pub(super) fn reports(&mut self) -> Vec<(u64, u64, Option<TaskStatus>)> {
        std::mem::take(&mut self.shared.lock().reports)
    }

    /// The links from other hosts copy threads started on since the last
    /// call, at their calibrated bandwidths.
    pub(super) fn remote_links(&mut self) -> Vec<LinkBandwidth> {
        std::mem::take(&mut self.shared.lock().remote_links)
    }

    /// Start a thread for the copy thread `thread`.
    fn spawn(&self, thread: ThreadKey) {
        let shared = Arc::clone(&self.shared);
        let network = Arc::clone(&self.network);
        let contexts = Arc::clone(&self.contexts);
        let spawned = std::thread::Builder::new()
            .name("uniserve-transfer-thread".into())
            .spawn(move || {
                let cost = match shared.cost(&thread) {
                    Ok(cost) => cost,
                    Err(message) => {
                        tracing::warn!(message, "a copy thread cannot copy");
                        shared.refuse(&thread, &message);
                        return;
                    }
                };
                match thread.runner {
                    Runner::Device(device) => {
                        device_thread(&shared, &thread, &contexts, device, cost);
                    }
                    Runner::Host => run(&shared, &thread, &mut HostCopier, cost),
                    Runner::Network => {
                        let mut copier = NetworkCopier {
                            network: &network,
                            bandwidth: cost.bytes_per_second as f64,
                            due: Instant::now(),
                        };
                        run(&shared, &thread, &mut copier, cost);
                    }
                }
            });
        if let Err(error) = spawned {
            fatal("starting a copy thread", &error);
        }
    }
}

/// What runs one copy thread's chunks: it submits a chunk's copy, and later
/// waits until the copy completed, as the host observes it.
trait Copier {
    /// A submitted chunk's copy, until it is waited for.
    type Flight;

    /// Whether a submitted copy runs on after [`Copier::submit`] returns
    /// (copy engines, READs), so that the thread keeps several in flight; the
    /// CPU's copies end within it.
    const ASYNCHRONOUS: bool;

    /// Submit `piece`'s copy; an error fails the piece's task, and no copy
    /// runs.
    fn submit(&mut self, piece: &Piece) -> Result<Self::Flight, String>;

    /// Wait until `flight` completed; returns why it failed.
    fn complete(&mut self, flight: Self::Flight) -> Option<String>;
}

/// Run `thread`'s chunks with `copier`, sized by the link's `cost`, until
/// nothing is queued or in flight.
fn run<C: Copier>(shared: &Shared, thread: &ThreadKey, copier: &mut C, cost: Cost) {
    let mut chunk = shared.chunk();
    let mut sizing = Sizing::of(cost, chunk, shared.page, C::ASYNCHRONOUS);
    let mut flight: VecDeque<(Piece, C::Flight)> = VecDeque::with_capacity(sizing.depth);
    loop {
        // A chunk time the server changed sizes the next chunk; the chunks in flight keep theirs.
        let paced = shared.chunk();
        if paced != chunk {
            chunk = paced;
            sizing = Sizing::of(cost, chunk, shared.page, C::ASYNCHRONOUS);
        }
        let piece = match shared.next(thread, sizing.bytes, !flight.is_empty()) {
            Next::Chunk(piece) => piece,
            Next::Busy => {
                // Nothing is queued: the oldest chunk in flight completes
                // first.
                if let Some((piece, submitted)) = flight.pop_front() {
                    let failure = copier.complete(submitted);
                    shared.complete(&piece, failure);
                }
                continue;
            }
            Next::Idle => return,
        };
        if !shared.pace(&thread.link, &piece, &sizing) {
            continue;
        }
        match copier.submit(&piece) {
            Ok(submitted) => flight.push_back((piece, submitted)),
            Err(message) => {
                shared.complete(&piece, Some(message));
                continue;
            }
        }
        if flight.len() >= sizing.depth
            && let Some((piece, submitted)) = flight.pop_front()
        {
            let failure = copier.complete(submitted);
            shared.complete(&piece, failure);
        }
    }
}

/// Run `thread`'s chunks on the copy engines of `device`, until nothing is
/// queued or in flight.
///
/// The thread holds the GPU and makes its stream in the GPU's context. A
/// context it cannot retain, or a stream it cannot make (the GPU has no
/// storage left for them), fails the copy thread's chunks, not the process:
/// no copy ran in the context.
fn device_thread(
    shared: &Shared,
    thread: &ThreadKey,
    contexts: &Contexts,
    device: DeviceId,
    cost: Cost,
) {
    // Locals drop in reverse order: the copier's stream, then the hold, which
    // may release the context. Every chunk in flight completed when `run`
    // returns.
    let started = contexts.hold(device).and_then(|hold| {
        hold.session()?;
        let stream = Stream::new(device)
            .map_err(|error| format!("creating a stream for transfers on {device}: {error}"))?;
        Ok((hold, stream))
    });
    let (_hold, stream) = match started {
        Ok(started) => started,
        Err(message) => {
            tracing::warn!(message, "a GPU thread cannot copy");
            shared.refuse(thread, &message);
            return;
        }
    };
    let mut copier = DeviceCopier {
        stream,
        runner: thread.runner,
    };
    run(shared, thread, &mut copier, cost);
}

/// The copy engines of one GPU, on a stream of the copy thread's.
struct DeviceCopier {
    stream: Stream,
    runner: Runner,
}

impl Copier for DeviceCopier {
    /// The blocking event recorded after the chunk's copy.
    type Flight = Event;

    const ASYNCHRONOUS: bool = true;

    fn submit(&mut self, piece: &Piece) -> Result<Event, String> {
        let destination = piece.destination.address(self.runner)?;
        let source = piece.source.address(self.runner)?;
        // SAFETY: both ranges lie within allocations the piece holds
        // mapped (the server checked every span against its allocation),
        // and the piece is held until the event after the copy completed.
        // The submitter keeps the source unchanged and the destination
        // unread until the task ends.
        let submitted = unsafe {
            if piece.rows > 1 {
                // Rows at a pitch run as one pitched copy.
                let mut batch = PitchedBatch::new(SourceOrder::Stream);
                batch
                    .push(Pitched {
                        destination: destination + piece.destination_offset,
                        destination_pitch: piece.destination_pitch as usize,
                        destination_layer: 0,
                        source: source + piece.source_offset,
                        source_pitch: piece.source_pitch as usize,
                        source_layer: 0,
                        width: piece.bytes as usize,
                        height: piece.rows as usize,
                        depth: 1,
                    })
                    .and_then(|()| batch.submit(&self.stream))
            } else {
                memcpy_async(
                    destination + piece.destination_offset,
                    source + piece.source_offset,
                    piece.bytes as usize,
                    &self.stream,
                )
            }
        };
        if let Err(error) = submitted {
            fatal("submitting a copy", &error);
        }
        // The thread blocks on the event when it waits for the copy, holding
        // no core meanwhile.
        let completion = Event::new(EventFlags {
            blocking: true,
            ..EventFlags::default()
        });
        if let Err(error) = completion.record(&self.stream) {
            fatal("recording a copy's completion", &error);
        }
        Ok(completion)
    }

    fn complete(&mut self, completion: Event) -> Option<String> {
        if let Err(error) = completion.synchronize() {
            fatal("waiting for a copy", &error);
        }
        None
    }
}

/// The CPU, between host buffers.
struct HostCopier;

impl Copier for HostCopier {
    type Flight = ();

    const ASYNCHRONOUS: bool = false;

    fn submit(&mut self, piece: &Piece) -> Result<(), String> {
        let destination = piece.destination.address(Runner::Host)?;
        let source = piece.source.address(Runner::Host)?;
        for row in 0..piece.rows {
            // SAFETY: every row lies within mappings the piece holds (the
            // server checked each span's rows against its allocation). `copy`
            // allows the two to overlap within one allocation.
            unsafe {
                std::ptr::copy(
                    (source + piece.source_offset + row * piece.source_pitch) as *const u8,
                    (destination + piece.destination_offset + row * piece.destination_pitch)
                        as *mut u8,
                    piece.bytes as usize,
                );
            }
        }
        Ok(())
    }

    fn complete(&mut self, (): ()) -> Option<String> {
        None
    }
}

/// This process's NIXL agent and Transfer Engine, with READs from another host's storage into
/// this host's and WRITEs the other way.
struct NetworkCopier<'a> {
    network: &'a Network,
    /// The rate of the link's READs, in bytes per second.
    bandwidth: f64,
    /// When the bytes of the READs posted so far are due at that rate.
    due: Instant,
}

/// A READ or a WRITE posted, by the transport that carries it.
enum Flight {
    Nixl(nixl::Transfer),
    Mooncake(mooncake::Batch),
}

impl Flight {
    /// Waits until the transfer completed, from `due` on ([`nixl::Transfer::wait`],
    /// [`mooncake::Batch::wait`]).
    fn wait(self, due: Instant) -> Result<(), String> {
        match self {
            Flight::Nixl(transfer) => transfer.wait(due).map_err(|error| error.to_string()),
            Flight::Mooncake(batch) => batch.wait(due).map_err(|error| error.to_string()),
        }
    }
}

impl Copier for NetworkCopier<'_> {
    /// The READ or WRITE, and when its bytes are due.
    type Flight = (Flight, Instant);

    const ASYNCHRONOUS: bool = true;

    fn submit(&mut self, piece: &Piece) -> Result<Self::Flight, String> {
        let transfer = post(self.network, piece)?;
        // Its bytes follow those of the READs posted before it.
        let seconds = piece.copied() as f64 / self.bandwidth;
        self.due = self.due.max(Instant::now()) + Duration::from_secs_f64(seconds);
        Ok((transfer, self.due))
    }

    fn complete(&mut self, (transfer, due): Self::Flight) -> Option<String> {
        // A READ that failed (another host or the network failed) fails its
        // task, not the process: no CUDA context took part in it.
        let failure = transfer
            .wait(due)
            .err()
            .map(|error| format!("a transfer with another host failed: {error}"));
        tracing::debug!(failure, "a transfer with another host ended");
        failure
    }
}

/// Post `piece` as one READ of `network`'s agent of the piece's transport: from another host's
/// storage, which its Transfer Server registered, into an allocation of this host's, registered
/// here at its first READ: device storage in its GPU's session, host storage with the
/// allocation; or as one WRITE the other way.
fn post(network: &Network, piece: &Piece) -> Result<Flight, String> {
    let (remote, mapped, hold, write) = match (&piece.source, &piece.destination) {
        (End::Remote(source), End::Given { mapped, hold }) => (source, mapped, hold, false),
        (End::Given { mapped, hold }, End::Remote(destination)) => {
            (destination, mapped, hold, true)
        }
        _ => {
            return Err(
                "a READ or a WRITE moves bytes between another host's storage and this host's"
                    .into(),
            );
        }
    };
    let (peer, remote_region) = match &remote.reach {
        Reach::Nixl { peer, region } => (peer, *region),
        Reach::Mooncake { segment, address } => {
            return post_mooncake(network, piece, (remote, *segment, *address), (mapped, hold))
                .map(Flight::Mooncake);
        }
    };
    let agent = network.agent()?;
    let local = match hold {
        Some(hold) => hold.session()?.nixl_region(agent, mapped)?,
        None => mapped.nixl_host_region(agent)?,
    };
    let len =
        usize::try_from(piece.bytes).map_err(|_| "a chunk exceeds the address space".to_owned())?;
    // The server checked this host's end of the span against its allocation, and the
    // exposure resolution the other end against its region's range. One READ or WRITE moves
    // the chunk's rows, a descriptor each.
    let at = |region: nixl::Region, offset: u64, pitch: u64, row: u64| nixl::Region {
        address: region.address + offset + row * pitch,
        len,
        ..region
    };
    let spans: Vec<nixl::Span> = (0..piece.rows)
        .map(|row| {
            let source = |region| at(region, piece.source_offset, piece.source_pitch, row);
            let destination = |region| {
                at(
                    region,
                    piece.destination_offset,
                    piece.destination_pitch,
                    row,
                )
            };
            if write {
                nixl::Span {
                    source: source(local),
                    destination: destination(remote_region),
                }
            } else {
                nixl::Span {
                    source: source(remote_region),
                    destination: destination(local),
                }
            }
        })
        .collect();
    let (transfer, what) = if write {
        (agent.write(peer, &spans), "WRITE into")
    } else {
        (agent.read(peer, &spans), "READ from")
    };
    let transfer =
        transfer.map_err(|error| format!("posting a {what} {}: {error}", remote.endpoint))?;
    tracing::debug!(
        task = piece.task,
        bytes = piece.bytes,
        endpoint = remote.endpoint,
        write,
        "posted a transfer with another host"
    );
    Ok(Flight::Nixl(transfer))
}

/// Posts `piece` over `network`'s Transfer Engine: a READ of the region of `remote` the
/// engine's `segment` registered from `address` on, into `mapped`, an allocation of this host's
/// registered here at its first transfer (in its GPU's session, which `hold` holds, for device
/// storage); or a WRITE the other way. One transfer moves each row of the chunk.
fn post_mooncake(
    network: &Network,
    piece: &Piece,
    (remote, segment, address): (&Source, mooncake::SegmentId, u64),
    (mapped, hold): (&Mapping, &Option<Hold>),
) -> Result<mooncake::Batch, String> {
    let engine = network.engine()?;
    let local = match hold {
        Some(hold) => hold.session()?.mooncake_address(engine, mapped)?,
        None => mapped.mooncake_host_address(engine)?,
    };
    let write = matches!(piece.destination, End::Remote(_));
    // The server checked this host's end of the span against its allocation, and the
    // exposure's resolution the other end against its region's range.
    let row = |base: u64, offset: u64, pitch: u64, row: u64| base + offset + row * pitch;
    let spans: Vec<mooncake::Span> = (0..piece.rows)
        .map(|index| {
            let source = |base| row(base, piece.source_offset, piece.source_pitch, index);
            let destination = |base| {
                row(
                    base,
                    piece.destination_offset,
                    piece.destination_pitch,
                    index,
                )
            };
            if write {
                mooncake::Span {
                    local: source(local),
                    remote: destination(address),
                    len: piece.bytes,
                }
            } else {
                mooncake::Span {
                    local: destination(local),
                    remote: source(address),
                    len: piece.bytes,
                }
            }
        })
        .collect();
    let (posted, what) = if write {
        (engine.write(segment, &spans), "WRITE into")
    } else {
        (engine.read(segment, &spans), "READ from")
    };
    let batch = posted.map_err(|error| {
        format!(
            "posting a Transfer Engine {what} {}: {error}",
            remote.endpoint
        )
    })?;
    tracing::debug!(
        task = piece.task,
        bytes = piece.bytes,
        endpoint = remote.endpoint,
        write,
        "posted a Transfer Engine transfer with another host"
    );
    Ok(batch)
}

/// End the process after a device fault: its CUDA contexts are no longer
/// trusted. The Storage Server fails every task and starts another.
pub(super) fn fatal(what: &str, error: &dyn std::fmt::Display) -> ! {
    tracing::error!(%error, "the Transfer Server failed {what}; it ends");
    std::process::exit(1)
}

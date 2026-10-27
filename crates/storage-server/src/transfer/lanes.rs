//! The lanes that run copies: one thread per link and copy runner, its
//! queue of pieces by priority, and its link's byte budget.
//!
//! A task's moves become pieces, each queued on the lane of its link and of
//! the GPU whose copy engine runs it (or the CPU, between host buffers). A
//! lane repeatedly takes one chunk off the front piece of its highest
//! priority that has one, waits until its link's budget covers the chunk,
//! and copies it:
//!
//! - **Chunks.** A chunk lasts the configured chunk time at the link's
//!   bandwidth, so that a copy engine is never held by one copy much longer
//!   than that, far below a scheduler step of the world executing beside
//!   the transfers; a task of a higher priority starts on the next chunk.
//! - **Budgets.** A link may move its budget's share of its bandwidth: a
//!   token bucket refilled at that rate, holding at most [`DEPTH`] chunks,
//!   and charged before each chunk. The share starts whole; a head halves
//!   it down to [`FLOOR_SHARE`] while its scheduler steps slow down, and
//!   after its restoration it grows back by [`RESTORE_PER_SECOND`] of the
//!   bandwidth per second, so the head's statistics, once a second, see
//!   each step of the recovery.
//! - **Completion.** A GPU lane keeps [`DEPTH`] chunks submitted on its own
//!   stream, so its copy engine never waits for the host, and counts a
//!   chunk copied only once the event recorded after it completed, as the
//!   host observes. A network lane keeps [`DEPTH`] NIXL READs posted and
//!   counts a chunk copied once its READ completed, as the host polls it:
//!   a READ passes through no CUDA stream. A task ends when its last chunk
//!   completed: complete, cancelled or failed. A cancelled or failed task's
//!   queued pieces leave at once, and it ends once its submitted chunks
//!   completed, since only then are its regions free of copies.
//! - **Links from other hosts.** Nobody measured them when the process
//!   started: a network lane times its chunks of [`MEASURE_BYTES`], one at
//!   a time, until [`MEASURE_SAMPLES`] of them took the link's bandwidth,
//!   and reports it; until then its chunks are of that size and wait for
//!   no budget.
//!
//! A device fault (a failed copy submission or completion) leaves the CUDA
//! context unusable, so the process ends: the Storage Server fails every
//! task it held and starts a new Transfer Server. A failed READ (another
//! host or the network failed) fails its task alone.

use std::collections::{HashMap, HashSet, VecDeque};
use std::os::fd::OwnedFd;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use cuda::{DeviceId, Event, EventFlags, Stream, memcpy_async};
use storage::protocol::{Link, LinkBandwidth, Priority, Throttle, TransferEnd};

use super::memory::{Fabric, Mapped};
use super::network::{Network, Source};

/// How often a running task reports its progress, at most.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// The smallest chunk, so that submitting a chunk stays cheap next to
/// copying it.
const MIN_CHUNK: u64 = 256 << 10;

/// Chunks are whole multiples of this.
const CHUNK_ALIGN: u64 = 64 << 10;

/// The chunk a GPU lane times to learn the bandwidth of a link nobody
/// measured (one from another host's GPU of the NVLink fabric).
const PROBE_CHUNK: u64 = 4 << 20;

/// The chunk a network lane times to learn the bandwidth of a link from
/// another host: as long as a measured copy when the process starts
/// (`devices::SAMPLE_BYTES`), so that the fixed cost of posting a READ and
/// seeing it complete is a few percent of its time at the line rate of the
/// host's network adapters.
const MEASURE_BYTES: u64 = 64 << 20;

/// The timed chunks of [`MEASURE_BYTES`] whose fastest gives a link from
/// another host its bandwidth: the first READ over a new connection also
/// sets the connection up.
const MEASURE_SAMPLES: usize = 2;

/// The chunks a GPU or network lane keeps in flight: one copying while the host
/// waits for the other.
const DEPTH: usize = 2;

/// The least share of its bandwidth a halved link keeps.
const FLOOR_SHARE: f64 = 1.0 / 64.0;

/// How fast a restored link's share grows back, per second.
const RESTORE_PER_SECOND: f64 = 1.0 / 8.0;

/// How long a GPU lane sleeps between checks of its oldest chunk.
const POLL: Duration = Duration::from_micros(20);

/// What runs a lane's copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Runner {
    /// The copy engines of this GPU, on a stream of the lane's.
    Device(DeviceId),
    /// The CPU, between host buffers.
    Host,
    /// This process's NIXL agent, with READs from another host through
    /// the network adapters.
    Network,
}

/// A lane: the link it moves bytes over and what runs its copies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct LaneKey {
    pub(super) link: Link,
    pub(super) runner: Runner,
}

/// One end of a piece: an allocation the server gave, another host's
/// allocation imported on the lane's GPU, or another host's memory that a
/// READ reaches.
#[derive(Clone)]
pub(super) enum End {
    Given(Arc<Mapped>),
    Fabric(Arc<Fabric>),
    Remote(Arc<Source>),
}

impl End {
    /// The address of byte 0 as `runner`'s copies use it.
    fn address(&self, runner: Runner) -> Result<u64, String> {
        match (self, runner) {
            (End::Given(mapped), Runner::Host) => mapped.host_address(),
            (End::Given(mapped), Runner::Device(device)) => match mapped.memory {
                storage::protocol::Memory::Device { .. } => mapped.device_address(),
                storage::protocol::Memory::Host => mapped.registered_address(device),
                storage::protocol::Memory::Remote { .. } => {
                    Err("the server gives allocations of this host only".into())
                }
            },
            (End::Fabric(fabric), Runner::Device(_)) => fabric.address(),
            (End::Fabric(_), Runner::Host) => Err("the CPU copies no GPU memory".into()),
            (End::Remote(_), _) | (_, Runner::Network) => {
                Err("another host's memory is read by a READ, not copied by address".into())
            }
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
    pub(super) bytes: u64,
}

/// A task as the lanes run it.
struct TaskState {
    /// Pieces queued and not taken whole.
    pieces: usize,
    /// Chunks submitted and not completed.
    running: usize,
    copied: u64,
    /// How the task ends, once a cancellation or a failure decided it.
    end: Option<TransferEnd>,
    reported: Instant,
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
            Some((since, from)) => {
                (from + RESTORE_PER_SECOND * now.duration_since(since).as_secs_f64()).min(1.0)
            }
            None => self.share,
        }
    }

    fn change(&mut self, throttle: Throttle, now: Instant) {
        let share = self.share(now);
        match throttle {
            Throttle::Halve => {
                self.share = (share / 2.0).max(FLOOR_SHARE);
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

    /// Charge `bytes` of a link of `bandwidth` bytes per second, whose
    /// bucket holds `capacity` bytes; returns how long the chunk waits.
    fn charge(&mut self, bytes: u64, bandwidth: f64, capacity: f64, now: Instant) -> Duration {
        let rate = self.share(now) * bandwidth;
        let elapsed = now.duration_since(self.refilled).as_secs_f64();
        self.refilled = now;
        self.tokens = (self.tokens + rate * elapsed).min(capacity) - bytes as f64;
        if self.tokens >= 0.0 || rate <= 0.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(-self.tokens / rate)
        }
    }
}

/// The pieces of one lane, by priority.
#[derive(Default)]
struct Queue {
    levels: [VecDeque<Piece>; 3],
}

/// What the lanes share with the main thread.
struct State {
    tasks: HashMap<u64, TaskState>,
    queues: HashMap<LaneKey, Queue>,
    budgets: HashMap<Link, Budget>,
    /// Each report: the task, the bytes it copied, and its end once it
    /// ended.
    reports: Vec<(u64, u64, Option<TransferEnd>)>,
    /// Links from other hosts that lanes measured, not reported yet.
    measured: Vec<LinkBandwidth>,
}

impl State {
    /// One chunk of at most `chunk` bytes off the front piece of the
    /// highest priority `lane` has.
    fn take(&mut self, lane: &LaneKey, chunk: u64) -> Option<Piece> {
        let queue = self.queues.get_mut(lane)?;
        let level = queue.levels.iter_mut().find(|level| !level.is_empty())?;
        let front = level.front_mut()?;
        let bytes = front.bytes.min(chunk);
        let mut taken = front.clone();
        taken.bytes = bytes;
        front.bytes -= bytes;
        front.source_offset += bytes;
        front.destination_offset += bytes;
        let whole = front.bytes == 0;
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

    /// Drop every queued piece of `task`.
    fn purge(&mut self, task: u64) {
        for queue in self.queues.values_mut() {
            for level in &mut queue.levels {
                level.retain(|piece| piece.task != task);
            }
        }
        if let Some(state) = self.tasks.get_mut(&task) {
            state.pieces = 0;
        }
    }

    /// Report `task` ended when nothing of it is queued or running, or its
    /// progress when it last reported long enough ago. Returns whether a
    /// report was made.
    fn report(&mut self, task: u64, now: Instant) -> bool {
        let Some(state) = self.tasks.get_mut(&task) else {
            return false;
        };
        if state.pieces == 0 && state.running == 0 {
            let end = state.end.take().unwrap_or(TransferEnd::Complete);
            let copied = state.copied;
            self.tasks.remove(&task);
            self.reports.push((task, copied, Some(end)));
            return true;
        }
        if now.duration_since(state.reported) >= PROGRESS_INTERVAL {
            state.reported = now;
            self.reports.push((task, state.copied, None));
            return true;
        }
        false
    }

    /// A chunk of `task` completed, copying `bytes`, or failed.
    fn complete(&mut self, task: u64, bytes: u64, failure: Option<String>, now: Instant) -> bool {
        let Some(state) = self.tasks.get_mut(&task) else {
            return false;
        };
        state.running -= 1;
        match failure {
            None => state.copied += bytes,
            Some(message) => {
                if state.end.is_none() {
                    state.end = Some(TransferEnd::Failed { message });
                }
                self.purge(task);
            }
        }
        self.report(task, now)
    }
}

/// The state behind its lock, the lanes' wake-up, and the main thread's.
struct Shared {
    state: Mutex<State>,
    /// Signalled when pieces are queued.
    work: Condvar,
    /// The write end of the main thread's wake pipe, written when a report
    /// is made.
    wake: OwnedFd,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn wake_main(&self) {
        let _ = rustix::io::write(&self.wake, b"x");
    }

    /// The next chunk of `lane`, waiting for one unless `busy` (the lane
    /// has chunks to complete meanwhile).
    fn next(&self, lane: &LaneKey, chunk: u64, busy: bool) -> Option<Piece> {
        let mut state = self.lock();
        loop {
            if let Some(piece) = state.take(lane, chunk) {
                return Some(piece);
            }
            if busy {
                return None;
            }
            state = self
                .work
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Wait until `link`'s budget covers `bytes`.
    fn pace(&self, link: &Link, bytes: u64, bandwidth: f64, capacity: f64) {
        let delay = {
            let now = Instant::now();
            let mut state = self.lock();
            state
                .budgets
                .entry(link.clone())
                .or_insert_with(|| Budget::new(now))
                .charge(bytes, bandwidth, capacity, now)
        };
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
    }

    fn complete(&self, piece: &Piece, failure: Option<String>) {
        let reported = self
            .lock()
            .complete(piece.task, piece.bytes, failure, Instant::now());
        if reported {
            self.wake_main();
        }
    }

    /// A lane measured `link`, from another host, at `bandwidth` bytes per
    /// second.
    fn measured(&self, link: &Link, bandwidth: f64) {
        tracing::info!(
            ?link,
            gigabytes_per_second = bandwidth / 1e9,
            "measured a link from another host"
        );
        self.lock().measured.push(LinkBandwidth {
            link: link.clone(),
            bytes_per_second: bandwidth as u64,
        });
        self.wake_main();
    }
}

/// The lanes and their shared state, as the main thread drives them.
pub(super) struct Lanes {
    shared: Arc<Shared>,
    running: HashSet<LaneKey>,
    /// The time a chunk lasts at its link's bandwidth.
    chunk: Duration,
    /// Each link's bandwidth as measured at start, in bytes per second.
    bandwidths: HashMap<Link, f64>,
    /// The NIXL agent network lanes post their READs with.
    network: Arc<Network>,
}

impl Lanes {
    pub(super) fn new(
        wake: OwnedFd,
        chunk: Duration,
        bandwidths: HashMap<Link, f64>,
        network: Arc<Network>,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    tasks: HashMap::new(),
                    queues: HashMap::new(),
                    budgets: HashMap::new(),
                    reports: Vec::new(),
                    measured: Vec::new(),
                }),
                work: Condvar::new(),
                wake,
            }),
            running: HashSet::new(),
            chunk,
            bandwidths,
            network,
        }
    }

    /// Queue task `task`'s pieces, each on its lane, at `priority`.
    pub(super) fn submit(&mut self, task: u64, priority: Priority, pieces: Vec<(LaneKey, Piece)>) {
        for (lane, _) in &pieces {
            if self.running.insert(lane.clone()) {
                self.spawn(lane.clone());
            }
        }
        let mut state = self.shared.lock();
        state.tasks.insert(
            task,
            TaskState {
                pieces: pieces.len(),
                running: 0,
                copied: 0,
                end: None,
                reported: Instant::now(),
            },
        );
        for (lane, piece) in pieces {
            state.queues.entry(lane).or_default().levels[priority.rank()].push_back(piece);
        }
        drop(state);
        self.shared.work.notify_all();
    }

    /// End task `task` at once as `end`, before it copied anything: it
    /// cannot run, or it was cancelled before its pieces were queued.
    pub(super) fn end(&mut self, task: u64, end: TransferEnd) {
        self.shared.lock().reports.push((task, 0, Some(end)));
        self.shared.wake_main();
    }

    /// Cancel task `task`: its queued pieces leave, and it ends once its
    /// submitted chunks completed.
    pub(super) fn cancel(&mut self, task: u64) {
        let mut state = self.shared.lock();
        let Some(entry) = state.tasks.get_mut(&task) else {
            return;
        };
        if entry.end.is_none() {
            entry.end = Some(TransferEnd::Cancelled);
        }
        state.purge(task);
        if state.report(task, Instant::now()) {
            drop(state);
            self.shared.wake_main();
        }
    }

    /// Change the budgets of `links`.
    pub(super) fn throttle(&mut self, links: &[Link], throttle: Throttle) {
        let now = Instant::now();
        let mut state = self.shared.lock();
        for link in links {
            state
                .budgets
                .entry(link.clone())
                .or_insert_with(|| Budget::new(now))
                .change(throttle, now);
        }
    }

    /// The reports made since the last call.
    pub(super) fn reports(&mut self) -> Vec<(u64, u64, Option<TransferEnd>)> {
        std::mem::take(&mut self.shared.lock().reports)
    }

    /// The links from other hosts measured since the last call.
    pub(super) fn measured(&mut self) -> Vec<LinkBandwidth> {
        std::mem::take(&mut self.shared.lock().measured)
    }

    /// Start the thread of `lane`.
    fn spawn(&self, lane: LaneKey) {
        let shared = Arc::clone(&self.shared);
        let network = Arc::clone(&self.network);
        let bandwidth = self.bandwidths.get(&lane.link).copied();
        let chunk = self.chunk;
        let spawned = std::thread::Builder::new()
            .name("uniserve-transfer-lane".into())
            .spawn(move || match lane.runner {
                Runner::Device(device) => device_lane(&shared, &lane, device, bandwidth, chunk),
                Runner::Host => host_lane(&shared, &lane, bandwidth, chunk),
                Runner::Network => network_lane(&shared, &lane, &network, bandwidth, chunk),
            });
        if let Err(error) = spawned {
            fatal("starting a lane's thread", &error);
        }
    }
}

/// The bytes of a chunk that lasts `chunk` at `bandwidth` bytes per second.
fn chunk_bytes(bandwidth: Option<f64>, chunk: Duration) -> u64 {
    let Some(bandwidth) = bandwidth else {
        return PROBE_CHUNK;
    };
    let bytes = (bandwidth * chunk.as_secs_f64()) as u64;
    (bytes / CHUNK_ALIGN * CHUNK_ALIGN).max(MIN_CHUNK)
}

/// Run `lane`'s chunks on the copy engines of `device`.
fn device_lane(
    shared: &Shared,
    lane: &LaneKey,
    device: DeviceId,
    mut bandwidth: Option<f64>,
    chunk: Duration,
) {
    let stream =
        Stream::new(device).unwrap_or_else(|error| fatal("creating a lane's stream", &error));
    let mut flight: VecDeque<(Piece, Event)> = VecDeque::with_capacity(DEPTH);
    loop {
        let bytes = chunk_bytes(bandwidth, chunk);
        let Some(piece) = shared.next(lane, bytes, !flight.is_empty()) else {
            // Nothing is queued: the oldest chunk in flight completes first.
            if let Some((piece, event)) = flight.pop_front() {
                wait(&event);
                shared.complete(&piece, None);
            }
            continue;
        };
        if let Some(bandwidth) = bandwidth {
            shared.pace(
                &lane.link,
                piece.bytes,
                bandwidth,
                (DEPTH as u64 * bytes) as f64,
            );
        }
        let addresses = piece
            .destination
            .address(lane.runner)
            .and_then(|destination| Ok((destination, piece.source.address(lane.runner)?)));
        let (destination, source) = match addresses {
            Ok(addresses) => addresses,
            Err(message) => {
                shared.complete(&piece, Some(message));
                continue;
            }
        };
        let started = Instant::now();
        // SAFETY: both ranges lie within allocations the piece holds
        // mapped (the server checked every span against its allocation),
        // and the piece is held until the event after the copy completed.
        // The submitter keeps the source unchanged and the destination
        // unread until the task ends.
        let submitted = unsafe {
            memcpy_async(
                destination + piece.destination_offset,
                source + piece.source_offset,
                piece.bytes as usize,
                &stream,
            )
        };
        if let Err(error) = submitted {
            fatal("submitting a copy", &error);
        }
        let event = Event::new(EventFlags::default());
        if let Err(error) = event.record(&stream) {
            fatal("recording a copy's completion", &error);
        }
        if bandwidth.is_none() {
            // A link nobody measured: its first chunk alone times it.
            wait(&event);
            let seconds = started.elapsed().as_secs_f64().max(1e-6);
            bandwidth = Some(piece.bytes as f64 / seconds);
            shared.complete(&piece, None);
            continue;
        }
        flight.push_back((piece, event));
        if flight.len() >= DEPTH
            && let Some((piece, event)) = flight.pop_front()
        {
            wait(&event);
            shared.complete(&piece, None);
        }
    }
}

/// Run `lane`'s chunks, between host buffers, on the CPU.
fn host_lane(shared: &Shared, lane: &LaneKey, bandwidth: Option<f64>, chunk: Duration) {
    loop {
        let bytes = chunk_bytes(bandwidth, chunk);
        let Some(piece) = shared.next(lane, bytes, false) else {
            continue;
        };
        if let Some(bandwidth) = bandwidth {
            shared.pace(
                &lane.link,
                piece.bytes,
                bandwidth,
                (DEPTH as u64 * bytes) as f64,
            );
        }
        let addresses = piece
            .destination
            .address(Runner::Host)
            .and_then(|destination| Ok((destination, piece.source.address(Runner::Host)?)));
        match addresses {
            Ok((destination, source)) => {
                // SAFETY: both ranges lie within mappings the piece holds
                // (the server checked every span against its allocation).
                // `copy` allows the two to overlap within one allocation.
                unsafe {
                    std::ptr::copy(
                        (source + piece.source_offset) as *const u8,
                        (destination + piece.destination_offset) as *mut u8,
                        piece.bytes as usize,
                    );
                }
                shared.complete(&piece, None);
            }
            Err(message) => shared.complete(&piece, Some(message)),
        }
    }
}

/// Run `lane`'s chunks, from another host's memory into this host's, as
/// NIXL READs this process's agent posts.
///
/// While the link is unmeasured its chunks are [`MEASURE_BYTES`] long and
/// run one at a time, each timed from its post to its completion; the
/// fastest of the first [`MEASURE_SAMPLES`] full chunks becomes the link's
/// bandwidth, which the lane reports. A shorter chunk (the tail of a piece)
/// is not timed: the fixed cost of a READ would understate the link.
fn network_lane(
    shared: &Shared,
    lane: &LaneKey,
    network: &Network,
    mut bandwidth: Option<f64>,
    chunk: Duration,
) {
    let mut flight: VecDeque<(Piece, nixl::Transfer)> = VecDeque::with_capacity(DEPTH);
    let mut samples = Vec::with_capacity(MEASURE_SAMPLES);
    loop {
        let bytes = match bandwidth {
            Some(_) => chunk_bytes(bandwidth, chunk),
            None => MEASURE_BYTES,
        };
        let Some(piece) = shared.next(lane, bytes, !flight.is_empty()) else {
            // Nothing is queued: the oldest READ in flight completes first.
            if let Some((piece, transfer)) = flight.pop_front() {
                let failure = finish(transfer);
                shared.complete(&piece, failure);
            }
            continue;
        };
        if let Some(bandwidth) = bandwidth {
            shared.pace(
                &lane.link,
                piece.bytes,
                bandwidth,
                (DEPTH as u64 * bytes) as f64,
            );
        }
        let started = Instant::now();
        let transfer = match post(network, &piece) {
            Ok(transfer) => transfer,
            Err(message) => {
                shared.complete(&piece, Some(message));
                continue;
            }
        };
        if bandwidth.is_none() {
            let failure = finish(transfer);
            if failure.is_none() && piece.bytes == MEASURE_BYTES {
                let seconds = started.elapsed().as_secs_f64().max(1e-6);
                samples.push(piece.bytes as f64 / seconds);
                if samples.len() == MEASURE_SAMPLES {
                    let fastest = samples.iter().copied().fold(0.0, f64::max);
                    bandwidth = Some(fastest);
                    shared.measured(&lane.link, fastest);
                }
            }
            shared.complete(&piece, failure);
            continue;
        }
        flight.push_back((piece, transfer));
        if flight.len() >= DEPTH
            && let Some((piece, transfer)) = flight.pop_front()
        {
            let failure = finish(transfer);
            shared.complete(&piece, failure);
        }
    }
}

/// Post `piece` as one READ of `network`'s agent: from another host's
/// memory, which its Transfer Server registered, into an allocation of this
/// host's, registered here at its first READ.
fn post(network: &Network, piece: &Piece) -> Result<nixl::Transfer, String> {
    let (End::Remote(source), End::Given(destination)) = (&piece.source, &piece.destination) else {
        return Err("a READ reads another host's memory into this host's".into());
    };
    let agent = network.agent()?;
    let local = destination.nixl_region(agent)?;
    let len =
        usize::try_from(piece.bytes).map_err(|_| "a chunk exceeds the address space".to_owned())?;
    // The server checked the destination span against its allocation, and
    // the exposure resolution the source span against its region's range.
    let span = nixl::Span {
        source: nixl::Region {
            address: source.region.address + piece.source_offset,
            len,
            ..source.region
        },
        destination: nixl::Region {
            address: local.address + piece.destination_offset,
            len,
            ..local
        },
    };
    agent
        .read(&source.peer, &[span])
        .map_err(|error| format!("posting a READ from {}: {error}", source.endpoint))
}

/// Poll `transfer` until it completed, as the host observes it; returns
/// why it failed when it did. A READ that failed (another host or the
/// network failed) fails its task, not the process: no CUDA context took
/// part in it.
fn finish(mut transfer: nixl::Transfer) -> Option<String> {
    loop {
        match transfer.poll() {
            Ok(true) => return None,
            Ok(false) => std::thread::sleep(POLL),
            Err(error) => return Some(format!("a READ from another host failed: {error}")),
        }
    }
}

/// Wait until `event` completed, as the host observes it.
fn wait(event: &Event) {
    loop {
        match event.query() {
            Ok(true) => return,
            Ok(false) => std::thread::sleep(POLL),
            Err(error) => fatal("waiting for a copy", &error),
        }
    }
}

/// End the process after a device fault: its CUDA contexts are no longer
/// trusted. The Storage Server fails every task and starts another.
pub(super) fn fatal(what: &str, error: &dyn std::fmt::Display) -> ! {
    tracing::error!(%error, "the Transfer Server failed {what}; it ends");
    std::process::exit(1)
}

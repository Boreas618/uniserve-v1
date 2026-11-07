//! One rank's side of an expert union, and the step agreement of the ranks of a host.
//!
//! The ranks of an expert union run every expert layer together, because a dispatch on one rank
//! waits on its device for the same dispatch on every other. A rank with a batch therefore needs
//! every peer inside the same step: a peer with a batch of its own runs it in that step, and a peer
//! without one joins, replaying the expert layers with no tokens of its own so its experts serve
//! the tokens routed to them.
//!
//! # The agreement
//!
//! [`Union::enter`] is that agreement. A rank about to replay a batch enters with its live token
//! count, and the round stays open until every rank has entered; the step's capacity is the largest
//! count. The capacity is exact, not rounded to a bucket: the transport reads it from device memory
//! ([`ops::moe::exchange::Transport::step`]), so no graph is captured per capacity.
//!
//! A rank with nothing to do must not spin, and must not be missed either. Only a rank with tokens
//! opens a round, and the rank that opens one wakes every peer through its job queue ([`Job::Join`]).
//! A woken rank that has a batch by then serves it first and enters with its tokens; otherwise it
//! enters with none. A wake that arrives after its round closed finds no round open and costs
//! nothing, which is why a join with no open round returns at once and a round is only ever opened
//! by tokens.
//!
//! Every rank's device thread runs one job at a time, so a rank enters each round at most once, and
//! every rank passes through the same sequence of rounds. That sequence is the order the transport's
//! kernels count their phases in.
//!
//! [`Job::Join`]: crate::batch::Job::Join

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use nn::ExchangeViews;
use ops::moe::exchange::{Group, Transport, TransportShape};

use crate::job::JobQueue;
use crate::runner::RunnerId;
use crate::{Error, Result};

/// How long a rank waits for its peers before it names them, in one wait of [`Union::enter`].
const STRAGGLER_REPORT: Duration = Duration::from_secs(60);

/// How recently a rank must have served a batch for a wake to wait for its next one.
///
/// A rank that is serving has its next batch moments away, and a join now would spend a whole step
/// on none of its tokens. Two serving ranks would then take turns instead of sharing every step.
const SERVING_WINDOW: Duration = Duration::from_millis(20);

/// How long a serving rank's wake waits for that batch before it joins with no tokens.
const NEXT_BATCH_GRACE: Duration = Duration::from_micros(300);

/// What one agreement decided for one rank.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Step {
    /// Rows per rank of the step: the most live tokens any rank sends.
    pub capacity: u32,
}

/// The rounds of a union: which ranks entered the open one and with how many tokens.
struct Rounds {
    /// Each rank's tokens in the open round, when it entered.
    entered: Vec<Option<u32>>,
    /// Whether a round is open.
    open: bool,
    /// Rounds closed so far: a waiting rank's round closed when this moved on.
    closed: u64,
    /// The capacity of the last round closed.
    capacity: u32,
    /// The union ended: a rank is not coming back, so nobody waits.
    ended: bool,
}

/// One gather in flight: what each rank brought, and the last gather's result.
struct Gathered {
    values: Vec<Option<Vec<u8>>>,
    rounds: u64,
    last: Arc<Vec<Vec<u8>>>,
}

/// Where a rank is woken: its device's job queue and the runner that joins there.
struct Wake {
    queue: Arc<JobQueue>,
    runner: RunnerId,
}

/// The ranks of a union on this host: they are threads of one process.
pub struct Union {
    ranks: usize,
    rounds: Mutex<Rounds>,
    entered: Condvar,
    gathered: Mutex<Gathered>,
    gathering: Condvar,
    wakes: Vec<OnceLock<Wake>>,
    /// Whether a wake of each rank is queued and not yet taken up.
    waking: Vec<AtomicBool>,
}

impl Union {
    /// A union of `ranks` members.
    pub fn new(ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            ranks,
            rounds: Mutex::new(Rounds {
                entered: vec![None; ranks],
                open: false,
                closed: 0,
                capacity: 0,
                ended: false,
            }),
            entered: Condvar::new(),
            gathered: Mutex::new(Gathered {
                values: vec![None; ranks],
                rounds: 0,
                last: Arc::new(Vec::new()),
            }),
            gathering: Condvar::new(),
            wakes: (0..ranks).map(|_| OnceLock::new()).collect(),
            waking: (0..ranks).map(|_| AtomicBool::new(false)).collect(),
        })
    }

    /// How many ranks the union has.
    pub const fn ranks(&self) -> usize {
        self.ranks
    }

    /// States where `rank` is woken, once its executor exists.
    pub fn bind(&self, rank: usize, queue: Arc<JobQueue>, runner: RunnerId) -> Result<()> {
        let slot = self
            .wakes
            .get(rank)
            .ok_or_else(|| Error::Invalid(format!("rank {rank} is outside the union")))?;
        slot.set(Wake { queue, runner })
            .map_err(|_| Error::Invariant(format!("rank {rank} of the union was bound twice")))
    }

    /// Ends the union: every rank waiting in it, and every rank that arrives, is refused.
    pub fn end(&self) {
        self.lock_rounds().ended = true;
        self.entered.notify_all();
    }

    fn lock_rounds(&self) -> MutexGuard<'_, Rounds> {
        self.rounds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Enters the union's next step with `tokens` live tokens of this rank's own.
    ///
    /// Answers the step once every rank has entered it, or `None` for a rank with no tokens when no
    /// step is open. Blocks this rank's device thread until then, which is what keeps its device
    /// from running ahead of the sequence of steps.
    pub fn enter(&self, rank: usize, tokens: u32) -> Result<Option<Step>> {
        let mut rounds = self.lock_rounds();
        if rounds.ended {
            return Err(Error::Invariant("the expert union ended".into()));
        }
        if tokens == 0 && !rounds.open {
            return Ok(None);
        }
        let opened = !rounds.open;
        rounds.open = true;
        if rounds.entered[rank].replace(tokens).is_some() {
            return Err(Error::Invariant(format!(
                "rank {rank} entered one expert step twice"
            )));
        }
        if rounds.entered.iter().all(Option::is_some) {
            rounds.capacity = rounds.entered.iter().flatten().copied().max().unwrap_or(0);
            rounds.entered.iter_mut().for_each(|entry| *entry = None);
            rounds.open = false;
            rounds.closed += 1;
            let capacity = rounds.capacity;
            drop(rounds);
            self.entered.notify_all();
            return Ok(Some(Step { capacity }));
        }
        let round = rounds.closed;
        if opened {
            drop(rounds);
            for peer in (0..self.ranks).filter(|peer| *peer != rank) {
                self.wake(peer)?;
            }
            rounds = self.lock_rounds();
        }
        while rounds.closed == round && !rounds.ended {
            let (guard, waited) = self
                .entered
                .wait_timeout(rounds, STRAGGLER_REPORT)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            rounds = guard;
            if waited.timed_out() && rounds.closed == round {
                let missing: Vec<usize> = (0..self.ranks)
                    .filter(|peer| rounds.entered[*peer].is_none())
                    .collect();
                tracing::warn!(rank, ?missing, "an expert step waits for these ranks");
            }
        }
        if rounds.closed == round {
            return Err(Error::Invariant(
                "the expert union ended during a step".into(),
            ));
        }
        Ok(Some(Step {
            capacity: rounds.capacity,
        }))
    }

    /// Queues a join on `rank`'s device, unless one is queued already.
    fn wake(&self, rank: usize) -> Result<()> {
        let Some(wake) = self.wakes[rank].get() else {
            return Err(Error::Invariant(format!(
                "rank {rank} of the expert union is not bound to a device"
            )));
        };
        if self.waking[rank].swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        wake.queue.join_step(wake.runner)
    }

    /// Notes that `rank` took up its wake: the next round may queue another.
    pub fn woken(&self, rank: usize) {
        self.waking[rank].store(false, Ordering::Release);
    }

    /// Whether `rank`'s device has a command waiting.
    fn has_work(&self, rank: usize) -> bool {
        self.wakes[rank]
            .get()
            .is_some_and(|wake| wake.queue.has_pending())
    }

    /// Gathers each rank's `mine`, in rank order.
    fn gather(&self, rank: usize, mine: Vec<u8>) -> Arc<Vec<Vec<u8>>> {
        let mut gathered = self
            .gathered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        gathered.values[rank] = Some(mine);
        if gathered.values.iter().all(Option::is_some) {
            let values = gathered
                .values
                .iter_mut()
                .map(|value| value.take().unwrap_or_default())
                .collect();
            gathered.last = Arc::new(values);
            gathered.rounds += 1;
            let last = Arc::clone(&gathered.last);
            drop(gathered);
            self.gathering.notify_all();
            return last;
        }
        let round = gathered.rounds;
        while gathered.rounds == round {
            gathered = self
                .gathering
                .wait(gathered)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        Arc::clone(&gathered.last)
    }
}

/// What a package's collective reaches this rank's union through.
struct Channel {
    union: Arc<Union>,
    rank: usize,
}

/// `UniserveGroup::all_gather`.
unsafe extern "C" fn all_gather(
    handle: *mut c_void,
    mine: *const c_void,
    peers: *mut c_void,
    bytes: i64,
) -> i32 {
    // SAFETY: `handle` is the `Channel` this rank's `Member` holds for the transport's life.
    let channel = unsafe { &*handle.cast::<Channel>() };
    let Ok(bytes) = usize::try_from(bytes) else {
        return 1;
    };
    // SAFETY: the package hands `bytes` readable bytes at `mine`.
    let value = unsafe { std::slice::from_raw_parts(mine.cast::<u8>(), bytes) }.to_vec();
    let gathered = channel.union.gather(channel.rank, value);
    for (index, value) in gathered.iter().enumerate() {
        if value.len() != bytes {
            return 1;
        }
        // SAFETY: the package hands `[ranks][bytes]` writable bytes at `peers`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                value.as_ptr(),
                peers.cast::<u8>().add(index * bytes),
                bytes,
            );
        }
    }
    0
}

/// `UniserveGroup::barrier`.
unsafe extern "C" fn barrier(handle: *mut c_void) -> i32 {
    // SAFETY: `handle` is the `Channel` this rank's `Member` holds for the transport's life.
    let channel = unsafe { &*handle.cast::<Channel>() };
    channel.union.gather(channel.rank, Vec::new());
    0
}

/// One rank's member of a union: where it enters steps, and its transport.
pub struct Member {
    union: Arc<Union>,
    rank: usize,
    shape: TransportShape,
    backend: Backing,
    /// When this rank last entered a step with tokens of its own.
    served: Option<std::time::Instant>,
    /// The direction whose program creates the transport: a fused backend has no other.
    mode: ops::moe::exchange::Mode,
    /// Whether this rank walks a bucket eagerly in step with the others, for a warmup.
    warming: Arc<AtomicBool>,
}

/// The transport and what must outlive it.
struct Backing {
    /// Read by the package through the group structure.
    channel: Box<Channel>,
    group: Box<Group>,
    transport: Option<Transport>,
    views: Option<ExchangeViews>,
}

// SAFETY: a member moves once, with its runner, to the rank's device thread, and is used there alone;
// the raw addresses it holds are its own boxes and the package's transport.
unsafe impl Send for Member {}

impl Member {
    /// Rank `rank` of `union`, over a transport of `shape`.
    pub fn new(union: Arc<Union>, shape: TransportShape) -> Self {
        let rank = shape.rank as usize;
        let mut channel = Box::new(Channel {
            union: Arc::clone(&union),
            rank,
        });
        let group = Box::new(Group {
            handle: std::ptr::from_mut(channel.as_mut()).cast(),
            communicator: std::ptr::null_mut(),
            ranks: i64::from(shape.ranks),
            rank: i64::from(shape.rank),
            device: i64::from(shape.device),
            all_gather,
            barrier,
            error: [0; 512],
        });
        Self {
            union,
            rank,
            shape,
            backend: Backing {
                channel,
                group,
                transport: None,
                views: None,
            },
            served: None,
            mode: ops::moe::exchange::Mode::Dispatch,
            warming: Arc::new(AtomicBool::new(false)),
        }
    }

    /// States that this rank's walks are a warmup's: eager, and in step with the other ranks'.
    pub fn set_warming(&self, warming: bool) {
        self.warming.store(warming, Ordering::Release);
    }

    /// This member's transport is created by a fused backend's program.
    #[must_use]
    pub fn fused(mut self) -> Self {
        self.mode = ops::moe::exchange::Mode::Fused;
        self
    }

    /// The rank's place in the union.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Whether this rank only holds experts: every walk it takes is a join.
    pub const fn experts_only(&self) -> bool {
        self.shape.attention_ranks > 0 && self.shape.rank >= self.shape.attention_ranks
    }

    /// The transport's views, once [`Self::ensure`] created it.
    pub fn views(&self) -> Option<&ExchangeViews> {
        self.backend.views.as_ref()
    }

    /// Creates the transport from `generation`'s program, collectively over the union.
    ///
    /// Every rank reaches this at its first capture, in step with the others, because their plans
    /// capture the same buckets in the same order and the first one is where a program exists.
    pub fn ensure(
        &mut self,
        generation: &ops::Generation,
        stream: &platform::Stream,
    ) -> Result<()> {
        if self.backend.transport.is_some() {
            return Ok(());
        }
        let shape = self.shape;
        let descriptor = ops::moe::exchange::Descriptor {
            mode: self.mode,
            hidden: shape.hidden,
            intermediate: shape.intermediate,
            top: shape.top,
            experts: shape.experts,
            max_tokens: shape.max_tokens,
            ranks: shape.ranks,
            attention: shape.attention_ranks,
            dtype: shape.dtype,
        };
        let kernel = generation
            .op::<ops::moe::ExpertExchange>(&descriptor)
            .map_err(|error| Error::Invalid(format!("the expert exchange's program: {error}")))?;
        let device = platform::DeviceId(shape.device);
        let before = device.memory_info().map_err(Error::Cuda)?.free;
        let transport = Transport::create(
            Arc::clone(kernel),
            std::ptr::from_ref(self.backend.group.as_ref()),
            shape,
            stream,
        )
        .map_err(|error| Error::Invalid(format!("creating the expert transport: {error}")))?;
        let held = before.saturating_sub(device.memory_info().map_err(Error::Cuda)?.free);
        // The plan reserved a bound before any program existed, so this is where it is checked.
        // The figure is the device's, which another process can move, so it refuses nothing: the
        // rank's booking after warmup holds it to the quota with its own process's figure.
        let planned = shape.device_bytes(self.mode == ops::moe::exchange::Mode::Fused);
        if held > planned {
            tracing::warn!(
                rank = self.rank,
                held,
                planned,
                "the expert transport took more of the device than the capacity plan reserved"
            );
        } else {
            tracing::info!(
                rank = self.rank,
                held,
                planned,
                "the expert transport's storage"
            );
        }
        let views = transport
            .views()
            .map_err(|error| Error::Invalid(format!("reading the expert transport: {error}")))?;
        self.backend.views = Some(ExchangeViews {
            transport: views.transport,
            ranks: shape.ranks,
            attention: shape.attention_ranks,
            capacity: shape.max_tokens,
            step_rows: shape.max_tokens,
            states: views.states,
            ids: views.ids,
            weights: views.weights,
            combined: views.combined,
            live: views.live,
            packed: views.packed,
            packed_scales: views.packed_scales,
            fused: views.fused,
            source_only: shape.rank < shape.attention_ranks,
            meet: {
                let union = Arc::clone(&self.union);
                let warming = Arc::clone(&self.warming);
                let rank = self.rank;
                Arc::new(move || {
                    if warming.load(Ordering::Acquire) {
                        union.gather(rank, Vec::new());
                    }
                    Ok(())
                })
            },
        });
        self.backend.transport = Some(transport);
        tracing::info!(
            rank = self.rank,
            ranks = shape.ranks,
            max_tokens = shape.max_tokens,
            fused = views.fused,
            "the expert transport is mapped"
        );
        Ok(())
    }

    /// Enters the union's next step and writes its capacity on `stream`.
    ///
    /// `tokens` is this rank's live tokens; zero joins a peer's step, and answers `None` when no
    /// peer opened one.
    pub fn step(&mut self, tokens: u32, stream: &platform::Stream) -> Result<Option<Step>> {
        if tokens > 0 {
            self.served = Some(std::time::Instant::now());
        }
        let Some(step) = self.union.enter(self.rank, tokens)? else {
            return Ok(None);
        };
        self.write(step.capacity, stream)?;
        Ok(Some(step))
    }

    /// Writes `capacity` for a walk every rank runs in step, as a warmup's walks are.
    pub fn write(&mut self, capacity: u32, stream: &platform::Stream) -> Result<()> {
        self.backend
            .transport
            .as_mut()
            .ok_or_else(|| Error::Invariant("an expert step before the transport exists".into()))?
            .step(capacity, stream)
            .map_err(|error| Error::Invalid(format!("writing the step's capacity: {error}")))
    }

    /// Takes up this rank's wake, and answers whether it joins the step now.
    ///
    /// A serving rank whose next batch arrives within [`NEXT_BATCH_GRACE`] does not: the wake goes
    /// behind that batch, which enters the step with its tokens, and finds the step closed.
    pub fn woken(&self) -> Result<bool> {
        self.union.woken(self.rank);
        let serving = self
            .served
            .is_some_and(|served| served.elapsed() < SERVING_WINDOW);
        if !serving {
            return Ok(true);
        }
        let began = std::time::Instant::now();
        while began.elapsed() < NEXT_BATCH_GRACE {
            if self.union.has_work(self.rank) {
                self.union.wake(self.rank)?;
                return Ok(false);
            }
            std::hint::spin_loop();
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rank with no tokens finds no round open and answers at once, and a round opened by a rank
    /// with tokens closes when every rank entered, at the largest count.
    #[test]
    fn a_round_is_opened_by_tokens_and_closes_at_the_largest_count() {
        let union = Union::new(1);
        assert_eq!(union.enter(0, 0).expect("an idle entry"), None);
        assert_eq!(
            union.enter(0, 7).expect("a step of one rank"),
            Some(Step { capacity: 7 })
        );
    }

    /// A union of `ranks` ranks whose wakes land in queues nobody serves.
    fn bound(ranks: usize) -> (Arc<Union>, Vec<Arc<JobQueue>>) {
        let union = Union::new(ranks);
        let queues: Vec<Arc<JobQueue>> = (0..ranks)
            .map(|_| JobQueue::new(platform::DeviceId(0), crate::stream::EventPool::new()))
            .collect();
        for (rank, queue) in queues.iter().enumerate() {
            union
                .bind(rank, Arc::clone(queue), RunnerId(0))
                .expect("a rank binds once");
        }
        (union, queues)
    }

    /// Every rank of a step leaves it with the same capacity, the largest count any rank brought,
    /// and the rank that opened the round woke the other.
    #[test]
    fn the_ranks_of_a_step_agree_on_the_largest_count() {
        let (union, queues) = bound(2);
        let steps: Vec<_> = std::thread::scope(|scope| {
            let opener = {
                let union = Arc::clone(&union);
                scope.spawn(move || union.enter(0, 3).expect("the opener's step"))
            };
            while !queues[1].has_pending() {
                std::thread::yield_now();
            }
            let peer = union.enter(1, 9).expect("the peer's step");
            vec![opener.join().expect("the opener ended"), peer]
        });
        assert_eq!(steps, vec![Some(Step { capacity: 9 }); 2]);
    }

    /// A wake that finds its step closed costs nothing, and queues no second wake meanwhile.
    #[test]
    fn a_late_join_finds_no_step_and_opens_none() {
        let (union, _queues) = bound(2);
        assert_eq!(union.enter(1, 0).expect("an idle entry"), None);
        union.end();
        assert!(union.enter(0, 1).is_err(), "an ended union takes no step");
    }

    #[test]
    fn the_gather_answers_every_rank_in_rank_order() {
        let union = Union::new(3);
        let answers: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..3)
                .map(|rank| {
                    let union = &union;
                    scope.spawn(move || union.gather(rank, vec![rank as u8]))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });
        for answer in answers {
            assert_eq!(*answer, vec![vec![0u8], vec![1], vec![2]]);
        }
    }
}

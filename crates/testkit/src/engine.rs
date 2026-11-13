//! A fake engine at the request interface: the engine's own control thread, request interface,
//! schedulers and policies (`engine::serve`), over fakes of what lies below the schedulers.
//!
//! The binary's start loads ranks and opens a session with its host's Storage Server; a fake
//! engine's start ([`fake_start`]) builds the same [`Engine`] over:
//!
//! - the session's layout as the engine states one ([`engine::text_layout`]), for the dense cache of
//!   [`crate::cache`] and the fixture's units;
//! - an [`InProcessPool`] that opens the session, owned by a `buffer-pool` thread that turns it as
//!   the Storage Server's thread turns its pool (`specs/buffer-pool.md` §14.2, "One thread": a
//!   turn, then a sleep of at most 200 µs while nothing is due), and runs the work a case paces it
//!   with between two turns, until the engine's close ends the session;
//! - one [`FakeExecutor`] per scheduler, shaped by the fixture's knobs, and what each worker
//!   group serves ([`crate::capacity`]);
//! - the tree's `fcfs` policy ([`crate::packaged_policy`]) within the groups' ceilings.
//!
//! As the binary's start grants the segments its ranks created and grows the KV regions into
//! them, the pool's thread grants each device region the session opened short of its numbering
//! one segment of the whole numbering, and each scheduler grows the region by `Resize` before the
//! engine serves. The engine holds no rank: no worker of a device, a data plane of no rank, no
//! head of a session with a Storage Server and no Build Server (`Engine::new`), so an operation
//! on the ranks answers with no rank's result, an operation on the KV cache's segments is refused,
//! and a preparation fails.
//!
//! [`spawn`] runs the control thread on an `engine` thread of the test, over two socket pairs
//! whose other ends play the daemon.

use std::os::unix::net::UnixStream;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use ::engine::executor::{Executor, Shadow};
use ::engine::scheduler::{self, Message};
use ::engine::{Engine, Start};
use anyhow::{Context as _, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender};
use proto::config::EngineArgs;
use service::Connection;
use storage::QueuePair;
use storage::protocol::Segment;
use storage::runtime::{Open, Opened, Region, Tier};

use crate::{FakeExecutor, InProcessPool};

/// The KV units of each scheduler's region a fixture states unless a case states others, the
/// padding unit included: 255 blocks of [`crate::BLOCK_TOKENS`] tokens.
pub const UNITS: u32 = 256;

/// The bytes of each ring of a fake engine's queue pairs.
const RING_BYTES: usize = 1 << 20;

/// The longest the buffer pool's thread sleeps between two turns while nothing arrives.
const TURN_WAIT: Duration = Duration::from_micros(200);

/// The bytes of one request slot's rows on one rank: an accounting figure, since no rank holds
/// the memory.
const SLOT_BYTES: u64 = 1 << 20;

/// Work the buffer pool's thread runs between two turns: a knob of [`InProcessPool`] a case paces
/// the pool with.
pub type Pace = Box<dyn FnOnce(&mut InProcessPool) + Send>;

/// What a fake engine is made of besides its launch.
pub struct Fixture {
    /// The KV units of each scheduler's device region, the padding unit included.
    pub units: u32,
    /// Makes each scheduler's fake executor of the one [`FakeExecutor::new`] gives: the case's
    /// knobs. A case that paces the executors keeps a clone of each.
    pub executor: Box<dyn Fn(FakeExecutor) -> FakeExecutor + Send>,
    /// The work the buffer pool's thread runs between two turns, in order; a case that sends
    /// none leaves it [`crossbeam_channel::never`].
    pub pool: Receiver<Pace>,
}

impl Fixture {
    /// A fixture of [`UNITS`] units whose executors are `executor`'s, and whose pool no case
    /// paces.
    pub fn new(executor: impl Fn(FakeExecutor) -> FakeExecutor + Send + 'static) -> Self {
        Self {
            units: UNITS,
            executor: Box::new(executor),
            pool: crossbeam_channel::never(),
        }
    }
}

/// The start of a fake engine of `fixture` (module documentation): what `engine::serve` builds
/// the started epoch with, from the launch it serves.
pub fn fake_start(fixture: Fixture) -> Start {
    Box::new(move |launch: &EngineArgs| start(launch, fixture))
}

/// Runs the engine of `launch` that `start` builds on an `engine` thread: the request
/// interface's hello and `Starting` ([`engine::connect`]), then the control thread
/// ([`engine::serve`]) until the daemon shuts the engine down or closes both interfaces. Returns
/// the daemon's end of the request interface, which the engine's hello waits on, the daemon's
/// end of the process interface, and the thread, which returns how the engine ended; a failure
/// is also written to standard error, where a test run carries it beside its assertion.
///
/// # Errors
///
/// When a socket pair or the thread cannot be made.
pub fn spawn(
    launch: EngineArgs,
    start: Start,
) -> Result<(Connection, Connection, JoinHandle<Result<()>>)> {
    let (requests, engine_requests) = UnixStream::pair().context("a request socket pair")?;
    let (process, engine_process) = UnixStream::pair().context("a process socket pair")?;
    let mut engine_requests = Connection::new(engine_requests)?;
    let engine_process = Connection::new(engine_process)?;

    let thread = std::thread::Builder::new()
        .name("engine".to_owned())
        .spawn(move || {
            let served = ::engine::connect(&mut engine_requests, &launch)
                .and_then(|()| ::engine::serve(engine_requests, engine_process, launch, start));
            if let Err(error) = &served {
                eprintln!("the test's engine ended: {error:#}");
            }
            served
        })
        .context("starting the engine's thread")?;
    Ok((
        Connection::new(requests)?,
        Connection::new(process)?,
        thread,
    ))
}

/// Builds the engine of `launch` over the fakes of `fixture` (module documentation).
fn start(launch: &EngineArgs, fixture: Fixture) -> Result<Engine> {
    let Fixture {
        units,
        executor,
        pool,
    } = fixture;
    let cache = crate::cache();
    let open = ::engine::text_layout(launch, &cache, units, 1, SLOT_BYTES)?;

    // Each scheduler's fake is made before the scheduler and its inbox, since the buffer pool
    // opens the session over the fakes and the schedulers over its answer: it sends to a channel
    // of its own, which a relay forwards to the inbox the engine makes for its scheduler.
    let classes: Vec<(u8, u32)> = (0..)
        .zip(&cache.groups)
        .map(|(class, group)| (class, group.block_size))
        .collect();
    let mut fakes = Vec::with_capacity(open.schedulers.len());
    let mut relayed = Vec::with_capacity(open.schedulers.len());
    for groups in &open.schedulers {
        let (sender, receiver) = crossbeam_channel::unbounded();
        fakes.push(executor(FakeExecutor::new(groups, classes.clone(), sender)));
        relayed.push(Some(receiver));
    }
    let (session, opened, pairs) = serve_pool(&open, fakes.clone(), pool)?;

    let capacity = crate::capacity(launch, &cache);
    let capacities: Vec<_> = open
        .groups
        .iter()
        .map(|group| (group.id, capacity.clone()))
        .collect();
    let ceilings = scheduler::policy::ceilings(capacities.iter().map(|(_, capacity)| capacity));
    let policy = crate::packaged_policy(&ceilings)?;

    let mut inboxes = Vec::with_capacity(open.schedulers.len());
    let engine = Engine::new(
        launch,
        &open,
        &opened,
        pairs,
        capacities,
        policy,
        |index, _, inbox| {
            let fake = fakes
                .get(index)
                .cloned()
                .with_context(|| format!("no fake executor of scheduler {index}"))?;
            let receiver = relayed
                .get_mut(index)
                .and_then(Option::take)
                .with_context(|| format!("scheduler {index}'s fake is relayed once"))?;
            relay(receiver, inbox.clone())?;
            inboxes.push(inbox);

            let shadow = Arc::new(Mutex::new(Shadow::default()));
            Ok((Box::new(fake) as Box<dyn Executor>, shadow))
        },
        session,
    )?;

    grow(&inboxes, &open, &opened)?;
    Ok(engine)
}

/// Forwards every message a fake sends to `receiver` to its scheduler's `inbox`, in order, on a
/// `relay` thread that ends once every clone of the fake is gone.
///
/// # Errors
///
/// When the thread cannot start.
fn relay(receiver: Receiver<Message>, inbox: Sender<Message>) -> Result<()> {
    std::thread::Builder::new()
        .name("relay".to_owned())
        .spawn(move || {
            for message in receiver {
                if inbox.send(message).is_err() {
                    return;
                }
            }
        })
        .context("starting a fake executor's relay")?;
    Ok(())
}

/// The units a device region of `stated` the session opened with `units` grows to, when it
/// opened short of its numbering: every unit past its reserved ones. A KV region is such a
/// region, which the layout opens with none.
fn short(stated: &Region, units: u32) -> Option<u32> {
    let numbered = stated.limit.saturating_sub(stated.reserved);
    (stated.tier == Tier::Device && units < numbered).then_some(numbered)
}

/// Starts the `buffer-pool` thread (module documentation) over a new [`InProcessPool`] that opens
/// the session of `open` over `fakes` and grants each region opened [`short`] one segment of its
/// whole numbering, then turns until the session ends, running `pace`'s work between two turns.
/// Returns what ends the session (the pool's thread stops, closing it, and is joined), the answer
/// to `Open` and the engine's end of each scheduler's queue pair.
///
/// # Errors
///
/// When the thread cannot start, or the pool refuses the layout or a grant.
#[expect(
    clippy::type_complexity,
    reason = "what ends the session, as `Engine::new` takes it, with the session's answer"
)]
fn serve_pool(
    open: &Open,
    fakes: Vec<FakeExecutor>,
    pace: Receiver<Pace>,
) -> Result<(
    Box<dyn FnOnce() -> Result<()> + Send>,
    Opened,
    Vec<QueuePair>,
)> {
    let (answer, answered) = crossbeam_channel::bounded(1);
    // The session ends when this channel's sender goes, sent nothing: at the session's end, or
    // with an engine that was never built.
    let (end, ended) = crossbeam_channel::bounded::<()>(0);
    let layout = open.clone();

    let thread = std::thread::Builder::new()
        .name("buffer-pool".to_owned())
        .spawn(move || {
            // The pool is not `Send`: it lives on this thread, which opens the session.
            let mut pool = InProcessPool::new(RING_BYTES);
            let opened = pool.open(&layout, fakes).and_then(|(opened, pairs)| {
                grant(&mut pool, &layout, &opened).map(|()| (opened, pairs))
            });
            let open = opened.is_ok();
            if answer.send(opened).is_err() || !open {
                return;
            }
            turn(&mut pool, &ended, pace);
            pool.close(layout.epoch);
        })
        .context("starting the buffer pool's thread")?;

    let (opened, pairs) = answered
        .recv()
        .context("the buffer pool's thread ended before it opened the session")?
        .map_err(|refusal| anyhow!("the buffer pool refused the session: {refusal}"))?;
    let session: Box<dyn FnOnce() -> Result<()> + Send> = Box::new(move || {
        drop(end);
        thread
            .join()
            .map_err(|_| anyhow!("the buffer pool's thread panicked"))
    });
    Ok((session, opened, pairs))
}

/// Grants each device region of the session of `open` that `opened` states [`short`] one segment
/// of its whole numbering, as the head grants the segments the ranks created.
///
/// # Errors
///
/// The pool's refusal of a grant.
fn grant(pool: &mut InProcessPool, open: &Open, opened: &Opened) -> Result<(), String> {
    for (stated, &(number, units)) in open.regions.iter().zip(&opened.regions) {
        if short(stated, units).is_some() {
            let segment = Segment {
                epoch: open.epoch,
                ordinal: 0,
            };
            pool.pool()
                .grant_segment(number, segment, 0, stated.limit)?;
        }
    }
    Ok(())
}

/// Turns `pool` as the Storage Server's thread turns its pool, until `ended` closes: a turn, then
/// a wait of at most [`TURN_WAIT`], during which a case's work from `pace` runs at once. A queue
/// pair the engine broke stops the turns, which the test's engine then waits on.
fn turn(pool: &mut InProcessPool, ended: &Receiver<()>, pace: Receiver<Pace>) {
    let mut pace = pace;
    loop {
        if let Err(error) = pool.turn() {
            eprintln!("the test's buffer pool stops: a queue pair is broken: {error}");
            return;
        }

        let mut paced = true;
        crossbeam_channel::select! {
            recv(ended) -> _ => return,
            recv(pace) -> work => match work {
                Ok(work) => work(pool),
                Err(_) => paced = false,
            },
            default(TURN_WAIT) => {}
        }
        // A case that let go of its sender paces nothing more.
        if !paced {
            pace = crossbeam_channel::never();
        }
    }
}

/// Grows each device region of the session of `open` that `opened` states [`short`] to its whole
/// numbering, by a `Resize` to the scheduler of `inboxes` whose queue pair carries the region's
/// capacity words, and awaits every reply: no unit enters a region before its segment is granted.
///
/// # Errors
///
/// When a scheduler stopped, or a region grew short of its numbering.
fn grow(inboxes: &[Sender<Message>], open: &Open, opened: &Opened) -> Result<()> {
    let mut replies = Vec::new();
    for (scheduler, registers) in opened.registers.iter().enumerate() {
        for &(number, _) in &registers.capacity {
            let Some(index) = opened.regions.iter().position(|&(n, _)| n == number) else {
                continue;
            };
            let Some(target) = short(&open.regions[index], opened.regions[index].1) else {
                continue;
            };
            let inbox = inboxes
                .get(scheduler)
                .with_context(|| format!("no inbox of scheduler {scheduler}"))?;
            let (reply, replied) = crossbeam_channel::bounded(1);
            inbox
                .send(Message::Resize(number, target, reply))
                .map_err(|_| anyhow!("scheduler {scheduler} stopped"))?;
            replies.push((scheduler, number, target, replied));
        }
    }

    for (scheduler, number, target, replied) in replies {
        let reached = replied
            .recv()
            .map_err(|_| anyhow!("scheduler {scheduler} stopped before region {number} grew"))?;
        if reached < target {
            bail!("region {number} grew to {reached} of its {target} units");
        }
    }
    Ok(())
}

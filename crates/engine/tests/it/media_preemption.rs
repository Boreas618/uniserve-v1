//! Video requests preempted at denoising-step boundaries and resumed from
//! the trajectories stored in the host latent pools, over the GPU-free
//! simulator and a Storage Server in the test's process.
//!
//! The simulator copies no bytes: what the tests observe is what the
//! scheduler asks of the workers at the executor boundary (the trajectory
//! offloads a batch carries, the stored pages a latent preparation restores,
//! the denoising calls a request runs) and the events its requests end
//! with. A preemption in normal serving (a sleep) and a world change (a
//! park, and a successor scheduler that resumes the request) take the same
//! path, whatever parallel configuration the successor's denoiser has, and
//! a request whose stored trajectory is missing starts from noise.
//!
//! The simulated denoising ranks state how their request slot pages hold a
//! request's samples as executors do (`CallOutput::latent_layouts`): one
//! `[48, 16]` float32 video sample, of which each rank holds a run of rows
//! from its first page on, three pages for the whole sample.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use engine::{
    BatchEvent, BatchResult, CacheChange, CommandSender, Event, EventRx, ExecutionBatch, Executor,
    ExecutorError, ExecutorInfo, ExecutorSubmitError, Request, SchedulerConfig, SimEngine,
    SimExecutor, Trajectory, WorkerGroupId,
};
use ipc::{
    CallKind, LatentPageCopy, LatentPoolInfo, LatentPoolRef, LatentRestore, MediaCall,
    RankLatentLayout, RequestId,
};
use proto::request::MigrationCause;

use crate::support;
use crate::support::storage::{DEPLOYMENT, HOST, StorageServer};
use crate::support::video::{video_request, video_worker_denoising_on};

/// Denoising steps of the simulated denoiser: enough that a preemption
/// after the second lands short of the last.
const STEPS: u32 = 6;

/// Latent pages one request slot owns on a denoising rank.
const SLOT_PAGES: u32 = 4;

/// Float32 values of one latent page (`video_worker`).
const PAGE_UNITS: u64 = 256;

/// Rows and channels of the simulated video sample: 64 bytes a row, three
/// pages of 1 KiB for all of them.
const ROWS: u64 = 48;
const WIDTH: u64 = 16;

/// The whole sample on one rank, and its rows on each of two
/// sequence-parallel ranks: two pages each.
const ONE_RANK: &[(u64, u64)] = &[(0, ROWS)];
const TWO_RANKS: &[(u64, u64)] = &[(0, 26), (26, ROWS - 26)];

/// Denoising batches submitted before the gate closes.
const BEFORE_PREEMPTION: u32 = 2;

/// The identity of what the simulated denoising computes, as a rank reports
/// it in its host latent pool.
const IDENTITY: &str = "denoiser";

/// The trajectory every test's request carries.
const TRAJECTORY: u64 = 77;

/// Holds the scheduler's submissions back once it submitted
/// [`BEFORE_PREEMPTION`] denoising batches, until the test opens it, so that
/// a preemption the test asks for meanwhile meets the request at a known
/// boundary: past the steps submitted, short of the last.
#[derive(Default)]
struct Gate {
    closed: AtomicBool,
    reached: AtomicBool,
    denoising: AtomicU32,
}

impl Gate {
    fn open(&self) {
        self.closed.store(false, Ordering::Release);
    }

    /// Waits until the gate closed.
    fn wait_reached(&self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !self.reached.load(Ordering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "the denoising never reached the gate"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }
}

/// The simulator behind a [`Gate`].
struct GatedExecutor {
    inner: SimExecutor,
    gate: Option<Arc<Gate>>,
}

impl Executor for GatedExecutor {
    fn info(&self) -> &ExecutorInfo {
        self.inner.info()
    }

    fn is_ready(&self, worker: &WorkerGroupId) -> bool {
        self.inner.is_ready(worker)
    }

    fn has_capacity(&self, worker: &WorkerGroupId) -> bool {
        !self
            .gate
            .as_ref()
            .is_some_and(|gate| gate.closed.load(Ordering::Acquire))
            && self.inner.has_capacity(worker)
    }

    fn command_has_capacity(&self, command: &ipc::BatchCommand) -> bool {
        self.inner.command_has_capacity(command)
    }

    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        let denoises = batch
            .requests
            .iter()
            .any(|(call, _)| call.kind == CallKind::Media(MediaCall::Denoising));
        self.inner.submit(batch)?;
        if let Some(gate) = &self.gate
            && denoises
            && gate.denoising.fetch_add(1, Ordering::AcqRel) + 1 == BEFORE_PREEMPTION
        {
            gate.closed.store(true, Ordering::Release);
            gate.reached.store(true, Ordering::Release);
        }
        Ok(())
    }

    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchResult>, ExecutorError> {
        self.inner.poll(timeout)
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        self.inner.close()
    }

    fn hook(&mut self, name: &str, payload: &str) -> Result<Vec<String>, ExecutorError> {
        self.inner.hook(name, payload)
    }
}

/// How a rank holding rows `start..start + count` of the sample lays them
/// out in its slot pages (`storage::manifest::StorageLayout`), from byte
/// zero on, as a video executor declares it.
fn share(start: u64, count: u64) -> String {
    serde_json::json!({
        "bytes": count * WIDTH * 4,
        "fields": [{
            "tensor": "video_latents",
            "source": "",
            "dtype": "float32",
            "element": 4,
            "offset": 0,
            "shape": [count, WIDTH],
            "strides": [WIDTH, 1],
            "logical": {
                "shape": [ROWS, WIDTH],
                "dims": ["row", "channel"],
                "units": [1, 0],
                "segments": [
                    [{"span": {"start": start, "count": count, "step": 1}, "local": 0}],
                    [{"span": {"start": 0, "count": WIDTH, "step": 1}, "local": 0}],
                ],
            },
            "encoding": {"kind": "affine"},
        }],
    })
    .to_string()
}

/// The pages a rank holding `count` rows stores.
fn pages_of(count: u64) -> u32 {
    (count * WIDTH * 4).div_ceil(PAGE_UNITS * 4) as u32
}

/// A scheduler of a simulated video denoiser whose ranks hold `shares` of
/// the sample and map the host latent pool `pool` of `server`, reporting
/// `identity`, with its loop running.
struct Denoiser {
    handle: CommandSender,
    thread: thread::JoinHandle<bool>,
    boundary: Receiver<BatchEvent>,
    /// The latent seed the scheduler keys trajectories by.
    seed: u64,
}

impl Denoiser {
    fn start(
        server: &StorageServer,
        pool: u64,
        identity: &str,
        gate: Option<Arc<Gate>>,
        shares: &[(u64, u64)],
    ) -> Self {
        let config = SchedulerConfig {
            max_batch: 32,
            ..Default::default()
        };
        Self::start_with(server, pool, identity, gate, shares, config)
    }

    /// [`Self::start`] with the scheduler's `config`.
    fn start_with(
        server: &StorageServer,
        pool: u64,
        identity: &str,
        gate: Option<Arc<Gate>>,
        shares: &[(u64, u64)],
        config: SchedulerConfig,
    ) -> Self {
        let mut sim = video_worker_denoising_on(1, 1, shares.len());
        let info = sim.mut_info_for_test();
        if let Some(denoiser) = info.video_denoiser.as_mut() {
            denoiser.denoise_steps = STEPS;
        }
        // Every rank maps the one pool; each rank's share determines its
        // pages, and its region names the share.
        sim.set_latent_pools(
            shares
                .iter()
                .map(|&(start, count)| {
                    (
                        HOST.to_owned(),
                        Some(LatentPoolInfo {
                            pool: pool_ref(pool),
                            slot_pages: SLOT_PAGES,
                            region: [start as u8, count as u8, shares.len() as u8, 0]
                                .repeat(8)
                                .try_into()
                                .unwrap(),
                            identity: identity.to_owned(),
                            endpoint: None,
                        }),
                    )
                })
                .collect(),
        );
        sim.set_latent_layouts(
            shares
                .iter()
                .enumerate()
                .map(|(rank, &(start, count))| RankLatentLayout {
                    rank: rank as u32,
                    layout: share(start, count),
                })
                .collect(),
        );
        let mut inner = SimExecutor::new(sim);
        let boundary = inner.observe();
        let mut scheduler = support::scheduler_with(
            Box::new(GatedExecutor { inner, gate }),
            SimEngine::special_tokens(),
            config,
        )
        .unwrap();
        scheduler.enable_trajectory_store(server.socket(), DEPLOYMENT, 0, HOST);
        let seed = scheduler.latent_seed();
        assert_ne!(seed, 0, "the host latent pool serves the denoiser");
        let (tx, rx) = crossbeam_channel::unbounded();
        let thread = thread::spawn(move || scheduler.run(rx));
        Self {
            handle: CommandSender::new(tx),
            thread,
            boundary,
            seed,
        }
    }

    /// Shuts the scheduler down and returns every batch it submitted.
    fn stop(self) -> Vec<ExecutionBatch> {
        self.handle.shutdown();
        assert!(!self.thread.join().unwrap(), "the engine loop died");
        self.boundary
            .iter()
            .filter_map(|event| match event {
                BatchEvent::Submitted(batch) => Some(batch),
                BatchEvent::Resolved { .. } => None,
            })
            .collect()
    }
}

/// The pool the fixture's server mapped as `pool`.
fn pool_ref(pool: u64) -> LatentPoolRef {
    LatentPoolRef {
        pool,
        shape: [7; 32],
        numa_node: None,
        page_bytes: PAGE_UNITS * 4,
    }
}

/// A server with one latent pool, which holds a slot's pages many times.
fn server() -> (StorageServer, u64) {
    let mut server = StorageServer::start(1 << 20);
    let pool = server.latent_pool("sim", PAGE_UNITS * 4, 64);
    (server, pool)
}

/// The `t2va` request every test submits, resuming `trajectory`.
fn request(trajectory: Trajectory) -> Request {
    numbered(1, trajectory)
}

/// The `t2va` request `id`, resuming `trajectory`.
fn numbered(id: u64, trajectory: Trajectory) -> Request {
    let Request::Diffusion(mut request) = video_request(id, 1) else {
        unreachable!("a video request");
    };
    request.trajectory = trajectory;
    Request::Diffusion(request)
}

/// The requests `batches` carry calls of, in the order of their first call.
fn first_calls(batches: &[ExecutionBatch]) -> Vec<RequestId> {
    let mut order = Vec::new();
    for (call, _) in batches.iter().flat_map(|batch| &batch.requests) {
        let id = call.request_key.request_id;
        if !order.contains(&id) {
            order.push(id);
        }
    }
    order
}

/// The events of `events` up to its terminal one, within thirty seconds.
fn until_terminal(events: &mut EventRx) -> Vec<Event> {
    let mut received = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(event) => {
                let terminal = event.is_terminal();
                received.push(event);
                if terminal {
                    return received;
                }
            }
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("the request did not end: {received:?}");
}

/// The trajectory a request left with, from its migratable end: the steps
/// stored and the record of where their pages lie, which the daemon
/// submits it again with.
fn left_with(events: &[Event], expected: MigrationCause, trajectory: u64) -> Trajectory {
    match events.last() {
        Some(Event::Migratable {
            cause,
            denoised_steps,
            stored_layout,
        }) if *cause == expected => Trajectory {
            id: trajectory,
            denoised_steps: *denoised_steps,
            state_seed: None,
            stored_layout: stored_layout.clone(),
        },
        other => panic!("expected a migratable end, got {other:?}"),
    }
}

/// Whether the request completed with its video.
fn completed(events: &[Event]) -> bool {
    events
        .iter()
        .any(|event| matches!(event, Event::Artifact(_)))
        && matches!(events.last(), Some(Event::Finished { .. }))
}

/// What the workers were asked of a request's admissions, each in order of
/// admission (its request epoch): the trajectory pages its batches
/// offloaded, the stored trajectory its latent preparation restored, and its
/// denoising calls.
#[derive(Debug, Default)]
struct Admission {
    offloaded: Vec<LatentPageCopy>,
    restored: Option<LatentRestore>,
    prepared: bool,
    denoising: u32,
}

/// Request 1's admissions.
fn admissions(batches: &[ExecutionBatch]) -> Vec<Admission> {
    admissions_of(batches, RequestId(1))
}

/// Request `request`'s admissions.
fn admissions_of(batches: &[ExecutionBatch], request: RequestId) -> Vec<Admission> {
    /// The admission of request epoch `epoch`, recorded from now on when new.
    fn of(epochs: &mut Vec<(u64, Admission)>, epoch: u64) -> &mut Admission {
        let index = match epochs.iter().position(|(known, _)| *known == epoch) {
            Some(index) => index,
            None => {
                epochs.push((epoch, Admission::default()));
                epochs.len() - 1
            }
        };
        &mut epochs[index].1
    }

    let mut epochs: Vec<(u64, Admission)> = Vec::new();
    for batch in batches {
        for (call, _) in &batch.requests {
            if call.request_key.request_id != request {
                continue;
            }
            let admission = of(&mut epochs, call.request_key.request_epoch);
            match call.kind {
                CallKind::Media(MediaCall::LatentPreparation) => {
                    admission.prepared = true;
                    admission.restored = call.latent_restore.clone();
                }
                CallKind::Media(MediaCall::Denoising) => admission.denoising += 1,
                _ => {}
            }
        }
        for offload in batch
            .page_offloads
            .iter()
            .flat_map(|offloads| &offloads.latents)
            .filter(|offload| offload.request_key.request_id == request)
        {
            of(&mut epochs, offload.request_key.request_epoch)
                .offloaded
                .extend(offload.pages.iter().cloned());
        }
    }
    epochs.into_iter().map(|(_, admission)| admission).collect()
}

/// Whether `copies` wrote every page each rank of `shares` holds of the
/// sample to `pool`, and no other.
fn every_page(copies: &[LatentPageCopy], pool: u64, shares: &[(u64, u64)]) -> bool {
    let mut written: Vec<(u32, u32)> = copies
        .iter()
        .filter(|copy| copy.pool == pool)
        .map(|copy| (copy.rank, copy.page))
        .collect();
    written.sort_unstable();
    let expected: Vec<(u32, u32)> = shares
        .iter()
        .enumerate()
        .flat_map(|(rank, &(_, count))| (0..pages_of(count)).map(move |page| (rank as u32, page)))
        .collect();
    copies.len() == expected.len() && written == expected
}

/// The stored ranks a restore names, each as the restoring ranks that read
/// it and the pages pinned for them.
fn sources(restore: &LatentRestore) -> Vec<(Vec<u32>, usize)> {
    restore
        .sources
        .iter()
        .map(|source| (source.readers.clone(), source.slots.len()))
        .collect()
}

/// Preempts request 1, which starts trajectory [`TRAJECTORY`] on a
/// scheduler whose ranks hold `shares`, by parking it at a step boundary
/// past the first and short of the last, and returns the trajectory it left
/// with, the parked scheduler's latent seed and its admission.
fn park(server: &StorageServer, pool: u64, shares: &[(u64, u64)]) -> (Trajectory, u64, Admission) {
    let gate = Arc::new(Gate::default());
    let serving = Denoiser::start(server, pool, IDENTITY, Some(Arc::clone(&gate)), shares);
    let started = Trajectory {
        id: TRAJECTORY,
        ..Trajectory::default()
    };
    let mut events = serving.handle.submit(request(started)).unwrap();
    gate.wait_reached();
    let parked = serving.handle.park();
    gate.open();
    let stored = left_with(
        &until_terminal(&mut events),
        MigrationCause::Drain,
        TRAJECTORY,
    );
    assert!(
        (BEFORE_PREEMPTION..STEPS).contains(&stored.denoised_steps),
        "the request is preempted past the steps submitted and short of the last: {}",
        stored.denoised_steps
    );
    parked
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler parked")
        .expect("the park succeeded");
    let seed = serving.seed;
    let [preempted] = admissions(&serving.stop())
        .try_into()
        .expect("the request was admitted once");
    assert!(
        every_page(&preempted.offloaded, pool, shares),
        "{preempted:?}"
    );
    (stored, seed, preempted)
}

/// A sleep, which gives back the workers' storage in normal serving,
/// preempts a denoising video request at its next step boundary: the
/// trajectory's pages reach the host latent pool, and the request leaves
/// with the steps it took. Submitted again once the workers woke, it
/// continues from that step: its latent preparation restores every page and
/// only the steps left are denoised, and it completes with its video.
#[test]
fn a_sleep_preempts_a_video_request_which_resumes_from_its_stored_step() {
    let (server, pool) = server();
    let gate = Arc::new(Gate::default());
    let denoiser = Denoiser::start(&server, pool, IDENTITY, Some(Arc::clone(&gate)), ONE_RANK);
    let trajectory = Trajectory {
        id: TRAJECTORY,
        ..Trajectory::default()
    };
    let mut events = denoiser.handle.submit(request(trajectory)).unwrap();

    gate.wait_reached();
    let slept = denoiser.handle.resize_cache(CacheChange::Sleep);
    gate.open();
    let stored = left_with(
        &until_terminal(&mut events),
        MigrationCause::Drain,
        TRAJECTORY,
    );
    assert!(
        (BEFORE_PREEMPTION..STEPS).contains(&stored.denoised_steps),
        "the request is preempted past the steps submitted and short of the last: {}",
        stored.denoised_steps
    );
    slept
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler slept")
        .expect("the sleep succeeded");
    denoiser
        .handle
        .resize_cache(CacheChange::Wake)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler woke")
        .expect("the wake succeeded");

    let steps = stored.denoised_steps;
    let mut resumed = denoiser.handle.submit(request(stored)).unwrap();
    assert!(completed(&until_terminal(&mut resumed)));

    let [preempted, resumed] = admissions(&denoiser.stop())
        .try_into()
        .expect("the request was admitted twice");
    assert_eq!(preempted.denoising, steps);
    assert!(
        every_page(&preempted.offloaded, pool, ONE_RANK),
        "{preempted:?}"
    );
    let restore = resumed
        .restored
        .expect("the preparation restores the trajectory");
    assert_eq!(restore.steps, steps);
    assert_eq!(sources(&restore), [(vec![0], pages_of(ROWS) as usize)]);
    assert_eq!(resumed.denoising, STEPS - steps);
    assert!(resumed.offloaded.is_empty());
}

/// A request that names a trajectory stored where no pool holds it any
/// more starts from noise: its latent preparation restores nothing, every
/// step is denoised, and it completes. So does one that names stored steps
/// without the record of where they lie.
#[test]
fn a_video_request_whose_stored_trajectory_is_missing_starts_from_noise() {
    // A trajectory stored on one host's server, and a successor whose pools
    // lie at another server that never held it.
    let (stored_on, pool) = server();
    let (stored, _, _) = park(&stored_on, pool, TWO_RANKS);
    let (elsewhere, other_pool) = server();
    let denoiser = Denoiser::start(&elsewhere, other_pool, IDENTITY, None, ONE_RANK);
    let mut events = denoiser.handle.submit(request(stored.clone())).unwrap();
    assert!(completed(&until_terminal(&mut events)));
    let mut unrecorded = denoiser
        .handle
        .submit(numbered(
            2,
            Trajectory {
                stored_layout: Vec::new(),
                ..stored
            },
        ))
        .unwrap();
    assert!(completed(&until_terminal(&mut unrecorded)));

    let batches = denoiser.stop();
    for request in [RequestId(1), RequestId(2)] {
        let [admission] = admissions_of(&batches, request)
            .try_into()
            .expect("the request was admitted once");
        assert!(admission.prepared);
        assert_eq!(admission.restored, None);
        assert_eq!(admission.denoising, STEPS);
    }
}

/// A request that resumes a stored trajectory is admitted before a request
/// that starts one, whichever was submitted first: the request slots take
/// the resumed requests first, so the trajectories a world still has to
/// restore are at most those of its latest preemption.
#[test]
fn a_resumed_video_request_is_admitted_before_one_that_starts() {
    let (server, pool) = server();
    let gate = Arc::new(Gate::default());
    let config = SchedulerConfig {
        max_batch: 32,
        max_running_requests: 1,
        ..Default::default()
    };
    let denoiser = Denoiser::start_with(
        &server,
        pool,
        IDENTITY,
        Some(Arc::clone(&gate)),
        ONE_RANK,
        config,
    );

    // Request 1 holds the one running request while the others queue: one
    // that starts its trajectory, then one that resumes another's.
    let mut running = denoiser
        .handle
        .submit(numbered(1, Trajectory::default()))
        .unwrap();
    gate.wait_reached();
    let mut queued = denoiser
        .handle
        .submit_all([
            numbered(2, Trajectory::default()),
            numbered(
                3,
                Trajectory {
                    id: TRAJECTORY,
                    denoised_steps: 3,
                    ..Trajectory::default()
                },
            ),
        ])
        .unwrap();
    let mut resuming = queued.pop().expect("the resumed request's events");
    let mut starting = queued.pop().expect("the starting request's events");
    gate.open();
    for events in [&mut running, &mut starting, &mut resuming] {
        assert!(completed(&until_terminal(events)));
    }

    assert_eq!(
        first_calls(&denoiser.stop()),
        [RequestId(1), RequestId(3), RequestId(2)]
    );
}

/// A world change: the serving scheduler parks, preempting the request at a
/// step boundary with its trajectory stored, and a successor whose
/// denoising computes the same continues it from that step. A successor
/// whose denoising computes other numerics finds no trajectory of its own
/// latent seed and starts from noise (`in_flight = recompute`), unless the
/// request names the parked world's seed (`in_flight = keep`), whose
/// trajectory it continues.
#[test]
fn a_parked_worlds_video_request_continues_on_its_successor() {
    let (server, pool) = server();
    let (resumed, parked_seed, _) = park(&server, pool, ONE_RANK);
    let steps = resumed.denoised_steps;

    // The same numerics: the successor continues the trajectory.
    let successor = Denoiser::start(&server, pool, IDENTITY, None, ONE_RANK);
    assert_eq!(successor.seed, parked_seed);
    let mut events = successor.handle.submit(request(resumed.clone())).unwrap();
    assert!(completed(&until_terminal(&mut events)));
    let [same] = admissions(&successor.stop()).try_into().unwrap();
    assert_eq!(same.restored.map(|restore| restore.steps), Some(steps));
    assert_eq!(same.denoising, STEPS - steps);

    // Other numerics: the trajectory is recomputed, or kept when the
    // request names the parked world's seed.
    let other = Denoiser::start(&server, pool, "another denoiser", None, ONE_RANK);
    assert_ne!(other.seed, parked_seed);
    let mut recomputed = other.handle.submit(request(resumed.clone())).unwrap();
    assert!(completed(&until_terminal(&mut recomputed)));
    let mut kept = other
        .handle
        .submit(request(Trajectory {
            state_seed: Some(parked_seed),
            ..resumed
        }))
        .unwrap();
    assert!(completed(&until_terminal(&mut kept)));
    let [recomputed, kept] = admissions(&other.stop()).try_into().unwrap();
    assert_eq!(recomputed.restored, None);
    assert_eq!(recomputed.denoising, STEPS);
    assert_eq!(kept.restored.map(|restore| restore.steps), Some(steps));
    assert_eq!(kept.denoising, STEPS - steps);
}

/// A world change of the denoiser's parallel configuration whose denoising
/// computes the same: the serving world's two sequence-parallel ranks
/// store their rows, and a successor of one rank continues the trajectory
/// from that step, its one rank reading both stored ranks' pages; the
/// reverse change hands the one stored rank's pages to both ranks. The
/// configurations' latent seeds are one, so the successor finds the pages
/// by the record the request left with.
#[test]
fn a_successor_of_another_parallel_configuration_continues_the_trajectory() {
    // Ulysses 2 to one rank.
    let (forward, pool) = server();
    let (resumed, parked_seed, _) = park(&forward, pool, TWO_RANKS);
    let steps = resumed.denoised_steps;
    let successor = Denoiser::start(&forward, pool, IDENTITY, None, ONE_RANK);
    assert_eq!(successor.seed, parked_seed);
    let mut events = successor.handle.submit(request(resumed)).unwrap();
    assert!(completed(&until_terminal(&mut events)));
    let [continued] = admissions(&successor.stop()).try_into().unwrap();
    let restore = continued
        .restored
        .expect("the preparation restores the trajectory");
    assert_eq!(restore.steps, steps);
    assert_eq!(
        sources(&restore),
        TWO_RANKS
            .iter()
            .map(|&(_, count)| (vec![0], pages_of(count) as usize))
            .collect::<Vec<_>>()
    );
    assert_eq!(continued.denoising, STEPS - steps);

    // One rank to Ulysses 2.
    let (reverse, pool) = server();
    let (resumed, _, _) = park(&reverse, pool, ONE_RANK);
    let steps = resumed.denoised_steps;
    let successor = Denoiser::start(&reverse, pool, IDENTITY, None, TWO_RANKS);
    let mut events = successor.handle.submit(request(resumed)).unwrap();
    assert!(completed(&until_terminal(&mut events)));
    let [continued] = admissions(&successor.stop()).try_into().unwrap();
    let restore = continued
        .restored
        .expect("the preparation restores the trajectory");
    assert_eq!(restore.steps, steps);
    assert_eq!(sources(&restore), [(vec![0, 1], pages_of(ROWS) as usize)]);
    assert_eq!(continued.denoising, STEPS - steps);
}

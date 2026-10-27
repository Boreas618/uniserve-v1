//! Video requests preempted at denoising-step boundaries and resumed from
//! the trajectories stored in the host latent pools, over the GPU-free
//! simulator and a Storage Server in the test's process.
//!
//! The simulator copies no bytes: what the tests observe is what the
//! scheduler asks of the workers at the executor boundary (the trajectory
//! offloads a batch carries, the stored trajectory a latent preparation
//! restores, the denoising calls a request runs) and the events its requests
//! end with. A preemption in normal serving (a sleep) and a world change (a
//! park, and a successor scheduler that resumes the request) take the same
//! path, and a request whose stored trajectory is missing starts from noise.

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
use ipc::{CallKind, LatentPageCopy, LatentPoolInfo, MediaCall, RequestId};
use proto::request::MigrationCause;

use crate::support;
use crate::support::storage::{DEPLOYMENT, HOST, StorageServer};
use crate::support::video::{video_request, video_worker};

/// Denoising steps of the simulated denoiser: enough that a preemption
/// after the second lands short of the last.
const STEPS: u32 = 6;

/// Latent pages one request slot owns on the denoising rank.
const SLOT_PAGES: u32 = 2;

/// Float32 values of one latent page (`video_worker`).
const PAGE_UNITS: u64 = 256;

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

/// A scheduler of a simulated video denoiser whose rank maps the host
/// latent pool `pool` of `server`, reporting `identity`, with its loop
/// running.
struct Denoiser {
    handle: CommandSender,
    thread: thread::JoinHandle<bool>,
    boundary: Receiver<BatchEvent>,
    /// The latent seed the scheduler keys trajectories by.
    seed: u64,
}

impl Denoiser {
    fn start(server: &StorageServer, pool: u64, identity: &str, gate: Option<Arc<Gate>>) -> Self {
        let mut sim = video_worker(1, 1);
        let info = sim.mut_info_for_test();
        if let Some(denoiser) = info.video_denoiser.as_mut() {
            denoiser.denoise_steps = STEPS;
        }
        sim.set_latent_pools(vec![(
            HOST.to_owned(),
            Some(LatentPoolInfo {
                pool,
                slot_pages: SLOT_PAGES,
                region: [1; 32],
                identity: identity.to_owned(),
                endpoint: None,
            }),
        )]);
        let mut inner = SimExecutor::new(sim);
        let boundary = inner.observe();
        let mut scheduler = support::scheduler_with(
            Box::new(GatedExecutor { inner, gate }),
            SimEngine::special_tokens(),
            SchedulerConfig {
                max_batch: 32,
                ..Default::default()
            },
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

/// A server with one latent pool, which holds a slot's pages many times.
fn server() -> (StorageServer, u64) {
    let mut server = StorageServer::start(1 << 20);
    let pool = server.latent_pool("sim", PAGE_UNITS * 4, 64);
    (server, pool)
}

/// The `t2va` request every test submits, resuming `trajectory`.
fn request(trajectory: Trajectory) -> Request {
    let Request::Diffusion(mut request) = video_request(1, 1) else {
        unreachable!("a video request");
    };
    request.trajectory = trajectory;
    Request::Diffusion(request)
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

/// The steps a request left with, from its migratable end.
fn left_with(events: &[Event], expected: MigrationCause) -> u32 {
    match events.last() {
        Some(Event::Migratable {
            cause,
            denoised_steps,
        }) if *cause == expected => *denoised_steps,
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

/// What the workers were asked of request 1's admissions, each in order of
/// admission (its request epoch): the trajectory pages its batches
/// offloaded, the stored step and pages its latent preparation restored,
/// and its denoising calls.
#[derive(Debug, Default)]
struct Admission {
    offloaded: Vec<LatentPageCopy>,
    restored: Option<(u32, Vec<LatentPageCopy>)>,
    prepared: bool,
    denoising: u32,
}

fn admissions(batches: &[ExecutionBatch]) -> Vec<Admission> {
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
            if call.request_key.request_id != RequestId(1) {
                continue;
            }
            let admission = of(&mut epochs, call.request_key.request_epoch);
            match call.kind {
                CallKind::Media(MediaCall::LatentPreparation) => {
                    admission.prepared = true;
                    admission.restored = call
                        .latent_restore
                        .as_ref()
                        .map(|restore| (restore.steps, restore.pages.clone()));
                }
                CallKind::Media(MediaCall::Denoising) => admission.denoising += 1,
                _ => {}
            }
        }
        for offload in batch
            .page_offloads
            .iter()
            .flat_map(|offloads| &offloads.latents)
        {
            of(&mut epochs, offload.request_key.request_epoch)
                .offloaded
                .extend(offload.pages.iter().cloned());
        }
    }
    epochs.into_iter().map(|(_, admission)| admission).collect()
}

/// The pages of every page of a request slot on rank 0 of `pool`.
fn every_page(copies: &[LatentPageCopy], pool: u64) -> bool {
    let mut pages: Vec<u32> = copies
        .iter()
        .filter(|copy| copy.rank == 0 && copy.pool == pool)
        .map(|copy| copy.page)
        .collect();
    pages.sort_unstable();
    copies.len() == SLOT_PAGES as usize && pages == (0..SLOT_PAGES).collect::<Vec<_>>()
}

/// A sleep, which gives back the workers' memory in normal serving,
/// preempts a denoising video request at its next step boundary: the
/// trajectory's pages reach the host latent pool, and the request leaves
/// with the steps it took. Submitted again once the workers woke, it
/// continues from that step: its latent preparation restores every page and
/// only the steps left are denoised, and it completes with its video.
#[test]
fn a_sleep_preempts_a_video_request_which_resumes_from_its_stored_step() {
    let (server, pool) = server();
    let gate = Arc::new(Gate::default());
    let denoiser = Denoiser::start(&server, pool, IDENTITY, Some(Arc::clone(&gate)));
    let trajectory = Trajectory {
        id: TRAJECTORY,
        ..Trajectory::default()
    };
    let mut events = denoiser.handle.submit(request(trajectory)).unwrap();

    gate.wait_reached();
    let slept = denoiser.handle.resize_cache(CacheChange::Sleep);
    gate.open();
    let stored = left_with(&until_terminal(&mut events), MigrationCause::Drain);
    assert!(
        (BEFORE_PREEMPTION..STEPS).contains(&stored),
        "the request is preempted past the steps submitted and short of the last: {stored}"
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

    let mut resumed = denoiser
        .handle
        .submit(request(Trajectory {
            denoised_steps: stored,
            ..trajectory
        }))
        .unwrap();
    assert!(completed(&until_terminal(&mut resumed)));

    let [preempted, resumed] = admissions(&denoiser.stop())
        .try_into()
        .expect("the request was admitted twice");
    assert_eq!(preempted.denoising, stored);
    assert!(every_page(&preempted.offloaded, pool), "{preempted:?}");
    let (steps, pages) = resumed
        .restored
        .expect("the preparation restores the trajectory");
    assert_eq!(steps, stored);
    assert!(every_page(&pages, pool), "{pages:?}");
    assert_eq!(resumed.denoising, STEPS - stored);
    assert!(resumed.offloaded.is_empty());
}

/// A request that names stored steps of a trajectory no pool holds starts
/// from noise: its latent preparation restores nothing, every step is
/// denoised, and it completes.
#[test]
fn a_video_request_whose_stored_trajectory_is_missing_starts_from_noise() {
    let (server, pool) = server();
    let denoiser = Denoiser::start(&server, pool, IDENTITY, None);
    let mut events = denoiser
        .handle
        .submit(request(Trajectory {
            id: TRAJECTORY,
            denoised_steps: 3,
            state_seed: None,
        }))
        .unwrap();
    assert!(completed(&until_terminal(&mut events)));

    let [admission] = admissions(&denoiser.stop())
        .try_into()
        .expect("the request was admitted once");
    assert!(admission.prepared);
    assert_eq!(admission.restored, None);
    assert_eq!(admission.denoising, STEPS);
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
    let gate = Arc::new(Gate::default());
    let serving = Denoiser::start(&server, pool, IDENTITY, Some(Arc::clone(&gate)));
    let trajectory = Trajectory {
        id: TRAJECTORY,
        ..Trajectory::default()
    };
    let mut events = serving.handle.submit(request(trajectory)).unwrap();
    gate.wait_reached();
    let parked = serving.handle.park();
    gate.open();
    let stored = left_with(&until_terminal(&mut events), MigrationCause::Drain);
    assert!((BEFORE_PREEMPTION..STEPS).contains(&stored), "{stored}");
    parked
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler parked")
        .expect("the park succeeded");
    let parked_seed = serving.seed;
    let [preempted] = admissions(&serving.stop())
        .try_into()
        .expect("the request was admitted once");
    assert!(every_page(&preempted.offloaded, pool), "{preempted:?}");

    let resumed = Trajectory {
        denoised_steps: stored,
        ..trajectory
    };
    // The same numerics: the successor continues the trajectory.
    let successor = Denoiser::start(&server, pool, IDENTITY, None);
    assert_eq!(successor.seed, parked_seed);
    let mut events = successor.handle.submit(request(resumed)).unwrap();
    assert!(completed(&until_terminal(&mut events)));
    let [same] = admissions(&successor.stop()).try_into().unwrap();
    assert_eq!(same.restored.map(|(steps, _)| steps), Some(stored));
    assert_eq!(same.denoising, STEPS - stored);

    // Other numerics: the trajectory is recomputed, or kept when the
    // request names the parked world's seed.
    let other = Denoiser::start(&server, pool, "another denoiser", None);
    assert_ne!(other.seed, parked_seed);
    let mut recomputed = other.handle.submit(request(resumed)).unwrap();
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
    assert_eq!(kept.restored.map(|(steps, _)| steps), Some(stored));
    assert_eq!(kept.denoising, STEPS - stored);
}

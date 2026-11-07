//! The device executor: one thread per device, the only issuer of work to that device's streams.
//!
//! The design (specs/worker.md 2.4) gives the executor thread one command at a time, in submission
//! order: serve a batch, prepare or switch a generation, release an old one, perform a storage
//! owner's copies and fences, or wait for the events deferred release hangs on. It blocks only on
//! the current command's device work, so the command it is performing is exactly what the control
//! thread's watchdog reads ([`Watchdog`]).
//!
//! This module is the handle every other thread holds. The scheduling mechanism submits a batch and
//! gets a [`Submission`] it can wait on; the control thread reads the watchdog and probes the
//! device's context for a sticky error (2.4); the program-layer update sends `precopy`,
//! `switchover` and `purge`, which take effect at batch boundaries (7.3).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ops::Generation;
use platform::DeviceId;

use crate::arena::Pools;
use crate::batch::{Batch, BatchOutput, Job, JobKind, Submission, SubmissionId};
use crate::job::{Command, JobQueue, jobs_of, phase_of};
use crate::plan::Bucket;
use crate::runner::{NamedRunner, RunnerId};
use crate::stream::events::{DRAIN_LIMIT, EventPool, JobId, drain_event};
use crate::{Error, Result};

/// A device executor's handle.
pub struct Executor {
    device: DeviceId,
    /// The phases this device's runners serve, recorded here because the runners themselves live
    /// on the device thread and a router above the executors has to choose one without asking it.
    phases: Vec<ipc::PhaseId>,
    /// What the device thread is performing, and `None` between commands (2.4).
    activity: Arc<std::sync::Mutex<Option<String>>>,
    /// The media computations this device's runners plan to capture, recorded here for the same
    /// reason: the mechanism asks whether a media computation is served before it forms the batch,
    /// and that answer cannot be a round trip to the device thread (2.4).
    media: crate::plan::Buckets,
    queue: Arc<JobQueue>,
    thread: Option<JoinHandle<()>>,
    /// The device's graph pools, shared by every runner that issues here (4.1).
    pools: Arc<std::sync::Mutex<Pools>>,
    /// What the device has done since it started, for the metrics report (7.3).
    counters: Arc<SharedCounters>,
    /// The operators the last program-layer update could not compare, each with its reason
    /// (7.3 step 2): the runners on this device write it as they prepare and shadow a generation,
    /// and the control thread reports it when the precopy finishes. Empty is a fact - every
    /// operator the update replaced was compared.
    uncompared: Arc<std::sync::Mutex<Vec<String>>>,
    /// The streams this device's runners compute on, which the counters record.
    streams: Vec<platform::Stream>,
    /// The stream the device's page copies run on, which the counters record beside them.
    copy_stream: platform::Stream,
    /// The rank's Storage Server client, bound after construction because the control thread builds
    /// it after the launch's claim while the executors exist before it: a sleep and a wake are the
    /// only calls that reach the ledger, and the device thread makes them (2.5).
    storage: Arc<std::sync::Mutex<Option<Arc<crate::memory::retained::Retained>>>>,
}

/// What a device has done since it started: the metrics report's counters (7.3).
///
/// `kernel_launches` counts the kernels of every replayed graph, summed over the device's runners;
/// it is what the report's `kernel_launches` carries. `device_ns` is the time the device thread
/// spent inside issued work, which is device time because the executor synchronizes after each
/// job and nothing else issues on this device's streams (2.4); with CUPTI profiling on, the
/// profiler's own kernel time replaces this figure in the report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Kernel launches issued, summed over every runner of the device.
    pub kernel_launches: u64,
    /// Device time spent in those launches, in nanoseconds.
    pub device_ns: u64,
}

/// The same counters as the device thread and its runners add to them.
///
/// Both are monotonic sums, so a reader on the control thread may take a reading while the device
/// thread adds to it: each field is one atomic add and one atomic load. A runner receives it from
/// [`Runner::attach_counters`](crate::runner::Runner::attach_counters) and adds the launches of
/// every replay; the device thread adds its own time around each job.
#[derive(Debug, Default)]
pub struct SharedCounters {
    kernel_launches: AtomicU64,
    device_ns: AtomicU64,
}

impl SharedCounters {
    /// Adds `launches` kernel launches.
    pub fn launches(&self, launches: u64) {
        self.kernel_launches.fetch_add(launches, Ordering::Relaxed);
    }

    /// Adds device time spent in one job.
    fn time(&self, elapsed: Duration) {
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.device_ns.fetch_add(nanos, Ordering::Relaxed);
    }

    /// The counters as the metrics report reads them.
    fn read(&self) -> Counters {
        Counters {
            kernel_launches: self.kernel_launches.load(Ordering::Relaxed),
            device_ns: self.device_ns.load(Ordering::Relaxed),
        }
    }
}

/// The device's graph storage, as the engine sizes it before the executor starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphStorage {
    /// Bytes the device may map for captured graphs (4.1's budget).
    pub bytes: u64,
    /// The device budget every mapping on the device is booked against.
    pub budget: u64,
    /// How many streams the executor issues on; the graph range is divided among them.
    pub streams: usize,
    /// Bytes of the device's scratch pool: the shadow pair's region and the state group's
    /// snapshot, one reservation (7.3, 4.5). Zero when the rank never shadows.
    pub scratch: u64,
    /// Bytes of the shadow region alone, which the paired forward's twins allocate.
    ///
    /// The reservation covers `shadow + snapshot`, and the two tenants take their own figures from
    /// it: giving the twins the reservation would leave the snapshot nothing to allocate, which a
    /// model with no state group cannot show (its snapshot is zero) and one with a state group
    /// shows on the first update it takes.
    pub shadow: u64,
    /// Bytes of the device's resident pool, which the storage owners carve through the same
    /// allocator (4.1). Zero when the rank keeps nothing resident in this process.
    pub resident: u64,
}

impl Executor {
    /// Starts the device thread, which owns the runners and the arena their graphs use.
    ///
    /// The pool set arrives from [`Executor::open`], which the engine calls while it opens the
    /// resident owners: it is the one allocator of the device, so every owner and every runner that
    /// issues here books against the same allowance (4.1). `scratch` is the shadow pair's region,
    /// which the engine sized beside the graph pool.
    ///
    /// `products` is the epoch's one product store, and it is passed in rather than built here:
    /// a product is produced by one device's runner and read by another's (a prompt's states are
    /// encoded on the device that holds the text model and read by the denoiser of the request's
    /// canvas), so there is one store per process and every device attaches the same one.
    ///
    /// From this call on the calling thread issues no device work for this device: everything
    /// reaches the device through [`Executor::submit`], the program-layer calls, and the storage
    /// owners' recorded jobs.
    /// Opens the device's one pool set, before anything that allocates on the device exists.
    ///
    /// The resident owners (4.1) carve their storage from this allocator, and they are opened
    /// before the executor thread - the runner that serves batches is built from them - so the pool
    /// set is built here and handed to [`Executor::spawn`] rather than built inside it.
    pub fn open(device: DeviceId, graph: &GraphStorage) -> Result<Arc<std::sync::Mutex<Pools>>> {
        let handle = platform::Device::new(device)?;
        // One book per device (4.1): the allocator books every pool kind against the same
        // allowance, so the three kinds cannot sum past it.
        let totals = Arc::new(Pools::totals_of(graph.budget));
        let allocator =
            platform::Allocator::new(&handle, Arc::clone(&totals) as Arc<dyn platform::Budget>)?;
        Ok(Arc::new(std::sync::Mutex::new(Pools::new(
            allocator,
            totals,
            graph.bytes,
            graph.scratch,
            graph.resident,
            graph.streams,
        )?)))
    }

    pub fn spawn(
        device: DeviceId,
        mut runners: Vec<NamedRunner>,
        events: Arc<EventPool>,
        pools: Arc<std::sync::Mutex<Pools>>,
        products: Arc<dyn crate::runners::products::Products>,
        scratch: u64,
        shadow: u64,
    ) -> Result<Self> {
        // The graph range is divided among the streams that map it, and the plan states that
        // count before the runners exist. Here they exist, so the two are compared: a runner the
        // division did not count would take its share from the ones before it and be refused at
        // its first capture as an allocator fault. A preparation takes one more range than the
        // serving generation does, so the check allows for it.
        {
            let pools = pools.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let needed = runners.len() + 1;
            if needed > pools.streams() {
                return Err(Error::Interface(format!(
                    "device {} serves {} runner(s) and a preparation, which need {} graph ranges, \
                     and its graph storage was divided among {}; a runner whose stream was not \
                     counted takes its share from the ones before it",
                    device.0,
                    runners.len(),
                    needed,
                    pools.streams(),
                )));
            }
        }
        let counters = Arc::new(SharedCounters::default());
        // What the device thread is performing, read by whoever reports a stall from another thread
        // (2.4): the thread that is stuck cannot say it, and a stall line naming the stage alone
        // says nothing about which job never answered.
        let activity: Arc<std::sync::Mutex<Option<String>>> = Arc::new(std::sync::Mutex::new(None));
        let uncompared = Arc::new(std::sync::Mutex::new(Vec::new()));
        let storage = Arc::new(std::sync::Mutex::new(None));
        let phases: Vec<ipc::PhaseId> = runners
            .iter()
            .flat_map(|named| named.phases.iter().copied())
            .collect();
        let media = crate::plan::Buckets::new(
            runners
                .iter()
                .flat_map(|named| named.runner.media_plan())
                .collect::<Vec<_>>(),
        );
        for named in &mut runners {
            named.runner.attach_pools(Arc::clone(&pools));
            named.runner.claim_arena()?;
            named.runner.attach_counters(Arc::clone(&counters));
            named.runner.attach_uncompared(Arc::clone(&uncompared));
            named.runner.attach_products(Arc::clone(&products));
        }
        // What the next generation captures and what it is planned in, stated before the first
        // capture rather than discovered by paying for it: the first generation of an epoch has no
        // serving generation to compare against, so every bucket is captured and every operator of
        // it is new.
        //
        // The two figures are not a sum. A stream's buckets are extents *inside* one buffer, and the
        // first capture maps that buffer whole - the reservation - whether the plan holds sixteen
        // buckets or nine thousand. `widest` is the largest extent a plan books inside it, and a
        // bucket whose walk runs past the range is refused by name at that bucket, before anything
        // writes past it (`nn::Error::Arena`). What a capture costs beyond that is a driver
        // executable per graph, which no pool books: the count is what a run should read.
        let captures: Vec<crate::plan::Bucket> = runners
            .iter()
            .flat_map(|named| named.runner.capture_plan())
            .collect();
        if !captures.is_empty() {
            let reservation = pools
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .reserved();
            let widest = captures.iter().max_by_key(|bucket| bucket.tokens);
            tracing::info!(
                device = device.0,
                captures = captures.len(),
                streams = 1,
                graph_reservation = reservation,
                widest = widest.map(|bucket| bucket.to_string()).unwrap_or_default(),
                "the next generation captures one graph per bucket, largest first; their activations \
                 are extents inside one range that the first capture maps whole"
            );
        }
        let streams: Vec<platform::Stream> = runners
            .iter()
            .filter_map(|named| named.runner.stream().cloned())
            .collect();
        let copy_stream = platform::Stream::new(device)?;
        let queue = JobQueue::new(device, Arc::clone(&events));
        let state = Device {
            device,
            runners,
            copy_stream: copy_stream.clone(),
            queue: Arc::clone(&queue),
            scratch,
            shadow,
            pools: Arc::clone(&pools),
            counters: Arc::clone(&counters),
            storage: Arc::clone(&storage),
            shadow_attached: false,
            activity: Arc::clone(&activity),
            sticky: None,
            waits: true,
            pending_landing: None,
            session: None,
            step_end: None,
            blocking: platform::cupti::blocking::Calls::default(),
            refold: false,
        };
        let thread_queue = Arc::clone(&queue);
        let thread = std::thread::Builder::new()
            .name(format!("executor-{}", device.0))
            .spawn(move || state.run(&thread_queue))
            .map_err(|error| {
                Error::Invalid(format!(
                    "the executor thread of {device} did not start: {error}"
                ))
            })?;
        Ok(Self {
            device,
            phases,
            media,
            activity,
            queue,
            thread: Some(thread),
            pools,
            counters,
            storage,
            uncompared,
            streams,
            copy_stream,
        })
    }

    /// The streams this device's runners compute on.
    pub fn streams(&self) -> &[platform::Stream] {
        &self.streams
    }

    /// The stream the device's page copies run on.
    pub fn copy_stream(&self) -> &platform::Stream {
        &self.copy_stream
    }

    /// The operators the last program-layer update could not compare, each with its reason (7.3
    /// step 2), in the order the device's runners recorded them.
    ///
    /// Read by the control thread when a precopy finishes, so the update's answer states what it
    /// has no evidence for instead of leaving that to a log line.
    pub fn uncompared(&self) -> Vec<String> {
        self.uncompared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// What the device thread is performing, or `None` when it is between commands (2.4).
    ///
    /// Read by whoever reports a stall: the thread that is stuck cannot say what it is stuck in, and
    /// a stall line that names only the stage says nothing about which job never answered.
    pub fn performing(&self) -> Option<String> {
        self.activity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// How many graph captures are open in this process on any stream.
    ///
    /// A hang inside a capture and a hang inside a launch look identical from outside and are
    /// different defects, so the stall report states this beside the job in hand (2.4). The count is
    /// the driver's own: a capture opens when `Capture::begin` returns and closes when it ends.
    pub fn capturing(&self) -> usize {
        platform::open_captures()
    }

    /// The device this executor drives.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Gives every runner the shadow harness of its device, carved from the device allocator
    /// (7.3): the main process calls this once, before the epoch serves.
    pub fn set_shadow(&self) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::SetShadow, &submission, JobKind::Shadow, None)?;
        Ok(submission)
    }

    /// The queue every thread submits device work through.
    pub fn queue(&self) -> &Arc<JobQueue> {
        &self.queue
    }

    /// Submits one batch and returns the submission the mechanism waits on.
    pub fn submit(&self, batch: Batch) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Batch(batch), &submission, JobKind::Batch, None)?;
        Ok(submission)
    }

    /// Runs one shadow batch: the paired forwards of `batch.generation`'s changed operators
    /// against the serving generation (7.3 step 2).
    pub fn shadow(&self, batch: Batch) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Shadow(batch), &submission, JobKind::Shadow, None)?;
        Ok(submission)
    }

    /// Asks every runner to capture its buckets of the serving generation, before the executor
    /// serves its first batch, and returns the submission that finishes it.
    pub fn warmup(&self) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Warmup, &submission, JobKind::Warmup, None)?;
        Ok(submission)
    }

    /// Asks every runner to prepare `next` and returns the submission that finishes it (7.3 step
    /// 2).
    pub fn precopy(&self, next: Arc<Generation>) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Precopy(next), &submission, JobKind::Precopy, None)?;
        Ok(submission)
    }

    /// Switches the serving generation at the next batch boundary (7.3 step 3).
    pub fn switchover(&self, generation: u64) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue.submit(
            Command::Switchover(generation),
            &submission,
            JobKind::Switchover,
            None,
        )?;
        Ok(submission)
    }

    /// Discards the update prepared for `generation` and keeps the serving one (7.3 step 4).
    pub fn revert(&self, generation: u64) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue.submit(
            Command::Revert(generation),
            &submission,
            JobKind::Revert,
            None,
        )?;
        Ok(submission)
    }

    /// Releases the graphs that still reference a generation older than `generation` (7.3 step 6).
    pub fn purge(&self, generation: u64) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue.submit(
            Command::Purge(generation),
            &submission,
            JobKind::Purge,
            None,
        )?;
        Ok(submission)
    }

    /// Releases everything that references `generation`, which was prepared and never served (7.3
    /// step 2).
    ///
    /// A failed precopy abandons the generation it prepared: its kernels unload with it, and the
    /// graphs and arena regions the runner rebound or recaptured for it are released here. `purge`
    /// cannot do this, because purge drops what is *older* than the serving generation and an
    /// abandoned generation is newer than it.
    pub fn discard(&self, generation: u64) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue.submit(
            Command::Discard(generation),
            &submission,
            JobKind::Discard,
            None,
        )?;
        Ok(submission)
    }

    /// Binds the rank's Storage Server client to this device, so a sleep can give segments back and
    /// a wake can map them again (2.5).
    ///
    /// Called once by the control thread after the executors are spawned: the client is built after
    /// the launch's claim, and the devices exist before it.
    pub fn bind_storage(&self, client: Arc<crate::memory::retained::Retained>) {
        *self
            .storage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(client);
    }

    /// Gives every segment of the resident cache back to the Storage Server (7.6).
    ///
    /// One call per cache, in the device queue's order: a batch submitted before it finishes first,
    /// and nothing submitted after it runs until admission opens again on the wake's answer.
    pub fn sleep(&self) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Sleep, &submission, JobKind::Sleep, None)?;
        Ok(submission)
    }

    /// Maps the released cache segments again, `limit` of them or all of them (2.5).
    ///
    /// The answer is the measurement of what came back; a shortfall is not an error, because the
    /// limit is a bound rather than a demand. An `Err` is: the addresses are valid and the K/V is
    /// not, so the caller must not admit a request on this device.
    pub fn wake(&self, limit: Option<u32>) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue
            .submit(Command::Wake { limit }, &submission, JobKind::Wake, None)?;
        Ok(submission)
    }

    /// Maps the host KV cache's page pools on this device's cache and answers what the rank reports
    /// of them. It waits: a scheduler sizes the host KV cache once, before it serves anything.
    pub fn map_host_cache(
        &self,
        tokens: u64,
        keys: &crate::batch::HostCacheKeys,
    ) -> Result<Vec<ipc::PagePoolInfo>> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        let pools = Arc::new(std::sync::Mutex::new(Vec::new()));
        self.queue.submit(
            Command::MapHostCache {
                tokens,
                keys: keys.clone(),
                pools: Arc::clone(&pools),
            },
            &submission,
            JobKind::HostCache,
            None,
        )?;
        match submission.wait() {
            crate::completion::Outcome::Success(_) => Ok(std::mem::take(
                &mut *pools
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )),
            crate::completion::Outcome::Failed(error) => {
                Err(Error::Invalid(format!("mapping the host KV cache: {error}")))
            }
            crate::completion::Outcome::Cancelled => {
                Err(Error::Invalid("mapping the host KV cache was cancelled".into()))
            }
        }
    }

    /// Copies `offloads` to the host KV cache in queue order, with no batch to carry them.
    pub fn copy_pages(&self, offloads: Vec<ipc::PageCopy>) -> Result<Submission> {
        self.storage_command(Command::CopyPages(offloads), JobKind::HostCache)
    }

    /// Gives the cache's segments from `from` on back to the Storage Server (7.6).
    pub fn release_cache(&self, from: u32) -> Result<Submission> {
        self.storage_command(Command::ReleaseCache { from }, JobKind::Sleep)
    }

    /// Unmaps the graph pools' storage at their reserved addresses, keeping every capture (7.6).
    pub fn suspend_graphs(&self) -> Result<Submission> {
        self.storage_command(Command::SuspendGraphs, JobKind::Sleep)
    }

    /// Maps the graph pools' storage at the addresses the captures hold (7.6).
    pub fn restore_graphs(&self) -> Result<Submission> {
        self.storage_command(Command::RestoreGraphs, JobKind::Wake)
    }

    /// Changes the rank's profiling session between two batches and answers what the session says.
    /// It waits for the device thread, so it is a work thread's call and never the control thread's.
    ///
    /// # Errors
    ///
    /// When the session refuses, or the device's thread is gone.
    pub fn profile(
        &self,
        change: crate::profiling::session::Change,
    ) -> Result<serde_json::Value> {
        let (answer, answered) = crossbeam_channel::bounded(1);
        let submission = self.storage_command(Command::Profile { change, answer }, JobKind::Profile)?;
        match answered.recv() {
            Ok(result) => result,
            Err(_) => match submission.wait() {
                crate::Outcome::Failed(error) => Err(Error::Shared(error)),
                _ => Err(Error::State("the rank's device ended before it answered")),
            },
        }
    }

    /// Runs `work`, which changes what the rank's weights map, in the queue's order: a batch
    /// submitted before it finished, and one submitted after it reads the new weights (7.6).
    pub fn rebind(&self, work: Arc<dyn Fn() -> Result<()> + Send + Sync>) -> Result<Submission> {
        self.storage_command(Command::Rebind(work), JobKind::Rebind)
    }

    fn storage_command(&self, command: Command, kind: JobKind) -> Result<Submission> {
        let id = SubmissionId(self.queue.next_submission());
        let (submission, _) = Submission::pending(id);
        self.queue.submit(command, &submission, kind, None)?;
        Ok(submission)
    }

    /// Whether one of this device's runners serves `phase` (4.4): what a router above the executors
    /// asks before it hands a batch over.
    pub fn serves(&self, phase: ipc::PhaseId) -> bool {
        self.phases.contains(&phase)
    }

    /// The phases this device's runners serve, in the order they were spawned.
    pub fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
    }

    /// The bucket this device would serve `computation` in, or `None` when its plan holds no graph
    /// for it (2.4's stated capability).
    ///
    /// The answer comes from the plan the runners stated at spawn, not from the device thread: the
    /// mechanism forms a media batch only where this answers, so a missing answer is the executor
    /// saying it does not serve the computation rather than a check nobody makes.
    ///
    /// # What the caller must state
    ///
    /// A computation that **states** an extent is matched to the graph captured for exactly that, so
    /// a 25-frame step cannot be served by a 13-frame graph. [`crate::plan::Extent::None`] is the
    /// caller saying it does not know - which is every media computation the engine forms, because a
    /// denoising step's layout is the model's arithmetic and a decode's clip is derived inside the
    /// runner - and the bucket's own extent is then the answer, with its rows and tokens still room
    /// to spare. **A caller that states nothing gets the bucket the runners planned; a caller that
    /// states something gets that or nothing.**
    pub fn media(&self, computation: &crate::plan::Computation) -> Option<crate::plan::Bucket> {
        self.media.select_media(computation)
    }

    /// The device's one pool set, for an owner that carves its own storage from the same allocator
    /// (the storage owners' resident planes, the paired forward's scratch pages).
    ///
    /// Nothing else on the device constructs a `Pools` (4.1): a second allocator would book the
    /// same device allowance twice.
    pub fn pools(&self) -> Arc<std::sync::Mutex<Pools>> {
        Arc::clone(&self.pools)
    }

    /// The device's mapped bytes per pool kind, from the one allocator that books them (4.1).
    ///
    /// The figures are one booking's totals rather than a sum over owners - one `Pools` is shared
    /// by every runner on the device, so a sum over runners would count the same mapping twice -
    /// and item 5's `graph + scratch + resident <= budget` is therefore decided by one book.
    pub fn memory(&self) -> crate::arena::DeviceMemory {
        self.pools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .memory()
    }

    /// What the device has done since it started: kernel launches and device time (7.3).
    pub fn counters(&self) -> Counters {
        self.counters.read()
    }

    /// What the control thread's watchdog reads: the command in flight and how long it took (2.4).
    pub fn watchdog(&self) -> Watchdog {
        Watchdog {
            jobs: self.queue.in_flight(),
        }
    }

    /// Probes this device's context for a sticky error, which the control thread does once a
    /// second while a command has run longer than a second (2.4).
    pub fn probe(&self) -> Result<Probe> {
        let context = platform::PrimaryContext::retain(self.device)?;
        Ok(match context.sticky_error()? {
            Some(error) => Probe::Sticky(error),
            None => Probe::Idle,
        })
    }

    /// Drains the device's commands and joins its thread. Every submission must have completed
    /// before the caller closes the executor.
    pub fn close(mut self) -> Result<()> {
        self.queue.close();
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| Error::State("the executor thread panicked"))?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Executor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Executor")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// One command in flight, as the control thread's stall rule reads it (2.4).
#[derive(Clone, Debug)]
pub struct Watch {
    /// The submission whose command is in flight, when the command belongs to one.
    pub submission: Option<SubmissionId>,
    /// The command's number.
    pub job: JobId,
    /// What kind of command it is.
    pub kind: JobKind,
    /// The batch it serves, when it serves one.
    pub batch: Option<crate::batch::BatchId>,
    /// When the device thread began it.
    pub started: std::time::Instant,
    /// How long it has been in flight at the reading.
    pub elapsed: Duration,
}

/// The watchdog's reading: the commands in flight on one device.
#[derive(Clone, Debug, Default)]
pub struct Watchdog {
    /// The commands in flight, in submission order (at most one).
    pub jobs: Vec<Watch>,
}

impl Watchdog {
    /// The command in flight longest, which the stall rule names when it fires.
    pub fn longest(&self) -> Option<&Watch> {
        self.jobs.iter().max_by_key(|watch| watch.elapsed)
    }

    /// The longest a command has been in flight, `Duration::ZERO` when none is.
    pub fn elapsed(&self) -> Duration {
        self.longest().map_or(Duration::ZERO, |watch| watch.elapsed)
    }
}

/// What one turn of the device thread finished (4.5's interleaving).
enum Step {
    /// The command finished; its submission completes with this output.
    Done(BatchOutput),
    /// The command put its remainder at the tail; its submission completes on that step.
    Requeued,
}

/// What a sticky-error probe found (2.4).
#[derive(Debug)]
pub enum Probe {
    /// The device's context is healthy.
    Idle,
    /// The device's context has a sticky error: the executor cannot serve another batch, and the
    /// main process must be replaced (2.2, 2.5).
    Sticky(platform::Error),
}

/// The device thread's state: the runners, the streams only it issues to, and the queue.
/// Milliseconds from the first time this was asked to `at`, one clock for every device's lines.
fn since_start(at: Instant) -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = *START.get_or_init(Instant::now);
    at.saturating_duration_since(start).as_millis() as u64
}

/// How often a device thread looks at a replay in flight while it waits for the next command.
const LANDING_POLL: Duration = Duration::from_micros(100);

/// A replay on its stream, and what is reported when it lands.
struct PendingLanding {
    /// Recorded behind the replay.
    landed: platform::Event,
    /// When its command was taken up.
    started: Instant,
    /// How long staging its batch took, in milliseconds.
    staged_ms: u64,
    /// The media call it serves.
    call: ipc::MediaCall,
    /// The batch it serves.
    batch: u64,
}

/// A command whose replay is running, with the answer its submission gets when it lands.
struct Launched {
    entry: crate::job::Entry,
    output: BatchOutput,
    pending: PendingLanding,
}

struct Device {
    /// Whether a job's device work is waited for before the job answers.
    waits: bool,
    /// The stream the device's page copies run on.
    copy_stream: platform::Stream,
    /// The replay the command just served left running.
    pending_landing: Option<PendingLanding>,
    device: DeviceId,
    runners: Vec<NamedRunner>,
    queue: Arc<JobQueue>,
    /// Bytes of the scratch pool's reservation: the shadow region and the snapshot (7.3, 4.5).
    scratch: u64,
    /// Bytes of the shadow region alone, which the twins allocate from that reservation.
    shadow: u64,
    /// The device's one graph pool, which also lends its allocator for the shadow pair's region.
    pools: Arc<std::sync::Mutex<Pools>>,
    /// What the device thread has done, shared with the handle the control thread reads.
    counters: Arc<SharedCounters>,
    /// Whether this device's runners hold a shadow harness, so `SetShadow` is sent once (7.3).
    shadow_attached: bool,
    /// What the job in hand is doing, for the report of a failed submission and for a stall report
    /// read from another thread. The device thread is the only place that knows it, and a failure
    /// crosses the thread boundary as a bare error.
    activity: Arc<std::sync::Mutex<Option<String>>>,
    /// The context's first sticky error, once one has been seen (2.4).
    ///
    /// A sticky error is not recoverable: every launch after it fails, so the executor issues
    /// nothing more and fails every command it is handed with this error. The main process learns
    /// of it through the control thread's probe and replaces the process (2.2, 2.5).
    sticky: Option<platform::Error>,
    /// The rank's Storage Server client, which a sleep and a wake reach the ledger through (2.5).
    storage: Arc<std::sync::Mutex<Option<Arc<crate::memory::retained::Retained>>>>,
    /// The profiling session that records this rank, while one does.
    session: Option<crate::profiling::session::Session>,
    /// The end marker of the step being served, launched right behind its replay: the batch number,
    /// its stage and the stream.
    step_end: Option<(u64, &'static str, platform::Stream)>,
    /// This thread's driver calls that waited for the device, when its last result left.
    blocking: platform::cupti::blocking::Calls,
    /// The weights changed while the graph pools were suspended, so the runners fill again what
    /// they fold from them once the pools are back.
    refold: bool,
}

impl Device {
    /// Serves the queue until it closes: one command at a time, in submission order.
    fn run(mut self, queue: &Arc<JobQueue>) {
        // This thread drives one device, so that device's context is current for its whole life. A
        // native module reads the device from the runtime (`cudaGetDevice`) to find its per-device
        // handles, and with no context current the runtime answers device zero: every other device
        // of the process would then launch through device zero's handle.
        let context = platform::PrimaryContext::retain(self.device);
        let _current = match &context {
            Ok(context) => context.enter().map_err(|error| error.to_string()),
            Err(error) => Err(error.to_string()),
        };
        if let Err(error) = &_current {
            tracing::error!(device = self.device.0, %error, "the device's context is not current");
        }
        // The driver calls this thread waits in are counted for as long as it serves
        // (`uniserve:worker_blocking_calls`): waiting for its own device work is its design, so the
        // figure is a record rather than a fault.
        let _watched = platform::cupti::blocking::Watched::enter();
        // Deferred storage release gets its waiting commands through this queue (2.4).
        let events = Arc::clone(queue.events());
        let submitter = Arc::clone(queue);
        events.set_submitter(Some(Box::new(move |device, events| {
            submitter.after(device, events)
        })));
        let mut launched: Option<Launched> = None;
        loop {
            let entry = match &launched {
                Some(in_flight) => match self.beside(queue, in_flight) {
                    Some(entry) => Some(entry),
                    None => {
                        if let Some(landing) = launched.take() {
                            self.land(queue, landing);
                        }
                        continue;
                    }
                },
                None => queue.next(None),
            };
            let Some(entry) = entry else {
                break;
            };
            if let Some(sticky) = &self.sticky {
                // The context is unusable: the command is not issued, and its submission fails
                // with the sticky error rather than with a launch error that would hide it.
                queue.complete(
                    entry,
                    Err(Error::Cuda(sticky.clone())),
                    Some("the device's context already had a sticky error".to_owned()),
                );
                continue;
            }
            let result = self.serve(&entry);
            let result = match (result, self.pending_landing.take()) {
                (Ok(Step::Done(output)), Some(pending)) => {
                    let landing = Launched {
                        entry,
                        output,
                        pending,
                    };
                    // A host call has nothing left on its stream, so it answers now and the replay
                    // in flight keeps running. A replay stays in flight, and the one before it
                    // lands with this one already on the stream.
                    if matches!(landing.pending.landed.query(), Ok(true)) {
                        self.land(queue, landing);
                    } else {
                        if let Some(previous) = launched.take() {
                            self.land(queue, previous);
                        }
                        launched = Some(landing);
                    }
                    continue;
                }
                (result, _) => result,
            };
            if let Some(landing) = launched.take() {
                self.land(queue, landing);
            }
            if matches!(result, Ok(Step::Requeued)) {
                // The entry put the rest of its command at the tail; its submission waits for the
                // last step, so nothing completes here.
                continue;
            }
            // What was in hand, read **after** the command: `perform` is what writes it, and it is
            // taken when a command completes, so a string read before the command is the one the
            // previous command left - empty. Reading it before the command therefore wrapped no
            // fault at all, and every device failure in a run named only its batch number.
            let detail = self
                .activity
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            // The work in flight is this thread's knowledge alone and is gone once the command ends:
            // it travels inside the failure, so every report of it - the device's line, the
            // submission's outcome, the main process's reason - names the bucket, the rows, the
            // generation and the operator without reaching back for state that is gone.
            let result = result.map_err(|error| match &detail {
                Some(detail) => Error::in_flight(detail.clone(), error),
                None => error,
            });
            if let Err(error) = &result {
                tracing::warn!(
                    %error,
                    in_flight = detail.as_deref().unwrap_or("no work in flight"),
                    device = self.device.0,
                    "a device command failed"
                );
            }
            let result = match result {
                Ok(Step::Done(output)) => Ok(output),
                Ok(Step::Requeued) => unreachable!("handled above"),
                Err(error) => Err(error),
            };
            if result.is_err() {
                // A failure may be the sticky error's first symptom; probe before the next command
                // so the executor stops issuing to a broken context (2.4).
                if let Ok(Some(sticky)) = self.probe_sticky() {
                    tracing::error!(
                        device = self.device.0,
                        %sticky,
                        "the device's context has a sticky error; no further command is issued"
                    );
                    self.sticky = Some(sticky);
                }
            }
            queue.complete(entry, result, detail);
        }
        if let Some(landing) = launched.take() {
            self.land(queue, landing);
        }
        events.set_submitter(None);
    }

    /// The next command to serve beside the replay in flight, or `None` once the replay should land
    /// first: it has finished, or the command next in line needs the device to itself.
    fn beside(&self, queue: &JobQueue, in_flight: &Launched) -> Option<crate::job::Entry> {
        loop {
            if let Some(entry) = queue.next_if(|entry| self.overlaps(entry)) {
                return Some(entry);
            }
            if queue.has_pending() || !matches!(in_flight.pending.landed.query(), Ok(false)) {
                return None;
            }
            std::thread::sleep(LANDING_POLL);
        }
    }

    /// Whether `entry` is a batch whose replay may be launched behind one in flight.
    fn overlaps(&self, entry: &crate::job::Entry) -> bool {
        let Command::Batch(batch) = &entry.command else {
            return false;
        };
        self.runner_of(phase_of(batch))
            .and_then(|runner| self.runner_ref(runner))
            .is_ok_and(|named| named.runner.overlaps_replays())
    }

    /// Waits for a launched replay and answers its submission.
    fn land(&mut self, queue: &JobQueue, launched: Launched) {
        let Launched {
            entry,
            output,
            pending,
        } = launched;
        let result = pending
            .landed
            .synchronize()
            .map(|()| output)
            .map_err(Error::from);
        tracing::info!(
            device = self.device.0,
            call = ?pending.call,
            batch = pending.batch,
            staged_ms = pending.staged_ms,
            total_ms = pending.started.elapsed().as_millis() as u64,
            began_ms = since_start(pending.started),
            ended_ms = since_start(Instant::now()),
            "a media batch was served"
        );
        if let Err(error) = &result {
            tracing::warn!(%error, device = self.device.0, "a device command failed");
            if let Ok(Some(sticky)) = self.probe_sticky() {
                self.sticky = Some(sticky);
            }
        }
        queue.complete(entry, result, None);
    }

    /// Reads this device's context for a sticky error, the check the stall rule ends in (2.4).
    fn probe_sticky(&self) -> Result<Option<platform::Error>> {
        let context = platform::PrimaryContext::retain(self.device)?;
        context.sticky_error().map_err(Into::into)
    }

    /// Performs one command and waits for its device work before returning (2.4: the executor
    /// blocks only on the current command's device work).
    ///
    /// A command that serves a batch answers with that step's output; every other command answers
    /// with an empty one. A multi-job command ([`Command::Steps`]) performs one job and puts the
    /// rest back at the tail of the queue, so a serving batch the scheduler submitted meanwhile is
    /// served between two of its jobs (4.5's interleaving).
    fn serve(&mut self, entry: &crate::job::Entry) -> Result<Step> {
        let label = entry.label.clone();
        match &entry.command {
            Command::Batch(batch) => {
                let runner = self.runner_of(phase_of(batch))?;
                let started = Instant::now();
                let jobs = self.runner_mut(runner)?.runner.submit(batch.clone())?;
                let staged = started.elapsed();
                // A batch's jobs are its replay and nothing else: serving it is one turn, because
                // the mechanism waits on this submission to build the next step. A runner whose
                // replays overlap leaves its stream running: launching a graph costs the device
                // milliseconds of idle, and the next step's launch fits behind this step's work.
                let named = self.runner_mut(runner)?;
                // A session's window is made of whole steps, so its rank runs them one at a time.
                let overlapping = named.runner.overlaps_replays();
                let stream = named.runner.stream().cloned();
                let overlapping = overlapping && self.session.is_none();
                self.begin_step(batch, stream.as_ref())?;
                self.waits = !overlapping;
                let output = self.perform(&label, jobs_of(runner, jobs));
                self.waits = true;
                let output = output
                    .and_then(|mut output| {
                        output.forward_stats = self.end_step(batch.id.0)?;
                        Ok(output)
                    })
                    .map(Step::Done);
                if overlapping && output.is_ok() {
                    let (Some(stream), Some(media)) = (stream, &batch.media) else {
                        return Err(Error::Invariant(
                            "a runner whose replays overlap serves a media batch on a stream"
                                .into(),
                        ));
                    };
                    let landed = platform::Event::new(platform::EventFlags::default());
                    landed.record(&stream)?;
                    self.pending_landing = Some(PendingLanding {
                        landed,
                        started,
                        staged_ms: staged.as_millis() as u64,
                        call: media.computation.call,
                        batch: batch.id.0,
                    });
                    return output;
                }
                // A media call is seconds of device work a few times per request, so one line each
                // is the request's timeline; a token batch's would be the log.
                if let Some(media) = &batch.media {
                    tracing::info!(
                        device = self.device.0,
                        call = ?media.computation.call,
                        batch = batch.id.0,
                        staged_ms = staged.as_millis() as u64,
                        total_ms = started.elapsed().as_millis() as u64,
                        began_ms = since_start(started),
                        ended_ms = since_start(Instant::now()),
                        "a media batch was served"
                    );
                }
                output
            }
            Command::Warmup => {
                let mut jobs = Vec::new();
                for named in &mut self.runners {
                    jobs.extend(jobs_of(named.id, named.runner.warmup()?));
                }
                self.steps(entry, &label, None, jobs)
            }
            Command::Shadow(batch) => {
                let runner = self.runner_of(phase_of(batch))?;
                let jobs = self.runner_mut(runner)?.runner.shadow(batch.clone())?;
                // A shadow batch is a set of independent paired forwards, one per changed operator:
                // they interleave with serving batches like a precopy's steps (7.3 step 2).
                self.steps(entry, &label, Some(batch.generation), jobs_of(runner, jobs))
            }
            Command::Precopy(next) => {
                let mut jobs = Vec::new();
                for named in &mut self.runners {
                    jobs.extend(jobs_of(named.id, named.runner.precopy(next)?));
                }
                self.steps(entry, &label, Some(next.number()), jobs)
            }
            Command::Steps { generation, jobs } => {
                self.steps(entry, &label, *generation, jobs.clone())
            }
            Command::Switchover(generation) => {
                for named in &mut self.runners {
                    named.runner.switchover(*generation)?;
                }
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::SetShadow => {
                // The shadow pair's region is allocator-backed and comes from the same device
                // allocator as the graph pools, reserved on first use; the harness then goes to
                // every runner, because the update may shadow an operator of any of them.
                // A second send in one process keeps the harness the first made: the region's
                // reservation is idempotent, but re-attaching twins would replace the `Shadow` an
                // update may be mid-comparison with, so the command answers empty instead.
                if self.shadow_attached {
                    return Ok(Step::Done(BatchOutput::default()));
                }
                let mut pools = self
                    .pools
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                // The reservation covers `shadow + snapshot` and is taken once; the twins take the
                // shadow region and the snapshot takes its own figure, so the two tenants of one
                // pool never claim each other's bytes.
                pools.reserve_scratch(self.scratch)?;
                // One region per runner, divided out of the one shadow figure the plan reserved:
                // a device may serve several runners (a text one and a media one is the case that
                // exposed this), and a region each of the WHOLE shadow asks the pool for more than
                // it was reserved - the second runner's request is then exactly the reservation
                // the first already took. The division is the plan's figure, divided here and
                // nowhere else, and a device that cannot give each runner a byte refuses by name
                // rather than failing at an allocator.
                let runners = self.runners.len().max(1) as u64;
                let each = self.shadow / runners;
                if each == 0 && !self.runners.is_empty() {
                    return Err(Error::Interface(format!(
                        "device {} reserves a {} -byte shadow region for {} runner(s), which leaves none of them a region to compare in",
                        self.device.0, self.shadow, runners,
                    )));
                }
                // A runner's share has two tenants: the twins a paired forward writes its outputs
                // to and the page set its KV lives in, which holds a copy of every row's context.
                // Each takes half, because a twin grows with a bucket's tokens and a page set with
                // its rows' context, and neither figure is known before a batch names them.
                let page_set = each / 2;
                for named in &mut self.runners {
                    let twins = ops::Twins::new(pools.allocator(), each - page_set)
                        .map_err(|error| Error::Interface(error.to_string()))?;
                    let shadow = crate::shadow::Shadow::new(self.device, twins)?;
                    named.runner.attach_shadow(shadow, page_set);
                }
                self.shadow_attached = true;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::MapHostCache { tokens, keys, pools } => {
                let client = self.storage()?;
                for cache in self.caches() {
                    let setup = crate::memory::prefix_cache::host::HostCacheSetup {
                        fingerprint: keys.fingerprints.clone(),
                        endpoint: keys.endpoint.clone(),
                        device: self.device,
                        copy: self.copy_stream.clone(),
                    };
                    let mut cache = cache
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *pools
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        setup.map(&client, &mut cache, *tokens)?;
                }
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::CopyPages(offloads) => {
                for cache in self.caches() {
                    cache
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .copy_pages(offloads)?;
                }
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Sleep => {
                let client = self.storage()?;
                let mut storage: &crate::memory::retained::Retained = &client;
                for cache in self.caches() {
                    crate::memory::sleep(&cache, &mut storage)?;
                }
                crate::profiling::storage_remapped();
                // The segments are given back by now; the device's own work is finished before the
                // queue hands this command over, so a synchronize here would only wait on nothing.
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Wake { limit } => {
                let client = self.storage()?;
                let mut storage: &crate::memory::retained::Retained = &client;
                // One call per cache, on the stream of the runner that reads it: a second call for a
                // cache two runners share would reset the units the first had just mapped (2.5).
                let mut woken: Vec<*const ()> = Vec::new();
                for named in &mut self.runners {
                    let Some(cache) = named.runner.cache() else {
                        continue;
                    };
                    let address = Arc::as_ptr(&cache) as *const ();
                    if woken.contains(&address) {
                        continue;
                    }
                    woken.push(address);
                    let stream = named
                        .runner
                        .stream()
                        .ok_or_else(|| {
                            Error::Invariant(format!(
                                "runner {} reads a cache but states no stream to wake it on",
                                named.id.0
                            ))
                        })?
                        .clone();
                    crate::memory::wake(&cache, &mut storage, &stream, *limit)?;
                }
                crate::profiling::storage_remapped();
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::ReleaseCache { from } => {
                let client = self.storage()?;
                let mut storage: &crate::memory::retained::Retained = &client;
                for cache in self.caches() {
                    let planes = {
                        let cache = crate::lock(&cache);
                        Arc::clone(cache.planes())
                    };
                    if planes.segments().is_some() {
                        planes.release_segments(*from, &mut storage)?;
                    }
                }
                crate::profiling::storage_remapped();
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::SuspendGraphs => {
                self.synchronize()?;
                self.pools
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .suspend()?;
                crate::profiling::storage_remapped();
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::RestoreGraphs => {
                self.pools
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .restore()?;
                crate::profiling::storage_remapped();
                if std::mem::take(&mut self.refold) {
                    for named in &mut self.runners {
                        named.runner.weights_changed()?;
                    }
                    self.synchronize()?;
                }
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Rebind(work) => {
                self.synchronize()?;
                work()?;
                crate::profiling::storage_remapped();
                // What a runner folds from the weights it computes in its arena, which a sleeping
                // rank has unmapped: that waits for the storage to come back, and no batch runs
                // before it does.
                let suspended = self
                    .pools
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .suspended();
                if suspended {
                    self.refold = true;
                    return Ok(Step::Done(BatchOutput::default()));
                }
                for named in &mut self.runners {
                    named.runner.weights_changed()?;
                }
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Revert(generation) => {
                for named in &mut self.runners {
                    named.runner.revert(*generation);
                }
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Purge(generation) => {
                for named in &mut self.runners {
                    named.runner.purge(*generation);
                }
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Discard(generation) => {
                for named in &mut self.runners {
                    named.runner.discard(*generation)?;
                }
                self.synchronize()?;
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::Profile { change, answer } => {
                self.synchronize()?;
                let answered = self.profile(change.clone());
                // A caller that stopped waiting has nothing to be told.
                let _ = answer.send(answered);
                Ok(Step::Done(BatchOutput::default()))
            }
            Command::After(events) => {
                // The command establishes that the work deferred on these events may run: wait
                // for them on the host, bounded like every host wait for device work.
                let deadline = std::time::Instant::now() + DRAIN_LIMIT;
                for event in events {
                    drain_event(event, deadline)?;
                }
                Ok(Step::Done(BatchOutput::default()))
            }
        }
    }

    /// Performs one job of a multi-job command and puts the rest at the tail (4.5).
    ///
    /// A capture pass and a precopy are sets of independent per-bucket jobs; performing them in
    /// one turn would stop serving for the whole set, which is the long pause the design forbids.
    /// One job per turn lets whatever the scheduler submitted in between be served first, and a
    /// quiet queue still runs the steps back to back.
    fn steps(
        &mut self,
        entry: &crate::job::Entry,
        label: &str,
        generation: Option<u64>,
        jobs: Vec<Job>,
    ) -> Result<Step> {
        let mut jobs = jobs;
        if jobs.is_empty() {
            return Ok(Step::Done(BatchOutput::default()));
        }
        let first = jobs.remove(0);
        let output = self.perform(label, vec![first])?;
        if jobs.is_empty() {
            return Ok(Step::Done(output));
        }
        self.queue
            .requeue(entry, Command::Steps { generation, jobs })?;
        Ok(Step::Requeued)
    }

    /// Performs a runner's jobs in order, each on the runner that returned it.
    ///
    /// The answer is the output of the jobs that produced one; a job list with no output (every
    /// program-layer command) answers empty.
    fn perform(&mut self, label: &str, jobs: Vec<Job>) -> Result<BatchOutput> {
        let mut output = BatchOutput::default();
        let last = jobs.len().saturating_sub(1);
        for (index, job) in jobs.into_iter().enumerate() {
            let runner = job
                .runner()
                .ok_or_else(|| Error::Invariant("a serving job names no runner".into()))?;
            // The job in hand names the bucket, generation and operator a failure is about; the
            // command's label says which submission was running it.
            *self
                .activity
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(format!("{label}, {}", job.describe()));
            // A capture is minutes of device work with no batch to show for it, and the warmup is
            // one capture per bucket in a row: the started line is what names the bucket a warmup
            // is stuck in, because a started line with no finished one is the only shape a hang
            // has. Every other job produces an answer a later line or a timeout already names.
            let walked = Self::walked_bucket(&job);
            if let Some(bucket) = walked {
                tracing::info!(
                    bucket = %bucket,
                    job = %job.describe(),
                    "a bucket's capture starts"
                );
            }
            let started = Instant::now();
            let outcome = match self.runner_mut(runner) {
                Ok(named) => named.runner.serve(&job),
                Err(error) => Err(error),
            }
            .and_then(|produced| {
                // The step's end marker goes right behind its last launch: launched after the wait
                // below, its span would hold the host's trip back as well as the device's work.
                if index == last
                    && let (Some((step, stage, stream)), Some(fences)) =
                        (self.step_end.take(), crate::profiling::fences::Fences::installed())
                {
                    fences.mark(&stream, step, stage, false)?;
                }
                if self.waits {
                    self.synchronize()?;
                }
                Ok(produced)
            });
            if let Some(bucket) = walked {
                let elapsed_ms = started.elapsed().as_millis() as u64;
                match &outcome {
                    Ok(_) => {
                        tracing::info!(
                            bucket = %bucket,
                            elapsed_ms,
                            "the bucket's capture finished"
                        );
                    }
                    Err(error) => {
                        // The reason first, on a line of its own: a reader's log tail trims a line
                        // at a fixed width, and a reason printed after a bucket name and an elapsed
                        // time is a reason no run shows. The bucket follows, so the two readings a
                        // failure needs - what stopped and which bucket - are one line apart.
                        tracing::warn!(%error, "a bucket's capture failed");
                        // The error text opens with the operator and its descriptor, and a
                        // descriptor is a hundred characters: whatever prints a trimmed line keeps
                        // the head and loses the REASON, which is the end of the text. Print the
                        // end on a line of its own, so the reason survives any width.
                        let text = error.to_string();
                        let tail: String = {
                            let mut chars = text.chars().rev().take(220).collect::<Vec<_>>();
                            chars.reverse();
                            chars.into_iter().collect()
                        };
                        tracing::warn!("a capture's reason, as the end of its text: {tail}");
                        tracing::info!(bucket = %bucket, elapsed_ms, "the bucket whose capture failed");
                    }
                }
            }
            let produced = outcome?;
            self.counters.time(started.elapsed());
            if !produced.calls.is_empty() {
                if !output.calls.is_empty() {
                    return Err(Error::Invariant(
                        "two jobs of one command produced tokens".into(),
                    ));
                }
                output = produced;
            }
        }
        Ok(output)
    }

    /// A step begins: its session is told, and its begin marker goes on the stream it replays on.
    fn begin_step(&mut self, batch: &Batch, stream: Option<&platform::Stream>) -> Result<()> {
        let stage = batch.shape.variant.stage();
        if let Some(mut session) = self.session.take() {
            // A collector starts with nothing of the rank in flight, or it would record the end of
            // the step before.
            let applied = if session.begin(batch.id.0, stage) {
                self.synchronize().map(|()| session.apply())
            } else {
                Ok(())
            };
            self.session = Some(session);
            applied?;
        }
        if let (Some(fences), Some(stream)) =
            (crate::profiling::fences::Fences::installed(), stream)
        {
            fences.mark(stream, batch.id.0, stage, true)?;
            self.step_end = Some((batch.id.0, stage, stream.clone()));
        }
        Ok(())
    }

    /// A step's forward ended: its session is told, and what the rank observed since its last result
    /// travels with this one.
    fn end_step(&mut self, step: u64) -> Result<Option<ipc::ForwardStats>> {
        if let Some(mut session) = self.session.take() {
            let applied = if session.end(step) {
                self.synchronize().map(|()| session.apply())
            } else {
                Ok(())
            };
            self.session = Some(session);
            applied?;
        }
        let mut stats = ipc::ForwardStats::default();
        let current = platform::cupti::blocking::Calls::current();
        crate::profiling::counters::add_blocking("device", &current.since(&self.blocking));
        self.blocking = current;
        stats.blocking_calls = crate::profiling::counters::take_blocking();
        let activity = platform::cupti::activity::take();
        if activity.dropped_records > 0 {
            tracing::warn!(
                "CUPTI dropped {} activity records; the streams' device time misses their \
                 operations",
                activity.dropped_records
            );
        }
        for stream in &activity.streams {
            for (state, us) in [
                ("busy", stream.busy_us),
                ("gap", stream.gap_us),
                ("late", stream.late_us),
            ] {
                if us > 0 {
                    *stats
                        .stream_us
                        .entry(format!("{}/{state}", stream.name))
                        .or_default() += us;
                }
            }
        }
        if let Some(fences) = crate::profiling::fences::Fences::installed() {
            fences.poll_device(self.device, false);
            let (histograms, lost) = fences.take(self.device);
            stats.fence_us = histograms;
            stats.fence_records_lost = lost;
        }
        if let Some(session) = &mut self.session {
            stats.profile = session.progress();
        }
        let observed = !stats.blocking_calls.is_empty()
            || !stats.stream_us.is_empty()
            || !stats.fence_us.is_empty()
            || stats.fence_records_lost > 0
            || !stats.profile.is_empty();
        Ok(observed.then_some(stats))
    }

    /// Performs one change of the rank's profiling session, with nothing of the rank in flight.
    fn profile(&mut self, change: crate::profiling::session::Change) -> Result<serde_json::Value> {
        use crate::profiling::session::{Change, Session};
        match change {
            Change::Start { start, rank } => {
                if let Some(session) = &self.session {
                    return Err(Error::Invalid(format!(
                        "profiling session {} records the rank",
                        session.id()
                    )));
                }
                let (session, answer) = Session::start(*start, rank, self.device)?;
                self.session = session;
                Ok(answer)
            }
            Change::Stop {
                session,
                reason,
                rank,
            } => match self.session.take() {
                Some(running) if running.id() == session => {
                    let mut answer = running.stop(&reason);
                    answer["rank"] = serde_json::json!(rank);
                    Ok(answer)
                }
                other => {
                    self.session = other;
                    Ok(serde_json::json!({ "rank": rank, "session": session, "selected": false }))
                }
            },
            Change::Release {
                session,
                collector,
                reason,
                rank,
            } => {
                let mut answer = match &mut self.session {
                    Some(running) if running.id() == session => {
                        running.release(collector, &reason)
                    }
                    _ => serde_json::json!({ "released": false }),
                };
                answer["rank"] = serde_json::json!(rank);
                Ok(answer)
            }
        }
    }

    /// The bucket a job walks, when it walks one.
    ///
    /// A capture and a rebind both record a walk of one bucket into a graph, and both are silent
    /// device work of minutes; a replay and a shadow answer something the caller is waiting on.
    fn walked_bucket(job: &Job) -> Option<Bucket> {
        match job {
            Job::Capture { bucket, .. } | Job::Rebind { bucket, .. } => Some(*bucket),
            _ => None,
        }
    }

    /// Waits for the streams this device's runners issued on.
    ///
    /// A runner owns the stream its graphs replay on, so waiting on those is exactly waiting for
    /// this device's issued work; the executor has no stream of its own to add.
    fn synchronize(&self) -> Result<()> {
        for named in &self.runners {
            if let Some(stream) = named.runner.stream() {
                stream.synchronize()?;
            }
        }
        Ok(())
    }

    /// The runner that serves `phase`.
    fn runner_of(&self, phase: ipc::PhaseId) -> Result<RunnerId> {
        self.runners
            .iter()
            .find(|named| named.serves(phase))
            .map(|named| named.id)
            .ok_or_else(|| {
                Error::Invariant(format!("no runner of this device serves phase {}", phase.0))
            })
    }

    /// The runner with `id`, to read.
    fn runner_ref(&self, id: RunnerId) -> Result<&NamedRunner> {
        self.runners
            .iter()
            .find(|named| named.id == id)
            .ok_or_else(|| Error::Invariant(format!("the device has no runner {}", id.0)))
    }

    /// The runner with `id`.
    /// The rank's Storage Server client, or the refusal a sleep and a wake give when none is bound.
    ///
    /// The handle comes back owned so the caller holds it for the length of the call; the slot keeps
    /// its own share, so a control thread that binds it once serves every park of the epoch.
    fn storage(&self) -> Result<Arc<crate::memory::retained::Retained>> {
        self.storage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or_else(|| {
                Error::State(
                    "this device has no Storage Server client: a standby cannot give its resident \
                     cache segments back or map them again"
                        .into(),
                )
            })
    }

    /// The distinct caches this device's runners read, one per cache.
    fn caches(&self) -> Vec<Arc<std::sync::Mutex<crate::memory::prefix_cache::PrefixCache>>> {
        let mut caches = Vec::new();
        let mut addresses: Vec<*const ()> = Vec::new();
        for named in &self.runners {
            let Some(cache) = named.runner.cache() else {
                continue;
            };
            let address = Arc::as_ptr(&cache) as *const ();
            if addresses.contains(&address) {
                continue;
            }
            addresses.push(address);
            caches.push(cache);
        }
        caches
    }

    fn runner_mut(&mut self, id: RunnerId) -> Result<&mut NamedRunner> {
        self.runners
            .iter_mut()
            .find(|named| named.id == id)
            .ok_or_else(|| Error::Invariant(format!("the device has no runner {}", id.0)))
    }
}

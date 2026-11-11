//! The scheduler's one door to execution.
//!
//! The step submits a batch, takes the batches that completed, and asks what
//! the pools can do; it does not know what runs a batch. This module is the
//! door: every path of the scheduler that executes something comes through
//! it, to the batch interface of this process's devices
//! (`ipc::execution::Batches`).
//!
//! The door also holds what ends the scheduler's wait. A scheduler with
//! nothing to do blocks in [`Submissions::wait`] until something arrives: a
//! command or a reader that freed room in an event channel
//! ([`Submissions::waker`]), a submission that settled on its device, or an
//! response on a Storage Server connection. Nothing polls for any of them: the
//! wake is an eventfd, waited on beside the connections in one `poll(2)`.

use std::collections::{BTreeMap, HashMap};
use std::os::fd::{BorrowedFd, OwnedFd};
use std::sync::Arc;
use std::time::Instant;

use rustix::event::{EventfdFlags, PollFd, PollFlags, Timespec};

use crate::WorkerGroupId;
use crate::worker::{WorkerError, WorkerInfo};
use ipc::execution::{Batch, BatchId, BatchOutput, Batches, Submission};
use ipc::execution::{EpochOperation, Outcome};

/// What the scheduler executes through: the batch interface of this
/// process's own devices (`ipc::execution::Batches`).
///
/// A batch reaches a device as the scheduler laid it out, and its submission
/// is held here until the device settles it; the scheduler takes the outcome
/// at its next step ([`Self::take_completed`]). The scheduler thread never
/// blocks on one device's completion: it waits for whatever arrives first.
///
/// Public because a scheduler is built with one: `Scheduler::new` takes a
/// `Submissions`, made with `over_batches`.
pub struct Submissions {
    batches: Arc<dyn Batches>,
    info: WorkerInfo,
    /// What ends the scheduler's wait ([`Self::wait`]): an eventfd that
    /// [`Self::waker`] hands out and every held submission signals when it
    /// settles.
    wake: Arc<OwnedFd>,
    /// The batches submitted and not yet taken as completed, in submission
    /// order.
    held: Vec<(BatchId, Submission)>,
    /// Shadow submissions in flight.
    ///
    /// A shadow batch's paired forwards compare themselves on the device, and
    /// the comparison's verdict is this submission's outcome - no other path
    /// carries it to the update. The scheduler reads them in `shadow_status`,
    /// which is why a submission is kept until it has finished.
    shadows: Vec<Submission>,
    /// The pool devices each batch in flight holds a slot of, by batch.
    ///
    /// A device of a pool admits `queue_depth` batches at once, and that bound is what makes a step
    /// a batch: a request becomes ready when its own batch ends, so a scheduler that refills at
    /// once sends it alone, the batch after it carries one row again, and requests that run
    /// together never meet in a step. While the slots are taken the ready requests wait for each
    /// other. The bound is each device's: a batch takes a slot on every device it runs on, so a
    /// step that runs on all of them takes one everywhere, and the clips a media request decodes
    /// on different devices go out together.
    slots: HashMap<BatchId, Vec<(WorkerGroupId, usize)>>,
    /// How many batches hold a slot of each device of each pool.
    occupied: HashMap<(WorkerGroupId, usize), usize>,
}

/// The state of the shadow submissions an update's pass asked for.
///
/// The pass gets its response by the comparison's verdict, not by the fact that the
/// batch was submitted: a paired forward that failed on the device is the
/// update's failure, and responding earlier let a wrong kernel reach the
/// switch.
pub(super) enum ShadowStatus {
    /// At least one shadow batch has not finished.
    Pending,
    /// Every shadow batch finished and every pair was within tolerance.
    Settled,
    /// A shadow batch failed, with the reason it failed for.
    Failed(String),
}

impl Submissions {
    /// The door over this process's own devices.
    ///
    /// The pools come from the worker's own inventory and nowhere else: a
    /// worker that states no pool is refused here rather than served with a
    /// guessed one, for the same reason `WorkerInfo::from_inventory` refuses
    /// a pool that states no rank or no longest sequence.
    pub fn over_batches(batches: Arc<dyn Batches>) -> anyhow::Result<Submissions> {
        let info = WorkerInfo::from_inventory(batches.inventory())?;
        anyhow::ensure!(
            !info.workers.is_empty(),
            "the in-process worker states no pool to schedule against"
        );
        // A batch carries the columns the kernels read, and every KV view is
        // shaped to the longest sequence the deployment was built for: without
        // that number no batch is servable, so it is a fact of the boot and not
        // something to discover at the first step as "nothing was dispatched".
        anyhow::ensure!(
            info.max_model_len > 0,
            "the in-process worker states no longest sequence; a policy cannot be shaped to a \
             KV view without one"
        );
        // Nonblocking: a wake adds to the counter and a wait reads it empty,
        // so wakes that arrive together end one wait.
        let wake = rustix::event::eventfd(0, EventfdFlags::NONBLOCK | EventfdFlags::CLOEXEC)
            .map_err(|error| anyhow::anyhow!("opening the scheduler's wake: {error}"))?;
        Ok(Submissions {
            batches,
            info,
            wake: Arc::new(wake),
            held: Vec::new(),
            shadows: Vec::new(),
            slots: HashMap::new(),
            occupied: HashMap::new(),
        })
    }

    /// What ends this scheduler's wait when something other than a batch's
    /// completion arrives: a command, or a reader that freed room in an event
    /// channel.
    ///
    /// The scheduler blocks while nothing arrives, so a command that did not
    /// wake it would wait for the next completion.
    pub fn waker(&self) -> crate::CommandWaker {
        let wake = Arc::clone(&self.wake);
        crate::CommandWaker::new(move || signal(&wake))
    }

    /// Blocks until something arrived since the last wait: a wake, or an
    /// response or event on one of `connections`, the scheduler's Storage
    /// Server connections. `deadline` is a bound of the scheduler's own that
    /// passes without any arrival (a starvation bound, a wait for a
    /// predecessor's index, a prefetch's patience, a drain's silence); with
    /// none, the wait has no end but an arrival.
    ///
    /// A wait that fails (an interrupted call) returns as an arrival does:
    /// the scheduler takes a step that finds nothing and waits again.
    pub(super) fn wait(&self, deadline: Option<Instant>, connections: &[BorrowedFd<'_>]) {
        let mut waited: Vec<PollFd<'_>> = Vec::with_capacity(1 + connections.len());
        waited.push(PollFd::new(&*self.wake, PollFlags::IN));
        waited.extend(
            connections
                .iter()
                .map(|connection| PollFd::from_borrowed_fd(*connection, PollFlags::IN)),
        );
        let remaining = deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
            .and_then(|remaining| Timespec::try_from(remaining).ok());
        let _ = rustix::event::poll(&mut waited, remaining.as_ref());

        // Every wake since the last wait is taken at once: the step that
        // follows reads everything that arrived.
        let mut counter = [0u8; 8];
        let _ = rustix::io::read(&*self.wake, &mut counter);
    }

    /// The pools' capabilities: their ranks, stages and limits.
    pub(super) fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Whether every rank of the pool `group` is ready to accept execution:
    /// the door serves the pools its inventory states and no others.
    pub(super) fn is_ready(&self, group: &WorkerGroupId) -> bool {
        self.info.workers.iter().any(|(id, _)| id == group)
    }

    /// Whether `group` is a stated pool with a slot free on some device: a
    /// batch may still be formed for it, and [`Self::reserve`] holds back one
    /// whose own devices are full.
    pub(super) fn has_capacity(&self, group: &WorkerGroupId) -> bool {
        self.is_ready(group)
            && (0..self.batches.device_count().max(1))
                .any(|device| self.slot_free(&(group.clone(), device)))
    }

    /// Whether one device of a pool has a slot free: fewer of its batches in
    /// flight than the pool's `queue_depth`.
    fn slot_free(&self, slot: &(WorkerGroupId, usize)) -> bool {
        let depth = self
            .info
            .layouts
            .get(&slot.0)
            .map_or(1, |layout| layout.queue_depth.max(1));
        self.occupied.get(slot).copied().unwrap_or(0) < depth
    }

    /// Takes a slot for `batch` on every device of `pools` it runs on, when
    /// each has one free, and responds whether it did. A batch that took its
    /// slots is submitted next; one that did not waits for a completion.
    ///
    /// # Errors
    ///
    /// When the worker cannot say which devices run the batch.
    pub(super) fn reserve(
        &mut self,
        batch: &Batch,
        pools: &[WorkerGroupId],
    ) -> Result<bool, WorkerError> {
        let devices = self
            .batches
            .devices(batch)
            .map_err(|error| anyhow::anyhow!(error))?;
        let slots: Vec<(WorkerGroupId, usize)> = pools
            .iter()
            .flat_map(|pool| devices.iter().map(move |index| (pool.clone(), *index)))
            .collect();
        if !slots
            .iter()
            .all(|slot| self.is_ready(&slot.0) && self.slot_free(slot))
        {
            return Ok(false);
        }
        for slot in &slots {
            *self.occupied.entry(slot.clone()).or_default() += 1;
        }
        self.slots.insert(batch.id, slots);
        Ok(true)
    }

    /// Gives back the slots `batch` held: it completed.
    pub(super) fn release(&mut self, batch: BatchId) {
        for slot in self.slots.remove(&batch).unwrap_or_default() {
            if let Some(count) = self.occupied.get_mut(&slot) {
                *count = count.saturating_sub(1);
            }
        }
    }

    /// Gives back every slot: the devices failed, and the scheduler retires
    /// what they held without waiting for their batches.
    pub(super) fn release_all(&mut self) {
        self.slots.clear();
        self.occupied.clear();
    }

    /// Submits `batch` to the devices of its group and holds its submission
    /// until they settle it.
    pub(super) fn submit(&mut self, batch: Batch) -> Result<(), WorkerError> {
        let id = batch.id;
        let submission = self
            .batches
            .submit(batch)
            .map_err(|error| anyhow::anyhow!(error))?;
        self.hold(id, submission);
        Ok(())
    }

    /// Copies `pages` to the host KV cache as batch `id`, which carries no
    /// call: pages that wait while nothing runs ride no forward.
    pub(super) fn copy_pages(
        &mut self,
        id: BatchId,
        pages: Vec<ipc::PageCopy>,
    ) -> Result<(), WorkerError> {
        let submission = self
            .batches
            .copy_pages(pages)
            .map_err(|error| anyhow::anyhow!(error))?;
        self.hold(id, submission);
        Ok(())
    }

    /// Copies preempted requests' trajectories to the host latent pools as
    /// batch `id`, which carries no call: a trajectory waits for nothing on
    /// the device but the batches that advanced it.
    pub(super) fn offload_latents(
        &mut self,
        id: BatchId,
        latents: Vec<ipc::LatentOffload>,
    ) -> Result<(), WorkerError> {
        let submission = self
            .batches
            .offload_latents(latents)
            .map_err(|error| anyhow::anyhow!(error))?;
        self.hold(id, submission);
        Ok(())
    }

    /// Keeps `submission` until it settles, which wakes the scheduler.
    fn hold(&mut self, id: BatchId, submission: Submission) {
        submission.on_settled(&self.settled());
        self.held.push((id, submission));
    }

    /// What a submission calls when it settles: the scheduler's wake.
    fn settled(&self) -> ipc::execution::Waker {
        let wake = Arc::clone(&self.wake);
        Arc::new(move || signal(&wake))
    }

    /// The oldest submitted batch that settled, with its outcome; `None`
    /// while every batch in flight is still on its device.
    ///
    /// The devices finish in their own order, so a batch is taken when it
    /// ends rather than behind an earlier batch of another device.
    pub(super) fn take_completed(&mut self) -> Option<(BatchId, Outcome<BatchOutput>)> {
        let (index, outcome) = self
            .held
            .iter()
            .enumerate()
            .find_map(|(index, (_, submission))| Some((index, submission.outcome()?)))?;
        let (id, _) = self.held.remove(index);
        Some((id, outcome))
    }

    /// Submits `batch`, which names another generation than the one serving,
    /// for the update's paired forwards: the worker compares them, and the
    /// batch reports nothing of its own.
    pub(super) fn shadow(&mut self, batch: Batch) -> Result<(), WorkerError> {
        let submission = self
            .batches
            .shadow(batch)
            .map_err(|error| anyhow::anyhow!(error))?;
        // The update's pass gets its response at the step after its verdict.
        submission.on_settled(&self.settled());
        self.shadows.push(submission);
        self.shadows
            .retain(|submission| submission.outcome().is_none());
        Ok(())
    }

    /// The state of this pass's shadow submissions, reading every one that has
    /// finished and responding to the first failure.
    ///
    /// A failure carries the device thread's label beside its error, so the
    /// update states which paired forward of which operator and bucket left
    /// tolerance rather than only the element that did.
    pub(super) fn shadow_status(&mut self) -> ShadowStatus {
        let mut failure: Option<String> = None;
        self.shadows
            .retain(|submission| match submission.outcome() {
                None => true,
                Some(Outcome::Success(_)) => false,
                Some(Outcome::Failed(error)) => {
                    let reason = match submission.failure_label() {
                        Some(label) => format!("{label}: {error}"),
                        None => error.to_string(),
                    };
                    failure.get_or_insert(reason);
                    false
                }
                Some(Outcome::Cancelled) => {
                    failure.get_or_insert_with(|| {
                        "a shadow batch was cancelled before its comparison ran".to_owned()
                    });
                    false
                }
            });
        match failure {
            Some(reason) => ShadowStatus::Failed(reason),
            None if self.shadows.is_empty() => ShadowStatus::Settled,
            None => ShadowStatus::Pending,
        }
    }

    /// The extent of the bucket the worker would serve `computation` in, when
    /// it serves one.
    ///
    /// The scheduler asks before it forms a media batch, because the batch
    /// states the extent its kernels were captured for and only the runner
    /// knows it: a bucket captured for one resolution serves no other, and
    /// deriving the extent in the scheduler would be a second statement of
    /// the runner's buckets. `None` is the response for a computation no device
    /// of this rank serves.
    pub(super) fn media(
        &self,
        phase: ipc::PhaseId,
        computation: &ipc::execution::Computation,
    ) -> Option<ipc::execution::Extent> {
        self.batches.media(phase, computation).ok().flatten()
    }

    /// Maps the host KV cache's page pools on every rank of the pools, each
    /// holding `tokens` tokens of every cache group.
    ///
    /// The ranks of this process map their pools through the worker that owns
    /// their caches: its device threads issue the copies.
    pub(super) fn map_host_cache(
        &mut self,
        tokens: u64,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, WorkerError> {
        let mapped = self
            .batches
            .map_host_cache(tokens)
            .map_err(|error| anyhow::anyhow!(error))?;
        mapped
            .into_iter()
            .map(|(group, ranks)| {
                let results = ranks
                    .iter()
                    .map(|pools| serde_json::to_string(&Some(pools)))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| anyhow::anyhow!(error))?;
                Ok((WorkerGroupId(group), results))
            })
            .collect()
    }

    /// Performs the storage `operation` on every rank and returns what each
    /// reported.
    pub(super) fn operate(
        &mut self,
        operation: &EpochOperation,
    ) -> Result<Vec<String>, WorkerError> {
        self.batches
            .operate(operation)
            .map_err(|error| anyhow::anyhow!(error))
    }

    /// Performs the storage `operation` on every rank and returns what each
    /// reported, by pool.
    ///
    /// This process's data path responds for its ranks in rank order. Every
    /// group of a scheduler indexes the one unit space its ranks agreed their
    /// cache over, so each group states what every rank reports.
    pub(super) fn operate_groups(
        &mut self,
        operation: &EpochOperation,
    ) -> Result<Vec<(WorkerGroupId, Vec<String>)>, WorkerError> {
        let results = self.operate(operation)?;
        if results.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self
            .info
            .workers
            .iter()
            .map(|(group, _)| (group.clone(), results.clone()))
            .collect())
    }

    /// Applies the expert `order` on every rank and returns what each
    /// reported.
    pub(super) fn commit_experts(
        &mut self,
        order: &BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<String>, WorkerError> {
        self.batches
            .commit_experts(order)
            .map_err(|error| anyhow::anyhow!(error))
    }
}

/// Ends the wait on `wake`. A counter that is already full ends it too, so
/// the write's failure is nothing to report.
fn signal(wake: &OwnedFd) {
    let _ = rustix::io::write(wake, &1u64.to_ne_bytes());
}

/// The phase id the device worker addresses a batch by.
///
/// The scheduler names phases with `scheduling`'s ids and the worker with
/// `ipc`'s; both are the same fixed vocabulary in the same order, and this is
/// the one place the two meet, so neither type grows a second numbering.
pub(super) fn wire_phase(kind: ipc::CallKind) -> Option<ipc::PhaseId> {
    let id = crate::views::phase_id(crate::views::call_phase(kind));
    u16::try_from(id.0).ok().map(ipc::PhaseId)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use ipc::execution::{Computation, Extent, MediaColumns, StagedRequest, Variant};
    use ipc::{CallKind, MediaCall};

    /// The door over a fake pool named `gpu` of `devices` devices, two batches
    /// in flight a device.
    fn door(devices: usize) -> (Submissions, WorkerGroupId) {
        let mut pool = testkit::pool("gpu");
        pool.layout.queue_depth = 2;
        let fake = testkit::FakeBatches::new(vec![pool]).with_devices(devices);
        let submissions =
            Submissions::over_batches(Arc::new(fake)).expect("the fake states a pool");
        let pool = submissions.info().workers[0].0.clone();
        (submissions, pool)
    }

    /// One denoising step of one request as batch `id` of pool `gpu`.
    fn denoise(id: u64) -> Batch {
        let computation = Computation {
            call: MediaCall::Denoising,
            rows: 1,
            tokens: 0,
            extent: Extent::Request {
                frames: 5,
                width: 128,
                height: 128,
                text_rows: 4,
                condition_rows: 0,
            },
        };
        let staged = StagedRequest {
            request: ipc::RequestKey::new(1, ipc::RequestId(9), 0),
            conditioning: vec![1, 2, 3],
            step: 2,
            steps: 8,
            seed: 0x5eed,
            width: 768,
            height: 768,
            frames: 121,
            rounds: 0,
            unit: 0,
        };
        Batch {
            id: BatchId(id),
            phase: wire_phase(CallKind::Media(MediaCall::Denoising)).expect("a media phase"),
            group: "gpu".to_owned(),
            calls: vec![ipc::execution::CallId(0)],
            shape: ipc::execution::Shape {
                rows: 1,
                tokens: 0,
                variant: Variant::Denoise,
            },
            columns: ipc::execution::BatchColumns::default(),
            generation: 0,
            cache_units: Vec::new(),
            block_tables: Vec::new(),
            page_loads: Vec::new(),
            page_offloads: Vec::new(),
            latent_restores: Vec::new(),
            host_outputs: Vec::new(),
            images: Vec::new(),
            frees: Vec::new(),
            media: Some(MediaColumns::staged(computation, &[staged])),
        }
    }

    /// A device holds `queue_depth` batches in flight, and a completed batch
    /// gives its slot back.
    ///
    /// This is what makes requests that run together meet in one step: each is
    /// ready when its own batch ends, so a pool that always took the next
    /// batch would serve them one row at a time.
    #[test]
    fn a_pool_holds_no_more_batches_than_its_queue_depth() {
        let (mut submissions, pool) = door(1);
        let pools = [pool.clone()];

        assert!(submissions.has_capacity(&pool), "an idle pool has room");
        for id in 1..=2 {
            assert!(
                submissions
                    .reserve(&denoise(id), &pools)
                    .expect("the fake states its devices"),
                "a batch within the depth takes its slot"
            );
        }
        assert!(!submissions.has_capacity(&pool), "the depth is taken");
        assert!(
            !submissions
                .reserve(&denoise(3), &pools)
                .expect("the fake states its devices"),
            "a batch past the depth waits for a slot"
        );

        submissions.release(BatchId(1));
        assert!(
            submissions.has_capacity(&pool),
            "a completion frees its slot"
        );
        assert!(
            submissions
                .reserve(&denoise(3), &pools)
                .expect("the fake states its devices"),
            "the batch that waited takes the freed slot"
        );
    }

    /// A batch takes a slot on every device it runs on: a pool of two devices
    /// at depth two holds two steps that run on both, and a third waits.
    #[test]
    fn a_batch_takes_a_slot_on_every_device_it_runs_on() {
        let (mut submissions, pool) = door(2);
        let pools = [pool.clone()];
        for id in 1..=2 {
            assert!(
                submissions
                    .reserve(&denoise(id), &pools)
                    .expect("the fake states its devices")
            );
        }
        assert!(
            !submissions.has_capacity(&pool),
            "both devices hold both steps"
        );
        submissions.release_all();
        assert!(
            submissions.has_capacity(&pool),
            "failed devices hold nothing"
        );
    }

    /// A submitted batch is taken once, when its device settled it, and its
    /// settling ends the scheduler's wait.
    #[test]
    fn a_settled_batch_wakes_the_wait_and_is_taken_once() {
        let mut pool = testkit::pool("gpu");
        pool.layout.queue_depth = 2;
        let fake = Arc::new(
            testkit::FakeBatches::new(vec![pool])
                .with_media_outputs(Arc::new(testkit::Media::new())),
        );
        fake.hold();
        let mut submissions =
            Submissions::over_batches(Arc::clone(&fake) as Arc<dyn Batches>).expect("a pool");

        submissions.submit(denoise(7)).expect("the fake accepts");
        assert!(
            submissions.take_completed().is_none(),
            "a batch on its device has no outcome"
        );

        fake.release(1);
        // The settled submission woke the wait, so this returns at once.
        submissions.wait(None, &[]);
        let (id, outcome) = submissions.take_completed().expect("the batch settled");
        assert_eq!(id, BatchId(7));
        assert!(matches!(outcome, Outcome::Success(_)));
        assert!(submissions.take_completed().is_none(), "taken once");
    }
}

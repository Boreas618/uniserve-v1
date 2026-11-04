//! One step's work across the boundary from the scheduling mechanism to the executor.
//!
//! The mechanism decides this step's work (specs/worker.md 4.4) and writes it in the layout the
//! kernels read; the runner pads it to a bucket and serves it (4.5). This module is that crossing:
//!
//! - [`Batch`] is one phase of one step: the host columns the mechanism wrote, the calls whose
//!   progress the batch advances, the shape the runner pads to, and the generation it serves with;
//! - [`BatchColumns`] is the host-side layout (the lead's ruling in exec-work M81 and M104): plain
//!   host vectors, because the mechanism owns no device and no stream. Staging them into the pinned
//!   arena, uploading them and building the device-side `nn::TokenBatch`/`nn::KvView` is the
//!   runner's work;
//! - [`Jobs`] is what a runner asks its device executor to issue for a batch, and [`Submission`]
//!   is what the caller holds until every job of the submission ended.
//!
//! A submission is one or more jobs because a program-layer update interleaves work with serving:
//! a batch is a `Replay` of its bucket's graph, a precopy is the `Capture`/`Rebind` jobs of the
//! buckets a changed operator uses, and switchover and purge are submissions of their own so they
//! land on a batch boundary (7.3).

use std::ops::Range;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ipc::PhaseId;

use crate::completion::Outcome;
use crate::plan::{Bucket, Shape};
use crate::stream::recorded;
use crate::{Error, Result, lock};

/// The scheduling mechanism's name for one submission: monotonic in the order the mechanism made
/// them, so the control thread can report which batch a stalled job belongs to (2.4).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SubmissionId(pub u64);

/// The scheduling mechanism's name for one batch, unique within an epoch.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BatchId(pub u64);

/// The mechanism's name for one call: a request's phase in this batch, not the request itself.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CallId(pub u64);

/// One batch: one phase of one step, as the mechanism assembles it.
///
/// `shape` is what the runner pads to. It is the mechanism's statement of the batch's size — the
/// rows and tokens it wrote — and the runner serves the batch from the smallest captured bucket
/// that holds it; a shape no bucket holds is an error, never an eager fallback (4.5).
pub struct Batch {
    /// The batch's number, for the stall report and the completion the mechanism reads (2.4).
    pub id: BatchId,
    /// The phase this batch advances, as the scheduler addresses it (`ipc::PhaseId`).
    pub phase: PhaseId,
    /// The calls whose progress this batch advances, in scheduling order.
    pub calls: Vec<CallId>,
    /// The rows and tokens the mechanism wrote, and what the batch computes.
    pub shape: Shape,
    /// The host columns the mechanism wrote, in the layout the phase's kernels read.
    pub columns: BatchColumns,
    /// The generation whose kernels serve this batch (4.2, 7.3).
    pub generation: u64,
}

/// One call of a batch: which request slot it evaluates and which rows it owns.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BatchCall {
    /// The runner's request slot (the design's `request_pool_idx`), which addresses the model's
    /// per-slot state; not the mechanism's request identity.
    pub request: u32,
    /// What kind of call the rows evaluate.
    pub kind: CallKind,
    /// The attention cache group whose KV pages the rows write and attend.
    pub cache_group: u32,
    /// The half-open range of the batch's token columns these rows occupy.
    pub rows: Range<u32>,
}

/// What a call of a causal language model evaluates in this slice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallKind {
    /// A chunk of prompt tokens, writing the KV cache.
    Prefill,
    /// One token per row drawn from the slot state.
    Decode,
}

/// The host columns of one batch, in the layout the phase's kernels read.
///
/// The mechanism writes these from the request state it owns; the executor stages them into the
/// pinned arena and uploads them on the batch's stream, then builds the device views the model
/// reads. Every vector has one entry per token, or one per row where a field says so; the runner
/// pads them to the bucket it serves and never reorders them.
#[derive(Clone, Default, Debug)]
pub struct BatchColumns {
    /// One entry per call, in scheduling order.
    pub calls: Vec<BatchCall>,
    /// Every query token, concatenated over the calls in `calls` order.
    pub input_ids: Vec<u32>,
    /// The absolute position of each query token in its request's sequence (RoPE).
    pub positions: Vec<u32>,
    /// One KV slot per query token: the cache row the KV-write kernel writes; `-1` for a token
    /// that does not persist past its visible prefix.
    pub slot_mapping: Vec<i64>,
    /// Per row: the cached prefix plus the query length, the length attention reads.
    pub seq_lens: Vec<u32>,
    /// Per row: how many query tokens this call evaluates.
    pub query_lens: Vec<u32>,
    /// Per row: whether the query interval persists in the prefix cache (`u8`, 0 or 1).
    pub write_cache: Vec<u8>,
    /// Per row: the token row whose hidden state feeds `logits`, for rows that sample.
    pub logits_rows: Vec<u32>,
    /// The total number of query tokens, so the runner sizes its bucket without recounting.
    pub num_tokens: u32,
    /// The KV page size in tokens, the same number the policy sees in `Room`.
    pub page_size: u32,
}

impl BatchColumns {
    /// The number of rows (sequences) the columns describe.
    pub fn rows(&self) -> usize {
        self.calls.len()
    }

    /// The number of query tokens the columns describe.
    pub fn tokens(&self) -> usize {
        self.input_ids.len()
    }

    /// Whether every column has the length its field promises for `shape`, so that a mis-sized
    /// batch fails before a launch rather than after one.
    pub fn check(&self, shape: &Shape) -> Result<()> {
        for (name, actual, expected) in [
            ("input_ids", self.input_ids.len(), self.num_tokens as usize),
            ("positions", self.positions.len(), self.num_tokens as usize),
            (
                "slot_mapping",
                self.slot_mapping.len(),
                self.num_tokens as usize,
            ),
        ] {
            if actual != expected {
                return Err(Error::Invalid(format!(
                    "{name} has {actual} entries, the batch has {} tokens",
                    self.num_tokens
                )));
            }
        }
        for (name, actual) in [
            ("seq_lens", self.seq_lens.len()),
            ("query_lens", self.query_lens.len()),
            ("write_cache", self.write_cache.len()),
            ("logits_rows", self.logits_rows.len()),
        ] {
            if actual != self.calls.len() {
                return Err(Error::Invalid(format!(
                    "{name} has {actual} entries, the batch has {} calls",
                    self.calls.len()
                )));
            }
        }
        if self.num_tokens > shape.tokens {
            return Err(Error::Invalid(format!(
                "the batch holds {} tokens, more than its shape's {}",
                self.num_tokens, shape.tokens
            )));
        }
        Ok(())
    }
}

/// The work a runner asks its device executor to issue for one submission.
///
/// The runner stages what the batch needs before it returns; a job is the device work that follows
/// the staging, in the order this list holds it. Each runner job names the runner that performs it,
/// because only the runner owns the model and the arena.
#[derive(Default)]
pub struct Jobs {
    jobs: Vec<Job>,
}

impl Jobs {
    /// A submission of `jobs`, in order.
    pub fn new(jobs: impl IntoIterator<Item = Job>) -> Self {
        Self {
            jobs: jobs.into_iter().collect(),
        }
    }

    /// A submission with nothing for the device to issue.
    pub fn none() -> Self {
        Self::default()
    }

    /// Every job, in order.
    pub fn iter(&self) -> impl Iterator<Item = &Job> {
        self.jobs.iter()
    }

    /// How many jobs the submission holds.
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Whether the submission holds no job.
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }
}

/// One job the device executor issues, in submission order.
pub enum Job {
    /// Replays the captured graph of `bucket`: the runner staged the batch's inputs already.
    Replay {
        /// The runner that serves the batch.
        runner: crate::runner::RunnerId,
        /// The bucket whose graph is replayed.
        bucket: Bucket,
        /// The generation the bucket's graph belongs to.
        generation: u64,
        /// The batch the job serves, for the stall report (2.4).
        batch: BatchId,
    },
    /// Captures `bucket` for `generation`, between serving batches (4.5 recapture).
    Capture {
        /// The runner that captures the bucket.
        runner: crate::runner::RunnerId,
        /// The bucket to capture.
        bucket: Bucket,
        /// The generation the capture is for.
        generation: u64,
    },
    /// Runs the paired forward of one changed operator and compares its two results (7.3).
    Shadow {
        /// The runner that walks the paired forward and compares.
        runner: crate::runner::RunnerId,
        /// The operator whose new kernel runs beside its old one.
        operator: &'static str,
        /// The operator's descriptor key, which names the launch's kernel (4.2).
        key: String,
        /// The bucket whose static activations the pair uses.
        bucket: Bucket,
        /// The generation the new kernel belongs to.
        generation: u64,
    },
    /// Rebinds `bucket`'s launches for `generation` from that generation's precopy walk.
    Rebind {
        /// The runner that walks the forward and rebinds the graph.
        runner: crate::runner::RunnerId,
        /// The bucket to rebind.
        bucket: Bucket,
        /// The generation the rebind is for.
        generation: u64,
    },
    /// A storage owner's recorded issue work, performed on its stream in submission order.
    Issue {
        /// The recorded copies, fills, fences and waits.
        work: recorded::Batch,
    },
    /// A job whose only work is the wait for the events the event pool deferred work on.
    After {
        /// The events whose completion the job establishes.
        events: Vec<Arc<platform::Event>>,
    },
}

impl Job {
    /// What kind of job this is, for the watchdog's report.
    pub fn kind(&self) -> JobKind {
        match self {
            Self::Replay { .. } => JobKind::Replay,
            Self::Capture { .. } => JobKind::Capture,
            Self::Rebind { .. } => JobKind::Rebind,
            Self::Shadow { .. } => JobKind::Shadow,
            Self::Issue { .. } => JobKind::Issue,
            Self::After { .. } => JobKind::After,
        }
    }

    /// The batch a serving job belongs to, when it serves one (2.4's stall report names it).
    pub fn batch(&self) -> Option<BatchId> {
        match self {
            Self::Replay { batch, .. } => Some(*batch),
            _ => None,
        }
    }

    /// The runner that performs the job, if a runner does.
    pub fn runner(&self) -> Option<crate::runner::RunnerId> {
        match self {
            Self::Replay { runner, .. }
            | Self::Capture { runner, .. }
            | Self::Rebind { runner, .. }
            | Self::Shadow { runner, .. } => Some(*runner),
            Self::Issue { .. } | Self::After { .. } => None,
        }
    }
}

impl std::fmt::Debug for Job {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Replay {
                runner,
                bucket,
                generation,
                batch,
            } => formatter
                .debug_struct("Replay")
                .field("runner", runner)
                .field("bucket", bucket)
                .field("generation", generation)
                .field("batch", batch)
                .finish(),
            Self::Capture {
                runner,
                bucket,
                generation,
            } => formatter
                .debug_struct("Capture")
                .field("runner", runner)
                .field("bucket", bucket)
                .field("generation", generation)
                .finish(),
            Self::Shadow {
                runner,
                operator,
                key,
                bucket,
                generation,
            } => formatter
                .debug_struct("Shadow")
                .field("runner", runner)
                .field("operator", operator)
                .field("key", key)
                .field("bucket", bucket)
                .field("generation", generation)
                .finish(),
            Self::Rebind {
                runner,
                bucket,
                generation,
            } => formatter
                .debug_struct("Rebind")
                .field("runner", runner)
                .field("bucket", bucket)
                .field("generation", generation)
                .finish(),
            Self::Issue { work } => formatter
                .debug_struct("Issue")
                .field("work", &work.len())
                .finish(),
            Self::After { events } => formatter
                .debug_struct("After")
                .field("events", &events.len())
                .finish(),
        }
    }
}

/// What the device executor is doing, as the control thread's stall report and probe name it (2.4).
///
/// A batch is one command even when it becomes several jobs: the stall report names the batch the
/// device is serving, which is what the scheduler can act on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JobKind {
    /// Serving one batch: staging it and replaying its bucket's graph.
    Batch,
    /// Capturing every bucket of the serving generation before it serves.
    Warmup,
    /// Preparing the next generation (7.3 step 2).
    Precopy,
    /// Switching the serving generation at a batch boundary (7.3 step 3).
    Switchover,
    /// Releasing an old generation's graphs (7.3 step 6).
    Purge,
    /// A storage job: the copies, fills and fences a storage owner recorded.
    Issue,
    /// A job that only waits for the events it was given.
    After,
}

impl JobKind {
    /// The kind's name, as a stall report prints it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Warmup => "warmup",
            Self::Precopy => "precopy",
            Self::Switchover => "switchover",
            Self::Purge => "purge",
            Self::Issue => "issue",
            Self::After => "after",
        }
    }
}

/// What the caller of a submission holds until its jobs ended.
///
/// A submission completes once the device executor has issued every job's work and the work
/// completed on the device (the executor thread waits on its own stream's work before reporting a
/// job, and with it the submission). The scheduler thread waits on the submission of a step at the
/// next step's hand-off, never on the device itself.
pub struct Submission {
    id: SubmissionId,
    state: Arc<SubmissionState>,
}

pub(crate) struct SubmissionState {
    outcome: Mutex<Option<Outcome>>,
    ready: Condvar,
}

impl Submission {
    /// A submission that has already completed with `outcome`.
    pub fn completed(id: SubmissionId, outcome: Outcome) -> Self {
        let state = Arc::new(SubmissionState {
            outcome: Mutex::new(Some(outcome)),
            ready: Condvar::new(),
        });
        Self { id, state }
    }

    /// A submission waiting for its jobs' completion, with the state the queue completes.
    pub(crate) fn pending(id: SubmissionId) -> (Self, Arc<SubmissionState>) {
        let state = Arc::new(SubmissionState {
            outcome: Mutex::new(None),
            ready: Condvar::new(),
        });
        (
            Self {
                id,
                state: Arc::clone(&state),
            },
            state,
        )
    }

    /// The submission's number, in the mechanism's order.
    pub fn id(&self) -> SubmissionId {
        self.id
    }

    /// The state the queue completes, and the submission's number.
    pub(crate) fn parts(&self) -> (Arc<SubmissionState>, SubmissionId) {
        (Arc::clone(&self.state), self.id)
    }

    /// Completes this submission with `outcome`, if something else did not already.
    pub(crate) fn complete(&self, outcome: Outcome) {
        Arc::clone(&self.state).finish(outcome);
    }

    /// Waits until the submission completed and answers its outcome.
    pub fn wait(&self) -> Outcome {
        let mut outcome = lock(&self.state.outcome);
        while outcome.is_none() {
            outcome = match self.state.ready.wait(outcome) {
                Ok(guard) => guard,
                // A panicking waiter does not make the submission incomplete.
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        outcome
            .clone()
            .unwrap_or(Outcome::Failed(Arc::new(Error::State(
                "a submission completed without an outcome",
            ))))
    }

    /// Waits at most `timeout` for the submission's completion.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Outcome> {
        let deadline = Instant::now() + timeout;
        let mut outcome = lock(&self.state.outcome);
        while outcome.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            outcome = match self.state.ready.wait_timeout(outcome, left) {
                Ok((guard, _)) => guard,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        outcome.clone()
    }

    /// The submission's outcome, when it completed.
    pub fn outcome(&self) -> Option<Outcome> {
        lock(&self.state.outcome).clone()
    }
}

impl SubmissionState {
    /// Completes the submission with `outcome`; the first completion wins.
    pub(crate) fn finish(self: Arc<Self>, outcome: Outcome) {
        let mut slot = lock(&self.outcome);
        if slot.is_none() {
            *slot = Some(outcome);
        }
        drop(slot);
        self.ready.notify_all();
    }
}

impl std::fmt::Debug for Submission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Submission")
            .field("id", &self.id)
            .field("completed", &self.outcome().is_some())
            .finish()
    }
}

impl std::fmt::Debug for SubmissionState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SubmissionState")
    }
}

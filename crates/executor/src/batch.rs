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

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ipc::{CallStatus, ComponentConfig, ErrorCode, PhaseId, TimingCounters, WorkerInfo};

use crate::completion::Outcome;
use crate::plan::{Bucket, Shape};
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

/// One call's completion, as the device half of 4.4 states it.
///
/// A batch advances a set of calls and each of them gets one entry here, whether or not it sampled
/// a token, because the mechanism's completion path is per call. A call this executor refuses never
/// reaches this type: [`Batches::submit`](crate::Batches::submit) answers `Error::Invalid`
/// synchronously for a batch it cannot serve, so an entry here is what execution did, and a
/// [`CallStatus::Error`] entry is always a device failure.
#[derive(Clone, Debug, PartialEq)]
pub struct CallOutput {
    /// The call the batch advanced.
    pub call: CallId,
    /// Whether execution performed the call.
    pub status: CallStatus,
    /// Why it failed, when it did: the class the mechanism acts on (`NumericalFault` fails the
    /// rows that computed non-finite logits, `InputError` rejects the request with this rank's
    /// message, `ResourceError` fails the call).
    pub error_code: Option<ErrorCode>,
    /// What this rank said about a refused input, when it refused one.
    pub error_message: Option<String>,
    /// The tokens the sampler accepted for this call, in the order the call's sampling rows took
    /// them. A call whose rows sampled nothing (a prefill chunk that ends no prompt, a padded row)
    /// has none, which is not the same as a failure.
    pub tokens: Vec<u32>,
    /// The slot's committed count for a text call; a latent trajectory reports its own step here
    /// once `ops::commit` gains a per-slot step column, and until then this field is the commit
    /// count for every call kind.
    ///
    /// It is read from the commit table's sampling position, which a text slot advances once per
    /// committed token and a latent slot advances once per denoising step.
    pub step_index: u32,
    /// The cache extent execution initialized, rejected speculative positions included: the slot's
    /// logical position after the commit.
    pub cache_computed_len: u32,
    /// The accepted prefix a successor may read: the slot's sampling position after the commit.
    pub cache_visible_len: u32,
    /// The log-probability of each accepted token, aligned with [`CallOutput::tokens`].
    ///
    /// Empty unless the deployment planned logprob buckets (4.5): producing them costs a
    /// full-vocabulary softmax on every sampling step and its `[rows, vocabulary]` buffer, so a
    /// deployment that serves without clients asking for them plans the plain buckets and every
    /// call reports this empty. **Empty means "not planned", never "zero".**
    pub token_logprobs: Vec<f32>,
    /// Ranked log-probabilities for each scored prompt position, in input order.
    pub prompt_logprobs: Vec<Vec<ipc::TokenLogprob>>,
    /// A token-denoising readout's candidate log-probabilities, in its own candidate order.
    pub candidate_logprobs: Vec<f32>,
    /// The finished artifact and its storage lifetime, when the call produced one.
    pub media_output: Option<ipc::MediaOutput>,
    /// The physical cache export, when the call publishes one.
    pub cache_output: Option<ipc::CacheTransfer>,
    /// The drafter's proposals after the last committed token.
    pub draft_token_ids: Vec<u32>,
    /// A video latent preparation's per-rank page layout.
    pub latent_layouts: Vec<ipc::RankLatentLayout>,
}

impl CallOutput {
    /// A completed call that sampled `tokens` and stands at `step_index`, from the slot state
    /// `state` the commit table left.
    pub fn completed(call: CallId, tokens: Vec<u32>, state: RowState) -> Self {
        Self {
            call,
            status: CallStatus::Ok,
            error_code: None,
            error_message: None,
            tokens,
            step_index: state.step_index,
            cache_computed_len: state.cache_computed_len,
            cache_visible_len: state.cache_visible_len,
            token_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            candidate_logprobs: Vec::new(),
            // A text call produces no artifact, publishes no export, drafts no token and lays out
            // no latent page: the mechanism reads these as empty, which is what happened.
            media_output: None,
            cache_output: None,
            draft_token_ids: Vec::new(),
            latent_layouts: Vec::new(),
        }
    }
}

/// Where the commit table left one call's slot, as the completion reads it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowState {
    /// The slot's trajectory position: the denoising trajectory's next step.
    pub step_index: u32,
    /// The cache extent execution initialized, rejected speculative positions included.
    pub cache_computed_len: u32,
    /// The accepted prefix a successor may read.
    pub cache_visible_len: u32,
}

/// What one serving step produced (4.4).
///
/// The mechanism learns it from the batch's [`Submission`], builds the next step's input ids from
/// the tokens, and folds the timing into the statistics of the calls' domain. A step that samples
/// nothing (a chunk of a prefill that ends no prompt, a warmup, a program-layer command) completes
/// with no call and nothing measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchOutput {
    /// One entry per call the batch advanced, in the order the batch named them.
    pub calls: Vec<CallOutput>,
    /// What the batch measured, in the segments this executor can observe.
    ///
    /// `queued_us` is the wait from the mechanism's submission to this device thread starting the
    /// batch, `device_us` the replay's launch to the stream's completion, and `host_us` the rest of
    /// the step on the host. `copy_us` is zero in-process and states a fact rather than a
    /// measurement: nothing is transferred for the completion, since the mechanism reads it here.
    pub timing: TimingCounters,
}

/// One batch: one phase of one step, as the mechanism assembles it.
///
/// `shape` is what the runner pads to. It is the mechanism's statement of the batch's size — the
/// rows and tokens it wrote — and the runner serves the batch from the smallest captured bucket
/// that holds it; a shape no bucket holds is an error, never an eager fallback (4.5).
#[derive(Clone)]
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
    /// What the mechanism allocated for this batch's cache pages: per group, the units it acquired
    /// and the `(source, destination)` copies a leased prefix needs before a call reads it.
    ///
    /// The mechanism computes these and owns the source lease, which runs to completion; the
    /// executor applies them on the batch's stream before the forward, so a destination holds the
    /// source's bytes by stream order and the graph needs no extra barrier. A batch that reads a
    /// fresh prefix carries no allocation, and one that reuses a cached prefix cannot be served
    /// correctly without it (`ipc::batch::CacheUnitAllocation`).
    pub cache_units: Vec<ipc::CacheUnitAllocation>,
    /// The batch's slot rows: one `ipc::BlockTable` per request slot and cache group, as the
    /// mechanism's own allocation of pages states them.
    ///
    /// The executor installs these into the cache's pooled table before the replay, because the
    /// gather that builds the attention kernels' block-table column reads **row `slot` of that
    /// table**. The row is the only place a request's pages live for the gather: a batch whose rows
    /// are not installed gathers whatever the slot held before - zeros, or another request's pages -
    /// and the graph then runs clean against the wrong memory. That failure looks like a model
    /// defect rather than a cache one, so it is refused at submit rather than left to the graph.
    pub block_tables: Vec<ipc::BlockTable>,
}

/// The commit table's own definition, so a producer outside this crate writes its rows against
/// the columns and modes the kernels read instead of a copy of them.
pub use ops::commit;

/// The host columns of one batch, in the layout the phase's kernels read.
///
/// The mechanism writes these from the request state it owns; the executor stages them into the
/// pinned arena, uploads them on the batch's stream and builds the device views the model reads.
/// The row layout is `nn::BatchRows`, the model's own host half — one definition, no conversion in
/// the mechanism. The runner pads to the bucket it serves and never reorders what the mechanism
/// wrote; a padded row carries `slot_mapping = -1` and a commit slot below zero, so it writes no
/// cache page and keeps no token state.
///
/// # The three counts
///
/// Every column here follows exactly one of three counts, and a column compared with the wrong one
/// is a mis-sized batch:
///
/// - **tokens** ([`BatchColumns::tokens`]): `input_ids` and the `[tokens]` columns of `rows` —
///   positions, slot mapping;
/// - **rows** ([`BatchColumns::rows`]): the sequences. `slots`, `parameters` and the `[rows]`
///   columns of `rows` — lengths, query offsets, the sampling rows — are one entry per row, and a
///   call may own several: a verification of three candidates is one call with three rows, and a
///   call's `BatchRow::rows` span names them;
/// - **calls** ([`BatchColumns::calls`]): the commit table, one row per call, because a call states
///   once how its step advances the slot's device state however many rows it evaluated.
///
/// A mechanism that fills these columns and an executor that stages them both read this, so the
/// rule is stated once; [`BatchColumns::check`] enforces it before a launch.
#[derive(Clone, Default, Debug)]
pub struct BatchColumns {
    /// The rows and their calls, as the model reads them: positions, slot mapping, lengths, query
    /// offsets, the sampling rows and the calls.
    pub rows: nn::BatchRows,
    /// `[tokens]` the ids the embedding op reads; the runner stages them to the descriptor's id
    /// element type.
    pub input_ids: Vec<i32>,
    /// `[rows]` the sampler's slot view: the runner's per-slot sampling state.
    pub slots: Vec<i32>,
    /// `rows x ops::sampler::PARAMETER_COLUMNS` f32 request parameters, row-major.
    pub parameters: Vec<f32>,
    /// `ops::commit`'s column table, column-major `[columns][rows]` i64; a negative commit slot
    /// belongs to a padded row and keeps its token state.
    pub commit: Vec<i64>,
    /// The KV page size in tokens, the same number the policy sees in `Room`.
    pub page_size: u32,
    /// The attention descriptor's static bound on a sequence's KV length.
    pub max_kv_len: u32,
}

impl BatchColumns {
    /// The number of rows (sequences) the columns describe.
    ///
    /// A row is one sequence's query, which is not the same as a call: a verify batch carries
    /// several rows for one call, and the per-row columns (`slots`, `parameters`) are as long as
    /// the rows. `rows.calls` counts the calls instead, which sized the sampler's views to one row
    /// when a call held three.
    pub fn rows(&self) -> usize {
        self.rows.rows()
    }

    /// The calls the columns describe.
    pub fn calls(&self) -> usize {
        self.rows.calls.len()
    }

    /// The number of query tokens the columns describe.
    pub fn tokens(&self) -> usize {
        self.input_ids.len()
    }

    /// Whether every column has the length its field promises for `shape`, so that a mis-sized
    /// batch fails before a launch rather than after one.
    ///
    /// Three counts, not one: `input_ids` is as long as the batch's tokens, `slots` and
    /// `parameters` as long as its rows — the sequences, which is `[rows]` in every per-row column
    /// and what [`BatchColumns::rows`] answers — and `commit` is one row per *call*, since a call
    /// states once how the step advances its slot however many rows it evaluated. A verify batch
    /// of three candidates is one call and three rows, and comparing the per-row columns with the
    /// call count refuses it.
    pub fn check(&self, shape: &Shape) -> Result<()> {
        let tokens = self.rows.tokens();
        // Two counts, and a column belongs to one of them: `slots`, `parameters`, `seq_lens` and
        // `logits_rows` are per **row** (a verify or branch call carries several rows for one call),
        // while the commit table states one row per **call** (`crates/engine`'s `columns.rs`).
        // Sizing one against the other refuses a valid multi-row call and lets a wrong commit table
        // through.
        let rows = self.rows();
        let calls = self.calls();
        let parameters = ops::sampler::PARAMETER_COLUMNS as usize;
        for (name, actual, expected) in [
            ("positions", self.rows.positions.len(), tokens),
            ("slot_mapping", self.rows.slot_mapping.len(), tokens),
            ("input_ids", self.input_ids.len(), tokens),
            ("seq_lens", self.rows.seq_lens.len(), rows),
            ("logits_rows", self.rows.logits_rows.len(), rows),
            ("slots", self.slots.len(), rows),
            ("parameters", self.parameters.len(), rows * parameters),
            (
                "commit",
                self.commit.len(),
                ops::commit::COMMIT_FIELDS * calls,
            ),
        ] {
            if actual != expected {
                return Err(Error::Invalid(format!(
                    "{name} has {actual} entries, the batch has {rows} rows, {calls} calls and \
                     {tokens} tokens"
                )));
            }
        }
        if tokens > shape.tokens as usize || rows > shape.rows as usize {
            return Err(Error::Invalid(format!(
                "the batch holds {rows} rows and {tokens} tokens, more than its shape's {} and {}",
                shape.rows, shape.tokens
            )));
        }
        if self.page_size == 0 {
            return Err(Error::Invalid("the batch's page size is zero".into()));
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
#[derive(Clone)]
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
}

impl Job {
    /// What this job is doing, for the report of a failed submission.
    ///
    /// The device thread knows the job it was performing and the caller does not, and the failure
    /// crosses a thread boundary as a bare error, so the description is written where the job is
    /// performed and carried beside that error.
    pub fn describe(&self) -> String {
        match self {
            Self::Replay {
                bucket,
                generation,
                batch,
                ..
            } => format!(
                "replaying batch {} from the bucket of {} tokens and {} rows of generation {generation}",
                batch.0, bucket.tokens, bucket.rows
            ),
            Self::Capture { bucket, generation, .. } => format!(
                "capturing the bucket of {} tokens and {} rows of generation {generation}",
                bucket.tokens, bucket.rows
            ),
            Self::Rebind { bucket, generation, .. } => format!(
                "rebinding the bucket of {} tokens and {} rows of generation {generation}",
                bucket.tokens, bucket.rows
            ),
            Self::Shadow {
                operator,
                key,
                bucket,
                generation,
                ..
            } => format!(
                "shadowing {operator} {key} in the bucket of {} tokens and {} rows of generation {generation}",
                bucket.tokens, bucket.rows
            ),
        }
    }

    /// What kind of job this is, for the watchdog's report.
    pub fn kind(&self) -> JobKind {
        match self {
            Self::Replay { .. } => JobKind::Replay,
            Self::Capture { .. } => JobKind::Capture,
            Self::Rebind { .. } => JobKind::Rebind,
            Self::Shadow { .. } => JobKind::Shadow,
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
        }
    }
}

/// What a scheduling mechanism asks of the executor that serves it (4.4, 4.5, 7.3).
///
/// The scheduler thread builds a step's batch and hands it here: `submit` serves it on the device
/// of the batch's phase, and `shadow` runs the same batch's paired forwards against the generation
/// it names without serving it. The main process implements this once over its per-device
/// executors; [`crate::device::Executor`] keeps the same two calls as inherent methods, so a
/// mechanism with a single device needs no router.
pub trait Batches: Send + Sync {
    /// Serves one batch: stage its inputs, choose its bucket, replay its graph.
    fn submit(&self, batch: Batch) -> Result<Submission>;

    /// Runs one shadow batch (7.3 step 2): the paired forwards of `batch.generation`'s changed
    /// operators against the serving generation's kernels, compared by each operator's tolerance.
    ///
    /// A shadow batch never writes resident memory and never serves a request; its failure is the
    /// update's, which the main process reports as `PrecopyFailed`.
    fn shadow(&self, batch: Batch) -> Result<Submission>;

    /// The pools this executor serves (4.2, 7.3): what the mechanism schedules against.
    ///
    /// The mechanism builds its `ExecutorInfo` from this and nothing else, so a pool whose ranks,
    /// cache geometry or model length is absent here is a pool it refuses to serve rather than one
    /// it assumes: a scheduler that guesses a rank's cache makes the host KV path disappear at
    /// runtime. An executor that states none cannot be served in process, and the mechanism says so
    /// when it builds its info.
    fn inventory(&self) -> &[Inventory] {
        &[]
    }
}

/// One pool of the deployment an executor serves (4.2, 7.3).
#[derive(Clone, Debug, Default)]
pub struct Inventory {
    /// The deployment's name for the pool, the identity the mechanism's `WorkerGroupId` carries.
    pub group: String,
    /// The pool's ranks in launch order, each with what it reported at handshake.
    pub ranks: Vec<WorkerInfo>,
    /// What the deployment launched the pool with; the ranks' reports do not carry it back.
    pub placement: Placement,
    /// The longest sequence the pool's model was built for: the attention descriptor's static bound
    /// on the keys one row reads.
    pub max_model_len: u32,
}

/// What a deployment launched one pool with (4.2).
#[derive(Clone, Debug, Default)]
pub struct Placement {
    /// Unresolved physical batches the pool admits at once, counted as at least one.
    pub queue_depth: usize,
    /// Ranks the pool was launched with, counted as at least one.
    pub ranks: usize,
    /// The pool's components by name, each with its member ranks, parallel degrees and distribution.
    pub components: std::collections::BTreeMap<String, ComponentConfig>,
}

/// The mechanism's `Batches` over a rank's devices (2.4, 4.4): a batch goes to the device whose
/// runners serve its phase.
///
/// A rank's executors are its devices, and a device's runners are the blocks of the model placed
/// on it; the mechanism addresses a batch by phase, not by device. This is the one place that
/// translation happens, so a scheduler never needs to know how many devices a rank holds. It is a
/// plain route on purpose: the phase list is fixed at spawn, so no lock, no schedule, no retry.
pub struct Executors {
    executors: Vec<Arc<crate::device::Executor>>,
    /// The pools this rank serves, as the deployment launched them (4.2): the scheduler builds its
    /// executor facts from this and refuses to start on an empty answer, so a rank states its pools
    /// before it is handed over rather than after the first batch finds none.
    inventory: Vec<Inventory>,
}

impl Executors {
    /// The rank's devices, in the order they were spawned.
    ///
    /// Every phase a batch can name must be served by exactly one of them; a rank that spawns two
    /// devices serving one phase is refused, because a batch could go to either and the two would
    /// disagree about the model's state.
    pub fn new(executors: Vec<crate::device::Executor>) -> Result<Self> {
        Self::shared(executors.into_iter().map(Arc::new).collect())
    }

    /// The rank's devices, as the handles the rest of the process already holds.
    ///
    /// A main process holds one handle per device from the moment it spawns them: the control
    /// thread reads their running jobs for the stall rule and their counters for the metrics,
    /// and a scheduler submits through the same handles. Sharing them rather than moving them in
    /// keeps one device one owner, and the phase check is the same one [`Executors::new`] makes.
    pub fn shared(executors: Vec<Arc<crate::device::Executor>>) -> Result<Self> {
        let mut served: Vec<PhaseId> = Vec::new();
        for executor in &executors {
            for phase in executor.phases() {
                if served.contains(phase) {
                    return Err(Error::Invalid(format!(
                        "two devices of this rank serve phase {phase:?}"
                    )));
                }
                served.push(*phase);
            }
        }
        Ok(Self {
            executors,
            inventory: Vec::new(),
        })
    }

    /// States the pools this rank serves, which the scheduler's executor facts are built from.
    ///
    /// The facts are the deployment's: its pool's name, its ranks as they reported at handshake,
    /// what the launch placed there and the model's maximum length. A rank that serves batches must
    /// state them - `Submissions::over_batches` refuses an executor that states no pool - so this is
    /// called where the launch facts are known, before the scheduler is built.
    #[must_use]
    pub fn stating(mut self, inventory: Vec<Inventory>) -> Self {
        self.inventory = inventory;
        self
    }

    /// The device that serves `phase`.
    fn route(&self, phase: PhaseId) -> Result<&Arc<crate::device::Executor>> {
        self.executors
            .iter()
            .find(|executor| executor.serves(phase))
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "no device of this rank serves phase {phase:?}; its {} device(s) serve {:?}",
                    self.executors.len(),
                    self.executors
                        .iter()
                        .flat_map(|executor| executor.phases().iter().copied())
                        .collect::<Vec<_>>()
                ))
            })
    }

    /// How many devices the rank hands batches to.
    pub fn len(&self) -> usize {
        self.executors.len()
    }

    /// Whether the rank has no device at all.
    pub fn is_empty(&self) -> bool {
        self.executors.is_empty()
    }
}

impl Batches for Executors {
    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }

    fn submit(&self, batch: Batch) -> Result<Submission> {
        self.route(batch.phase)?.submit(batch)
    }

    fn shadow(&self, batch: Batch) -> Result<Submission> {
        self.route(batch.phase)?.shadow(batch)
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
    /// A replay of a captured bucket inside a serving command.
    Replay,
    /// A capture of one bucket (warmup or a recapture of the update's generation).
    Capture,
    /// A bucket's rebind - or its recapture, when the walk shows the launch structure changed
    /// (4.5).
    Rebind,
    /// A paired forward of one changed operator (7.3).
    Shadow,
    /// Capturing every bucket of the serving generation before it serves.
    Warmup,
    /// Preparing the next generation (7.3 step 2).
    Precopy,
    /// Switching the serving generation at a batch boundary (7.3 step 3).
    Switchover,
    /// The discard of a prepared generation (7.3 step 4's revert).
    Revert,
    /// Releasing an old generation's graphs (7.3 step 6).
    Purge,
    /// Releasing a prepared generation that never served (an abandoned precopy, 7.3 step 2).
    Discard,
    /// A job that only waits for the events it was given.
    After,
}

impl JobKind {
    /// The kind's name, as a stall report prints it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Replay => "replay",
            Self::Capture => "capture",
            Self::Rebind => "rebind",
            Self::Shadow => "shadow",
            Self::Warmup => "warmup",
            Self::Precopy => "precopy",
            Self::Switchover => "switchover",
            Self::Revert => "revert",
            Self::Purge => "purge",
            Self::Discard => "discard",
            Self::After => "after",
        }
    }
}

/// What the caller of a submission holds until its jobs ended.
///
/// A submission completes once the device executor has performed every job's work and the work
/// completed on the device (the executor thread waits on its own stream's work before reporting a
/// job, and with it the submission). A serving submission completes with the step's
/// [`BatchOutput`] — the tokens its sampling rows sampled; every other submission completes with an
/// empty one. The scheduler thread waits on the submission of a step at the next step's hand-off,
/// never on the device itself.
pub struct Submission {
    id: SubmissionId,
    state: Arc<SubmissionState>,
}

pub(crate) struct SubmissionState {
    outcome: Mutex<Option<Outcome<BatchOutput>>>,
    /// What the submission was doing when it failed, written where the failure happened: the
    /// caller cannot know it, and an error alone does not say which bucket or operator broke.
    failure: Mutex<Option<String>>,
    ready: Condvar,
}

impl Submission {
    /// A submission that has already completed with `outcome`.
    pub fn completed(id: SubmissionId, outcome: Outcome<BatchOutput>) -> Self {
        let state = Arc::new(SubmissionState {
            outcome: Mutex::new(Some(outcome)),
            failure: Mutex::new(None),
            ready: Condvar::new(),
        });
        Self { id, state }
    }

    /// A submission waiting for its jobs' completion, with the state the queue completes.
    pub(crate) fn pending(id: SubmissionId) -> (Self, Arc<SubmissionState>) {
        let state = Arc::new(SubmissionState {
            outcome: Mutex::new(None),
            failure: Mutex::new(None),
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

    /// Waits until the submission completed and answers its outcome.
    pub fn wait(&self) -> Outcome<BatchOutput> {
        let mut outcome = lock(&self.state.outcome);
        while outcome.is_none() {
            outcome = match self.state.ready.wait(outcome) {
                Ok(guard) => guard,
                // A panicking waiter does not make the submission incomplete.
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        outcome.clone().unwrap_or(Outcome::Failed(Arc::new(Error::State(
            "a submission completed without an outcome",
        ))))
    }

    /// Waits at most `timeout` for the submission's completion.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Outcome<BatchOutput>> {
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
    pub fn outcome(&self) -> Option<Outcome<BatchOutput>> {
        lock(&self.state.outcome).clone()
    }

    /// What the submission was doing when it failed, when it did.
    ///
    /// The device thread writes it where the failure happens - the job it was performing and the
    /// bucket, generation or operator that named it - because the caller holds only the bare error.
    pub fn failure_label(&self) -> Option<String> {
        lock(&self.state.failure).clone()
    }
}

impl SubmissionState {
    /// Records what this submission was doing when it failed, for the report that reads it.
    pub(crate) fn note_failure(&self, label: String) {
        let mut slot = lock(&self.failure);
        if slot.is_none() {
            *slot = Some(label);
        }
    }

    /// Completes the submission with `outcome`; the first completion wins.
    pub(crate) fn finish(self: Arc<Self>, outcome: Outcome<BatchOutput>) {
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

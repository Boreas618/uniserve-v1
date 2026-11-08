//! One step's work across the boundary from the scheduling mechanism to the worker.
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
//! - [`Jobs`] is what a runner asks its device worker to issue for a batch, and [`Submission`]
//!   is what the caller holds until every job of the submission ended.
//!
//! A submission is one or more jobs because a program-layer update interleaves work with serving:
//! a batch is a `Replay` of its bucket's graph, a precopy is the `Capture`/`Rebind` jobs of the
//! buckets a changed operator uses, and switchover and purge are submissions of their own so they
//! land on a batch boundary (7.3).

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ipc::{CallStatus, ErrorCode, MainProcessInfo, PhaseId, TimingCounters};

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
/// a token, because the mechanism's completion path is per call. A call this worker refuses never
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
    /// What the batch measured, in the segments this worker can observe.
    ///
    /// `queued_us` is the wait from the mechanism's submission to this device thread starting the
    /// batch, `device_us` the replay's launch to the stream's completion, and `host_us` the rest of
    /// the step on the host. `copy_us` is zero in-process and states a fact rather than a
    /// measurement: nothing is transferred for the completion, since the mechanism reads it here.
    pub timing: TimingCounters,
    /// What this rank's counters, probes and profiling session accumulated since its last result,
    /// which the scheduler adds into its statistics. Empty and absent while nothing observes it.
    pub forward_stats: Option<ipc::ForwardStats>,
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
    /// worker applies them on the batch's stream before the forward, so a destination holds the
    /// source's bytes by stream order and the graph needs no extra barrier. A batch that reads a
    /// fresh prefix carries no allocation, and one that reuses a cached prefix cannot be served
    /// correctly without it (`ipc::batch::CacheUnitAllocation`).
    pub cache_units: Vec<ipc::CacheUnitAllocation>,
    /// The batch's slot rows: one `ipc::BlockTable` per request slot and cache group, as the
    /// mechanism's own allocation of pages states them.
    ///
    /// The worker installs these into the cache's pooled table before the replay, because the
    /// gather that builds the attention kernels' block-table column reads **row `slot` of that
    /// table**. The row is the only place a request's pages live for the gather: a batch whose rows
    /// are not installed gathers whatever the slot held before - zeros, or another request's pages -
    /// and the graph then runs clean against the wrong memory. That failure looks like a model
    /// defect rather than a cache one, so it is refused at submit rather than left to the graph.
    pub block_tables: Vec<ipc::BlockTable>,
    /// Pages of the host KV cache the batch's new units are filled with, on the runner's copy stream
    /// before its forward: a request whose prefix the host held continues from them.
    pub page_loads: Vec<ipc::PageCopy>,
    /// Published pages the batch copies to the host KV cache beside its forward; the batch's output
    /// waits for the copies, and the mechanism commits the pages once it read that output.
    pub page_offloads: Vec<ipc::PageCopy>,
    /// The stored trajectories the batch's latent preparations open at, by request: the pages of
    /// the host latent pools the engine pinned for each resumed request, which the denoiser copies
    /// into its trajectory before the call completes (`ipc::Call::latent_restore`). A preparation
    /// with none opens its trajectory at step zero.
    pub latent_restores: Vec<(ipc::RequestKey, ipc::LatentRestore)>,
    /// The encoder outputs of the batch's requests in the host encoder pool, by request
    /// (`ipc::Call::host_output`): an import reads the output's rows from the pages instead of
    /// computing them, and an export writes the rows it computed to the reserved pages before the
    /// call completes.
    pub host_outputs: Vec<(ipc::RequestKey, ipc::HostOutput)>,
    /// A media computation's columns, and `None` for every causal-LM batch.
    ///
    /// One batch is one computation (2.4): a media batch states its inputs here and carries no token
    /// column a language model reads, and a token batch carries no media inputs. The runner that
    /// receives one knows which class it serves, so the field is read by the media runners alone.
    pub media: Option<MediaColumns>,
}

/// The commit table's own definition, so a producer outside this crate writes its rows against
/// the columns and modes the kernels read instead of a copy of them.
pub use ops::commit;

/// The host columns of one batch, in the layout the phase's kernels read.
///
/// The mechanism writes these from the request state it owns; the worker stages them into the
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
/// A mechanism that fills these columns and a worker that stages them both read this, so the
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

/// The answers to a replay that served `calls`, leaving `calls` empty.
///
/// A media call's result is a product, which its runner published under the request's name, so a
/// call reports that it ran and nothing else.
pub(crate) fn answered(calls: &mut Vec<CallId>) -> Vec<CallOutput> {
    calls
        .drain(..)
        .map(|call| CallOutput::completed(call, Vec::new(), RowState::default()))
        .collect()
}

/// One media computation's columns: the buffers the model prepared, per request (4.5).
///
/// A media computation is not staged from typed token columns. What a denoiser, an encoder or a
/// decoder reads is what the model itself prepared for each request (`Denoiser::prepare_evaluation`
/// and its siblings), so a media batch's columns are data: the worker copies them to device
/// storage sized from the computation's extent and never interprets them, which is what keeps model
/// mathematics out of the worker. The computation travels with them so the columns can state the
/// rules of their own family without a shared struct growing a field only one family reads.
#[derive(Clone, Debug)]
pub struct MediaColumns {
    /// The computation these columns are for, and the extent its kernels are captured for.
    pub computation: crate::plan::Computation,
    /// Per request, in batch order, the buffers the model prepared for it.
    pub inputs: Vec<Vec<Vec<u8>>>,
}

impl MediaColumns {
    /// Whether the columns state the computation they are for (4.5).
    ///
    /// The rules are the family's own: one set of inputs per request, no token column a language
    /// model reads, and - except for an encode, which is sized by rows and tokens - an extent,
    /// because a bucket captured for one canvas serves no other.
    pub fn check(&self) -> Result<()> {
        let computation = self.computation;
        if !computation.stated() {
            return Err(Error::Invalid(format!(
                "a {:?} batch is a computation no variant states",
                computation.call
            )));
        }
        let rows = computation.rows as usize;
        if self.inputs.len() != rows {
            return Err(Error::Invalid(format!(
                "a {:?} batch of {rows} rows carries {} request(s) of media inputs",
                computation.call,
                self.inputs.len()
            )));
        }
        // Three computations are sized by no extent at all: an encode by rows and tokens, and the
        // artifact's two host calls by the request's own products - an audio track's samples come
        // from the decoder's window, and a mux reads what the encodes already wrote. A device
        // computation, by contrast, is captured for one canvas, so a batch that states none cannot
        // say which graph serves it.
        let host = matches!(
            computation.variant(),
            Some(variant)
                if variant.host() || variant == crate::plan::Variant::Encode
        );
        if computation.extent == crate::plan::Extent::None && !host {
            return Err(Error::Invalid(format!(
                "a {:?} batch states no extent: its kernels are captured for one canvas, and a                  batch cannot say which",
                computation.call
            )));
        }
        Ok(())
    }

    /// The columns of a media batch, from the facts the mechanism states per request.
    ///
    /// The mechanism owns the computation and, per request, the request's own stated parameters:
    /// its conditioning as the model family planned it, the diffusion step it runs, and the seed
    /// and canvas its noise is drawn from. It does not own what the model prepares on the device
    /// — the encoded conditioning rows, the latent trajectory — so a staged set carries the first
    /// and never a placeholder for the second.
    pub fn staged(
        computation: crate::plan::Computation,
        requests: &[StagedRequest],
    ) -> MediaColumns {
        MediaColumns {
            computation,
            inputs: requests.iter().map(StagedRequest::encode).collect(),
        }
    }

    /// One request's staged inputs, as [`MediaColumns::staged`] wrote them.
    ///
    /// The rows are this batch's requests in batch order, so a runner reads the row the call it
    /// serves was placed at. A row that does not carry the two buffers this spelling states, or
    /// whose parameter record is the wrong width, is refused by name: a runner that read a
    /// truncated record would run a step index nobody stated.
    pub fn request(&self, index: usize) -> Result<StagedRequest> {
        let Some(row) = self.inputs.get(index) else {
            return Err(Error::Invalid(format!(
                "a {:?} batch carries {} request(s); row {index} is not one of them",
                self.computation.call,
                self.inputs.len()
            )));
        };
        StagedRequest::decode(row)
    }

    /// The bytes of every request's every buffer: what the staging path allocates for.
    pub fn bytes(&self) -> usize {
        self.inputs
            .iter()
            .flat_map(|request| request.iter())
            .map(Vec::len)
            .sum()
    }
}

/// One request's staged media inputs, as the mechanism states them (4.5).
///
/// The record is the mechanism-to-model seam for one request of a media computation, and it holds
/// what the mechanism owns and nothing else: the request's conditioning exactly as the model family
/// planned it, the diffusion step the batch is, the seed and canvas the request states, and the
/// request's identity. The model's own device-side preparation (the encoded conditioning rows, the
/// latent trajectory) is the model's and is not carried here, because a mechanism that filled it
/// would be inventing either a tensor it cannot read or a shape the model never stated.
///
/// # The bytes
///
/// A staged request is two buffers, in this order:
///
/// - **0**, the conditioning blob, byte for byte as `DiffusionRequest::conditioning` holds it;
/// - **1**, sixty bytes little-endian: `step: u32`, `steps: u32`, `seed: u64`, `width: u32`,
///   `height: u32`, `frames: u32`, then the identity - `engine_id: u64`, `request_id: u64`,
///   `request_epoch: u64` - and `handed: u32` and `unit: u32`.
///
/// Little-endian and fixed width so that a worker and an engine of different architectures read one
/// record, and so that a runner reads offsets rather than a parser.
///
/// # Why the identity travels with the row
///
/// An in-process batch numbers its calls by their position (`Batch::calls` is a list of indices), so
/// a row tells a runner *where* a request is and never *which* request it is. A step that must find
/// what the previous step of the same request left behind - a latent trajectory, an encoded
/// conditioning - has nothing to key on without this, and a request whose row moved between two
/// batches would take another request's state. The mechanism owns the key, so the mechanism states
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagedRequest {
    /// The request this row's state belongs to: the only name a step has for what it must find
    /// again, since a batch numbers its calls by position.
    pub request: ipc::RequestKey,
    /// The request's conditioning, as the model family planned it.
    pub conditioning: Vec<u8>,
    /// The zero-based diffusion step this batch is.
    pub step: u32,
    /// The steps the request's schedule runs in total.
    pub steps: u32,
    /// The request's noise seed: the draw is addressed by it and the element's index.
    pub seed: u64,
    /// The output raster's width: the request's canvas.
    pub width: u32,
    /// The output raster's height.
    pub height: u32,
    /// The output frames after the model's alignment.
    pub frames: u32,
    /// The encoded rounds this call hands over, which only a muxing call states: the final muxing
    /// call hands none, and that is what makes it the one that finishes the artifact.
    pub handed: u32,
    /// The first timeline unit a unit-measured call covers: the clip a video decode is. Zero for
    /// every other call.
    pub unit: u32,
}

impl StagedRequest {
    /// The two buffers a staged request is (see the type's byte layout).
    pub fn encode(&self) -> Vec<Vec<u8>> {
        let mut record = Vec::with_capacity(PARAMETER_BYTES);
        record.extend_from_slice(&self.step.to_le_bytes());
        record.extend_from_slice(&self.steps.to_le_bytes());
        record.extend_from_slice(&self.seed.to_le_bytes());
        record.extend_from_slice(&self.width.to_le_bytes());
        record.extend_from_slice(&self.height.to_le_bytes());
        record.extend_from_slice(&self.frames.to_le_bytes());
        record.extend_from_slice(&self.request.engine_id.to_le_bytes());
        record.extend_from_slice(&self.request.request_id.0.to_le_bytes());
        record.extend_from_slice(&self.request.request_epoch.to_le_bytes());
        record.extend_from_slice(&self.handed.to_le_bytes());
        record.extend_from_slice(&self.unit.to_le_bytes());
        vec![self.conditioning.clone(), record]
    }

    /// Reads a staged request's two buffers.
    pub fn decode(row: &[Vec<u8>]) -> Result<Self> {
        let [conditioning, record] = row else {
            return Err(Error::Invalid(format!(
                "a media request's row carries {} buffer(s); a staged request is the conditioning \
                 blob and its parameter record",
                row.len()
            )));
        };
        if record.len() != PARAMETER_BYTES {
            // The refusal names the **shape**, not the width it expects: the width is
            // `PARAMETER_BYTES`, which every column this record gains moves by a field, and a
            // refusal that prints it states a fact about the build rather than about what was
            // handed over. What the reader needs is which record this is and what arrived.
            return Err(Error::Invalid(format!(
                "a media request's parameter record is step, steps, seed, width, height, frames, \
                 the request's three identities, the rounds it hands over and its first unit; this \
                 one is {} bytes, which is not that shape",
                record.len()
            )));
        }
        let word = |at: usize| {
            u32::from_le_bytes([record[at], record[at + 1], record[at + 2], record[at + 3]])
        };
        let word64 = |at: usize| {
            u64::from_le_bytes([
                record[at],
                record[at + 1],
                record[at + 2],
                record[at + 3],
                record[at + 4],
                record[at + 5],
                record[at + 6],
                record[at + 7],
            ])
        };
        Ok(Self {
            request: ipc::RequestKey::new(word64(28), ipc::RequestId(word64(36)), word64(44)),
            conditioning: conditioning.clone(),
            step: word(0),
            steps: word(4),
            seed: word64(8),
            width: word(16),
            height: word(20),
            frames: word(24),
            handed: word(52),
            unit: word(56),
        })
    }
}

/// The width of a staged request's parameter record: `step`, `steps`, `seed`, `width`, `height` and
/// `frames`, then the request's identity (`engine_id`, `request_id`, `request_epoch`), then `handed`
/// and `unit`.
const PARAMETER_BYTES: usize = (4 + 4 + 8 + 4 + 4 + 4) + (8 + 8 + 8) + 4 + 4;

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
    pub fn check(&self, shape: &Shape) -> Result<()> {
        let tokens = self.rows.tokens();
        // Two counts, and a column belongs to one of them: `slots`, `parameters`, `seq_lens` and
        // `logits_rows` are per **row** (a verify or branch call carries several rows for one call),
        // while the commit table states one row per **call** (`crates/scheduler`'s `columns.rs`).
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

/// The work a runner asks its device worker to issue for one submission.
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

/// One job the device worker issues, in submission order.
#[derive(Clone)]
pub enum Job {
    /// Takes this rank's part in the expert step a peer opened, with no tokens of its own.
    Join {
        /// The runner that joins.
        runner: crate::runner::RunnerId,
    },
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
    /// Serves one call on the host: a media call whose work is encoding, muxing or reading, and
    /// whose answer is bytes rather than a device result (2.4's fourth kind of blocking work).
    ///
    /// The job carries no bucket because a host call captures no graph: the runner staged the bytes
    /// this job reads when it submitted the batch, and performs the work on the process's media
    /// lane, so nothing here reaches a device stream. A `CallOutput` from such a job carries the
    /// artifact (`media_output`) and no tokens, which is what the mechanism's completion reads.
    Host {
        /// The runner that serves the call.
        runner: crate::runner::RunnerId,
        /// The call this job answers.
        call: CallId,
    },
    /// Performs one latent preparation on the denoiser that will step the trajectory: the call
    /// states how the denoiser's pages hold the request's samples and, for a resumed request,
    /// copies the stored pages the engine pinned into the trajectory on the denoiser's stream.
    ///
    /// The job carries no bucket because a preparation captures no graph; the runner noted the
    /// request's layout when it submitted the batch.
    Prepare {
        /// The runner that prepares the trajectory.
        runner: crate::runner::RunnerId,
        /// The call this job answers.
        call: CallId,
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
            Self::Join { .. } => "joining an expert step".to_owned(),
            Self::Replay {
                bucket,
                generation,
                batch,
                ..
            } => format!(
                "replaying batch {} from the bucket of {} tokens and {} rows of generation {generation}",
                batch.0, bucket.tokens, bucket.rows
            ),
            Self::Capture {
                bucket, generation, ..
            } => format!("capturing bucket {bucket} of generation {generation}"),
            Self::Rebind {
                bucket, generation, ..
            } => format!("rebinding bucket {bucket} of generation {generation}"),
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
            Self::Host { call, .. } => format!("serving call {} on the host", call.0),
            Self::Prepare { call, .. } => format!("preparing the trajectory of call {}", call.0),
        }
    }

    /// What kind of job this is, for the watchdog's report.
    pub fn kind(&self) -> JobKind {
        match self {
            Self::Join { .. } | Self::Replay { .. } => JobKind::Replay,
            Self::Capture { .. } => JobKind::Capture,
            Self::Rebind { .. } => JobKind::Rebind,
            Self::Shadow { .. } => JobKind::Shadow,
            Self::Host { .. } => JobKind::Host,
            Self::Prepare { .. } => JobKind::Prepare,
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
            Self::Join { runner }
            | Self::Replay { runner, .. }
            | Self::Capture { runner, .. }
            | Self::Rebind { runner, .. }
            | Self::Shadow { runner, .. }
            | Self::Host { runner, .. }
            | Self::Prepare { runner, .. } => Some(*runner),
        }
    }
}

impl std::fmt::Debug for Job {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Join { runner } => formatter
                .debug_struct("Join")
                .field("runner", runner)
                .finish(),
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
            Self::Host { runner, call } => formatter
                .debug_struct("Host")
                .field("runner", runner)
                .field("call", call)
                .finish(),
            Self::Prepare { runner, call } => formatter
                .debug_struct("Prepare")
                .field("runner", runner)
                .field("call", call)
                .finish(),
        }
    }
}

/// What a scheduling mechanism asks of the worker that serves it (4.4, 4.5, 7.3).
///
/// The scheduler thread builds a step's batch and hands it here: `submit` serves it on the device
/// of the batch's phase, and `shadow` runs the same batch's paired forwards against the generation
/// it names without serving it. The main process implements this once over its per-device
/// workers; [`crate::device::Worker`] keeps the same two calls as inherent methods, so a
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

    /// The bucket this worker would serve `computation` in, or `None` when it serves no such
    /// computation (2.4's stated capability).
    ///
    /// The mechanism asks before it forms a media batch: a media call is served only where a
    /// runner's plan states the computation, and the answer is the bucket that plan planned it
    /// into, so the refusal the mechanism makes names a capability the worker does not state
    /// rather than a check that was deleted. A missing phase is its own refusal: a batch cannot be
    /// sent to a device that does not exist.
    fn media(
        &self,
        _phase: ipc::PhaseId,
        _computation: &crate::plan::Computation,
    ) -> Result<Option<crate::plan::Bucket>> {
        Ok(None)
    }

    /// The pools this worker serves (4.2, 7.3): what the mechanism schedules against.
    ///
    /// The mechanism builds its `WorkerInfo` from this and nothing else, so a pool whose ranks,
    /// cache geometry or model length is absent here is a pool it refuses to serve rather than one
    /// it assumes: a scheduler that guesses a rank's cache makes the host KV path disappear at
    /// runtime. A worker that states none cannot be served in process, and the mechanism says so
    /// when it builds its info.
    fn inventory(&self) -> &[Inventory] {
        &[]
    }

    /// Runs the storage operation `name` with the JSON `payload` on every rank, while no batch runs
    /// (7.6), and answers each rank's JSON result.
    ///
    /// A worker with no storage of its own has nothing to change and nothing to report, which
    /// is an empty answer: the mechanism reads it as a cache that is not split into segments.
    fn hook(&self, _name: &str, _payload: &str) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Puts the prepared expert redistribution of `order` in place, while no batch runs (7.6), and
    /// answers each rank's result.
    fn commit_experts(
        &self,
        _order: &std::collections::BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Maps the host KV cache's page pools on this rank's devices, `tokens` tokens of each cache
    /// group, and copies pages through them from now on. Answers each pool's name with its ranks'
    /// pools, in rank order, as the mechanism keys the host KV cache by them. A worker that keeps
    /// no host KV cache maps nothing, and the mechanism plans no host write.
    fn map_host_cache(&self, _tokens: u64) -> Result<Vec<(String, Vec<Vec<ipc::PagePoolInfo>>)>> {
        Ok(Vec::new())
    }

    /// Copies `offloads` to the host KV cache with no batch to carry them: the mechanism's pages
    /// that wait while nothing runs. The submission's output means the copies finished.
    fn copy_pages(&self, _offloads: Vec<ipc::PageCopy>) -> Result<Submission> {
        Err(crate::Error::Invalid(
            "this worker keeps no host KV cache to copy pages to".to_owned(),
        ))
    }

    /// Copies the trajectories of preempted requests to the host latent pools the engine reserved
    /// slots in, with no batch to carry them (`ipc::LatentOffload`): every call of each request
    /// completed, so each copy waits only for the batches that advanced the trajectory. The
    /// submission's output means the copies finished.
    fn offload_latents(&self, _offloads: Vec<ipc::LatentOffload>) -> Result<Submission> {
        Err(crate::Error::Invalid(
            "this worker maps no host latent pool to offload a trajectory to".to_owned(),
        ))
    }
}

/// What a rank's pages are computed from, which keys them in the host KV cache: an epoch that
/// computes the same pages finds what another epoch of the deployment left there.
#[derive(Clone, Debug, Default)]
pub struct HostCacheKeys {
    /// Each cache group's upstream component identities, in group order.
    pub fingerprints: Vec<String>,
    /// Where an engine on another host reaches this rank's Storage Server.
    pub endpoint: Option<String>,
}

/// One pool of the deployment a worker serves (4.2, 7.3).
#[derive(Clone, Debug, Default)]
pub struct Inventory {
    /// The deployment's name for the pool, the identity the mechanism's `WorkerGroupId` carries.
    pub group: String,
    /// The pool's ranks in launch order, each with what it reported at handshake.
    pub ranks: Vec<MainProcessInfo>,
    /// What the deployment launched the pool with; the ranks' reports do not carry it back.
    pub layout: PoolLayout,
    /// The longest sequence the pool's model was built for: the attention descriptor's static bound
    /// on the keys one row reads.
    pub max_model_len: u32,
}

/// What a deployment launched one pool with (4.2).
#[derive(Clone, Debug, Default)]
pub struct PoolLayout {
    /// Unresolved physical batches the pool admits at once, counted as at least one.
    pub queue_depth: usize,
    /// Ranks the pool was launched with, counted as at least one.
    pub ranks: usize,
    /// The pool's components by name, each with its member ranks, parallel degrees and distribution.
    pub components: std::collections::BTreeMap<String, proto::config::BlockConfig>,
}

/// The mechanism's `Batches` over a node's devices (2.4, 4.4): a batch goes to every device whose
/// runners serve its phase.
///
/// A node's workers are its devices, and a device's runners are the blocks of the model placed
/// on it; the mechanism addresses a batch by phase, not by device. This is the one place that
/// translation happens, so a scheduler never needs to know how many devices it drives. A phase one
/// device serves is that device's. A phase several devices serve is a parallel group's: each runs
/// its part of one batch, so each must see it. The scheduler's thread is the one thread that submits,
/// so every device's queue holds the group's batches in one order, which is the order their
/// collectives need. It is a plain route on purpose: the phase list is fixed at spawn, so no lock,
/// no schedule, no retry.
pub struct Workers {
    workers: Vec<Arc<crate::device::Worker>>,
    /// The pools this rank serves, as the deployment launched them (4.2): the scheduler builds its
    /// worker facts from this and refuses to start on an empty answer, so a rank states its pools
    /// before it is handed over rather than after the first batch finds none.
    inventory: Vec<Inventory>,
    /// What keys this rank's pages in the host KV cache, when the deployment keeps one.
    host_cache: Option<HostCacheKeys>,
    /// What the storage operations of 7.6 reach, on a rank that keeps storage at a Storage Server.
    data: Option<Arc<crate::data_plane::DataPlane>>,
}

impl Workers {
    /// The node's devices, in the order they were spawned.
    pub fn new(workers: Vec<crate::device::Worker>) -> Result<Self> {
        Self::shared(workers.into_iter().map(Arc::new).collect())
    }

    /// The rank's devices, as the handles the rest of the process already holds.
    ///
    /// A main process holds one handle per device from the moment it spawns them: the control
    /// thread reads their running jobs for the stall rule and their counters for the metrics,
    /// and a scheduler submits through the same handles. Sharing them rather than moving them in
    /// keeps one device one owner.
    pub fn shared(workers: Vec<Arc<crate::device::Worker>>) -> Result<Self> {
        Ok(Self {
            workers,
            inventory: Vec::new(),
            host_cache: None,
            data: None,
        })
    }

    /// States that this rank keeps a host KV cache, with what keys its pages.
    #[must_use]
    pub fn keeping_host_cache(mut self, keys: HostCacheKeys) -> Self {
        self.host_cache = Some(keys);
        self
    }

    /// States what the storage operations of 7.6 reach on this rank.
    #[must_use]
    pub fn with_data_plane(mut self, data: Arc<crate::data_plane::DataPlane>) -> Self {
        self.data = Some(data);
        self
    }

    /// States the pools this rank serves, which the scheduler's worker facts are built from.
    ///
    /// The facts are the deployment's: its pool's name, its ranks as they reported at handshake,
    /// what the launch placed there and the model's maximum length. A rank that serves batches must
    /// state them - `Submissions::over_batches` refuses a worker that states no pool - so this is
    /// called where the launch facts are known, before the scheduler is built.
    #[must_use]
    pub fn stating(mut self, inventory: Vec<Inventory>) -> Self {
        self.inventory = inventory;
        self
    }

    /// The device that serves `phase`.
    fn route(&self, phase: PhaseId) -> Result<&Arc<crate::device::Worker>> {
        self.workers
            .iter()
            .find(|worker| worker.serves(phase))
            .ok_or_else(|| {
                route_refusal(
                    phase,
                    &self
                        .workers
                        .iter()
                        .flat_map(|worker| worker.phases().iter().copied())
                        .collect::<Vec<_>>(),
                )
            })
    }

    /// The one device a unit-measured call runs on, when `batch` is one.
    ///
    /// A video decode is one unit of the timeline and needs no other unit, so the devices that serve
    /// the phase take the units in turn instead of each decoding the same one: a device's clip is
    /// independent of its neighbours' but for the tail the decoder hands across. A video encode
    /// follows its unit, so the device that decoded a clip encodes it from its own column.
    fn device_of_unit(&self, batch: &Batch) -> Result<Option<&Arc<crate::device::Worker>>> {
        let Some(media) = &batch.media else {
            return Ok(None);
        };
        if !matches!(
            media.computation.call,
            ipc::MediaCall::VideoDecoding | ipc::MediaCall::VideoEncoding
        ) {
            return Ok(None);
        }
        let serving: Vec<&Arc<crate::device::Worker>> = self
            .workers
            .iter()
            .filter(|worker| worker.serves(batch.phase))
            .collect();
        if serving.is_empty() {
            return Err(route_refusal(batch.phase, &[]));
        }
        let unit = media.request(0)?.unit as usize;
        Ok(Some(serving[unit % serving.len()]))
    }

    /// Hands `batch` to every device that serves its phase, as one submission.
    fn on_every_device(
        &self,
        batch: Batch,
        submit: impl Fn(&crate::device::Worker, Batch) -> Result<Submission>,
    ) -> Result<Submission> {
        let first = self.route(batch.phase)?;
        let mut peers = Vec::new();
        for peer in self
            .workers
            .iter()
            .filter(|worker| worker.serves(batch.phase) && !Arc::ptr_eq(worker, first))
        {
            peers.push(submit(peer, batch.clone())?);
        }
        Ok(submit(first, batch)?.joined(peers))
    }

    /// How many devices the rank hands batches to.
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Whether the rank has no device at all.
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }
}

/// The refusal of a batch whose phase no device of this rank serves.
///
/// A phase number is a plan's, and the plan that numbers the batches and the plan that splits the
/// model must agree on it; when they do not, the batch reaches this rank and nothing here can serve
/// it. Both numbers are named - the phase asked for and the phases this rank's devices serve -
/// because the disagreement is between two plans and one figure alone cannot show it. The answer is
/// `Invalid`: the batch is not servable here, which is a refusal rather than a broken device. What a
/// reader of the number does with that distinction is the caller's - the in-process submission path
/// currently wraps every worker error alike, so an `Invalid` and a broken context read the same
/// there.
fn route_refusal(phase: PhaseId, served: &[PhaseId]) -> Error {
    Error::Invalid(format!(
        "no device of this rank serves phase {}; its devices serve {:?}",
        phase.0,
        served.iter().map(|served| served.0).collect::<Vec<_>>()
    ))
}

impl Batches for Workers {
    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }

    fn media(
        &self,
        phase: ipc::PhaseId,
        computation: &crate::plan::Computation,
    ) -> Result<Option<crate::plan::Bucket>> {
        Ok(self.route(phase)?.media(computation))
    }

    fn submit(&self, batch: Batch) -> Result<Submission> {
        match self.device_of_unit(&batch)? {
            Some(worker) => worker.submit(batch),
            None => self.on_every_device(batch, |worker, mut batch| {
                // A page copy names the rank that makes it, and a rank is a device here.
                if let Some(rank) = self
                    .workers
                    .iter()
                    .position(|held| std::ptr::eq(held.as_ref(), worker))
                {
                    batch.page_loads.retain(|copy| copy.rank as usize == rank);
                    batch
                        .page_offloads
                        .retain(|copy| copy.rank as usize == rank);
                }
                worker.submit(batch)
            }),
        }
    }

    fn shadow(&self, batch: Batch) -> Result<Submission> {
        self.on_every_device(batch, |worker, batch| worker.shadow(batch))
    }

    fn map_host_cache(&self, tokens: u64) -> Result<Vec<(String, Vec<Vec<ipc::PagePoolInfo>>)>> {
        let Some(keys) = &self.host_cache else {
            return Ok(Vec::new());
        };
        let ranks = self
            .workers
            .iter()
            .map(|worker| worker.map_host_cache(tokens, keys))
            .collect::<Result<Vec<_>>>()?;
        Ok(self
            .inventory
            .iter()
            .map(|pool| (pool.group.clone(), ranks.clone()))
            .collect())
    }

    fn copy_pages(&self, offloads: Vec<ipc::PageCopy>) -> Result<Submission> {
        let mut submissions = self
            .workers
            .iter()
            .enumerate()
            .map(|(rank, worker)| {
                let own = offloads
                    .iter()
                    .filter(|copy| copy.rank as usize == rank)
                    .cloned()
                    .collect();
                worker.copy_pages(own)
            })
            .collect::<Result<Vec<_>>>()?;
        let first = submissions
            .pop()
            .ok_or_else(|| crate::Error::Invalid("this rank holds no device".to_owned()))?;
        Ok(first.joined(submissions))
    }

    fn offload_latents(&self, offloads: Vec<ipc::LatentOffload>) -> Result<Submission> {
        // A page copy names the rank that makes it, and a rank is a device here.
        let mut submissions = self
            .workers
            .iter()
            .enumerate()
            .map(|(rank, worker)| {
                let own = offloads
                    .iter()
                    .map(|offload| ipc::LatentOffload {
                        request_key: offload.request_key,
                        pages: offload
                            .pages
                            .iter()
                            .filter(|copy| copy.rank as usize == rank)
                            .cloned()
                            .collect(),
                    })
                    .filter(|offload| !offload.pages.is_empty())
                    .collect();
                worker.offload_latents(own)
            })
            .collect::<Result<Vec<_>>>()?;
        let first = submissions
            .pop()
            .ok_or_else(|| crate::Error::Invalid("this rank holds no device".to_owned()))?;
        Ok(first.joined(submissions))
    }

    fn hook(&self, name: &str, payload: &str) -> Result<Vec<String>> {
        match &self.data {
            Some(data) => data.operate(name, payload),
            None => Ok(Vec::new()),
        }
    }

    fn commit_experts(
        &self,
        order: &std::collections::BTreeMap<String, Vec<u32>>,
    ) -> Result<Vec<String>> {
        match &self.data {
            Some(data) => data.commit_experts(order),
            None => Ok(Vec::new()),
        }
    }
}

/// What the device worker is doing, as the control thread's stall report and probe name it (2.4).
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
    /// Serving one media call on the host: encoding, muxing, or the reading of media (2.4).
    Host,
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
    /// Giving the resident cache segments back, for a standby's park (2.5).
    Sleep,
    /// Mapping the host KV cache's page pools, or copying pages to them with no batch to ride on.
    HostCache,
    /// A latent preparation: stating a trajectory's pages, and restoring a stored one into them.
    Prepare,
    /// Copying preempted requests' trajectories to the host latent pool.
    LatentOffload,
    /// Mapping the released cache segments again, with the unit resets that follow (2.5).
    Wake,
    /// A change of the rank's profiling session.
    Profile,
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
            Self::Host => "host",
            Self::Warmup => "warmup",
            Self::Precopy => "precopy",
            Self::Switchover => "switchover",
            Self::Revert => "revert",
            Self::Purge => "purge",
            Self::Discard => "discard",
            Self::Sleep => "sleep",
            Self::HostCache => "host cache",
            Self::Prepare => "prepare",
            Self::LatentOffload => "latent offload",
            Self::Wake => "wake",
            Self::Profile => "profile",
            Self::After => "after",
        }
    }
}

/// What the caller of a submission holds until its jobs ended.
///
/// A submission completes once the device worker has performed every job's work and the work
/// completed on the device (the worker thread waits on its own stream's work before reporting a
/// job, and with it the submission). A serving submission completes with the step's
/// [`BatchOutput`] — the tokens its sampling rows sampled; every other submission completes with an
/// empty one. The scheduler thread waits on the submission of a step at the next step's hand-off,
/// never on the device itself.
pub struct Submission {
    id: SubmissionId,
    state: Arc<SubmissionState>,
    /// The same batch on the other ranks that serve it: the submission completes when they all did.
    peers: Vec<Submission>,
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
        Self {
            id,
            state,
            peers: Vec::new(),
        }
    }

    /// This submission and the same batch's on `peers`, as one.
    ///
    /// The ranks of a group run one batch together, so it has ended when every one of them ended:
    /// this rank's outcome is the answer, and a peer's failure replaces it, because a rank that
    /// failed leaves the others' output without its part.
    #[must_use]
    pub fn joined(mut self, peers: Vec<Submission>) -> Self {
        self.peers = peers;
        self
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
                peers: Vec::new(),
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
        let own = self.wait_own();
        let peers: Vec<Outcome<BatchOutput>> = self.peers.iter().map(Self::wait).collect();
        Self::joined_outcome(own, peers)
    }

    /// This rank's outcome unless a peer's says the batch did not complete.
    fn joined_outcome(
        own: Outcome<BatchOutput>,
        peers: Vec<Outcome<BatchOutput>>,
    ) -> Outcome<BatchOutput> {
        match own {
            Outcome::Success(_) => peers
                .into_iter()
                .find(|peer| !matches!(peer, Outcome::Success(_)))
                .unwrap_or(own),
            failed => failed,
        }
    }

    /// Waits until this rank's part completed and answers its outcome.
    fn wait_own(&self) -> Outcome<BatchOutput> {
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
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Outcome<BatchOutput>> {
        let deadline = Instant::now() + timeout;
        let own = self.wait_own_until(deadline)?;
        let mut peers = Vec::with_capacity(self.peers.len());
        for peer in &self.peers {
            peers.push(peer.wait_timeout(deadline.saturating_duration_since(Instant::now()))?);
        }
        Some(Self::joined_outcome(own, peers))
    }

    /// Waits until `deadline` for this rank's part.
    fn wait_own_until(&self, deadline: Instant) -> Option<Outcome<BatchOutput>> {
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
        let own = lock(&self.state.outcome).clone()?;
        let mut peers = Vec::with_capacity(self.peers.len());
        for peer in &self.peers {
            peers.push(peer.outcome()?);
        }
        Some(Self::joined_outcome(own, peers))
    }

    /// What the submission was doing when it failed, when it did.
    ///
    /// The device thread writes it where the failure happens - the job it was performing and the
    /// bucket, generation or operator that named it - because the caller holds only the bare error.
    pub fn failure_label(&self) -> Option<String> {
        lock(&self.state.failure)
            .clone()
            .or_else(|| self.peers.iter().find_map(Self::failure_label))
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

#[cfg(test)]
mod joined_tests {
    use super::*;

    fn done(id: u64) -> Submission {
        Submission::completed(SubmissionId(id), Outcome::Success(BatchOutput::default()))
    }

    /// A batch a group runs together has ended when every rank's part ended, and the rank that
    /// failed decides the answer: the others' output lacks its part.
    #[test]
    fn a_joined_submission_waits_for_every_rank_and_answers_a_peers_failure() {
        let (pending, state) = Submission::pending(SubmissionId(2));
        let joined = done(1).joined(vec![pending]);
        assert!(joined.outcome().is_none(), "a peer has not ended");
        state.finish(Outcome::Failed(Arc::new(Error::State("a peer failed"))));
        assert!(matches!(joined.outcome(), Some(Outcome::Failed(_))));
        assert!(matches!(joined.wait(), Outcome::Failed(_)));

        let joined = done(3).joined(vec![done(4), done(5)]);
        assert!(matches!(joined.wait(), Outcome::Success(_)));
    }
}

#[cfg(test)]
mod media_tests {
    use super::*;
    use crate::plan::{Computation, Extent, Variant};
    use nn::capability::{LatentShape, Layout};

    fn layout() -> Layout {
        Layout {
            video: LatentShape {
                channels: 16,
                frames: 13,
                height: 60,
                width: 34,
            },
            audio: None,
            text_rows: 256,
            condition_rows: 0,
            condition_elements: 0,
        }
    }

    /// A media batch's columns are the model's prepared buffers, one set per request, and the
    /// family's rules are its own: a computation no variant states, a request count the batch does
    /// not carry, and a denoising step without its canvas are each refused before a launch - while
    /// an encode, which is sized by rows and tokens, states no extent and is accepted.
    #[test]
    fn a_media_batch_states_one_set_of_inputs_per_request_and_its_own_extent() {
        let denoise = MediaColumns {
            computation: Computation {
                call: ipc::MediaCall::Denoising,
                rows: 1,
                tokens: 0,
                extent: Extent::Latents(layout()),
            },
            inputs: vec![vec![vec![0u8; 64]]],
        };
        assert_eq!(denoise.bytes(), 64);
        denoise.check().expect("one request, its canvas stated");

        // A mux reads what the encodes already wrote, so no canvas sizes it: a **host** call
        // states `Extent::None` and is accepted, which is what separates it from the device
        // computation whose extent is asserted below. The case that used to sit here made muxing
        // the example of a call no variant states, and that stopped being true when it gained one.
        let muxed = MediaColumns {
            computation: Computation {
                call: ipc::MediaCall::Muxing,
                rows: 1,
                tokens: 0,
                extent: Extent::None,
            },
            inputs: vec![Vec::new()],
        };
        muxed
            .check()
            .expect("a host call is sized by the request's own products, not a canvas");
        assert_eq!(muxed.bytes(), 0, "a mux stages no extent of its own");

        let wrong_rows = MediaColumns {
            inputs: vec![Vec::new(), Vec::new()],
            ..denoise.clone()
        };
        let refusal = wrong_rows
            .check()
            .expect_err("two request sets for a one-row batch");
        assert!(
            refusal.to_string().contains("2 request(s)"),
            "the refusal names the count: {refusal}"
        );

        let no_canvas = MediaColumns {
            computation: Computation {
                call: ipc::MediaCall::Denoising,
                rows: 1,
                tokens: 0,
                extent: Extent::None,
            },
            ..denoise.clone()
        };
        let refusal = no_canvas
            .check()
            .expect_err("a denoising step states its canvas");
        assert!(
            refusal.to_string().contains("states no extent"),
            "the refusal names the missing extent: {refusal}"
        );

        let encode = MediaColumns {
            computation: Computation {
                call: ipc::MediaCall::TextEncoding,
                rows: 1,
                tokens: 300,
                extent: Extent::None,
            },
            inputs: vec![vec![vec![1u8; 8]]],
        };
        assert_eq!(encode.computation.variant(), Some(Variant::Encode));
        encode
            .check()
            .expect("an encode is sized by rows and tokens");
    }

    /// A staged request is the mechanism's record of one request, and it round-trips through the
    /// two buffers it is: the conditioning verbatim and the parameters little-endian at fixed
    /// offsets, so a runner reads offsets rather than a parser. A row of the wrong shape is
    /// refused by name - a truncated record would be a step index nobody stated.
    #[test]
    fn a_staged_request_is_the_conditioning_blob_and_its_parameter_record() {
        let staged = StagedRequest {
            unit: 0,
            request: ipc::RequestKey::new(7, ipc::RequestId(41), 3),
            conditioning: vec![9, 8, 7],
            step: 3,
            steps: 12,
            seed: 0x0102_0304_0506_0708,
            width: 768,
            height: 512,
            frames: 121,
            handed: 0,
        };
        let row = staged.encode();
        assert_eq!(row.len(), 2, "a staged request is two buffers");
        assert_eq!(
            row[0],
            vec![9, 8, 7],
            "the conditioning is carried verbatim"
        );
        assert_eq!(row[1].len(), PARAMETER_BYTES);
        assert_eq!(&row[1][0..4], &3u32.to_le_bytes(), "step");
        assert_eq!(&row[1][4..8], &12u32.to_le_bytes(), "steps");
        assert_eq!(
            &row[1][8..16],
            &0x0102_0304_0506_0708u64.to_le_bytes(),
            "seed"
        );
        let decoded = StagedRequest::decode(&row).expect("the record round-trips");
        assert_eq!(decoded, staged);
        assert_eq!(
            decoded.request,
            ipc::RequestKey::new(7, ipc::RequestId(41), 3),
            "the identity is the row's name for what a step must find again"
        );

        let columns = MediaColumns::staged(
            Computation {
                call: ipc::MediaCall::Denoising,
                rows: 1,
                tokens: 0,
                extent: Extent::Latents(layout()),
            },
            std::slice::from_ref(&staged),
        );
        assert_eq!(columns.request(0).expect("one row"), staged);
        let out_of_range = columns.request(1).expect_err("one row, asked for two");
        assert!(
            format!("{out_of_range}").contains("row 1 is not one of them"),
            "{out_of_range}"
        );
    }

    /// A staged request's two buffers are both load-bearing, and the refusals name which one is
    /// wrong: a document's worth of buffers, or a record of another width, is not this record.
    #[test]
    fn a_staged_request_of_the_wrong_shape_is_refused_by_name() {
        let short = vec![vec![0u8; 3], vec![0u8; PARAMETER_BYTES - 1]];
        let error = StagedRequest::decode(&short).expect_err("a truncated record");
        let text = format!("{error}");
        assert!(
            text.contains("step, steps, seed, width, height, frames"),
            "the refusal does not name the record's shape: {text}"
        );
        assert!(
            text.contains(&format!("{} bytes", PARAMETER_BYTES - 1)),
            "the refusal does not name the record it was given: {text}"
        );

        let one = vec![vec![1u8; 3]];
        let error = StagedRequest::decode(&one).expect_err("one buffer is not the record");
        assert!(
            format!("{error}").contains("carries 1 buffer(s)"),
            "{error}"
        );
    }

    /// A batch whose phase no device of this rank serves is refused where it is routed, and the
    /// refusal names the phase the plan asked for and the phases this rank's devices serve: the
    /// disagreement is between two plans, and one figure alone cannot show it. It is a refusal and
    /// not a device failure - the batch is not servable here, which is not the same as a broken
    /// device - and it is stated as such, because a reader that can tell the two apart can end one
    /// batch instead of the epoch.
    #[test]
    fn a_phase_no_device_serves_is_refused_by_both_numbers() {
        let refusal = route_refusal(ipc::PhaseId(3), &[ipc::PhaseId(0), ipc::PhaseId(1)]);
        assert!(
            matches!(refusal, Error::Invalid(_)),
            "a batch nothing here can serve was refused as a device failure: {refusal:?}"
        );
        let text = refusal.to_string();
        assert!(
            text.contains("phase 3") && text.contains("[0, 1]"),
            "the refusal names neither the phase asked for nor the ones served: {text}"
        );
    }
}

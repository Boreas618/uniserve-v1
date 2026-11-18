//! The batch the scheduler hands to its executor, and what the batch produced.
//!
//! A [`Batch`] is one phase of one step on one worker group: the requests it advances, one call
//! each, the host columns a token phase's runner copies into its input tensors ([`Columns`]), the
//! rows of a media phase ([`MediaRow`]), and the requests the group starts ([`NewSlot`]). It is
//! plain data: no device address, block table, copy list or free list. A row names a request's
//! memory by positions in the request's tables at the buffer pool (a token's cache position, a
//! recurrent-state checkpoint's position in the state table, an encoder output's position in the
//! encoder-output table, a media product's position in the product table), and the worker
//! computes every address from its device block tables, which only the buffer pool's completions
//! feed.
//!
//! A [`BatchOutput`] answers a batch with one [`CallOutput`] per request, in the batch's order.
//!
//! Everything here is host data: no device, no kernel, no model. The commit table's and the
//! sampler's columns are the module vocabulary's (`proto::module::{commit, sampler}`), and a
//! denoising step's latent layout is the video vocabulary's (`proto::video::Layout`).
//!
//! [`Capacity`] is what a worker group's workers serve, stated once they are ready, with the cache
//! groups it names ([`cache`]) and a media group's pipeline, whose products a request's phases
//! write and read (`nn::pipeline`); [`philox`] holds the counter-based draws a sampled row's
//! random value comes from. A media read's row carries its request's condition media and a
//! muxing call's output its generated artifact ([`MediaOutput`]), each a sealed memory file the
//! process shares.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use proto::video::Layout;
pub use scheduling::Phase;
use serde::{Deserialize, Serialize};
use storage::runtime::GroupId;

use proto::MappedMedia;
use proto::report::ForwardStats;
use proto::request::{RequestId, TokenLogprob};

use self::philox::DrawAddress;

/// Result type for semantic batch validation.
pub(crate) type ValidationResult<T> = std::result::Result<T, ValidationError>;

/// Describes a batch or a batch's output that violates the rules spanning its parts.
#[derive(Debug, thiserror::Error)]
#[error("invalid worker message: {0}")]
pub struct ValidationError(String);

impl ValidationError {
    /// Constructs a validation error from contextual text.
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// What the message violates, without the protocol's prefix.
    pub fn reason(&self) -> &str {
        &self.0
    }
}

// The validation macros below are textually scoped: they are visible only after their
// definitions, in this file and in the modules declared after them.
macro_rules! invalid_message {
    ($($arg:tt)*) => {
        ValidationError::message(format!($($arg)*))
    };
}

macro_rules! bail_invalid {
    ($($arg:tt)*) => {
        return Err(invalid_message!($($arg)*))
    };
}

macro_rules! ensure_valid {
    ($condition:expr, $($arg:tt)*) => {
        if !$condition {
            bail_invalid!($($arg)*);
        }
    };
}

pub mod cache;
pub mod capacity;
pub mod philox;

pub use capacity::Capacity;

/// The scheduler's name for one batch, unique within an epoch.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct BatchId(pub u64);

/// The kernel and policy generations a scheduler stamps on the batches it forms
/// ([`Batch::kernels`], [`Batch::policy`]): one value per engine, shared by its schedulers and
/// set by the generation update at a batch boundary of every scheduler.
#[derive(Debug, Default)]
pub struct Generations {
    kernels: AtomicU64,
    policy: AtomicU64,
}

impl Generations {
    /// Stamps of `kernels` and `policy`.
    pub fn new(kernels: u64, policy: u64) -> Self {
        Self {
            kernels: AtomicU64::new(kernels),
            policy: AtomicU64::new(policy),
        }
    }

    /// The kernel generation the next batch is stamped with.
    pub fn kernels(&self) -> u64 {
        self.kernels.load(Ordering::Acquire)
    }

    /// The policy generation the next batch is stamped with.
    pub fn policy(&self) -> u64 {
        self.policy.load(Ordering::Acquire)
    }

    /// Stamps the batches formed from here on with kernel generation `kernels`.
    pub fn set_kernels(&self, kernels: u64) {
        self.kernels.store(kernels, Ordering::Release);
    }

    /// Stamps the batches formed from here on with policy generation `policy`.
    pub fn set_policy(&self, policy: u64) {
        self.policy.store(policy, Ordering::Release);
    }
}

/// One phase of one step on one worker group, as the scheduler assembled it.
///
/// `shape` is the scheduler's statement of the batch's size, the rows and tokens it wrote, and
/// the worker serves the batch from the smallest captured bucket that holds it; a shape no
/// bucket holds is an error, never an eager fallback.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    /// The batch's number, which its output and a stall report name it by.
    pub id: BatchId,
    /// The worker group whose ranks run the batch.
    pub group: GroupId,
    /// What every call of the batch computes.
    pub phase: Phase,
    /// The newest kernel generation whose graphs the batch may replay. Each bucket serves from
    /// the newest generation its graphs were switched to on the group's ranks, at most this one:
    /// the scheduler stamps the number of a generation whose buckets are being switched from the
    /// boundary its switch begins at, and the kept one again from the boundary a rollback takes
    /// effect at, so no batch formed after a rollback replays a graph of the rolled-back
    /// generation.
    pub kernels: u64,
    /// The generation of the scheduling program the batch was formed under. It changes at one
    /// batch boundary of every scheduler and involves no graph.
    pub policy: u64,
    /// The configuration revision of the step values the batch was assembled under: the
    /// snapshot its step began with (`proto::config::Tuning`), the same on every rank.
    pub revision: u64,
    /// The rows, tokens, variant and extent the scheduler wrote: what selects the bucket.
    pub shape: Shape,
    /// The requests the batch advances, one call each, in row order: the order of
    /// `columns.rows.calls` in a token phase and of `media` in a media phase.
    pub requests: Vec<RequestId>,
    /// A token phase's host columns, which its runner copies into its input tensors; for a text
    /// encoding, the prompt columns it encodes; empty for every other media phase.
    pub columns: Columns,
    /// A media phase's rows, one per request; empty for a token phase.
    pub media: Vec<MediaRow>,
    /// The requests this group has not served before this batch, strictly increasing by slot:
    /// the worker starts each one's sampling state on its slot before the batch samples.
    pub new_slots: Vec<NewSlot>,
}

/// What one batch produced.
///
/// The scheduler builds the next step's input ids from the tokens and folds the timing into the
/// statistics of the batch's phase. A batch that advances no request, an expert union member's
/// join, completes with no call and nothing measured.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BatchOutput {
    /// One per entry of [`Batch::requests`], in its order: an output's position names its request.
    pub calls: Vec<CallOutput>,
    /// What the batch measured, in the segments the worker can observe.
    ///
    /// `queued_us` is the wait from the submission to the device thread starting the batch,
    /// `device_us` the replay's launch to the stream's completion, and `host_us` the rest of the
    /// step on the host. `copy_us` is zero in-process: nothing is transferred for the output.
    pub timing: TimingCounters,
    /// What the rank's counters, probes and profiling session accumulated since its last output,
    /// which the scheduler adds into its statistics. Absent while nothing observes it.
    pub forward_stats: Option<ForwardStats>,
}

/// One call's output: what execution did for one request of a batch.
///
/// A batch advances one call per request and each gets one output, whether or not it sampled a
/// token, so the scheduler applies outputs per request. A batch the worker refuses produces no
/// output at all, so a [`CallStatus::Error`] output is always a failure of execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallOutput {
    /// Whether execution performed the call.
    pub status: CallStatus,
    /// Why it failed, when it did: the class the scheduler acts on (`NumericalFault` fails the
    /// rows that computed non-finite logits, `InputError` rejects the request with this rank's
    /// message, `ResourceError` fails the call).
    pub error_code: Option<ErrorCode>,
    /// What this rank said about a refused input, when it refused one.
    pub error_message: Option<String>,
    /// The tokens the sampler accepted for this call, in the order the call's sampling rows took
    /// them. A call whose rows sampled nothing (a prefill chunk that ends no prompt) has none,
    /// which is not the same as a failure.
    pub tokens: Vec<u32>,
    /// The slot's commit count after the call, read from the commit table's sampling position:
    /// a text slot advances it once per committed token, a latent slot once per denoising step.
    pub step_index: u32,
    /// The cache extent execution initialized, rejected speculative positions included: the slot's
    /// logical position after the commit.
    pub cache_computed_len: u32,
    /// The accepted prefix a successor may read: the slot's sampling position after the commit.
    pub cache_visible_len: u32,
    /// The scores of each accepted token, aligned with [`CallOutput::tokens`], on the raw
    /// distribution: `log_softmax` of the model's logits before penalties, temperature and
    /// filtering (vLLM's default `raw_logprobs`).
    ///
    /// A token's entries are the token itself (its log-probability and its one-based competition
    /// rank), then its position's most likely tokens, the largest first, as many as the
    /// deployment's `max_logprobs`; the token may be among them. A causal-LM step reports them for
    /// every token it sampled, whether or not the request asked, because a captured graph cannot
    /// branch on a request parameter; the scheduler keeps the candidates each request asks for and
    /// drops the rest. A call that sampled nothing reports none.
    pub token_logprobs: Vec<Vec<TokenLogprob>>,
    /// The scores of each prompt position the call scored, in input order, in the form of
    /// [`CallOutput::token_logprobs`]: the prompt token itself, then the position's most likely
    /// tokens.
    pub prompt_logprobs: Vec<Vec<TokenLogprob>>,
    /// The finished artifact, when the call produced one.
    pub media_output: Option<MediaOutput>,
    /// The drafter's proposals after the last committed token.
    pub draft_token_ids: Vec<u32>,
}

/// Terminal status of one call's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallStatus {
    /// Execution completed successfully.
    Ok,
    /// Execution failed, with the class [`CallOutput::error_code`] states.
    Error,
}

/// The class of a rank's failure: the one vocabulary of a failed call's
/// completion and of an error response.
///
/// The rank classifies every failure it reports with these classes, and the
/// engine decides by them: an invariant violation is a control-plane defect
/// that stops the engine, an input error is a rejected request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The rank does not serve the call's kind.
    UnsupportedCall,
    /// A message, a call or a launch value breaks the protocol.
    InvalidDescriptor,
    /// The deployment asks for a rank layout, a device or a kernel the rank
    /// cannot provide.
    UnsupportedSetup,
    /// A request's inputs are invalid for its computation.
    InputError,
    /// Device or numerical computation failed.
    ComputeError,
    /// The computation produced non-finite values (NaN or infinity) where
    /// its numerics admit none, as among a forward's logits: a fault of the
    /// computation, never of the request.
    NumericalFault,
    /// A reserved or physical resource could not satisfy the work.
    ResourceError,
    /// An invariant the rank relies on is broken.
    InvariantViolation,
    /// The rank failed in a way that leaves it unable to serve.
    FatalWorkerFailure,
}

impl ErrorCode {
    /// The class's name as the protocol spells it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedCall => "unsupported_call",
            Self::InvalidDescriptor => "invalid_descriptor",
            Self::UnsupportedSetup => "unsupported_setup",
            Self::InputError => "input_error",
            Self::ComputeError => "compute_error",
            Self::NumericalFault => "numerical_fault",
            Self::ResourceError => "resource_error",
            Self::InvariantViolation => "invariant_violation",
            Self::FatalWorkerFailure => "fatal_worker_failure",
        }
    }

    /// Whether a failure of this class leaves the rank unsafe to serve
    /// further work.
    pub const fn fatal(self) -> bool {
        matches!(self, Self::InvariantViolation | Self::FatalWorkerFailure)
    }
}

impl fmt::Display for ErrorCode {
    /// Formats the class as its protocol name.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Per-call timing counters. Accounting only; never state identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TimingCounters {
    /// Time spent waiting for worker execution, in microseconds.
    pub queued_us: u64,
    /// Time spent executing device work, in microseconds.
    pub device_us: u64,
    /// Time spent transferring completion data, in microseconds.
    pub copy_us: u64,
    /// Time spent processing the completion on the host, in microseconds.
    pub host_us: u64,
}

/// One materialized media artifact: a sealed anonymous memory file the media host wrote and
/// mapped (`proto::MappedMedia`), which the scheduler hands to the daemon, which maps the same file
/// through its descriptor. Within the process the artifact is shared, never copied; the peer link
/// carries its bytes, which the receiving process writes into a memory file of its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaOutput(#[serde(with = "medium_bytes")] pub Arc<MappedMedia>);

/// The serde form of a medium (`proto::MappedMedia`): its bytes as one byte string, which the peer
/// link's MessagePack carries as binary, read back into a sealed memory file of the receiver's.
mod medium_bytes {
    use std::sync::Arc;

    use proto::MappedMedia;
    use proto::request::ArtifactMedia as _;
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        medium: &Arc<MappedMedia>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(medium.bytes())
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<MappedMedia>, D::Error> {
        let bytes = serde_bytes::ByteBuf::deserialize(deserializer)?;
        MappedMedia::copy(&bytes)
            .map(Arc::new)
            .map_err(D::Error::custom)
    }
}

/// The serde form of a row's condition media ([`MediaRow::media`]): each medium as one byte
/// string, as [`medium_bytes`] states it.
mod media_bytes {
    use std::sync::Arc;

    use proto::MappedMedia;
    use proto::request::ArtifactMedia as _;
    use serde::de::Error as _;
    use serde::ser::SerializeSeq as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        media: &[Arc<MappedMedia>],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(media.len()))?;
        for medium in media {
            sequence.serialize_element(serde_bytes::Bytes::new(medium.bytes()))?;
        }
        sequence.end()
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<Arc<MappedMedia>>, D::Error> {
        let media: Vec<serde_bytes::ByteBuf> = Deserialize::deserialize(deserializer)?;
        media
            .iter()
            .map(|bytes| MappedMedia::copy(bytes).map(Arc::new))
            .collect::<std::io::Result<_>>()
            .map_err(D::Error::custom)
    }
}

/// The serde form of a media row's image: the encoded file as one byte string, which the peer
/// link's MessagePack carries as binary rather than as a sequence of integers.
mod image_bytes {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        image: &Option<Arc<[u8]>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match image {
            Some(bytes) => serializer.serialize_some(serde_bytes::Bytes::new(bytes)),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Arc<[u8]>>, D::Error> {
        let image: Option<serde_bytes::ByteBuf> = Deserialize::deserialize(deserializer)?;
        Ok(image.map(|bytes| Arc::from(bytes.into_vec())))
    }
}

/// The host columns of one token batch, in the layout the phase's kernels read.
///
/// The scheduler writes these from the request state it owns; the runner stages them into its
/// pinned arena, uploads them on the batch's stream and builds the device views the model reads.
/// The runner pads to the bucket it serves and never reorders what the scheduler wrote; a padded
/// row carries a cache position of `-1` and a commit slot below zero, so it writes no cache entry
/// and keeps no token state.
///
/// # The three counts
///
/// Every column here follows exactly one of three counts, and a column compared with the wrong one
/// is a mis-sized batch:
///
/// - **tokens** ([`Columns::tokens`]): `input_ids` and the `[tokens]` columns of `rows`:
///   positions and cache positions;
/// - **rows** ([`Columns::rows`]): the sequences. `slots`, `parameters`, `draws` and the `[rows]`
///   columns of `rows` (lengths, the sampling rows) are one entry per row, and a call may own
///   several: a verification of three candidates is one call with three rows, and the call's
///   [`Row::rows`] span names them;
/// - **calls** ([`Columns::calls`]): the commit table, one row per call, because a call states
///   once how its step advances the slot's device state however many rows it evaluated.
///
/// The scheduler that fills these columns and the runner that stages them both read this, so the
/// rule is stated once; [`Batch::check`] enforces it before a launch.
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct Columns {
    /// The rows and their calls, as the model reads them: positions, cache positions, lengths,
    /// query offsets, the sampling rows and the calls.
    pub rows: Rows,
    /// `[tokens]` the ids the embedding op reads; the runner stages them to the descriptor's id
    /// element type.
    pub input_ids: Vec<i32>,
    /// `[rows]` the sampler's slot view: the runner's per-slot sampling state.
    pub slots: Vec<i32>,
    /// `rows x proto::module::sampler::PARAMETER_COLUMNS` f32 request parameters, row-major.
    pub parameters: Vec<f32>,
    /// `[rows]` where each row's draw lies: its request's Philox stream and the logical position
    /// of the token it samples ([`philox`]). The runner turns each into the row's uniform
    /// before the replay, so a sampled token depends on its request's seed and position alone. A
    /// greedy row and a row that samples nothing carry one too; its draw is never read.
    pub draws: Vec<DrawAddress>,
    /// `slot_commit` (`proto::module::commit`)'s column table, column-major `[columns][calls]`
    /// i64; a negative commit slot belongs to a padded row and keeps its token state.
    pub commit: Vec<i64>,
    /// The KV page size in tokens, the same number the policy sees in `Room`.
    pub page_size: u32,
    /// The attention descriptor's static bound on a sequence's KV length.
    pub max_kv_len: u32,
    /// `[rows]` the absolute token count after which each row's recurrent state is kept for the
    /// prefix cache, zero for a row that keeps none; empty for a model that keeps no recurrent
    /// state. The model's recurrent layers write that state into the row's slot of the checkpoint
    /// banks, and the runner moves it into the block a [`StateCopy`] names.
    pub checkpoints: Vec<i32>,
    /// The recurrent-state copies between the request slots' banks and their state tables' blocks
    /// this batch needs: every restore before the forward, every checkpoint after it.
    pub state_copies: Vec<StateCopy>,
    /// One per call, in call order: what the call does for its request's draft cache, which a
    /// runner that speculates with its model's drafter reads; empty for a batch whose calls draft
    /// nothing.
    pub drafts: Vec<DraftWork>,
    /// The image blocks the batch's prefill rows write, in token order: each block's token rows
    /// carry no id the embedding reads, and the runner writes the encoder output's rows there.
    pub vision: Vec<VisionRows>,
    /// `[rows]` whether each row adds its slot's bias row ([`NewSlot::bias`] and
    /// [`NewSlot::allowed`]) to its logits before penalties, temperature and filtering; empty when
    /// no row does.
    pub biased: Vec<bool>,
    /// `[rows]` the tokens each row may not sample, strictly increasing, whose logits are `-inf`:
    /// the end-of-sequence tokens under a `min_tokens` floor and the last token of every bad word
    /// whose prefix the generated tokens end with. They move with a row's history, so they are
    /// stated per batch; empty when no row bans any.
    pub banned: Vec<Vec<u32>>,
    /// `[tokens]` the prompt token each token's logits score, `-1` for a token that scores none;
    /// empty for a batch that scores no prompt token. A call that scores its prompt names, for
    /// each of its tokens, the prompt token after it: the next input token, and for its last the
    /// first prompt token after the call, so a prompt is scored at every position but its first
    /// however it is chunked (vLLM's alignment). A batch that names one is served by
    /// [`Variant::PrefillScored`].
    pub targets: Vec<i32>,
}

/// The row columns of one token batch: plain host vectors in the layout of the device views a
/// model reads (`nn::TokenBatch`).
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rows {
    /// The calls of the batch, one per request, in the order of [`Batch::requests`].
    pub calls: Vec<Row>,
    /// `[tokens]` rotary positions. An image block's rows share one rotary position, so this is
    /// not a token's place in its request's cache.
    pub positions: Vec<i32>,
    /// `[tokens]` each token's cache position in its request: the logical position its key and
    /// value are written at, `-1` where its row writes nothing. A position, never a unit: the
    /// runner finds the unit through its device block table of the row's cache group.
    pub cache_positions: Vec<i32>,
    /// `[rows]` keys visible to each row's last query.
    pub seq_lens: Vec<i32>,
    /// `[rows + 1]` packed query offsets.
    pub query_offsets: Vec<i32>,
    /// `[rows]` hidden rows whose logits are wanted.
    pub logits_rows: Vec<i32>,
}

impl Rows {
    /// The number of packed query tokens.
    pub fn tokens(&self) -> usize {
        self.positions.len()
    }

    /// The number of rows (sequences) of the batch.
    pub fn rows(&self) -> usize {
        self.seq_lens.len()
    }
}

/// One call of a token batch: which request slot, which phase, which rows it occupies.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Row {
    /// The request slot the call runs on, in the batch's worker group.
    pub slot: u32,
    /// What the call computes: a prefill (or a refill) chunk, a decode or a verification.
    pub kind: Phase,
    /// The attention cache group the call's keys and values belong to.
    pub cache_group: u32,
    /// The half-open row range the call owns in the batch's device columns.
    pub rows: std::ops::Range<u32>,
}

/// The kind of forward a captured graph serves: the bucket key the worker selects by.
///
/// A bucket only ever holds a forward of its own variant: a decode row cannot be served by a
/// prefill graph, and a prefill that projects logits has different outputs from one that only
/// writes the KV cache.
///
/// # The token variants
///
/// | variant | writes | sampler |
/// --- | --- | ---
/// | [`Variant::PrefillWrite`] | the KV cache | runs over every row, and every row's draw is discarded (`logits_rows` are all `-1`) |
/// | [`Variant::PrefillOutput`] | the KV cache **and** every sampling row's logits | runs over every row of the bucket; a row that ends a prompt keeps its token |
/// | [`Variant::PrefillGreedy`] | the same, for a batch whose prompt endings are all greedy | runs, greedily |
/// | [`Variant::PrefillScored`] | the same, and the scores of every prompt token the batch's `targets` name | runs over every row, one draw per row |
/// | [`Variant::DecodeGreedy`] | one token per row | runs, greedily |
/// | [`Variant::DecodeSampled`] | one token per row | runs, one draw per row |
///
/// A prefill bucket carries the sampler because the logits are computed anyway, and because it is
/// what makes a first token possible without a decode step spent only on choosing it. The draw is
/// the expensive tail: its candidate selection visits every row of the bucket for milliseconds,
/// which a batch of greedy prompt endings and a batch that ends no prompt both have no use for. So
/// those two take the greedy tail, and only a batch with a drawing row pays for the draw.
///
/// **A greedy row in a drawing bucket is served by the draw:** the runner stages it with a top-k
/// width of one, so its candidate set is its largest shaped logit and its draw selects it, the
/// token the greedy launch sequence would choose (ties to the lowest id on both paths). A batch may
/// therefore mix greedy and drawn rows, prompt endings or decodes, and is served as a drawing one.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum Variant {
    /// Decode rows drawn from the slot state, sampled greedily.
    DecodeGreedy,
    /// Decode rows drawn from the slot state, sampled with the bucket's sampler graph.
    DecodeSampled,
    /// Prefill rows of host tokens writing the KV cache, keeping no hidden state.
    PrefillWrite,
    /// Prefill rows of host tokens that also project each row's last hidden state.
    PrefillOutput,
    /// A [`Self::PrefillOutput`] whose rows that end a prompt are all greedy: each row's first
    /// token is the argmax of its shaped logits, and nothing is drawn.
    PrefillGreedy,
    /// A [`Self::PrefillOutput`] that also scores prompt tokens (`prompt_logprobs`): every token
    /// whose target the batch names ([`Columns::targets`]) has its logits projected and its
    /// target's log-probability, rank and candidates reported on the raw distribution.
    ///
    /// A variant of its own because the scoring projects every token of the bucket rather than
    /// one per row, which a prefill that scores nothing does not pay for: a batch none of whose
    /// requests asks for prompt logprobs is served by the other prefill variants, with their
    /// launches. Its prompt endings are drawn, a greedy one as a draw over its largest candidate.
    PrefillScored,
    /// One encode of prompt rows: the text model's and the vision tower's outputs, kept as the
    /// conditioning the denoiser reads.
    Encode,
    /// One denoising step of one latent layout: the diffusion model's velocity, applied to the
    /// latents in place (a request's steps are one batch each, never a loop inside a batch).
    Denoise,
    /// One clip decoded from latents into a raster.
    Decode,
    /// One audio track decoded from latents into interleaved PCM.
    ///
    /// A variant of its own because the graph is: a clip's decode is a stack of three-dimensional
    /// convolutions over a voxel grid, and a track's is the vocoder's upsampling and dilated
    /// one-dimensional convolutions. One bucket cannot hold both, and a capture is keyed by the
    /// graph a bucket runs.
    AudioDecode,
    /// One video unit encoded on the host: the postprocessor's `[frames, rows, columns, 3]` raster
    /// into the encoded unit the assembly appends.
    VideoEncode,
    /// One request's audio track encoded on the host: the decoder's interleaved PCM into the track
    /// the assembly carries.
    AudioEncode,
    /// One request's encoded units and track assembled into its MP4, on the host.
    Mux,
    /// One request's trajectory opened. The denoiser draws the request's noise at its first step,
    /// so this call computes nothing: it is the scheduler's step between a prompt's encoding and
    /// the first denoising call.
    LatentPrepare,
    /// One request's condition media read on the host: its bytes decoded and laid out as the
    /// model's encoders read them, in storage a later call of the request reads.
    MediaRead,
    /// An encoder's call over an image: a chat request's image encoded by its vision or latent
    /// encoder ([`Phase::VisionEncode`], [`Phase::LatentEncode`]) into the encoder output its
    /// prefill reads. A video request has no such call: its text encoding runs the vision tower
    /// over its conditions' vision blocks, under this key and eagerly, and its latent preparation
    /// encodes its condition latents.
    ConditionEncode,
    /// A rank's part in an expert step it has no tokens for: every expert layer, and nothing else.
    ExpertJoin,
    /// One verification per row: a committed token followed by its drafts, each token evaluated
    /// as the decode step at its position and sampled greedily; the row commits the drafts its
    /// own samples agree with and the sample after them. A bucket of `rows` verifications holds
    /// `rows` times one more than the deployment's drafts tokens.
    VerifyGreedy,
    /// A [`Self::VerifyGreedy`] whose tokens are drawn, a greedy row as a draw over its one
    /// largest candidate.
    VerifySampled,
    /// The drafter's forward behind a prefill or a verification of the same rows and tokens:
    /// its entries over the positions whose next token the step before it committed, then the
    /// drafts each row proposes, chained on the device. No batch states it: the worker issues
    /// it behind the batch it follows.
    Draft,
}

/// The stage a forward is measured under: the variants of one computation together, as a
/// profiling session's `stages` select steps and the fence probes' histograms group them.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Stage {
    Decode,
    Verify,
    Draft,
    Prefill,
    TextEncoding,
    Denoising,
    VideoDecoding,
    AudioDecoding,
    VideoEncoding,
    AudioEncoding,
    Muxing,
    LatentPreparation,
    MediaReading,
    ConditionEncoding,
    ExpertJoin,
}

impl Stage {
    /// The stage's name, as a profiling session's request selects it and its file records it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Verify => "verify",
            Self::Draft => "draft",
            Self::Prefill => "prefill",
            Self::TextEncoding => "text_encoding",
            Self::Denoising => "denoising",
            Self::VideoDecoding => "video_decoding",
            Self::AudioDecoding => "audio_decoding",
            Self::VideoEncoding => "video_encoding",
            Self::AudioEncoding => "audio_encoding",
            Self::Muxing => "muxing",
            Self::LatentPreparation => "latent_preparation",
            Self::MediaReading => "media_reading",
            Self::ConditionEncoding => "condition_encoding",
            Self::ExpertJoin => "expert_join",
        }
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Variant {
    /// The stage this variant is measured under.
    pub const fn stage(self) -> Stage {
        match self {
            Self::DecodeGreedy | Self::DecodeSampled => Stage::Decode,
            Self::VerifyGreedy | Self::VerifySampled => Stage::Verify,
            Self::Draft => Stage::Draft,
            Self::PrefillWrite
            | Self::PrefillOutput
            | Self::PrefillGreedy
            | Self::PrefillScored => Stage::Prefill,
            Self::Encode => Stage::TextEncoding,
            Self::Denoise => Stage::Denoising,
            Self::Decode => Stage::VideoDecoding,
            Self::AudioDecode => Stage::AudioDecoding,
            Self::VideoEncode => Stage::VideoEncoding,
            Self::AudioEncode => Stage::AudioEncoding,
            Self::Mux => Stage::Muxing,
            Self::LatentPrepare => Stage::LatentPreparation,
            Self::MediaRead => Stage::MediaReading,
            Self::ConditionEncode => Stage::ConditionEncoding,
            Self::ExpertJoin => Stage::ExpertJoin,
        }
    }

    /// The variant's name, as a capture report prints it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DecodeGreedy => "decode_greedy",
            Self::DecodeSampled => "decode_sampled",
            Self::PrefillWrite => "prefill_write",
            Self::PrefillOutput => "prefill_output",
            Self::PrefillGreedy => "prefill_greedy",
            Self::PrefillScored => "prefill_scored",
            Self::Encode => "encode",
            Self::Denoise => "denoise",
            Self::Decode => "decode",
            Self::AudioDecode => "audio_decode",
            Self::VideoEncode => "video_encode",
            Self::AudioEncode => "audio_encode",
            Self::Mux => "mux",
            Self::LatentPrepare => "latent_prepare",
            Self::MediaRead => "media_read",
            Self::ConditionEncode => "condition_encode",
            Self::ExpertJoin => "expert_join",
            Self::VerifyGreedy => "verify_greedy",
            Self::VerifySampled => "verify_sampled",
            Self::Draft => "draft",
        }
    }

    /// Whether this variant is a media computation rather than a causal-LM forward: a batch of
    /// it carries [`MediaRow`]s, and its bucket an [`Extent`].
    ///
    /// The two families never share a bucket: a causal-LM runner refuses a media variant by name
    /// rather than serving it from a graph shaped for another computation.
    pub const fn media(self) -> bool {
        matches!(
            self,
            Self::Encode
                | Self::Denoise
                | Self::Decode
                | Self::AudioDecode
                | Self::VideoEncode
                | Self::AudioEncode
                | Self::Mux
                | Self::LatentPrepare
                | Self::MediaRead
                | Self::ConditionEncode
        )
    }

    /// Whether this variant is performed on the host rather than on a device.
    ///
    /// These calls capture no graph: their work is reading, encoding and muxing, which the
    /// process's media thread pool does. A bucket of one states which device serves the call and
    /// never an extent, so no resolution narrows it.
    pub const fn host(self) -> bool {
        matches!(
            self,
            Self::VideoEncode
                | Self::AudioEncode
                | Self::Mux
                | Self::MediaRead
                | Self::ConditionEncode
        )
    }

    /// Whether a batch of this host variant is the work of every process that holds ranks of its
    /// group, each doing the part of it whose blocks its own ranks hold: a video encode round
    /// reads each of its units in the run of the rank that decoded it (the decode rounds' deal,
    /// `runner::decoder::RoundConfig`), and the process that holds that rank encodes the unit
    /// into the unit's own encoded block there. A batch of any other host variant reads blocks
    /// one process holds, and runs once (`crate::executor::Route`).
    pub const fn dealt(self) -> bool {
        matches!(self, Self::VideoEncode)
    }

    /// Whether a bucket of this variant picks each row's token by taking the largest candidate
    /// rather than by drawing one.
    ///
    /// A greedy request states temperature zero, whose shaping divides by one, so the row's largest
    /// candidate *is* its token. A bucket of greedy rows takes the argmax launch sequence; one with
    /// a drawn row draws, and serves its greedy rows as a draw over their one largest candidate. A
    /// prefill that ends no prompt keeps no token, so it takes the cheaper of the two.
    pub const fn greedy(self) -> bool {
        matches!(
            self,
            Self::DecodeGreedy | Self::PrefillGreedy | Self::PrefillWrite | Self::VerifyGreedy
        )
    }

    /// Whether a bucket of this variant runs the sampler.
    ///
    /// Every bucket the causal-LM runner serves does, which is every variant this predicate is
    /// asked about: a media bucket is refused by name before a graph is chosen for it, so it never
    /// reaches this question. A prefill's sampler is a tail over logits the forward computes
    /// anyway; the scheduler says which rows end a prompt by putting them in `logits_rows`, and a
    /// row that ends none has its draw discarded.
    pub const fn samples(self) -> bool {
        true
    }

    /// Whether a bucket of this variant scores the prompt tokens its batch's targets name.
    pub const fn scores(self) -> bool {
        matches!(self, Self::PrefillScored)
    }

    /// Whether a forward of this variant draws its inputs from the slot state one token per row.
    pub const fn is_decode(self) -> bool {
        matches!(self, Self::DecodeGreedy | Self::DecodeSampled)
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// One forward's size: what it asks a captured graph to hold.
///
/// `tokens` is the number of tokens the forward writes to the KV cache and `rows` the number of
/// sequences it evaluates: a decode evaluates one token per row, so its `tokens == rows`, and a
/// prefill a chunk of host tokens, which may be more than its rows. A media computation is
/// captured for one extent as well, so its shape states it; a token forward's is
/// [`Extent::None`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct Shape {
    pub rows: u32,
    pub tokens: u32,
    pub variant: Variant,
    pub extent: Extent,
}

impl Shape {
    /// A decode forward of `rows` rows.
    pub const fn decode(rows: u32, variant: Variant) -> Self {
        Self {
            rows,
            tokens: rows,
            variant,
            extent: Extent::None,
        }
    }

    /// A prefill or encode forward of `rows` rows holding `tokens` tokens.
    pub const fn prefill(rows: u32, tokens: u32, variant: Variant) -> Self {
        Self {
            rows,
            tokens,
            variant,
            extent: Extent::None,
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}/{} rows/{} tokens",
            self.variant, self.rows, self.tokens
        )
    }
}

/// What a media bucket's computation is sized by, where a causal-LM bucket is sized by tokens
/// alone.
///
/// The language model's four variants leave it `None`, so their buckets are `(variant, rows,
/// tokens)` exactly as before and no causal-LM path reads this field. A media bucket states the
/// shape its graph is captured for: a denoising step is one latent layout, a decode is one clip.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum Extent {
    /// No media extent: a causal-LM bucket, sized by its tokens.
    None,
    /// One denoising step's latent layout (`Layout`).
    Latents(Layout),
    /// What a request asks for, which the mechanism states to find its bucket: the output frames and
    /// the resolution. A layout's frame count is the request's own, because the video's extent sets the
    /// sparse selection budget and the audio timeline, so a longer bucket is not a padded one.
    Request {
        /// Output frames after the model's alignment.
        frames: u32,
        /// Raster columns of a frame.
        width: u32,
        /// Raster rows of a frame.
        height: u32,
        /// Prompt rows the layout must hold.
        text_rows: u32,
        /// Condition rows the layout must hold, as the denoiser packs them. A layout's condition
        /// rows are real rows of every step, so a request rides the smallest layout that holds its
        /// own, and a request without any the layout that holds none.
        condition_rows: u32,
    },
    /// One decoded clip's raster.
    Clip {
        /// Frames in the clip.
        frames: u32,
        /// Raster rows of a frame.
        height: u32,
        /// Raster columns of a frame.
        width: u32,
    },
}

/// A request this group has not served before the batch: its slot's sampling state starts here.
///
/// A request does not always start on a group with nothing generated: a request given from its
/// prefill group to a decode group generated its first token on the prefill group, and a migrated
/// or preempted request generated its tokens before it was admitted again. Its penalties count
/// those tokens all the same, so the slot's counts start at the request's history rather than at
/// zero. Its logit bias and allowed set are fixed for its lifetime, so the slot's row of the
/// worker's bias table, one `[vocabulary]` f32 row per slot, is written once here and every row
/// of the request flagged in [`Columns::biased`] reads it.
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewSlot {
    /// The request slot whose sampling state starts.
    pub slot: u32,
    /// The tokens the request generated before this batch, which its penalties count, one count
    /// per occurrence: empty for a request that generated none, or whose penalties count nothing.
    pub history: Vec<u32>,
    /// `(token, bias)` pairs added to the logits of the request's biased rows, strictly
    /// increasing in token, each bias finite; a token outside the vocabulary is ignored, as no
    /// logit carries it.
    pub bias: Vec<(u32, f32)>,
    /// The tokens the request may sample, strictly increasing, when it restricts them: every other
    /// token's bias row entry is `-inf`, whatever its bias.
    pub allowed: Option<Vec<u32>>,
    /// The token ids before the request's first token in this batch, the oldest first, which a
    /// model whose layers read token ids reads as the history of that token: the last
    /// [`Capacity::token_history`](crate::batch::Capacity::token_history) of them, fewer when
    /// the request holds fewer, and none for a model that reads no token ids. The runner stands
    /// the model's fill id in for each position before the request's first token
    /// (`nn::family::TokenHistory::fill`).
    pub context: Vec<u32>,
}

/// One copy of a request slot's recurrent state between its banks and a block of its state table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateCopy {
    /// The request slot whose banks the copy reads or writes.
    pub slot: u32,
    /// The checkpoint's position in the request's state table, whose block the runner finds
    /// through its device block table.
    pub position: u32,
    /// Which way the state moves.
    pub direction: StateDirection,
}

/// Which way a [`StateCopy`] moves a slot's recurrent state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateDirection {
    /// The page's checkpoint into the slot's live banks, before the forward: a row that resumes
    /// from a cached prefix continues from the state its prefix's last token left.
    Restore,
    /// The slot's checkpoint banks into the page, after the forward: the row's state at the page's
    /// last token, which a later request reusing the prefix resumes from.
    Keep,
}

/// One image block of a prefill batch: `rows` consecutive token rows from `token`, holding the
/// rows of an encoder output, which rotate from logical position `start`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionRows {
    /// The batch's token index of the block's first row.
    pub token: u32,
    /// The block's rows: the encoder output's rows, one cache position each.
    pub rows: u32,
    /// The logical position the block's first row rotates from: the position after the context
    /// before it.
    pub start: u32,
    /// The encoder output's position in the encoder-output table of the request whose row holds
    /// the block.
    pub feature: u32,
}

/// What a token call does for its request's draft cache, in a deployment
/// that speculates with its model's drafter.
///
/// The draft cache entry at position `i` is the drafter's evaluation of the
/// target's hidden state at `i` with the token at `i + 1`. A forward call
/// writes the entries of its positions whose next token it knows: the
/// tokens it evaluates, its own sampled token, or `lookahead`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DraftWork {
    /// Draft tokens the call proposes after its last committed token, which
    /// the request's next verification may verify; at most the deployment's
    /// draft limit, and zero for none.
    pub proposals: u32,
    /// The token after the call's last input token, which the draft cache
    /// entry of its last position reads; absent when the call samples that
    /// token itself or knows none.
    pub lookahead: Option<u32>,
    /// The first position of a call that evaluates tokens its request's
    /// cache already holds, its input tokens from there on, to fill the
    /// request's draft cache: it starts from the state checkpoint at that
    /// position (or from the beginning), writes no target K/V and leaves
    /// its request slot's state as it was. Absent for every other call.
    pub refill: Option<u32>,
}

/// One request's row of a media batch: what the scheduler owns of the request and its call.
///
/// The row holds the request's conditioning exactly as the model family prepared it, the
/// diffusion step and timeline unit the call is, the seed and resolution the request states, and
/// where its products lie as positions in the request's product table. What a model prepares on
/// the device (the encoded conditioning rows, the latent trajectory) is the model's, in blocks the
/// worker finds through its tables by slot and position.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaRow {
    /// The request the row advances.
    pub request: RequestId,
    /// The request slot the call runs on, in the batch's worker group.
    pub slot: u32,
    /// The request's conditioning, as the model family prepared it.
    pub conditioning: Vec<u8>,
    /// The fetched media of the request's conditions, in condition order, on a media read's row
    /// alone (empty on every other row): what the reader decodes. The peer link carries their
    /// bytes.
    #[serde(with = "media_bytes")]
    pub media: Vec<Arc<MappedMedia>>,
    /// The zero-based diffusion step this call is.
    pub step: u32,
    /// The steps the request's schedule runs in total.
    pub steps: u32,
    /// The request's noise seed: the draw is addressed by it and the element's index.
    pub seed: u64,
    /// The output raster's width: the request's resolution.
    pub width: u32,
    /// The output raster's height.
    pub height: u32,
    /// The output frames after the model's alignment.
    pub frames: u32,
    /// The encoded rounds this call hands over, which only a muxing call states: the final muxing
    /// call hands none, and that is what makes it the one that finishes the artifact.
    pub rounds: u32,
    /// The first timeline unit a unit-measured call covers: the first clip of the round a video
    /// decode or its encode is. Zero for every other call.
    pub unit: u32,
    /// The rank of the batch's worker group that runs this row when the call is that one rank's
    /// alone. `None` when every rank of the group runs its share: a video decode round, whose
    /// units are dealt to the ranks of the decoder group, in order in an even round and in reverse
    /// in an odd one, each decoding its own, and the encode of a round, which each process runs
    /// over the units its ranks decoded ([`Variant::dealt`]).
    pub rank: Option<u32>,
    /// A chat image's encoded file, for the vision or latent encoding of a token request's
    /// image; `None` for every other row.
    #[serde(with = "image_bytes")]
    pub image: Option<Arc<[u8]>>,
    /// The declared results the call reads, then those it writes, as positions in the request's
    /// product table, which holds one block per declared result in the order the scheduler asked
    /// for them at admission. A video decode round after the first reads the splice tail its
    /// previous round wrote and writes its own, one declared result per unit.
    pub results: Vec<u32>,
    /// How many leading entries of `results` the call reads; the entries after them it writes.
    pub reads: u32,
}

impl Batch {
    /// Checks the rules that span the batch's parts, so a batch that crosses a process boundary is
    /// refused before any of it is staged:
    ///
    /// - the batch names each request once;
    /// - a token batch carries no media row, one call row per request on distinct slots, and its
    ///   columns follow the three counts of [`Columns`] and fit its shape;
    /// - its state copies run on slots of its calls, restores before checkpoints and one restore
    ///   per slot, and one checkpoint per row that keeps its state;
    /// - a media batch carries one row per request, no more rows than its shape, an extent unless
    ///   it is an encode or a host call, a read count within each row's results, and one rank for
    ///   all its rows or none;
    /// - the new slots are strictly increasing, and each bias row is canonical.
    ///
    /// # Errors
    ///
    /// [`ValidationError`] naming the first rule the batch breaks.
    pub fn check(&self) -> ValidationResult<()> {
        let mut requests = HashSet::with_capacity(self.requests.len());
        ensure_valid!(
            self.requests
                .iter()
                .all(|request| requests.insert(*request)),
            "a batch repeats a request"
        );

        if self.shape.variant.media() {
            self.check_media()?;
        } else {
            ensure_valid!(
                self.media.is_empty(),
                "a {} batch carries media rows",
                self.shape.variant
            );
            ensure_valid!(
                self.columns.calls() == self.requests.len(),
                "a token batch states {} call rows for {} requests",
                self.columns.calls(),
                self.requests.len()
            );
            let mut slots = HashSet::with_capacity(self.requests.len());
            ensure_valid!(
                self.columns
                    .rows
                    .calls
                    .iter()
                    .all(|call| slots.insert(call.slot)),
                "a batch carries two calls on one slot"
            );
            self.columns.check(&self.shape)?;
            self.columns.check_state_copies(&slots)?;
        }

        self.check_new_slots()
    }

    /// The rules of a media batch: one row per request within the shape, the extent its graph is
    /// captured for, each row's read count within its results, and one rank or none.
    fn check_media(&self) -> ValidationResult<()> {
        let variant = self.shape.variant;
        ensure_valid!(
            self.media.len() == self.requests.len(),
            "a {variant} batch of {} requests carries {} media rows",
            self.requests.len(),
            self.media.len()
        );
        ensure_valid!(
            self.media.len() <= self.shape.rows as usize,
            "a {variant} batch carries {} media rows, more than its shape's {}",
            self.media.len(),
            self.shape.rows
        );

        // An encode is sized by rows and tokens, and a host call by the request's own products; a
        // device computation is captured for one resolution, so a batch that states none cannot
        // say which graph serves it.
        ensure_valid!(
            self.shape.extent != Extent::None || variant.host() || variant == Variant::Encode,
            "a {variant} batch states no extent: its kernels are captured for one resolution, \
             and a batch cannot say which"
        );

        ensure_valid!(
            self.media
                .iter()
                .all(|row| row.reads as usize <= row.results.len()),
            "a media row reads more results than it names"
        );
        ensure_valid!(
            self.media
                .windows(2)
                .all(|pair| pair[0].rank == pair[1].rank),
            "a {variant} batch's rows name different ranks to run on"
        );
        Ok(())
    }

    /// The rule of the new slots: strictly increasing by slot, each bias row canonical.
    fn check_new_slots(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.new_slots
                .windows(2)
                .all(|pair| pair[0].slot < pair[1].slot),
            "the batch's new slots are not strictly increasing"
        );
        for slot in &self.new_slots {
            let biased: Vec<u32> = slot.bias.iter().map(|(token, _)| *token).collect();
            ensure_valid!(
                increasing(&biased)
                    && slot.allowed.as_deref().is_none_or(increasing)
                    && slot.bias.iter().all(|(_, bias)| bias.is_finite()),
                "slot {}'s bias row is not canonical: its tokens must be strictly increasing and \
                 its biases finite",
                slot.slot
            );
        }
        Ok(())
    }
}

/// Whether `tokens` are strictly increasing.
fn increasing(tokens: &[u32]) -> bool {
    tokens.windows(2).all(|pair| pair[0] < pair[1])
}

impl Columns {
    /// The number of rows (sequences) the columns describe.
    ///
    /// A row is one sequence's query, which is not the same as a call: a verify batch carries
    /// several rows for one call, and the per-row columns (`slots`, `parameters`) are as long as
    /// the rows.
    pub fn rows(&self) -> usize {
        self.rows.rows()
    }

    /// The calls the columns describe.
    pub fn calls(&self) -> usize {
        self.rows.calls.len()
    }

    /// The number of query tokens the columns describe.
    pub fn tokens(&self) -> usize {
        self.rows.tokens()
    }

    /// Whether every column has the length its count promises and the counts fit `shape`, so that
    /// a mis-sized batch fails before a launch rather than after one.
    fn check(&self, shape: &Shape) -> ValidationResult<()> {
        let tokens = self.tokens();
        let rows = self.rows();
        let calls = self.calls();
        let parameters = proto::module::sampler::PARAMETER_COLUMNS as usize;

        for (name, actual, expected) in [
            ("positions", self.rows.positions.len(), tokens),
            ("cache_positions", self.rows.cache_positions.len(), tokens),
            ("input_ids", self.input_ids.len(), tokens),
            ("seq_lens", self.rows.seq_lens.len(), rows),
            ("logits_rows", self.rows.logits_rows.len(), rows),
            ("slots", self.slots.len(), rows),
            ("parameters", self.parameters.len(), rows * parameters),
            ("draws", self.draws.len(), rows),
            (
                "commit",
                self.commit.len(),
                proto::module::commit::COMMIT_FIELDS * calls,
            ),
        ] {
            ensure_valid!(
                actual == expected,
                "{name} has {actual} entries, the batch has {rows} rows, {calls} calls and \
                 {tokens} tokens"
            );
        }
        ensure_valid!(
            tokens <= shape.tokens as usize && rows <= shape.rows as usize,
            "the batch holds {rows} rows and {tokens} tokens, more than its shape's {} and {}",
            shape.rows,
            shape.tokens
        );
        ensure_valid!(self.page_size != 0, "the batch's page size is zero");

        // The optional columns: each is empty or follows its count.
        for (name, actual, expected) in [
            ("targets", self.targets.len(), tokens),
            ("checkpoints", self.checkpoints.len(), rows),
            ("drafts", self.drafts.len(), calls),
            ("biased", self.biased.len(), rows),
            ("banned", self.banned.len(), rows),
        ] {
            ensure_valid!(
                actual == 0 || actual == expected,
                "{name} has {actual} entries, the batch has {rows} rows, {calls} calls and \
                 {tokens} tokens"
            );
        }
        let scores = self.targets.iter().any(|target| *target >= 0);
        ensure_valid!(
            scores == shape.variant.scores(),
            "a {} batch {} prompt targets",
            shape.variant,
            if shape.variant.scores() {
                "names no"
            } else {
                "names"
            }
        );
        ensure_valid!(
            self.banned.iter().all(|tokens| increasing(tokens)),
            "a row's banned tokens are not strictly increasing"
        );
        Ok(())
    }

    /// Checks the recurrent-state copies against the calls of the batch: a copy runs on the slot
    /// of one of the batch's calls, every restore precedes every checkpoint (the restores run
    /// before the forward and the checkpoints after it), a slot restores at most once, and each
    /// checkpoint answers one row's non-zero `checkpoints` entry.
    fn check_state_copies(&self, slots: &HashSet<u32>) -> ValidationResult<()> {
        let mut restored = HashSet::with_capacity(self.state_copies.len());
        for copy in &self.state_copies {
            ensure_valid!(
                slots.contains(&copy.slot),
                "a state copy names a slot no call of the batch runs on"
            );
            if copy.direction == StateDirection::Restore {
                ensure_valid!(
                    restored.insert(copy.slot),
                    "a batch restores one slot's state twice"
                );
            }
        }
        ensure_valid!(
            self.state_copies
                .is_sorted_by_key(|copy| copy.direction != StateDirection::Restore),
            "a state checkpoint precedes a restore"
        );

        let keeps = self
            .state_copies
            .iter()
            .filter(|copy| copy.direction == StateDirection::Keep)
            .count();
        let kept_rows = self
            .checkpoints
            .iter()
            .filter(|tokens| **tokens != 0)
            .count();
        ensure_valid!(
            keeps == kept_rows,
            "the batch's state checkpoints do not answer its rows' checkpoint tokens"
        );
        Ok(())
    }
}

impl BatchOutput {
    /// Checks that the output answers `batch`, one call output per request, and that every call
    /// output's fields agree with its status.
    ///
    /// # Errors
    ///
    /// [`ValidationError`] naming the first rule the output breaks.
    pub fn check(&self, batch: &Batch) -> ValidationResult<()> {
        ensure_valid!(
            self.calls.len() == batch.requests.len(),
            "batch {} advanced {} requests and its output answers {}",
            batch.id.0,
            batch.requests.len(),
            self.calls.len()
        );
        self.calls.iter().try_for_each(CallOutput::check)
    }
}

impl CallOutput {
    /// Checks that the output's fields agree with its status. Rules that relate an output to the
    /// phase of its call belong to the scheduler, which holds the request.
    fn check(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.cache_visible_len <= self.cache_computed_len,
            "a call output's visible cache length exceeds its computed length"
        );
        ensure_valid!(
            self.error_code.is_some() == (self.status == CallStatus::Error),
            "a call output carries an error code exactly when it failed"
        );
        ensure_valid!(
            self.error_message.is_none() || self.error_code == Some(ErrorCode::InputError),
            "only an input-error output states why the rank refused its request"
        );
        ensure_valid!(
            self.token_logprobs.is_empty() || self.token_logprobs.len() == self.tokens.len(),
            "a call output's token log-probabilities do not align with its tokens"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use proto::module::commit::COMMIT_FIELDS;
    use proto::module::sampler::PARAMETER_COLUMNS;
    use proto::video::{LatentShape, Layout};

    use super::*;

    /// A verification of two requests: the first verifies two candidates, the second one, so the
    /// batch has 2 calls, 3 rows and, with a committed token and a draft per row but the last,
    /// 5 tokens. The three counts differ, so a column sized by the wrong one is caught.
    fn verification() -> Batch {
        let tokens = 5;
        let rows = 3;
        let calls = 2;

        Batch {
            id: BatchId(4),
            group: GroupId(1),
            phase: Phase::Verify,
            kernels: 0,
            policy: 0,
            revision: 0,
            shape: Shape::prefill(4, 8, Variant::VerifyGreedy),
            requests: vec![RequestId(7), RequestId(9)],
            columns: Columns {
                rows: Rows {
                    calls: vec![
                        Row {
                            slot: 1,
                            kind: Phase::Verify,
                            cache_group: 0,
                            rows: 0..2,
                        },
                        Row {
                            slot: 2,
                            kind: Phase::Verify,
                            cache_group: 0,
                            rows: 2..3,
                        },
                    ],
                    positions: vec![10, 11, 12, 13, 30],
                    cache_positions: vec![10, 11, 12, 13, 30],
                    seq_lens: vec![12, 14, 31],
                    query_offsets: vec![0, 2, 4, 5],
                    logits_rows: vec![1, 3, 4],
                },
                input_ids: vec![5; tokens],
                slots: vec![1, 1, 2],
                parameters: vec![0.0; rows * PARAMETER_COLUMNS as usize],
                draws: vec![DrawAddress::default(); rows],
                commit: vec![0; COMMIT_FIELDS * calls],
                page_size: 16,
                max_kv_len: 4096,
                ..Columns::default()
            },
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    fn refusal(batch: &Batch) -> String {
        batch.check().unwrap_err().to_string()
    }

    #[test]
    fn a_batch_whose_three_counts_agree_is_accepted() {
        verification().check().unwrap();
    }

    /// Each count is checked against the columns that follow it: a token column, a row column
    /// and the commit table each refused when one entry is missing, and the commit table refused
    /// when it is sized by rows rather than calls.
    #[test]
    fn a_column_that_disagrees_with_its_count_is_refused() {
        let mut short_tokens = verification();
        short_tokens.columns.rows.cache_positions.pop();
        assert!(refusal(&short_tokens).contains("cache_positions"));

        let mut short_rows = verification();
        short_rows.columns.slots.pop();
        assert!(refusal(&short_rows).contains("slots"));

        let mut per_row_commit = verification();
        per_row_commit.columns.commit = vec![0; COMMIT_FIELDS * 3];
        assert!(refusal(&per_row_commit).contains("commit"));

        let mut short_drafts = verification();
        short_drafts.columns.drafts = vec![DraftWork::default()];
        assert!(refusal(&short_drafts).contains("drafts"));
    }

    /// The counts must fit the shape the bucket is chosen by, and a token batch names one call
    /// row per request.
    #[test]
    fn a_batch_larger_than_its_shape_or_its_requests_is_refused() {
        let mut few_rows = verification();
        few_rows.shape.rows = 2;
        assert!(refusal(&few_rows).contains("more than its shape"));

        let mut few_tokens = verification();
        few_tokens.shape.tokens = 4;
        assert!(refusal(&few_tokens).contains("more than its shape"));

        let mut one_request = verification();
        one_request.requests.pop();
        assert!(refusal(&one_request).contains("2 call rows for 1 requests"));
    }

    /// A media batch has one row per request, and a device computation states the extent its
    /// graph is captured for while an encode, sized by rows and tokens, need not.
    #[test]
    fn a_media_batch_has_one_row_per_request_and_states_its_extent() {
        let row = MediaRow {
            request: RequestId(3),
            slot: 1,
            conditioning: Vec::new(),
            media: Vec::new(),
            step: 0,
            steps: 8,
            seed: 1,
            width: 64,
            height: 64,
            frames: 9,
            rounds: 0,
            unit: 0,
            rank: None,
            image: None,
            results: vec![0, 1],
            reads: 1,
        };
        let layout = Layout {
            video: LatentShape {
                channels: 16,
                frames: 3,
                height: 8,
                width: 8,
            },
            audio: None,
            text_rows: 16,
            condition_rows: 0,
            condition_elements: 0,
        };
        let denoise = Batch {
            id: BatchId(5),
            group: GroupId(0),
            phase: Phase::Denoise,
            kernels: 0,
            policy: 0,
            revision: 0,
            shape: Shape {
                rows: 1,
                tokens: 0,
                variant: Variant::Denoise,
                extent: Extent::Latents(layout),
            },
            requests: vec![RequestId(3)],
            columns: Columns::default(),
            media: vec![row.clone()],
            new_slots: Vec::new(),
        };
        denoise.check().unwrap();

        let mut two_rows = denoise.clone();
        two_rows.media.push(MediaRow {
            request: RequestId(4),
            ..row
        });
        assert!(refusal(&two_rows).contains("1 requests carries 2 media rows"));

        let mut no_extent = denoise.clone();
        no_extent.shape.extent = Extent::None;
        assert!(refusal(&no_extent).contains("states no extent"));

        let mut encode = no_extent;
        encode.phase = Phase::TextEncode;
        encode.shape.variant = Variant::Encode;
        encode.check().unwrap();
    }

    /// An output answers its batch by position: one call output per request.
    #[test]
    fn an_output_answers_each_request_once() {
        let batch = verification();
        let call = CallOutput {
            status: CallStatus::Ok,
            error_code: None,
            error_message: None,
            tokens: vec![5],
            step_index: 1,
            cache_computed_len: 13,
            cache_visible_len: 12,
            token_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            media_output: None,
            draft_token_ids: Vec::new(),
        };
        let mut output = BatchOutput {
            calls: vec![call.clone(), call],
            ..BatchOutput::default()
        };
        output.check(&batch).unwrap();

        output.calls.pop();
        assert!(output.check(&batch).is_err());
    }
}

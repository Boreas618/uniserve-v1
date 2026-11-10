//! Sampling: the request's penalties, its candidate filtering, its draw, the log-probabilities it
//! reports, and the slot state a step reads and advances.
//!
//! The module is the worker's one owner of sampling: the [`Sampler`] layer, the per-slot penalty
//! count table it reads ([`Sampling`]), and the commit table ([`CommitColumns`], `ops::commit`)
//! that installs each row's token into its slot after the draw. The causal-LM runner composes them
//! inside a bucket's captured graph.
//!
//! One sampling step is a fixed sequence of operator launches over one row per active request:
//!
//! ```text
//! logits [rows, vocabulary]        → logits_shape   → shaped [rows, vocabulary] f32
//! logits                           → log_softmax   → logprobs [rows, vocabulary] f32  (when asked)
//! shaped                           → top_k          → values [rows, width] f32, ids [rows, width] i32
//! values + ids                     → filter_candidates → values (dropped = -inf), valid [rows]
//! values + ids                     → candidate_order → the same rows in ascending token id
//! ordered values                   → softmax        → probabilities [rows, width] f32
//! probabilities + draws            → inverse_cdf    → tokens [rows]
//! logprobs + tokens                → logprob_gather → logp [rows] f32                    (when asked)
//! ```
//!
//! The reported log-probabilities are the raw distribution's: `log_softmax` of the model's logits
//! before penalties, temperature and filtering, written over the shaped row once the selection no
//! longer reads it. A greedy step reports them the same way after its argmax.
//!
//! A greedy row takes the second of two launch sequences instead, and only its first step:
//!
//! ```text
//! logits [rows, vocabulary]        → logits_shape   → shaped [rows, vocabulary] f32
//! shaped                           → argmax         → tokens [rows]                     (the id is the column)
//! ```
//!
//! The order of the arena reservations below is that sequence, so a captured bucket's plan is
//! static: the same layers on the same extents reserve the same addresses on capture and on every
//! replay.
//!
//! ## The draws are inputs
//!
//! The sampler never generates randomness. `draws` is `[rows]` f32 in `[0, 1)`, computed by the
//! runner from the request's seed and the token position it is sampling (Philox, addressed per
//! element), so a restored or replayed step draws the same number.
//!
//! ## Greedy rows
//!
//! A row whose temperature is zero is **not** a draw: it is [`Sampler::greedy`]'s row. The shaper
//! divides such a row by one rather than by zero (`ops::sampler`'s doc cites the rule), so the row's
//! shaped logits keep the logits' order and its largest candidate is the temperature-zero token. The
//! temperature of each row is a request parameter, so the worker knows which rows are greedy and
//! routes them: `argmax` over the row's logits for those, this path's draw for the rest. No draw can
//! replace that step, because the draw path's candidates are sorted by token id before the CDF, so
//! a draw of zero selects the lowest token id rather than the largest logit.
//!
//! ## Rows that cannot be sampled
//!
//! `filter_candidates` reports, per row, whether the row's kept candidates form a usable
//! distribution (a row can end up empty when a nucleus or min-p threshold and a top-k width leave
//! nothing). The sampler does not branch on it — the value is device data and the step is inside a
//! captured graph — but leaves it in the candidate working set's `valid` row for the worker, and
//! samples the row as its first candidate, which is deterministic and reproducible. Whether such a
//! request continues, is retried with different parameters, or fails is the worker's decision, not
//! a numerical one.

/// Lanes one candidate program walks at a time: a chunk of the row, not the row.
///
/// Both programs that walk a row of candidates - the distribution and the draw - take it, so a
/// candidate width may be the whole vocabulary without either holding a row in registers.
const CANDIDATE_TILE: u32 = 8192;
/// The lanes one `top_k` pass reads: the only size that operator's key names, since the vocabulary
/// and the kept count are launch scalars. Smaller than the candidate tile because an ordering pass
/// holds two planes and a nibble's ranks at once.
const TOP_TILE: u32 = 2048;

use ops::norm::{Softmax, SoftmaxArgs, SoftmaxDescriptor};
use ops::sampler::{
    ArgMax, ArgMaxArgs, ArgMaxDescriptor, CandidateOrder, FilterArgs, FilterCandidates,
    FilterDescriptor, InverseCdf, InverseCdfArgs, InverseCdfDescriptor, LogSoftmax, LogSoftmaxArgs,
    LogSoftmaxDescriptor, LogitsShape, LogprobArgs, LogprobDescriptor, LogprobGather, OrderArgs,
    OrderDescriptor, ShapeArgs, ShapeDescriptor, TopK, TopKArgs, TopKDescriptor,
};
use ops::{DType, Shape, TensorView};
use proto::program::sampler::PARAMETER_COLUMNS;

use nn::layers::launch;
use nn::{Arena, Forward};

use platform::DeviceId;

use crate::plan::Buckets;
use crate::runners::columns::{Column, encode_i32};
use crate::{Error, Result};

/// A view or a construction argument the sampling step cannot take, named by its owner.
fn refused(owner: &'static str, problem: impl Into<String>) -> Error {
    Error::Invalid(format!("{owner}: {}", problem.into()))
}

/// The request state a sampling step reads.
///
/// The worker owns it and writes it in the layout the operators read: one row of parameters per
/// row being sampled, the slot whose history penalizes the row, and the slot's counts.
pub struct SamplerState {
    /// `[slots, vocabulary]` i32: how many times each slot has already seen each token.
    pub counts: TensorView,
    /// `[rows, PARAMETER_COLUMNS]` f32: temperature, top-k width, top-p, min-p, repetition,
    /// frequency and presence in [`proto::program::sampler::PARAMETER_COLUMNS`] order.
    ///
    /// That order is the operators' and the worker writes it. A column's meaning is its position
    /// alone, so a writer using another order would turn a top-k width into a nucleus threshold
    /// and a nucleus threshold into a min-p threshold on every request, which no shape check can
    /// see. This doc and `Sampler::check` are the only two places in this
    /// crate that know the count, and the check uses the constant, so a column added to the
    /// operators needs nothing here.
    pub parameters: TensorView,
    /// `[rows]` i32: the slot each row belongs to.
    pub slots: TensorView,
    /// `[rows, extras]` i32 tokens counted besides the slot's own history, when the state carries
    /// any; `-1` counts nothing.
    pub extras: Option<TensorView>,
}

/// The candidate working set of one sampling step, reserved from the forward's arena.
pub struct Candidates {
    /// `[rows, vocabulary]` f32 penalized, temperature-scaled logits.
    pub shaped: TensorView,
    /// `[rows, width]` f32 the kept candidate logits, descending.
    pub values: TensorView,
    /// `[rows, width]` i32 the vocabulary ids of the candidates.
    pub ids: TensorView,
    /// `[rows, width]` f32 the candidates' probabilities, in the reference's token order.
    pub probabilities: TensorView,
    /// `[rows, width]` f32 the filtered candidates in ascending token id order, whole to the width:
    /// the slots no candidate took hold `-inf`, which carries no mass.
    pub ordered_values: TensorView,
    /// `[rows, width]` i32 the ids of `ordered_values`, in the same order, whole to the width: the
    /// slots no candidate took hold `-1`, which is outside every vocabulary. A slot is therefore
    /// never memory nothing wrote, and it cannot read as the legitimate token `0`.
    pub ordered_ids: TensorView,
    /// `[rows]` i32: 1 when the row's candidates form a usable distribution.
    pub valid: TensorView,
    /// `[rows, 4 · width + 2 · TOP_K_BINS]` i32: the ordering's scratch, which `top_k` fills.
    pub scratch: TensorView,
    /// `[rows, vocabulary]` i32: the id space's counts, which `candidate_order` owns for a launch.
    ///
    /// It is the size of the vocabulary and not of the candidate row because the ordering counts
    /// ranks rather than comparing candidates: a deployment with top-k disabled orders a row as wide
    /// as the vocabulary, and the quadratic form of that comparison was sixteen seconds a token.
    pub counts: TensorView,
}

impl Candidates {
    /// Reserve the working set of `rows` rows from `arena`, in the order the operators read it.
    pub fn reserve(arena: &Arena, rows: u64, vocabulary: u64, width: u64) -> Result<Self> {
        Ok(Self {
            shaped: arena.tensor(DType::F32, Shape::new(&[rows, vocabulary]))?,
            values: arena.tensor(DType::F32, Shape::new(&[rows, width]))?,
            ids: arena.tensor(DType::I32, Shape::new(&[rows, width]))?,
            probabilities: arena.tensor(DType::F32, Shape::new(&[rows, width]))?,
            ordered_values: arena.tensor(DType::F32, Shape::new(&[rows, width]))?,
            ordered_ids: arena.tensor(DType::I32, Shape::new(&[rows, width]))?,
            valid: arena.tensor(DType::I32, Shape::new(&[rows]))?,
            scratch: arena.tensor(DType::I32, ops::sampler::top_k_workspace(rows, width))?,
            counts: arena.tensor(DType::I32, Shape::new(&[rows, vocabulary]))?,
        })
    }
}

/// What one sampling step wrote.
pub struct Samples {
    /// `[rows]` the sampled token of each row.
    pub tokens: TensorView,
    /// `[rows]` f32 the log-probability of each sampled token, when the step reports them.
    pub logprobabilities: Option<TensorView>,
}

impl Samples {
    /// Reserve the step's outputs from `arena`: the tokens, and the log-probabilities when asked.
    pub fn reserve(arena: &Arena, tokens: DType, rows: u64, logprobs: bool) -> Result<Self> {
        Ok(Self {
            tokens: arena.tensor(tokens, Shape::new(&[rows]))?,
            logprobabilities: match logprobs {
                true => Some(arena.tensor(DType::F32, Shape::new(&[rows]))?),
                false => None,
            },
        })
    }
}

/// The sampling step of a language model.
pub struct Sampler {
    vocabulary: u32,
    slots: u32,
    width: u32,
    extras: u32,
    block: u32,
    tokens: DType,
}

impl Sampler {
    /// A sampler over a `vocabulary`-wide logit row, for at most `slots` request slots.
    ///
    /// `width` is how many candidates a row keeps: the largest top-k width the requests this
    /// sampler serves may ask for, capped at the vocabulary. `extras` is how many token columns a
    /// row carries besides its slot's counts (a draft's tokens, a verified prefix), which only
    /// changes the shaping operator's descriptor.
    pub fn new(vocabulary: u32, slots: u32, width: u32, extras: u32) -> Result<Self> {
        if vocabulary == 0 {
            return Err(refused(
                "Sampler",
                "a sampler needs a non-empty vocabulary".to_owned(),
            ));
        }
        if slots == 0 {
            return Err(refused(
                "Sampler",
                "a sampler needs at least one row".to_owned(),
            ));
        }
        if width == 0 || width > vocabulary {
            return Err(refused(
                "Sampler",
                format!("a candidate width of {width} is outside 1..={vocabulary}"),
            ));
        }
        Ok(Self {
            vocabulary,
            slots,
            width,
            extras,
            block: launch::row_tile(vocabulary),
            tokens: DType::I32,
        })
    }

    /// The vocabulary width of a logit row.
    pub const fn vocabulary(&self) -> u32 {
        self.vocabulary
    }

    /// How many candidates a row keeps.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The element type of a sampled token.
    pub const fn tokens(&self) -> DType {
        self.tokens
    }

    /// Sample one token per row of `logits`, and the log-probability of each when `samples` carries
    /// them.
    ///
    /// `logits` is `[rows, vocabulary]` in the head's output dtype; `draws` is `[rows]` f32 in
    /// `[0, 1)` from the worker. The step's working set comes from the forward's arena, so the
    /// call is inside a captured bucket's plan.
    ///
    /// The candidates reach [`InverseCdf`] in ascending token id order: [`TopK`] writes them by
    /// descending logit, [`FilterCandidates`] preserves that, and [`CandidateOrder`] permutes both
    /// the values and the ids into token order, which is what makes a fixed draw select the same
    /// token whatever order the top-k program kept the candidates in. Greedy rows do not go
    /// through this method at all — see [`Self::greedy`].
    pub fn forward(
        &self,
        f: &Forward,
        logits: TensorView,
        state: &SamplerState,
        draws: TensorView,
        samples: &Samples,
    ) -> Result<()> {
        let vocabulary = u64::from(self.vocabulary);
        let width = u64::from(self.width);
        let rows = self.check(logits, state, samples)?;
        draws
            .require_shape("Sampler", Shape::new(&[rows]))
            .map_err(Error::from)?;
        if draws.dtype != DType::F32 {
            return Err(refused(
                "Sampler",
                format!("expected f32 draws, found {}", draws.dtype),
            ));
        }

        let arena = f.arena();
        let candidates = Candidates::reserve(arena, rows, vocabulary, width)?;

        // 1. Penalties and temperature.
        let shape = self.shape_descriptor(logits.dtype);
        let args = ShapeArgs {
            logits,
            counts: state.counts,
            slots: state.slots,
            parameters: state.parameters,
            extras: state.extras,
            out: candidates.shaped,
        };

        f.operate::<LogitsShape>(&shape, args)?;

        // 2. The candidate set: the `width` largest, then the row's own thresholds.
        let top = self.top_descriptor();
        let args = TopKArgs {
            logits: candidates.shaped,
            values: candidates.values,
            ids: candidates.ids,
            workspace: candidates.scratch,
        };

        f.operate::<TopK>(&top, args)?;

        let filter = self.filter_descriptor();
        let args = FilterArgs {
            values: candidates.values,
            ids: candidates.ids,
            parameters: state.parameters,
            valid: candidates.valid,
        };

        f.operate::<FilterCandidates>(&filter, args)?;

        // 3. Token order, because the draw must not depend on the order the top-k program kept the
        // candidates in: ordered by token id before the inverse CDF, a fixed draw selects the same
        // token for the same candidate set.
        let order = self.order_descriptor();
        let args = OrderArgs {
            values: candidates.values,
            ids: candidates.ids,
            ordered_values: candidates.ordered_values,
            ordered_ids: candidates.ordered_ids,
            counts: candidates.counts,
            vocabulary: u64::from(self.vocabulary),
        };

        f.operate::<CandidateOrder>(&order, args)?;

        // 4. The candidates' distribution, then one draw per row.
        let softmax = self.softmax_descriptor();
        let args = SoftmaxArgs {
            x: candidates.ordered_values,
            out: candidates.probabilities,
        };

        f.operate::<Softmax>(&softmax, args)?;

        let draw = self.draw_descriptor();
        let args = InverseCdfArgs {
            probabilities: candidates.probabilities,
            ids: candidates.ordered_ids,
            draws,
            tokens: samples.tokens,
        };

        f.operate::<InverseCdf>(&draw, args)?;

        // 5. The reported log-probabilities. The shaped row is dead once the draw has read the
        // candidates, so it holds the scores and the step reserves nothing more for them.
        self.report(f, logits, candidates.shaped, samples)?;

        Ok(())
    }

    /// Write the log-probability of each row's sampled token into `samples`, when it carries a
    /// row for them.
    ///
    /// The scores are those of the model's raw distribution: `log_softmax` of the unshaped
    /// `logits`, before penalties, temperature and candidate filtering, which is what
    /// `docs/architecture.md` states the engine reports and vLLM's default (`raw_logprobs`)
    /// answers. `scratch` is a `[rows, vocabulary]` f32 view the caller no longer reads; the
    /// scores of the whole vocabulary are written there before one per row is gathered.
    fn report(
        &self,
        f: &Forward,
        logits: TensorView,
        scratch: TensorView,
        samples: &Samples,
    ) -> Result<()> {
        let Some(logprobabilities) = samples.logprobabilities else {
            return Ok(());
        };
        let softmax = self.log_softmax_descriptor(logits.dtype);
        let args = LogSoftmaxArgs {
            x: logits,
            out: scratch,
        };

        f.operate::<LogSoftmax>(&softmax, args)?;

        let gather = self.logprob_descriptor();
        let args = LogprobArgs {
            logprobabilities: scratch,
            tokens: samples.tokens,
            out: logprobabilities,
        };

        f.operate::<LogprobGather>(&gather, args)?;
        Ok(())
    }

    /// The descriptors this layer launches with, each built in exactly one place.
    ///
    /// Every field is layer geometry or a fixed element type — the vocabulary, the candidate width,
    /// the row block, the token width — and both sampling paths launch these. The row count of a
    /// draw is a launch value and appears in none of them.
    ///
    /// Penalties and temperature over the whole vocabulary: `logits`' element type is the one thing
    /// a deployment chooses.
    fn shape_descriptor(&self, logits: DType) -> ShapeDescriptor {
        ShapeDescriptor {
            dtype: logits,
            vocabulary: self.vocabulary,
            block: self.block,
            extras: self.extras,
            divide: true,
            pdl: launch::PDL,
        }
    }

    /// The candidate set: the row's `width` candidates in descending score order.
    ///
    /// What the emission owes, because every later step reads it as an order rather than a set: each
    /// row's first `width` lanes hold that row's candidates, `values` never increases along the row,
    /// ties are ordered by ascending token id (a draw must not depend on which of two equal scores a
    /// program happened to meet first), and the lanes past `width` in the buffers are not written.
    /// `filter_candidates` reads the largest at lane 0 and the request's own `top_k` at index
    /// `top_k - 1`, and `candidate_order` then sorts by id — so an emission that is a set and not an
    /// order silently changes what a request draws.
    fn top_descriptor(&self) -> TopKDescriptor {
        TopKDescriptor {
            dtype: DType::F32,
            vocabulary: self.vocabulary,
            width: self.width,
            tile: TOP_TILE,
            ids: DType::I32,
            pdl: launch::PDL,
        }
    }

    /// The row's own thresholds applied to its candidates.
    fn filter_descriptor(&self) -> FilterDescriptor {
        FilterDescriptor {
            width: self.width,
            tile: CANDIDATE_TILE,
            ids: DType::I32,
            pdl: launch::PDL,
        }
    }

    /// The candidates reordered by token id, so a draw does not depend on the order the top-k
    /// program kept.
    fn order_descriptor(&self) -> OrderDescriptor {
        OrderDescriptor {
            values: DType::F32,
            ids: DType::I32,
            pdl: launch::PDL,
        }
    }

    /// The candidates' distribution.
    ///
    /// The tile is a **chunk of the row**, not the row: the program walks the width in tiles of
    /// `CANDIDATE_TILE` lanes and keeps only a running maximum and a running denominator, so the
    /// candidate width may be the whole vocabulary. A tile derived from the width would ask for one
    /// register per candidate.
    fn softmax_descriptor(&self) -> SoftmaxDescriptor {
        SoftmaxDescriptor {
            dtype: DType::F32,
            columns: self.width,
            tile: CANDIDATE_TILE,
            pdl: launch::PDL,
        }
    }

    /// One inverse-CDF draw per row, over the ordered candidates.
    ///
    /// The tile is a chunk of the row, as the distribution's is: the draw walks the candidates in
    /// tiles and keeps a running cumulative, so the width may be the whole vocabulary.
    fn draw_descriptor(&self) -> InverseCdfDescriptor {
        InverseCdfDescriptor {
            width: self.width,
            tile: CANDIDATE_TILE,
            ids: DType::I32,
            tokens: self.tokens,
            pdl: launch::PDL,
        }
    }

    /// The greedy path's argmax over the shaped row.
    fn argmax_descriptor(&self) -> ArgMaxDescriptor {
        ArgMaxDescriptor {
            dtype: DType::F32,
            width: self.vocabulary,
            ids: self.tokens,
            pdl: launch::PDL,
        }
    }

    /// The reported log-probabilities over the raw distribution of the whole vocabulary, read in
    /// the logits' own element type.
    fn log_softmax_descriptor(&self, logits: DType) -> LogSoftmaxDescriptor {
        LogSoftmaxDescriptor {
            dtype: logits,
            columns: self.vocabulary,
            tile: self.block,
            pdl: launch::PDL,
        }
    }

    /// The reported log-probability of the token each row drew.
    fn logprob_descriptor(&self) -> LogprobDescriptor {
        LogprobDescriptor {
            columns: self.vocabulary,
            ids: self.tokens,
            pdl: launch::PDL,
        }
    }

    /// What both sampling paths read: the logit rows, the request state, and the outputs.
    ///
    /// [`Self::forward`] calls this and then checks its draws; [`Self::greedy`] calls it and launches.
    /// It is private because the two paths are the whole of this layer's interface — a caller does not
    /// choose a half-step — but it is one method so the two paths cannot disagree about what a valid
    /// step is.
    fn check(&self, logits: TensorView, state: &SamplerState, samples: &Samples) -> Result<u64> {
        let vocabulary = u64::from(self.vocabulary);
        logits
            .require_row("Sampler", logits.dtype, vocabulary)
            .map_err(Error::from)?;
        let rows = logits.rows();
        if rows == 0 {
            return Err(refused(
                "Sampler",
                "a sampling step has at least one row".to_owned(),
            ));
        }
        if !matches!(logits.dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(refused(
                "Sampler",
                format!(
                    "{} logits are not a shape the shaper reads: f32, bf16 or f16",
                    logits.dtype
                ),
            ));
        }
        if samples.tokens.dtype != self.tokens {
            return Err(refused(
                "Sampler",
                format!(
                    "expected {} tokens, found {}",
                    self.tokens, samples.tokens.dtype
                ),
            ));
        }
        samples
            .tokens
            .require_shape("Sampler", Shape::new(&[rows]))
            .map_err(Error::from)?;
        if let Some(logprobabilities) = samples.logprobabilities {
            logprobabilities
                .require_shape("Sampler", Shape::new(&[rows]))
                .map_err(Error::from)?;
            if logprobabilities.dtype != DType::F32 {
                return Err(refused(
                    "Sampler",
                    format!(
                        "expected f32 log-probabilities, found {}",
                        logprobabilities.dtype
                    ),
                ));
            }
        }
        state
            .parameters
            .require_shape("Sampler", Shape::new(&[rows, u64::from(PARAMETER_COLUMNS)]))
            .map_err(Error::from)?;
        if state.parameters.dtype != DType::F32 {
            return Err(refused(
                "Sampler",
                format!("expected f32 parameters, found {}", state.parameters.dtype),
            ));
        }
        state
            .slots
            .require_shape("Sampler", Shape::new(&[rows]))
            .map_err(Error::from)?;
        state
            .counts
            .require_row("Sampler", DType::I32, vocabulary)
            .map_err(Error::from)?;
        if state.counts.rows() != u64::from(self.slots) {
            return Err(refused(
                "Sampler",
                format!(
                    "{} count rows for {} slots",
                    state.counts.rows(),
                    self.slots
                ),
            ));
        }
        match (self.extras, state.extras) {
            (0, None) => {}
            (0, Some(_)) => {
                return Err(refused(
                    "Sampler",
                    "the sampler reads no extra tokens and the state carries some".to_owned(),
                ));
            }
            (_, Some(_)) => {}
            (extras, None) => {
                return Err(refused(
                    "Sampler",
                    format!("the sampler reads {extras} extra tokens and the state has none"),
                ));
            }
        }
        if let Some(extra) = state.extras {
            extra
                .require_shape("Sampler", Shape::new(&[rows, u64::from(self.extras)]))
                .map_err(Error::from)?;
            if extra.dtype != DType::I32 {
                return Err(refused(
                    "Sampler",
                    format!("expected i32 extra tokens, found {}", extra.dtype),
                ));
            }
        }

        Ok(rows)
    }

    /// Take each row's largest shaped logit as its token, and report its log-probability when
    /// `samples` carries one.
    ///
    /// This is the greedy step (`torch.max` over the shaped row, the lowest index among ties), for
    /// the rows whose temperature is zero. It runs the same shaper as [`Self::forward`] and nothing
    /// else: no candidate set, no cutoffs, no draw. It is a separate launch sequence
    /// rather than a branch inside `forward`, because the worker's buckets are homogeneous in the
    /// path their rows take and a bucket's captured plan is one of the two.
    pub fn greedy(
        &self,
        f: &Forward,
        logits: TensorView,
        state: &SamplerState,
        samples: &Samples,
    ) -> Result<()> {
        let rows = self.check(logits, state, samples)?;
        let vocabulary = u64::from(self.vocabulary);
        let arena = f.arena();
        let shaped = arena.tensor(DType::F32, Shape::new(&[rows, vocabulary]))?;

        let shape = self.shape_descriptor(logits.dtype);
        let args = ShapeArgs {
            logits,
            counts: state.counts,
            slots: state.slots,
            parameters: state.parameters,
            extras: state.extras,
            out: shaped,
        };

        f.operate::<LogitsShape>(&shape, args)?;

        // The column of a dense vocabulary row is the token id, so the index is the token.
        let argmax = self.argmax_descriptor();
        let args = ArgMaxArgs {
            values: shaped,
            out: samples.tokens,
        };

        f.operate::<ArgMax>(&argmax, args)?;

        // The shaped row is dead once the argmax has read it, so it holds the scores.
        self.report(f, logits, shaped, samples)?;

        Ok(())
    }
}

/// The commit table of a warm pose: `COMMIT_FIELDS` columns of `rows` i64 values, every row's slot
/// and task `-1` and every other field zero.
///
/// The kernel treats a row with a negative slot as having no destination (no store of any kind) and
/// a row with a negative task as having sampled nothing, so this table commits nothing while the
/// launch that reads it is still recorded in the bucket's graph - which is what makes the graph a
/// warmup captures the graph a serving batch replays.
pub(crate) fn inert_commit(rows: usize) -> Vec<i64> {
    let mut table = vec![0_i64; proto::program::commit::COMMIT_FIELDS * rows];
    for row in 0..rows {
        table[proto::program::commit::column::SLOT * rows + row] = -1;
        table[proto::program::commit::column::TASK * rows + row] = -1;
    }
    table
}

/// The descriptor a plain decode step's commit uses, from the table's capacity and the vocabulary.
///
/// `ops::commit` checks the table's leading stride against `stride`, every operand's type against
/// `index`, and the block against `count`; the arithmetic is here, once, so a host test can state it
/// without a device and the runner cannot drift from it.
pub(crate) fn commit_descriptor(
    capacity: u64,
    block: u32,
    vocabulary: u32,
) -> ops::commit::CommitDescriptor {
    ops::commit::CommitDescriptor {
        // The table and the sampled ids are i32 end to end: `Sampler` emits i32 tokens, and the
        // operator checks `tokens` against `index`.
        index: DType::I32,
        verified: DType::I32,
        penalties: DType::I32,
        // The table is `[COMMIT_FIELDS, capacity]` and a column is contiguous.
        stride: capacity,
        // One next-input token per slot, in a `[slots, 1]` matrix.
        future_stride: 1,
        vocab: vocabulary,
        block,
    }
}

/// Every `slot_commit` descriptor a runner of `buckets` can launch, one per distinct block.
///
/// The commit is a runner-owned operator: no model calls it, so the epoch's program set does not
/// contain it unless the runner asks. Its descriptor carries the vocabulary and the launch block
/// (the smallest power of two covering the bucket's rows) while the table's capacity is the largest
/// bucket's rows for every bucket, so the keys a deployment needs are one per power of two its
/// buckets reach - eight for the default catalog.
pub fn commit_descriptors(
    buckets: &Buckets,
    vocabulary: u32,
) -> Vec<ops::commit::CommitDescriptor> {
    let capacity = buckets
        .iter()
        .map(|bucket| u64::from(bucket.rows))
        .max()
        .unwrap_or(1);
    let mut blocks: Vec<u32> = buckets
        .iter()
        .map(|bucket| commit_block(bucket.rows))
        .collect();
    blocks.sort_unstable();
    blocks.dedup();
    blocks
        .into_iter()
        .map(|block| commit_descriptor(capacity, block, vocabulary))
        .collect()
}

/// The commit table the device reads, from the mechanism's `[COMMIT_FIELDS, rows]` i64 table.
///
/// The device table is `[COMMIT_FIELDS, capacity]` i32: the staged rows fill its first `rows`
/// columns, and the rest keep whatever the previous batch left there behind `dim(1) >= count`.
pub(crate) fn commit_table(commit: &[i64], rows: usize, capacity: usize) -> Result<Vec<i32>> {
    let fields = proto::program::commit::COMMIT_FIELDS;
    if commit.len() != fields * rows {
        return Err(Error::Invariant(format!(
            "the commit table holds {} values, {fields} fields of {rows} rows",
            commit.len()
        )));
    }
    let mut table = vec![0_i32; fields * capacity];
    // Rows past the batch's own are **inert**, not zero: a zero slot is slot 0, a real destination,
    // so a launch that ever ran over the whole capacity would commit thirteen phantom tokens into
    // the first slot the deployment serves. The staged table already pads its own rows with the
    // operator's sentinels; this makes the tail inert as well, so the invariant survives whatever
    // `count` a future launch passes (`ops::commit`'s two sentinels).
    for row in rows.min(capacity)..capacity {
        table[proto::program::commit::column::SLOT * capacity + row] = -1;
        table[proto::program::commit::column::TASK * capacity + row] = -1;
    }
    for field in 0..fields {
        for row in 0..rows.min(capacity) {
            let value = commit[field * rows + row];
            table[field * capacity + row] = i32::try_from(value).map_err(|_| {
                Error::Invalid(format!("a commit column value {value} does not fit i32"))
            })?;
        }
    }
    Ok(table)
}

/// The block a commit launch needs for `rows` rows: the smallest power of two that covers them.
pub(crate) fn commit_block(rows: u32) -> u32 {
    rows.max(1).next_power_of_two()
}

/// The token-state commit's operands (`ops::commit`), bound once and rewritten per batch.
///
/// A batch's sampled tokens must advance its slots' state - the histories the repetition,
/// frequency and presence penalties subtract, and the positions a verify row carries. Without the
/// launch a slot's state never moves, so the second decode step is computed against an empty
/// history. It runs once per batch, after the sampler and before the read-back, inside the bucket's
/// capture, so a replay advances the state exactly once.
///
/// Every operand is allocated when the runner binds and only the table's contents change per
/// batch: the warm pose writes it with every row's slot and task `-1`, which the kernel treats as
/// inert on every store, so the graph recorded at warmup is the graph a serving batch replays.
pub(crate) struct CommitColumns {
    /// `[COMMIT_FIELDS, capacity]` i32, column-major: the table the mechanism's rows fill.
    pub(crate) table: Column,
    /// `[rows]` i32 whether a row's token is valid, and whether its slot is active.
    pub(crate) valid: Column,
    pub(crate) active: Column,
    /// `[rows]` u8 continuation flags and `[rows]` i32 accepted points, one per sampled task: a plain
    /// decode's rows use neither, and the commit reads them at the row's task all the same.
    pub(crate) continuation: Column,
    pub(crate) points: Column,
    /// `[slots]` the per-slot next-input predicate, logical position, sampling position and
    /// verification length. A decode row advances the two positions by one.
    pub(crate) predicates: Column,
    pub(crate) logical: Column,
    pub(crate) sampling: Column,
    pub(crate) verified: Column,
    /// `[slots]` i32 one next-input token per slot: a decode without speculation drafts none.
    pub(crate) future: Column,
    /// The table's leading stride, which the descriptor states.
    pub(crate) capacity: u64,
    /// The launch's block, which must cover every row of the largest bucket.
    pub(crate) block: u32,
    /// The slots the per-slot state columns address.
    pub(crate) slots: u32,
    /// `[slots]` i32 the per-slot trajectory positions, read back after a replay: the commit kernel
    /// advances them, so only the device knows where a call's slot stands. Two, because a replay's
    /// positions are read while the next replay's copy is on its way.
    pub(crate) readback: [platform::PinnedBuffer; 2],
}

impl CommitColumns {
    /// The commit operands of a runner whose largest bucket has `rows` rows over `slots` slots.
    pub(crate) fn new(device: DeviceId, rows: u32, slots: u32) -> Result<Self> {
        let capacity = u64::from(rows.max(1));
        let mut columns = Self {
            table: Column::new(
                device,
                DType::I32,
                u64::from(proto::program::commit::COMMIT_FIELDS as u32) * capacity,
            )?,
            valid: Column::new(device, DType::I32, capacity)?,
            active: Column::new(device, DType::I32, capacity)?,
            continuation: Column::new(device, DType::U8, capacity)?,
            points: Column::new(device, DType::I32, capacity)?,
            predicates: Column::new(device, DType::U8, u64::from(slots.max(1)))?,
            logical: Column::new(device, DType::I32, u64::from(slots.max(1)))?,
            sampling: Column::new(device, DType::I32, u64::from(slots.max(1)))?,
            verified: Column::new(device, DType::I32, u64::from(slots.max(1)))?,
            future: Column::new(device, DType::I32, u64::from(slots.max(1)))?,
            capacity,
            block: commit_block(rows),
            slots: slots.max(1),
            readback: [
                platform::PinnedBuffer::new(
                    device,
                    2 * slots.max(1) as usize * std::mem::size_of::<i32>(),
                )?,
                platform::PinnedBuffer::new(
                    device,
                    2 * slots.max(1) as usize * std::mem::size_of::<i32>(),
                )?,
            ],
        };
        // The roles a decode step leaves inert hold their starting values; only `table` is written
        // again per batch.
        let mut encoded = Vec::new();
        let ones = vec![1_i32; capacity as usize];
        encode_i32(&mut encoded, &ones);
        columns.valid.write(&encoded)?;
        columns.active.write(&encoded)?;
        let zeros = vec![0_i32; slots.max(1) as usize];
        encode_i32(&mut encoded, &zeros);
        for column in [
            &mut columns.logical,
            &mut columns.sampling,
            &mut columns.verified,
            &mut columns.future,
        ] {
            column.write(&encoded)?;
        }
        let zeros_i32 = vec![0_i32; capacity as usize];
        encode_i32(&mut encoded, &zeros_i32);
        columns.points.write(&encoded)?;
        let zeros_u8 = vec![0_u8; capacity as usize];
        columns.continuation.write(&zeros_u8)?;
        let zeros_u8 = vec![0_u8; slots.max(1) as usize];
        columns.predicates.write(&zeros_u8)?;
        Ok(columns)
    }

    /// The slots the per-slot state columns address.
    pub(crate) fn slots(&self) -> usize {
        self.slots as usize
    }

    /// States slot `slot`'s next input, for a token the host chose after the graph drew another.
    pub(crate) fn set_next_input(
        &self,
        slot: usize,
        token: i32,
        stream: &platform::Stream,
    ) -> Result<()> {
        let offset = (slot * std::mem::size_of::<i32>()) as u64;
        // SAFETY: the column holds one i32 per slot and the caller's slot is inside it; no launch
        // that reads the column is in flight, and the wait below ends the copy's use of the source.
        unsafe {
            *self.future.host.as_ptr().cast::<i32>().add(slot) = token;
            platform::memcpy_async(
                self.future.device.ptr.address() + offset,
                self.future.host.address() + offset,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }
        stream.synchronize()?;
        Ok(())
    }

    /// Moves the columns the host rewrites or reads to device storage. The others hold values no
    /// step changes, or values only the kernel touches, and stay where they are.
    pub(crate) fn place(
        &mut self,
        device: DeviceId,
        pools: &mut crate::arena::Pools,
        stream: &platform::Stream,
    ) -> Result<()> {
        for column in [
            &mut self.table,
            &mut self.logical,
            &mut self.sampling,
            &mut self.future,
        ] {
            column.place(device, pools, stream)?;
        }
        Ok(())
    }

    /// The device address of the per-slot trajectory positions.
    pub(crate) fn sampling_ptr(&self) -> ops::DevicePtr {
        self.sampling.device.ptr
    }

    /// The device address of the per-slot logical positions: the cache extent execution
    /// initialized.
    pub(crate) fn logical_ptr(&self) -> ops::DevicePtr {
        self.logical.device.ptr
    }

    /// The bytes one readback buffer holds, which bounds what one copy may read.
    pub(crate) fn readback_len(&self) -> usize {
        self.readback[0].len()
    }

    /// The host address a readback copies the trajectory positions into.
    pub(crate) fn readback_address(&self, turn: usize) -> u64 {
        self.readback[turn].address()
    }

    /// The host side of the readback, as the positions the copy filled.
    pub(crate) fn readback_ptr(&self, turn: usize) -> *const u8 {
        self.readback[turn].as_ptr()
    }

    /// Writes the staged commit table - `[COMMIT_FIELDS, rows]` i64, column-major - as the i32 table
    /// the descriptor asks for, then hands every operand to the device.
    pub(crate) fn stage(
        &mut self,
        commit: &[i64],
        staged_rows: usize,
        capacity: usize,
    ) -> Result<()> {
        if commit.is_empty() {
            return Ok(());
        }
        let table = commit_table(commit, staged_rows, capacity)?;
        let mut encoded = Vec::new();
        encode_i32(&mut encoded, &table);
        self.table.write(&encoded)?;
        Ok(())
    }

    /// The `[COMMIT_FIELDS, capacity]` i32 view the kernel reads.
    pub(crate) fn table_view(&self) -> TensorView {
        TensorView::with_strides(
            self.table.device.ptr,
            DType::I32,
            ops::Shape::new(&[proto::program::commit::COMMIT_FIELDS as u64, self.capacity]),
            ops::Strides::new(&[self.capacity, 1]),
        )
    }

    /// Copies the one operand a batch rewrites into its device buffer.
    pub(crate) fn upload(&mut self, stream: &platform::Stream) -> Result<()> {
        self.table.push(stream)
    }
}

/// The sampling step's own device state: the slot histories its penalties read.
pub(crate) struct Sampling {
    pub(crate) sampler: Sampler,
    /// `[slots, vocabulary]` i32 counts of the tokens each slot has already seen, which the
    /// penalties subtract; zeroed once when the runner binds and advanced by `ops::commit`.
    pub(crate) counts: TensorView,
    pub(crate) _counts: platform::DeviceAllocation,
}

impl Sampling {
    /// The sampler and its zeroed count table for `slots` request slots over `vocabulary` tokens.
    ///
    /// A request may ask for any top-k it likes, so the candidate width is the vocabulary: a
    /// narrower buffer would silently sample a different set.
    pub(crate) fn new(
        device: DeviceId,
        stream: &platform::Stream,
        vocabulary: u32,
        slots: u32,
    ) -> Result<Self> {
        let sampler = Sampler::new(vocabulary, slots, vocabulary, 0)?;
        let bytes = u64::from(slots) * u64::from(vocabulary) * 4;
        let allocation = platform::DeviceAllocation::linear(
            device,
            usize::try_from(bytes).map_err(|_| {
                Error::Invalid("the sampler's count table is larger than memory".into())
            })?,
        )?;
        // The table starts at zero: a slot has seen no token before its first call. A fresh device
        // allocation holds whatever its pages held last, so the zeros are written by the device,
        // stream-ordered before the first sample reads a count.
        // SAFETY: the range is this allocation's own `slots × vocabulary` i32 words, and the
        // allocation outlives the stream's work on it (the runner owns both).
        unsafe {
            platform::memset_d32(
                allocation.address(),
                0,
                usize::try_from(bytes / 4).map_err(|_| {
                    Error::Invalid("the sampler's count table is larger than memory".into())
                })?,
                stream,
            )?;
        }
        stream.synchronize()?;
        let counts = TensorView::new(
            ops::DevicePtr::new(allocation.address()),
            DType::I32,
            ops::Shape::new(&[u64::from(slots), u64::from(vocabulary)]),
        );
        Ok(Self {
            sampler,
            counts,
            _counts: allocation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ops::DevicePtr;

    /// A view of `shape` at a host address: these tests never launch, so the address is a marker.
    fn view(shape: &[u64]) -> TensorView {
        TensorView::new(DevicePtr::new(0x1000), DType::F32, Shape::new(shape))
    }

    /// The state's own rows — the slot counts, the slot ids and the extra tokens — are integers.
    fn ints(shape: &[u64]) -> TensorView {
        TensorView::new(DevicePtr::new(0x1000), DType::I32, Shape::new(shape))
    }

    #[test]
    fn a_step_is_built_for_a_vocabulary_and_a_candidate_width() {
        let sampler = Sampler::new(151_936, 8, 256, 0).unwrap();
        assert_eq!(sampler.vocabulary(), 151_936);
        assert_eq!(sampler.width(), 256);
        assert_eq!(sampler.tokens(), DType::I32);
        // The width is how many candidates a row keeps: not zero, not more than the vocabulary.
        assert!(Sampler::new(151_936, 8, 0, 0).is_err());
        assert!(Sampler::new(1_024, 8, 2_048, 0).is_err());
        assert!(Sampler::new(0, 8, 16, 0).is_err());
        assert!(Sampler::new(1_024, 0, 16, 0).is_err());
    }

    /// The working set and the outputs lie in one arena, in the order the operators read them, so a
    /// captured bucket's plan is static.
    #[test]
    fn the_working_set_is_reserved_in_the_order_the_step_reads_it() {
        let arena = Arena::new(DevicePtr::new(0x10_0000), 1 << 20);
        let rows = 4;
        let candidates = Candidates::reserve(&arena, rows, 1_024, 32).unwrap();
        let samples = Samples::reserve(&arena, DType::I32, rows, true).unwrap();
        let order = [
            candidates.shaped,
            candidates.values,
            candidates.ids,
            candidates.probabilities,
            candidates.ordered_values,
            candidates.ordered_ids,
            candidates.valid,
            samples.tokens,
            samples.logprobabilities.unwrap(),
        ];
        let mut previous = 0;
        for view in order {
            assert!(view.ptr.address() > previous);
            assert_eq!(view.ptr.address() % nn::forward::ALIGNMENT, 0);
            previous = view.ptr.address();
        }
        assert_eq!(candidates.shaped.dtype, DType::F32);
        assert_eq!(candidates.shaped.shape.dims(), &[rows, 1_024]);
        assert_eq!(candidates.values.shape.dims(), &[rows, 32]);
        assert_eq!(candidates.ids.dtype, DType::I32);
        assert_eq!(candidates.ordered_values.shape.dims(), &[rows, 32]);
        assert_eq!(candidates.ordered_ids.dtype, DType::I32);
        assert_eq!(candidates.valid.shape.dims(), &[rows]);
        assert_eq!(samples.tokens.shape.dims(), &[rows]);
        // Without log-probabilities the step reserves neither the row of log-probabilities nor the
        // whole-vocabulary log-softmax output.
        let arena = Arena::new(DevicePtr::new(0x10_0000), 1 << 20);
        let _ = Candidates::reserve(&arena, rows, 1_024, 32).unwrap();
        let samples = Samples::reserve(&arena, DType::I32, rows, false).unwrap();
        assert!(samples.logprobabilities.is_none());
    }

    /// The refusals a caller meets before either path launches. `check` is the one body both paths
    /// run, so a request it accepts is one both paths accept and a request it refuses names the
    /// argument, not the path — which is the point of the tests being here rather than nowhere: the
    /// paths themselves need a live device, and this is the half of them a host can hold to account.
    #[test]
    fn a_step_refuses_the_state_it_cannot_read() {
        let sampler = Sampler::new(1_024, 4, 64, 2).unwrap();
        let state = |extras: bool| SamplerState {
            counts: ints(&[4, 1_024]),
            parameters: view(&[2, u64::from(PARAMETER_COLUMNS)]),
            slots: ints(&[2]),
            extras: extras.then(|| ints(&[2, 2])),
        };
        let samples = Samples {
            tokens: TensorView::new(DevicePtr::new(0x2000), DType::I32, Shape::new(&[2])),
            logprobabilities: Some(TensorView::new(
                DevicePtr::new(0x2000),
                DType::F32,
                Shape::new(&[2]),
            )),
        };
        sampler
            .check(view(&[2, 1_024]), &state(true), &samples)
            .expect("a step whose state is its own");

        // A row of the wrong width, a step with no rows, and a logit type the shaper cannot read.
        for (logits, named) in [
            (view(&[2, 512]), "1024"),
            (view(&[0, 1_024]), "one row"),
            (
                TensorView::new(DevicePtr::new(0x1000), DType::I32, Shape::new(&[2, 1_024])),
                "f32, bf16 or f16",
            ),
        ] {
            let text = format!(
                "{}",
                sampler.check(logits, &state(true), &samples).unwrap_err()
            );
            assert!(text.contains(named), "{text}");
        }

        // The request state: a row of parameters too narrow, counts that are not one row per slot,
        // and the two ways the extra columns and the sampler's declaration can disagree.
        let narrow = SamplerState {
            counts: ints(&[4, 1_024]),
            parameters: view(&[2, 4]),
            slots: ints(&[2]),
            extras: Some(ints(&[2, 2])),
        };
        assert!(sampler.check(view(&[2, 1_024]), &narrow, &samples).is_err());
        let miscounted = SamplerState {
            counts: ints(&[3, 1_024]),
            parameters: view(&[2, u64::from(PARAMETER_COLUMNS)]),
            slots: ints(&[2]),
            extras: Some(ints(&[2, 2])),
        };
        let text = format!(
            "{}",
            sampler
                .check(view(&[2, 1_024]), &miscounted, &samples)
                .unwrap_err()
        );
        assert!(text.contains('3') && text.contains('4'), "{text}");
        let absent = format!(
            "{}",
            sampler
                .check(view(&[2, 1_024]), &state(false), &samples)
                .unwrap_err()
        );
        assert!(absent.contains('2'), "{absent}");

        // And the sampler that reads no extra tokens refuses a state that carries them, rather than
        // demanding a zero-width view of them.
        let plain = Sampler::new(1_024, 4, 64, 0).unwrap();
        let text = format!(
            "{}",
            plain
                .check(view(&[2, 1_024]), &state(true), &samples)
                .unwrap_err()
        );
        assert!(text.contains("no extra tokens"), "{text}");
    }

    /// A greedy step reads no draws, so the state a greedy caller must supply is exactly the state a
    /// draw caller must supply: what it leaves out is the uniform draw, nothing else.
    #[test]
    fn a_greedy_step_needs_no_draws_to_be_a_valid_step() {
        let sampler = Sampler::new(1_024, 4, 64, 0).unwrap();
        let state = SamplerState {
            counts: ints(&[4, 1_024]),
            parameters: view(&[2, u64::from(PARAMETER_COLUMNS)]),
            slots: ints(&[2]),
            extras: None,
        };
        let samples = Samples {
            tokens: TensorView::new(DevicePtr::new(0x2000), DType::I32, Shape::new(&[2])),
            logprobabilities: None,
        };
        assert_eq!(
            sampler.check(view(&[2, 1_024]), &state, &samples).unwrap(),
            2
        );
        // The tokens a greedy step writes are whatever type the sampler declares, and a step whose
        // output is a different one is refused by name rather than launched into a wrong view.
        let narrow = Samples {
            tokens: TensorView::new(DevicePtr::new(0x2000), DType::I64, Shape::new(&[2])),
            logprobabilities: None,
        };
        let text = format!(
            "{}",
            sampler
                .check(view(&[2, 1_024]), &state, &narrow)
                .unwrap_err()
        );
        assert!(text.contains("i32"), "{text}");
    }
}

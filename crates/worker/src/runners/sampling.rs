//! Sampling: the request's penalties, its candidate filtering, its draw, the log-probabilities it
//! reports, and the slot state a step reads and advances.
//!
//! The module is the worker's one owner of sampling: the [`Sampler`] layer, the per-slot penalty
//! count table and bias table it reads ([`Sampling`]), each batch's logit adjustments
//! ([`Adjustments`]), and the commit table ([`CommitColumns`], `ops::commit`) that installs each
//! row's token into its slot after the draw. The causal-LM runner composes them inside a bucket's
//! captured graph.
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
//! logprobs + tokens                → logprob_gather → logp [rows] f32, rank [rows] i32  (when asked)
//! logprobs                         → top_logprobs ×2 → top [rows, n] f32, ids [rows, n] (n > 0)
//! ```
//!
//! The reported log-probabilities are the raw distribution's: `log_softmax` of the model's logits
//! before the logit adjustments, penalties, temperature and filtering - vLLM's default
//! (`logprobs_mode = raw_logprobs`), so a logit bias, an allowed set or a banned token changes which
//! token is chosen and not the score reported for it. Both paths write them over the shaped row
//! once the selection no longer reads it. Each chosen token is reported with its competition rank
//! in that distribution, and with the row's `n` most likely tokens when the sampler reports
//! candidates ([`Sampler::reporting`]): `n` is the deployment's `max_logprobs`, a request takes the
//! first of them it asks for, and a captured graph computes them for every row because whether a
//! row's request asked is not a fact it can branch on.
//!
//! ## Logit adjustments
//!
//! `logits_shape` applies a row's adjustments before its penalties, in vLLM's order: its slot's
//! row of the bias table (the request's logit bias, `-inf` outside its allowed set) when the row's
//! flag names it, then `-inf` at every token its ban mask sets (the last token of a bad word whose
//! prefix the history ends with, the end-of-sequence tokens under a `min_tokens` floor). Both
//! paths shape through it, so a greedy row's argmax is over the adjusted logits too. A row with no
//! adjustment selects its own logits unchanged, so a batch that asks for none samples bit for bit
//! as without them, with the same launches: the operator reads three more operands and loads none
//! of them for such a row.
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
//! A row whose temperature is zero asks for its largest shaped logit. The shaper divides such a row
//! by one rather than by zero (`ops::sampler`'s doc cites the rule), so the row's shaped logits keep
//! the logits' order and its largest candidate is the temperature-zero token. A bucket of greedy
//! rows takes [`Sampler::greedy`]'s launch sequence, an argmax. A greedy row in a drawing bucket -
//! beside a drawn row in the same batch - is staged with a top-k width of one, so its candidate set
//! is that one token and its draw, whatever the uniform, selects it: the candidates sorted by token
//! id before the CDF are a set of one. The two paths agree on ties, because `top_k` orders equal
//! scores by ascending id and `argmax` answers the lowest index.
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
/// The lanes one program of the candidates' first selection reads (`ops::sampler::TopLogprobs`): a
/// vocabulary-wide row is selected in this many lanes' chunks in parallel, and the chunks'
/// candidates in one more launch.
const LOGPROB_TILE: u32 = 4096;
/// The widest row one `top_logprobs` program selects from, which bounds the chunks' candidates the
/// second selection reads.
const LOGPROB_MAX_TILE: u32 = 8192;

use ops::norm::{Softmax, SoftmaxArgs, SoftmaxDescriptor};
use ops::sampler::{
    ArgMax, ArgMaxArgs, ArgMaxDescriptor, CandidateOrder, FilterArgs, FilterCandidates,
    FilterDescriptor, InverseCdf, InverseCdfArgs, InverseCdfDescriptor, LogSoftmax, LogSoftmaxArgs,
    LogSoftmaxDescriptor, LogitsShape, LogprobArgs, LogprobDescriptor, LogprobGather, OrderArgs,
    OrderDescriptor, ShapeArgs, ShapeDescriptor, TopK, TopKArgs, TopKDescriptor, TopLogprobsArgs,
    TopLogprobsDescriptor,
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
    /// `[slots, vocabulary]` f32 the slots' bias rows ([`Sampling::bias`]).
    pub bias: TensorView,
    /// `[rows, 2]` i32 each row's adjustments: its bias flag and its ban mask's index
    /// ([`Adjustments`]).
    pub adjustments: TensorView,
    /// `[masks, ceil(vocabulary / 32)]` i32 the batch's ban masks.
    pub bans: TensorView,
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
    /// The scores of each sampled token, when the step reports them.
    pub scores: Option<Scores>,
}

impl Samples {
    /// Reserve the step's outputs from `arena`: the tokens, and their scores with `candidates`
    /// top tokens of a `vocabulary`-wide row when `candidates` is stated.
    pub fn reserve(
        arena: &Arena,
        tokens: DType,
        rows: u64,
        vocabulary: u32,
        candidates: Option<u32>,
    ) -> Result<Self> {
        Ok(Self {
            tokens: arena.tensor(tokens, Shape::new(&[rows]))?,
            scores: match candidates {
                Some(candidates) => Some(Scores::reserve(arena, rows, vocabulary, candidates)?),
                None => None,
            },
        })
    }
}

/// The scores of one token per row on the model's raw distribution: its log-probability, its rank,
/// and the row's most likely tokens.
#[derive(Clone, Copy, Debug)]
pub struct Scores {
    /// `[rows]` f32 the token's log-probability.
    pub logprobs: TensorView,
    /// `[rows]` i32 the token's one-based competition rank in its row, `0` for a row whose token
    /// names no position (`ops::sampler::LogprobGather`).
    pub ranks: TensorView,
    /// The row's most likely tokens, when the sampler reports any.
    pub top: Option<TopLogprobs>,
}

/// A row's `n` most likely tokens and their log-probabilities, the largest first and ties in
/// ascending id order (`ops::sampler::TopLogprobs`' order).
#[derive(Clone, Copy, Debug)]
pub struct TopLogprobs {
    /// `[rows, n]` f32 the candidates' log-probabilities, descending.
    pub values: TensorView,
    /// `[rows, n]` i32 the candidates' ids.
    pub ids: TensorView,
    /// `[rows, chunks · n]` f32 and i32: each chunk's candidates, which the first selection writes
    /// and the second reads.
    pub chunk_values: TensorView,
    pub chunk_ids: TensorView,
}

impl TopLogprobs {
    /// The candidates of `rows` rows of a `vocabulary`-wide distribution, `candidates` a row, with
    /// their chunks' views beside them, from `values` and `ids` and `arena`.
    pub fn with_chunks(
        arena: &Arena,
        values: TensorView,
        ids: TensorView,
        rows: u64,
        vocabulary: u32,
        candidates: u32,
    ) -> Result<Self> {
        let chunks = chunk_candidates(vocabulary, candidates);
        Ok(Self {
            values,
            ids,
            chunk_values: arena.tensor(DType::F32, Shape::new(&[rows, chunks]))?,
            chunk_ids: arena.tensor(DType::I32, Shape::new(&[rows, chunks]))?,
        })
    }
}

/// The candidates a `vocabulary`-wide row's chunks keep, `candidates` a chunk.
fn chunk_candidates(vocabulary: u32, candidates: u32) -> u64 {
    ops::sampler::top_logprobs_chunks(u64::from(vocabulary), LOGPROB_TILE) * u64::from(candidates)
}

impl Scores {
    /// Reserve the scores of `rows` rows of a `vocabulary`-wide distribution from `arena`, with
    /// `candidates` top tokens a row when it is above zero.
    pub fn reserve(arena: &Arena, rows: u64, vocabulary: u32, candidates: u32) -> Result<Self> {
        let logprobs = arena.tensor(DType::F32, Shape::new(&[rows]))?;
        let ranks = arena.tensor(DType::I32, Shape::new(&[rows]))?;
        let top = match candidates {
            0 => None,
            candidates => {
                let width = u64::from(candidates);
                Some(TopLogprobs::with_chunks(
                    arena,
                    arena.tensor(DType::F32, Shape::new(&[rows, width]))?,
                    arena.tensor(DType::I32, Shape::new(&[rows, width]))?,
                    rows,
                    vocabulary,
                    candidates,
                )?)
            }
        };
        Ok(Self {
            logprobs,
            ranks,
            top,
        })
    }

    /// The candidates a row reports, `0` when the scores carry none.
    pub fn candidates(&self) -> u64 {
        self.top.map_or(0, |top| top.values.shape.last())
    }

    /// The scores of rows `start..start + count`: the same buffers, narrowed to those rows.
    pub fn rows(&self, start: u64, count: u64) -> Result<Self> {
        let narrow = |view: TensorView| view.narrow(0, start, count).map_err(Error::from);
        Ok(Self {
            logprobs: narrow(self.logprobs)?,
            ranks: narrow(self.ranks)?,
            top: match self.top {
                Some(top) => Some(TopLogprobs {
                    values: narrow(top.values)?,
                    ids: narrow(top.ids)?,
                    chunk_values: narrow(top.chunk_values)?,
                    chunk_ids: narrow(top.chunk_ids)?,
                }),
                None => None,
            },
        })
    }
}

/// One scored token as the completion reports it: the token itself with its log-probability and
/// rank, then the row's top candidates read back beside it, the largest first.
///
/// `values` and `ids` are the row's candidates as `top_logprobs` wrote them, and a candidate's rank is its
/// competition rank among them - one plus the candidates strictly above it - which is its rank in
/// the row, since every entry above a kept candidate is kept before it. A candidate whose id is
/// outside the vocabulary (a row with fewer finite entries than candidates) is not reported. The
/// token itself may be among its candidates; the scheduler keeps the candidates each request asks
/// for and drops that repetition.
pub fn scored(
    token: u32,
    logprob: f32,
    rank: u32,
    values: &[f32],
    ids: &[i32],
    vocabulary: u32,
) -> Vec<ipc::TokenLogprob> {
    let mut entries = Vec::with_capacity(1 + values.len());
    entries.push(ipc::TokenLogprob {
        token_id: token,
        logprob,
        rank,
    });
    for (index, (&value, &id)) in values.iter().zip(ids).enumerate() {
        let Some(id) = u32::try_from(id).ok().filter(|id| *id < vocabulary) else {
            continue;
        };
        let above = values[..index]
            .iter()
            .filter(|other| **other > value)
            .count();
        entries.push(ipc::TokenLogprob {
            token_id: id,
            logprob: value,
            rank: above as u32 + 1,
        });
    }
    entries
}

/// The sampling step of a language model.
pub struct Sampler {
    vocabulary: u32,
    slots: u32,
    width: u32,
    extras: u32,
    block: u32,
    tokens: DType,
    /// The most likely tokens a scored row reports beside its own ([`Self::reporting`]).
    candidates: u32,
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
            candidates: 0,
        })
    }

    /// The same sampler, reporting each scored row's `candidates` most likely tokens beside its
    /// own: the deployment's `max_logprobs`, which bounds what a request may ask for.
    ///
    /// # Errors
    ///
    /// A count above the vocabulary, which has no that many tokens to rank.
    pub fn reporting(mut self, candidates: u32) -> Result<Self> {
        if candidates > self.vocabulary {
            return Err(refused(
                "Sampler",
                format!(
                    "{candidates} reported candidates of a vocabulary of {}",
                    self.vocabulary
                ),
            ));
        }
        if candidates > 0 {
            merge_tile(self.vocabulary, candidates)?;
        }
        self.candidates = candidates;
        Ok(self)
    }

    /// The most likely tokens a scored row reports beside its own.
    pub const fn candidates(&self) -> u32 {
        self.candidates
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
    ///
    /// A greedy row (temperature zero) is shaped with a divisor of one and drawn from the
    /// candidates its `top_k` keeps; the worker stages a width of one for it, so its draw is the
    /// argmax of its shaped row and a drawing bucket serves greedy rows beside drawn ones.
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

        // 1. Logit adjustments, penalties and temperature.
        self.shape(f, logits, state, candidates.shaped)?;

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

    /// Adjust, penalize and temperature-scale `logits` into `out`, as both paths do.
    fn shape(
        &self,
        f: &Forward,
        logits: TensorView,
        state: &SamplerState,
        out: TensorView,
    ) -> Result<()> {
        let shape = self.shape_descriptor(logits.dtype);
        let args = ShapeArgs {
            logits,
            counts: state.counts,
            slots: state.slots,
            parameters: state.parameters,
            extras: state.extras,
            bias: state.bias,
            adjustments: state.adjustments,
            bans: state.bans,
            out,
        };

        f.operate::<LogitsShape>(&shape, args)?;
        Ok(())
    }

    /// Write the scores of each row's sampled token into `samples`, when it carries them.
    ///
    /// The scores are those of the model's raw distribution: `log_softmax` of the unshaped
    /// `logits`, before penalties, temperature and candidate filtering, which is what
    /// `docs/architecture.md` states the engine reports and vLLM's default (`raw_logprobs`)
    /// answers. `scratch` is a `[rows, vocabulary]` f32 view the caller no longer reads; the
    /// scores of the whole vocabulary are written there before each row's are read.
    fn report(
        &self,
        f: &Forward,
        logits: TensorView,
        scratch: TensorView,
        samples: &Samples,
    ) -> Result<()> {
        let Some(scores) = samples.scores else {
            return Ok(());
        };
        self.log_softmax(f, logits, scratch)?;
        self.score(f, scratch, samples.tokens, &scores)
    }

    /// Write `log_softmax` of each row of `logits` into `out`, a `[rows, vocabulary]` f32 view:
    /// the raw distribution a row's scores are read from.
    pub fn log_softmax(&self, f: &Forward, logits: TensorView, out: TensorView) -> Result<()> {
        let softmax = self.log_softmax_descriptor(logits.dtype);
        let args = LogSoftmaxArgs { x: logits, out };

        f.operate::<LogSoftmax>(&softmax, args)?;
        Ok(())
    }

    /// Score one token per row of `logprobabilities`, a `[rows, vocabulary]` f32 view of the raw
    /// distribution: its log-probability and rank, and the row's most likely tokens when `scores`
    /// carries room for them. A negative token names no position and scores nothing (rank `0`);
    /// its candidates are written all the same, and its reader ignores them.
    pub fn score(
        &self,
        f: &Forward,
        logprobabilities: TensorView,
        tokens: TensorView,
        scores: &Scores,
    ) -> Result<()> {
        let rows = logprobabilities.rows();
        let gather = self.logprob_descriptor();
        let args = LogprobArgs {
            logprobabilities,
            tokens,
            out: scores.logprobs,
            ranks: scores.ranks,
        };

        f.operate::<LogprobGather>(&gather, args)?;

        let Some(top) = scores.top else {
            return Ok(());
        };
        let width = top.values.shape.last();
        if top.values.shape.dims() != [rows, width] || top.ids.shape.dims() != [rows, width] {
            return Err(refused(
                "Sampler",
                format!(
                    "{rows} scored rows with candidates of {:?} and ids of {:?}",
                    top.values.shape.dims(),
                    top.ids.shape.dims()
                ),
            ));
        }
        // The candidates are the row's largest log-probabilities: each `LOGPROB_TILE`-lane chunk's
        // first, by one program a chunk, then the row's among the chunks' by one program a row.
        // Both launches keep ties in ascending id order.
        let candidates = u32::try_from(width)
            .map_err(|_| refused("Sampler", "a candidate count wider than u32"))?;
        let chunks = TopLogprobsDescriptor {
            candidates,
            tile: LOGPROB_TILE,
            indirect: false,
            pdl: launch::PDL,
        };
        let args = TopLogprobsArgs {
            values: logprobabilities,
            ids: None,
            out_values: top.chunk_values,
            out_ids: top.chunk_ids,
        };

        f.operate::<ops::sampler::TopLogprobs>(&chunks, args)?;

        let row = TopLogprobsDescriptor {
            tile: merge_tile(self.vocabulary, candidates)?,
            indirect: true,
            ..chunks
        };
        let args = TopLogprobsArgs {
            values: top.chunk_values,
            ids: Some(top.chunk_ids),
            out_values: top.values,
            out_ids: top.ids,
        };

        f.operate::<ops::sampler::TopLogprobs>(&row, args)?;
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

    /// The reported log-probability and rank of the token each row drew: the rank's count walks
    /// the row in the same tiles the log-softmax does.
    fn logprob_descriptor(&self) -> LogprobDescriptor {
        LogprobDescriptor {
            columns: self.vocabulary,
            ids: self.tokens,
            tile: self.block,
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
        if let Some(scores) = samples.scores {
            for (name, view, dtype) in [
                ("log-probabilities", scores.logprobs, DType::F32),
                ("ranks", scores.ranks, DType::I32),
            ] {
                view.require_shape("Sampler", Shape::new(&[rows]))
                    .map_err(Error::from)?;
                if view.dtype != dtype {
                    return Err(refused(
                        "Sampler",
                        format!("expected {dtype} {name}, found {}", view.dtype),
                    ));
                }
            }
            if scores.candidates() != u64::from(self.candidates) {
                return Err(refused(
                    "Sampler",
                    format!(
                        "the sampler reports {} candidates and the scores hold {}",
                        self.candidates,
                        scores.candidates()
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
        state
            .bias
            .require_row("Sampler", DType::F32, vocabulary)
            .map_err(Error::from)?;
        if state.bias.rows() != u64::from(self.slots) {
            return Err(refused(
                "Sampler",
                format!("{} bias rows for {} slots", state.bias.rows(), self.slots),
            ));
        }
        state
            .adjustments
            .require_shape("Sampler", Shape::new(&[rows, 2]))
            .map_err(Error::from)?;
        state
            .bans
            .require_row(
                "Sampler",
                DType::I32,
                u64::from(ops::sampler::ban_words(self.vocabulary)),
            )
            .map_err(Error::from)?;
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
    /// the rows whose temperature is zero: the shaped row is the adjusted, penalized logits, so a
    /// greedy request honours its logit bias, its allowed set and its banned tokens, as vLLM's
    /// greedy sampler does. It runs the same shaper as [`Self::forward`] and nothing
    /// else: no candidate set, no cutoffs, no draw. It is a separate launch sequence
    /// rather than a branch inside `forward`, because a bucket's captured plan is one of the two:
    /// a bucket whose rows are all greedy takes this one, and a bucket with a drawn row takes
    /// `forward`, which serves its greedy rows as a draw over their one largest candidate.
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

        self.shape(f, logits, state, shaped)?;

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

/// The lanes the second candidates' selection reads: the chunks' candidates of a
/// `vocabulary`-wide row, rounded up to a power of two, which one program must hold.
///
/// # Errors
///
/// A count whose chunks keep more candidates than one program holds: `max_logprobs` is bounded
/// by `LOGPROB_MAX_TILE` over the vocabulary's chunks (215 over a 151936-token vocabulary).
fn merge_tile(vocabulary: u32, candidates: u32) -> Result<u32> {
    let width = chunk_candidates(vocabulary, candidates);
    let tile = width.next_power_of_two();
    if tile > u64::from(LOGPROB_MAX_TILE) {
        return Err(refused(
            "Sampler",
            format!(
                "{candidates} reported candidates keep {width} of a {vocabulary}-token vocabulary's \
                 chunks, more than the {LOGPROB_MAX_TILE} one selection holds"
            ),
        ));
    }
    Ok(tile as u32)
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
    /// The host side of [`CommitColumns::start`]: `[4, slots]` i32 cells (next input token,
    /// logical position, sampling position, verified length) and `[slots]` u8 predicates, one cell
    /// per slot, so a slot's start is copied from cells no other slot's start rewrites.
    starts: platform::PinnedBuffer,
}

/// The state a slot starts with when its first call on this worker is a decode: a request whose
/// prompt another worker computed and whose first token that worker sampled, as a decode group
/// continues a prefill group's request. It is what that worker's explicit commit left in its own
/// slot, which the decode then advances by one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotStart {
    /// The request's slot.
    pub(crate) slot: u32,
    /// The token the decode reads as its input: the last token the request generated.
    pub(crate) token: i32,
    /// The token's logical position, which is also the cache extent before it.
    pub(crate) logical: i32,
    /// The position of the draw the decode samples with.
    pub(crate) sampling: i32,
    /// The cache length the request's slot has verified: its tokens before the input token.
    pub(crate) verified: i32,
}

/// The i32 fields of a [`SlotStart`] after its slot: token, logical, sampling and verified.
const START_FIELDS: usize = 4;

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
            starts: platform::PinnedBuffer::new(
                device,
                slots.max(1) as usize * (START_FIELDS * std::mem::size_of::<i32>() + 1),
            )?,
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

    /// Starts `start.slot`'s state on `stream`, before the step that reads it: its next input
    /// token, its logical and sampling positions and verified length, and a set predicate.
    ///
    /// A decode reads its input token from the slot's state and the commit after it advances the
    /// positions by one, so a slot whose first call here is a decode holds the state the worker that
    /// sampled the request's last token committed. Each value is a stream-ordered copy of one cell
    /// into the slot's element of its column, so no other slot's state, which only the device holds,
    /// is touched.
    ///
    /// # Errors
    ///
    /// Fails for a slot past the columns' slots, or when a copy cannot be issued.
    pub(crate) fn start(&mut self, start: SlotStart, stream: &platform::Stream) -> Result<()> {
        let slots = self.slots as usize;
        let slot = start.slot as usize;
        if slot >= slots {
            return Err(Error::Invalid(format!(
                "a slot start names slot {slot} of a commit state of {slots} slots"
            )));
        }
        let word = std::mem::size_of::<i32>();
        let values = [start.token, start.logical, start.sampling, start.verified];
        let flags = START_FIELDS * slots * word;
        // SAFETY: the cells written are this slot's, and a copy from them that may still be pending
        // is a previous start of the same slot, which a slot's next request cannot overtake: the
        // slot is started again only after the request that held it ended, its steps completed.
        unsafe {
            let cells = self.starts.as_mut_slice();
            for (field, value) in values.iter().enumerate() {
                let at = (field * slots + slot) * word;
                cells[at..at + word].copy_from_slice(&value.to_ne_bytes());
            }
            cells[flags + slot] = 1;
        }
        let base = self.starts.address();
        let targets = [&self.future, &self.logical, &self.sampling, &self.verified];
        for (field, column) in targets.into_iter().enumerate() {
            // SAFETY: the destination is element `slot` of an i32 column of `slots` elements, the
            // source this slot's cell of the pinned start buffer; both outlive the copy, which the
            // stream orders before the step that reads the slot.
            unsafe {
                platform::memcpy_async(
                    column.device.ptr.address() + (slot * word) as u64,
                    base + ((field * slots + slot) * word) as u64,
                    word,
                    stream,
                )?;
            }
        }
        // SAFETY: as above, for element `slot` of the u8 predicates.
        unsafe {
            platform::memcpy_async(
                self.predicates.device.ptr.address() + slot as u64,
                base + (flags + slot) as u64,
                1,
                stream,
            )?;
        }
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

/// One batch's logit adjustments ([`ipc::execution::LogitAdjustments`]) in the layout
/// `logits_shape` reads, encoded on the host into buffers sized when the runner binds.
///
/// Per row, two i32: whether the row adds its slot's bias row, and the index of its ban mask, `-1`
/// for a row that bans nothing. The ban masks are one bit per token (`ops::sampler::ban_words`
/// words a mask), one mask per row that bans any token, in row order, so a batch whose rows ban
/// nothing stages no mask at all. A padded row adds nothing and bans nothing.
#[derive(Default, Debug)]
pub struct Adjustments {
    /// `[rows, 2]` the rows' bias flags and mask indices, for the largest bucket.
    rows: Vec<i32>,
    /// `[rows, words]` room for one mask per row; the first `masks` are the batch's.
    bans: Vec<i32>,
    masks: usize,
    words: usize,
    vocabulary: u32,
    /// The bias rows the batch installs before it samples.
    installs: Vec<ipc::execution::SlotBias>,
}

impl Adjustments {
    /// Sizes the buffers for `rows` rows of a `vocabulary`-wide row.
    pub fn reserve(&mut self, rows: usize, vocabulary: u32) {
        self.words = ops::sampler::ban_words(vocabulary) as usize;
        self.vocabulary = vocabulary;
        self.rows = vec![0; 2 * rows];
        self.bans = vec![0; rows * self.words];
        self.masks = 0;
        self.installs = Vec::new();
    }

    /// Encodes the adjustments of a batch of `rows` rows, padded to `bucket_rows`.
    ///
    /// A banned token outside the vocabulary bans nothing: no logit carries it.
    pub fn stage(
        &mut self,
        adjustments: &ipc::execution::LogitAdjustments,
        rows: usize,
        bucket_rows: usize,
    ) -> Result<()> {
        if 2 * bucket_rows > self.rows.len() {
            return Err(Error::Invariant(format!(
                "the adjustments hold {} rows, the bucket {bucket_rows}",
                self.rows.len() / 2
            )));
        }
        self.masks = 0;
        for row in 0..bucket_rows {
            let biased = row < rows && adjustments.biased.get(row).copied().unwrap_or(false);
            self.rows[2 * row] = i32::from(biased);
            self.rows[2 * row + 1] = -1;
            let banned = match adjustments.banned.get(row) {
                Some(banned) if row < rows && !banned.is_empty() => banned,
                _ => continue,
            };
            let index = self.masks;
            let mask = &mut self.bans[index * self.words..(index + 1) * self.words];
            mask.fill(0);
            for &token in banned.iter().filter(|token| **token < self.vocabulary) {
                // Bit `t % 32` of word `t / 32`; the kernel reads the word as i32 and shifts it,
                // and bit 31 is the sign bit of the same word.
                mask[(token / 32) as usize] |= (1_u32 << (token % 32)) as i32;
            }
            self.rows[2 * row + 1] = i32::try_from(index)
                .map_err(|_| Error::Invariant("a ban mask index overflows i32".into()))?;
            self.masks += 1;
        }
        self.installs.clear();
        self.installs.extend_from_slice(&adjustments.installs);
        Ok(())
    }

    /// The `[rows, 2]` flags and mask indices of the first `rows` rows.
    pub fn rows(&self, rows: usize) -> &[i32] {
        &self.rows[..2 * rows]
    }

    /// The batch's ban masks, `[masks, words]`.
    pub fn bans(&self) -> &[i32] {
        &self.bans[..self.masks * self.words]
    }

    /// The bias rows the batch installs.
    pub fn installs(&self) -> &[ipc::execution::SlotBias] {
        &self.installs
    }
}

/// The sampling step's own device state: the slot histories its penalties read, and the slot
/// bias rows its logit adjustments add.
pub(crate) struct Sampling {
    pub(crate) sampler: Sampler,
    /// `[slots, vocabulary]` i32 counts of the tokens each slot has already seen, which the
    /// penalties subtract; zeroed once when the runner binds, started for each request that takes
    /// a slot ([`Self::start_counts`]) and advanced by `ops::commit`.
    pub(crate) counts: TensorView,
    pub(crate) _counts: platform::DeviceAllocation,
    /// `[slots, vocabulary]` f32 each slot's bias row: the logit bias of its request's biased
    /// tokens, `-inf` outside its allowed set, `0` elsewhere. Zeroed when the runner binds and
    /// written by [`Self::install`]; a row reads it only when the batch flags it, and the scheduler
    /// flags a request's rows only once a batch at or before them installed its row.
    pub(crate) bias: TensorView,
    _bias: platform::DeviceAllocation,
    /// The bias table's pinned host image, one row per slot, which an install writes and copies
    /// from. A slot's row is rewritten only by the next install of that slot, which belongs to a
    /// later request after the slot's previous request finished and its batches completed (or
    /// repeats the same request's identical row), so no copy in flight reads a row while it changes.
    mirror: platform::PinnedBuffer,
    /// The two halves a fresh slot's seeded count row is built in, taken by the batches in turn
    /// and grown to the most rows one batch has seeded ([`Self::start_counts`]).
    seeds: [Option<platform::PinnedBuffer>; 2],
    /// The half of `seeds` the next batch writes.
    seed_turn: usize,
    device: DeviceId,
    vocabulary: u32,
    slots: u32,
}

/// Writes into `row`, one slot's `[vocabulary]` i32 penalty counts, one count per occurrence of
/// each token of `history`: the counts the commits of those tokens leave on a slot that held the
/// request from its first token (`ops::commit` adds one per committed token). A token outside the
/// vocabulary counts nothing, as the commit counts nothing for one.
pub(crate) fn seed_row(row: &mut [i32], history: &[u32]) {
    row.fill(0);
    for &token in history {
        if let Some(count) = row.get_mut(token as usize) {
            *count += 1;
        }
    }
}

impl Sampling {
    /// The same state, its sampler reporting each scored row's `candidates` most likely tokens
    /// ([`Sampler::reporting`]).
    pub(crate) fn reporting(mut self, candidates: u32) -> Result<Self> {
        self.sampler = self.sampler.reporting(candidates)?;
        Ok(self)
    }

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
        // The bias table is as large as the count table: one f32 per slot and token. It starts at
        // zero, as the counts do, so a row read before any install would add nothing.
        let words = usize::try_from(bytes / 4)
            .map_err(|_| Error::Invalid("the sampler's bias table is larger than memory".into()))?;
        let bias_allocation =
            platform::DeviceAllocation::linear(device, words * std::mem::size_of::<f32>())?;
        // SAFETY: as for the counts: this allocation's own `slots × vocabulary` 32-bit words.
        unsafe {
            platform::memset_d32(bias_allocation.address(), 0, words, stream)?;
        }
        stream.synchronize()?;
        let bias = TensorView::new(
            ops::DevicePtr::new(bias_allocation.address()),
            DType::F32,
            ops::Shape::new(&[u64::from(slots), u64::from(vocabulary)]),
        );
        let mirror = platform::PinnedBuffer::new(device, words * std::mem::size_of::<f32>())?;
        Ok(Self {
            sampler,
            counts,
            _counts: allocation,
            bias,
            _bias: bias_allocation,
            mirror,
            seeds: [None, None],
            seed_turn: 0,
            device,
            vocabulary,
            slots,
        })
    }

    /// Starts the penalty counts of each fresh slot on `stream`, before the step that reads them:
    /// a request that takes a slot counts its own tokens and none of the slot's previous request.
    ///
    /// A slot whose request generated nothing yet (or counts nothing) is cleared by one device
    /// memset. A slot whose request already generated tokens elsewhere - on its prefill group
    /// before a hand-off, or before a migration or a preemption admitted it again - starts at
    /// those tokens' counts instead: the row is built in this batch's half of the seed staging
    /// and copied over the slot's row. Both are stream-ordered after the previous request's last
    /// commit, so nothing on the host waits for the device.
    pub(crate) fn start_counts(
        &mut self,
        fresh: &[ipc::execution::FreshSlot],
        stream: &platform::Stream,
    ) -> Result<()> {
        let width = self.vocabulary as usize;
        let row_bytes = width * std::mem::size_of::<i32>();
        if let Some(fresh) = fresh.iter().find(|fresh| fresh.slot >= self.slots) {
            return Err(Error::Invalid(format!(
                "a fresh slot {} of a count table of {} slots",
                fresh.slot, self.slots
            )));
        }
        let seeded = fresh
            .iter()
            .filter(|fresh| !fresh.history.is_empty())
            .count();
        // The batches take the two halves in turn: the half this batch writes was last read by the
        // copies of the batch before the previous one, which completed before this batch was
        // staged, as the runner's columns' halves do.
        let turn = self.seed_turn;
        self.seed_turn = 1 - turn;
        if seeded * row_bytes
            > self.seeds[turn]
                .as_ref()
                .map_or(0, platform::PinnedBuffer::len)
        {
            // A hand-off or a resumed request is rare, so the staging grows to the most rows a
            // batch has seeded rather than being reserved for every slot at bind.
            self.seeds[turn] = Some(platform::PinnedBuffer::new(
                self.device,
                seeded * row_bytes,
            )?);
        }
        let mut row_index = 0;
        for fresh in fresh {
            let offset = (fresh.slot as usize * row_bytes) as u64;
            let destination = self.counts.ptr.address() + offset;
            if fresh.history.is_empty() {
                // SAFETY: the range is slot `slot`'s own `vocabulary` i32 of the count table, and
                // the table outlives the stream's work on it (the runner owns both).
                unsafe {
                    platform::memset_d32(destination, 0, width, stream)?;
                }
                continue;
            }
            let staging = self.seeds[turn]
                .as_ref()
                .ok_or(Error::State("a seeded slot has no staging"))?;
            // SAFETY: the staging holds `seeded` rows of `vocabulary` i32 and `row_index` is below
            // `seeded`; no copy in flight reads this batch's half (the turn's invariant above).
            let row = unsafe {
                std::slice::from_raw_parts_mut(
                    staging.as_ptr().cast::<i32>().add(row_index * width),
                    width,
                )
            };
            seed_row(row, &fresh.history);
            // SAFETY: the source is the staging's row just written and the destination the slot's
            // row of the count table, both `vocabulary` i32; the staging row stays as written until
            // the copy has run (the turn's invariant).
            unsafe {
                platform::memcpy_async(
                    destination,
                    staging.address() + (row_index * row_bytes) as u64,
                    row_bytes,
                    stream,
                )?;
            }
            row_index += 1;
        }
        Ok(())
    }

    /// Writes each install's slot row of the bias table, on `stream`, before the step that reads
    /// it.
    ///
    /// The row is built in the slot's row of the pinned image - `-inf` everywhere when the request
    /// restricts its tokens and `0` otherwise, `0` at each allowed token, then each bias at a token
    /// the request may sample - and copied to the device row. The copy is stream-ordered before
    /// the replay that reads it, so nothing waits for it. A token outside the vocabulary is ignored.
    pub(crate) fn install(
        &mut self,
        installs: &[ipc::execution::SlotBias],
        stream: &platform::Stream,
    ) -> Result<()> {
        let width = self.vocabulary as usize;
        let row_bytes = width * std::mem::size_of::<f32>();
        for install in installs {
            if install.slot >= self.slots {
                return Err(Error::Invalid(format!(
                    "a bias row for slot {} of a table of {} slots",
                    install.slot, self.slots
                )));
            }
            let slot = install.slot as usize;
            // SAFETY: the image holds `slots × vocabulary` f32 and `slot` is below `slots`; no
            // copy in flight reads this row (the type's own invariant).
            let row = unsafe {
                std::slice::from_raw_parts_mut(
                    self.mirror.as_ptr().cast::<f32>().add(slot * width),
                    width,
                )
            };
            match &install.allowed {
                Some(allowed) => {
                    row.fill(f32::NEG_INFINITY);
                    for &token in allowed.iter().filter(|token| (**token as usize) < width) {
                        row[token as usize] = 0.0;
                    }
                }
                None => row.fill(0.0),
            }
            for &(token, bias) in install
                .bias
                .iter()
                .filter(|(token, _)| (*token as usize) < width)
            {
                let entry = &mut row[token as usize];
                if *entry != f32::NEG_INFINITY {
                    *entry = bias;
                }
            }
            let offset = (slot * row_bytes) as u64;
            // SAFETY: both ranges are the slot's own `vocabulary` f32 of tables of the same layout,
            // and the image's row stays as written until the copy has run (the type's invariant).
            unsafe {
                platform::memcpy_async(
                    self.bias.ptr.address() + offset,
                    self.mirror.address() + offset,
                    row_bytes,
                    stream,
                )?;
            }
        }
        Ok(())
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
        let samples = Samples::reserve(&arena, DType::I32, rows, 1_024, Some(4)).unwrap();
        let scores = samples.scores.unwrap();
        let top = scores.top.unwrap();
        let order = [
            candidates.shaped,
            candidates.values,
            candidates.ids,
            candidates.probabilities,
            candidates.ordered_values,
            candidates.ordered_ids,
            candidates.valid,
            samples.tokens,
            scores.logprobs,
            scores.ranks,
            top.values,
            top.ids,
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
        assert_eq!(top.values.shape.dims(), &[rows, 4]);
        assert_eq!(top.ids.dtype, DType::I32);
        // Without scores the step reserves none of them; scores without candidates reserve no top
        // tokens.
        let arena = Arena::new(DevicePtr::new(0x10_0000), 1 << 20);
        let _ = Candidates::reserve(&arena, rows, 1_024, 32).unwrap();
        let samples = Samples::reserve(&arena, DType::I32, rows, 1_024, None).unwrap();
        assert!(samples.scores.is_none());
        let samples = Samples::reserve(&arena, DType::I32, rows, 1_024, Some(0)).unwrap();
        assert!(samples.scores.unwrap().top.is_none());
    }

    /// A scored token is reported first, then its row's candidates in their order with their
    /// competition ranks - tied candidates share the rank of the first of them - and a candidate
    /// the row had no value for is not reported.
    #[test]
    fn a_scored_token_reports_itself_then_its_ranked_candidates() {
        let entries = scored(
            7,
            -1.5,
            3,
            &[-0.25, -1.0, -1.0, -1.5, f32::NEG_INFINITY],
            &[2, 9, 11, 7, 1_024],
            1_024,
        );
        let listed: Vec<(u32, f32, u32)> = entries
            .iter()
            .map(|entry| (entry.token_id, entry.logprob, entry.rank))
            .collect();
        assert_eq!(
            listed,
            vec![
                (7, -1.5, 3),
                (2, -0.25, 1),
                (9, -1.0, 2),
                (11, -1.0, 2),
                (7, -1.5, 4),
            ]
        );
        // Without candidates only the token is reported.
        assert_eq!(scored(5, -0.5, 1, &[], &[], 8).len(), 1);
    }

    /// A sampler reports at most as many candidates as its vocabulary has tokens.
    #[test]
    fn a_sampler_reports_no_more_candidates_than_its_vocabulary() {
        let sampler = Sampler::new(1_024, 4, 64, 0)
            .unwrap()
            .reporting(20)
            .unwrap();
        assert_eq!(sampler.candidates(), 20);
        assert!(Sampler::new(16, 4, 16, 0).unwrap().reporting(17).is_err());
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
            bias: view(&[4, 1_024]),
            adjustments: ints(&[2, 2]),
            bans: ints(&[1, 32]),
        };
        let samples = Samples {
            tokens: TensorView::new(DevicePtr::new(0x2000), DType::I32, Shape::new(&[2])),
            scores: Some(Scores {
                logprobs: TensorView::new(DevicePtr::new(0x2000), DType::F32, Shape::new(&[2])),
                ranks: ints(&[2]),
                top: None,
            }),
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
            bias: view(&[4, 1_024]),
            adjustments: ints(&[2, 2]),
            bans: ints(&[1, 32]),
        };
        assert!(sampler.check(view(&[2, 1_024]), &narrow, &samples).is_err());
        let miscounted = SamplerState {
            counts: ints(&[3, 1_024]),
            parameters: view(&[2, u64::from(PARAMETER_COLUMNS)]),
            slots: ints(&[2]),
            extras: Some(ints(&[2, 2])),
            bias: view(&[4, 1_024]),
            adjustments: ints(&[2, 2]),
            bans: ints(&[1, 32]),
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

    /// A batch's adjustments reach the shaper as one flag pair per row and one ban mask per row that
    /// bans a token: bit `t % 32` of word `t / 32`, the sign bit included; a padded row, a token past
    /// the vocabulary and a row that bans nothing set nothing.
    #[test]
    fn a_batch_stages_one_ban_mask_per_banning_row() {
        let mut adjustments = Adjustments::default();
        adjustments.reserve(4, 100);
        let batch = ipc::execution::LogitAdjustments {
            installs: Vec::new(),
            biased: vec![true, false, true],
            banned: vec![vec![0, 31, 99, 100], Vec::new(), vec![32]],
        };
        adjustments.stage(&batch, 3, 4).unwrap();
        assert_eq!(adjustments.rows(4), &[1, 0, 0, -1, 1, 1, 0, -1]);
        let words = ops::sampler::ban_words(100) as usize;
        assert_eq!(words, 4);
        let bans = adjustments.bans();
        assert_eq!(bans.len(), 2 * words);
        assert_eq!(bans[0], 1 | i32::MIN, "tokens 0 and 31 share word 0");
        assert_eq!(
            bans[3],
            1 << 3,
            "token 99 is bit 3 of word 3, and 100 is outside"
        );
        assert_eq!(&bans[words..], &[0, 1, 0, 0], "token 32 is bit 0 of word 1");

        // A batch that adjusts nothing stages no mask and flags no row.
        adjustments
            .stage(&ipc::execution::LogitAdjustments::default(), 3, 4)
            .unwrap();
        assert!(adjustments.bans().is_empty());
        assert_eq!(adjustments.rows(4), &[0, -1, 0, -1, 0, -1, 0, -1]);
        // A bucket wider than the reservation is refused.
        assert!(adjustments.stage(&batch, 3, 5).is_err());
    }

    /// A fresh slot's seeded counts are its request's history as the commits count it: one per
    /// occurrence, nothing for a token outside the vocabulary, and nothing left of the row's
    /// previous request.
    #[test]
    fn a_seeded_count_row_counts_each_occurrence_of_its_history() {
        let mut row = vec![5_i32; 8];
        seed_row(&mut row, &[3, 1, 3, 9, 3]);
        assert_eq!(row, vec![0, 1, 0, 3, 0, 0, 0, 0]);
        seed_row(&mut row, &[]);
        assert_eq!(row, vec![0; 8]);
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
            bias: view(&[4, 1_024]),
            adjustments: ints(&[2, 2]),
            bans: ints(&[1, 32]),
        };
        let samples = Samples {
            tokens: TensorView::new(DevicePtr::new(0x2000), DType::I32, Shape::new(&[2])),
            scores: None,
        };
        assert_eq!(
            sampler.check(view(&[2, 1_024]), &state, &samples).unwrap(),
            2
        );
        // The tokens a greedy step writes are whatever type the sampler declares, and a step whose
        // output is a different one is refused by name rather than launched into a wrong view.
        let narrow = Samples {
            tokens: TensorView::new(DevicePtr::new(0x2000), DType::I64, Shape::new(&[2])),
            scores: None,
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

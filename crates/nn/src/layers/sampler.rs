//! Sampling: the request's penalties, its candidate filtering, its draw, and the log-probabilities
//! it reports.
//!
//! One sampling step is a fixed sequence of operator launches over one row per active request
//! (specs/worker.md 9, "采样"; `uniserve/sampling/top_k.py` today):
//!
//! ```text
//! logits [rows, vocabulary]        → logits_shape   → shaped [rows, vocabulary] f32
//! shaped                           → log_softmax   → logprobs [rows, vocabulary] f32  (when asked)
//! shaped                           → top_k          → values [rows, width] f32, ids [rows, width] i32
//! values + ids                     → filter_candidates → values (dropped = -inf), valid [rows]
//! values + ids                     → candidate_order → the same rows in ascending token id
//! ordered values                   → softmax        → probabilities [rows, width] f32
//! probabilities + draws            → inverse_cdf    → tokens [rows]
//! logprobs + tokens                → logprob_gather → logp [rows] f32                    (when asked)
//! ```
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
//! The layer never generates randomness. `draws` is `[rows]` f32 in `[0, 1)`, computed by the
//! executor from the request's seed and the token position it is sampling (Philox, addressed per
//! element), so a restored or replayed step draws the same number.
//!
//! ## Greedy rows
//!
//! A row whose temperature is zero is **not** a draw: it is [`Self::argmax`]'s row. The shaper
//! divides such a row by one rather than by zero (`ops::sampler`'s doc cites the rule), so the row's
//! shaped logits keep the logits' order and its largest candidate is the temperature-zero token. The
//! temperature of each row is a request parameter, so the executor knows which rows are greedy and
//! routes them: `argmax` over the row's logits for those, this path's draw for the rest. No draw can
//! replace that step, because the draw path's candidates are sorted by token id before the CDF (the
//! step the reference takes at `uniserve/sampling/top_k.py:307-314`), so a draw of zero selects the
//! lowest token id rather than the largest logit.
//!
//! ## Rows that cannot be sampled
//!
//! `filter_candidates` reports, per row, whether the row's kept candidates form a usable
//! distribution (a row can end up empty when a nucleus or min-p threshold and a top-k width leave
//! nothing). The layer does not branch on it — the value is device data and the layer is inside a
//! captured graph — but leaves it in the candidate working set's `valid` row for the executor, and
//! samples the row as its first candidate, which is deterministic and reproducible. Whether such a
//! request continues, is retried with different parameters, or fails is the executor's decision, not
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
    OrderDescriptor, PARAMETER_COLUMNS, ShapeArgs, ShapeDescriptor, TopK, TopKArgs, TopKDescriptor,
};
use ops::{DType, Op, Shape, TensorView};

use crate::error::{Error, Result};
use crate::forward::{Arena, Forward};
use crate::layers::launch;

/// The request state a sampling step reads.
///
/// The executor owns it and writes it in the layout the operators read: one row of parameters per
/// row being sampled, the slot whose history penalizes the row, and the slot's counts.
pub struct SamplerState {
    /// `[slots, vocabulary]` i32: how many times each slot has already seen each token.
    pub counts: TensorView,
    /// `[rows, PARAMETER_COLUMNS]` f32: temperature, top-k width, top-p, min-p, repetition,
    /// frequency and presence in [`ops::sampler::PARAMETER_COLUMNS`] order.
    ///
    /// That order is the operators' and the executor writes it. The reference's own table
    /// (`uniserve/sampling/top_k.py:41-43`) is `temperature, top-p, min-p, typical-p, repetition,
    /// frequency, presence, top-k`, so it agrees on the four columns this layer's shaper reads
    /// (0, 4, 5, 6) and differs on the three its cutoffs read: our 1 is the top-k width where the
    /// reference's 1 is the nucleus threshold, our 2 is that threshold where its 2 is min-p, and our
    /// 3 is min-p where its 3 is typical-p. Aligning ours to it would therefore turn a top-k width
    /// into a nucleus threshold and a nucleus threshold into a min-p threshold on every request,
    /// which no shape check can see. This doc and `Sampler::check` are the only two places in this
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
    /// `[rows, width]` f32 the filtered candidates in ascending token id order.
    pub ordered_values: TensorView,
    /// `[rows, width]` i32 the ids of `ordered_values`, in the same order.
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
            return Err(Error::shape(
                "Sampler",
                "a sampler needs a non-empty vocabulary".to_owned(),
            ));
        }
        if slots == 0 {
            return Err(Error::shape(
                "Sampler",
                "a sampler needs at least one row".to_owned(),
            ));
        }
        if width == 0 || width > vocabulary {
            return Err(Error::shape(
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
    /// `[0, 1)` from the executor. The step's working set comes from the forward's arena, so the
    /// call is inside a captured bucket's plan.
    ///
    /// The candidates reach [`InverseCdf`] in ascending token id order: [`TopK`] writes them by
    /// descending logit, [`FilterCandidates`] preserves that, and [`CandidateOrder`] permutes both
    /// the values and the ids into token order, which is what makes a fixed draw select the same
    /// token as the reference's (`uniserve/sampling/top_k.py:307-314`). Greedy rows do not go through
    /// this method at all — see [`Self::greedy`].
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
            return Err(Error::shape(
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
        // candidates in: the reference reorders them by token id before its inverse CDF
        // (`uniserve/sampling/top_k.py:307-314`), and a fixed draw selects the same token only when
        // the columns mean the same thing on both sides.
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

        // 5. The reported log-probabilities, over the shaped distribution of the whole vocabulary.
        if let Some(logprobabilities) = samples.logprobabilities {
            let softmax = self.log_softmax_descriptor();
            let logprobs = arena.tensor(DType::F32, Shape::new(&[rows, vocabulary]))?;
            let args = LogSoftmaxArgs {
                x: candidates.shaped,
                out: logprobs,
            };

            f.operate::<LogSoftmax>(&softmax, args)?;

            let gather = self.logprob_descriptor();
            let args = LogprobArgs {
                logprobabilities: logprobs,
                tokens: samples.tokens,
                out: logprobabilities,
            };

            f.operate::<LogprobGather>(&gather, args)?;
        }

        Ok(())
    }

    /// The descriptors this layer launches with, each built in exactly one place.
    ///
    /// Every field is layer geometry or a fixed element type — the vocabulary, the candidate width,
    /// the row block, the token width — so a request space enumerates them without a stream, an
    /// arena or a batch, and both sampling paths launch these. The row count of a draw is a launch
    /// value and appears in none of them.
    ///
    /// Penalties and temperature over the whole vocabulary: `logits`' element type is the one thing
    /// a deployment chooses.
    pub fn shape_descriptor(&self, logits: DType) -> ShapeDescriptor {
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
    pub fn top_descriptor(&self) -> TopKDescriptor {
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
    pub fn filter_descriptor(&self) -> FilterDescriptor {
        FilterDescriptor {
            width: self.width,
            tile: CANDIDATE_TILE,
            ids: DType::I32,
            pdl: launch::PDL,
        }
    }

    /// The candidates reordered by token id, so a draw does not depend on the order the top-k
    /// program kept.
    pub fn order_descriptor(&self) -> OrderDescriptor {
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
    pub fn softmax_descriptor(&self) -> SoftmaxDescriptor {
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
    pub fn draw_descriptor(&self) -> InverseCdfDescriptor {
        InverseCdfDescriptor {
            width: self.width,
            tile: CANDIDATE_TILE,
            ids: DType::I32,
            tokens: self.tokens,
            pdl: launch::PDL,
        }
    }

    /// The greedy path's argmax over the shaped row.
    pub fn argmax_descriptor(&self) -> ArgMaxDescriptor {
        ArgMaxDescriptor {
            dtype: DType::F32,
            width: self.vocabulary,
            ids: self.tokens,
            pdl: launch::PDL,
        }
    }

    /// The reported log-probabilities over the shaped distribution of the whole vocabulary.
    pub fn log_softmax_descriptor(&self) -> LogSoftmaxDescriptor {
        LogSoftmaxDescriptor {
            dtype: DType::F32,
            columns: self.vocabulary,
            tile: self.block,
            pdl: launch::PDL,
        }
    }

    /// The reported log-probability of the token each row drew.
    pub fn logprob_descriptor(&self) -> LogprobDescriptor {
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
            return Err(Error::shape(
                "Sampler",
                "a sampling step has at least one row".to_owned(),
            ));
        }
        if !matches!(logits.dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::shape(
                "Sampler",
                format!(
                    "{} logits are not a shape the shaper reads: f32, bf16 or f16",
                    logits.dtype
                ),
            ));
        }
        if samples.tokens.dtype != self.tokens {
            return Err(Error::shape(
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
                return Err(Error::shape(
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
            return Err(Error::shape(
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
            return Err(Error::shape(
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
                return Err(Error::shape(
                    "Sampler",
                    "the sampler reads no extra tokens and the state carries some".to_owned(),
                ));
            }
            (_, Some(_)) => {}
            (extras, None) => {
                return Err(Error::shape(
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
                return Err(Error::shape(
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
    /// This is the reference's greedy step (`torch.max` over the shaped row, `uniserve/sampling/
    /// greedy.py`), for the rows whose temperature is zero. It runs the same shaper as [`Self::forward`]
    /// and nothing else: no candidate set, no cutoffs, no draw. It is a separate launch sequence
    /// rather than a branch inside `forward`, because the executor's buckets are homogeneous in the
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

        if let Some(logprobabilities) = samples.logprobabilities {
            let softmax = self.log_softmax_descriptor();
            let logprobs = arena.tensor(DType::F32, Shape::new(&[rows, vocabulary]))?;
            let args = LogSoftmaxArgs {
                x: shaped,
                out: logprobs,
            };

            f.operate::<LogSoftmax>(&softmax, args)?;

            let gather = self.logprob_descriptor();
            let args = LogprobArgs {
                logprobabilities: logprobs,
                tokens: samples.tokens,
                out: logprobabilities,
            };

            f.operate::<LogprobGather>(&gather, args)?;
        }

        Ok(())
    }

    /// The bound the sampled tokens' distributions are compared with: the shaping and filtering
    /// steps round FP32 values a few times, while a chosen token is exact.
    pub fn tolerance(&self) -> ops::Tolerance {
        LogitsShape::tolerance(&ShapeDescriptor {
            dtype: DType::F32,
            vocabulary: self.vocabulary,
            block: self.block,
            extras: self.extras,
            divide: true,
            pdl: launch::PDL,
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
            assert_eq!(view.ptr.address() % crate::forward::ALIGNMENT, 0);
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

    #[test]
    fn the_bound_is_the_shaping_steps_own_roundings() {
        let sampler = Sampler::new(1_024, 8, 64, 0).unwrap();
        let bound = sampler.tolerance();
        assert!(
            bound.rtol > 0.0 && bound.rtol < 1.0e-6,
            "rtol = {}",
            bound.rtol
        );
        assert!(bound.allows(1.0, 1.0 + 1.0e-7));
        assert!(!bound.allows(1.0, 1.0 + 1.0e-3));
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

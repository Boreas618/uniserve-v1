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
//! values                           → filter_candidates → values (dropped = -inf), valid [rows]
//! values                           → softmax        → probabilities [rows, width] f32
//! probabilities + draws            → inverse_cdf    → tokens [rows]
//! logprobs + tokens                → logprob_gather → logp [rows] f32                    (when asked)
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
//! element), so a restored or replayed step draws the same number. A row without a positive
//! temperature is sampled by the same path with a draw of `0`, which selects its largest candidate —
//! the greedy token — without a second code path.
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

use ops::norm::{Softmax, SoftmaxArgs, SoftmaxDescriptor};
use ops::sampler::{
    ArgMax, ArgMaxArgs, ArgMaxDescriptor, FilterArgs, FilterCandidates, FilterDescriptor,
    InverseCdf, InverseCdfArgs, InverseCdfDescriptor, LogSoftmax, LogSoftmaxArgs,
    LogSoftmaxDescriptor, LogitsShape, LogprobArgs, LogprobDescriptor, LogprobGather,
    PARAMETER_COLUMNS, ShapeArgs, ShapeDescriptor, TopK, TopKArgs, TopKDescriptor,
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
    /// frequency and presence in [`ops::logits::PARAMETER_COLUMNS`] order.
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
    /// `[rows, width]` f32 the candidates' probabilities after filtering.
    pub probabilities: TensorView,
    /// `[rows]` i32: 1 when the row's candidates form a usable distribution.
    pub valid: TensorView,
}

impl Candidates {
    /// Reserve the working set of `rows` rows from `arena`, in the order the operators read it.
    pub fn reserve(arena: &Arena, rows: u64, vocabulary: u64, width: u64) -> Result<Self> {
        Ok(Self {
            shaped: arena.tensor(DType::F32, Shape::new(&[rows, vocabulary]))?,
            values: arena.tensor(DType::F32, Shape::new(&[rows, width]))?,
            ids: arena.tensor(DType::I32, Shape::new(&[rows, width]))?,
            probabilities: arena.tensor(DType::F32, Shape::new(&[rows, width]))?,
            valid: arena.tensor(DType::I32, Shape::new(&[rows]))?,
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
        draws
            .require_shape("Sampler", Shape::new(&[rows]))
            .map_err(Error::from)?;
        if draws.dtype != DType::F32 {
            return Err(Error::shape(
                "Sampler",
                format!("expected f32 draws, found {}", draws.dtype),
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
            (0, None) | (_, Some(_)) => {}
            (extras, None) => {
                return Err(Error::shape(
                    "Sampler",
                    format!("the sampler reads {extras} extra tokens and the state has none"),
                ));
            }
        }
        if let Some(extras) = state.extras {
            extras
                .require_shape("Sampler", Shape::new(&[rows, u64::from(self.extras)]))
                .map_err(Error::from)?;
            if extras.dtype != DType::I32 {
                return Err(Error::shape(
                    "Sampler",
                    format!("expected i32 extra tokens, found {}", extras.dtype),
                ));
            }
        }

        let arena = f.arena();
        let candidates = Candidates::reserve(arena, rows, vocabulary, width)?;

        // 1. Penalties and temperature.
        let shape = ShapeDescriptor {
            dtype: logits.dtype,
            vocabulary: self.vocabulary,
            block: self.block,
            extras: self.extras,
            divide: true,
            pdl: launch::PDL,
        };
        let args = ShapeArgs {
            logits,
            counts: state.counts,
            slots: state.slots,
            parameters: state.parameters,
            extras: state.extras,
            out: candidates.shaped,
        };
        let kernel = f.kernels().op::<LogitsShape>(&shape)?;
        LogitsShape::launch(kernel, &shape, args, f.stream)?;

        // 2. The candidate set: the `width` largest, then the row's own thresholds.
        let top = TopKDescriptor {
            dtype: DType::F32,
            vocabulary: self.vocabulary,
            width: self.width,
            ids: DType::I32,
            pdl: launch::PDL,
        };
        let args = TopKArgs {
            logits: candidates.shaped,
            values: candidates.values,
            ids: candidates.ids,
        };
        let kernel = f.kernels().op::<TopK>(&top)?;
        TopK::launch(kernel, &top, args, f.stream)?;

        let filter = FilterDescriptor {
            width: self.width,
            ids: DType::I32,
            pdl: launch::PDL,
        };
        let args = FilterArgs {
            values: candidates.values,
            ids: candidates.ids,
            parameters: state.parameters,
            valid: candidates.valid,
        };
        let kernel = f.kernels().op::<FilterCandidates>(&filter)?;
        FilterCandidates::launch(kernel, &filter, args, f.stream)?;

        // 3. The candidates' distribution, then one draw per row.
        let softmax = SoftmaxDescriptor {
            dtype: DType::F32,
            columns: self.width,
            tile: launch::row_tile(self.width),
            pdl: launch::PDL,
        };
        let args = SoftmaxArgs {
            x: candidates.values,
            out: candidates.probabilities,
        };
        let kernel = f.kernels().op::<Softmax>(&softmax)?;
        Softmax::launch(kernel, &softmax, args, f.stream)?;

        let draw = InverseCdfDescriptor {
            width: self.width,
            ids: DType::I32,
            tokens: self.tokens,
            pdl: launch::PDL,
        };
        let args = InverseCdfArgs {
            probabilities: candidates.probabilities,
            ids: candidates.ids,
            draws,
            tokens: samples.tokens,
        };
        let kernel = f.kernels().op::<InverseCdf>(&draw)?;
        InverseCdf::launch(kernel, &draw, args, f.stream)?;

        // 4. The reported log-probabilities, over the shaped distribution of the whole vocabulary.
        if let Some(logprobabilities) = samples.logprobabilities {
            let softmax = LogSoftmaxDescriptor {
                dtype: DType::F32,
                columns: self.vocabulary,
                tile: self.block,
                pdl: launch::PDL,
            };
            let logprobs = arena.tensor(DType::F32, Shape::new(&[rows, vocabulary]))?;
            let args = LogSoftmaxArgs {
                x: candidates.shaped,
                out: logprobs,
            };
            let kernel = f.kernels().op::<LogSoftmax>(&softmax)?;
            LogSoftmax::launch(kernel, &softmax, args, f.stream)?;

            let gather = LogprobDescriptor {
                columns: self.vocabulary,
                ids: self.tokens,
                pdl: launch::PDL,
            };
            let args = LogprobArgs {
                logprobabilities: logprobs,
                tokens: samples.tokens,
                out: logprobabilities,
            };
            let kernel = f.kernels().op::<LogprobGather>(&gather)?;
            LogprobGather::launch(kernel, &gather, args, f.stream)?;
        }

        Ok(())
    }

    /// The greedy token of each row of `values`: the index of the largest candidate.
    ///
    /// A row without a positive temperature is sampled through the same draw path with a draw of
    /// zero, which selects its first candidate; this call is for diagnostics and for a caller that
    /// wants the index rather than the token.
    pub fn argmax(&self, f: &Forward, values: TensorView, out: TensorView) -> Result<()> {
        let rows = values.rows();
        values
            .require_row("Sampler", values.dtype, u64::from(self.width))
            .map_err(Error::from)?;
        out.require_shape("Sampler", Shape::new(&[rows]))
            .map_err(Error::from)?;
        let descriptor = ArgMaxDescriptor {
            dtype: values.dtype,
            width: self.width,
            ids: DType::I32,
            pdl: launch::PDL,
        };
        let args = ArgMaxArgs { values, out };
        let kernel = f.kernels().op::<ArgMax>(&descriptor)?;
        ArgMax::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
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
}

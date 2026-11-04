//! How far apart two implementations of one operator may be.
//!
//! Shadow execution runs the old and the new kernel of an operator on the same inputs and compares
//! their outputs (specs/worker.md 7.3). The tolerance is the operator's, not the program's: it says
//! how much a *correct* implementation may differ from another correct one, so the bound follows
//! from the operator's mathematics — its reduction length, its accumulation dtype and its output
//! dtype — and never from a measurement of the pair being compared.
//!
//! ## The bound
//!
//! For a sum of `n` terms evaluated with round-to-nearest in a dtype with unit round-off `u`,
//! floating-point arithmetic gives
//!
//! ```text
//! |fl(Σ aᵢbᵢ) − Σ aᵢbᵢ| ≤ γₙ Σ|aᵢbᵢ|,   γₙ = n·u / (1 − n·u)
//! ```
//!
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1). Two implementations
//! therefore differ by at most `2γₙ Σ|aᵢbᵢ|`, and once the result is stored in the output dtype one
//! more rounding of `u_out` follows, giving
//!
//! ```text
//! |new − old| ≤ 2γₙ Σ|aᵢbᵢ| + u_out·|result|.
//! ```
//!
//! That is the bound this type is written in, and it is always relative to a magnitude:
//!
//! - [`Tolerance::rounding`] is for a reduction whose result is a *scale* of its inputs — a sum of
//!   squares for a norm, a sum of positive weights — where `Σ|aᵢbᵢ|` and the result are the same
//!   order, so the bound is relative to the result itself with factor `2γₙ`.
//! - [`Tolerance::dot`] is for a reduction of mixed-sign products — a matrix product, a
//!   convolution, an attention score — where the result can cancel and `Σ|aᵢbᵢ|` can exceed it.
//!   For summed products of independent signs, `|result| ≈ √n·σ` and `Σ|aᵢbᵢ| ≈ n·σ`, so the bound
//!   relative to the result is `2γₙ√n`; this is the standard statistical reading of the same
//!   inequality, not a relaxation of it. A row that cancels harder than that is not covered by any
//!   data-free bound, and an operator that has such rows declares an envelope with
//!   [`Tolerance::with_atol`] and says in its documentation what data the envelope is for.
//! - [`Tolerance::exact`] is for an operator that only moves values: a gather, a scatter, a sort of
//!   indices. Two implementations of a copy agree bit for bit.
//!
//! The comparison is `|candidate − reference| ≤ atol + rtol·|reference|`.

use crate::tensor::DType;

/// How far one implementation of an operator may be from another.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Tolerance {
    /// The bound that does not scale with the reference value.
    pub atol: f32,
    /// The bound relative to the reference value.
    pub rtol: f32,
}

impl Tolerance {
    /// Two implementations that move values agree bit for bit.
    pub const fn exact() -> Self {
        Self {
            atol: 0.0,
            rtol: 0.0,
        }
    }

    /// A bound stated directly.
    pub const fn new(atol: f32, rtol: f32) -> Self {
        Self { atol, rtol }
    }

    /// `2γₙ(u) + u_out`: two implementations of a reduction whose result is the scale of its
    /// inputs, with the reduction accumulated in `accumulate` and rounded once to `output`.
    pub fn rounding(reduction: u64, accumulate: DType, output: DType) -> Self {
        Self {
            atol: 0.0,
            rtol: relative(2.0 * gamma(reduction, accumulate) + output.unit_roundoff()),
        }
    }

    /// `2γₙ(u)√n + u_out`: two implementations of a sum of mixed-sign products, with the reduction
    /// accumulated in `accumulate` and rounded once to `output`.
    pub fn dot(reduction: u64, accumulate: DType, output: DType) -> Self {
        let reduction = reduction as f64;
        Self {
            atol: 0.0,
            rtol: relative(
                2.0 * gamma(reduction as u64, accumulate) * reduction.sqrt()
                    + output.unit_roundoff(),
            ),
        }
    }

    /// The same bound with an envelope for rows that cancel harder than [`Tolerance::dot`] covers.
    ///
    /// The envelope is a declaration about the data an operator sees, not something derivable from
    /// the descriptor; the operator's documentation says which data it is for.
    pub const fn with_atol(self, atol: f32) -> Self {
        Self { atol, ..self }
    }

    /// Whether `candidate` is within this bound of `reference`.
    pub fn allows(&self, reference: f32, candidate: f32) -> bool {
        let difference = (f64::from(candidate) - f64::from(reference)).abs();
        let bound = f64::from(self.atol) + f64::from(self.rtol) * f64::from(reference).abs();
        difference <= bound
    }
}

/// `γₙ = n·u/(1 − n·u)` for a length-`n` reduction accumulated in `accumulate`.
///
/// Saturates at 1 for a reduction so long that `n·u ≥ 1`, where the expression stops being a bound;
/// no operator here reduces over enough terms for that to happen in FP32 (`n` would be 2²⁴).
pub fn gamma(reduction: u64, accumulate: DType) -> f64 {
    let terms = reduction as f64 * accumulate.unit_roundoff();
    if terms >= 1.0 {
        1.0
    } else {
        terms / (1.0 - terms)
    }
}

/// An `f64` bound as the `f32` a tolerance is stored in.
fn relative(value: f64) -> f32 {
    value as f32
}

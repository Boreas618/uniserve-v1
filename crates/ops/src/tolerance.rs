//! How far apart two implementations of one operator may be.
//!
//! Shadow execution runs the old and the new kernel of an operator on the same inputs and compares
//! their outputs. The tolerance is the operator's, not the program's: it says
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
//!   inequality, not a relaxation of it. One element of such an output can cancel far below its
//!   own `Σ|aᵢbᵢ|`, so the bound is taken against the largest magnitude of the output it belongs
//!   to, which is the same order as that sum, rather than against the element's own value.
//! - [`Tolerance::exact`] is for an operator that only moves values: a gather, a scatter, a sort of
//!   indices. Two implementations of a copy agree bit for bit.
//!
//! The comparison is `|candidate − reference| ≤ atol + rtol·magnitude`, where the magnitude is
//! `|reference|`, or the output's largest `|reference|` for a bound that [`Tolerance::cancels`].

use uniserve_tensor::DType;

/// How far one implementation of an operator may be from another.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Tolerance {
    /// The bound that does not scale with the reference value.
    pub atol: f32,
    /// The bound relative to the reference value.
    pub rtol: f32,
    /// Whether an element may cancel below its own terms, so that `rtol` is relative to the largest
    /// magnitude of the output and not to the element.
    pub cancels: bool,
}

impl Tolerance {
    /// Two implementations that move values agree bit for bit.
    pub const fn exact() -> Self {
        Self {
            atol: 0.0,
            rtol: 0.0,
            cancels: false,
        }
    }

    /// A bound stated directly.
    pub const fn new(atol: f32, rtol: f32) -> Self {
        Self {
            atol,
            rtol,
            cancels: false,
        }
    }

    /// `2γₙ(u) + u_out`: two implementations of a reduction whose result is the scale of its
    /// inputs, with the reduction accumulated in `accumulate` and rounded once to `output`.
    pub fn rounding(reduction: u64, accumulate: DType, output: DType) -> Self {
        Self {
            atol: 0.0,
            rtol: relative(2.0 * gamma(reduction, accumulate) + unit_roundoff(output)),
            cancels: false,
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
                    + unit_roundoff(output),
            ),
            cancels: true,
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
        self.allows_within(reference.abs(), reference, candidate)
    }

    /// Whether `candidate` is within this bound of `reference`, an element of an output whose
    /// largest magnitude is `peak`.
    pub fn allows_within(&self, peak: f32, reference: f32, candidate: f32) -> bool {
        let difference = (f64::from(candidate) - f64::from(reference)).abs();
        let magnitude = if self.cancels { peak } else { reference.abs() };
        let bound = f64::from(self.atol) + f64::from(self.rtol) * f64::from(magnitude);
        difference <= bound
    }
}

/// The unit round-off `u` of an element type: half the gap between one and the next value above
/// it, which is the factor every rounding bound is written with (Higham, *Accuracy and Stability
/// of Numerical Algorithms*, 2nd ed., §2.2). An integer rounds nothing.
pub fn unit_roundoff(dtype: DType) -> f64 {
    match dtype {
        DType::F64 => 1.110_223_024_625_156_5e-16, // 2^-53
        DType::F32 => 5.960_464_477_539_063e-8,    // 2^-24
        DType::BF16 => 3.906_25e-3,                // 2^-8
        DType::F16 => 4.882_812_5e-4,              // 2^-11
        DType::F8E4M3 | DType::F6E2M3 => 6.25e-2,  // 2^-4
        DType::F6E3M2 => 1.25e-1,                  // 2^-3
        DType::F4 => 2.5e-1,                       // 2^-2
        DType::F8E8M0 => 5e-1,                     // 2^-1: a power of two and nothing between
        DType::U8 | DType::U32 | DType::I16 | DType::I32 | DType::I64 => 0.0,
        // An element type the tensor library adds has no bound until it is stated here: a NaN
        // bound holds for no pair of values, so a comparison under it fails instead of passing.
        _ => f64::NAN,
    }
}

/// The element type a reduction over `dtype` accumulates in.
///
/// Every operator here accumulates a floating reduction in FP32, so the bounds are written with
/// `u = 2^-24` whatever the input dtype is; FP64 alone accumulates in itself, because FP32 would
/// lose what its elements hold.
pub const fn accumulate(dtype: DType) -> DType {
    match dtype {
        DType::F64 => DType::F64,
        _ => DType::F32,
    }
}

/// `γₙ = n·u/(1 − n·u)` for a length-`n` reduction accumulated in `accumulate`.
///
/// Saturates at 1 for a reduction so long that `n·u ≥ 1`, where the expression stops being a bound;
/// no operator here reduces over enough terms for that to happen in FP32 (`n` would be 2²⁴).
pub fn gamma(reduction: u64, accumulate: DType) -> f64 {
    let terms = reduction as f64 * unit_roundoff(accumulate);
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

//! Diffusion timestep conditioning: the sinusoidal embedding of a time and the MLP that projects it.
//!
//! [`sinusoidal`] is host arithmetic over the schedule's times; [`TimestepEmbedding`] is the
//! `Linear → SiLU → Linear` pair a DiT applies to it (`TimestepEmbedding` of diffusers and
//! `TimestepEmbedder` of the multimodal references).
//!
//! The embedding keeps the arithmetic the shipped baseline uses, which diverges from the references
//! in its precision: the angles are evaluated in f64 and each cosine and sine rounded once to f32,
//! where the references compute in f32. The row is every cosine and then every sine of the same
//! angles (the GLIDE/DiT order, not an interleaving).

use ops::Operand;

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::{Bias, Linear, Parallel, Silu};
use crate::weights::Scope;

/// The base of the frequency ladder: `10000^(-i / half)`.
const MAX_PERIOD: f32 = 10_000.0;

/// The sinusoidal embedding of each time, `[times, dim]` FP32 row-major.
///
/// Row `r`, element `i` is `cos(t_r · f_i)` for `i < half` and the sine of the same angle for
/// `i ≥ half`, `half = dim / 2`, with `f_i = exp(-ln(10000) · i / half)`. Every step is FP32, as the
/// diffusion references evaluate it (`torch.exp(-math.log(10000) * torch.arange(half,
/// dtype=float32) / half)`, then `t * freqs` and the cosine and sine in FP32).
///
/// # Errors
///
/// No time, or a width with no cosine half.
pub fn sinusoidal(times: &[f32], dim: u32) -> Result<Vec<f32>> {
    if times.is_empty() {
        return Err(Error::shape(
            "sinusoidal",
            "an embedding needs at least one time".to_owned(),
        ));
    }
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(Error::shape(
            "sinusoidal",
            format!("a {dim}-wide embedding has no cosine half"),
        ));
    }
    let half = dim / 2;
    let log = MAX_PERIOD.ln();
    let frequencies: Vec<f32> = (0..half)
        .map(|index| (-log * index as f32 / half as f32).exp())
        .collect();
    let mut values = Vec::with_capacity(times.len() * dim as usize);
    for time in times {
        let angles: Vec<f32> = frequencies
            .iter()
            .map(|frequency| time * frequency)
            .collect();
        values.extend(angles.iter().map(|angle| angle.cos()));
        values.extend(angles.iter().map(|angle| angle.sin()));
    }
    Ok(values)
}

/// The widths of a [`TimestepEmbedding`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestepSpec {
    /// The sinusoidal embedding's width.
    pub frequency_dim: u32,
    /// The hidden width between the two projections.
    pub hidden: u32,
    /// The output width.
    pub out: u32,
}

/// `linear_2(silu(linear_1(x)))` over a sinusoidal embedding.
pub struct TimestepEmbedding {
    first: Linear,
    second: Linear,
}

impl TimestepEmbedding {
    /// The pair the module at `scope` states as `linear_1` and `linear_2`, each with its bias.
    ///
    /// A model that keeps the embedding in a representation of its own pins `scope`.
    ///
    /// # Errors
    ///
    /// A projection the loader refuses.
    pub fn load(scope: &mut Scope<'_, '_>, spec: TimestepSpec) -> Result<Self> {
        let mut projection = |name: &str, inputs: u32, outputs: u32| {
            Linear::load(
                &mut scope.scope(name),
                inputs,
                outputs,
                Bias::Present,
                Parallel::None,
            )
        };
        Ok(Self {
            first: projection("linear_1", spec.frequency_dim, spec.hidden)?,
            second: projection("linear_2", spec.hidden, spec.out)?,
        })
    }

    /// The element type the pair computes in: its weights'.
    pub fn dtype(&self) -> ops::DType {
        self.first.weight().dtype
    }

    /// `linear_2(silu(linear_1(embedding)))` of `embedding` `[rows, frequency_dim]`: a fresh
    /// `[rows, out]` tensor of the embedding's element type. The two intermediates are reserved
    /// after the output and handed back before return.
    ///
    /// # Errors
    ///
    /// A view a projection refuses, or an arena that cannot hold the output and intermediates.
    pub fn forward(&self, f: &Forward, embedding: Operand) -> Result<Operand> {
        // The second projection's rows are the embedding's, which is all its reservation reads.
        let out = self.second.reserve(f, embedding)?;
        f.scoped(|f| {
            let hidden = self.first.forward(f, embedding)?;
            let activated = Silu.forward(f, hidden)?;
            self.second.launch(f, activated, out)
        })?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The row is every cosine, then every sine of the same angles: a permuted row is a different
    /// conditioning that every downstream check accepts.
    #[test]
    fn the_embedding_is_every_cosine_then_every_sine() {
        let dim = 8;
        let time = 0.25_f32;
        let values = sinusoidal(&[time], dim).unwrap();
        assert_eq!(values.len(), dim as usize);
        let half = dim as usize / 2;
        for index in 0..half {
            // The references' FP32 ladder: `exp(-ln(10000) * i / half)`, then `t * f`.
            let frequency = (-10_000_f32.ln() * index as f32 / half as f32).exp();
            let angle = time * frequency;
            assert_eq!(values[index], angle.cos(), "element {index}");
            assert_eq!(
                values[half + index],
                angle.sin(),
                "element {}",
                half + index
            );
        }
        assert!(sinusoidal(&[], dim).is_err());
        assert!(sinusoidal(&[time], 7).is_err());
    }
}

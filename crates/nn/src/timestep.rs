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

use ops::{Shape, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::{Bias, Linear, Parallel, Silu};
use crate::weights::Scope;

/// The base of the frequency ladder: `10000^(-i / half)`.
const MAX_PERIOD: f64 = 10_000.0;

/// The sinusoidal embedding of each time, `[times, dim]` FP32 row-major.
///
/// Row `r`, element `i` is `cos(t_r · 10000^(-i/half))` for `i < half` and the sine of the same
/// angle for `i ≥ half`, `half = dim / 2`; the angle is evaluated in f64 and each value rounded once
/// to f32.
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
    let half = (dim / 2) as usize;
    let mut values = Vec::with_capacity(times.len() * dim as usize);
    for time in times {
        let angles: Vec<f64> = (0..half)
            .map(|index| {
                let exponent = 2.0 * index as f64 / f64::from(dim);
                f64::from(*time) * MAX_PERIOD.powf(-exponent)
            })
            .collect();
        values.extend(angles.iter().map(|angle| angle.cos() as f32));
        values.extend(angles.iter().map(|angle| angle.sin() as f32));
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

    /// `out = linear_2(silu(linear_1(embedding)))`, `embedding` `[rows, frequency_dim]` and `out`
    /// `[rows, out]`; the two intermediates are reserved from the arena and released before return.
    ///
    /// # Errors
    ///
    /// A view a projection refuses, or an arena that cannot hold the intermediates.
    pub fn forward(&self, f: &Forward, embedding: TensorView, out: TensorView) -> Result<()> {
        let arena = f.arena();
        let mark = arena.mark();
        let shape = Shape::new(&[embedding.rows(), u64::from(self.first.n())]);
        let hidden = arena.tensor(embedding.dtype, shape)?;
        let activated = arena.tensor(embedding.dtype, shape)?;
        self.first.forward(f, embedding, hidden)?;
        Silu.forward(f, hidden, activated)?;
        self.second.forward(f, activated, out)?;
        arena.release(mark);
        Ok(())
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
            let angle = f64::from(time) * 10_000_f64.powf(-(index as f64) / half as f64);
            assert_eq!(values[index], angle.cos() as f32, "element {index}");
            assert_eq!(
                values[half + index],
                angle.sin() as f32,
                "element {}",
                half + index
            );
        }
        assert!(sinusoidal(&[], dim).is_err());
        assert!(sinusoidal(&[time], 7).is_err());
    }
}

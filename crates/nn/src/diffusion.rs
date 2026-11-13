//! Flow-matching schedules, the solver formulas a diffusion trajectory advances with, and the
//! sinusoidal embedding of a time.
//!
//! A schedule is host arithmetic over a checkpoint's fixed statement: the sigma points one
//! modality's trajectory visits, from its noise level down to the clean endpoint `0`. A model
//! states them to execution as [`Step`]s ([`steps`]), and each step's forward computes the solver
//! formula on the latents:
//!
//! ```text
//! time_shift(u, s)   s u / (1 + (s − 1) u)                      the rectified-flow time shift
//! uniform_sigmas     u_i = linspace(1, 0, points);  sigma_i = time_shift(u_i) in FP32, deduplicated
//! ladder_sigmas      u_i = rung_i / train_steps in FP64;  sigma_i = time_shift(u_i) in FP64,
//!                    rounded once to FP32, then the clean endpoint 0
//! euler_step         x + v · (sigma_i − sigma_{i+1})
//! scale_noise        x + x · (sigma − 1)                        a fresh draw at the first level
//! add_noise          t · x + (1 − t) · noise                    a condition's anchor
//! sinusoidal         [cos(t · f_i) …, sin(t · f_i) …]            f_i = 10000^(−i / half)
//! ```
//!
//! ## Precision
//!
//! The precision of each grid is the released schedulers' and is part of the result: the uniform
//! grid evaluates its shift in FP32 (the released grid is an FP32 tensor), the ladder in FP64
//! with one rounding at the end. Consecutive equal FP32 values of a uniform grid collapse into
//! one point, as `torch.unique_consecutive` collapses them; a ladder keeps one point per rung,
//! because each rung is one trained evaluation. Every solver formula forms its products and its
//! sum in FP32 and rounds once into the sample's element type.
//!
//! ## Host values in a step
//!
//! A step's coefficient is a host value of the schedule. Each solver function moves it to the
//! sample's device as a one-element tensor, a host-to-device copy, so a captured forward reads
//! it as an input and one capture serves every step of the trajectory. The functions return the
//! advanced sample; a model writes it into the latents execution lends with `slice_set`.

use tensor::{DType, Device, Tensor};

use crate::capability::Step;
use crate::error::{Error, Result};

/// The base of the sinusoidal embedding's frequencies: `10000^(−i / half)`.
const MAX_PERIOD: f32 = 10_000.0;

/// The rectified-flow time shift `s u / (1 + (s − 1) u)`, evaluated in `T`.
///
/// The precision is the caller's statement: a uniform grid shifts in FP32 and a ladder in FP64,
/// and the two differ in the last bits of the resulting sigma.
pub fn time_shift<T>(value: T, shift: T) -> T
where
    T: Copy
        + From<u8>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let one = T::from(1);
    shift * value / (one + (shift - one) * value)
}

/// `linspace(start, end, count)` in FP64: `start + i · (end − start) / (count − 1)`.
///
/// Both endpoints are exact: the clean end of a schedule must be zero and not the residue that
/// `start + n · step` leaves, which would scale the final sample by it. A `count` below two is
/// the start alone.
pub fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count < 2 {
        return vec![start; count];
    }
    let step = (end - start) / (count - 1) as f64;

    (0..count)
        .map(|index| match index {
            0 => start,
            last if last == count - 1 => end,
            _ => start + index as f64 * step,
        })
        .collect()
}

/// `values` with consecutive bitwise-equal entries dropped, as `torch.unique_consecutive` drops
/// them.
pub fn dedup(values: Vec<f32>) -> Vec<f32> {
    let mut points: Vec<f32> = Vec::with_capacity(values.len());

    for value in values {
        if points
            .last()
            .is_none_or(|previous| previous.to_bits() != value.to_bits())
        {
            points.push(value);
        }
    }
    points
}

/// The sigma points of a uniform grid under `shift`, decreasing to the clean endpoint `0`:
/// `points` points of a descending `linspace(1, 0, points)`, shifted in FP32 and deduplicated.
pub fn uniform_sigmas(points: u32, shift: f32) -> Vec<f32> {
    dedup(
        linspace(1.0, 0.0, points as usize)
            .into_iter()
            .map(|value| time_shift(value as f32, shift))
            .collect(),
    )
}

/// The sigma points of a distilled student's trained rungs under `shift`: the unshifted levels
/// `rung / train_steps` of a `train_steps`-step training schedule, shifted in FP64 and rounded
/// once, then the clean endpoint `0`.
///
/// A ladder is not deduplicated: its length is its trained evaluations, and two rungs that
/// round to one FP32 sigma are still two evaluations.
pub fn ladder_sigmas(rungs: &[u32], train_steps: f64, shift: f32) -> Vec<f32> {
    let shift = f64::from(shift);

    let mut sigmas: Vec<f32> = rungs
        .iter()
        .map(|rung| time_shift(f64::from(*rung) / train_steps, shift) as f32)
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// The steps of a trajectory through `sigmas`: step `i` starts at `sigmas[i]` and ends at
/// `sigmas[i + 1]`. Fewer than two points take no step.
pub fn steps(sigmas: &[f32]) -> Vec<Step> {
    let count = sigmas.len().saturating_sub(1) as u32;

    sigmas
        .windows(2)
        .zip(0..)
        .map(|(pair, index)| Step {
            index,
            count,
            sigma: pair[0],
            sigma_next: pair[1],
        })
        .collect()
}

/// `value` as a one-element FP32 tensor on `like`'s device: a host value a forward reads.
fn host_scalar(value: f32, like: &Tensor) -> Result<Tensor> {
    Ok(Tensor::new(&[value], &Device::Cpu)?.to_device(like.device())?)
}

/// Checks that `other`, the tensor `name`, covers `sample` element for element.
fn covers(owner: &'static str, name: &str, other: &Tensor, sample: &Tensor) -> Result<()> {
    if other.dims() != sample.dims() || other.dtype() != sample.dtype() {
        return Err(Error::shape(
            owner,
            format!(
                "a {name} of {} {:?} does not cover a sample of {} {:?}",
                ops::abi::name(other.dtype()),
                other.dims(),
                ops::abi::name(sample.dtype()),
                sample.dims()
            ),
        ));
    }
    Ok(())
}

/// The Euler step of one trajectory: `sample + velocity · (sigma − sigma_next)`, in `sample`'s
/// shape and element type.
///
/// The coefficient is the FP32 difference of `step`'s two levels. The clean-sample form of the
/// released solvers, `x · r + (x + v · sigma) · (1 − r)` with `r = sigma_next / sigma`, closes
/// algebraically to this step.
///
/// # Errors
///
/// A velocity of another shape or element type than the sample.
pub fn euler_step(sample: &Tensor, velocity: &Tensor, step: Step) -> Result<Tensor> {
    covers("euler_step", "velocity", velocity, sample)?;
    let coefficient = host_scalar(step.sigma - step.sigma_next, sample)?;

    let advanced = (sample.to_dtype(DType::F32)?
        + velocity.to_dtype(DType::F32)?.broadcast_mul(&coefficient)?)?;
    Ok(advanced.to_dtype(sample.dtype())?)
}

/// A fresh draw scaled to the schedule's first level `sigma`: `sample + sample · (sigma − 1)`,
/// for a schedule that does not start from unit noise.
///
/// # Errors
///
/// What a tensor operation refuses.
pub fn scale_noise(sample: &Tensor, sigma: f32) -> Result<Tensor> {
    let coefficient = host_scalar(sigma - 1.0, sample)?;
    let sample32 = sample.to_dtype(DType::F32)?;

    let scaled = (&sample32 + sample32.broadcast_mul(&coefficient)?)?;
    Ok(scaled.to_dtype(sample.dtype())?)
}

/// A clean `sample` noised to the flow-matching level `t`: `t · sample + (1 − t) · noise`.
///
/// `t` is the clean fraction (`1` is the clean sample), the coordinate a condition anchored at
/// a fixed level is stated in. The coefficients are `t` and the FP32 difference `1 − t`.
///
/// # Errors
///
/// Noise of another shape or element type than the sample.
pub fn add_noise(sample: &Tensor, noise: &Tensor, t: f32) -> Result<Tensor> {
    covers("add_noise", "noise", noise, sample)?;
    let (clean, noisy) = (host_scalar(t, sample)?, host_scalar(1.0 - t, sample)?);

    let mixed = (sample.to_dtype(DType::F32)?.broadcast_mul(&clean)?
        + noise.to_dtype(DType::F32)?.broadcast_mul(&noisy)?)?;
    Ok(mixed.to_dtype(sample.dtype())?)
}

/// The sinusoidal embedding of each time: `[times, dim]` in FP32 on the CPU device.
///
/// `times` is a CPU tensor of one axis. Row `r`, element `i` is `cos(t_r · f_i)` for `i < half`
/// and the sine of the same angle for `i ≥ half`, `half = dim / 2`, with
/// `f_i = exp(−ln(10000) · i / half)`: every cosine and then every sine (the GLIDE/DiT order,
/// not an interleaving). Every step is FP32, as the diffusion references evaluate it.
///
/// # Errors
///
/// No time, times of more than one axis, or a width with no cosine half.
pub fn sinusoidal(times: &Tensor, dim: u32) -> Result<Tensor> {
    const OWNER: &str = "sinusoidal";

    let &[count] = times.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("the times are one axis, not {:?}", times.dims()),
        ));
    };
    if count == 0 {
        return Err(Error::shape(
            OWNER,
            "an embedding needs at least one time".to_owned(),
        ));
    }
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(Error::shape(
            OWNER,
            format!("a {dim}-wide embedding has no cosine half"),
        ));
    }
    let half = dim / 2;
    let scalar = |value: f32| Tensor::new(value, &Device::Cpu);

    // `exp(−ln(10000) · i / half)`: the product, then the quotient, then the exponential.
    let frequencies = Tensor::arange(0_f32, half as f32, &Device::Cpu)?
        .broadcast_mul(&scalar(-MAX_PERIOD.ln())?)?
        .broadcast_div(&scalar(half as f32)?)?
        .exp()?;
    let angles = times
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&frequencies.unsqueeze(0)?)?;

    Ok(Tensor::cat(&[angles.cos()?, angles.sin()?], 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shift_of_one_is_the_identity_and_the_endpoints_are_fixed() {
        for value in [0.0_f64, 0.25, 0.5, 1.0] {
            assert_eq!(time_shift(value, 1.0), value);
        }
        assert_eq!(time_shift(1.0_f32, 12.0), 1.0);
        assert_eq!(time_shift(0.0_f32, 12.0), 0.0);

        // A stronger shift moves an interior point towards the noisy end.
        assert!(time_shift(0.5_f32, 12.0) > time_shift(0.5_f32, 3.0));
    }

    #[test]
    fn linspace_hits_both_endpoints_exactly() {
        let grid = linspace(0.999, 0.0, 33);
        assert_eq!(grid.len(), 33);
        assert_eq!(grid[0], 0.999);
        assert_eq!(grid[32], 0.0);
        assert!(grid.windows(2).all(|pair| pair[0] > pair[1]));
        assert_eq!(linspace(1.0, 0.0, 1), vec![1.0]);
    }

    #[test]
    fn dedup_drops_only_consecutive_repeats() {
        assert_eq!(
            dedup(vec![1.0, 1.0, 0.5, 0.5, 0.25, 0.5]),
            vec![1.0, 0.5, 0.25, 0.5]
        );

        // Bitwise: a negative zero is not a zero.
        assert_eq!(dedup(vec![0.0, -0.0]).len(), 2);
    }

    #[test]
    fn a_uniform_grid_shifts_in_single_precision_and_collapses_equal_points() {
        assert_eq!(uniform_sigmas(5, 1.0), vec![1.0, 0.75, 0.5, 0.25, 0.0]);

        let shifted = uniform_sigmas(50, 12.0);
        assert_eq!(shifted.len(), 50);
        assert_eq!(shifted[1], time_shift((1.0 - 1.0 / 49.0_f64) as f32, 12.0));

        // `s u / (1 + (s − 1) u)` tends to one as `s` grows, so the interior points collapse.
        assert_eq!(uniform_sigmas(4, 1e30), vec![1.0, 0.0]);
    }

    #[test]
    fn a_ladder_shifts_in_double_precision_and_keeps_every_rung() {
        let sigmas = ladder_sigmas(&[999, 874, 749, 624, 500, 375, 250, 125], 1000.0, 10.0);
        assert_eq!(sigmas.len(), 9);
        assert_eq!(sigmas[0], time_shift(0.999_f64, 10.0) as f32);
        assert_eq!(sigmas[8], 0.0);

        // Two rungs that round to one FP32 sigma stay two evaluations.
        let close = ladder_sigmas(&[1_000_000_000, 999_999_999], 1e9, 12.0);
        assert_eq!(close.len(), 3);
        assert_eq!(close[0], close[1]);
    }

    #[test]
    fn the_steps_follow_the_sigma_intervals_down_to_zero() {
        let sigmas = uniform_sigmas(50, 12.0);
        let trajectory = steps(&sigmas);
        assert_eq!(trajectory.len(), 49);

        for (index, step) in trajectory.iter().enumerate() {
            assert_eq!((step.index, step.count), (index as u32, 49));
            assert_eq!(
                (step.sigma, step.sigma_next),
                (sigmas[index], sigmas[index + 1])
            );
            assert!(step.sigma > step.sigma_next);
        }
        assert_eq!(trajectory[48].sigma_next, 0.0);
        assert!(steps(&[1.0]).is_empty());
    }

    #[test]
    fn the_clean_sample_form_closes_to_the_euler_step() {
        let (sigma, sigma_next) = (0.75_f32, 0.25_f32);
        let ratio = sigma_next / sigma;

        for (x, v) in [(0.5_f32, 1.25_f32), (-2.0, 0.125), (0.0, -3.0)] {
            let released = x * ratio + (x + v * sigma) * (1.0 - ratio);
            let euler = x + v * (sigma - sigma_next);
            assert!((released - euler).abs() < 1e-6);
        }
    }
}

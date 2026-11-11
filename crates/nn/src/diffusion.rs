//! Flow-matching schedules and the Euler step a diffusion model's trajectory advances with.
//!
//! A schedule is host arithmetic over a checkpoint's fixed statement: the sigma points one
//! modality's trajectory visits, from its noise level down to the clean endpoint `0`. The Euler
//! step is the device half: it advances a sample from one sigma point to the next along the
//! network's velocity.
//!
//! ```text
//! time_shift(u, s)   s u / (1 + (s - 1) u)                      the rectified-flow time shift
//! Grid::Uniform      u_i = linspace(1, 0, points);  sigma_i = time_shift(u_i) in FP32, deduplicated
//! Grid::Ladder       u_i = rung_i / train_steps in FP64;  sigma_i = time_shift(u_i) in FP64,
//!                    rounded once to FP32, then the clean endpoint 0
//! Euler              x += v · (sigma_i - sigma_{i+1})
//! add_noise          x  = t · x + (1 - t) · noise                       a condition's anchor
//! TimestepTable      the per-block modulation at each distinct level, gathered per step
//! ```
//!
//! ## Precision
//!
//! The precision of each grid is the released schedulers' and is part of the result: the uniform
//! grid evaluates its shift in FP32 (the released grid is an FP32 tensor), the ladder in FP64 with one
//! rounding at the end. Consecutive equal FP32 values of a uniform grid collapse into one point, as
//! `torch.unique_consecutive` collapses them; a ladder keeps one point per rung, because each rung is
//! one trained evaluation.
//!
//! ## The device step
//!
//! A step's coefficient is a host value of a fixed schedule, so [`Euler`] publishes the coefficients
//! as one FP32 `[steps, 1]` constant at load and gathers the evaluated step's word on the device,
//! indexed by the caller's step table. Every evaluation then launches against the same buffers, so
//! one captured graph serves the whole trajectory; only the step table's index changes per replay.

use ops::{DType, Shape, Strides, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::{Axpby, IndexRows, Residual, Scale};
use crate::timestep::sinusoidal;
use crate::weights::{Scope, Weight};

/// The rectified-flow time shift `s u / (1 + (s - 1) u)`, evaluated in `T`.
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

/// `linspace(start, end, count)` in FP64: `start + i · (end - start) / (count - 1)`.
///
/// Both endpoints are exact: the clean end of a schedule must be zero and not the residue that
/// `start + n · step` leaves, which would scale the final sample by it. A `count` below two is the
/// start alone.
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

/// A flow-matching grid: where a trajectory's sigma points lie before and after the time shift.
#[derive(Clone, PartialEq, Debug)]
pub enum Grid {
    /// `points` points of a descending `linspace(1, 0, points)`, shifted in FP32 and deduplicated.
    Uniform {
        /// Sigma points, the clean endpoint included.
        points: u32,
    },
    /// A distilled student's trained rungs: unshifted levels of a `train_steps`-step training
    /// schedule, shifted in FP64, then the clean endpoint.
    Ladder {
        /// The rungs, one evaluation each.
        rungs: Vec<u32>,
        /// The training schedule's step count the rungs are stated on.
        train_steps: f64,
    },
}

impl Grid {
    /// The grid's sigma points under `shift`, decreasing to the clean endpoint `0`.
    ///
    /// A ladder is not deduplicated: its length is its trained evaluations, and two rungs that
    /// round to one FP32 sigma are still two evaluations.
    pub fn sigmas(&self, shift: f32) -> Vec<f32> {
        match self {
            Self::Uniform { points } => dedup(
                linspace(1.0, 0.0, *points as usize)
                    .into_iter()
                    .map(|value| time_shift(value as f32, shift))
                    .collect(),
            ),
            Self::Ladder { rungs, train_steps } => {
                let shift = f64::from(shift);
                let mut sigmas: Vec<f32> = rungs
                    .iter()
                    .map(|rung| time_shift(f64::from(*rung) / train_steps, shift) as f32)
                    .collect();
                sigmas.push(0.0);
                sigmas
            }
        }
    }
}

/// The Euler step of one trajectory: `sample += velocity · (sigma_i - sigma_{i+1})`.
///
/// The step is `scaled_residual` with a one-word scale gathered on the device from the published
/// `[steps, 1]` coefficient table, so the launch reads one buffer whose contents change per
/// evaluation. The clean-sample form of the released solvers, `x · r + (x + v · sigma) · (1 - r)`
/// with `r = sigma_next / sigma`, closes algebraically to this step.
pub struct Euler {
    /// `[steps, 1]` FP32: evaluation `i` advances the sample by `coefficients[i]`.
    coefficients: Weight,
    /// Scales a fresh draw to the schedule's first level, `sample += sample · (sigma_0 - 1)`, for a
    /// schedule that does not start from unit noise.
    start: Option<Residual>,
}

impl Euler {
    /// The step coefficients of a trajectory through `sigmas`: `sigma_i - sigma_{i+1}` in FP32.
    pub fn coefficients(sigmas: &[f32]) -> Vec<f32> {
        sigmas.windows(2).map(|pair| pair[0] - pair[1]).collect()
    }

    /// The step over `sigmas`, publishing its coefficients as `<scope>.step_coefficients`.
    ///
    /// `start` is the level a fresh draw is scaled to before the first evaluation, when the schedule
    /// scales its draw at all; it is published as `<scope>.start_coefficient`, `start - 1`.
    ///
    /// # Errors
    ///
    /// Fewer than two sigma points, or a constant the loader refuses.
    pub fn load(scope: &mut Scope<'_, '_>, sigmas: &[f32], start: Option<f32>) -> Result<Self> {
        if sigmas.len() < 2 {
            return Err(Error::shape(
                "Euler",
                format!(
                    "a trajectory of {} sigma points takes no step",
                    sigmas.len()
                ),
            ));
        }
        let coefficients = Self::coefficients(sigmas);
        let table = scope.constant_f32(
            "step_coefficients",
            &[coefficients.len() as u64, 1],
            &coefficients,
        )?;
        let start = match start {
            None => None,
            Some(sigma) => {
                let scale = scope.constant_f32("start_coefficient", &[1], &[sigma - 1.0])?;
                Some(Residual::new(Scale::Channel(scale), None)?)
            }
        };
        Ok(Self {
            coefficients: table,
            start,
        })
    }

    /// How many evaluations the trajectory takes.
    pub fn steps(&self) -> u32 {
        self.coefficients.shape().dim(0) as u32
    }

    /// Advance `sample` in place by `velocity` over the step `step` names.
    ///
    /// `step` is the caller's `[1]` i32 step index on the device; the step's FP32 coefficient is
    /// gathered into the forward's arena. `sample` and `velocity` are contiguous views of one
    /// element count and one element type.
    ///
    /// # Errors
    ///
    /// Views of different element counts or types, a strided view, or an operator's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        sample: TensorView,
        velocity: TensorView,
        step: TensorView,
    ) -> Result<()> {
        let (sample, velocity) = (elements(sample)?, elements(velocity)?);
        if sample.shape != velocity.shape || sample.dtype != velocity.dtype {
            return Err(Error::shape(
                "Euler",
                format!(
                    "a velocity of {:?} {} does not cover a sample of {:?} {}",
                    velocity.shape.dims(),
                    velocity.dtype,
                    sample.shape.dims(),
                    sample.dtype
                ),
            ));
        }
        let gathered = IndexRows.gather(f, self.coefficients.view(), step)?;
        let scale = Weight {
            backing: crate::Backing::new(gathered.ptr, 4),
            view: gathered.reshape(Shape::new(&[1])),
        };
        // A one-channel scale: every element of the sample's single column reads it.
        Residual::new(Scale::Channel(scale), None)?.forward(f, sample, velocity, None)?;
        Ok(())
    }

    /// Scale a fresh draw `sample` in place to the schedule's first level.
    ///
    /// Nothing is launched for a schedule that starts from unit noise.
    ///
    /// # Errors
    ///
    /// A strided view, or the operator's refusal.
    pub fn prepare_latents(&self, f: &Forward, sample: TensorView) -> Result<()> {
        let Some(start) = self.start.as_ref() else {
            return Ok(());
        };
        let sample = elements(sample)?;
        start.forward(f, sample, sample, None)?;
        Ok(())
    }
}

/// Noise a clean `sample` in place to the flow-matching level `t`: `sample ← t · sample + (1 − t) ·
/// noise`.
///
/// `t` is the clean fraction (`1` is the clean sample), the coordinate a condition anchored at a
/// fixed level is stated in. The coefficients are `t` and the FP32 difference `1 − t`; the products
/// and the sum are FP32 and round once into `sample`'s element type. `sample` and `noise` are
/// `[rows, columns]` views of one shape.
///
/// # Errors
///
/// Views of different shapes, or the operator's refusal.
pub fn add_noise(f: &Forward, sample: TensorView, noise: TensorView, t: f32) -> Result<()> {
    Axpby.forward(f, sample, noise, 1.0 - t, t)
}

/// The geometry of a [`TimestepTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestepTableSpec {
    /// Width of the sinusoidal embedding the levels are evaluated at.
    pub frequency_dim: u32,
    /// The blocks the table holds one modulation table for.
    pub layers: u32,
    /// Columns of one block's modulation row.
    pub layer_columns: u64,
    /// Columns of the final norm's modulation row.
    pub output_columns: u64,
    /// Element type of the modulation tables.
    pub dtype: DType,
}

/// A diffusion stack's timestep conditioning, precomputed at each distinct noise level of a fixed
/// schedule and gathered per step on the device.
///
/// A step conditions its rows on several timestep groups (a modality's level, a condition's
/// level), and every block reads one modulation row per group. The schedule is fixed, so each
/// block's modulation projection is evaluated once per **distinct** level, and a step gathers its
/// groups' rows by entry index:
///
/// ```text
/// frequencies  [entries, frequency_dim]           FP32   the sinusoidal embedding of each level
/// entries      [steps, groups]                    i32    the entry each (step, group) reads
/// layers       [layers, entries, layer_columns]          each block's modulation at each entry
/// output       [entries, output_columns]                 the final norm's modulation at each entry
/// ```
///
/// `frequencies` and `entries` are constants of the schedule, published at load. The two
/// modulation tables are model-computed constants the runtime allocates: the table states them
/// ([`Self::tables`]), the model fills them once after loading from `frequencies` through its own
/// projections, and every step reads them ([`Self::bind`]). A step's index is a device value, so
/// one captured graph serves every step.
pub struct TimestepTable {
    frequencies: Weight,
    entries: Weight,
    /// `[layers, entries, layer_columns]`: each block's modulation at each entry, written once by
    /// the model's projections after the load ([`Scope::writable_zeros`]).
    layers: Weight,
    /// `[entries, output_columns]`: the final norm's modulation at each entry, written likewise.
    output: Weight,
}

impl TimestepTable {
    /// The distinct levels of `levels` and the entry each level reads.
    ///
    /// `levels` is step-major, `groups` per step. Levels are compared bitwise, and an entry is the
    /// position of its level's first occurrence, so the entries follow the schedule's order. The
    /// result is `(times, entries)`: one level per entry, and one entry per element of `levels`.
    ///
    /// # Errors
    ///
    /// No group, levels that are not whole steps, or more entries than an `i32` indexes.
    pub fn entries(levels: &[f32], groups: u32) -> Result<(Vec<f32>, Vec<i32>)> {
        if groups == 0 || levels.is_empty() || !levels.len().is_multiple_of(groups as usize) {
            return Err(Error::shape(
                "TimestepTable",
                format!(
                    "{} levels are not whole steps of {groups} groups",
                    levels.len()
                ),
            ));
        }
        let mut times: Vec<f32> = Vec::new();
        let mut entries = Vec::with_capacity(levels.len());
        for level in levels {
            let position = match times
                .iter()
                .position(|time| time.to_bits() == level.to_bits())
            {
                Some(position) => position,
                None => {
                    times.push(*level);
                    times.len() - 1
                }
            };
            entries.push(i32::try_from(position).map_err(|_| {
                Error::shape(
                    "TimestepTable",
                    "more entries than an i32 indexes".to_owned(),
                )
            })?);
        }
        Ok((times, entries))
    }

    /// The table of `levels` (step-major, `groups` per step) under `scope`.
    ///
    /// Publishes `frequencies` and `entries` as constants, and reserves the modulation tables as
    /// this module's writable constants, which the model's projections fill once after the load
    /// ([`Self::layer_table`], [`Self::output_table`]).
    ///
    /// # Errors
    ///
    /// As [`Self::entries`], an embedding width [`sinusoidal`] refuses, or a constant the loader
    /// refuses.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        levels: &[f32],
        groups: u32,
        spec: TimestepTableSpec,
    ) -> Result<Self> {
        let (times, table) = Self::entries(levels, groups)?;
        let entries = times.len() as u64;
        let frequencies = scope.constant_f32(
            "frequencies",
            &[entries, u64::from(spec.frequency_dim)],
            &sinusoidal(&times, spec.frequency_dim)?,
        )?;
        // One row per step of the levels, which may be more rows than distinct entries.
        let steps = (table.len() / groups as usize) as u64;
        let entry_table = scope.constant_i32("entries", &[steps, u64::from(groups)], &table)?;
        let layers = scope.writable_zeros(
            "layers",
            spec.dtype,
            &[u64::from(spec.layers), entries, spec.layer_columns],
        )?;
        let output = scope.writable_zeros("output", spec.dtype, &[entries, spec.output_columns])?;
        Ok(Self {
            frequencies,
            entries: entry_table,
            layers,
            output,
        })
    }

    /// The groups one step reads.
    pub fn groups(&self) -> u64 {
        self.entries.shape().last()
    }

    /// The steps the entry table covers.
    pub fn steps(&self) -> u64 {
        self.entries.shape().dim(0)
    }

    /// `[entries, frequency_dim]` FP32: the embedding the model's projections read.
    pub fn frequencies(&self) -> TensorView {
        self.frequencies.view()
    }

    /// The entries step `step` reads: `step` is `[1]` i32 on the device, and the answer a fresh
    /// `[1, groups]` i32 tensor.
    ///
    /// # Errors
    ///
    /// The gather's refusal, or an arena that cannot hold the entries.
    pub fn entries_of(&self, f: &Forward, step: TensorView) -> Result<TensorView> {
        IndexRows.gather(f, self.entries.view(), step)
    }
}

impl TimestepTable {
    /// Block `layer`'s `[entries, layer_columns]` table, which the model writes once after the load
    /// and every step gathers from.
    ///
    /// # Errors
    ///
    /// A block the table does not hold.
    pub fn layer_table(&self, layer: usize) -> Result<TensorView> {
        let layers = self.layers.view();
        let count = layers.shape.dim(0);
        if layer as u64 >= count {
            return Err(Error::shape(
                "TimestepTable",
                format!("{count} block tables, not block {layer}"),
            ));
        }
        let shape = Shape::new(&[layers.shape.dim(1), layers.shape.dim(2)]);
        Ok(layers.narrow(0, layer as u64, 1)?.reshape(shape))
    }

    /// The final norm's `[entries, output_columns]` table, written once after the load.
    pub fn output_table(&self) -> TensorView {
        self.output.view()
    }

    /// Block `layer`'s modulation rows at `entries` (the `[1, groups]` view
    /// [`Self::entries_of`] answered): a fresh `[groups, layer_columns]` tensor.
    ///
    /// # Errors
    ///
    /// A block the table does not hold, the gather's refusal, or an arena that cannot hold the rows.
    pub fn layer(&self, f: &Forward, layer: usize, entries: TensorView) -> Result<TensorView> {
        IndexRows.gather(f, self.layer_table(layer)?, flat(entries))
    }

    /// The final norm's modulation rows at `entries`: a fresh `[groups, output_columns]` tensor.
    ///
    /// # Errors
    ///
    /// The gather's refusal, or an arena that cannot hold the rows.
    pub fn output(&self, f: &Forward, entries: TensorView) -> Result<TensorView> {
        IndexRows.gather(f, self.output.view(), flat(entries))
    }
}

/// A `[1, groups]` entry list as the one-axis index list a gather takes.
///
/// [`TimestepTable::entries_of`] answers `[1, groups]`, one row of the entry table; the block and
/// output gathers read the same bytes as `[groups]` indices.
fn flat(entries: TensorView) -> TensorView {
    entries.reshape(Shape::new(&[entries.shape.numel()]))
}

/// A contiguous view as the `[elements, 1]` column the elementwise step reads.
///
/// The step is elementwise, so the sample keeps its mathematical shape and the operator sees one
/// element per row, the shape a `[1]` scale broadcasts over.
fn elements(sample: TensorView) -> Result<TensorView> {
    if sample.strides != Strides::row_major(sample.shape) {
        return Err(Error::shape(
            "Euler",
            format!(
                "a sample of {:?} at strides {:?} is not contiguous",
                sample.shape.dims(),
                sample.strides.as_slice()
            ),
        ));
    }
    Ok(TensorView::new(
        sample.ptr,
        sample.dtype,
        Shape::new(&[sample.shape.numel(), 1]),
    ))
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
        let linear = Grid::Uniform { points: 5 }.sigmas(1.0);
        assert_eq!(linear, vec![1.0, 0.75, 0.5, 0.25, 0.0]);

        let shifted = Grid::Uniform { points: 50 }.sigmas(12.0);
        assert_eq!(shifted.len(), 50);
        assert_eq!(shifted[1], time_shift((1.0 - 1.0 / 49.0_f64) as f32, 12.0));

        // `s u / (1 + (s - 1) u)` tends to one as `s` grows, so the interior points collapse.
        assert_eq!(Grid::Uniform { points: 4 }.sigmas(1e30), vec![1.0, 0.0]);
    }

    #[test]
    fn a_ladder_shifts_in_double_precision_and_keeps_every_rung() {
        let ladder = Grid::Ladder {
            rungs: vec![999, 874, 749, 624, 500, 375, 250, 125],
            train_steps: 1000.0,
        };
        let sigmas = ladder.sigmas(10.0);
        assert_eq!(sigmas.len(), 9);
        assert_eq!(sigmas[0], time_shift(0.999_f64, 10.0) as f32);
        assert_eq!(*sigmas.last().unwrap(), 0.0);

        // Two rungs that round to one FP32 sigma stay two evaluations.
        let close = Grid::Ladder {
            rungs: vec![1_000_000_000, 999_999_999],
            train_steps: 1e9,
        };
        let sigmas = close.sigmas(12.0);
        assert_eq!(sigmas.len(), 3);
        assert_eq!(sigmas[0], sigmas[1]);
    }

    #[test]
    fn a_coefficient_is_the_sigma_interval_and_the_last_lands_on_zero() {
        let sigmas = Grid::Uniform { points: 50 }.sigmas(12.0);
        let coefficients = Euler::coefficients(&sigmas);
        assert_eq!(coefficients.len(), 49);
        assert!(coefficients.iter().all(|coefficient| *coefficient > 0.0));
        assert_eq!(coefficients[48], sigmas[48]);
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

//! The denoiser's solver: one Euler step of a diffusion trajectory per evaluation.
//!
//! A network evaluates a velocity at a noise level; the solver advances the sample from that level
//! to the next. MiniMax-H3 uses the released clean-sample Euler solver
//! (`uniserve/diffusion/solver.py::CleanSampleEulerSolver`), whose two statements
//!
//! ```text
//! clean = 1 - timestep                  # = sigma, rounded to the sample's dtype
//! v     = v * clean + x                 # the clean estimate, one rounding
//! ratio = sigma_next / sigma            # FP32
//! x     = x * ratio + v * (1 - ratio)
//! ```
//!
//! close algebraically to one Euler step in the sigma domain,
//!
//! ```text
//! x += v * (sigma - sigma_next)
//! ```
//!
//! because `sigma * (1 - ratio) = sigma - sigma_next`. The operator that computes it is
//! `scaled_residual` (`hidden += update * scale`), whose scale is a device view: since a step's
//! coefficient is a host value of the checkpoint's fixed schedule, this module publishes the
//! coefficients as one FP32 constant table per modality at load, and builds one [`ScaledResidual`]
//! per evaluation whose scale is the `[1]` view at that evaluation's element. A captured graph
//! therefore addresses one coefficient by address, and one capture serves one evaluation - the
//! placement of that bucket is the executor's (see the `DenoiserRunner` agreement).
//!
//! ## What the solver writes
//!
//! Nothing but the sample. The samples are the executor's buffers, the velocity is the network's
//! output for this evaluation, and both arrive borrowed. A model holds no host-to-device path, which
//! is why the coefficients are constants rather than a per-step upload.
//!
//! ## Precision
//!
//! `scaled_residual` reads its residual and its update in one element type. The released denoiser
//! predicts in FP32 over a trajectory it keeps in the sample's own type, so that the Euler
//! arithmetic stays FP32 and only the store rounds. This model instead declares the output heads'
//! precision as the latents' type (specs/worker.md 9, the `torch.autocast` row: each layer declares
//! its precision), so the trajectory and the velocity agree in type, the arithmetic inside the
//! operator is FP32, and the store rounds once per step. The difference from the released pipeline
//! is one rounding per step of the velocity; it is inside the operator's tolerance (specs/worker.md
//! 7.3) and is the `quality` preset's own representation.

use nn::layers::ScaledResidual;
use nn::weights::{Weight, WeightLoader};
use nn::{DType, DevicePtr, Forward, Shape, TensorView};

use crate::schedule::{Modality, Schedule};
use crate::{Error, Result};

/// The elements of a latent tensor as the operator's row matrix sees them.
///
/// The Euler update is elementwise, so any contiguous view is one element per row; the latents keep
/// their mathematical shape (`[channels, frames, height, width]`) and the operator sees
/// `[elements, 1]`, which is the shape a `[1]` scale broadcasts over.
pub fn elements(sample: TensorView) -> Result<TensorView> {
    if sample.strides != nn::Strides::row_major(sample.shape) {
        return Err(Error::checkpoint(
            "H3 latents must be contiguous for the solver's elementwise update".to_owned(),
        ));
    }
    let count = sample.shape.numel();
    Ok(TensorView::new(
        sample.ptr,
        sample.dtype,
        Shape::new(&[count, 1]),
    ))
}

/// The Euler solver of a checkpoint's schedule.
///
/// One value per modality: the video and audio timelines advance together but carry their own
/// shifts and so their own coefficients.
pub struct Solver {
    video: Track,
    audio: Track,
}

/// One modality's steps: one scaled addition per evaluation, and the level a draw starts from.
struct Track {
    /// `steps[i]` advances evaluation `i`: `sample += velocity * (sigma_i - sigma_{i+1})`.
    steps: Vec<ScaledResidual>,
    /// Scales a fresh draw to the schedule's first level: `sample += sample * (sigma_0 - 1)`.
    ///
    /// A parallel-decoding student's first node is not unit noise; the released and distilled
    /// grids start at `sigma_0 = 1`, where the layer is omitted because the draw is already right.
    start: Option<ScaledResidual>,
}

impl Solver {
    /// Publish the schedule's coefficients and build the steps.
    ///
    /// `loader` is the loading boundary's weight loader: the two coefficient tables become
    /// constants of the retained store, so every deployment of this checkpoint reads the same
    /// numbers and a graph addresses them by address.
    pub fn load(schedule: &Schedule, loader: &mut WeightLoader<'_>) -> Result<Self> {
        Ok(Self {
            video: Track::load(schedule, Modality::Video, loader)?,
            audio: Track::load(schedule, Modality::Audio, loader)?,
        })
    }

    /// How many evaluations one modality has.
    pub fn steps(&self, modality: Modality) -> u32 {
        match modality {
            Modality::Video => self.video.steps.len() as u32,
            Modality::Audio => self.audio.steps.len() as u32,
        }
    }

    /// Advance `sample` in place to the next noise level: `sample += velocity * Δsigma`.
    ///
    /// `evaluation` is the step's index in the schedule; the runner gets it from
    /// [`Schedule::steps`] and passes it back in [`nn::Step::index`](nn::Step).
    pub fn step(
        &self,
        f: &Forward,
        modality: Modality,
        sample: TensorView,
        velocity: TensorView,
        evaluation: u32,
    ) -> Result<()> {
        let track = self.track(modality);
        let scaled = track
            .steps
            .get(evaluation as usize)
            .ok_or_else(|| {
                Error::checkpoint(format!(
                    "H3 {} evaluation {evaluation} is outside the {} the schedule has",
                    modality.as_str(),
                    track.steps.len()
                ))
            })?;
        let (sample, velocity) = self.check(sample, velocity, modality)?;
        scaled.forward(f, sample, velocity).map_err(Error::from)
    }

    /// Scale a fresh draw to the level the schedule starts from.
    ///
    /// The released and distilled grids start from unit noise, where this does nothing; a
    /// parallel-decoding student scales by its first node's sigma.
    pub fn start(&self, f: &Forward, modality: Modality, sample: TensorView) -> Result<()> {
        let Some(start) = self.track(modality).start.as_ref() else {
            return Ok(());
        };
        let sample = elements(sample)?;
        start.forward(f, sample, sample).map_err(Error::from)
    }

    /// The coefficients of one step, as the published table holds them.
    ///
    /// This is the schedule's arithmetic without a launch, for the loading boundary's tests and for
    /// the executor's own diagnostics.
    pub fn coefficients(schedule: &Schedule, modality: Modality) -> Result<Vec<f32>> {
        let steps = schedule.steps(modality)?;
        Ok(steps
            .into_iter()
            .map(|step| step.sigma_next - step.sigma)
            .collect())
    }

    fn track(&self, modality: Modality) -> &Track {
        match modality {
            Modality::Video => &self.video,
            Modality::Audio => &self.audio,
        }
    }

    /// Check that a step's two views are the same contiguous element count in one type.
    fn check(
        &self,
        sample: TensorView,
        velocity: TensorView,
        modality: Modality,
    ) -> Result<(TensorView, TensorView)> {
        let (sample, velocity) = (elements(sample)?, elements(velocity)?);
        if sample.shape != velocity.shape {
            return Err(Error::checkpoint(format!(
                "H3 {} velocity {:?} does not cover the sample's {:?}",
                modality.as_str(),
                velocity.shape.dims(),
                sample.shape.dims()
            )));
        }
        if sample.dtype != velocity.dtype {
            return Err(Error::checkpoint(format!(
                "H3 {} velocity is {} while the trajectory is {}; the step's operator reads one \
                 type",
                modality.as_str(),
                velocity.dtype,
                sample.dtype
            )));
        }
        Ok((sample, velocity))
    }
}

impl Track {
    fn load(
        schedule: &Schedule,
        modality: Modality,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let coefficients = Solver::coefficients(schedule, modality)?;
        let table = publish(
            loader,
            &format!("denoiser.{}.step_coefficients", modality.as_str()),
            &coefficients,
        )?;
        let steps = (0..coefficients.len())
            .map(|index| ScaledResidual::new(element(&table, index), None).map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;

        // The start coefficient is `sigma_0 - 1`; at unit noise the draw needs no launch.
        let sigma = schedule.start_sigma(modality)?;
        let start = if sigma == 1.0 {
            None
        } else {
            let table = publish(
                loader,
                &format!("denoiser.{}.start_coefficient", modality.as_str()),
                &[sigma - 1.0],
            )?;
            Some(ScaledResidual::new(element(&table, 0), None).map_err(Error::from)?)
        };
        Ok(Self { steps, start })
    }
}

/// Publish `values` as an FP32 constant and answer the weight that views it.
fn publish(loader: &mut WeightLoader<'_>, name: &str, values: &[f32]) -> Result<Weight> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let shape = Shape::new(&[values.len() as u64]);
    loader
        .constant(name, DType::F32, shape, &bytes)
        .map_err(Error::from)
}

/// The `[1]` view at element `index` of a published table.
///
/// The table is contiguous row-major FP32, so element `index` starts `index` words after its first
/// address. The view borrows the table's backing, which the returned weight keeps alive.
fn element(table: &Weight, index: usize) -> Weight {
    let address = table.view.ptr.address() + (index as u64) * 4;
    Weight {
        backing: table.backing.clone(),
        view: TensorView::new(DevicePtr::new(address), DType::F32, Shape::new(&[1])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Shifts;
    use crate::config::ScheduleSpec;

    fn base() -> Schedule {
        let spec = ScheduleSpec::uniform(
            50,
            Shifts {
                video: 12.0,
                audio: 3.0,
            },
        )
        .expect("the base checkpoint's schedule");
        Schedule::new(&spec).expect("an aligned base schedule")
    }

    fn blocks() -> Schedule {
        let spec = ScheduleSpec::blocks(
            32,
            vec![0, 4, 8, 12, 16, 20, 24, 28, 32],
            0.999,
            Shifts {
                video: 12.0,
                audio: 3.0,
            },
        )
        .expect("a block grid");
        Schedule::new(&spec).expect("an aligned block schedule")
    }

    #[test]
    fn a_coefficient_is_minus_the_sigma_interval() {
        let schedule = base();
        let coefficients = Solver::coefficients(&schedule, Modality::Video).expect("coefficients");
        let steps = schedule.steps(Modality::Video).expect("steps");
        assert_eq!(coefficients.len(), steps.len());
        for (coefficient, step) in coefficients.iter().zip(&steps) {
            assert!(*coefficient < 0.0);
            assert!((coefficient - (step.sigma_next - step.sigma)).abs() < 1e-6);
        }
        // The last step lands exactly on the clean endpoint.
        assert!((coefficients[48] - -steps[48].sigma).abs() < 1e-6);
    }

    #[test]
    fn the_released_clean_sample_form_closes_to_the_same_step() {
        // The released solver reaches `x + v Δsigma` through the clean estimate:
        //   x0 = x + v sigma;  x' = x r + x0 (1 - r),  r = sigma_next / sigma.
        let (sigma, sigma_next) = (0.75f32, 0.25f32);
        let ratio = sigma_next / sigma;
        for (x, v) in [(0.5f32, 1.25f32), (-2.0, 0.125), (0.0, -3.0)] {
            let clean = x + v * sigma;
            let released = x * ratio + clean * (1.0 - ratio);
            let euler = x + v * (sigma_next - sigma);
            assert!(
                (released - euler).abs() < 1e-6,
                "released {released} and euler {euler} disagree"
            );
        }
    }

    #[test]
    fn the_base_schedule_takes_forty_nine_steps() {
        let schedule = base();
        let coefficients = Solver::coefficients(&schedule, Modality::Audio).expect("coefficients");
        assert_eq!(coefficients.len(), 49);
    }

    #[test]
    fn a_parallel_decoding_student_starts_below_unit_noise() {
        let schedule = blocks();
        let sigma = schedule
            .start_sigma(Modality::Video)
            .expect("a start level");
        assert!((sigma - 0.999).abs() < 1e-6);
        // Its draw is scaled by `sigma`, which the start coefficient expresses as `sigma - 1`.
        assert!((sigma - 1.0 - -0.001).abs() < 1e-6);
        // The released grid starts at unit noise and needs no launch.
        assert_eq!(base().start_sigma(Modality::Video).expect("a level"), 1.0);
    }

    #[test]
    fn a_latent_view_is_one_element_per_row() {
        let view = TensorView::new(
            DevicePtr::new(0),
            DType::BF16,
            Shape::new(&[24, 5, 8, 8]),
        );
        let elements = elements(view).expect("a contiguous latent tensor");
        assert_eq!(elements.shape.dims(), &[24 * 5 * 8 * 8, 1]);
    }

    #[test]
    fn a_strided_latent_tensor_is_refused() {
        let mut view = TensorView::new(DevicePtr::new(0), DType::BF16, Shape::new(&[24, 5]));
        view.strides = nn::Strides::new(&[2, 1]);
        assert!(elements(view).is_err());
    }

    #[test]
    fn the_step_view_addresses_one_word_of_the_table() {
        let table = Weight {
            backing: nn::Backing::new(DevicePtr::new(0), 32),
            view: TensorView::new(DevicePtr::new(0x1000), DType::F32, Shape::new(&[8])),
        };
        let third = element(&table, 2);
        assert_eq!(third.view.ptr.address(), 0x1008);
        assert_eq!(third.view.shape.dims(), &[1]);
        assert_eq!(third.dtype(), DType::F32);
    }
}

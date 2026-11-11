//! The denoiser's solver: one Euler step of each modality's trajectory per evaluation.
//!
//! A network evaluates a velocity at a noise level; the solver advances the sample from that level
//! to the next. MiniMax-H3 uses the released clean-sample Euler solver, which closes algebraically
//! to the shared flow-matching step [`nn::diffusion::Euler`], `x += v · (sigma - sigma_next)`.
//! This module binds one such step per modality to the checkpoint's evaluated [`Schedule`].
//!
//! ## What the solver writes
//!
//! Nothing but the sample. The samples are the worker's buffers and the velocity is the network's
//! output for this evaluation. The coefficients are published constants of the schedule, and the
//! step gathers its own by the evaluation index it uploads.
//!
//! ## Precision
//!
//! `scaled_residual` reads its residual and its update in one element type. The released denoiser
//! predicts in FP32 over a trajectory it keeps in the sample's own type, so that the Euler
//! arithmetic stays FP32 and only the store rounds. This model instead declares the output heads'
//! precision as the latents' type, so the trajectory and the velocity agree in type, the arithmetic
//! inside the operator is FP32, and the store rounds once per step.

use nn::diffusion::Euler;
use nn::weights::WeightLoader;
use nn::{Forward, TensorView};

use crate::schedule::{Modality, Schedule};
use crate::{Error, Result};

/// The Euler solver of a checkpoint's schedule.
///
/// One [`Euler`] step per modality: the video and audio timelines advance together but carry their
/// own shifts and so their own coefficients.
pub struct Solver {
    video: Euler,
    audio: Euler,
}

impl Solver {
    /// Publish the schedule's coefficients and build the steps.
    ///
    /// Each modality's coefficient table becomes a constant of the retained store under
    /// `denoiser.<modality>`, so every deployment of this checkpoint reads the same numbers and a
    /// graph addresses them by address.
    pub fn load(schedule: &Schedule, loader: &mut WeightLoader<'_>) -> Result<Self> {
        let mut euler = |modality: Modality| -> Result<Euler> {
            let mut scope = loader.scope(format!("denoiser.{}", modality.as_str()));
            // Only the grid that scales its draw states a start level (`Schedule::start_sigma`): a
            // released or distilled grid starts from unit noise and needs no launch at all.
            Ok(Euler::load(
                &mut scope,
                &schedule.sigmas(modality),
                schedule.start_sigma(modality)?,
            )?)
        };
        Ok(Self {
            video: euler(Modality::Video)?,
            audio: euler(Modality::Audio)?,
        })
    }

    /// How many evaluations one modality has.
    pub fn steps(&self, modality: Modality) -> u32 {
        self.euler(modality).steps()
    }

    /// Advance `sample` in place to the next noise level: `sample += velocity * Δsigma`.
    ///
    /// `evaluation` is the step's index in the schedule, on the host and as the `[1]` i32 the
    /// coefficient is gathered by on the device.
    pub fn step(
        &self,
        f: &Forward,
        modality: Modality,
        (sample, velocity): (TensorView, TensorView),
        (evaluation, index): (u32, TensorView),
    ) -> Result<()> {
        let steps = self.steps(modality);
        if evaluation >= steps {
            return Err(Error::checkpoint(format!(
                "H3 {} evaluation {evaluation} is outside the {steps} the schedule has",
                modality.as_str()
            )));
        }
        Ok(self.euler(modality).forward(f, sample, velocity, index)?)
    }

    /// Scale a fresh draw to the level the schedule starts from.
    ///
    /// The released and distilled grids start from unit noise, where this does nothing; a
    /// parallel-decoding student scales by its first node's sigma.
    pub fn start(&self, f: &Forward, modality: Modality, sample: TensorView) -> Result<()> {
        Ok(self.euler(modality).prepare_latents(f, sample)?)
    }

    fn euler(&self, modality: Modality) -> &Euler {
        match modality {
            Modality::Video => &self.video,
            Modality::Audio => &self.audio,
        }
    }
}

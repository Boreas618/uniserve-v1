//! The image denoiser's runner.
//!
//! FIXME: `ImageDiffusion` is the image family of `DenoiserRunner` and
//! belongs in `runner/denoiser.rs` as that type. It keeps its own type and
//! path while `executor/forward/token/` names
//! `runner::diffusion::image::{ConditionRow, ImageDiffusion}`.

pub(crate) mod image;

//! The programs a deployment compiles: one `operator<TAB>key` row per launch the model can make.
//!
//! A kernel store is built for a target before any weight is read, so the deployment's key set must
//! be stated from the model's own geometry: each component says which launches its `forward` makes,
//! through the same construction that launches them (`nn`'s `descriptor_of` accessors and the model's
//! own config arithmetic). The alternative — a table in another crate that restates the configuration
//! — is a second implementation of the model, and it drifts silently: the store then lacks a program
//! the model asks for, and the refusal arrives at a launch instead of at a build.
//!
//! Rows are deduplicated and ordered, because a key set is a set: two layers of one shape share one
//! program.

use ops::Op;

use crate::config::{Config, MAX_FRAMES};
use crate::{Error, Result};

/// The launches a component makes, as the operator and the key its descriptor states.
#[derive(Default)]
pub struct Programs {
    rows: Vec<(String, String)>,
}

impl Programs {
    /// Add one launch.
    ///
    /// The operator is the one the launch names ([`Op::NAME`]) and the key is the descriptor's own
    /// text, so a row is exactly what a store is asked for at that launch.
    pub fn launch<O: Op>(&mut self, descriptor: &O::Descriptor) {
        self.rows
            .push((O::NAME.to_owned(), ops::descriptor::key(descriptor)));
    }

    /// Add rows another component stated, keeping its operators and keys as they are.
    ///
    /// A denoiser composes two stacks and their orchestration; absorbing each stack's rows rather than
    /// restating them is what keeps one construction per program.
    pub fn extend(&mut self, rows: impl IntoIterator<Item = (String, String)>) {
        self.rows.extend(rows);
    }

    /// The rows, deduplicated and ordered by operator and then key.
    pub fn finish(mut self) -> Vec<(String, String)> {
        self.rows.sort();
        self.rows.dedup();
        self.rows
    }

    /// Whether no launch has been added.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// The element type of each component a deployment runs.
///
/// One fact per component, and the checkpoint's own under `dtype = "auto"` (the design's precision
/// rule: nothing is converted at load). A single `dtype` handed to every component is the shape of
/// every element-type defect this crate has had - a video VAE asked for the stack's type states rows
/// no launch asks for, and a transformer asked for the VAE's states none of the ones it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Types {
    /// The Qwen3-VL text encoder.
    pub text_encoder: nn::DType,
    /// The denoiser's two stacks.
    pub transformer: nn::DType,
    /// The video VAE, decoder and encoder.
    pub video: nn::DType,
    /// The audio VAE, decoder and encoder.
    pub audio: nn::DType,
}

/// What a deployment serves, stated once: its canvases, its components' element types, the clip
/// lengths its packing accepts, and the rows those imply.
///
/// Every downstream reader reads this value rather than restating a part of it - a canvas list in a
/// tool, an element type in a fixture header, a rung in a script is a copy of a deployment fact, and
/// the copies are where this crate's row sets drifted. It holds nothing a runtime owns: no weights,
/// no device, no stream, no pool, so a host with the checkpoint's configuration and no shard states
/// every row.
pub struct Deployment {
    /// The checkpoint's configuration, normalized at its loading boundary.
    pub config: Config,
    /// Each component's element type, which is the checkpoint's own unless the caller states it.
    pub types: Types,
    /// The rasters the deployment renders, from the DiT that states its canvases.
    pub canvases: Vec<(u32, u32)>,
    /// The clip lengths the delivery serves, in video frames.
    ///
    /// **The delivery states these; the packing only accepts them.** They are not every length the
    /// packing admits: the audio rows key on a track's extent today, so a deployment that listed all
    /// 58 admissible clips would ship 58 widths per stage for requests its stages never drive.
    /// [`Deployment::admissible`] gives the packing's own set, and [`Deployment::new`] refuses a rung
    /// outside it by name - so a delivery cannot state a clip the model would refuse.
    pub rungs: Vec<u32>,
}

impl Deployment {
    /// A deployment of this checkpoint: the canvases it offers and the clips the delivery serves.
    ///
    /// A rung the packing refuses is refused here by name, because the alternative is a row set that
    /// covers a clip no request can ask for and a first launch that fails at the kernel lookup.
    pub fn new(config: Config, types: Types, rungs: Vec<u32>) -> Result<Self> {
        let dies = config.denoisers.values().next().ok_or_else(|| {
            Error::config("the checkpoint states no DiT, so it states no canvases".to_owned())
        })?;
        let canvases = dies
            .offered_canvases()
            .iter()
            .map(|canvas| (canvas.height, canvas.width))
            .collect();
        let admissible = Self::admissible();
        for rung in &rungs {
            if !admissible.contains(rung) {
                return Err(Error::config(format!(
                    "a {rung}-frame clip is not one the packing accepts: the rule admits {} frames \
                     up to {MAX_FRAMES}",
                    admissible.len()
                )));
            }
        }
        Ok(Self {
            config,
            types,
            canvases,
            rungs,
        })
    }

    /// The clip lengths the packing accepts, computed from its own rule rather than listed.
    ///
    /// The rule is `17n + 5` up to [`MAX_FRAMES`]; the first rung is the shortest clip one window
    /// holds. A listed ladder is a second statement of the rule and drifts from it at the first
    /// change - which is why the ladder this crate shipped for two days was wrong at both ends.
    pub fn admissible() -> Vec<u32> {
        let mut rungs = Vec::new();
        let mut frames = crate::CLIP_FRAMES + crate::CLIP_LATENTS;
        while frames <= MAX_FRAMES {
            rungs.push(frames);
            frames += crate::CLIP_LATENTS;
        }
        rungs
    }

    /// The audio latent frames of each rung.
    pub fn latents(&self) -> Vec<u32> {
        self.rungs
            .iter()
            .map(|frames| crate::packing::audio_latent_frames(*frames))
            .collect()
    }

    /// The track lengths, in samples, the encoder's residual units are sized in.
    pub fn samples(&self) -> Vec<u32> {
        self.latents()
            .iter()
            .filter_map(|latent| {
                u32::try_from(crate::audio_vae::track_samples(
                    &self.config.audio_vae,
                    *latent,
                ))
                .ok()
            })
            .collect()
    }

    /// The rows the two audio VAEs and the video VAE state: the components whose whole argument is
    /// the deployment's own canvases, rungs and element types.
    ///
    /// A component that also takes a request extent (the prompt's length, the trajectory's elements,
    /// the packed row bound) is stated by [`Deployment::rows`], which is the one function downstream
    /// reads.
    pub fn vae_rows(&self) -> Result<Vec<(String, String)>> {
        let mut programs = Programs::default();
        let latents = self.latents();
        let samples = self.samples();
        programs.extend(crate::audio_vae::AudioDecoder::programs(
            &self.config.audio_vae,
            &latents,
        )?);
        programs.extend(crate::encoding::AudioEncoder::programs(
            &self.config.audio_vae,
            &samples,
        )?);
        programs.extend(crate::video_vae::VideoDecoder::programs(
            &self.config.video_vae,
            self.types.video,
        )?);
        programs.extend(crate::output::VideoPostprocessor::programs(
            self.types.video,
            &self.canvases,
        ));
        programs.extend(crate::encoding::VideoEncoder::programs(
            &self.config.video_vae,
            &self.canvases,
        )?);
        Ok(programs.finish())
    }
}

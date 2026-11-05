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

use crate::config::{AttentionKind, Config, MAX_FRAMES};
use crate::denoiser::Arrangement;
use crate::modulation::Groups;
use crate::partition::Partition;
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
    /// **A supplied clip and a generated clip obey different rules, and this is the supplied one.**
    /// A generated timeline is `17n + 5` frames ([`Deployment::generated`]); a clip a request hands
    /// the model - a reference video, the track `ref2va` conditions on - is any length the encoder
    /// divides ([`crate::encoding::Clip::latent_frames`]). The audio rows follow the second rule, so
    /// a delivery that listed only the first would refuse the lengths its own stages drive.
    /// [`Deployment::new`] refuses a rung the encoder refuses, by name.
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
        let dies = dies.clone();
        for rung in &rungs {
            if let Err(error) = dies.clip.latent_frames(*rung) {
                return Err(Error::config(format!(
                    "a {rung}-frame clip is not one the encoder accepts: {error}"
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

    /// The clip lengths a **generated** timeline may have, computed from the packing's rule rather
    /// than listed: `17n + 5` up to [`MAX_FRAMES`].
    ///
    /// A listed ladder is a second statement of the rule and drifts from it at the first change.
    /// These are not the rungs a delivery serves - see [`Deployment::rungs`], which is the supplied
    /// clip's rule - and confusing the two is how a ladder of stage lengths came to be read as a list
    /// of clips no request can name.
    pub fn generated() -> Vec<u32> {
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

    /// The rows the text encoder and its vision tower state, at the prompt length the deployment
    /// holds.
    ///
    /// The tower's own statement is scoped: it keys its learned position table on the sequence it is
    /// handed, while the table's compiled grid is a second dimension. A deployment therefore states
    /// the table itself ([`Deployment::tower_table`]) until the tower takes the grid.
    pub fn text_rows(&self, partition: &Partition, tokens: u32) -> Result<Vec<(String, String)>> {
        let mut programs = Programs::default();
        programs.extend(crate::text_encoder::TextEncoder::programs(
            &self.config.text_encoder,
            partition,
            self.types.text_encoder,
            tokens,
        )?);
        programs.extend(crate::vision::VisionTower::programs(
            &self.config.text_encoder.vision,
            self.types.text_encoder,
            tokens,
        )?);
        Ok(programs.finish())
    }

    /// The rows the two DiTs state: their stacks, the orchestration around them, and the
    /// conditioner.
    ///
    /// Three facts are the caller's and are handed in rather than held, because they are the
    /// deployment's *capacity* rather than the checkpoint's shape - the sequence's arrangement, the
    /// tensor-parallel placement, and whether the modulation cache is the conditioned one. The
    /// packed row bound is not among them: it is derived from the arrangement by the layout rule
    /// (`Layouts::rows_bound`), which is the one place that answers it.
    ///
    /// `elements` is empty by construction. The solver's residual is one operator over the whole
    /// trajectory, so its width is a request's element count - a row for it would compile a program
    /// for numbers no request states until it arrives, which is the class this file keeps naming:
    /// **a runtime quantity that sizes a launch is an argument, not a key.**
    pub fn denoiser_rows(
        &self,
        partition: &Partition,
        arrangement: &Arrangement,
        groups: Groups,
    ) -> Result<Vec<(String, String)>> {
        let mut programs = Programs::default();
        for config in self.config.denoisers.values() {
            let rules = crate::denoiser::Layouts::new(config.clone(), *arrangement)?;
            let rows = rules.rows_bound()?;
            let tiles = match config.attention {
                AttentionKind::Dense => 0,
                _ => rows
                    .div_ceil(config.attention.capacity_rows())
                    .next_power_of_two(),
            };
            programs.extend(crate::denoiser::Denoiser::programs(
                &config.transformer,
                config.attention,
                self.types.transformer,
                rows,
                groups,
                partition,
                &[],
                tiles,
            )?);
        }
        Ok(programs.finish())
    }

    /// The rows the two audio VAEs and the video VAE state, at the request extents a walk states.
    ///
    /// The rasters and the clip ladder are arguments because they are the delivery's: a walk states
    /// which clips it drives, and [`Deployment::own_vae_rows`] is the same statement at this
    /// checkpoint's own canvases and the caller's rungs. A component that also takes the prompt's
    /// length or the trajectory's elements is stated by [`Deployment::text_rows`] and
    /// [`Deployment::denoiser_rows`].
    pub fn vae_rows(
        &self,
        rasters: &[(u32, u32)],
        latents: &[u32],
        samples: &[u32],
    ) -> Result<Vec<(String, String)>> {
        let mut programs = Programs::default();
        programs.extend(crate::audio_vae::AudioDecoder::programs(
            &self.config.audio_vae,
            latents,
        )?);
        programs.extend(crate::encoding::AudioEncoder::programs(
            &self.config.audio_vae,
            samples,
        )?);
        programs.extend(crate::video_vae::VideoDecoder::programs(
            &self.config.video_vae,
            self.types.video,
        )?);
        programs.extend(crate::output::VideoPostprocessor::programs(
            self.types.video,
            rasters,
        ));
        programs.extend(crate::encoding::VideoEncoder::programs(
            &self.config.video_vae,
            rasters,
        )?);
        Ok(programs.finish())
    }

    /// The same rows at this checkpoint's own canvases and the rungs the delivery states.
    pub fn own_vae_rows(&self) -> Result<Vec<(String, String)>> {
        self.vae_rows(&self.canvases, &self.latents(), &self.samples())
    }
}

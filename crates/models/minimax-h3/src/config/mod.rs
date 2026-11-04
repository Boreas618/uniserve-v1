//! MiniMax-H3's typed configuration: the checkpoint's architecture, read once at the loading
//! boundary and immutable afterwards.
//!
//! `uniserve/models/minimax_h3/config.py` normalizes a checkpoint into one configuration before
//! any module is built; this module does the same for the fields the Rust model reads. A
//! checkpoint is the base release (a pipeline index over the DiT partitions, text encoder, video
//! VAE and audio VAE it holds) or a FastH3 export (one DiT partition and schedulers, with a
//! contract naming the base revision it reads the rest from).
//!
//! ## What is a fact of the checkpoint and what is a fact of the request
//!
//! The dimensions, the schedule, the attention kind, the canvases a DMD student generates and the
//! latent statistics are the checkpoint's and are stated here. How a request selects a canvas, how
//! many frames a duration gives and how condition media are planned are the request rules'
//! (`crates/models/rules`), and are not duplicated here - except the twelve canvases a DMD student
//! was trained on, which are a fact of *that* checkpoint rather than of the rule.
//!
//! ## Timeline
//!
//! H3 counts video in frames at [`FPS`], in clips of [`CLIP_FRAMES`] frames that the video VAE
//! encodes to [`CLIP_LATENTS`] latent frames, and audio in latent steps of [`AUDIO_HOP`] samples
//! at [`AUDIO_SAMPLE_RATE`]. The checkpoint's own VAE configuration must agree with these, which
//! [`Config::normalize`] checks: a checkpoint whose compression differs is one this crate's
//! packing and reconstruction cannot serve.

mod audio_vae;
mod canvas;
mod denoiser;
mod field;
mod text;
mod video_vae;

use std::collections::BTreeMap;
use std::path::Path;

use nn::VideoTask;

pub use audio_vae::AudioVaeConfig;
pub use canvas::{CANVAS_SHORT_EDGE, Canvas, DMD_CANVASES, NAMED_ASPECT_RATIOS, NAMED_CANVASES};
pub use denoiser::{
    AttentionKind, DenoiserConfig, SPARSE_TILE_ROWS, ScheduleGrid, ScheduleSpec, Shifts,
    TransformerConfig,
};
pub use text::{
    Activation, PixelConfig, RETAINED_LAYERS, TextEncoderConfig, VisionActivation, VisionConfig,
};
pub use video_vae::{SpatialPadding, VideoVaeConfig};

use crate::checkpoint::{ContractSchedule, Dit, Layout, Sidecars};
use crate::{Error, Result};

/// Frames per second of a generated video.
pub const FPS: u32 = 24;

/// Frames of one video VAE clip: the temporal footprint of one encoding step.
pub const CLIP_FRAMES: u32 = 17;

/// Latent frames one [`CLIP_FRAMES`]-frame clip encodes to.
pub const CLIP_LATENTS: u32 = 5;

/// Latent frames the final clip drops from a generated length; the next clip's overlap repeats
/// them.
pub const DROPPED_LATENTS: u32 = 3;

/// Samples per second of the generated audio track.
pub const AUDIO_SAMPLE_RATE: u32 = 32_000;

/// Audio latent steps per second: the generated audio stream's latent rate.
pub const AUDIO_LATENTS_PER_SECOND: u32 = 40;

/// Samples one audio latent step covers.
pub const AUDIO_HOP: u32 = AUDIO_SAMPLE_RATE / AUDIO_LATENTS_PER_SECOND;

/// Channels of the generated audio: interleaved stereo.
pub const AUDIO_CHANNELS: u32 = 2;

/// A canvas side is a multiple of this many pixels, so it survives the video VAE's 16x spatial
/// compression on a whole number of 2x2 transformer patches.
pub const CANVAS_MULTIPLE: u32 = 32;

/// Steps of the training schedule a DMD student's rungs are stated on.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Sigma points of the released full-step grid, the clean endpoint included.
pub const UNIFORM_GRID_POINTS: u32 = 50;

/// A checkpoint's typed configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// The Qwen3-VL text encoder that conditions a request.
    pub text_encoder: TextEncoderConfig,
    /// Each DiT partition the checkpoint holds, by network.
    pub denoisers: BTreeMap<Dit, DenoiserConfig>,
    /// The video VAE: the decoder every task uses and the encoder `ref2va` uses.
    pub video_vae: VideoVaeConfig,
    /// The audio VAE: the BigVGAN decoder every task uses and the encoder `ref2va` uses.
    pub audio_vae: AudioVaeConfig,
}

impl Config {
    /// Read the checkpoint at `root`.
    ///
    /// `base` is the snapshot of the base revision a FastH3 export pins, from which the
    /// directories the export omits are read ([`crate::checkpoint::Contract::base_revision`]);
    /// it is `None` for a checkpoint that holds every directory.
    pub fn read(root: &Path, base: Option<&Path>) -> Result<Self> {
        let layout = Layout::detect(root)?;
        let sidecars = Sidecars::read(root, base, &layout)?;
        Self::normalize(&layout, &sidecars)
    }

    /// Build the configuration from a recognized layout and its parsed sidecars.
    ///
    /// This is the whole of the loading boundary's reading: everything is checked here - the
    /// language layers against the DiT's conditioning width, the VAEs' compression against the
    /// timeline the packing assumes, the sparse attention against its tile shapes - so no layer
    /// below sees a value it cannot use.
    pub fn normalize(layout: &Layout, sidecars: &Sidecars) -> Result<Self> {
        let text_encoder = TextEncoderConfig::parse(&sidecars.text_encoder, &sidecars.processor)?;
        let video_vae = VideoVaeConfig::parse(&sidecars.video_vae)?;
        let audio_vae = AudioVaeConfig::parse(&sidecars.audio_vae)?;
        let shifts = Shifts {
            video: shift(&sidecars.scheduler, "scheduler/scheduler_config.json")?,
            audio: shift(
                &sidecars.audio_scheduler,
                "audio_scheduler/scheduler_config.json",
            )?,
        };
        // A student's contract states the same two shifts next to its scheduler sidecars; a
        // checkpoint whose two statements disagree is refused rather than resolved silently.
        if let Some(contract) = &layout.contract {
            if contract.video_shift != shifts.video || contract.audio_shift != shifts.audio {
                return Err(Error::config(format!(
                    "fastvideo_inference.json shifts ({}, {}) disagree with the scheduler \
                     sidecars ({}, {})",
                    contract.video_shift, contract.audio_shift, shifts.video, shifts.audio
                )));
            }
        }

        let mut denoisers = BTreeMap::new();
        for (dit, tasks) in &layout.dits {
            let values = sidecars.dits.get(dit).ok_or_else(|| {
                Error::config(format!(
                    "the checkpoint holds no {}/config.json",
                    dit.directory()
                ))
            })?;
            let transformer = TransformerConfig::parse(
                field::object(values, &format!("{}/config.json", dit.directory()))?,
                &format!("{}/config.json", dit.directory()),
            )?;
            let (schedule, attention) = match &layout.contract {
                None => (
                    ScheduleSpec::uniform(UNIFORM_GRID_POINTS, shifts)?,
                    AttentionKind::Dense,
                ),
                Some(contract) => (
                    match &contract.schedule {
                        ContractSchedule::Rungs(rungs) => {
                            ScheduleSpec::dmd(rungs.clone(), NUM_TRAIN_TIMESTEPS, shifts)?
                        }
                        ContractSchedule::Nodes { nodes, max_t } => ScheduleSpec::blocks(
                            transformer.output_heads,
                            nodes.clone(),
                            *max_t,
                            shifts,
                        )?,
                    },
                    AttentionKind::sparse(
                        contract.attention.tile,
                        contract.attention.sparsity,
                        contract.attention.reference_keep,
                    )?,
                ),
            };
            let denoiser = DenoiserConfig::new(transformer, schedule, attention, tasks.clone())?;
            denoisers.insert(*dit, denoiser);
        }

        let config = Self {
            text_encoder,
            denoisers,
            video_vae,
            audio_vae,
        };
        config.validate()?;
        Ok(config)
    }

    /// The DiT a deployment places, when the checkpoint holds only one.
    ///
    /// A base release holds both partitions and a deployment places one, which the deployment
    /// names; an export holds exactly one.
    pub fn only_denoiser(&self) -> Option<(Dit, &DenoiserConfig)> {
        let mut dits = self.denoisers.iter();
        match (dits.next(), dits.next()) {
            (Some((dit, config)), None) => Some((*dit, config)),
            _ => None,
        }
    }

    /// The tasks the checkpoint can serve: each DiT's, in canonical order.
    pub fn tasks(&self) -> Vec<VideoTask> {
        let mut tasks = Vec::new();
        for task in [
            VideoTask::TextToVideoAudio,
            VideoTask::FirstLastToVideoAudio,
            VideoTask::ReferenceToVideoAudio,
        ] {
            if self
                .denoisers
                .values()
                .any(|denoiser| denoiser.tasks.contains(&task))
            {
                tasks.push(task);
            }
        }
        tasks
    }

    /// The DiT that serves `task`, when the checkpoint holds one.
    pub fn denoiser_for(&self, task: VideoTask) -> Option<(Dit, &DenoiserConfig)> {
        self.denoisers
            .iter()
            .find_map(|(dit, config)| config.tasks.contains(&task).then_some((*dit, config)))
    }

    /// Check the components against each other and against the timeline.
    fn validate(&self) -> Result<()> {
        // H3's packing and reconstruction hard-code one clip timeline: 17 frames encode to 5
        // latent frames, of which the final clip drops 3, at 16x spatial and 4x temporal
        // compression.
        let video = &self.video_vae;
        if (
            video.clip_length,
            video.clip_latents(),
            video.final_clip_latents(),
            video.spatial_compression(),
            video.temporal_compression(),
        ) != (
            CLIP_FRAMES,
            CLIP_LATENTS,
            CLIP_LATENTS - DROPPED_LATENTS,
            16,
            4,
        ) {
            return Err(Error::config(format!(
                "MiniMax-H3 video_vae must encode {CLIP_FRAMES}-frame clips to {CLIP_LATENTS} \
                 latent frames, dropping {DROPPED_LATENTS} of them from the final clip, at 16x \
                 spatial and 4x temporal compression; this checkpoint states {} frames to {} \
                 latents, keeping {}, at {}x spatial and {}x temporal",
                video.clip_length,
                video.clip_latents(),
                video.final_clip_latents(),
                video.spatial_compression(),
                video.temporal_compression()
            )));
        }
        // Soundtracks are resampled to the timeline's sample rate, so the audio VAE must run at
        // it and hop the timeline's latent step.
        let audio = &self.audio_vae;
        if audio.sampling_rate != AUDIO_SAMPLE_RATE {
            return Err(Error::config(format!(
                "MiniMax-H3 audio_vae samples at {} Hz; the timeline's audio runs at \
                 {AUDIO_SAMPLE_RATE} Hz",
                audio.sampling_rate
            )));
        }
        if audio.hop() != AUDIO_HOP || audio.decoder_hop() != AUDIO_HOP {
            return Err(Error::config(format!(
                "MiniMax-H3 audio_vae must hop {AUDIO_HOP} samples per latent; this checkpoint \
                 encodes {} and decodes {}",
                audio.hop(),
                audio.decoder_hop()
            )));
        }
        // Each DiT's latent widths and conditioning width are the VAEs' and the text encoder's.
        for (dit, denoiser) in &self.denoisers {
            let transformer = &denoiser.transformer;
            if self.text_encoder.hidden_size != transformer.text_dim {
                return Err(Error::config(format!(
                    "H3 text features are {} wide and the {} DiT conditions on {}",
                    self.text_encoder.hidden_size,
                    dit.directory(),
                    transformer.text_dim
                )));
            }
            if video.latent_channels != transformer.video_channels {
                return Err(Error::config(format!(
                    "H3 video latents have {} channels and the {} DiT reads {}",
                    video.latent_channels,
                    dit.directory(),
                    transformer.video_channels
                )));
            }
            if audio.latent_channels != transformer.audio_channels {
                return Err(Error::config(format!(
                    "H3 audio latents have {} channels and the {} DiT reads {}",
                    audio.latent_channels,
                    dit.directory(),
                    transformer.audio_channels
                )));
            }
        }
        Ok(())
    }
}

/// The noise level shift a scheduler sidecar states.
fn shift(sidecar: &serde_json::Value, path: &str) -> Result<f32> {
    let values = field::object(sidecar, path)?;
    Ok(field::positive(values, "shift", path)? as f32)
}

/// The released checkpoint's sidecars, as the tests of this crate use them.
///
/// They are the values `docs/minimax_h3/minimax_h3.md` and the released checkpoints state: a
/// 64-layer Qwen3-VL text encoder of which H3 keeps 50, a 50-block DiT conditioned on 5120-wide
/// text, a 17-frame video VAE of 24 latent channels, and a 32-channel audio VAE at 32 kHz.
#[cfg(test)]
pub(crate) mod tests_support {
    use std::collections::BTreeMap;

    use nn::VideoTask;
    use serde_json::Value;

    use super::*;
    use crate::checkpoint::{Contract, Layout, Sidecars, SparseContract};

    /// Parse a fixture's JSON.
    pub(crate) fn json(body: &str) -> Value {
        serde_json::from_str(body).expect("a fixture is JSON")
    }

    /// The text encoder checkpoint's configuration.
    pub(crate) fn text_encoder() -> Value {
        json(
            r#"{
                "image_token_id": 151655,
                "video_token_id": 151656,
                "tie_word_embeddings": false,
                "text_config": {
                    "attention_bias": false,
                    "head_dim": 128,
                    "hidden_act": "silu",
                    "hidden_size": 5120,
                    "intermediate_size": 25600,
                    "max_position_embeddings": 262144,
                    "num_attention_heads": 64,
                    "num_hidden_layers": 64,
                    "num_key_value_heads": 8,
                    "rms_norm_eps": 1e-06,
                    "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"},
                    "rope_theta": 5000000,
                    "vocab_size": 151936
                },
                "vision_config": {
                    "deepstack_visual_indexes": [8, 16, 24],
                    "depth": 27,
                    "hidden_act": "gelu_pytorch_tanh",
                    "hidden_size": 1152,
                    "in_channels": 3,
                    "intermediate_size": 4304,
                    "num_heads": 16,
                    "num_position_embeddings": 2304,
                    "out_hidden_size": 5120,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2
                }
            }"#,
        )
    }

    /// The processor's pixel normalization.
    pub(crate) fn processor() -> Value {
        json(r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}"#)
    }

    /// The video VAE's configuration.
    pub(crate) fn video_vae() -> Value {
        let statistics = |value: f64| Value::Array((0..24).map(|_| Value::from(value)).collect());
        serde_json::json!({
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 24,
            "block_out_channels": [128, 256, 256, 512, 512, 1024],
            "layers_per_block": 2,
            "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
            "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "spatial_padding_mode": "reflect",
            "decoder_num_layers": 36,
            "decoder_num_attention_heads": 32,
            "decoder_attention_head_dim": 64,
            "decoder_num_register_tokens": 4,
            "decoder_ffn_mult": 4,
            "decoder_rope_theta": 100.0,
            "decoder_rope_dim_ratio": 0.75,
            "decoder_norm_eps": 1e-05,
            "clip_length": 17,
            "token_drop": 3,
            "latents_mean": statistics(0.0),
            "latents_std": statistics(1.0),
        })
    }

    /// The audio VAE's configuration.
    pub(crate) fn audio_vae() -> Value {
        serde_json::json!({
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 4, 5, 5],
            "latent_dim": 2048,
            "latent_channels": 32,
            "decoder_dim": 1024,
            "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
            "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
            "num_attention_heads": 8,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "sampling_rate": 32000,
            "latents_mean": [0.0, 0.0],
            "latents_std": [1.0, 1.0],
        })
    }

    /// One DiT partition's configuration.
    pub(crate) fn transformer() -> Value {
        json(
            r#"{
                "num_attention_heads": 56,
                "attention_head_dim": 128,
                "hidden_size": 5376,
                "num_layers": 50,
                "num_refiner_layers": 2,
                "ffn_dim": 14336,
                "in_channels": 24,
                "audio_in_channels": 32,
                "patch_size": [1, 2, 2],
                "text_dim": 5120,
                "freq_dim": 256,
                "time_embed_hidden_dim": 5376,
                "time_embed_dim": 2688,
                "rope_freq_dim": 16,
                "rope_theta": 10000.0,
                "norm_eps": 1e-05,
                "qk_norm_eps": 1e-05,
                "final_norm_eps": 1e-05
            }"#,
        )
    }

    /// The base release's layout: both DiT partitions, no contract.
    pub(crate) fn base_layout() -> Layout {
        Layout {
            dits: vec![
                (
                    Dit::Base,
                    vec![
                        VideoTask::TextToVideoAudio,
                        VideoTask::FirstLastToVideoAudio,
                    ],
                ),
                (Dit::Reference, vec![VideoTask::ReferenceToVideoAudio]),
            ],
            contract: None,
            packed: false,
        }
    }

    /// The base release's sidecars.
    pub(crate) fn sidecars(layout: &Layout) -> Sidecars {
        let mut dits = BTreeMap::new();
        for (dit, _) in &layout.dits {
            dits.insert(*dit, transformer());
        }
        Sidecars {
            text_encoder: text_encoder(),
            processor: processor(),
            video_processor: None,
            video_vae: video_vae(),
            audio_vae: audio_vae(),
            scheduler: json(r#"{"shift": 12.0}"#),
            audio_scheduler: json(r#"{"shift": 3.0}"#),
            dits,
        }
    }

    /// A FastH3 DMD contract for the base partition.
    pub(crate) fn dmd_contract() -> Contract {
        Contract {
            task: VideoTask::TextToVideoAudio,
            dit: Dit::Base,
            base_revision: None,
            video_shift: 10.0,
            audio_shift: 3.0,
            schedule: ContractSchedule::Rungs(vec![999, 874, 749, 624, 500, 375, 250, 125]),
            attention: SparseContract {
                tile: 64,
                sparsity: 0.8,
                reference_keep: None,
            },
        }
    }

    /// The released base configuration.
    pub(crate) fn released_config() -> Config {
        let layout = base_layout();
        Config::normalize(&layout, &sidecars(&layout)).expect("the released configuration")
    }
}

#[cfg(test)]
mod tests {
    use nn::VideoTask;

    use super::tests_support::*;
    use super::*;

    #[test]
    fn a_base_release_reads_both_partitions_with_the_uniform_grid() {
        let layout = base_layout();
        let config = Config::normalize(&layout, &sidecars(&layout)).expect("a base release");
        assert_eq!(config.denoisers.len(), 2);
        assert_eq!(
            config.tasks(),
            vec![
                VideoTask::TextToVideoAudio,
                VideoTask::FirstLastToVideoAudio,
                VideoTask::ReferenceToVideoAudio,
            ]
        );
        let (dit, denoiser) = config
            .denoiser_for(VideoTask::ReferenceToVideoAudio)
            .expect("the reference task");
        assert_eq!(dit, Dit::Reference);
        assert_eq!(denoiser.attention, AttentionKind::Dense);
        assert_eq!(
            denoiser.schedule,
            ScheduleSpec::uniform(
                UNIFORM_GRID_POINTS,
                Shifts {
                    video: 12.0,
                    audio: 3.0
                }
            )
            .expect("the grid")
        );
        assert_eq!(denoiser.canvases, Some(&NAMED_CANVASES[..]));
        assert!(config.only_denoiser().is_none());
    }

    #[test]
    fn a_dmd_export_reads_its_ladder_its_sparse_attention_and_its_twelve_canvases() {
        let layout = Layout {
            dits: vec![(
                Dit::Base,
                vec![
                    VideoTask::TextToVideoAudio,
                    VideoTask::FirstLastToVideoAudio,
                ],
            )],
            contract: Some(dmd_contract()),
            packed: true,
        };
        let mut sidecars = sidecars(&layout);
        sidecars.scheduler = json(r#"{"shift": 10.0}"#);
        let config = Config::normalize(&layout, &sidecars).expect("a DMD export");
        let (dit, denoiser) = config.only_denoiser().expect("one partition");
        assert_eq!(dit, Dit::Base);
        assert_eq!(denoiser.schedule.steps(), 8);
        assert_eq!(
            denoiser.attention,
            AttentionKind::sparse(64, 0.8, None).expect("sparse")
        );
        assert_eq!(denoiser.canvases, Some(&DMD_CANVASES[..]));
    }

    #[test]
    fn a_configuration_refuses_components_that_disagree_with_each_other() {
        let layout = base_layout();

        let mut widths = sidecars(&layout);
        widths.text_encoder["text_config"]["hidden_size"] = json("4096");
        widths.text_encoder["vision_config"]["out_hidden_size"] = json("4096");
        let error = Config::normalize(&layout, &widths).expect_err("the DiT conditioning width");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("conditions on")),
            "{error}"
        );

        let mut channels = sidecars(&layout);
        let statistics = |value: f64| {
            serde_json::Value::Array((0..16).map(|_| serde_json::Value::from(value)).collect())
        };
        channels.video_vae["latent_channels"] = json("16");
        channels.video_vae["latents_mean"] = statistics(0.0);
        channels.video_vae["latents_std"] = statistics(1.0);
        let error = Config::normalize(&layout, &channels).expect_err("the latent channels");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("channels")),
            "{error}"
        );
    }

    #[test]
    fn a_configuration_refuses_a_vae_timeline_the_packing_cannot_serve() {
        let layout = base_layout();
        let mut timeline = sidecars(&layout);
        timeline.video_vae["temporal_downsample_factors"] = json("[1, 2, 1, 1, 1, 1]");
        let error = Config::normalize(&layout, &timeline).expect_err("a different compression");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("temporal")),
            "{error}"
        );

        let mut clock = sidecars(&layout);
        clock.audio_vae["sampling_rate"] = json("44100");
        let error = Config::normalize(&layout, &clock).expect_err("a different sample clock");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("sampling")),
            "{error}"
        );

        let mut hop = sidecars(&layout);
        hop.audio_vae["decoder_rates"] = json("[5, 5, 2, 2, 2, 2, 1]");
        let error = Config::normalize(&layout, &hop).expect_err("a different hop");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("hop")),
            "{error}"
        );
    }

    #[test]
    fn a_contract_whose_shifts_disagree_with_its_schedulers_is_refused() {
        let layout = Layout {
            dits: vec![(Dit::Base, vec![VideoTask::TextToVideoAudio])],
            contract: Some(dmd_contract()),
            packed: false,
        };
        // The sidecars state the base release's video shift of 12, not the student's 10.
        let error = Config::normalize(&layout, &sidecars(&layout)).expect_err("shift mismatch");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("disagree")),
            "{error}"
        );
    }

    #[test]
    fn a_denoiser_is_found_by_the_task_it_serves() {
        let config = released_config();
        assert_eq!(
            config
                .denoiser_for(VideoTask::FirstLastToVideoAudio)
                .map(|(dit, _)| dit),
            Some(Dit::Base)
        );
    }
}

//! MiniMax-H3's typed configuration: the checkpoint's sidecars, read once at the loading
//! boundary and immutable afterwards.
//!
//! A checkpoint is normalized into one [`Config`] before any module is built: the base release
//! (a pipeline index over the networks, the text encoder, the video VAE and the audio VAE it
//! holds), or a FastH3 export or adapter (one network whose schedule and attention its
//! `fastvideo_inference.json`, or the release it was extracted from, states). Every module's
//! `load` consumes the typed struct of its own submodule and reparses nothing.
//!
//! A packed ModelOpt export states its calibration in a manifest at its root; it is read here
//! into the quantization each stage's configuration declares ([`modelopt`]), which the stage's
//! loader is given before any of its weights is read.
//!
//! The request-level values the family's request rules are parameterized by (the frame rule,
//! the resolution rule, the durations and the audio clock) are request processing's
//! ([`processing::minimax_h3`]); this configuration checks that the checkpoint agrees with
//! them.

mod audio_vae;
mod denoiser;
mod modelopt;
mod text;
mod video_vae;

use std::collections::BTreeMap;

use nn::PixelNorm;
use nn::config::Fields;
use nn::family::Stage;
use nn::quantization::ModeloptManifest;
use processing::qwen_vl::ProcessorConfig;
use serde_json::Value;

pub use audio_vae::AudioVaeConfig;
pub use denoiser::{DenoiserConfig, ScheduleConfig, TransformerConfig, VsaConfig};
pub use processing::minimax_h3::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_LATENTS_PER_SECOND, AUDIO_SAMPLE_RATE, FPS, FRAMES,
    LATENT_COMPRESSION, PATCH_SIDE,
};
pub use text::RETAINED_LAYERS;
pub use video_vae::VideoVaeConfig;

use crate::{Error, Result, VideoTask};

/// The video VAE's temporal compression: one latent frame covers this many video frames.
pub const TEMPORAL_COMPRESSION: u32 = 4;

/// The longest generated clip a deployment's bounds are taken over, in frames.
pub const MAX_FRAMES: u32 = 1024;

/// Steps of the training schedule a DMD student's rungs are stated on.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Sigma points of the released full-step grid, the clean endpoint included.
pub const UNIFORM_GRID_POINTS: u32 = 50;

/// The networks a release may hold, of which a deployment places one.
pub const NETWORKS: [Stage; 2] = [Stage::Transformer, Stage::TransformerRef];

/// The directory of the video VAE, which holds the decoder's weights and the encoder's.
pub const VIDEO_VAE: &str = "vae";

/// The rungs a fused FastH3 adapter's student was distilled with: the four DMD rungs of the
/// `t2av` network at the base release's shifts
/// (`FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`).
const ADAPTER_RUNGS: [u32; 4] = [999, 749, 500, 250];

/// The sparse attention of a fused FastH3 adapter that assigns the blocks' compression gates:
/// 64-row tiles at a sparsity of 0.9.
const ADAPTER_VSA: (u32, f64) = (64, 0.9);

/// The network that serves `task`: the base `transformer` for `t2va` and `fl2va`, the
/// `transformer_ref` for `ref2va`.
pub const fn network(task: VideoTask) -> Stage {
    match task {
        VideoTask::TextToVideoAudio | VideoTask::FirstLastToVideoAudio => Stage::Transformer,
        VideoTask::ReferenceToVideoAudio => Stage::TransformerRef,
    }
}

/// A checkpoint's typed configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// The Qwen3-VL text encoder that conditions a request.
    pub text_encoder: qwen3::Config,
    /// The pixel normalization the text encoder's processor applies.
    pub pixels: PixelNorm,
    /// The text encoder's Qwen-VL processor sizes, `None` when the checkpoint states no
    /// `processor/video_preprocessor_config.json`.
    pub processor: Option<ProcessorConfig>,
    /// Each network the checkpoint holds.
    pub denoisers: BTreeMap<Stage, DenoiserConfig>,
    /// The video VAE: the decoder every task uses and the encoder a conditioned task uses.
    pub video_vae: VideoVaeConfig,
    /// The audio VAE: the vocoder every task uses and the encoder `ref2va` uses.
    pub audio_vae: AudioVaeConfig,
}

impl Config {
    /// Reads and normalizes the configuration of `checkpoint`, once.
    ///
    /// Each sidecar is read from the first root of the checkpoint that holds it, so an export
    /// that continues a base release reads what it omits from the base. A FastH3 adapter fused
    /// into the base `transformer` reads as the student it was extracted from.
    ///
    /// # Errors
    ///
    /// A sidecar that is missing or states a value this crate does not serve.
    pub fn open(checkpoint: &nn::Checkpoint) -> Result<Self> {
        let adapter = checkpoint
            .stage(Stage::Transformer.name())
            .ok()
            .and_then(nn::Checkpoint::adapter)
            .map(|fusion| {
                fusion
                    .assigned()
                    .any(|name| name.ends_with(".attn.to_gate_compress.weight"))
            });
        let root = checkpoint.root().display().to_string();

        Self::read(|relative| Ok(checkpoint.sidecar(relative)?), adapter, &root)
    }

    /// The configuration of the sidecars `sidecar` returns by checkpoint-relative path.
    ///
    /// `adapter` is whether a fused FastH3 adapter assigns compression gates, `None` when none
    /// is fused; `root` names the checkpoint in refusals. Everything is checked here (the text
    /// encoder against the networks' conditioning width, the VAEs' compression against the
    /// frame rule, the student's statement against its schedulers), so no module sees a value
    /// it cannot use.
    pub(crate) fn read(
        sidecar: impl Fn(&str) -> Result<Option<Value>>,
        adapter: Option<bool>,
        root: &str,
    ) -> Result<Self> {
        let require = |relative: &str| -> Result<Value> {
            sidecar(relative)?.ok_or_else(|| {
                Error::checkpoint(format!(
                    "no root of the checkpoint at {root} holds {relative}"
                ))
            })
        };

        // A packed export's calibration, which every stage's declaration is read from.
        let manifest = sidecar(ModeloptManifest::FILE)?
            .map(|manifest| ModeloptManifest::parse(&manifest))
            .transpose()?;
        if let Some(manifest) = &manifest {
            modelopt::check(manifest)?;
        }
        let declared = |directory: &str| modelopt::declaration(manifest.as_ref(), directory);

        let mut text_encoder = text::text_encoder(&require("text_encoder/config.json")?)?;
        text_encoder.quantization = declared(Stage::TextEncoder.name())?;
        let image_processor = require("processor/preprocessor_config.json")?;
        let pixels = qwen3::config::pixel_norm(&Fields::of(
            "processor/preprocessor_config.json",
            &image_processor,
        )?)?;
        let processor = sidecar("processor/video_preprocessor_config.json")?
            .map(|video| ProcessorConfig::from_values(&image_processor, &video))
            .transpose()
            .map_err(|error| Error::config(format!("the checkpoint's processor: {error}")))?;

        let mut video_vae = VideoVaeConfig::parse(&require(&format!("{VIDEO_VAE}/config.json"))?)?;
        video_vae.quantization = declared(VIDEO_VAE)?;
        let audio_vae = AudioVaeConfig::parse(&require("audio_vae/config.json")?)?;
        let shifts = (
            shift(&require("scheduler/scheduler_config.json")?, "scheduler")?,
            shift(
                &require("audio_scheduler/scheduler_config.json")?,
                "audio_scheduler",
            )?,
        );

        let student = sidecar("fastvideo_inference.json")?;
        let student = student
            .as_ref()
            .map(|value| Fields::of("fastvideo_inference.json", value))
            .transpose()?;
        let networks = match (&student, adapter) {
            (Some(student), _) => vec![student_network(student)?],
            (None, Some(_)) => vec![Stage::Transformer],
            (None, None) => released_networks(&sidecar, root)?,
        };

        let mut denoisers = BTreeMap::new();
        for stage in networks {
            let relative = format!("{}/config.json", stage.name());
            let transformer =
                TransformerConfig::parse(&Fields::of(&relative, &require(&relative)?)?)?;

            let (schedule, vsa) = match (&student, adapter) {
                (Some(student), _) => student_denoising(student, &transformer, shifts)?,
                (None, Some(gated)) => (
                    ScheduleConfig::ladder(
                        &ADAPTER_RUNGS,
                        NUM_TRAIN_TIMESTEPS,
                        shifts.0,
                        shifts.1,
                    )?,
                    gated
                        .then(|| VsaConfig::new(ADAPTER_VSA.0, ADAPTER_VSA.1, None))
                        .transpose()?,
                ),
                (None, None) => (
                    ScheduleConfig::uniform(UNIFORM_GRID_POINTS, shifts.0, shifts.1)?,
                    None,
                ),
            };

            let tasks = VideoTask::ALL
                .into_iter()
                .filter(|task| network(*task) == stage)
                .collect();
            let mut denoiser = DenoiserConfig::new(transformer, schedule, vsa, tasks)?;
            denoiser.quantization = declared(stage.name())?;
            denoisers.insert(stage, denoiser);
        }

        let config = Self {
            text_encoder,
            pixels,
            processor,
            denoisers,
            video_vae,
            audio_vae,
        };
        config.validate()?;
        Ok(config)
    }

    /// The text encoder's vision tower, which every H3 text encoder carries.
    ///
    /// # Errors
    ///
    /// A configuration built without one, which the reader refuses.
    pub fn vision(&self) -> Result<&qwen3::VisionConfig> {
        self.text_encoder.vision.as_ref().ok_or_else(|| {
            Error::config("H3's text encoder is a Qwen3-VL model with a vision tower".to_owned())
        })
    }

    /// The tasks the checkpoint can serve: each network's, in canonical order.
    pub fn tasks(&self) -> Vec<VideoTask> {
        VideoTask::ALL
            .into_iter()
            .filter(|task| self.denoisers.contains_key(&network(*task)))
            .collect()
    }

    /// Checks the stages against each other and against the frame rule.
    fn validate(&self) -> Result<()> {
        // The packing and the reconstruction are written for one clip rule, the frame rule's,
        // at 16x spatial and 4x temporal compression.
        let video = &self.video_vae;
        let stated = (
            video.clip_length,
            video.clip_latents(),
            video.final_clip_latents(),
            video.spatial_compression(),
            video.temporal_compression(),
        );
        let served = (
            FRAMES.clip_frames,
            FRAMES.clip_latent_frames,
            FRAMES.tail_latent_frames,
            LATENT_COMPRESSION,
            TEMPORAL_COMPRESSION,
        );
        if stated != served {
            return Err(Error::config(format!(
                "the MiniMax-H3 video_vae must encode {}-frame clips to {} latent frames, \
                 keeping {} of them in the final clip, at {LATENT_COMPRESSION}x spatial and \
                 {TEMPORAL_COMPRESSION}x temporal compression; this checkpoint states {} frames \
                 to {} latents, keeping {}, at {}x spatial and {}x temporal",
                served.0, served.1, served.2, stated.0, stated.1, stated.2, stated.3, stated.4
            )));
        }

        // Soundtracks are resampled to the request rule's sample rate, so the audio VAE must
        // run at it and hop the rule's latent step.
        let audio = &self.audio_vae;
        if audio.sampling_rate != AUDIO_SAMPLE_RATE {
            return Err(Error::config(format!(
                "the MiniMax-H3 audio_vae samples at {} Hz; the request rule's audio runs at \
                 {AUDIO_SAMPLE_RATE} Hz",
                audio.sampling_rate
            )));
        }
        if audio.hop() != AUDIO_HOP || audio.decoder_hop() != AUDIO_HOP {
            return Err(Error::config(format!(
                "the MiniMax-H3 audio_vae must hop {AUDIO_HOP} samples per latent; this \
                 checkpoint encodes {} and decodes {}",
                audio.hop(),
                audio.decoder_hop()
            )));
        }

        // Each network's latent widths and conditioning width are the VAEs' and the text
        // encoder's.
        for (stage, denoiser) in &self.denoisers {
            let transformer = &denoiser.transformer;

            if self.text_encoder.text.hidden_size != transformer.text_dim {
                return Err(Error::config(format!(
                    "H3 text features are {} wide and the {} network conditions on {}",
                    self.text_encoder.text.hidden_size,
                    stage.name(),
                    transformer.text_dim
                )));
            }
            if video.latent_channels != transformer.video_channels {
                return Err(Error::config(format!(
                    "H3 video latents have {} channels and the {} network reads {}",
                    video.latent_channels,
                    stage.name(),
                    transformer.video_channels
                )));
            }
            if audio.latent_channels != transformer.audio_channels {
                return Err(Error::config(format!(
                    "H3 audio latents have {} channels and the {} network reads {}",
                    audio.latent_channels,
                    stage.name(),
                    transformer.audio_channels
                )));
            }
        }
        Ok(())
    }
}

/// The networks a base release holds: those its pipeline index names and whose configuration
/// the checkpoint holds.
fn released_networks(
    sidecar: &impl Fn(&str) -> Result<Option<Value>>,
    root: &str,
) -> Result<Vec<Stage>> {
    let class = processing::minimax_h3::PIPELINE_CLASS;

    let index = sidecar("model_index.json")?.ok_or_else(|| {
        Error::checkpoint(format!(
            "{root} holds neither fastvideo_inference.json nor model_index.json"
        ))
    })?;
    let index = Fields::of("model_index.json", &index)?;
    if let Some(stated) = index.optional_text("_class_name")?
        && stated != class
    {
        return Err(Error::checkpoint(format!(
            "{root} is a {stated} checkpoint, not a {class} pipeline root; the task exports \
             under FL2VA/ are not pipeline roots"
        )));
    }

    let mut networks = Vec::new();
    for stage in NETWORKS {
        let held = sidecar(&format!("{}/config.json", stage.name()))?.is_some();
        if index.contains(stage.name()) && held {
            networks.push(stage);
        }
    }
    if networks.is_empty() {
        return Err(Error::checkpoint(format!(
            "{root}/model_index.json names no network the checkpoint holds"
        )));
    }
    Ok(networks)
}

/// The network a FastH3 export distils: the base network of the task it states (`task`, or
/// `model_type` in the OmniRef exports).
fn student_network(student: &Fields<'_>) -> Result<Stage> {
    let stated = match student.optional_text("task")? {
        Some(task) => task,
        None => student.optional_text("model_type")?.ok_or_else(|| {
            Error::checkpoint("fastvideo_inference.json states no task or model_type".to_owned())
        })?,
    };

    match stated {
        "t2av" => Ok(network(VideoTask::TextToVideoAudio)),
        "ref2va" => Ok(network(VideoTask::ReferenceToVideoAudio)),
        other => Err(Error::checkpoint(format!(
            "unsupported FastH3 checkpoint: its task must be t2av or ref2va, got {other:?}"
        ))),
    }
}

/// A FastH3 export's schedule and sparse attention, from its `fastvideo_inference.json`.
///
/// The shifts are the scheduler sidecars'; an export that states them too must state the same
/// two, and one whose statements disagree is refused. Its schedule is a DMD ladder (`dmd_denoising_steps`) or a
/// parallel-decoding block grid (`pdd_step_indices` with `grid_max_t`), exactly one of them;
/// its attention backend is `DENSE` (or unstated) or `VIDEO_SPARSE_ATTN_H3` with the `vsa_*`
/// values.
fn student_denoising(
    student: &Fields<'_>,
    transformer: &TransformerConfig,
    shifts: (f32, f32),
) -> Result<(ScheduleConfig, Option<VsaConfig>)> {
    for (key, shift) in [
        ("video_scheduler_shift", shifts.0),
        ("audio_scheduler_shift", shifts.1),
    ] {
        if let Some(stated) = student.optional_number(key)?
            && stated as f32 != shift
        {
            return Err(Error::config(format!(
                "fastvideo_inference.json.{key} is {stated}, which disagrees with the \
                 scheduler sidecar's shift of {shift}"
            )));
        }
    }

    let schedule = match (
        student.optional_indices("dmd_denoising_steps")?,
        student.optional_indices("pdd_step_indices")?,
    ) {
        (Some(rungs), None) => {
            ScheduleConfig::ladder(&rungs, NUM_TRAIN_TIMESTEPS, shifts.0, shifts.1)?
        }
        (None, Some(nodes)) => ScheduleConfig::blocks(
            transformer.output_heads,
            nodes,
            student.number("grid_max_t")? as f32,
            shifts.0,
            shifts.1,
        )?,
        (Some(_), Some(_)) => {
            return Err(Error::checkpoint(
                "fastvideo_inference.json states both dmd_denoising_steps and pdd_step_indices"
                    .to_owned(),
            ));
        }
        (None, None) => {
            return Err(Error::checkpoint(
                "unsupported FastH3 checkpoint: fastvideo_inference.json states neither \
                 dmd_denoising_steps nor pdd_step_indices"
                    .to_owned(),
            ));
        }
    };

    let vsa = match student.optional_text("attention_backend")? {
        None | Some("DENSE") => None,
        Some("VIDEO_SPARSE_ATTN_H3") => Some(VsaConfig::new(
            student.index("vsa_tile_size")?,
            student.number("vsa_sparsity")?,
            student.optional_number("vsa_ref_keep_rate")?,
        )?),
        Some(other) => {
            return Err(Error::checkpoint(format!(
                "fastvideo_inference.json states the attention backend {other:?}, which this \
                 build does not serve: it serves DENSE and VIDEO_SPARSE_ATTN_H3"
            )));
        }
    };
    Ok((schedule, vsa))
}

/// The noise level shift the scheduler sidecar of `stage` states.
fn shift(sidecar: &Value, stage: &str) -> Result<f32> {
    let owner = format!("{stage}/scheduler_config.json");

    Ok(Fields::of(owner, sidecar)?.positive("shift")? as f32)
}

/// One per-channel latent statistic of a VAE sidecar: a number for every latent channel.
fn statistics(values: &Fields<'_>, key: &str, channels: u32) -> Result<Vec<f32>> {
    Ok(values
        .fixed_numbers(key, channels as usize)?
        .into_iter()
        .map(|value| value as f32)
        .collect())
}

/// The released checkpoint's sidecars, as the tests of this crate use them.
///
/// They are the values `docs/minimax_h3/minimax_h3.md` and the released checkpoints state: a
/// 64-layer Qwen3-VL text encoder of which H3 keeps 50, a 50-block network conditioned on
/// 5120-wide text, a 17-frame video VAE of 24 latent channels, and a 32-channel audio VAE at
/// 32 kHz.
#[cfg(test)]
pub(crate) mod tests_support {
    use std::collections::BTreeMap;

    use serde_json::Value;

    use super::*;

    /// A fixture's JSON.
    pub(crate) fn json(body: &str) -> Value {
        serde_json::from_str(body).unwrap()
    }

    /// The text encoder checkpoint's configuration.
    pub(crate) fn text_encoder() -> Value {
        json(
            r#"{
                "architectures": ["Qwen3VLForConditionalGeneration"],
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
                    "rope_scaling": {
                        "mrope_interleaved": true,
                        "mrope_section": [24, 20, 20],
                        "rope_type": "default"
                    },
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
            "latents_mean": vec![0.0_f64; 32],
            "latents_std": vec![1.0_f64; 32],
        })
    }

    /// One network's configuration.
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

    /// The base release's sidecars, by checkpoint-relative path: a pipeline index over both
    /// networks.
    pub(crate) fn release() -> BTreeMap<String, Value> {
        let index = r#"{"_class_name": "MiniMaxH3ModularPipeline",
                        "transformer": [], "transformer_ref": []}"#;

        [
            ("model_index.json", json(index)),
            ("text_encoder/config.json", text_encoder()),
            ("processor/preprocessor_config.json", processor()),
            ("vae/config.json", video_vae()),
            ("audio_vae/config.json", audio_vae()),
            (
                "scheduler/scheduler_config.json",
                json(r#"{"shift": 12.0}"#),
            ),
            (
                "audio_scheduler/scheduler_config.json",
                json(r#"{"shift": 3.0}"#),
            ),
            ("transformer/config.json", transformer()),
            ("transformer_ref/config.json", transformer()),
        ]
        .into_iter()
        .map(|(path, value)| (path.to_owned(), value))
        .collect()
    }

    /// A FastH3 DMD export of the base network over `files`: its inference file, its
    /// scheduler's shift and no reference network.
    pub(crate) fn dmd_export(
        mut files: BTreeMap<String, Value>,
        backend: &str,
    ) -> BTreeMap<String, Value> {
        files.remove("model_index.json");
        files.remove("transformer_ref/config.json");
        files.insert(
            "scheduler/scheduler_config.json".to_owned(),
            json(r#"{"shift": 10.0}"#),
        );
        files.insert(
            "fastvideo_inference.json".to_owned(),
            serde_json::json!({
                "task": "t2av",
                "video_scheduler_shift": 10.0,
                "audio_scheduler_shift": 3.0,
                "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
                "attention_backend": backend,
                "vsa_tile_size": 64,
                "vsa_sparsity": 0.8,
            }),
        );
        files
    }

    /// The configuration of `files`, with a fused adapter's gating when one is stated.
    pub(crate) fn read(files: &BTreeMap<String, Value>, adapter: Option<bool>) -> Result<Config> {
        Config::read(
            |relative| Ok(files.get(relative).cloned()),
            adapter,
            "fixture",
        )
    }

    /// The released base configuration.
    pub(crate) fn released_config() -> Config {
        read(&release(), None).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;

    #[test]
    fn a_base_release_reads_both_networks_with_the_uniform_grid() {
        let config = released_config();

        assert_eq!(config.denoisers.len(), 2);
        assert_eq!(config.tasks(), VideoTask::ALL.to_vec());

        let denoiser = &config.denoisers[&Stage::TransformerRef];
        assert_eq!(denoiser.tasks, vec![VideoTask::ReferenceToVideoAudio]);
        assert_eq!(denoiser.vsa, None);
        assert_eq!(
            denoiser.schedule,
            ScheduleConfig::uniform(UNIFORM_GRID_POINTS, 12.0, 3.0).unwrap()
        );

        let base = &config.denoisers[&Stage::Transformer];
        assert_eq!(
            base.tasks,
            [
                VideoTask::TextToVideoAudio,
                VideoTask::FirstLastToVideoAudio
            ]
        );
        assert_eq!(base.capacity_unit(), 64);
        assert!(base.conditioned());
    }

    #[test]
    fn a_task_export_or_an_empty_root_is_not_a_pipeline_root() {
        let mut export = release();
        export.insert(
            "model_index.json".to_owned(),
            json(r#"{"_class_name": "MiniMaxH3Pipeline", "transformer": []}"#),
        );
        let error = read(&export, None).unwrap_err();
        assert!(
            error.to_string().contains("not a MiniMaxH3ModularPipeline"),
            "{error}"
        );

        let mut empty = release();
        empty.remove("model_index.json");
        let error = read(&empty, None).unwrap_err();
        assert!(error.to_string().contains("model_index.json"), "{error}");
    }

    #[test]
    fn a_release_holds_the_networks_its_index_names_and_its_root_holds() {
        let mut files = release();
        files.remove("transformer_ref/config.json");
        let config = read(&files, None).unwrap();
        assert_eq!(
            config.denoisers.keys().collect::<Vec<_>>(),
            [&Stage::Transformer]
        );

        files.remove("transformer/config.json");
        assert!(read(&files, None).is_err(), "no network is held");
    }

    #[test]
    fn a_dmd_export_reads_its_ladder_and_its_sparse_attention() {
        let config = read(&dmd_export(release(), "VIDEO_SPARSE_ATTN_H3"), None).unwrap();
        assert_eq!(
            config.denoisers.keys().collect::<Vec<_>>(),
            [&Stage::Transformer]
        );

        let denoiser = &config.denoisers[&Stage::Transformer];
        assert_eq!(denoiser.schedule.num_steps(), 8);
        assert_eq!(denoiser.vsa, Some(VsaConfig::new(64, 0.8, None).unwrap()));
        assert_eq!(denoiser.capacity_unit(), 64);
    }

    /// The export states its backend, and the statement decides: a student that kept its
    /// distillation's sparse values while naming dense attention attends densely, and an
    /// unknown backend is refused by name.
    #[test]
    fn an_export_attends_with_the_backend_it_states() {
        let config = read(&dmd_export(release(), "DENSE"), None).unwrap();
        assert_eq!(config.denoisers[&Stage::Transformer].vsa, None);

        let error = read(&dmd_export(release(), "FLASH"), None).unwrap_err();
        assert!(error.to_string().contains("FLASH"), "{error}");
    }

    #[test]
    fn an_export_states_one_schedule_of_a_served_task() {
        let mut files = dmd_export(release(), "DENSE");
        files.get_mut("fastvideo_inference.json").unwrap()["task"] = json(r#""t2v""#);
        assert!(read(&files, None).is_err(), "t2v is not served");

        let mut files = dmd_export(release(), "DENSE");
        let student = files.get_mut("fastvideo_inference.json").unwrap();
        student["pdd_step_indices"] = json("[0, 8, 16, 32]");
        assert!(read(&files, None).is_err(), "two schedules");
    }

    #[test]
    fn a_fused_adapter_is_the_four_rung_student_it_was_extracted_from() {
        for (gated, vsa) in [
            (true, Some(VsaConfig::new(64, 0.9, None).unwrap())),
            (false, None),
        ] {
            let config = read(&release(), Some(gated)).unwrap();
            assert_eq!(
                config.denoisers.keys().collect::<Vec<_>>(),
                [&Stage::Transformer]
            );

            let denoiser = &config.denoisers[&Stage::Transformer];
            assert_eq!(denoiser.schedule.num_steps(), 4);
            assert_eq!(denoiser.vsa, vsa);
        }
    }

    #[test]
    fn a_configuration_refuses_stages_that_disagree_with_each_other() {
        let mut widths = release();
        let encoder = widths.get_mut("text_encoder/config.json").unwrap();
        encoder["text_config"]["hidden_size"] = json("4096");
        encoder["vision_config"]["out_hidden_size"] = json("4096");
        let error = read(&widths, None).unwrap_err();
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("conditions on")),
            "{error}"
        );

        let mut channels = release();
        let statistics = |value: f64| Value::Array((0..16).map(|_| Value::from(value)).collect());
        let vae = channels.get_mut("vae/config.json").unwrap();
        vae["latent_channels"] = json("16");
        vae["latents_mean"] = statistics(0.0);
        vae["latents_std"] = statistics(1.0);
        let error = read(&channels, None).unwrap_err();
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("channels")),
            "{error}"
        );
    }

    #[test]
    fn a_configuration_refuses_a_vae_the_frame_rule_cannot_serve() {
        let mut compression = release();
        compression.get_mut("vae/config.json").unwrap()["temporal_downsample_factors"] =
            json("[1, 2, 1, 1, 1, 1]");
        let error = read(&compression, None).unwrap_err();
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("temporal")),
            "{error}"
        );

        let mut clock = release();
        clock.get_mut("audio_vae/config.json").unwrap()["sampling_rate"] = json("44100");
        let error = read(&clock, None).unwrap_err();
        assert!(
            matches!(error, Error::Config(ref problem)
                if problem.contains("44100") && problem.contains("32000 Hz")),
            "{error}"
        );
    }

    /// The four-step preview export states no shift: its schedule takes the scheduler
    /// sidecars' shifts.
    #[test]
    fn an_export_that_states_no_shift_takes_its_schedulers() {
        let mut files = dmd_export(release(), "DENSE");
        let student = files.get_mut("fastvideo_inference.json").unwrap();
        let student = student.as_object_mut().unwrap();
        student.remove("video_scheduler_shift");
        student.remove("audio_scheduler_shift");

        let config = read(&files, None).unwrap();
        let schedule = &config.denoisers[&Stage::Transformer].schedule;
        assert_eq!((schedule.video_shift, schedule.audio_shift), (10.0, 3.0));
    }

    #[test]
    fn an_export_whose_shifts_disagree_with_its_schedulers_is_refused() {
        let mut files = dmd_export(release(), "DENSE");
        // The scheduler states the base release's video shift of 12, not the student's 10.
        files.insert(
            "scheduler/scheduler_config.json".to_owned(),
            json(r#"{"shift": 12.0}"#),
        );
        let error = read(&files, None).unwrap_err();
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("disagree")),
            "{error}"
        );
    }

    #[test]
    fn the_processors_normalization_is_read_with_the_patch_sizes() {
        let config = released_config();

        assert_eq!(config.pixels.mean, [0.5, 0.5, 0.5]);
        assert_eq!(config.pixels.std, [0.5, 0.5, 0.5]);
        assert_eq!(config.processor, None, "no video processor file is stated");

        let mut unscaled = release();
        unscaled.insert(
            "processor/preprocessor_config.json".to_owned(),
            json(r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.0, 0.5]}"#),
        );
        assert!(read(&unscaled, None).is_err());
    }

    /// The released four-step export's manifest, trimmed to the first block of each calibrated
    /// stage: the fixture the manifest's reader is tested on.
    fn manifest() -> Value {
        json(include_str!(
            "../../../../nn/tests/fixtures/modelopt_manifest.json"
        ))
    }

    /// A packed export's manifest becomes each stage's declaration: the network's feed-forward
    /// projections and the video decoder's attention and feed-forward projections are
    /// calibrated under the names their stages store them by, and the text encoder, which the
    /// manifest lists and does not calibrate, is dense.
    #[test]
    fn a_packed_export_declares_the_projections_each_stage_stores_packed() {
        let mut files = dmd_export(release(), "VIDEO_SPARSE_ATTN_H3");
        files.insert(ModeloptManifest::FILE.to_owned(), manifest());
        let config = read(&files, None).unwrap();

        // An activation's tensor scale is its calibrated magnitude over 6 · 448.
        let scale = |amax: f64| Some((amax / 2688.0) as f32);
        let network = &config.denoisers[&Stage::Transformer].quantization;
        assert!(network.is_encoded());
        for (module, amax) in [
            ("ff.net.0.proj", 485.011_444_091_796_9),
            ("ff.net.2", 602.315_246_582_031_2),
        ] {
            assert_eq!(
                network
                    .activation_scale(&format!("transformer_blocks.0.{module}"))
                    .unwrap(),
                scale(amax),
                "{module}"
            );
        }

        let decoder = &config.video_vae.quantization;
        for (module, amax) in [
            ("attn.to_q", 5_855.696_289_062_5),
            ("attn.to_k", 5_855.696_289_062_5),
            ("attn.to_v", 5_855.696_289_062_5),
            ("attn.to_out.0", 4_231.270_019_531_25),
            ("ff.net.0.proj", 3_796.949_218_75),
            ("ff.net.2", 540.490_356_445_312_5),
        ] {
            assert_eq!(
                decoder
                    .activation_scale(&format!("decoder.transformer_blocks.0.{module}"))
                    .unwrap(),
                scale(amax),
                "{module}"
            );
        }

        assert_eq!(
            config.text_encoder.quantization,
            nn::quantization::Declaration::dense()
        );
    }

    /// A calibration this crate has no stored projection for is refused by the path the
    /// manifest lists, and a manifest the reader refuses is refused as the reader names it.
    #[test]
    fn a_manifest_this_family_cannot_place_is_refused_by_the_listed_path() {
        let mut files = dmd_export(release(), "VIDEO_SPARSE_ATTN_H3");
        let listed = "text_encoder.network.layers.0.mlp.down";

        let mut unplaced = manifest();
        unplaced["components"]["text_encoder"] = serde_json::json!({
            "enabled": true,
            "modules": {listed: {"activation_amax": 12.5}},
        });
        files.insert(ModeloptManifest::FILE.to_owned(), unplaced);
        let error = read(&files, None).unwrap_err();
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains(listed)),
            "{error}"
        );

        let mut inconsistent = manifest();
        inconsistent["components"]["video_vae"]["enabled"] = serde_json::json!(false);
        files.insert(ModeloptManifest::FILE.to_owned(), inconsistent);
        let error = read(&files, None).unwrap_err();
        assert!(
            matches!(error, Error::Nn(nn::Error::Config { ref field, .. }) if field == "modules"),
            "{error}"
        );
    }
}

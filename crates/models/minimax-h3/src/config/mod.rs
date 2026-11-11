//! MiniMax-H3's typed configuration: the checkpoint's architecture, read once at the loading
//! boundary and immutable afterwards.
//!
//! A checkpoint is normalized into one configuration before any module is built, for the fields
//! the model reads: the base release (a pipeline index over the DiT partitions, text encoder,
//! video VAE and audio VAE it holds), or a FastH3 export or adapter (one DiT partition whose
//! schedule and attention its `fastvideo_inference.json`, or the release it was extracted from,
//! states). The request-level values the family-neutral request rules are parameterized by -
//! the frame rule, the canvas rule, the durations and the audio clock - are request processing's
//! ([`processing::minimax_h3`]); this configuration checks that the checkpoint agrees with them.

mod audio_vae;
mod denoiser;
mod text;
mod video_vae;

use std::collections::BTreeMap;

use nn::config::Fields;
use serde_json::Value;

use crate::checkpoint::PIPELINE_CLASS;
use crate::packing::LATENT_COMPRESSION;
use crate::{Error, Result, VideoTask};
use nn::family::Stage;

pub use audio_vae::AudioVaeConfig;
pub use denoiser::{
    AttentionKind, DenoiserConfig, SPARSE_TILE_ROWS, ScheduleGrid, ScheduleSpec, Shifts,
    TransformerConfig,
};
pub use processing::minimax_h3::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_LATENTS_PER_SECOND, AUDIO_SAMPLE_RATE, CANVAS_RULE, FPS,
    FRAMES, MAX_SECONDS, MIN_SECONDS, NAMED_ASPECT_RATIOS,
};
pub use text::RETAINED_LAYERS;
pub use video_vae::VideoVaeConfig;

/// The video VAE's temporal compression: one latent frame covers this many video frames.
pub const TEMPORAL_COMPRESSION: u32 = 4;

/// The longest generated clip a deployment's bounds are taken over, in frames.
///
/// Every capacity a deployment sizes - the packed row bound, a decoder's canvas extent, the audio
/// encoder's attention - is taken over a request no longer than this.
pub const MAX_FRAMES: u32 = 1024;

/// Steps of the training schedule a DMD student's rungs are stated on.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Sigma points of the released full-step grid, the clean endpoint included.
pub const UNIFORM_GRID_POINTS: u32 = 50;

/// The canvases every H3 checkpoint generates: the canvas rule's canvas of each named aspect ratio,
/// with the ratio, in [`NAMED_ASPECT_RATIOS`]' order (1536x672 for 21:9 to 768x1344 for 9:16).
///
/// # Errors
///
/// Fails when the rule refuses a named ratio.
pub fn named_canvases() -> Result<Vec<((u32, u32), proto::video::Canvas)>> {
    NAMED_ASPECT_RATIOS
        .iter()
        .map(|&(width, height)| {
            CANVAS_RULE
                .canvas(f64::from(width), f64::from(height))
                .map(|canvas| ((width, height), canvas))
                .map_err(|error| Error::config(format!("the canvas rule: {error}")))
        })
        .collect()
}

/// The FastH3 rungs and attention a fused adapter's student was distilled with: the four DMD
/// rungs of the `t2av` partition at the base release's shifts, and VSA at 64-row tiles and 0.9
/// sparsity when the adapter assigns the layers' compression gates
/// (`FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`).
const ADAPTER_RUNGS: [u32; 4] = [999, 749, 500, 250];

/// A checkpoint's typed configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// The Qwen3-VL text encoder that conditions a request.
    pub text_encoder: qwen3::Config,
    /// The pixel normalization the text encoder's processor applies.
    pub pixels: nn::capability::PixelNorm,
    /// Each DiT partition the checkpoint holds, by network.
    pub denoisers: BTreeMap<Stage, DenoiserConfig>,
    /// The video VAE: the decoder every task uses and the encoder `ref2va` uses.
    pub video_vae: VideoVaeConfig,
    /// The audio VAE: the BigVGAN decoder every task uses and the encoder `ref2va` uses.
    pub audio_vae: AudioVaeConfig,
    /// The text encoder's Qwen-VL processor geometry, `None` when the checkpoint states no
    /// `processor/video_preprocessor_config.json`.
    pub processor: Option<processing::qwen_vl::ProcessorConfig>,
}

impl Config {
    /// Read and normalize the configuration of `checkpoint`, once.
    ///
    /// Each sidecar is read from the first root of the checkpoint that holds it, so an export that
    /// continues a base release reads what it omits from the base. A FastH3 adapter fused into the
    /// base `transformer` reads as the student it was extracted from.
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

    /// Build the configuration from `sidecar`, which answers a checkpoint-relative sidecar or
    /// `None`. `adapter` is whether a fused FastH3 adapter assigns compression gates, `None` when
    /// none is fused; `root` names the checkpoint in refusals.
    ///
    /// This is the whole of the loading boundary's reading: everything is checked here - the
    /// text encoder against the DiT's conditioning width, the VAEs' compression against the
    /// timeline the packing assumes, the student's statement against its schedulers - so no layer
    /// below sees a value it cannot use.
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
        let text_encoder = text::text_encoder(&require("text_encoder/config.json")?)?;
        let image_processor = require("processor/preprocessor_config.json")?;
        let pixels = qwen3::vision::pixel_norm(&Fields::of(
            "processor/preprocessor_config.json",
            &image_processor,
        )?)?;
        let processor = sidecar("processor/video_preprocessor_config.json")?
            .map(|video| {
                processing::qwen_vl::ProcessorConfig::from_values(&image_processor, &video)
            })
            .transpose()
            .map_err(|error| Error::config(format!("the checkpoint's processor: {error}")))?;
        let video_vae = VideoVaeConfig::parse(&require("vae/config.json")?)?;
        let audio_vae = AudioVaeConfig::parse(&require("audio_vae/config.json")?)?;
        let shifts = Shifts {
            video: shift(
                &require("scheduler/scheduler_config.json")?,
                "scheduler/scheduler_config.json",
            )?,
            audio: shift(
                &require("audio_scheduler/scheduler_config.json")?,
                "audio_scheduler/scheduler_config.json",
            )?,
        };

        let student = sidecar("fastvideo_inference.json")?;
        let student = student
            .as_ref()
            .map(|value| Fields::of("fastvideo_inference.json", value))
            .transpose()?;
        let dits = match (&student, adapter) {
            (Some(student), _) => vec![student_dit(student)?],
            (None, Some(_)) => vec![Stage::Transformer],
            (None, None) => released_dits(&sidecar, root)?,
        };

        let mut denoisers = BTreeMap::new();
        for dit in dits {
            let relative = format!("{}/config.json", dit.name());
            let transformer =
                TransformerConfig::parse(&Fields::of(&relative, &require(&relative)?)?)?;
            let (schedule, attention) = match (&student, adapter) {
                (Some(student), _) => student_denoising(student, &transformer, shifts)?,
                (None, Some(gated)) => (
                    ScheduleSpec::dmd(ADAPTER_RUNGS.to_vec(), NUM_TRAIN_TIMESTEPS, shifts)?,
                    if gated {
                        AttentionKind::sparse(64, 0.9, None)?
                    } else {
                        AttentionKind::Dense
                    },
                ),
                (None, None) => (
                    ScheduleSpec::uniform(UNIFORM_GRID_POINTS, shifts)?,
                    AttentionKind::Dense,
                ),
            };
            let denoiser = DenoiserConfig::new(
                transformer,
                schedule,
                attention,
                crate::checkpoint::dit_tasks(dit),
            )?;
            denoisers.insert(dit, denoiser);
        }

        let config = Self {
            text_encoder,
            pixels,
            denoisers,
            video_vae,
            audio_vae,
            processor,
        };
        config.validate()?;
        Ok(config)
    }

    /// The tasks the checkpoint can serve: each DiT's, in canonical order.
    pub fn tasks(&self) -> Vec<VideoTask> {
        VideoTask::ALL
            .into_iter()
            .filter(|task| self.denoisers.contains_key(&crate::checkpoint::dit(*task)))
            .collect()
    }

    /// The text encoder's vision tower, which every H3 text encoder carries.
    ///
    /// # Errors
    ///
    /// A configuration built without one, which the reader refuses.
    pub fn vision(&self) -> Result<&qwen3::config::VisionConfig> {
        self.text_encoder.vision.as_ref().ok_or_else(|| {
            Error::config("H3's text encoder is a Qwen3-VL model with a vision tower".to_owned())
        })
    }

    /// Check the components against each other and against the timeline.
    fn validate(&self) -> Result<()> {
        // H3's packing and reconstruction hard-code one clip timeline, the frame rule's, at 16x
        // spatial and 4x temporal compression.
        let video = &self.video_vae;
        if (
            video.clip_length,
            video.clip_latents(),
            video.final_clip_latents(),
            video.spatial_compression(),
            video.temporal_compression(),
        ) != (
            FRAMES.clip_frames,
            FRAMES.clip_latent_frames,
            FRAMES.tail_latent_frames,
            LATENT_COMPRESSION,
            TEMPORAL_COMPRESSION,
        ) {
            return Err(Error::config(format!(
                "MiniMax-H3 video_vae must encode {}-frame clips to {} latent frames, keeping {} of \
                 them in the final clip, at {LATENT_COMPRESSION}x spatial and \
                 {TEMPORAL_COMPRESSION}x temporal compression; this checkpoint states {} frames to \
                 {} latents, keeping {}, at {}x spatial and {}x temporal",
                FRAMES.clip_frames,
                FRAMES.clip_latent_frames,
                FRAMES.tail_latent_frames,
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
            if self.text_encoder.text.hidden_size != transformer.text_dim {
                return Err(Error::config(format!(
                    "H3 text features are {} wide and the {} DiT conditions on {}",
                    self.text_encoder.text.hidden_size,
                    dit.name(),
                    transformer.text_dim
                )));
            }
            if video.latent_channels != transformer.video_channels {
                return Err(Error::config(format!(
                    "H3 video latents have {} channels and the {} DiT reads {}",
                    video.latent_channels,
                    dit.name(),
                    transformer.video_channels
                )));
            }
            if audio.latent_channels != transformer.audio_channels {
                return Err(Error::config(format!(
                    "H3 audio latents have {} channels and the {} DiT reads {}",
                    audio.latent_channels,
                    dit.name(),
                    transformer.audio_channels
                )));
            }
        }
        Ok(())
    }
}

/// The DiT partitions a base release holds: those its pipeline index names and whose
/// configuration the checkpoint holds.
fn released_dits(
    sidecar: &impl Fn(&str) -> Result<Option<Value>>,
    root: &str,
) -> Result<Vec<Stage>> {
    let index = sidecar("model_index.json")?.ok_or_else(|| {
        Error::checkpoint(format!(
            "{root} holds neither fastvideo_inference.json nor model_index.json"
        ))
    })?;
    let index = Fields::of("model_index.json", &index)?;
    if let Some(class) = index.optional_text("_class_name")?
        && class != PIPELINE_CLASS
    {
        return Err(Error::checkpoint(format!(
            "{root} is a {class} checkpoint, not a {PIPELINE_CLASS} pipeline root; the task \
             exports under FL2VA/ are not pipeline roots"
        )));
    }
    let mut dits = Vec::new();
    for dit in crate::checkpoint::DITS {
        if index.contains(dit.name()) && sidecar(&format!("{}/config.json", dit.name()))?.is_some()
        {
            dits.push(dit);
        }
    }
    if dits.is_empty() {
        return Err(Error::checkpoint(format!(
            "{root}/model_index.json names no DiT partition the checkpoint holds"
        )));
    }
    Ok(dits)
}

/// The DiT partition a FastH3 export distils: the base network of the task it states (`task`, or
/// `model_type` in the OmniRef exports).
fn student_dit(student: &Fields<'_>) -> Result<Stage> {
    let stated = match student.optional_text("task")? {
        Some(task) => task,
        None => student.optional_text("model_type")?.ok_or_else(|| {
            Error::checkpoint("fastvideo_inference.json states no task or model_type".to_owned())
        })?,
    };
    match stated {
        "t2av" => Ok(crate::checkpoint::dit(VideoTask::TextToVideoAudio)),
        "ref2va" => Ok(crate::checkpoint::dit(VideoTask::ReferenceToVideoAudio)),
        other => Err(Error::checkpoint(format!(
            "unsupported FastH3 checkpoint: its task must be t2av or ref2va, got {other:?}"
        ))),
    }
}

/// A FastH3 export's schedule and attention, from its `fastvideo_inference.json`.
///
/// The export states the same two shifts as its scheduler sidecars, and a checkpoint whose two
/// statements disagree is refused. Its schedule is a DMD ladder (`dmd_denoising_steps`) or a
/// parallel-decoding block grid (`pdd_step_indices` with `grid_max_t`), exactly one of them; its
/// attention backend is `DENSE` (or unstated) or `VIDEO_SPARSE_ATTN_H3` with the `vsa_*` knobs,
/// whose sparsity stays FP64 because `ceil((1 - sparsity) · tiles)` is a tile count.
fn student_denoising(
    student: &Fields<'_>,
    transformer: &TransformerConfig,
    shifts: Shifts,
) -> Result<(ScheduleSpec, AttentionKind)> {
    let stated = (
        student.number("video_scheduler_shift")? as f32,
        student.number("audio_scheduler_shift")? as f32,
    );
    if stated != (shifts.video, shifts.audio) {
        return Err(Error::config(format!(
            "fastvideo_inference.json shifts {stated:?} disagree with the scheduler sidecars \
             ({}, {})",
            shifts.video, shifts.audio
        )));
    }
    let schedule = match (
        student.optional_indices("dmd_denoising_steps")?,
        student.optional_indices("pdd_step_indices")?,
    ) {
        (Some(rungs), None) => ScheduleSpec::dmd(rungs, NUM_TRAIN_TIMESTEPS, shifts)?,
        (None, Some(nodes)) => ScheduleSpec::blocks(
            transformer.output_heads,
            nodes,
            student.number("grid_max_t")? as f32,
            shifts,
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
    let attention = match student.optional_text("attention_backend")? {
        None | Some("DENSE") => AttentionKind::Dense,
        Some("VIDEO_SPARSE_ATTN_H3") => AttentionKind::sparse(
            student.index("vsa_tile_size")?,
            student.number("vsa_sparsity")?,
            student.optional_number("vsa_ref_keep_rate")?,
        )?,
        Some(other) => {
            return Err(Error::checkpoint(format!(
                "fastvideo_inference.json states the attention backend {other:?}, which this \
                 build does not serve: it serves DENSE and VIDEO_SPARSE_ATTN_H3"
            )));
        }
    };
    Ok((schedule, attention))
}

/// The noise level shift a scheduler sidecar states.
pub(crate) fn shift(sidecar: &serde_json::Value, path: &str) -> Result<f32> {
    Ok(Fields::of(path, sidecar)?.positive("shift")? as f32)
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
/// 64-layer Qwen3-VL text encoder of which H3 keeps 50, a 50-block DiT conditioned on 5120-wide
/// text, a 17-frame video VAE of 24 latent channels, and a 32-channel audio VAE at 32 kHz.
#[cfg(test)]
pub(crate) mod tests_support {
    use std::collections::BTreeMap;

    use serde_json::Value;

    use super::*;

    /// Parse a fixture's JSON.
    pub(crate) fn json(body: &str) -> Value {
        serde_json::from_str(body).expect("a fixture is JSON")
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
            "latents_mean": vec![0.0_f64; 32],
            "latents_std": vec![1.0_f64; 32],
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

    /// The base release's sidecars, by checkpoint-relative path: a pipeline index over both DiT
    /// partitions.
    pub(crate) fn release() -> BTreeMap<String, Value> {
        let mut files = BTreeMap::new();
        for (path, value) in [
            (
                "model_index.json",
                json(
                    r#"{"_class_name": "MiniMaxH3ModularPipeline",
                        "transformer": [], "transformer_ref": []}"#,
                ),
            ),
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
        ] {
            files.insert(path.to_owned(), value);
        }
        files
    }

    /// A FastH3 DMD export of the base partition over `files`: its inference file, its scheduler's
    /// shift and no reference partition.
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
        read(&release(), None).expect("the released configuration")
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;

    #[test]
    fn a_base_release_reads_both_partitions_with_the_uniform_grid() {
        let config = released_config();
        assert_eq!(config.denoisers.len(), 2);
        assert_eq!(config.tasks(), VideoTask::ALL.to_vec());
        let denoiser = &config.denoisers[&Stage::TransformerRef];
        assert_eq!(denoiser.tasks, vec![VideoTask::ReferenceToVideoAudio]);
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
    }

    #[test]
    fn a_task_export_or_an_empty_root_is_not_a_pipeline_root() {
        let mut export = release();
        export.insert(
            "model_index.json".to_owned(),
            json(r#"{"_class_name": "MiniMaxH3Pipeline", "transformer": []}"#),
        );
        let error = read(&export, None).expect_err("a task export is refused");
        assert!(
            error.to_string().contains("not a MiniMaxH3ModularPipeline"),
            "{error}"
        );

        let mut empty = release();
        empty.remove("model_index.json");
        let error = read(&empty, None).expect_err("no index and no student");
        assert!(error.to_string().contains("model_index.json"), "{error}");
    }

    #[test]
    fn a_release_holds_the_partitions_its_index_names_and_its_root_holds() {
        let mut files = release();
        files.remove("transformer_ref/config.json");
        let config = read(&files, None).expect("the base partition alone");
        assert_eq!(
            config.denoisers.keys().collect::<Vec<_>>(),
            [&Stage::Transformer]
        );
        files.remove("transformer/config.json");
        assert!(read(&files, None).is_err(), "no partition is held");
    }

    #[test]
    fn a_dmd_export_reads_its_ladder_and_its_sparse_attention() {
        let config = read(&dmd_export(release(), "VIDEO_SPARSE_ATTN_H3"), None).expect("an export");
        assert_eq!(
            config.denoisers.keys().collect::<Vec<_>>(),
            [&Stage::Transformer]
        );
        let denoiser = &config.denoisers[&Stage::Transformer];
        assert_eq!(denoiser.schedule.steps(), 8);
        assert_eq!(
            denoiser.attention,
            AttentionKind::sparse(64, 0.8, None).expect("sparse")
        );
    }

    /// The export states its backend, and the statement decides: a student that kept its
    /// distillation's sparse knobs while naming dense attention attends densely, and an unknown
    /// backend is refused by name.
    #[test]
    fn an_export_attends_with_the_backend_it_states() {
        let config = read(&dmd_export(release(), "DENSE"), None).expect("a dense export");
        assert_eq!(
            config.denoisers[&Stage::Transformer].attention,
            AttentionKind::Dense
        );
        let error = read(&dmd_export(release(), "FLASH"), None).expect_err("unknown backend");
        assert!(error.to_string().contains("FLASH"), "{error}");
    }

    #[test]
    fn an_export_states_one_schedule_of_a_served_task() {
        let mut files = dmd_export(release(), "DENSE");
        files.get_mut("fastvideo_inference.json").expect("the file")["task"] = json(r#""t2v""#);
        assert!(read(&files, None).is_err(), "t2v is not served");

        let mut files = dmd_export(release(), "DENSE");
        let student = files.get_mut("fastvideo_inference.json").expect("the file");
        student["pdd_step_indices"] = json("[0, 8, 16, 32]");
        assert!(read(&files, None).is_err(), "two schedules");
    }

    #[test]
    fn a_fused_adapter_is_the_four_rung_student_it_was_extracted_from() {
        for (gated, attention) in [
            (true, AttentionKind::sparse(64, 0.9, None).expect("sparse")),
            (false, AttentionKind::Dense),
        ] {
            let config = read(&release(), Some(gated)).expect("the student");
            assert_eq!(
                config.denoisers.keys().collect::<Vec<_>>(),
                [&Stage::Transformer]
            );
            let denoiser = &config.denoisers[&Stage::Transformer];
            assert_eq!(denoiser.schedule.steps(), 4);
            assert_eq!(denoiser.attention, attention);
        }
    }

    #[test]
    fn a_configuration_refuses_components_that_disagree_with_each_other() {
        let mut widths = release();
        let encoder = widths
            .get_mut("text_encoder/config.json")
            .expect("the encoder");
        encoder["text_config"]["hidden_size"] = json("4096");
        encoder["vision_config"]["out_hidden_size"] = json("4096");
        let error = read(&widths, None).expect_err("the DiT conditioning width");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("conditions on")),
            "{error}"
        );

        let mut channels = release();
        let statistics = |value: f64| {
            serde_json::Value::Array((0..16).map(|_| serde_json::Value::from(value)).collect())
        };
        let vae = channels.get_mut("vae/config.json").expect("the VAE");
        vae["latent_channels"] = json("16");
        vae["latents_mean"] = statistics(0.0);
        vae["latents_std"] = statistics(1.0);
        let error = read(&channels, None).expect_err("the latent channels");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("channels")),
            "{error}"
        );
    }

    #[test]
    fn a_configuration_refuses_a_vae_timeline_the_packing_cannot_serve() {
        let mut timeline = release();
        timeline.get_mut("vae/config.json").expect("the VAE")["temporal_downsample_factors"] =
            json("[1, 2, 1, 1, 1, 1]");
        let error = read(&timeline, None).expect_err("a different compression");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("temporal")),
            "{error}"
        );

        let mut clock = release();
        clock.get_mut("audio_vae/config.json").expect("the VAE")["sampling_rate"] = json("44100");
        let error = read(&clock, None).expect_err("a different sample clock");
        assert!(
            matches!(error, Error::Config(ref problem)
                if problem.contains("44100") && problem.contains("32000 Hz")),
            "{error}"
        );
    }

    #[test]
    fn an_export_whose_shifts_disagree_with_its_schedulers_is_refused() {
        let mut files = dmd_export(release(), "DENSE");
        // The scheduler states the base release's video shift of 12, not the student's 10.
        files.insert(
            "scheduler/scheduler_config.json".to_owned(),
            json(r#"{"shift": 12.0}"#),
        );
        let error = read(&files, None).expect_err("shift mismatch");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("disagree")),
            "{error}"
        );
    }
}

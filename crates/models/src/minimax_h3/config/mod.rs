//! MiniMax-H3's typed configuration: the checkpoint's sidecars, read once at the loading
//! boundary and immutable afterwards.
//!
//! A composed checkpoint is normalized into one [`Config`] before any module is built: the base
//! release (the networks its pipeline index declares, each on the released full-step grid), or a
//! FastH3 student, an export or an adapter fused into the base release (one network on the
//! schedule and attention its contract states, [`Contract`]). The components each comes from are
//! named by [`Component`]. Every module's `load` consumes the typed struct of its own submodule
//! and reparses nothing.
//!
//! A packed ModelOpt export states its calibration in a manifest at its root; it is read here
//! into the quantization each stage's configuration declares ([`modelopt`]), which the stage's
//! loader is given before any of its weights is read. The deployment's precision policy is
//! applied to a dense checkpoint's declarations when the family opens the checkpoint, naming
//! the projections each stage converts at load, and refused for a packed one ([`precision`]).
//!
//! The request-level values the family's request rules are parameterized by (the frame rule,
//! the resolution rule, the durations and the audio clock) are request processing's
//! ([`processing::minimax_h3`]); this configuration checks that the checkpoint agrees with
//! them.

mod audio_vae;
mod contract;
mod denoiser;
mod modelopt;
pub(crate) mod precision;
mod text;
mod video_vae;

use std::collections::BTreeMap;

use nn::config::Fields;
use nn::quantization::ModeloptManifest;
use nn::{PipelineIndex, PixelNorm, Result};
use processing::qwen_vl::ProcessorConfig;
use serde_json::Value;

pub use audio_vae::AudioVaeConfig;
pub use contract::{BaseRevision, Contract, StudentSchedule};
pub use denoiser::{DenoiserConfig, ScheduleConfig, TransformerConfig, VsaConfig};
pub use processing::minimax_h3::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_LATENTS_PER_SECOND, AUDIO_SAMPLE_RATE, FPS, FRAMES,
    LATENT_COMPRESSION, PATCH_PIXELS, PATCH_SIDE,
};
pub use text::RETAINED_LAYERS;
pub use video_vae::VideoVaeConfig;

use crate::minimax_h3::stage::Stage;
use crate::minimax_h3::{ID, VideoTask};

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

/// A component of a MiniMax-H3 pipeline, by the name its index declares it under: a stage that
/// holds a shard set, or a directory that holds only configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Component {
    /// The Qwen3-VL text encoder.
    TextEncoder,
    /// The text encoder's Qwen-VL processor.
    Processor,
    /// The base network, of `t2va` and `fl2va`.
    Transformer,
    /// The reference network, of `ref2va`.
    TransformerRef,
    /// The video VAE, which holds the decoder's weights and the encoder's.
    Vae,
    /// The audio VAE, which holds the vocoder's weights and the encoder's.
    AudioVae,
    /// The video scheduler.
    Scheduler,
    /// The audio scheduler.
    AudioScheduler,
}

impl Component {
    /// Every component, in the order a pipeline index lists them.
    pub const ALL: [Self; 8] = [
        Self::TextEncoder,
        Self::Processor,
        Self::Vae,
        Self::AudioVae,
        Self::Transformer,
        Self::TransformerRef,
        Self::Scheduler,
        Self::AudioScheduler,
    ];

    /// The component's name in a pipeline index.
    pub const fn name(self) -> &'static str {
        match self {
            Self::TextEncoder => "text_encoder",
            Self::Processor => "processor",
            Self::Transformer => "transformer",
            Self::TransformerRef => "transformer_ref",
            Self::Vae => "vae",
            Self::AudioVae => "audio_vae",
            Self::Scheduler => "scheduler",
            Self::AudioScheduler => "audio_scheduler",
        }
    }

    /// The component a pipeline index names `name`, `None` for one this family reads nothing of.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|component| component.name() == name)
    }

    /// Whether the component holds a shard set.
    pub const fn weights(self) -> bool {
        matches!(
            self,
            Self::TextEncoder
                | Self::Transformer
                | Self::TransformerRef
                | Self::Vae
                | Self::AudioVae
        )
    }

    /// The deployment stage of the network the component holds, `None` for one that holds none.
    pub const fn network(self) -> Option<Stage> {
        match self {
            Self::Transformer => Some(Stage::Transformer),
            Self::TransformerRef => Some(Stage::TransformerRef),
            _ => None,
        }
    }

    /// The component that holds the network a deployment places as `stage`.
    ///
    /// # Errors
    ///
    /// When `stage` places no network.
    pub fn holding(stage: Stage) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|component| component.network() == Some(stage))
            .ok_or_else(|| ID.config(format!("{} places no MiniMax-H3 network", stage.name())))
    }

    /// The checkpoint-relative path of `file` in the component's directory.
    pub fn file(self, file: &str) -> String {
        format!("{}/{file}", self.name())
    }
}

/// The networks a composed checkpoint serves, as its composition states them.
#[derive(Clone, Debug, PartialEq)]
pub enum Networks {
    /// A base release's networks, each on the released full-step grid.
    Released(Vec<Stage>),
    /// One student's network, on the schedule and attention its contract states.
    Student(Contract),
}

/// The networks a base release serves: those its pipeline index declares.
///
/// # Errors
///
/// When the index declares no network.
pub fn released_networks(index: &PipelineIndex) -> Result<Vec<Stage>> {
    let networks: Vec<Stage> = Component::ALL
        .into_iter()
        .filter(|component| index.declares(component.name()))
        .filter_map(Component::network)
        .collect();
    if networks.is_empty() {
        return Err(ID.checkpoint("the pipeline index declares no MiniMax-H3 network".to_owned()));
    }
    Ok(networks)
}

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
    pub text_encoder: crate::qwen3::Config,
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
    /// Reads and normalizes the configuration of the composed `checkpoint`, whose networks
    /// `networks` states, once.
    ///
    /// Each sidecar is read from the directory of the component that holds it, as the
    /// checkpoint's composition placed it ([`nn::Checkpoint::sidecar`]).
    ///
    /// # Errors
    ///
    /// A sidecar that is missing or states a value this family does not serve.
    pub fn open(checkpoint: &nn::Checkpoint, networks: &Networks) -> Result<Self> {
        let root = checkpoint.root().display().to_string();

        Self::read(|relative| checkpoint.sidecar(relative), networks, &root)
    }

    /// The configuration of the sidecars `sidecar` returns by checkpoint-relative path, for the
    /// networks `networks` states.
    ///
    /// `root` names the checkpoint in refusals. Everything is checked here (the text encoder
    /// against the networks' conditioning width, the VAEs' compression against the frame rule,
    /// the student's contract against its schedulers), so no module sees a value it cannot use.
    pub(crate) fn read(
        sidecar: impl Fn(&str) -> Result<Option<Value>>,
        networks: &Networks,
        root: &str,
    ) -> Result<Self> {
        let require = |relative: &str| -> Result<Value> {
            sidecar(relative)?.ok_or_else(|| {
                ID.checkpoint(format!("the checkpoint at {root} holds no {relative}"))
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

        let text_file = Component::TextEncoder.file("config.json");
        let mut text_encoder = text::text_encoder(&require(&text_file)?)?;
        text_encoder.quantization = declared(Component::TextEncoder.name())?;
        let image_file = Component::Processor.file("preprocessor_config.json");
        let image_processor = require(&image_file)?;
        let pixels = Fields::of(&image_file, &image_processor)?.pixel_norm()?;
        let processor = sidecar(&Component::Processor.file("video_preprocessor_config.json"))?
            .map(|video| ProcessorConfig::from_values(&image_processor, &video))
            .transpose()
            .map_err(|error| ID.config(format!("the checkpoint's processor: {error}")))?;

        let mut video_vae = VideoVaeConfig::parse(&require(&Component::Vae.file("config.json"))?)?;
        video_vae.quantization = declared(Component::Vae.name())?;
        let audio_vae = AudioVaeConfig::parse(&require(&Component::AudioVae.file("config.json"))?)?;
        let shifts = (
            shift(
                &require(&Component::Scheduler.file("scheduler_config.json"))?,
                Component::Scheduler,
            )?,
            shift(
                &require(&Component::AudioScheduler.file("scheduler_config.json"))?,
                Component::AudioScheduler,
            )?,
        );

        let (stages, student) = match networks {
            Networks::Released(stages) => (stages.clone(), None),
            Networks::Student(contract) => (vec![contract.network], Some(contract)),
        };

        let mut denoisers = BTreeMap::new();
        for stage in stages {
            let component = Component::holding(stage)?;
            let relative = component.file("config.json");
            let transformer =
                TransformerConfig::parse(&Fields::of(&relative, &require(&relative)?)?)?;

            let (schedule, vsa) = match student {
                Some(contract) => student_denoising(contract, &transformer, shifts)?,
                None => (
                    ScheduleConfig::uniform(UNIFORM_GRID_POINTS, shifts.0, shifts.1)?,
                    None,
                ),
            };

            let tasks = VideoTask::ALL
                .into_iter()
                .filter(|task| network(*task) == stage)
                .collect();
            let mut denoiser = DenoiserConfig::new(transformer, schedule, vsa, tasks)?;
            denoiser.quantization = declared(component.name())?;
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
    pub fn vision(&self) -> Result<&crate::qwen3::VisionConfig> {
        self.text_encoder.vision.as_ref().ok_or_else(|| {
            ID.config("H3's text encoder is a Qwen3-VL model with a vision tower".to_owned())
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
            return Err(ID.config(format!(
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
            return Err(ID.config(format!(
                "the MiniMax-H3 audio_vae samples at {} Hz; the request rule's audio runs at \
                 {AUDIO_SAMPLE_RATE} Hz",
                audio.sampling_rate
            )));
        }
        if audio.hop() != AUDIO_HOP || audio.decoder_hop() != AUDIO_HOP {
            return Err(ID.config(format!(
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
                return Err(ID.config(format!(
                    "H3 text features are {} wide and the {} network conditions on {}",
                    self.text_encoder.text.hidden_size,
                    stage.name(),
                    transformer.text_dim
                )));
            }
            if video.latent_channels != transformer.video_channels {
                return Err(ID.config(format!(
                    "H3 video latents have {} channels and the {} network reads {}",
                    video.latent_channels,
                    stage.name(),
                    transformer.video_channels
                )));
            }
            if audio.latent_channels != transformer.audio_channels {
                return Err(ID.config(format!(
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

/// A FastH3 student's schedule and sparse attention, as its contract states them.
///
/// The shifts are the scheduler sidecars'; a contract that states them too must state the same
/// two, and one whose statements disagree is refused. A DMD ladder's rungs are timesteps of the
/// training schedule; a parallel-decoding block grid spans the network's output heads.
fn student_denoising(
    contract: &Contract,
    transformer: &TransformerConfig,
    shifts: (f32, f32),
) -> Result<(ScheduleConfig, Option<VsaConfig>)> {
    for ((key, stated), shift) in ["video_scheduler_shift", "audio_scheduler_shift"]
        .into_iter()
        .zip(contract.shifts)
        .zip([shifts.0, shifts.1])
    {
        if let Some(stated) = stated
            && stated as f32 != shift
        {
            return Err(ID.config(format!(
                "{}.{key} is {stated}, which disagrees with the scheduler sidecar's shift of \
                 {shift}",
                Contract::FILE
            )));
        }
    }

    let schedule = match &contract.schedule {
        StudentSchedule::Ladder(rungs) => {
            ScheduleConfig::ladder(rungs, NUM_TRAIN_TIMESTEPS, shifts.0, shifts.1)?
        }
        StudentSchedule::Blocks { nodes, grid_max_t } => ScheduleConfig::blocks(
            transformer.output_heads,
            nodes.clone(),
            *grid_max_t as f32,
            shifts.0,
            shifts.1,
        )?,
    };
    Ok((schedule, contract.vsa))
}

/// The noise level shift the scheduler sidecar of `component` states.
fn shift(sidecar: &Value, component: Component) -> Result<f32> {
    let owner = component.file("scheduler_config.json");

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

/// The released checkpoint's sidecars, as the tests of this module use them.
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
                        "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
                        "transformer_ref": ["diffusers", "MiniMaxH3Transformer3DModel"]}"#;

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

    /// A FastH3 DMD export of the base network over `files`: its contract, its scheduler's
    /// shift and no reference network.
    pub(crate) fn dmd_export(
        mut files: BTreeMap<String, Value>,
        backend: &str,
    ) -> BTreeMap<String, Value> {
        files.remove("transformer_ref/config.json");
        files.insert(
            "scheduler/scheduler_config.json".to_owned(),
            json(r#"{"shift": 10.0}"#),
        );
        files.insert(Contract::FILE.to_owned(), dmd_contract(backend));
        files
    }

    /// The contract of an eight-rung DMD student of the base network, attending with `backend`.
    pub(crate) fn dmd_contract(backend: &str) -> Value {
        serde_json::json!({
            "schema_version": Contract::SCHEMA,
            "task": "t2av",
            "video_scheduler_shift": 10.0,
            "audio_scheduler_shift": 3.0,
            "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
            "attention_backend": backend,
            "vsa_tile_size": 64,
            "vsa_sparsity": 0.8,
        })
    }

    /// The configuration the family composes of `files`, a checkpoint's sidecars: the student
    /// `adapter` contracts when an adapter is fused, else the student the checkpoint's own
    /// contract states, else the networks its pipeline index declares.
    pub(crate) fn read(
        files: &BTreeMap<String, Value>,
        adapter: Option<&Contract>,
    ) -> Result<Config> {
        let networks = match (adapter, files.get(Contract::FILE)) {
            (Some(contract), _) => Networks::Student(contract.clone()),
            (None, Some(contract)) => Networks::Student(Contract::parse(contract)?),
            (None, None) => {
                let index = files.get("model_index.json").ok_or_else(|| {
                    ID.checkpoint("the fixture states no model_index.json".to_owned())
                })?;
                let index = PipelineIndex::parse(std::path::Path::new("model_index.json"), index)?;
                Networks::Released(released_networks(&index)?)
            }
        };

        Config::read(
            |relative| Ok(files.get(relative).cloned()),
            &networks,
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
    fn a_release_serves_the_networks_its_index_declares() {
        let mut files = release();
        files.insert(
            "model_index.json".to_owned(),
            json(
                r#"{"_class_name": "MiniMaxH3ModularPipeline",
                    "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"]}"#,
            ),
        );
        let config = read(&files, None).unwrap();
        assert_eq!(
            config.denoisers.keys().collect::<Vec<_>>(),
            [&Stage::Transformer]
        );

        files.insert(
            "model_index.json".to_owned(),
            json(r#"{"_class_name": "MiniMaxH3ModularPipeline"}"#),
        );
        let error = read(&files, None).unwrap_err();
        assert!(
            error.to_string().contains("no MiniMax-H3 network"),
            "{error}"
        );
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
        files.get_mut(Contract::FILE).unwrap()["task"] = json(r#""t2v""#);
        assert!(read(&files, None).is_err(), "t2v is not served");

        let mut files = dmd_export(release(), "DENSE");
        let student = files.get_mut(Contract::FILE).unwrap();
        student["pdd_step_indices"] = json("[0, 8, 16, 32]");
        assert!(read(&files, None).is_err(), "two schedules");
    }

    /// A fused adapter serves the student its contract states, on the base release's
    /// schedulers: its network alone, its rungs and its attention.
    #[test]
    fn a_fused_adapter_serves_the_student_its_contract_states() {
        for (backend, vsa) in [
            (
                "VIDEO_SPARSE_ATTN_H3",
                Some(VsaConfig::new(64, 0.9, None).unwrap()),
            ),
            ("DENSE", None),
        ] {
            let contract = Contract::parse(&serde_json::json!({
                "schema_version": Contract::SCHEMA,
                "task": "t2av",
                "dmd_denoising_steps": [999, 749, 500, 250],
                "attention_backend": backend,
                "vsa_tile_size": 64,
                "vsa_sparsity": 0.9,
            }))
            .unwrap();
            let config = read(&release(), Some(&contract)).unwrap();
            assert_eq!(
                config.denoisers.keys().collect::<Vec<_>>(),
                [&Stage::Transformer]
            );

            let denoiser = &config.denoisers[&Stage::Transformer];
            assert_eq!(denoiser.schedule.num_steps(), 4);
            assert_eq!(
                (denoiser.schedule.video_shift, denoiser.schedule.audio_shift),
                (12.0, 3.0)
            );
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
            matches!(error, nn::Error::Model { refusal: nn::Refusal::Config, ref problem, .. } if problem.contains("conditions on")),
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
            matches!(error, nn::Error::Model { refusal: nn::Refusal::Config, ref problem, .. } if problem.contains("channels")),
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
            matches!(error, nn::Error::Model { refusal: nn::Refusal::Config, ref problem, .. } if problem.contains("temporal")),
            "{error}"
        );

        let mut clock = release();
        clock.get_mut("audio_vae/config.json").unwrap()["sampling_rate"] = json("44100");
        let error = read(&clock, None).unwrap_err();
        assert!(
            matches!(error, nn::Error::Model { refusal: nn::Refusal::Config, ref problem, .. }
                if problem.contains("44100") && problem.contains("32000 Hz")),
            "{error}"
        );
    }

    /// The four-step preview export states no shift: its schedule takes the scheduler
    /// sidecars' shifts.
    #[test]
    fn an_export_that_states_no_shift_takes_its_schedulers() {
        let mut files = dmd_export(release(), "DENSE");
        let student = files.get_mut(Contract::FILE).unwrap();
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
            matches!(error, nn::Error::Model { refusal: nn::Refusal::Config, ref problem, .. } if problem.contains("disagree")),
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
        json(include_str!("modelopt_manifest.json"))
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

    /// A calibration this family has no stored projection for is refused by the path the
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
            matches!(error, nn::Error::Model { refusal: nn::Refusal::Config, ref problem, .. } if problem.contains(listed)),
            "{error}"
        );

        let mut inconsistent = manifest();
        inconsistent["components"]["video_vae"]["enabled"] = serde_json::json!(false);
        files.insert(ModeloptManifest::FILE.to_owned(), inconsistent);
        let error = read(&files, None).unwrap_err();
        assert!(
            matches!(error, nn::Error::Config { ref field, .. } if field == "modules"),
            "{error}"
        );
    }
}

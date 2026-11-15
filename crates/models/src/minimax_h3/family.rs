//! MiniMax-H3 and FastH3 as execution loads them and request processing serves them
//! ([`nn::family`]).
//!
//! [`FAMILY`] claims the `MiniMaxH3ModularPipeline` class. A checkpoint opens to its
//! [`Config`] (an export and the base release it continues read as one), a load builds the
//! stages the deployment places on a rank with the deployment's sequence and tensor splits, and
//! the loaded [`MinimaxH3`] states the pipeline's own rules to execution: the `17n + 5` frame
//! rule, its resolutions, the requests its denoiser's graphs are captured for, and the
//! description its request processing admits requests against.
//!
//! ## Stages
//!
//! A deployment places stages under the names of its deployment file ([`Stage`]):
//! `text_encoder` (the Qwen3-VL model, an [`nn::Encoder`]), `transformer` (the network of
//! `t2va` and `fl2va`) or `transformer_ref` (the network of `ref2va`), of which a rank holds
//! one, `video_decoder`, `audio_decoder`, `latent_encoder` (the condition encoders, which
//! live with the denoiser that reads them) and `video_postprocessor` (the clip splice, which
//! holds no weights and is placed wherever `video_decoder` is). A stage this rank does not place
//! is absent from the loaded pipeline.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use nn::family::{
    ConditionBounds, Description, Family, Model, Stage, Stages, VideoCheckpoint, VideoFamily,
    VideoModel, VideoPlacement, VideoServing,
};
use nn::pipeline::{Block, Extent, Phase, Pipeline, Product, ProductSpec};
use nn::{
    Checkpoint, ConditionRole, ConditionShape, DType, Decoder, Encoder, MediaFormat, Splicer,
    VideoRequest, WeightLoader,
};
use processing::minimax_h3::{
    CANVAS_RULE, ConditionPacking, DenoiserInfo, MIN_SECONDS, PreparedCondition, REFERENCE_LIMITS,
    ReferenceLimits, ServedCanvas, named_canvases,
};
use proto::config::{Partition, Quantization};

use crate::minimax_h3::audio_vae::{AudioDecoder, AudioEncoder};
use crate::minimax_h3::config::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_SAMPLE_RATE, FPS, FRAMES, NETWORKS, RETAINED_LAYERS,
    VIDEO_VAE, network, precision,
};
use crate::minimax_h3::denoiser::{Denoiser, SAMPLE_DTYPE};
use crate::minimax_h3::splice::VideoPostprocessor;
use crate::minimax_h3::video_vae::{self, VideoDecoder, VideoEncoder};
use crate::minimax_h3::{Config, Error, Result, VideoTask, packing};
use crate::qwen3::Qwen3VlModel;

/// The MiniMax-H3 family: a diffusers modular pipeline, which request processing serves
/// through its video processor ([`processing::minimax_h3::H3Family`]).
pub const FAMILY: Family = Family {
    request: processing::minimax_h3::FAMILY,
    model: Model::Video(&MinimaxH3Family),
    step_stack: crate::minimax_h3::transformer::TRANSFORMER,
};

/// The release a FastH3 adapter is fused into.
const FASTH3_BASE_MODEL: &str = "MiniMaxAI/MiniMax-H3";

/// The family's value, which [`FAMILY`] refers to.
struct MinimaxH3Family;

/// Whether `metadata` is the header of a FastH3 adapter over the base release: the FastVideo
/// adapter format, a `FastVideo/…FastH3…` student, and the base release as its base model.
fn fasth3_adapter(metadata: &std::collections::BTreeMap<String, String>) -> bool {
    let student = metadata
        .get("finetuned_model")
        .map(|name| name.to_lowercase())
        .unwrap_or_default();
    let base = metadata.get("base_model").map(String::as_str);

    metadata.get("format").map(String::as_str) == Some(nn::adapter::FASTVIDEO_FORMAT)
        && student.starts_with("fastvideo/")
        && student.contains("fasth3")
        && base.is_some_and(|base| base.eq_ignore_ascii_case(FASTH3_BASE_MODEL))
}

impl VideoFamily for MinimaxH3Family {
    fn open(
        &self,
        checkpoint: &Checkpoint,
        quantization: &Quantization,
    ) -> nn::Result<Box<dyn VideoCheckpoint>> {
        // A checkpoint is one directory, or an export and the base release it continues.
        if checkpoint.roots().len() > 2 {
            return Err(nn::Error::Unsupported(format!(
                "a MiniMax-H3 checkpoint is one directory and the base release it continues, \
                 not {} directories",
                checkpoint.roots().len()
            )));
        }

        let mut config = Config::open(checkpoint)?;
        precision::apply(quantization, &mut config)?;
        Ok(Box::new(config))
    }

    /// A FastH3 adapter edits the base release's `transformer`, the network its student was
    /// distilled from; fused there, the checkpoint reads as that student.
    fn adapter_stage(
        &self,
        checkpoint: &Checkpoint,
        adapter: &nn::Adapter,
    ) -> nn::Result<Option<&'static str>> {
        if !fasth3_adapter(adapter.metadata()) {
            return Err(nn::Error::Unsupported(format!(
                "{} is not a FastH3 adapter (its metadata names the student {:?} over {:?}); \
                 MiniMax-H3 fuses the FastH3 releases of FastVideo over {FASTH3_BASE_MODEL}",
                adapter.path().display(),
                adapter.metadata().get("finetuned_model"),
                adapter.metadata().get("base_model"),
            )));
        }

        // An adapter edits the base release: a FastH3 export is already a student.
        if checkpoint.sidecar("fastvideo_inference.json")?.is_some() {
            return Err(nn::Error::Unsupported(format!(
                "{} is a FastH3 export, and a FastH3 adapter is fused into the base release \
                 {FASTH3_BASE_MODEL} it was extracted from",
                checkpoint.root().display(),
            )));
        }
        Ok(Some(Stage::Transformer.name()))
    }
}

impl VideoCheckpoint for Config {
    fn pipeline(&self) -> Pipeline {
        pipeline()
    }

    fn load(
        &self,
        checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &VideoPlacement,
    ) -> nn::Result<Arc<dyn VideoModel>> {
        Ok(Arc::new(MinimaxH3::load(
            self, checkpoint, loader, placement,
        )?))
    }
}

/// MiniMax-H3's pipeline: a conditioned request reads its media, its prompt is encoded with
/// the vision features of its conditions, its latents are prepared (encoding its condition
/// latents) and denoised one step per call, its video is decoded in rounds of media units
/// spliced onto each other and its soundtrack once, each decoded round and the track are
/// encoded, and the encoded rounds are muxed into the artifact. After denoising, completed
/// decodes are encoded before further decodes are asked for.
///
/// The release holds two networks that denoise (the base and the reference-conditioned
/// one), of which a deployment places one; the condition encoders are what a latent
/// preparation computes with, held by the denoising ranks; the audio decoder decodes a
/// request's track whole, on its block's first rank.
pub fn pipeline() -> Pipeline {
    use Extent::{Request, Unit};
    use Phase::{
        AudioDecode, AudioEncode, Denoise, LatentPrepare, MediaRead, Mux, TextEncode, VideoDecode,
        VideoEncode,
    };

    let product = |product, writer, reader, extent| ProductSpec {
        product,
        writer,
        reader,
        extent,
    };
    let block = |stage: Stage, phases: &[Phase], partition| Block {
        name: stage.name().to_owned(),
        phases: phases.to_vec(),
        partition,
    };
    Pipeline {
        phases: vec![
            MediaRead,
            TextEncode,
            LatentPrepare,
            Denoise,
            VideoEncode,
            AudioEncode,
            VideoDecode,
            AudioDecode,
            Mux,
        ],
        products: vec![
            product(Product::ConditionPixels, MediaRead, LatentPrepare, Request),
            product(Product::ConditionSamples, MediaRead, LatentPrepare, Request),
            product(Product::VisionPixels, MediaRead, TextEncode, Request),
            product(Product::Conditioning, TextEncode, LatentPrepare, Request),
            product(Product::VideoLatents, Denoise, VideoDecode, Request),
            product(Product::AudioLatents, Denoise, AudioDecode, Request),
            product(Product::VideoUnits, VideoDecode, VideoEncode, Request),
            product(Product::VideoTail, VideoDecode, VideoDecode, Unit),
            product(Product::EncodedUnits, VideoEncode, Mux, Unit),
            product(Product::AudioSamples, AudioDecode, AudioEncode, Request),
        ],
        blocks: vec![
            block(Stage::TextEncoder, &[TextEncode], Partition::Group),
            block(
                Stage::Transformer,
                &[LatentPrepare, Denoise],
                Partition::Group,
            ),
            block(
                Stage::TransformerRef,
                &[LatentPrepare, Denoise],
                Partition::Group,
            ),
            block(
                Stage::VideoDecoder,
                &[VideoDecode],
                Partition::Units { carries: true },
            ),
            block(Stage::AudioDecoder, &[AudioDecode], Partition::First),
            block(Stage::LatentEncoder, &[], Partition::With(LatentPrepare)),
        ],
    }
}

/// A loaded MiniMax-H3 pipeline: the stages this rank places, each built once, the
/// configuration they were read with, the deployment's element type and the tasks its network
/// serves.
pub struct MinimaxH3 {
    config: Config,
    dtype: DType,
    tasks: Vec<VideoTask>,
    text_encoder: Option<Arc<Qwen3VlModel>>,
    denoiser: Option<(Stage, Arc<Denoiser>)>,
    splicer: Option<Arc<VideoPostprocessor>>,
    video_decoder: Option<Arc<VideoDecoder>>,
    audio_decoder: Option<Arc<AudioDecoder>>,
}

impl MinimaxH3 {
    /// Loads the stages `placement` places of `checkpoint`, whose configuration `config` is:
    /// the stages its tasks need that the rank's group places.
    ///
    /// A stage this build cannot construct is refused by name and not omitted: a deployment
    /// that places one fails at load.
    fn load(
        config: &Config,
        checkpoint: &Checkpoint,
        loader: &WeightLoader<'_>,
        placement: &VideoPlacement,
    ) -> Result<Self> {
        let serving = &placement.serving;
        let tasks = served_tasks(
            &config.tasks(),
            placement.stages.as_ref(),
            checkpoint.root(),
        )?;
        let stages = stages(config, &tasks, placement.stages.as_ref())?;
        let split = placement.denoiser;

        // The tensor split and the Ulysses degree divide the attention heads between their
        // ranks; an all-gather degree splits rows alone.
        for (stage, denoiser) in &config.denoisers {
            let heads = denoiser.transformer.num_attention_heads;
            let shares = split.ulysses.split.size * split.tensor.size;
            if shares == 0 || !heads.is_multiple_of(shares) {
                return Err(Error::config(format!(
                    "a Ulysses degree of {} in a tensor split of {} does not divide the {heads} \
                     attention heads of {}",
                    split.ulysses.split.size,
                    split.tensor.size,
                    stage.name()
                )));
            }
        }

        // The text capacity is the deployment's longest prompt; the condition capacity is the
        // deployment's when it states one, and otherwise the widest its tasks ask for.
        let text_rows = *serving.text_capacities.last().ok_or_else(|| {
            Error::config(
                "this deployment states no prompt capacity, so no layout could hold a prompt"
                    .to_owned(),
            )
        })?;
        let condition_rows = match serving.condition_rows {
            Some(rows) => rows,
            None => tasks
                .iter()
                .map(|task| condition_capacity(config, *task))
                .collect::<Result<Vec<u32>>>()?
                .into_iter()
                .max()
                .unwrap_or(0),
        };

        let mut model = Self {
            config: config.clone(),
            dtype: placement.dtype,
            tasks: tasks.clone(),
            text_encoder: None,
            denoiser: None,
            splicer: None,
            video_decoder: None,
            audio_decoder: None,
        };

        if stages.contains(Stage::TextEncoder) {
            let processor = config.processor.ok_or_else(|| {
                Error::checkpoint(
                    "the checkpoint states no processor/video_preprocessor_config.json, so the \
                     text encoder's vision tower cannot size an image"
                        .to_owned(),
                )
            })?;
            let mut stage = loader.reborrow(checkpoint.stage(Stage::TextEncoder.name())?)?;
            stage.declare(&config.text_encoder.quantization)?;

            model.text_encoder = Some(Arc::new(Qwen3VlModel::load(
                &config.text_encoder,
                stage.root(),
                (processor, config.pixels),
                placement.tensor,
                RETAINED_LAYERS,
                false,
            )?));
        }

        for stage in NETWORKS {
            if !stages.contains(stage) {
                continue;
            }
            let denoiser = config.denoisers.get(&stage).ok_or_else(|| {
                Error::checkpoint(format!("the checkpoint holds no {} network", stage.name()))
            })?;

            // The two condition encoders are the denoiser's conditioning half.
            let encoders = if stages.contains(Stage::LatentEncoder) {
                let mut video = loader.reborrow(checkpoint.stage(VIDEO_VAE)?)?;
                video.declare(&config.video_vae.quantization)?;
                let audio = loader.reborrow(checkpoint.stage("audio_vae")?)?;

                Some((
                    VideoEncoder::load(&config.video_vae, video.root())?,
                    AudioEncoder::load(&config.audio_vae, audio.root())?,
                ))
            } else {
                None
            };

            let mut network = loader.reborrow(checkpoint.stage(stage.name())?)?;
            network.declare(&denoiser.quantization)?;

            model.denoiser = Some((
                stage,
                Arc::new(Denoiser::load(
                    denoiser,
                    (text_rows, condition_rows),
                    split,
                    encoders,
                    network.root(),
                )?),
            ));
        }

        if stages.contains(Stage::VideoPostprocessor) {
            model.splicer = Some(Arc::new(VideoPostprocessor::new(
                video_vae::ACTIVATION,
                video_vae::temporal_clips(&config.video_vae),
            )?));
        }
        if stages.contains(Stage::AudioDecoder) {
            let stage = loader.reborrow(checkpoint.stage("audio_vae")?)?;

            model.audio_decoder = Some(Arc::new(AudioDecoder::load(
                &config.audio_vae,
                stage.root(),
            )?));
        }
        if stages.contains(Stage::VideoDecoder) {
            // The deployment's sizes are checked here, where a size the checkpoint does not
            // generate would otherwise be found at the first decode.
            served_resolutions(&serving.sizes)?;
            let mut stage = loader.reborrow(checkpoint.stage(VIDEO_VAE)?)?;
            stage.declare(&config.video_vae.quantization)?;

            model.video_decoder = Some(Arc::new(VideoDecoder::load(
                &config.video_vae,
                stage.root(),
            )?));
        }
        Ok(model)
    }

    /// The placed denoiser network.
    fn network(&self) -> Option<&Arc<Denoiser>> {
        self.denoiser.as_ref().map(|(_, network)| network)
    }

    /// How the placed denoiser lays a request's conditions into its rows, as request
    /// processing sizes them: whole tiles under sparse attention, row for row under dense
    /// attention, and the rows it holds when this rank places the condition encoders.
    fn condition_packing(&self, denoiser: &Denoiser) -> ConditionPacking {
        let capacity = if denoiser.places_encoders() {
            denoiser.condition_capacity()
        } else {
            0
        };

        match denoiser.config().vsa {
            Some(vsa) => ConditionPacking {
                capacity,
                tile_rows: vsa.tile_rows,
                video_tile: vsa.tile_shape(),
            },
            None => ConditionPacking {
                capacity,
                tile_rows: 1,
                video_tile: [1, 1, 1],
            },
        }
    }
}

impl VideoModel for MinimaxH3 {
    fn text_encoder(&self) -> Option<Arc<dyn Encoder + Send + Sync>> {
        self.text_encoder
            .as_ref()
            .map(|encoder| Arc::clone(encoder) as Arc<dyn Encoder + Send + Sync>)
    }

    fn text_width(&self) -> u32 {
        self.config.text_encoder.text.hidden_size
    }

    fn denoiser(&self) -> Option<Arc<dyn nn::Denoiser + Send + Sync>> {
        self.network()
            .map(|denoiser| Arc::clone(denoiser) as Arc<dyn nn::Denoiser + Send + Sync>)
    }

    fn denoiser_stage(&self) -> &'static str {
        match self.denoiser {
            Some((Stage::TransformerRef, _)) => Stage::TransformerRef.name(),
            _ => Stage::Transformer.name(),
        }
    }

    fn sample_type(&self) -> DType {
        SAMPLE_DTYPE
    }

    fn video_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>> {
        self.video_decoder
            .as_ref()
            .map(|decoder| Arc::clone(decoder) as Arc<dyn Decoder + Send + Sync>)
    }

    fn splicer(&self) -> Option<Arc<dyn Splicer + Send + Sync>> {
        self.splicer
            .as_ref()
            .map(|splicer| Arc::clone(splicer) as Arc<dyn Splicer + Send + Sync>)
    }

    fn audio_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>> {
        self.audio_decoder
            .as_ref()
            .map(|decoder| Arc::clone(decoder) as Arc<dyn Decoder + Send + Sync>)
    }

    fn audio_rate(&self) -> u32 {
        self.audio_decoder
            .as_ref()
            .map_or(AUDIO_SAMPLE_RATE, |decoder| decoder.sampling_rate())
    }

    fn frame_rate(&self) -> u32 {
        FPS
    }

    fn decode_window(&self) -> u32 {
        // One decode call decodes the decoder's window of latent frames, each covering the
        // VAE's temporal compression of output frames.
        video_vae::temporal_clips(&self.config.video_vae).window()
            * self.config.video_vae.temporal_compression()
    }

    /// The video encoder reads planar pixels normalized by [`video_vae::PIXEL_NORM`], the
    /// vision tower patch rows by the processor's statistics, and the audio encoder channel
    /// rows padded to whole latent hops.
    fn format(&self) -> nn::Result<MediaFormat> {
        Ok(MediaFormat {
            pixels: video_vae::PIXEL_NORM,
            patches: crate::qwen3::vision::patch_format(
                self.config.vision()?,
                self.config.pixels,
                self.dtype,
            ),
            hop: u64::from(AUDIO_HOP),
        })
    }

    fn timeline(&self, seconds: f64) -> nn::Result<u32> {
        Ok(timeline(seconds)?)
    }

    fn video_units(&self, frames: u32) -> nn::Result<Vec<u32>> {
        // The decoder's temporal clips are the one statement of the clip schedule.
        let clips = video_vae::temporal_clips(&self.config.video_vae);

        Ok(video_vae::splices(&clips, frames)?
            .into_iter()
            .map(|splice| splice.keep)
            .collect())
    }

    fn condition_bounds(&self) -> ConditionBounds {
        // A deployment that places no condition encoder takes no condition at all.
        let rows = self
            .network()
            .filter(|denoiser| denoiser.places_encoders())
            .map_or(0, |denoiser| denoiser.condition_capacity());

        // A condition row is one latent patch, a square of the resolution rule's multiple, of
        // one latent frame, which covers up to a clip's frames over its latent frames, rounded
        // up; a track's latent step is two rows.
        let frames = FRAMES.clip_frames.div_ceil(FRAMES.clip_latent_frames);
        let cell = CANVAS_RULE.multiple;
        let merge = self
            .config
            .vision()
            .map_or(1, |vision| vision.spatial_merge_size);

        ConditionBounds {
            rows,
            pixels: rows.saturating_mul(cell * cell * frames),
            samples: (rows / AUDIO_CHANNELS).saturating_mul(AUDIO_HOP),
            patches_per_token: merge * merge,
        }
    }

    fn candidates(
        &self,
        serving: &VideoServing,
    ) -> nn::Result<Vec<(VideoRequest, Vec<Vec<ConditionShape>>)>> {
        let denoiser = self.network().ok_or_else(|| {
            Error::config("this rank places no denoiser, so no denoising step could run".to_owned())
        })?;

        Ok(candidates(
            &self.tasks,
            serving,
            &denoiser.condition_ladder(),
            &self.condition_packing(denoiser),
            self.config.processor.as_ref(),
        )?)
    }

    fn warm_prompts(&self, serving: &VideoServing) -> nn::Result<Vec<(Vec<[u32; 3]>, Vec<u32>)>> {
        let placeholder = self
            .config
            .text_encoder
            .vision
            .as_ref()
            .map(|vision| vision.image_token_id);
        Ok(warm_prompts(
            &self.tasks,
            serving,
            (self.config.processor.as_ref(), placeholder),
        )?)
    }

    /// What the placed denoiser serves under `serving`: the tasks of the placed network, its
    /// fixed schedule and shifts, the deployment's resolutions, the reference limits and the
    /// condition packing.
    fn description(&self, serving: &VideoServing) -> nn::Result<Description> {
        let denoiser = self.network().ok_or_else(|| {
            Error::config("this rank places no denoiser, so it serves no task".to_owned())
        })?;
        let tasks: Vec<VideoTask> = VideoTask::ALL
            .into_iter()
            .filter(|task| self.tasks.contains(task))
            .collect();
        let schedule = &denoiser.config().schedule;
        let steps = schedule.num_steps();

        let info = DenoiserInfo {
            tasks: tasks.iter().map(|task| task.as_str().to_owned()).collect(),
            schedule_points: steps + 1,
            video_shift: f64::from(schedule.video_shift),
            audio_shift: f64::from(schedule.audio_shift),
            canvases: served_resolutions(&serving.sizes)?,
            references: if tasks.contains(&VideoTask::ReferenceToVideoAudio) {
                REFERENCE_LIMITS
            } else {
                ReferenceLimits::default()
            },
            condition_packing: self.condition_packing(denoiser),
        };
        info.validate()
            .map_err(|reason| Error::config(format!("the denoiser's description: {reason}")))?;

        Ok(Description {
            takes_conditions: tasks.iter().any(|task| task.conditioned()),
            denoise_steps: steps,
            details: info
                .encode()
                .map_err(|error| Error::config(format!("encoding the description: {error}")))?,
        })
    }
}

/// The stages a rank builds to serve `tasks` over a checkpoint of `config`: the stages the tasks
/// need that the rank's group places (`placed`; `None` places the whole pipeline).
///
/// Every task needs the text encoder, the two decoders and the splice, because every task
/// returns video and audio; its network (`t2va` and `fl2va` run the base `transformer`,
/// `ref2va` the `transformer_ref`); and the condition encoders when it carries conditions. A
/// rank holds one network, so a group that places both is refused.
fn stages(
    config: &Config,
    tasks: &[VideoTask],
    placed: Option<&BTreeSet<String>>,
) -> Result<Stages> {
    if tasks.is_empty() {
        return Err(Error::config(
            "a deployment serves at least one task".to_owned(),
        ));
    }
    let stated = config.tasks();

    let mut needed = Stages::of([
        Stage::TextEncoder,
        Stage::VideoDecoder,
        Stage::AudioDecoder,
        Stage::VideoPostprocessor,
    ]);
    for task in tasks {
        if !stated.contains(task) {
            return Err(Error::config(format!(
                "this checkpoint states {stated:?} and not {task:?}"
            )));
        }

        needed = needed.with(network(*task));
        if task.conditioned() {
            needed = needed.with(Stage::LatentEncoder);
        }
    }

    // The clip splice holds no weights and places the video decoder's clips into the
    // timeline, so it goes wherever the video decoder goes.
    let places = |names: &BTreeSet<String>, stage: Stage| {
        names.contains(stage.name())
            || (stage == Stage::VideoPostprocessor && names.contains(Stage::VideoDecoder.name()))
    };
    let stages: Stages = needed
        .iter()
        .filter(|stage| placed.is_none_or(|names| places(names, *stage)))
        .collect();
    if stages.contains(Stage::Transformer) && stages.contains(Stage::TransformerRef) {
        return Err(Error::config(
            "a rank cannot hold both the base and the reference network: they are one stage"
                .to_owned(),
        ));
    }
    Ok(stages)
}

/// The tasks a rank's stages serve: those the checkpoint states, on the network the rank's
/// group places.
///
/// A deployment that places the whole pipeline serves what the checkpoint states when that is
/// one network's. A group that places no network (a text encoder's group, a decoder's) serves
/// beside whichever network the deployment places elsewhere, so its stages serve every task the
/// checkpoint states.
fn served_tasks(
    stated: &[VideoTask],
    stages: Option<&BTreeSet<String>>,
    checkpoint: &Path,
) -> Result<Vec<VideoTask>> {
    if stated.is_empty() {
        return Err(Error::config(format!(
            "the checkpoint {} states no task, so no network can be sized for one",
            checkpoint.display()
        )));
    }

    let Some(stages) = stages else {
        let first = network(stated[0]);
        if stated.iter().all(|task| network(*task) == first) {
            return Ok(stated.to_vec());
        }

        return Err(Error::config(format!(
            "the checkpoint {} holds two networks and this deployment places the whole \
             pipeline: a deployment file that places one of them chooses it",
            checkpoint.display()
        )));
    };

    if !NETWORKS.iter().any(|stage| stages.contains(stage.name())) {
        return Ok(stated.to_vec());
    }

    let served: Vec<VideoTask> = stated
        .iter()
        .copied()
        .filter(|task| stages.contains(network(*task).name()))
        .collect();
    if served.is_empty() {
        return Err(Error::config(format!(
            "the checkpoint {} states {stated:?} and the deployment places no network of them",
            checkpoint.display()
        )));
    }
    Ok(served)
}

/// A keyframe of `resolution`, anchoring the frame of `role`.
fn keyframe(role: ConditionRole, resolution: &ServedCanvas) -> ConditionShape {
    ConditionShape {
        role,
        frames: 1,
        height: resolution.canvas.height,
        width: resolution.canvas.width,
        audio_samples: 0,
    }
}

/// The condition rows a deployment reserves to serve `task` at the largest resolution the
/// checkpoint generates, in whole capacity units: the keyframes the task anchors, or a
/// reference clip and its soundtrack, row for row. A task with no conditions still reserves
/// one unit.
fn condition_capacity(config: &Config, task: VideoTask) -> Result<u32> {
    let stage = network(task);
    let denoiser = config.denoisers.get(&stage).ok_or_else(|| {
        Error::config(format!("the checkpoint holds no {} network", stage.name()))
    })?;

    let mut rows = 0;
    for resolution in resolutions()? {
        let conditions = match task {
            VideoTask::TextToVideoAudio => Vec::new(),
            VideoTask::FirstLastToVideoAudio => vec![
                keyframe(ConditionRole::FirstFrame, &resolution),
                keyframe(ConditionRole::LastFrame, &resolution),
            ],
            // A reference with a soundtrack is one clip of the frame rule's frames and a track
            // as long as the clip.
            VideoTask::ReferenceToVideoAudio => vec![ConditionShape {
                role: ConditionRole::Reference,
                frames: FRAMES.clip_frames,
                height: resolution.canvas.height,
                width: resolution.canvas.width,
                audio_samples: FRAMES.clip_frames * AUDIO_SAMPLE_RATE / FPS,
            }],
        };

        let taken: usize = conditions
            .iter()
            .map(|condition| {
                let (audio, visual) = packing::condition_rows(condition);
                audio + visual
            })
            .sum();
        rows = rows.max(taken as u32);
    }

    Ok(rows.max(1).next_multiple_of(denoiser.capacity_unit()))
}

/// The frames of the least video of the frame rule that lasts at least `seconds`.
fn timeline(seconds: f64) -> Result<u32> {
    let frames = processing::requested_frames(seconds, FPS);
    if !seconds.is_finite() || frames == 0 || frames > u32::MAX - FRAMES.clip_frames {
        return Err(Error::config(format!(
            "{seconds} seconds is not a positive number of frames at {FPS} frames per second"
        )));
    }
    Ok(FRAMES.align_up(frames))
}

/// The resolutions every H3 checkpoint generates: the resolution rule's of each named aspect
/// ratio.
fn resolutions() -> Result<Vec<ServedCanvas>> {
    named_canvases().map_err(|error| Error::config(format!("the resolution rule: {error}")))
}

/// The resolutions of the sizes a deployment serves, each with the aspect ratio that names it,
/// in the deployment's order.
fn served_resolutions(sizes: &[(u32, u32)]) -> Result<Vec<ServedCanvas>> {
    if sizes.is_empty() {
        return Err(Error::config(
            "this deployment serves no size, so no raster could be decoded".to_owned(),
        ));
    }
    let generated = resolutions()?;

    sizes
        .iter()
        .map(|&(width, height)| {
            generated
                .iter()
                .find(|served| served.canvas.width == width && served.canvas.height == height)
                .cloned()
                .ok_or_else(|| {
                    Error::config(format!(
                        "this deployment serves {width}x{height}, which the checkpoint does not \
                         generate: it generates {}",
                        generated
                            .iter()
                            .map(|served| format!(
                                "{}x{}",
                                served.canvas.width, served.canvas.height
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })
        })
        .collect()
}

/// The requests a deployment captures its denoiser's graphs for: one per (task, size, frame
/// count, prompt capacity, condition rung) the deployment serves. An `fl2va` request also
/// states the keyframe sets a request of its layout may open with ([`keyframe_openings`]), and
/// a `ref2va` one the reference sets that bound those its requests open with
/// ([`reference_openings`]).
fn candidates(
    tasks: &[VideoTask],
    serving: &VideoServing,
    ladder: &[u32],
    packing: &ConditionPacking,
    processor: Option<&processing::qwen_vl::ProcessorConfig>,
) -> Result<Vec<(VideoRequest, Vec<Vec<ConditionShape>>)>> {
    let longest = timeline(serving.max_seconds)?;
    let shortest = timeline(MIN_SECONDS)?;
    let frame_counts: Vec<u32> = (shortest..=longest)
        .step_by(FRAMES.clip_frames as usize)
        .collect();

    let mut candidates = Vec::new();
    for resolution in served_resolutions(&serving.sizes)? {
        for task in tasks.iter().copied() {
            // `fl2va` anchors both keyframes. A `ref2va` reference's raster follows the
            // client's media, so its request counts a reference, which makes it a `ref2va`
            // request, states no condition and rides its rung's condition rows.
            let keyframes = u32::from(task == VideoTask::FirstLastToVideoAudio);
            let references = u32::from(task == VideoTask::ReferenceToVideoAudio);
            let rungs: Vec<Option<u32>> = if task.conditioned() {
                ladder.iter().copied().map(Some).collect()
            } else {
                vec![None]
            };
            let openings = match task {
                VideoTask::FirstLastToVideoAudio => {
                    keyframe_openings(&resolution, ladder, packing, processor)?
                }
                VideoTask::ReferenceToVideoAudio => {
                    reference_openings(&resolution, longest, ladder, packing)
                }
                VideoTask::TextToVideoAudio => Vec::new(),
            };

            for frames in frame_counts.iter().copied() {
                for capacity in &serving.text_capacities {
                    for rung in rungs.iter().copied() {
                        let request = VideoRequest {
                            width: resolution.canvas.width,
                            height: resolution.canvas.height,
                            frames,
                            fps: FPS,
                            first_frames: keyframes,
                            last_frames: keyframes,
                            references,
                            text_tokens: *capacity,
                            conditions: Vec::new(),
                            vision_spans: Vec::new(),
                            condition_rows: rung,
                        };
                        let opened = openings
                            .iter()
                            .filter(|(served, _)| Some(*served) == rung)
                            .map(|(_, set)| set.clone())
                            .collect();

                        candidates.push((request, opened));
                    }
                }
            }
        }
    }
    Ok(candidates)
}

/// The keyframe sets an `fl2va` request of `resolution` may present, each with the rung of
/// `ladder` that serves it: every non-empty subset of the anchors, each keyframe fitted to the
/// resolution by request processing's own rule, on the smallest rung holding the rows it packs
/// into.
fn keyframe_openings(
    resolution: &ServedCanvas,
    ladder: &[u32],
    packing: &ConditionPacking,
    processor: Option<&processing::qwen_vl::ProcessorConfig>,
) -> Result<Vec<(u32, Vec<ConditionShape>)>> {
    let processor = processor.ok_or_else(|| {
        Error::checkpoint(
            "the deployment serves fl2va and the checkpoint states no \
             processor/video_preprocessor_config.json, so a keyframe cannot be prepared"
                .to_owned(),
        )
    })?;
    let (width, height) = (resolution.canvas.width, resolution.canvas.height);

    // An image of the resolution's own size: the preparation fits every image onto it.
    let image = processing::ImageMetadata { width, height };
    let anchors = [ConditionRole::FirstFrame, ConditionRole::LastFrame];

    let mut openings = Vec::new();
    for mask in 1_u32..(1 << anchors.len()) {
        let roles: Vec<ConditionRole> = anchors
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, role)| *role)
            .collect();

        let prepared = roles
            .iter()
            .enumerate()
            .map(|(order, role)| {
                PreparedCondition::keyframe(
                    order,
                    *role,
                    &image,
                    resolution.canvas,
                    order,
                    true,
                    processor,
                )
            })
            .collect::<std::result::Result<Vec<_>, String>>()
            .map_err(|error| Error::config(format!("a keyframe of {width}x{height}: {error}")))?;

        let rows = processing::minimax_h3::condition_rows(&prepared, resolution.canvas, packing);
        let Some(rung) = ladder.iter().copied().find(|rung| *rung >= rows) else {
            continue;
        };

        let shapes = roles
            .iter()
            .map(|role| keyframe(*role, resolution))
            .collect();
        openings.push((rung, shapes));
    }
    Ok(openings)
}

/// The reference sets a `ref2va` request of `resolution` opens its layout's profile with, each
/// with the rung of `ladder` that serves it: per rung, the largest image an admitted request
/// presents that the rung's rows hold, the longest and widest video with its soundtrack they
/// hold, and the longest track alone they hold.
///
/// A reference's raster and length follow the client's media, so no set of the deployment's is
/// a request's own. The profile run opens each set, which states every kernel a reference's
/// encode launches (their keys follow element types and not extents) and the most of the
/// runner's region an opening takes: an opening encodes its references one after another, and
/// on its rung none is larger in pixels, in frames or in samples than one of these. An image
/// reference is encoded at a short edge of at most the declared edge
/// ([`REFERENCE_LIMITS`]`.image_short_edge`) within the canvas rule's aspects, so the largest
/// is that edge by four times it, fewer cells where the rung holds fewer, as square as they
/// divide. A video reference is put on the canvas of its aspect and lasts at most the generated
/// video: the longest lasts `longest` frames, the deployment's longest output, at
/// `resolution`'s canvas, its frames and then its sides halved until its rows fit the rung.
fn reference_openings(
    resolution: &ServedCanvas,
    longest: u32,
    ladder: &[u32],
    packing: &ConditionPacking,
) -> Vec<(u32, Vec<ConditionShape>)> {
    // A condition row is one latent patch of the raster's cells, which the packing tiles.
    let cell = processing::minimax_h3::LATENT_COMPRESSION * processing::minimax_h3::PATCH_SIDE;
    let reference = |frames: u32, height: u32, width: u32, audio_samples: u32| ConditionShape {
        role: ConditionRole::Reference,
        frames,
        height,
        width,
        audio_samples,
    };
    let rows = |shape: &ConditionShape| -> u32 {
        let (audio, visual) = packing::condition_rows(shape);
        let grid = [
            FRAMES.latent_frames(shape.frames),
            shape.height / cell,
            shape.width / cell,
        ];
        let visual = match (shape.frames, visual) {
            (_, 0) => processing::minimax_h3::rules::VisualRows::None,
            (1, rows) => processing::minimax_h3::rules::VisualRows::Patches(rows as u32),
            _ => processing::minimax_h3::rules::VisualRows::Grid(grid),
        };
        let tiles = processing::minimax_h3::rules::condition_tiles(audio as u32, visual, packing);
        tiles.saturating_mul(packing.tile_rows.max(1))
    };
    let samples = |frames: u32| frames * AUDIO_SAMPLE_RATE / FPS;

    let mut openings: Vec<(u32, Vec<ConditionShape>)> = Vec::new();
    for &rung in ladder {
        let mut sets = Vec::new();

        // The image of the most cells the rung holds within the declared edge and the widest
        // aspect, as square as they divide.
        let tile = packing.tile_rows.max(1);
        let cells = rung / tile * tile;
        let (narrow, wide) = (CANVAS_RULE.widest.1.max(1), CANVAS_RULE.widest.0);
        let short = (REFERENCE_LIMITS.image_short_edge / cell).max(1);
        let side = (f64::from(cells).sqrt() as u32).clamp(1, short);
        let long = (cells / side).min(side * wide / narrow);
        if long > 0 {
            sets.push(reference(1, side * cell, long * cell, 0));
        }

        // The longest video at the widest raster the rung holds, with its soundtrack.
        let (mut height, mut width) = (resolution.canvas.height, resolution.canvas.width);
        'video: while height >= cell && width >= cell {
            let mut frames = FRAMES.align_down(longest);
            while let Some(count) = frames {
                let video = reference(count, height, width, samples(count));
                if rows(&video) <= rung {
                    sets.push(video);
                    break 'video;
                }
                frames = count
                    .checked_sub(FRAMES.clip_frames)
                    .filter(|shorter| FRAMES.is_legal(*shorter));
            }
            height = (height / 2 / cell) * cell;
            width = (width / 2 / cell) * cell;
        }

        // The longest track alone the rung holds.
        let mut track = samples(longest);
        while track > 0 {
            let audio = reference(0, 0, 0, track);
            if rows(&audio) <= rung {
                sets.push(audio);
                break;
            }
            track /= 2;
        }

        // Each set on the smallest rung that holds it, as a request of it rides.
        for set in sets {
            let taken = rows(&set);
            if let Some(served) = ladder.iter().copied().find(|served| *served >= taken) {
                let opening = (served, vec![set]);
                if !openings.contains(&opening) {
                    openings.push(opening);
                }
            }
        }
    }
    openings
}

/// The prompts with vision blocks whose eager encodes a deployment's text encoder is profiled
/// over ([`VideoModel::warm_prompts`]).
///
/// - `fl2va` fixes its grids: per served resolution, a prompt of each keyframe count, each
///   keyframe at the processor's grid of the resolution.
/// - `ref2va` presents its references at grids that follow the client's media, so it warms the
///   prompts that bound every presentation it admits ([`reference_prompts`]).
/// - `t2va` presents none.
///
/// A prompt's ids present each block as the text encoder reads a vision block: one `placeholder`
/// (the tower's image token) per merged row, between two text tokens that stand for the block's
/// `<|vision_start|>` and `<|vision_end|>` markers. An encode's work follows from the placeholder
/// runs and the prompt's length alone, so any text token stands for a marker.
#[expect(
    clippy::type_complexity,
    reason = "each prompt's grids and ids, as `VideoModel::warm_prompts` states them"
)]
fn warm_prompts(
    tasks: &[VideoTask],
    serving: &VideoServing,
    (processor, placeholder): (Option<&processing::qwen_vl::ProcessorConfig>, Option<u32>),
) -> Result<Vec<(Vec<[u32; 3]>, Vec<u32>)>> {
    /// The keyframes an `fl2va` prompt presents at most: the first frame and the last.
    const KEYFRAMES: u32 = 2;

    let keyframes = tasks.contains(&VideoTask::FirstLastToVideoAudio);
    let references = tasks.contains(&VideoTask::ReferenceToVideoAudio);
    if !keyframes && !references {
        return Ok(Vec::new());
    }
    let processor = processor.ok_or_else(|| {
        Error::checkpoint(
            "the deployment serves vision blocks and the checkpoint states no \
             processor/video_preprocessor_config.json, so the vision grid a block presents \
             cannot be sized"
                .to_owned(),
        )
    })?;
    let placeholder = placeholder.ok_or_else(|| {
        Error::checkpoint(
            "the deployment serves vision blocks and the text encoder states no vision tower, \
             so no prompt presents one"
                .to_owned(),
        )
    })?;

    let mut prompts: Vec<(Vec<[u32; 3]>, Vec<u32>)> = Vec::new();
    if keyframes {
        for resolution in served_resolutions(&serving.sizes)? {
            let (width, height) = (resolution.canvas.width, resolution.canvas.height);
            let grid = processor.image_grid(height, width).map_err(|error| {
                Error::config(format!("a keyframe of {width}x{height}: {error}"))
            })?;

            // Each temporal step of a block presents one run of merged tokens.
            let tokens = grid.t * processor.block_tokens(grid);
            let grids = [[grid.t, grid.h, grid.w]];
            for count in 1..=KEYFRAMES {
                let prompt = presented(&grids.repeat(count as usize), &[tokens], placeholder);
                if !prompts.contains(&prompt) {
                    prompts.push(prompt);
                }
            }
        }
    }
    if references {
        for prompt in reference_prompts(serving, processor.merge_size, placeholder) {
            if !prompts.contains(&prompt) {
                prompts.push(prompt);
            }
        }
    }
    Ok(prompts)
}

/// The prompts that bound every `ref2va` presentation a deployment serving `serving` admits:
/// per prompt capacity, the capacity's tokens presenting vision rows in one block, and in as
/// many blocks as a request presents at most, every token but the blocks' markers a vision
/// row; each block an image of one time step and `merge` (the tower's merge side) patch rows,
/// its merged rows sharing the blocks' rows as evenly as they divide.
///
/// An admitted presentation rides a capacity of at least its tokens (the deployment refuses a
/// prompt longer than its largest), and every condition with a vision view presents one grid,
/// so it presents at most that capacity less two markers per grid as vision rows, in at most
/// [`REFERENCE_LIMITS`]' images and videos. The text encoder's host values and the region its
/// eager encode computes in grow with the vision rows and the grids that present them, and an
/// image of one time step costs the most of any grid per merged row: the position table is
/// sampled for every frame, and a video's time steps share their frame's. The capacity's
/// presentation in one grid holds the largest grid any admitted prompt encodes, and in the most
/// grids the most host copies (`Graphs::input_peak` holds every forward of no more copies and
/// no more bytes than a profiled one).
fn reference_prompts(
    serving: &VideoServing,
    merge: u32,
    placeholder: u32,
) -> Vec<(Vec<[u32; 3]>, Vec<u32>)> {
    // The markers around each block's run of placeholders.
    const MARKERS: u32 = 2;
    let blocks =
        (REFERENCE_LIMITS.images + REFERENCE_LIMITS.videos).clamp(1, REFERENCE_LIMITS.total.max(1));
    let merge = merge.max(1);

    let mut prompts = Vec::new();
    for &capacity in &serving.text_capacities {
        for count in [1, blocks] {
            // Each block holds at least one vision row beside its markers.
            if capacity < count * (MARKERS + 1) {
                continue;
            }
            let rows = capacity - count * MARKERS;
            let shares: Vec<u32> = (0..count)
                .map(|block| rows / count + u32::from(block < rows % count))
                .collect();

            // A block of `share` merged rows is `merge` patch rows by `merge · share` columns.
            let grids: Vec<[u32; 3]> = shares
                .iter()
                .map(|share| [1, merge, merge * share])
                .collect();
            prompts.push(presented(&grids, &shares, placeholder));
        }
    }
    prompts
}

/// The prompt that presents `grids`, block `i` of `tokens[i]` merged rows (the last entry
/// standing for every later block), as the text encoder reads a vision block: its placeholders
/// between two marker tokens.
fn presented(grids: &[[u32; 3]], tokens: &[u32], placeholder: u32) -> (Vec<[u32; 3]>, Vec<u32>) {
    /// The text token that stands for a block's markers.
    const MARKER: u32 = 0;

    let mut ids = Vec::new();
    for index in 0..grids.len() {
        let rows = tokens.get(index).or(tokens.last()).copied().unwrap_or(0);
        ids.push(MARKER);
        ids.extend(std::iter::repeat_n(placeholder, rows as usize));
        ids.push(MARKER);
    }
    (grids.to_vec(), ids)
}

#[cfg(test)]
mod tests {
    use processing::qwen_vl::ProcessorConfig;

    use super::*;

    /// The released processor: 16-pixel patches merged 2x2, an image budget of 64Ki to 16Mi
    /// pixels.
    fn processor() -> ProcessorConfig {
        ProcessorConfig::parse(
            r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
            r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 4096, "longest_edge": 25165824}}"#,
        )
        .unwrap()
    }

    fn serving() -> VideoServing {
        VideoServing {
            sizes: vec![(1344, 768)],
            max_seconds: 4.0,
            text_capacities: vec![2048],
            condition_rows: None,
        }
    }

    /// The pipeline execution runs is one it can run: its products' phases and its blocks'
    /// phases are phases of it, and each product and block is declared once.
    #[test]
    fn the_pipeline_is_a_declaration_execution_runs() {
        let pipeline = pipeline();
        pipeline.check().unwrap();
        assert_eq!(pipeline.serving(Phase::Denoise).count(), 2);
        assert_eq!(pipeline.kind(Product::EncodedUnits), Some(8));
    }

    /// A deployment that places the whole pipeline serves what the checkpoint states when that
    /// is one network's, and a checkpoint that states no task serves nothing.
    #[test]
    fn the_served_tasks_are_those_of_the_checkpoints_one_network() {
        let checkpoint = Path::new("/models/fasth3");
        let tasks = [
            VideoTask::TextToVideoAudio,
            VideoTask::FirstLastToVideoAudio,
        ];
        assert_eq!(served_tasks(&tasks, None, checkpoint).unwrap(), tasks);

        let none = served_tasks(&[], None, checkpoint).unwrap_err();
        assert!(none.to_string().contains("states no task"), "{none}");

        let both = [
            VideoTask::TextToVideoAudio,
            VideoTask::ReferenceToVideoAudio,
        ];
        assert!(
            served_tasks(&both, None, checkpoint).is_err(),
            "two networks need a deployment file to choose between them"
        );
    }

    /// A deployment that places one network serves that network's tasks, and one that places
    /// none of the checkpoint's networks is refused.
    #[test]
    fn a_placed_network_chooses_the_tasks() {
        let checkpoint = Path::new("/models/minimax-h3");
        let both = [
            VideoTask::TextToVideoAudio,
            VideoTask::ReferenceToVideoAudio,
        ];
        let placed = |names: &[&str]| -> BTreeSet<String> {
            names.iter().map(|name| (*name).to_owned()).collect()
        };

        assert_eq!(
            served_tasks(
                &both,
                Some(&placed(&["transformer_ref", "text_encoder"])),
                checkpoint
            )
            .unwrap(),
            [VideoTask::ReferenceToVideoAudio]
        );

        // A group that places no network serves beside the one the deployment places.
        assert_eq!(
            served_tasks(&both, Some(&placed(&["text_encoder"])), checkpoint).unwrap(),
            both
        );

        // A network the checkpoint does not hold serves no task.
        let base = [VideoTask::TextToVideoAudio];
        assert!(served_tasks(&base, Some(&placed(&["transformer_ref"])), checkpoint).is_err());
    }

    /// Every task places the text encoder, the decoders and the splice with its network; a
    /// conditioned task places the condition encoders too, and two networks are refused.
    #[test]
    fn a_tasks_stages_are_its_network_and_what_every_task_returns() {
        let config = crate::minimax_h3::config::tests_support::released_config();

        let plain = stages(&config, &[VideoTask::TextToVideoAudio], None).unwrap();
        assert!(plain.contains(Stage::Transformer) && plain.contains(Stage::VideoPostprocessor));
        assert!(!plain.contains(Stage::LatentEncoder));

        let reference = stages(&config, &[VideoTask::ReferenceToVideoAudio], None).unwrap();
        assert!(reference.contains(Stage::TransformerRef));
        assert!(reference.contains(Stage::LatentEncoder));

        let both = [
            VideoTask::TextToVideoAudio,
            VideoTask::ReferenceToVideoAudio,
        ];
        assert!(stages(&config, &both, None).is_err());

        // A group builds only the stages it places: a text encoder's group holds no network,
        // and a network's group without the condition encoders holds none of them.
        let placed = |names: &[&str]| -> BTreeSet<String> {
            names.iter().map(|name| (*name).to_owned()).collect()
        };
        let text = stages(&config, &both, Some(&placed(&["text_encoder"]))).unwrap();
        assert_eq!(text, Stages::of([Stage::TextEncoder]));

        let flow = stages(
            &config,
            &[VideoTask::ReferenceToVideoAudio],
            Some(&placed(&["transformer_ref", "video_decoder"])),
        )
        .unwrap();
        // The clip splice comes with the video decoder that a deployment names.
        assert_eq!(
            flow,
            Stages::of([
                Stage::TransformerRef,
                Stage::VideoDecoder,
                Stage::VideoPostprocessor
            ])
        );
        let decoders = stages(
            &config,
            &[VideoTask::TextToVideoAudio],
            Some(&placed(&["video_decoder", "audio_decoder"])),
        )
        .unwrap();
        assert!(decoders.contains(Stage::VideoPostprocessor));
        let track = stages(
            &config,
            &[VideoTask::TextToVideoAudio],
            Some(&placed(&["audio_decoder"])),
        )
        .unwrap();
        assert_eq!(track, Stages::of([Stage::AudioDecoder]));
    }

    /// A video is `17n + 5` frames, and a duration is served by the least such count covering
    /// it; a duration of no frame is refused.
    #[test]
    fn a_duration_aligns_to_the_least_frame_count_covering_it() {
        assert_eq!(timeline(4.0).unwrap(), 107);
        assert_eq!(timeline(5.0).unwrap(), 124);
        assert_eq!(timeline(15.0).unwrap(), 362);
        assert_eq!(timeline(0.5).unwrap(), 22, "one clip and its tail");
        assert!(timeline(0.0).is_err());
        assert!(timeline(f64::NAN).is_err());
    }

    /// A size the checkpoint does not generate is refused by name, with the sizes it does.
    #[test]
    fn a_size_the_checkpoint_does_not_generate_is_refused() {
        let served = served_resolutions(&[(1344, 768), (768, 1344)]).unwrap();
        assert_eq!(served.len(), 2);
        assert_eq!(served[0].aspect_ratio, "16:9");

        let message = served_resolutions(&[(1000, 1000)]).unwrap_err().to_string();
        assert!(
            message.contains("1000x1000") && message.contains("1344x768"),
            "{message}"
        );
    }

    /// The condition capacity a task reserves by default: two keyframes of the largest
    /// resolution for `fl2va`, a reference clip and its soundtrack for `ref2va`, and one unit
    /// for a task without conditions.
    #[test]
    fn a_task_reserves_the_condition_rows_of_its_largest_request() {
        let config = crate::minimax_h3::config::tests_support::released_config();
        let capacity = |task| condition_capacity(&config, task).unwrap();

        assert_eq!(capacity(VideoTask::TextToVideoAudio), 64);

        // A 1344x768 keyframe is 42x24 patches: 1008 rows, and two of them 2016.
        assert_eq!(capacity(VideoTask::FirstLastToVideoAudio), 2048);

        // A 17-frame clip is two latent frames, and its 22667-sample track 29 latent steps of
        // two channels: 2016 + 58 rows.
        assert_eq!(capacity(VideoTask::ReferenceToVideoAudio), 2112);
    }

    /// An `fl2va` deployment admits, per served resolution, a prompt of one keyframe and one
    /// of both anchors, each keyframe at the processor's grid of the resolution; a `t2va` one
    /// admits none.
    #[test]
    fn a_keyframe_presents_the_grid_of_its_resolution() {
        // 768x1344 pixels are 48x84 patches of one temporal step, which merge to 1008 tokens:
        // a keyframe's run of placeholders between its two markers.
        const IMAGE: u32 = 151_655;
        let prompts = warm_prompts(
            &[VideoTask::FirstLastToVideoAudio],
            &serving(),
            (Some(&processor()), Some(IMAGE)),
        )
        .unwrap();
        let grids: Vec<&Vec<[u32; 3]>> = prompts.iter().map(|(grids, _)| grids).collect();
        assert_eq!(grids, [&vec![[1, 48, 84]], &vec![[1, 48, 84], [1, 48, 84]]]);
        for (grids, ids) in &prompts {
            let placeholders = ids.iter().filter(|id| **id == IMAGE).count();
            assert_eq!(placeholders, 1008 * grids.len());
            assert_eq!(ids.len(), 1010 * grids.len());
            // Each block is one run of placeholders, apart from the next.
            let runs = ids
                .chunk_by(|left, right| (*left == IMAGE) == (*right == IMAGE))
                .filter(|run| run[0] == IMAGE)
                .count();
            assert_eq!(runs, grids.len());
        }

        assert_eq!(
            warm_prompts(&[VideoTask::TextToVideoAudio], &serving(), (None, None)).unwrap(),
            Vec::new()
        );
        assert!(
            warm_prompts(
                &[VideoTask::FirstLastToVideoAudio],
                &serving(),
                (None, Some(IMAGE))
            )
            .is_err(),
            "an fl2va deployment cannot size a keyframe without the processor"
        );
        assert!(
            warm_prompts(
                &[VideoTask::FirstLastToVideoAudio],
                &serving(),
                (Some(&processor()), None)
            )
            .is_err(),
            "an fl2va deployment presents no keyframe without a vision tower"
        );
    }

    /// A `ref2va` deployment, whose references present grids that follow the client's media,
    /// warms per prompt capacity the capacity's tokens presenting vision rows in one block and
    /// in the most blocks a request presents (its images and videos), every token beside the
    /// blocks' markers a vision row, each block's grid merging to its run of placeholders.
    #[test]
    fn a_reference_deployment_warms_the_most_vision_rows_each_capacity_presents() {
        const IMAGE: u32 = 151_655;
        let serving = VideoServing {
            text_capacities: vec![1024, 8192],
            ..serving()
        };
        let prompts = warm_prompts(
            &[VideoTask::ReferenceToVideoAudio],
            &serving,
            (Some(&processor()), Some(IMAGE)),
        )
        .unwrap();

        let most = REFERENCE_LIMITS.images + REFERENCE_LIMITS.videos;
        let shape: Vec<(usize, usize)> = prompts
            .iter()
            .map(|(grids, ids)| (ids.len(), grids.len()))
            .collect();
        assert_eq!(
            shape,
            [
                (1024, 1),
                (1024, most as usize),
                (8192, 1),
                (8192, most as usize)
            ]
        );
        for (grids, ids) in &prompts {
            // Two markers a block, and every other token a placeholder.
            let placeholders = ids.iter().filter(|id| **id == IMAGE).count();
            assert_eq!(placeholders, ids.len() - 2 * grids.len());

            // Each block is one run, of the merged rows its grid's patches make.
            let runs: Vec<usize> = ids
                .chunk_by(|left, right| (*left == IMAGE) == (*right == IMAGE))
                .filter(|run| run[0] == IMAGE)
                .map(<[u32]>::len)
                .collect();
            let merged: Vec<usize> = grids
                .iter()
                .map(|[time, rows, columns]| (time * rows * columns / 4) as usize)
                .collect();
            assert_eq!(runs, merged);
        }
    }

    /// A `ref2va` layout's profile opens with, per rung, the largest image, the longest and
    /// widest video with its soundtrack and the longest track alone the rung's rows hold, each
    /// on the smallest rung that holds it, as a request of it rides, whatever the client's media
    /// are; a rung too small for a video at the canvas holds one of halved sides.
    #[test]
    fn a_reference_rung_opens_with_the_largest_references_it_holds() {
        // OmniRef's tiling: 128-row tiles, a video's token grid in tiles of 4 x 4 x 8 tokens.
        let packing = ConditionPacking {
            capacity: 49_152,
            tile_rows: 128,
            video_tile: [4, 4, 8],
        };
        let ladder = [1024, 49_152];
        let candidates = candidates(
            &[VideoTask::ReferenceToVideoAudio],
            &serving(),
            &ladder,
            &packing,
            Some(&processor()),
        )
        .unwrap();
        assert_eq!(candidates.len(), 2, "one request per rung");

        let mut kinds = Vec::new();
        for (request, sets) in &candidates {
            let rung = request.condition_rows.unwrap();
            for set in sets {
                let [reference] = set.as_slice() else {
                    panic!("one reference a set: {set:?}");
                };
                assert_eq!(reference.role, ConditionRole::Reference);
                kinds.push((reference.frames > 0, reference.audio_samples > 0));

                // The smallest rung that holds the set is the one it opens.
                let (audio, visual) = packing::condition_rows(reference);
                assert!(audio + visual <= rung as usize, "{reference:?} on {rung}");
            }
        }
        // An image, a video with its soundtrack and a track alone.
        for kind in [(true, false), (true, true), (false, true)] {
            assert!(kinds.contains(&kind), "{kinds:?}");
        }

        // The widest rung holds the largest image a request presents: the declared short edge
        // by four times it.
        let image = candidates
            .iter()
            .flat_map(|(_, sets)| sets.iter().flatten())
            .filter(|reference| reference.frames == 1)
            .map(|reference| (reference.height, reference.width))
            .max();
        assert_eq!(image, Some((2048, 8192)));

        // The widest rung holds the longest video at the canvas; the narrowest one of halved
        // sides.
        let video = |rung: u32| {
            let (_, sets) = candidates
                .iter()
                .find(|(request, _)| request.condition_rows == Some(rung))
                .unwrap();
            sets.iter()
                .flatten()
                .find(|reference| reference.frames > 1)
                .copied()
                .unwrap()
        };
        assert_eq!(
            (
                video(49_152).frames,
                video(49_152).width,
                video(49_152).height
            ),
            (107, 1344, 768)
        );
        assert!(video(1024).width < 1344);
    }

    /// An `fl2va` layout opens with the keyframe sets its rung serves, each keyframe at the
    /// resolution's raster: one keyframe (either anchor) on the rung that holds one frame,
    /// both on the rung that holds two, and a `t2va` layout with none.
    #[test]
    fn a_rung_opens_with_the_keyframes_it_holds() {
        // Row for row: a 1344x768 keyframe is 42x24 latent patches, 1008 rows.
        let packing = ConditionPacking {
            capacity: 2048,
            tile_rows: 1,
            video_tile: [1, 1, 1],
        };
        let candidates = candidates(
            &[
                VideoTask::TextToVideoAudio,
                VideoTask::FirstLastToVideoAudio,
            ],
            &serving(),
            &[1024, 2048],
            &packing,
            Some(&processor()),
        )
        .unwrap();

        // One request per task and rung at the one size, frame count and prompt capacity.
        assert_eq!(candidates.len(), 3);
        assert!(candidates.iter().all(|(request, _)| request.frames == 107));

        // A request states the task it is captured for by the conditions it counts.
        let tasks: Vec<VideoTask> = candidates
            .iter()
            .map(|(request, _)| {
                VideoTask::of(
                    request.first_frames,
                    request.last_frames,
                    request.references,
                )
            })
            .collect();
        assert_eq!(
            tasks,
            [
                VideoTask::TextToVideoAudio,
                VideoTask::FirstLastToVideoAudio,
                VideoTask::FirstLastToVideoAudio,
            ]
        );
        let reference = super::candidates(
            &[VideoTask::ReferenceToVideoAudio],
            &serving(),
            &[1024],
            &packing,
            None,
        )
        .unwrap();
        assert_eq!(reference[0].0.references, 1);
        assert_eq!(reference[0].0.condition_rows, Some(1024));

        let resolution = &served_resolutions(&[(1344, 768)]).unwrap()[0];
        let opened = |rung: Option<u32>| -> Vec<Vec<ConditionShape>> {
            candidates
                .iter()
                .find(|(request, _)| request.condition_rows == rung)
                .unwrap()
                .1
                .clone()
        };

        assert_eq!(opened(None), Vec::<Vec<ConditionShape>>::new());
        assert_eq!(
            opened(Some(1024)),
            vec![
                vec![keyframe(ConditionRole::FirstFrame, resolution)],
                vec![keyframe(ConditionRole::LastFrame, resolution)],
            ]
        );
        assert_eq!(
            opened(Some(2048)),
            vec![vec![
                keyframe(ConditionRole::FirstFrame, resolution),
                keyframe(ConditionRole::LastFrame, resolution),
            ]]
        );
    }

    #[test]
    fn a_fasth3_adapter_names_its_student_and_the_base_release() {
        let metadata = |student: &str, base: &str| {
            std::collections::BTreeMap::from([
                (
                    "format".to_owned(),
                    nn::adapter::FASTVIDEO_FORMAT.to_owned(),
                ),
                ("finetuned_model".to_owned(), student.to_owned()),
                ("base_model".to_owned(), base.to_owned()),
            ])
        };

        assert!(fasth3_adapter(&metadata(
            "FastVideo/FastVideo-FastH3-4-step-v1",
            FASTH3_BASE_MODEL
        )));
        assert!(!fasth3_adapter(&metadata(
            "someone/h3-style",
            FASTH3_BASE_MODEL
        )));
        assert!(!fasth3_adapter(&metadata(
            "FastVideo/FastVideo-FastH3-4-step-v1",
            "other/base"
        )));
    }
}

//! MiniMax-H3 and FastH3 as execution loads them and request processing serves them
//! ([`nn::family`]).
//!
//! [`FAMILY`] claims the `MiniMaxH3ModularPipeline` class. A deployment's sources compose to one
//! checkpoint and its [`Config`], a load builds the stages the deployment places on a rank with
//! the deployment's sequence and tensor splits, and the loaded [`MinimaxH3`] states the
//! pipeline's own rules to execution: the `17n + 5` frame rule, its resolutions, the kinds of
//! request its network serves with the rows their conditions take and the largest references a
//! layout holds, how a prompt presents a keyframe, and the description its request processing
//! admits requests against. Execution enumerates the requests it captures and the prompts it
//! warms from them.
//!
//! ## Composition
//!
//! A deployment names a checkpoint, and may name the base release it continues (`base`) and an
//! adapter directory (`lora_path`) ([`Sources`]). The family composes them by what the
//! checkpoint's own files declare, and each component ([`Component`]) comes from exactly one
//! source:
//!
//! | Checkpoint | Declared by | Network | Schedulers | Text encoder, processor, VAEs |
//! | --- | --- | --- | --- | --- |
//! | Base release | its pipeline index, and no contract | each the index declares, on the full-step grid | the release | the release |
//! | Student export | its contract ([`Contract`]) | the contract's | the export | the base the contract names, which the deployment's `base` supplies; the export when it names none |
//! | Base release and adapter | the contract beside the adapter | the contract's: the release's network with the adapter fused | the release | the release |
//!
//! A component's directory is the one the checkpoint's pipeline index declares for it, wherever
//! its source lies. A component absent from its source, a component an export holds that its
//! contract takes from the base, a base the checkpoint does not continue, and an adapter without
//! a contract are refused.
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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use nn::family::{
    ConditionBounds, ConditionMedia, Description, Family, Model, RequestKind, VideoCheckpoint,
    VideoFamily, VideoModel, VideoPlacement, VisionTokens,
};
use nn::pipeline::{Block, Extent, Phase, Pipeline, Product, ProductSpec};
use nn::{
    Adapter, AudioFormat, Checkpoint, ConditionRole, ConditionShape, DType, Decoder, Encoder,
    MediaFormat, PipelineIndex, Result, Source, Sources, Splicer, WeightLoader,
};
use processing::minimax_h3::rules::{VisualRows, condition_tiles};
use processing::minimax_h3::{
    CANVAS_RULE, ConditionPacking, DenoiserInfo, MIN_SECONDS, PIPELINE_CLASS, PreparedCondition,
    REFERENCE_LIMITS, ReferenceLimits, ServedCanvas, named_canvases,
};
use proto::config::{Partition, Quantization};
use proto::video::Canvas;

use crate::minimax_h3::adapter::FastVideoLora;
use crate::minimax_h3::audio_vae::{AudioDecoder, AudioEncoder};
use crate::minimax_h3::config::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_SAMPLE_RATE, BaseRevision, Component, Contract, FPS, FRAMES,
    NETWORKS, Networks, RETAINED_LAYERS, network, precision, released_networks,
};
use crate::minimax_h3::denoiser::{Denoiser, SAMPLE_DTYPE};
use crate::minimax_h3::splice::VideoPostprocessor;
use crate::minimax_h3::stage::{Stage, Stages};
use crate::minimax_h3::video_vae::{self, VideoDecoder, VideoEncoder};
use crate::minimax_h3::{Config, ID, VideoTask, packing};
use crate::qwen3::Qwen3VlModel;

/// The MiniMax-H3 family: a diffusers modular pipeline, which request processing serves
/// through its video processor ([`processing::minimax_h3::H3Family`]).
pub const FAMILY: Family = Family {
    request: processing::minimax_h3::FAMILY,
    model: Model::Video(&MinimaxH3Family),
    step_stack: crate::minimax_h3::transformer::TRANSFORMER,
};

/// The family's value, which [`FAMILY`] refers to.
struct MinimaxH3Family;

impl VideoFamily for MinimaxH3Family {
    fn open(
        &self,
        sources: &Sources,
        quantization: &Quantization,
    ) -> nn::Result<(Checkpoint, Box<dyn VideoCheckpoint>)> {
        let (checkpoint, networks) = compose(sources)?;
        let mut config = Config::open(&checkpoint, &networks)?;
        precision::apply(quantization, &mut config)?;
        Ok((checkpoint, Box::new(config)))
    }
}

/// The components every deployment reads besides its networks.
const PIPELINE: [Component; 6] = [
    Component::TextEncoder,
    Component::Processor,
    Component::Vae,
    Component::AudioVae,
    Component::Scheduler,
    Component::AudioScheduler,
];

/// The components a student export holds besides its network: the schedulers its contract's
/// schedule is stated over.
const STUDENT: [Component; 2] = [Component::Scheduler, Component::AudioScheduler];

/// The checkpoint `sources` compose to, with the networks it serves, which its configuration is
/// read for ([`Config::open`]; [module documentation](self)).
///
/// # Errors
///
/// When the checkpoint declares another pipeline, its contract or the adapter's does not parse,
/// a base or an adapter is stated that the checkpoint does not take (or a base it continues is
/// not stated), a component is absent from its source or held by an export that takes it from
/// the base, or the adapter does not fuse into the network it edits.
pub fn compose(sources: &Sources) -> Result<(Checkpoint, Networks)> {
    let checkpoint = &sources.checkpoint;
    let index = pipeline_index(checkpoint)?;
    let export = Contract::read(checkpoint)?;
    let adapter = sources.adapter.as_ref().map(open_adapter).transpose()?;

    // The networks the checkpoint serves, and the base release its other components come from.
    let (networks, base) = match (export, &adapter) {
        (Some(_), Some(_)) => {
            return Err(ID.checkpoint(format!(
                "{} is a FastH3 student export (it states {}), and an adapter is fused into \
                 the base release",
                checkpoint.directory().display(),
                Contract::FILE
            )));
        }
        (Some(contract), None) => {
            let base = match (&contract.base, &sources.base) {
                (Some(revision), Some(base)) => {
                    continues(revision, base)?;
                    Some(base)
                }
                (Some(revision), None) => {
                    return Err(ID.checkpoint(format!(
                        "{} continues the base release {revision}, and the deployment states \
                         no snapshot of it (`base`)",
                        checkpoint.directory().display()
                    )));
                }
                (None, Some(base)) => {
                    return Err(ID.checkpoint(format!(
                        "the deployment states the base {}, and {} continues none: its {} \
                         states no base_model_revision",
                        base.directory().display(),
                        checkpoint.directory().display(),
                        Contract::FILE
                    )));
                }
                (None, None) => None,
            };
            (Networks::Student(contract), base)
        }
        (None, adapter) => {
            if let Some(base) = &sources.base {
                return Err(ID.checkpoint(format!(
                    "{} is a base release, which continues none, and the deployment states \
                     the base {}",
                    checkpoint.directory().display(),
                    base.directory().display()
                )));
            }
            match adapter {
                Some((contract, _)) => {
                    if let Some(revision) = &contract.base {
                        continues(revision, checkpoint)?;
                    }
                    (Networks::Student(contract.clone()), None)
                }
                None => (Networks::Released(released_networks(&index)?), None),
            }
        }
    };

    // Each component from the one source that holds it: a student's network and schedulers
    // from the export, the rest of the pipeline from the base it continues.
    let held: Vec<Stage> = match &networks {
        Networks::Released(stages) => stages.clone(),
        Networks::Student(contract) => vec![contract.network],
    };
    let mut components = PIPELINE.to_vec();
    for stage in held {
        components.push(Component::holding(stage)?);
    }

    let mut stages = BTreeMap::new();
    let mut directories = BTreeMap::new();
    for component in components {
        let name = component.name();
        let source = match base {
            Some(base) if component.network().is_none() && !STUDENT.contains(&component) => {
                // An export that also holds what its contract takes from the base would be
                // served half from files it does not name.
                if index.place(name, checkpoint)?.directory().exists() {
                    return Err(ID.checkpoint(format!(
                        "{} holds `{name}`, which its {} takes from the base release it \
                         continues",
                        checkpoint.directory().display(),
                        Contract::FILE
                    )));
                }
                base
            }
            _ => checkpoint,
        };

        if component.weights() {
            stages.insert(name.to_owned(), index.stage(name, source)?);
        } else {
            directories.insert(name.to_owned(), index.directory(name, source)?);
        }
    }

    // An adapter edits the network its contract names, and the fusion checks every edit against
    // the network's own tensors.
    if let Some((contract, adapter)) = adapter {
        let name = Component::holding(contract.network)?.name();
        let network = stages.remove(name).ok_or_else(|| {
            ID.checkpoint(format!("the adapter edits `{name}`, which is not composed"))
        })?;
        stages.insert(name.to_owned(), network.attach(adapter)?);
    }

    let composed = Checkpoint::compose(checkpoint.clone(), stages, directories)?;
    Ok((composed, networks))
}

/// The pipeline index of the checkpoint at `source`, which declares this family's pipeline.
fn pipeline_index(source: &Source) -> Result<PipelineIndex> {
    let index = PipelineIndex::read(source, FAMILY.request.root)?;
    if index.class() != PIPELINE_CLASS {
        return Err(ID.checkpoint(format!(
            "{} declares a {} pipeline, and this family serves {PIPELINE_CLASS}",
            source.directory().display(),
            index.class()
        )));
    }
    Ok(index)
}

/// The adapter the directory `directory` holds and the contract of the student it is.
fn open_adapter(directory: &Source) -> Result<(Contract, Adapter)> {
    let contract = Contract::read(directory)?.ok_or_else(|| {
        ID.checkpoint(format!(
            "the adapter directory {} holds no {}: an adapter states the student it is (its \
             task, schedule and attention) in the contract its release publishes",
            directory.directory().display(),
            Contract::FILE
        ))
    })?;
    Ok((contract, Adapter::open(directory, &FastVideoLora)?))
}

/// Checks that `source` is a snapshot of `revision`, when its caller resolved it as one.
///
/// A local directory states no commit, so the deployment that names it vouches for it.
fn continues(revision: &BaseRevision, source: &Source) -> Result<()> {
    let Source::Snapshot(snapshot) = source else {
        return Ok(());
    };
    if snapshot.repository() != revision.repository || snapshot.revision() != revision.revision {
        return Err(ID.checkpoint(format!(
            "the student continues {revision}, and the deployment's snapshot is {}@{}",
            snapshot.repository(),
            snapshot.revision()
        )));
    }
    Ok(())
}

impl VideoCheckpoint for Config {
    fn pipeline(&self) -> Pipeline {
        pipeline()
    }

    /// The most rows any task a rank placing `stages` serves reserves ([`condition_capacity`]).
    fn condition_rows(&self, stages: Option<&BTreeSet<String>>) -> nn::Result<u32> {
        let tasks = served_tasks(&self.tasks(), stages)?;

        Ok(tasks
            .iter()
            .map(|task| condition_capacity(self, *task))
            .collect::<Result<Vec<u32>>>()?
            .into_iter()
            .max()
            .unwrap_or(0))
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
        let tasks = served_tasks(&config.tasks(), placement.stages.as_ref())?;
        let stages = stages(config, &tasks, placement.stages.as_ref())?;
        let split = placement.denoiser;

        // The tensor split and the Ulysses degree divide the attention heads between their
        // ranks; an all-gather degree splits rows alone.
        for (stage, denoiser) in &config.denoisers {
            let heads = denoiser.transformer.num_attention_heads;
            let shares = split.ulysses.split.size() * split.tensor.size();
            if shares == 0 || !heads.is_multiple_of(shares) {
                return Err(ID.config(format!(
                    "a Ulysses degree of {} in a tensor split of {} does not divide the {heads} \
                     attention heads of {}",
                    split.ulysses.split.size(),
                    split.tensor.size(),
                    stage.name()
                )));
            }
        }

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
                ID.checkpoint(
                    "the checkpoint states no processor/video_preprocessor_config.json, so the \
                     text encoder's vision tower cannot size an image"
                        .to_owned(),
                )
            })?;
            let mut stage = loader.reborrow(checkpoint.stage(Component::TextEncoder.name())?)?;
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
                ID.checkpoint(format!("the checkpoint holds no {} network", stage.name()))
            })?;

            // The two condition encoders are the denoiser's conditioning half.
            let encoders = if stages.contains(Stage::LatentEncoder) {
                let mut video = loader.reborrow(checkpoint.stage(Component::Vae.name())?)?;
                video.declare(&config.video_vae.quantization)?;
                let audio = loader.reborrow(checkpoint.stage(Component::AudioVae.name())?)?;

                Some((
                    VideoEncoder::load(&config.video_vae, video.root())?,
                    AudioEncoder::load(&config.audio_vae, audio.root())?,
                ))
            } else {
                None
            };

            let mut network =
                loader.reborrow(checkpoint.stage(Component::holding(stage)?.name())?)?;
            network.declare(&denoiser.quantization)?;

            model.denoiser = Some((
                stage,
                Arc::new(Denoiser::load(
                    denoiser,
                    placement.capacity,
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
            let stage = loader.reborrow(checkpoint.stage(Component::AudioVae.name())?)?;

            model.audio_decoder = Some(Arc::new(AudioDecoder::load(
                &config.audio_vae,
                stage.root(),
            )?));
        }
        if stages.contains(Stage::VideoDecoder) {
            let mut stage = loader.reborrow(checkpoint.stage(Component::Vae.name())?)?;
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

    /// The placed denoiser network, which `need` asks of the rank.
    fn placed(&self, need: &str) -> Result<&Arc<Denoiser>> {
        self.network()
            .ok_or_else(|| ID.config(format!("this rank places no denoiser, so {need}")))
    }

    /// The processor a keyframe is prepared and a vision block sized by.
    fn processor(&self) -> Result<&processing::qwen_vl::ProcessorConfig> {
        self.config.processor.as_ref().ok_or_else(|| {
            ID.checkpoint(
                "the checkpoint states no processor/video_preprocessor_config.json, so a \
                 keyframe cannot be prepared nor a vision block sized"
                    .to_owned(),
            )
        })
    }

    /// How the placed denoiser lays a request's conditions into its rows, as request
    /// processing sizes them: whole tiles under sparse attention, row for row under dense
    /// attention, and the rows it holds when this rank places the condition encoders.
    fn condition_packing(&self, denoiser: &Denoiser) -> Result<ConditionPacking> {
        let capacity = if denoiser.places_encoders() {
            denoiser.condition_capacity()
        } else {
            0
        };

        match denoiser.config().vsa {
            Some(vsa) => ConditionPacking::tiled(capacity, vsa.tile_rows, vsa.tile_shape())
                .ok_or_else(|| ID.config("a sparse attention tile is empty".to_owned())),
            None => Ok(ConditionPacking::dense(capacity)),
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
            audio: AudioFormat {
                channels: AUDIO_CHANNELS,
                hop: u64::from(AUDIO_HOP),
            },
        })
    }

    fn timeline(&self, seconds: f64) -> nn::Result<u32> {
        timeline(seconds)
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
        let cell = processing::minimax_h3::PATCH_PIXELS;
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

    fn canvases(&self) -> nn::Result<Vec<(u32, u32)>> {
        Ok(resolutions()?
            .into_iter()
            .map(|served| (served.canvas.width, served.canvas.height))
            .collect())
    }

    /// Every frame count of the `17n + 5` rule from the shortest duration request processing
    /// admits ([`MIN_SECONDS`]) to the least covering `seconds`.
    fn timelines(&self, seconds: f64) -> nn::Result<Vec<u32>> {
        let longest = timeline(seconds)?;
        let shortest = timeline(MIN_SECONDS)?;

        Ok((shortest..=longest)
            .step_by(FRAMES.clip_frames as usize)
            .collect())
    }

    /// `t2va` carries no condition; `fl2va` anchors the first frame, the last or both, each a
    /// keyframe at the canvas presented in the prompt; `ref2va` carries references of the
    /// client's media, each image and video presented in the prompt, up to the declared limits
    /// ([`REFERENCE_LIMITS`]).
    fn requests(&self) -> Vec<RequestKind> {
        self.tasks
            .iter()
            .map(|task| match task {
                VideoTask::TextToVideoAudio => RequestKind {
                    roles: Vec::new(),
                    media: ConditionMedia::None,
                    vision_blocks: 0,
                },
                VideoTask::FirstLastToVideoAudio => RequestKind {
                    roles: vec![ConditionRole::FirstFrame, ConditionRole::LastFrame],
                    media: ConditionMedia::Canvas,
                    vision_blocks: 2,
                },
                VideoTask::ReferenceToVideoAudio => RequestKind {
                    roles: vec![ConditionRole::Reference],
                    media: ConditionMedia::Client,
                    vision_blocks: (REFERENCE_LIMITS.images + REFERENCE_LIMITS.videos)
                        .clamp(1, REFERENCE_LIMITS.total.max(1)),
                },
            })
            .collect()
    }

    fn capacity_unit(&self) -> nn::Result<u32> {
        let denoiser = self.placed("no layout is sized")?;
        Ok(denoiser.config().capacity_unit())
    }

    /// A keyframe is prepared as request processing prepares it, fitted to the canvas, and
    /// takes the tiles its patches pack into; a reference takes the tiles of its own extents
    /// ([`reference_rows`]).
    fn condition_rows(
        &self,
        conditions: &[ConditionShape],
        (width, height): (u32, u32),
    ) -> nn::Result<u32> {
        let denoiser = self.placed("no condition is laid out")?;
        let packing = self.condition_packing(denoiser)?;
        let canvas = Canvas { width, height };

        let mut rows: u32 = 0;
        for (order, condition) in conditions.iter().enumerate() {
            let taken = if condition.role == ConditionRole::Reference {
                reference_rows(condition, &packing)
            } else {
                let processor = self.processor()?;
                let image = processing::ImageMetadata {
                    width: condition.width,
                    height: condition.height,
                };
                let prepared = PreparedCondition::keyframe(
                    order,
                    condition.role,
                    &image,
                    canvas,
                    order,
                    true,
                    processor,
                )
                .map_err(|error| ID.config(format!("a keyframe of {width}x{height}: {error}")))?;

                processing::minimax_h3::condition_rows(&[prepared], canvas, &packing)
            };
            rows = rows.saturating_add(taken);
        }
        Ok(rows)
    }

    fn largest_conditions(
        &self,
        canvas: (u32, u32),
        frames: u32,
        rows: u32,
    ) -> nn::Result<Vec<ConditionShape>> {
        let denoiser = self.placed("no condition is laid out")?;
        if !self.tasks.contains(&VideoTask::ReferenceToVideoAudio) {
            return Ok(Vec::new());
        }
        Ok(largest_references(
            canvas,
            frames,
            rows,
            &self.condition_packing(denoiser)?,
        ))
    }

    /// A placeholder is the tower's image token, and the processor merges its patches in
    /// squares of its merge side.
    fn vision_tokens(&self) -> nn::Result<VisionTokens> {
        let processor = self.processor()?;
        let placeholder = self
            .config
            .text_encoder
            .vision
            .as_ref()
            .map(|vision| vision.image_token_id)
            .ok_or_else(|| {
                ID.checkpoint(
                    "the text encoder states no vision tower, so no prompt presents a vision \
                     block"
                        .to_owned(),
                )
            })?;

        Ok(VisionTokens {
            placeholder,
            merge: processor.merge_size.max(1),
        })
    }

    fn vision_grid(&self, width: u32, height: u32) -> nn::Result<[u32; 3]> {
        let grid = self
            .processor()?
            .image_grid(height, width)
            .map_err(|error| ID.config(format!("a keyframe of {width}x{height}: {error}")))?;
        Ok([grid.t, grid.h, grid.w])
    }

    /// What the placed denoiser serves at `sizes`: the tasks of the placed network, its fixed
    /// schedule and shifts, the deployment's resolutions, the reference limits and the condition
    /// packing of the capacity it was loaded for.
    fn description(&self, sizes: &[(u32, u32)]) -> nn::Result<Description> {
        let denoiser = self.placed("it serves no task")?;
        let tasks: Vec<VideoTask> = VideoTask::ALL
            .into_iter()
            .filter(|task| self.tasks.contains(task))
            .collect();
        let schedule = &denoiser.config().schedule;
        let steps = schedule.num_steps();

        let info = DenoiserInfo {
            tasks: tasks.clone(),
            schedule_points: steps + 1,
            video_shift: f64::from(schedule.video_shift),
            audio_shift: f64::from(schedule.audio_shift),
            canvases: served_resolutions(sizes)?,
            references: if tasks.contains(&VideoTask::ReferenceToVideoAudio) {
                REFERENCE_LIMITS
            } else {
                ReferenceLimits::default()
            },
            condition_packing: self.condition_packing(denoiser)?,
        };
        info.validate()
            .map_err(|reason| ID.config(format!("the denoiser's description: {reason}")))?;

        Ok(Description {
            takes_conditions: tasks.iter().any(|task| task.conditioned()),
            denoise_steps: steps,
            details: info.encode(),
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
        return Err(ID.config("a deployment serves at least one task".to_owned()));
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
            return Err(ID.config(format!(
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
        return Err(ID.config(
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
fn served_tasks(stated: &[VideoTask], stages: Option<&BTreeSet<String>>) -> Result<Vec<VideoTask>> {
    if stated.is_empty() {
        return Err(ID.config(
            "the checkpoint states no task, so no network can be sized for one".to_owned(),
        ));
    }

    let Some(stages) = stages else {
        let first = network(stated[0]);
        if stated.iter().all(|task| network(*task) == first) {
            return Ok(stated.to_vec());
        }

        return Err(ID.config(
            "the checkpoint holds two networks and this deployment places the whole pipeline: \
             a deployment file that places one of them chooses it"
                .to_owned(),
        ));
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
        return Err(ID.config(format!(
            "the checkpoint states {stated:?} and the deployment places no network of them"
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
    let denoiser = config
        .denoisers
        .get(&stage)
        .ok_or_else(|| ID.config(format!("the checkpoint holds no {} network", stage.name())))?;

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
        return Err(ID.config(format!(
            "{seconds} seconds is not a positive number of frames at {FPS} frames per second"
        )));
    }
    Ok(FRAMES.align_up(frames))
}

/// The resolutions every H3 checkpoint generates: the resolution rule's of each named aspect
/// ratio.
fn resolutions() -> Result<Vec<ServedCanvas>> {
    named_canvases().map_err(|error| ID.config(format!("the resolution rule: {error}")))
}

/// The resolutions of the sizes a deployment serves, each with the aspect ratio that names it,
/// in the deployment's order.
fn served_resolutions(sizes: &[(u32, u32)]) -> Result<Vec<ServedCanvas>> {
    if sizes.is_empty() {
        return Err(
            ID.config("this deployment serves no size, so no raster could be decoded".to_owned())
        );
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
                    ID.config(format!(
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

/// Denoiser rows one reference `shape` takes in a layout of `packing`: its track's rows in whole
/// tiles, then an image's patch rows in whole tiles, or the video tiles that cover a video's
/// `(latent frames, patch rows, patch columns)` token grid, each tile `packing.tile_rows` rows.
fn reference_rows(shape: &ConditionShape, packing: &ConditionPacking) -> u32 {
    // A condition row is one latent patch of the raster's cells, which the packing tiles.
    let cell = processing::minimax_h3::PATCH_PIXELS;
    let (audio, visual) = packing::condition_rows(shape);
    let grid = [
        FRAMES.latent_frames(shape.frames),
        shape.height / cell,
        shape.width / cell,
    ];
    let visual = match (shape.frames, visual) {
        (_, 0) => VisualRows::None,
        (1, rows) => VisualRows::Patches(rows as u32),
        _ => VisualRows::Grid(grid),
    };

    condition_tiles(audio as u32, visual, packing).saturating_mul(packing.tile_rows.get())
}

/// The largest references a layout of `rows` condition rows holds, for a request at `canvas`
/// lasting at most `longest` frames: the largest image an admitted request presents that the
/// rows hold, the longest and widest video with its soundtrack they hold, and the longest track
/// alone they hold, in that order, each present when the rows hold any of its form.
///
/// A reference's raster and length follow the client's media, so these bound what a request's
/// references are and are none of a request's own: an opening of the layout encodes its
/// references one after another, and none is larger in pixels, in frames or in samples than one
/// of these. An image reference is encoded at a short edge of at most the declared edge
/// ([`REFERENCE_LIMITS`]`.image_short_edge`) within the canvas rule's aspects, so the largest
/// is that edge by four times it, fewer cells where the rows hold fewer, as square as they
/// divide. A video reference is put on the canvas of its aspect and lasts at most the generated
/// video: the longest lasts `longest` frames at `canvas`, its frames and then its sides halved
/// until its rows fit.
fn largest_references(
    (width, height): (u32, u32),
    longest: u32,
    rows: u32,
    packing: &ConditionPacking,
) -> Vec<ConditionShape> {
    let cell = processing::minimax_h3::PATCH_PIXELS;
    let reference = |frames: u32, height: u32, width: u32, audio_samples: u32| ConditionShape {
        role: ConditionRole::Reference,
        frames,
        height,
        width,
        audio_samples,
    };
    let samples = |frames: u32| frames * AUDIO_SAMPLE_RATE / FPS;
    let holds = |shape: &ConditionShape| reference_rows(shape, packing) <= rows;

    let mut largest = Vec::new();

    // The image of the most cells the rows hold within the declared edge and the widest aspect,
    // as square as they divide.
    let tile = packing.tile_rows.get();
    let cells = rows / tile * tile;
    let (narrow, wide) = (CANVAS_RULE.widest.1.max(1), CANVAS_RULE.widest.0);
    let short = (REFERENCE_LIMITS.image_short_edge / cell).max(1);
    let side = (f64::from(cells).sqrt() as u32).clamp(1, short);
    let long = (cells / side).min(side * wide / narrow);
    if long > 0 {
        largest.push(reference(1, side * cell, long * cell, 0));
    }

    // The longest video at the widest raster the rows hold, with its soundtrack.
    let (mut height, mut width) = (height, width);
    'video: while height >= cell && width >= cell {
        let mut frames = FRAMES.align_down(longest);
        while let Some(count) = frames {
            let video = reference(count, height, width, samples(count));
            if holds(&video) {
                largest.push(video);
                break 'video;
            }
            frames = count
                .checked_sub(FRAMES.clip_frames)
                .filter(|shorter| FRAMES.is_legal(*shorter));
        }
        height = (height / 2 / cell) * cell;
        width = (width / 2 / cell) * cell;
    }

    // The longest track alone the rows hold.
    let mut track = samples(longest);
    while track > 0 {
        let audio = reference(0, 0, 0, track);
        if holds(&audio) {
            largest.push(audio);
            break;
        }
        track /= 2;
    }
    largest
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let tasks = [
            VideoTask::TextToVideoAudio,
            VideoTask::FirstLastToVideoAudio,
        ];
        assert_eq!(served_tasks(&tasks, None).unwrap(), tasks);

        let none = served_tasks(&[], None).unwrap_err();
        assert!(none.to_string().contains("states no task"), "{none}");

        let both = [
            VideoTask::TextToVideoAudio,
            VideoTask::ReferenceToVideoAudio,
        ];
        assert!(
            served_tasks(&both, None).is_err(),
            "two networks need a deployment file to choose between them"
        );
    }

    /// A deployment that places one network serves that network's tasks, and one that places
    /// none of the checkpoint's networks is refused.
    #[test]
    fn a_placed_network_chooses_the_tasks() {
        let both = [
            VideoTask::TextToVideoAudio,
            VideoTask::ReferenceToVideoAudio,
        ];
        let placed = |names: &[&str]| -> BTreeSet<String> {
            names.iter().map(|name| (*name).to_owned()).collect()
        };

        assert_eq!(
            served_tasks(&both, Some(&placed(&["transformer_ref", "text_encoder"])),).unwrap(),
            [VideoTask::ReferenceToVideoAudio]
        );

        // A group that places no network serves beside the one the deployment places.
        assert_eq!(
            served_tasks(&both, Some(&placed(&["text_encoder"]))).unwrap(),
            both
        );

        // A network the checkpoint does not hold serves no task.
        let base = [VideoTask::TextToVideoAudio];
        assert!(served_tasks(&base, Some(&placed(&["transformer_ref"]))).is_err());
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

    /// The largest references a rung of OmniRef's tiling holds: an image, a video with its
    /// soundtrack and a track alone, none taking more rows than the rung, whatever the client's
    /// media are. The widest rung holds the largest image a request presents (the declared short
    /// edge by four times it) and the longest video at the canvas; a rung too small for a video
    /// at the canvas holds one of halved sides.
    #[test]
    fn a_rung_holds_the_largest_reference_of_each_form() {
        // OmniRef's tiling: 128-row tiles, a video's token grid in tiles of 4 x 4 x 8 tokens.
        let packing = ConditionPacking::tiled(49_152, 128, [4, 4, 8]).unwrap();
        // A four-second request at 1344x768: 107 frames.
        let largest = |rows| largest_references((1344, 768), 107, rows, &packing);

        for rows in [1024, 49_152] {
            let references = largest(rows);
            let forms: Vec<(bool, bool)> = references
                .iter()
                .map(|reference| (reference.frames > 0, reference.audio_samples > 0))
                .collect();
            assert_eq!(
                forms,
                [(true, false), (true, true), (false, true)],
                "{rows}"
            );

            for reference in &references {
                assert_eq!(reference.role, ConditionRole::Reference);
                assert!(
                    reference_rows(reference, &packing) <= rows,
                    "{reference:?} in {rows} rows"
                );
            }
        }

        let widest = largest(49_152);
        assert_eq!((widest[0].height, widest[0].width), (2048, 8192));
        assert_eq!(
            (widest[1].frames, widest[1].width, widest[1].height),
            (107, 1344, 768)
        );
        assert!(largest(1024)[1].width < 1344);
    }
}

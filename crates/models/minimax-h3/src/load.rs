//! The loading boundary of a deployment: the blocks one rank places, built from the checkpoint.
//!
//! An H3 checkpoint is a diffusers pipeline directory whose components are its subdirectories, and
//! `nn::Checkpoint` learns such a root: `checkpoint.component("transformer")` gives one
//! component's shard set with each shard's own identity, and `WeightLoader::reborrow` gives one
//! loader per component over the same store and precision policy (specs/worker.md 8). The blocks a
//! rank holds are the [`Partition`]'s, so this module only builds them, and registers each under the
//! name its runner addresses: the capability's name, not the checkpoint directory's.
//!
//! ## What the caller gets back
//!
//! [`MinimaxH3`]: the blocks the executor serves, and the same components behind them as handles.
//! A block is boxed as `Arc<dyn Encoder>` or `Arc<dyn Decoder>` and those traits publish no
//! descriptors, so a deployment that can answer "which programs must be compiled for me?" can only
//! answer it from the concrete components - and that is what the handles are for. They are the same
//! values the blocks hold, so the weights exist once.
//!
//! ## Testing
//!
//! A load publishes every tensor through `nn::weights::RetainedWeights`, which only the executor
//! implements, so this module has no unit test of its own: the block constructors are tested where
//! they live, and the composition is exercised by the caller's load of a real checkpoint. What can
//! be checked here without a device is checked - `Partition`'s rules, the packing's tables, and the
//! component names and directories.
//!
//! ## Precision
//!
//! With the design's `dtype = "auto"` nothing is converted at load: each weight keeps the
//! representation the checkpoint stores it in - BF16 for the DiT partitions and the text encoder,
//! FP16 for the video VAE decoder - and each layer declares the precision it computes in
//! (specs/worker.md 1, 9).

use nn::{Block, BlockName};
use std::sync::Arc;

use crate::audio_vae::AudioDecoder;
use crate::checkpoint::{Dit, Layout, Sidecars};
use crate::config::Config;
use crate::denoiser::{ConditionEncoders, Denoiser, Layouts};
use crate::schedule::Schedule;
use crate::text_encoder::TextEncoder;
use crate::transformer::StudentHeads;
use crate::{Error, Partition, Result};

/// The block type is the partition's; re-exported here because a block and its loading are read
/// together, and the layers beside them name their block this way.
pub use crate::partition::Component;

/// A loaded deployment: the blocks this rank serves, and the components they were built from.
///
/// The runner needs the first and only the first: [`MinimaxH3::into_model`] hands it over and the
/// executor serves each block by the capability it implements. The second is what makes the
/// deployment able to answer for itself: a block is boxed as `Arc<dyn Encoder>` and the trait does
/// not publish descriptors, so the model's programs can only be enumerated from the components
/// before they are erased - `keys` walks these handles, and nothing else in the crate reads them.
///
/// Every handle is the *same* value the block holds, not a copy: the component is constructed once,
/// shared by the model and by the walk, and the weights exist once.
pub struct MinimaxH3 {
    model: nn::Model,
    text_encoder: Option<Arc<TextEncoder>>,
    transformer: Option<Arc<Denoiser>>,
    transformer_ref: Option<Arc<Denoiser>>,
    video_postprocessor: Option<Arc<crate::output::VideoPostprocessor>>,
    video_decoder: Option<Arc<crate::video_vae::VideoDecoder>>,
    audio_decoder: Option<Arc<AudioDecoder>>,
}

impl std::fmt::Debug for MinimaxH3 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MinimaxH3")
            .field("model", &self.model)
            .field(
                "text_encoder",
                &self.text_encoder.as_ref().map(|_| "TextEncoder"),
            )
            .field(
                "transformer",
                &self.transformer.as_ref().map(|_| "Denoiser"),
            )
            .field(
                "transformer_ref",
                &self.transformer_ref.as_ref().map(|_| "Denoiser"),
            )
            .field(
                "video_postprocessor",
                &self
                    .video_postprocessor
                    .as_ref()
                    .map(|_| "VideoPostprocessor"),
            )
            .field(
                "video_decoder",
                &self.video_decoder.as_ref().map(|_| "VideoDecoder"),
            )
            .field(
                "audio_decoder",
                &self.audio_decoder.as_ref().map(|_| "AudioDecoder"),
            )
            .finish()
    }
}

impl MinimaxH3 {
    /// The blocks this rank serves, as the executor consumes them.
    pub fn model(&self) -> &nn::Model {
        &self.model
    }

    /// The blocks, consuming the deployment: what a runner is handed once nothing else needs to
    /// enumerate the deployment's programs.
    pub fn into_model(self) -> nn::Model {
        self.model
    }

    /// The text encoder, when this rank places it.
    pub fn text_encoder(&self) -> Option<&Arc<TextEncoder>> {
        self.text_encoder.as_ref()
    }

    /// The base DiT partition, when this rank places it.
    pub fn transformer(&self) -> Option<&Arc<Denoiser>> {
        self.transformer.as_ref()
    }

    /// The reference DiT partition, when this rank places it.
    ///
    /// A deployment places at most one of the two (`Partition::validate` refuses both), so exactly
    /// one of this and [`MinimaxH3::transformer`] is `Some` when a denoiser is placed at all.
    pub fn transformer_ref(&self) -> Option<&Arc<Denoiser>> {
        self.transformer_ref.as_ref()
    }

    /// The weightless post-processing block, when this rank places it.
    pub fn video_postprocessor(&self) -> Option<&Arc<crate::output::VideoPostprocessor>> {
        self.video_postprocessor.as_ref()
    }

    /// The video decoder, when this rank places it.
    pub fn video_decoder(&self) -> Option<&Arc<crate::video_vae::VideoDecoder>> {
        self.video_decoder.as_ref()
    }

    /// The audio decoder, when this rank places it.
    pub fn audio_decoder(&self) -> Option<&Arc<AudioDecoder>> {
        self.audio_decoder.as_ref()
    }
}

/// Refuse a component whose names the export lacks, before any shard is opened.
///
/// A block that composes its names with `format!` inside its layers reports a missing tensor as a
/// read error from inside a layer, naming no block - the class that cost the first device run a
/// round (`transformer_blocks.0.ff.net.0.proj.weight.weight`). The block's `keys` is the same list
/// as data, so this answers it against the index first and names the block, which half of it is
/// short, and the first few absent tensors.
///
/// The list is rank-independent: a degree splits a tensor's rows and never renames it.
fn require(
    source: &nn::Checkpoint,
    keys: &[String],
    block: &str,
    what: &str,
    directory: &str,
) -> Result<()> {
    let missing = crate::checkpoint::absent(keys, |key| source.tensor(key).is_some());
    if missing.is_empty() {
        return Ok(());
    }
    Err(Error::checkpoint(format!(
        "the {block} block asks {} tensors for its {what} and {} of them are not in \
         {directory}/: {}",
        keys.len(),
        missing.len(),
        missing[..missing.len().min(8)].join(", ")
    )))
}

/// Load the deployment this rank places.
///
/// The place and the blocks are the `partition`'s: it names the degrees and the rank's position in
/// them, the blocks this rank holds, and the row capacities those blocks are sized for. The
/// checkpoint is one file set per pipeline: an export that continues a base release is opened by
/// the engine as one checkpoint spanning both, so this function never learns about bases.
///
/// Nothing is converted at load unless the checkpoint says so: with the design's `dtype = "auto"`
/// each block keeps the representation its weights are stored in (BF16 DiT and text encoder, FP16
/// video VAE decoder), and each layer declares the precision it computes in (specs/worker.md 1).
///
/// A block this build cannot construct is refused with [`Error::Unsupported`], naming it, rather
/// than omitted: a deployment that places one must fail at load, not on the device.
pub fn load(
    checkpoint: &nn::Checkpoint,
    partition: Partition,
    loader: &mut nn::weights::WeightLoader<'_>,
) -> Result<MinimaxH3> {
    let (layout, sidecars) = sidecars(checkpoint)?;
    let config = Config::normalize(&layout, &sidecars)?;
    partition.validate(&config)?;

    // The two condition encoders are the denoiser block's half, and they are loaded before the
    // loop because a block is boxed as it is built: the partition places `transformer` before
    // `latent_encoder`, so by the time the encoders' own component is reached the denoiser is a
    // `Block::Denoiser` and the concrete block that holds them is out of reach. A deployment whose
    // tasks carry no conditions places none.
    let mut encoders = if partition.blocks.contains(Component::LatentEncoder) {
        // The encoders' names are data too (`keys` is the other column of the table each one's
        // load reads), so a name the export lacks is refused here rather than surfacing as a read
        // error from inside a stack of strides and convolutions. The audio encoder is checked in
        // the same pass and refuses just as plainly: `ref2va` reads both, and a checkpoint that
        // holds no audio weights should say so before any stage of it is built.
        let video = crate::encoding::VideoEncoder::keys(&config.video_vae)?;
        require(
            checkpoint.component("vae")?,
            &video,
            "latent_encoder",
            "video encoder",
            "vae",
        )?;
        let audio = crate::encoding::AudioEncoder::keys(&config.audio_vae)?;
        require(
            checkpoint.component("audio_vae")?,
            &audio,
            "latent_encoder",
            "audio encoder",
            "audio_vae",
        )?;
        Some(ConditionEncoders::load(&config, checkpoint, loader)?)
    } else {
        None
    };

    let mut blocks = Vec::new();
    let mut text_encoder: Option<Arc<TextEncoder>> = None;
    let mut transformer: Option<Arc<Denoiser>> = None;
    let mut transformer_ref: Option<Arc<Denoiser>> = None;
    let mut video_postprocessor: Option<Arc<crate::output::VideoPostprocessor>> = None;
    let mut video_decoder: Option<Arc<crate::video_vae::VideoDecoder>> = None;
    let mut audio_decoder: Option<Arc<AudioDecoder>> = None;
    for component in partition.blocks.iter() {
        let block = match component {
            Component::TextEncoder => {
                let source = checkpoint.component("text_encoder")?;
                // The encoder composes its names with `format!` inside its layers, so a name the
                // export lacks used to surface as a read error from inside a layer, naming no
                // block. `keys` is the same list as data - the language stack and the vision tower
                // are separate lists because they are separate components of one block, and the
                // report says which of the two is short. Both live in one index, so one source
                // answers both, and no degree renames a tensor: the degree splits rows.
                let encoder_keys = TextEncoder::keys(&config.text_encoder);
                let tower_keys = crate::vision::VisionTower::keys(&config.text_encoder.vision);
                require(
                    source,
                    &encoder_keys,
                    component.name(),
                    "language stack",
                    "text_encoder",
                )?;
                require(
                    source,
                    &tower_keys,
                    component.name(),
                    "vision tower",
                    "text_encoder",
                )?;
                let dtype = component.activation();
                let mut shards = loader.reborrow(source)?;
                let encoder = Arc::new(TextEncoder::load(
                    &mut shards,
                    &config.text_encoder,
                    &partition,
                    dtype,
                )?);
                text_encoder = Some(Arc::clone(&encoder));
                Block::Encoder(encoder)
            }
            Component::Transformer | Component::TransformerRef => {
                let dit = match component {
                    Component::Transformer => Dit::Base,
                    _ => Dit::Reference,
                };
                let denoiser = config.denoisers.get(&dit).ok_or_else(|| {
                    Error::checkpoint(format!(
                        "the checkpoint states no {} DiT, so the {} block cannot be placed (it \
                         states {})",
                        dit.directory(),
                        component.name(),
                        config
                            .denoisers
                            .keys()
                            .map(|dit| dit.directory())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?;
                let arrangement = partition.arrangement(denoiser)?;
                // The attention's scratch is allocated for the largest layout this deployment can
                // reach, over every canvas the DiT offers and every legal frame count.
                let rows = Layouts::new(denoiser.clone(), arrangement)?.rows_bound()?;
                let source = checkpoint.component(dit.directory())?;
                // Every tensor this partition's denoiser reads, asked of the block itself: `load`
                // states no name of its own. A name the checkpoint lacks is refused here rather
                // than surfacing as a read error deep in `Denoiser::load`, so the report names the
                // block, its directory and every absent tensor at once. The list is rank-independent
                // because the DiT shards the sequence and never the weights - `Denoiser::load` takes
                // the partition for its exchange rank, not for a weight slice - so one rank's check
                // answers for every rank of the deployment.
                let keys = Denoiser::keys(denoiser);
                require(source, &keys, component.name(), "denoiser", dit.directory())?;
                let mut shards = loader.reborrow(source)?;
                // A parallel-decoding student stacks one head per fine interval; the boundary that
                // owns the checkpoint fuses them into one head per evaluation block and publishes
                // them through the same loader, so the model reads fused tables and nothing else.
                let schedule = Schedule::new(&denoiser.schedule)?;
                let heads = if denoiser.transformer.output_heads > 1 {
                    Some(StudentHeads::fuse(
                        source,
                        &schedule,
                        &denoiser.transformer,
                        &mut shards,
                    )?)
                } else {
                    None
                };
                let mut denoiser = Denoiser::load(
                    denoiser.clone(),
                    arrangement,
                    rows,
                    &partition,
                    &mut shards,
                    heads,
                )?;
                // The encoders are this block's conditioning half. Only one DiT partition can be
                // placed on a rank (`Partition::validate` refuses both), so at most one arm takes
                // them; `attach_encoders` refuses a second attach as well.
                if let Some(encoders) = encoders.take() {
                    denoiser.attach_encoders(encoders)?;
                }
                let denoiser = Arc::new(denoiser);
                match dit {
                    Dit::Base => transformer = Some(Arc::clone(&denoiser)),
                    Dit::Reference => transformer_ref = Some(Arc::clone(&denoiser)),
                }
                Block::Denoiser(denoiser)
            }
            Component::VideoPostprocessor => {
                // Weightless: the splice's only constants are the reference's ImageNet means and standard
                // deviations, uploaded once as the two `[3]` weights the byte step reads.
                let dtype = Component::VideoDecoder.activation();
                let bytes = |values: [f32; 3]| -> Vec<u8> {
                    values
                        .iter()
                        .flat_map(|value| value.to_ne_bytes())
                        .collect()
                };
                let shape = nn::Shape::new(&[3]);
                let mean = loader.constant(
                    "video_postprocessor.pixel_mean",
                    nn::DType::F32,
                    shape,
                    &bytes(crate::output::PIXEL_MEAN),
                )?;
                let std = loader.constant(
                    "video_postprocessor.pixel_std",
                    nn::DType::F32,
                    shape,
                    &bytes(crate::output::PIXEL_STD),
                )?;
                let postprocessor =
                    Arc::new(crate::output::VideoPostprocessor::new(dtype, mean, std)?);
                video_postprocessor = Some(Arc::clone(&postprocessor));
                nn::Block::Decoder(postprocessor)
            }
            Component::AudioDecoder => {
                // The audio decoder has no layout tables - every table it reads is a weight bound
                // at load - so it constructs from its own component and its sidecar alone.
                let source = checkpoint.component("audio_vae")?;
                require(
                    source,
                    &AudioDecoder::keys(&config.audio_vae)?,
                    component.name(),
                    "decoder",
                    "audio_vae",
                )?;
                let mut shards = loader.reborrow(source)?;
                let decoder = Arc::new(AudioDecoder::load(&config.audio_vae, &mut shards)?);
                audio_decoder = Some(Arc::clone(&decoder));
                Block::Decoder(decoder)
            }
            Component::VideoDecoder => {
                require(
                    checkpoint.component("vae")?,
                    &crate::video_vae::VideoDecoder::keys(&config.video_vae)?,
                    component.name(),
                    "decoder",
                    "vae",
                )?;
                // One block serves every canvas: the tile grid is arithmetic on a request's raster,
                // so the block is loaded against the checkpoint's tile - the largest tile any canvas
                // of this checkpoint produces, since the grid is uniform wherever the canvas holds a
                // whole tile - and the seam blends are built here, at load.
                let spatial = config.video_vae.spatial_compression();
                let canvas = crate::video_vae::Canvas::clip(
                    &config.video_vae,
                    crate::video_vae::TILE_PIXELS / spatial,
                    crate::video_vae::TILE_PIXELS / spatial,
                );
                // The clip's rows are packed in the attention's order, so the decoder needs it at
                // load. It is the denoisers' and it must be one: two partitions that attend
                // differently would pack one clip two ways, and one table can only gather one.
                let mut kinds = config.denoisers.values().map(|dit| dit.attention);
                let attention = kinds.next().ok_or_else(|| {
                    Error::checkpoint(
                        "the checkpoint states no DiT, so no attention kind packs the video \
                         decoder's clip rows"
                            .to_owned(),
                    )
                })?;
                if kinds.any(|kind| kind != attention) {
                    return Err(Error::Unsupported(format!(
                        "the {} block gathers a clip's rows in the denoisers' attention order, and \
                         this checkpoint's DiT partitions attend differently",
                        component.name()
                    )));
                }
                // The decoder's three reference steps before its transformer - denormalise,
                // post-quantise, project - are one affine, composed here from the checkpoint's own
                // bytes. Its element type is the type those weights are stored in: `dtype = "auto"`
                // converts nothing, and the reference reads the projection's type for the same reason.
                let vae = checkpoint.component(component.directories()[0])?;
                let (width, dtype, weight, bias) =
                    crate::video_vae::folded_input(vae, &config.video_vae)?;
                let latent = u64::from(config.video_vae.latent_channels);
                let weight = loader.constant(
                    "video_decoder.input.weight",
                    dtype,
                    nn::Shape::new(&[width, latent]),
                    &weight,
                )?;
                let bias = loader.constant(
                    "video_decoder.input.bias",
                    dtype,
                    nn::Shape::new(&[width]),
                    &bias,
                )?;
                let input =
                    nn::layers::Linear::new(weight, Some(bias), ops::gemm::WeightOrder::NxK)?;
                // A VAE directory holds both halves; the decoder reads its own keys from it.
                let source = checkpoint.component(component.directories()[0])?;
                let mut shards = loader.reborrow(source)?;
                let decoder = Arc::new(crate::video_vae::VideoDecoder::load(
                    &config.video_vae,
                    canvas,
                    attention,
                    dtype,
                    input,
                    &mut shards,
                )?);
                video_decoder = Some(Arc::clone(&decoder));
                Block::Decoder(decoder)
            }
            Component::LatentEncoder => {
                // No block of its own: the two condition encoders are the `denoiser` block's
                // conditioning half, loaded above and attached when that block is built. This
                // component names the two checkpoint directories they read from, and `validate`
                // requires a DiT beside it.
                continue;
            }
        };
        // The runner addresses a block by the capability it serves, not by the checkpoint's
        // directory: both DiT partitions are the `denoiser` block.
        blocks.push((BlockName::new(component.block()), block));
    }

    // A conditioned deployment is served as the checkpoint states it: the encoders are attached to
    // the denoiser block, and `prepare_conditioning` writes a condition's audio rows and its visual
    // rows from them. What a runner must not do is offer a conditioned request to a deployment that
    // places no encoder - `Denoiser::places_encoders` answers that, and `encode_media` refuses a
    // request that carries media it holds no encoder for.

    Ok(MinimaxH3 {
        model: nn::Model::new(blocks),
        text_encoder,
        transformer,
        transformer_ref,
        video_postprocessor,
        video_decoder,
        audio_decoder,
    })
}

/// The checkpoint's pipeline layout and its sidecar documents.
///
/// One function on purpose: a checkpoint that spans a base release is still read from the one root
/// `Checkpoint` was opened at, and when the union form lands (a checkpoint whose components come
/// from two roots) this is the only place that changes.
fn sidecars(checkpoint: &nn::Checkpoint) -> Result<(Layout, Sidecars)> {
    // The pipeline's own `model_index.json` names the components; each component's sidecars are
    // read from the root that holds it, because a checkpoint may span an export and the base
    // release it continues.
    let layout = Layout::detect(checkpoint.root())?;
    let sidecars = Sidecars::read_components(checkpoint, &layout)?;
    Ok((layout, sidecars))
}

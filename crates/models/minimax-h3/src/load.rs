//! The loading boundary of a deployment: the blocks one rank places, built from the checkpoint.
//!
//! An H3 checkpoint is a diffusers pipeline directory whose components are its subdirectories, and
//! `nn::Checkpoint` learns such a root: `checkpoint.component("transformer")` gives one
//! component's shard set with each shard's own identity, and `WeightLoader::reborrow` gives one
//! loader per component over the same store and precision policy (specs/worker.md 8). The blocks a
//! rank holds are the [`Partition`]'s, so this module only builds them, and registers each under the
//! name its runner addresses: the capability's name, not the checkpoint directory's.
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
) -> Result<nn::Model> {
    let (layout, sidecars) = sidecars(checkpoint)?;
    let config = Config::normalize(&layout, &sidecars)?;
    partition.validate(&config)?;

    // The two condition encoders are the denoiser block's half, and they are loaded before the
    // loop because a block is boxed as it is built: the partition places `transformer` before
    // `latent_encoder`, so by the time the encoders' own component is reached the denoiser is a
    // `Block::Denoiser` and the concrete block that holds them is out of reach. A deployment whose
    // tasks carry no conditions places none.
    let mut encoders = if partition.blocks.contains(Component::LatentEncoder) {
        Some(ConditionEncoders::load(&config, checkpoint, loader)?)
    } else {
        None
    };

    let mut blocks = Vec::new();
    for component in partition.blocks.iter() {
        let block = match component {
            Component::TextEncoder => {
                let source = checkpoint.component("text_encoder")?;
                let dtype = component.activation();
                let mut shards = loader.reborrow(source)?;
                let encoder =
                    TextEncoder::load(&mut shards, &config.text_encoder, &partition, dtype)?;
                Block::Encoder(Box::new(encoder))
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
                Block::Denoiser(Box::new(denoiser))
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
                let postprocessor = crate::output::VideoPostprocessor::new(dtype, mean, std)?;
                nn::Block::Decoder(Box::new(postprocessor))
            }
            Component::AudioDecoder => {
                // The audio decoder has no layout tables - every table it reads is a weight bound
                // at load - so it constructs from its own component and its sidecar alone.
                let source = checkpoint.component("audio_vae")?;
                let mut shards = loader.reborrow(source)?;
                let decoder = AudioDecoder::load(&config.audio_vae, &mut shards)?;
                Block::Decoder(Box::new(decoder))
            }
            Component::VideoDecoder => {
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
                let decoder = crate::video_vae::VideoDecoder::load(
                    &config.video_vae,
                    canvas,
                    attention,
                    dtype,
                    input,
                    &mut shards,
                )?;
                Block::Decoder(Box::new(decoder))
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

    // The encoders are constructed and attached, and nothing writes a condition row yet:
    // `Denoiser::prepare_conditioning` returns before the video encoder's posterior epilogue
    // exists. A deployment that would serve `fl2va` or `ref2va` is therefore refused by the tasks
    // it names rather than loaded and served with a video conditioned on nothing - a load that
    // succeeds here would be a different model than the one the checkpoint states.
    if partition.blocks.contains(Component::LatentEncoder) {
        let mut conditioned: Vec<String> = config
            .denoisers
            .values()
            .flat_map(|denoiser| denoiser.tasks.iter())
            .filter(|task| !matches!(task, nn::capability::VideoTask::TextToVideoAudio))
            .map(|task| format!("{task:?}"))
            .collect();
        conditioned.sort();
        conditioned.dedup();
        if !conditioned.is_empty() {
            return Err(Error::Unsupported(format!(
                "this deployment places the condition encoders for {}, and neither task can be \
                 served yet: the conditioning pass writes no condition row until the video \
                 encoder's posterior epilogue exists, so a conditioned request would produce a \
                 video conditioned on nothing",
                conditioned.join(", ")
            )));
        }
    }

    Ok(nn::Model::new(blocks))
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

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
use crate::denoiser::{Denoiser, Layouts};
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
                let denoiser = Denoiser::load(
                    denoiser.clone(),
                    arrangement,
                    rows,
                    &partition,
                    &mut shards,
                    heads,
                )?;
                Block::Denoiser(Box::new(denoiser))
            }
            Component::AudioDecoder => {
                // The audio decoder has no layout tables - every table it reads is a weight bound
                // at load - so it constructs from its own component and its sidecar alone.
                let source = checkpoint.component("audio_vae")?;
                let mut shards = loader.reborrow(source)?;
                let decoder = AudioDecoder::load(&config.audio_vae, &mut shards)?;
                Block::Decoder(Box::new(decoder))
            }
            Component::VideoDecoder | Component::LatentEncoder => {
                // The video decoder is sized by the canvas a request serves and the condition
                // encoders become capability blocks as their layers land; until then a deployment
                // that places one is refused here rather than served by a half-built block.
                return Err(Error::Unsupported(format!(
                    "the {} block has no construction yet, so it cannot be placed",
                    component.name()
                )));
            }
        };
        // The runner addresses a block by the capability it serves, not by the checkpoint's
        // directory: both DiT partitions are the `denoiser` block.
        blocks.push((BlockName::new(component.block()), block));
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

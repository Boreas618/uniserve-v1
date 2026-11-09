//! The loading boundary of a deployment: the blocks one rank places, built from the checkpoint.
//!
//! An H3 checkpoint is a diffusers pipeline directory whose components are its subdirectories, and
//! `nn::Checkpoint` learns such a root: `checkpoint.component("transformer")` gives one
//! component's shard set with each shard's own identity, and `WeightLoader::reborrow` gives one
//! loader per component over the same store and precision policy. The blocks a rank holds are the
//! [`Partition`]'s, so this module only builds them.
//!
//! ## What the caller gets back
//!
//! [`MinimaxH3`]: the concrete components this rank places, each behind one `Arc`. The model's
//! family hands them to execution as capability objects (`crate::family`), and the concrete
//! handles are what the model's programs are enumerated from, so the weights exist once.
//!
//! ## Testing
//!
//! A load publishes every tensor through `nn::weights::RetainedWeights`, which only the worker
//! implements, so this module has no unit test of its own: the block constructors are tested where
//! they live, and the composition is exercised by the caller's load of a real checkpoint. What can
//! be checked here without a device is checked - `Partition`'s rules, the packing's tables, and the
//! component names and directories.
//!
//! ## Precision
//!
//! With the design's `dtype = "auto"` nothing is converted at load: each weight keeps the
//! representation the checkpoint stores it in, and each layer declares the precision it computes
//! in. That representation is a fact about the export, not a preset: the released checkpoint is
//! BF16 everywhere but its video VAE decoder, while an export of the same model can store FP32
//! throughout, and `auto` must follow each of them. [`stored_dtype`] is the one place that reads
//! it.

use std::sync::Arc;

use crate::audio_vae::BigVgan;
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

/// A loaded deployment: the components this rank places, each constructed once.
pub struct MinimaxH3 {
    text_encoder: Option<Arc<TextEncoder>>,
    transformer: Option<Arc<Denoiser>>,
    transformer_ref: Option<Arc<Denoiser>>,
    video_postprocessor: Option<Arc<crate::output::VideoPostprocessor>>,
    video_decoder: Option<Arc<crate::video_vae::VideoDecoder>>,
    audio_decoder: Option<Arc<BigVgan>>,
}

impl std::fmt::Debug for MinimaxH3 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MinimaxH3")
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
                &self.audio_decoder.as_ref().map(|_| "BigVgan"),
            )
            .finish()
    }
}

impl MinimaxH3 {
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
    pub fn audio_decoder(&self) -> Option<&Arc<BigVgan>> {
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

/// Refuse a deployment whose text encoder and DiT are stored in different element types.
///
/// The refiner between them reads the encoder's `[rows, text_dim]` features and projects them with
/// the **DiT's** `token_refiner.input`, so a mixed pair is one gemm whose activation is the encoder's
/// type and whose weight is the DiT's - a cell no backend can serve (`cuBLASLt refused an argument`),
/// and one that surfaces deep inside a forward rather than at the boundary that composed it.
///
/// **The released pipeline converts instead**: it casts the features to the input projection's
/// weight type once, before the projection. This crate's refiner has one element type for the
/// weight and the activation (the `scaled_add`/`gemm` rule: no mixed cells), so the cast cannot be
/// expressed today and a mixed pair is refused here. When a convert exists, the reference's line is
/// what to implement, and this refusal becomes the cast's precondition rather than a limit. `dtype
/// = "auto"` converts nothing, so the two must already agree; a deployment that states a type gets
/// one conversion at load, which is what an explicit `dtype` is for.
pub fn refiner_types(text_encoder: nn::DType, dit: nn::DType, dit_directory: &str) -> Result<()> {
    if text_encoder == dit {
        return Ok(());
    }
    Err(Error::checkpoint(format!(
        "the text encoder is stored in {text_encoder} and `{dit_directory}` in {dit}, and the \
         refiner between them projects the encoder's features with the DiT's own weights: that is one \
         gemm of a {text_encoder} activation with a {dit} weight, which no compiled cell can run. \
         With `dtype = \"auto\"` nothing is converted, so this export needs a deployment that states \
         one element type for both."
    )))
}

/// The one floating-point type a component's weights are stored in.
///
/// `dtype = "auto"` converts nothing at load, so this is the element type a block built from that
/// component keeps and computes in. It is read from the checkpoint - the tensors' own headers, which
/// `nn::Checkpoint` has without mapping a byte - because the export need not declare it: a diffusers
/// sidecar that omits `torch_dtype` states nothing, and a preset in this crate would be a second
/// statement of a number the files already hold.
///
/// This is the **one** type a consumer takes for a whole component - a module whose tables are
/// published in a single dtype, or a container that must agree with its export. It is **not** the
/// answer to "what does `auto` load": `auto` loads each weight in its own stored type, and a
/// component may store several ([`stored_types`] lists them), because the reference pins a type per
/// tensor. A component of two is therefore refused here by name - because *this caller* needs one
/// type - and never as a statement that the checkpoint is wrong. Integer tensors - a table, an index
/// - are not weights and do not count.
pub fn stored_types(
    checkpoint: &nn::Checkpoint,
    component: &str,
) -> Result<Vec<(nn::DType, String)>> {
    let source = checkpoint.component(component)?;
    // The element types present, with one tensor that carries each, so a question about them is
    // answerable by name.
    let mut found: Vec<(nn::DType, String)> = Vec::new();
    for name in source.names() {
        let Some(location) = source.tensor(name) else {
            continue;
        };
        if !location.dtype.is_float() {
            continue;
        }
        if !found.iter().any(|(dtype, _)| *dtype == location.dtype) {
            found.push((location.dtype, name.to_owned()));
        }
    }
    Ok(found)
}

/// The element type a consumer takes for a whole component, refusing one of two by name.
pub fn stored_dtype(checkpoint: &nn::Checkpoint, component: &str) -> Result<nn::DType> {
    let found = stored_types(checkpoint, component)?;
    match found.len() {
        1 => Ok(found[0].0),
        0 => Err(Error::checkpoint(format!(
            "the checkpoint's `{component}` component holds no floating-point tensor, so there is \
             no representation to load its weights in"
        ))),
        _ => Err(Error::checkpoint(format!(
            "the checkpoint's `{component}` component stores {} floating-point types ({}); the \
             consumer that asked takes one element type for a whole component, and a component of \
             several is legal under `dtype = \"auto\"` - each weight loads in its own stored type - \
             so this is a limit of that consumer, not of the export",
            found.len(),
            found
                .iter()
                .map(|(dtype, example)| format!("{dtype} (such as `{example}`)"))
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// The element type of one tensor of a component.
///
/// A component may store several types (a release that keeps a loading derivation's operand in
/// FP32, say), so a question about one weight must be asked of that weight. The name must be in the
/// component's index; a name it lacks is refused here rather than defaulted.
pub fn tensor_dtype(checkpoint: &nn::Checkpoint, component: &str, name: &str) -> Result<nn::DType> {
    let source = checkpoint.component(component)?;
    let location = source.tensor(name).ok_or_else(|| {
        Error::checkpoint(format!(
            "the checkpoint's `{component}` component has no tensor `{name}`, so its element type \
             cannot be read"
        ))
    })?;
    Ok(location.dtype)
}

/// Load the deployment this rank places.
///
/// The place and the blocks are the `partition`'s: it names the degrees and the rank's position in
/// them, the blocks this rank holds, and the row capacities those blocks are sized for. The
/// checkpoint is one file set per pipeline: an export that continues a base release is opened by
/// the engine as one checkpoint spanning both, so this function never learns about bases.
///
/// `dtype` is the deployment's element type: `Some` converts every weight once as it is published,
/// and the retained key names the conversion, so a deployment over an export of another type is
/// legal and its keys say what it did. `None` is the design's `auto`: each block keeps the
/// representation its weights are stored in ([`stored_dtype`]), and each layer declares the
/// precision it computes in. Under `auto` nothing converts, so a component of
/// two floating-point types is refused here rather than mixed in a forward.
///
/// A block this build cannot construct is refused with [`Error::Unsupported`], naming it, rather
/// than omitted: a deployment that places one must fail at load, not on the device.
pub fn load(
    checkpoint: &nn::Checkpoint,
    partition: Partition,
    dtype: Option<nn::DType>,
    canvases: &[(u32, u32)],
    loader: &mut nn::weights::WeightLoader<'_>,
) -> Result<MinimaxH3> {
    let (layout, sidecars) = sidecars(checkpoint)?;
    let config = Config::normalize(&layout, &sidecars)?;
    partition.validate(&config)?;
    // The deployment's element type, for the parts this function builds itself: a stated type is
    // what every weight is converted to, and `None` is each component's own representation.
    let deployment = |component: Component| -> Result<nn::DType> {
        match dtype {
            Some(stated) => Ok(stated),
            None => stored_dtype(checkpoint, component.directories()[0]),
        }
    };
    // A component that stores more than one floating-point type is legal under `auto`: the
    // checkpoint pins a type per tensor, and a pin is the *request*'s, so each weight is loaded at
    // the type its own tensor is pinned to and nothing is silently mixed. What must never happen is
    // a launch whose activation and weight differ, and that is refused where it happens - by the
    // request that states no pin, and by the operator's own launch check, which names the cell. Two
    // components, one number: the refiner spans the text encoder and the DiT, so their stored types
    // must agree before a single weight is read.
    if partition.blocks.contains(Component::TextEncoder) {
        for dit in [Dit::Base, Dit::Reference] {
            let placed = match dit {
                Dit::Base => partition.blocks.contains(Component::Transformer),
                Dit::Reference => partition.blocks.contains(Component::TransformerRef),
            };
            if placed {
                // The probe is the refiner's first attention projection: it is the gemm this
                // check is about - the encoder's features multiplied by a weight of the DiT - and
                // the export states it. The refiner has no input projection to probe: its layers
                // are `token_refiner.refiner_blocks.<n>` and `token_refiner.final_norm`, which is
                // the prefix the reference reads too.
                refiner_types(
                    deployment(Component::TextEncoder)?,
                    tensor_dtype(
                        checkpoint,
                        dit.directory(),
                        "token_refiner.refiner_blocks.0.attn.to_q.weight",
                    )?,
                    dit.directory(),
                )?;
            }
        }
    }

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

    let mut text_encoder: Option<Arc<TextEncoder>> = None;
    let mut transformer: Option<Arc<Denoiser>> = None;
    let mut transformer_ref: Option<Arc<Denoiser>> = None;
    let mut video_postprocessor: Option<Arc<crate::output::VideoPostprocessor>> = None;
    let mut video_decoder: Option<Arc<crate::video_vae::VideoDecoder>> = None;
    let mut audio_decoder: Option<Arc<BigVgan>> = None;
    for component in partition.blocks.iter() {
        match component {
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
                // The text encoder's representation is the checkpoint's, like every other block's:
                // `TextEncoder::load` converts to whatever it is handed, so a constant here would
                // be the one conversion in the load that no file states.
                let dtype = deployment(Component::TextEncoder)?;
                let mut shards = loader.reborrow(source)?;
                let encoder = Arc::new(TextEncoder::load(
                    &mut shards,
                    &config.text_encoder,
                    &partition,
                    dtype,
                )?);
                text_encoder = Some(encoder);
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
                    Dit::Base => transformer = Some(denoiser),
                    Dit::Reference => transformer_ref = Some(denoiser),
                }
            }
            Component::VideoPostprocessor => {
                // Weightless: the splice's only constants are the reference's ImageNet means and standard
                // deviations, uploaded once as the two `[3]` weights the byte step reads. It reads and
                // writes the video decoder's rasters, so it holds their element type, which the
                // decoder declares.
                let dtype = crate::video_vae::ACTIVATION;
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
                    "minimax-h3",
                )?;
                let std = loader.constant(
                    "video_postprocessor.pixel_std",
                    nn::DType::F32,
                    shape,
                    &bytes(crate::output::PIXEL_STD),
                    "minimax-h3",
                )?;
                let postprocessor =
                    Arc::new(crate::output::VideoPostprocessor::new(dtype, mean, std)?);
                video_postprocessor = Some(postprocessor);
            }
            Component::AudioDecoder => {
                // The audio decoder has no layout tables - every table it reads is a weight bound
                // at load - so it constructs from its own component and its sidecar alone.
                let source = checkpoint.component("audio_vae")?;
                require(
                    source,
                    &BigVgan::keys(&config.audio_vae)?,
                    component.name(),
                    "decoder",
                    "audio_vae",
                )?;
                let mut shards = loader.reborrow(source)?;
                let decoder = Arc::new(BigVgan::load(&config.audio_vae, &mut shards)?);
                audio_decoder = Some(decoder);
            }
            Component::VideoDecoder => {
                require(
                    checkpoint.component("vae")?,
                    &crate::video_vae::VideoDecoder::keys(&config.video_vae)?,
                    component.name(),
                    "decoder",
                    "vae",
                )?;
                // The block is built for the deployment's raster, which the caller states in
                // pixels: the decoder's register rows, its voxel-row table and its sequence offsets
                // are that raster's constants, published here at load, and the seam blends are
                // built here too so that no layer is built on a request's path.
                // The decoder converts the deployment's pixel rasters to its latent canvases and
                // checks each is whole cells; this file only passes them through.
                // A VAE directory holds both halves; the decoder reads its own keys from it.
                let source = checkpoint.component(component.directories()[0])?;
                let mut shards = loader.reborrow(source)?;
                let decoder = Arc::new(crate::video_vae::VideoDecoder::load(
                    &config.video_vae,
                    canvases,
                    &mut shards,
                )?);
                video_decoder = Some(decoder);
            }
            Component::LatentEncoder => {
                // No block of its own: the two condition encoders are the `denoiser` block's
                // conditioning half, loaded above and attached when that block is built. This
                // component names the two checkpoint directories they read from, and `validate`
                // requires a DiT beside it.
                continue;
            }
        }
    }

    // A conditioned deployment is served as the checkpoint states it: the encoders are attached to
    // the denoiser block, and `prepare_conditioning` writes a condition's audio rows and its visual
    // rows from them. What a runner must not do is offer a conditioned request to a deployment that
    // places no encoder - `Denoiser::places_encoders` answers that, and `encode_media` refuses a
    // request that carries media it holds no encoder for.

    Ok(MinimaxH3 {
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

//! Encoder runners: vision patch encoders, latent encoders, text encoders
//! and a video model's condition latent encoders.
//!
//! An [`EncoderRunner`] serves one encoder call of a bound component: the
//! module at a path of the loaded model (a component's root, or a nested
//! encoder such as a text encoder's vision tower bound with it). It owns the
//! encoder's Rust buffers, its graph table and its call policy.
//!
//! A text encoder and the vision tower that encodes its prompt's packed
//! vision blocks compute through the Python half,
//! `uniserve_models.runner.encoder.EncoderRunner`, constructed directly
//! with its stream's shared scratch ([`EncoderRunner::text`]): the text
//! encoder's half owns the execution context, its tower borrows it, and
//! `encode` takes the tensors of one forward as arguments.
//!
//! An image model's vision and latent encoders compute through the same
//! Python class, constructed over a bound component by
//! `EncoderRunner.bound` ([`EncoderRunner::bind`]):
//!
//! | kind | module | methods |
//! | --- | --- | --- |
//! | [`EncoderInfo::Vision`] | `PatchEncoder` | `encode`, `prepare_resident`, `warm`; packed: `encode` over the slot tensors, `pack_images` |
//! | [`EncoderInfo::Latent`] | `PatchAutoencoder` | `encode`, `prepare_resident`, `warm` |
//!
//! Such a runner receives its device's product arena once; an unpacked
//! `encode` writes its results into product [`Region`]s of it, in the
//! product's dtype, and returns the results' shapes, which the caller
//! publishes.
//!
//! **Images.** Inline request images are decoded and transformed on a host
//! lane by `EncoderRunner.prepare_image` ([`prepare_input`]), which returns an
//! opaque [`PreparedImage`] in page-locked memory; resident image products
//! are transformed on the device by `prepare_resident`. Both reach `encode`
//! as handles. Unpacked encoders encode images of one pixel shape per
//! numerical call, in first-appearance order, so an image's features do not
//! depend on the other images of its batch.
//!
//! **Packed vision.** An encoder with `max_patches` encodes images packed in
//! fixed slots of `max_patches` patch rows (`PatchEncoder.encode_packed`):
//! one shared pixel buffer, slot grids and a feature buffer, captured once
//! per slot count `capacity..=1` (largest first, so smaller captures reuse
//! the pool blocks of larger ones) with `capacity = min(max_batch_calls,
//! 16)`. A batch replays full chunks of `capacity` images, then the exact
//! remainder. Rust writes the slot grids ([`packed_grids`]) before every
//! call; images are independent of their slot neighbours by the encoder's
//! contract, so a chunk may mix image sizes.
//!
//! **Text.** Prompts are zero-padded to the smallest configured text
//! capacity that holds them ([`text_capacity`]); causal attention keeps the
//! padding out of the prompt's rows, and one captured graph per capacity
//! serves every prompt length up to it: the runner writes the padded tokens,
//! launches the graph and copies the prompt's feature rows into the product,
//! all natively. Prompts with vision tokens (rotary coordinates and vision
//! rows) run on the Python thread.
//!
//! Every runner call is ordered on the runner's stream; the caller orders
//! that stream after the producers of its inputs and records the producer
//! fence of its outputs on it.

pub(crate) mod inputs;
pub(crate) mod runners;
mod start;
pub(crate) mod warmup;

use std::collections::HashMap;
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::{Canvas, VideoCondition};
use uniserve_cuda::{CopyBatch, SourceOrder, Stream};
use uniserve_ipc::{Batch as BatchPlan, CallKind, ForwardStats, MediaCall};

use crate::config::Device;
use crate::cuda_graph::GraphPool;
use crate::cuda_graph::{GraphMode, GraphTable, Lookup};
use crate::executor::forward::batch::MediaBatch;
use crate::executor::forward::batch::image::ImageParts;
use crate::executor::forward::batch::image::{ImageBatch, allocations, submit};
use crate::executor::forward::batch::{
    Parts, bind_rows, decode_range, product_name, publish, publish_bound, row_width, stream_of,
};
use crate::executor::forward::placement::{OutputShape, Placement, bind_outputs};
use crate::executor::forward::products::Publications;
use crate::ffi::{Backing, DType, Function, Handle, integer_list};
use crate::lane::{Completion, Job};
use crate::model::interface::Interface;
use crate::parallel::joins::GraphFunctions;
use crate::profiling::kernels::KernelRecord;
use crate::runner::methods::{Argument, MethodCall, Methods, RunnerClass, kernel_records};
use crate::runner::{ImageSetup, RunnerSetup};
use crate::storage::memory::{Memory, StorageView};
use crate::storage::{Metadata, OutputPlacement, ValueRange};
use crate::stream::Issue;
use crate::{Error, Result};

const ENCODER: &str = "uniserve_models.runner.encoder.EncoderRunner";

/// The largest packed vision slot count: it bounds retained activations
/// independently of the admitted batch size. Larger batches replay full
/// chunks and then their exact remaining slot count.
pub const MAX_PACKED_IMAGES: usize = 16;

/// The image transform a preparation applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tower {
    /// The understanding tower (`ImageProcessor.vit`): vision encoding.
    Vit,
    /// The latent codec tower (`ImageProcessor.vae`): latent encoding.
    Vae,
}

impl Tower {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vit => "vit",
            Self::Vae => "vae",
        }
    }
}

/// One image prepared for an encoder: the opaque Python value and the facts
/// planning needs.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    /// The prepared pixels (`uniserve_models.media.pixels.PreparedImage`).
    pub value: Handle,
    /// `[patches, row]` for a patch tower, `[channels, height, width]`
    /// otherwise.
    pub shape: Vec<i64>,
    /// The `[rows, columns]` patch grid of a patch tower.
    pub grid: Option<[u32; 2]>,
    /// The canvas the encoder's features describe.
    pub height: u32,
    pub width: u32,
}

impl PreparedImage {
    /// The image `value` a preparation returned, with what `facts`
    /// (`EncoderRunner.image_facts`) reports of it.
    fn from_value(facts: &Function, value: Any) -> Result<Self> {
        let image = Handle::new(value)?;
        let values = crate::ffi::integers(facts.call_values(&[image.to_any()])?)
            .map_err(|error| Error::numerical(facts.name(), &error))?;
        Self::from_facts(facts, image, &values)
    }

    /// The prepared `image` with the `values` that `facts` reported of it.
    fn from_facts(facts: &Function, image: Handle, values: &[i64]) -> Result<Self> {
        let [height, width, rows, columns, ref shape @ ..] = values[..] else {
            return Err(Error::Interface(format!(
                "{} reports a canvas, a grid and a pixel shape",
                facts.name()
            )));
        };
        let dimension = |extent: i64, name: &str| -> Result<u32> {
            u32::try_from(extent)
                .ok()
                .filter(|&extent| extent > 0)
                .ok_or_else(|| {
                    Error::Interface(format!("{} reports an invalid {name}", facts.name()))
                })
        };
        // A patch tower reports its grid; other towers report zeros.
        let grid = match (rows, columns) {
            (0, 0) => None,
            (rows, columns) if rows > 0 && columns > 0 => Some([rows as u32, columns as u32]),
            _ => {
                return Err(Error::Interface(format!(
                    "{} reports a grid that is not two positive extents",
                    facts.name()
                )));
            }
        };
        Ok(Self {
            value: image,
            shape: shape.to_vec(),
            grid,
            height: dimension(height, "height")?,
            width: dimension(width, "width")?,
        })
    }

    /// Patch rows of a packed image (`rows * columns`).
    pub fn patches(&self) -> Option<u32> {
        self.grid.map(|[rows, columns]| rows * columns)
    }
}

/// Decode and transform one inline request image on the calling (host)
/// thread with the model's image policy.
///
/// `input_images` is the request's input image count, which a shared pixel
/// budget divides; `pin` places the pixels in page-locked memory for an
/// asynchronous copy to a CUDA device.
pub fn prepare_input(
    function: &Function,
    facts: &Function,
    loaded: &Handle,
    tower: Tower,
    payload: &str,
    input_images: u32,
    pin: bool,
) -> Result<PreparedImage> {
    if input_images == 0 {
        return Err(Error::Invalid(
            "request admission declares no input images".into(),
        ));
    }
    let value = function.call_values(&[
        loaded.to_any(),
        Any::from(tvm_ffi::String::from(tower.name())),
        Any::from(tvm_ffi::String::from(payload)),
        Any::from(i64::from(input_images)),
        Any::from(pin),
    ])?;
    PreparedImage::from_value(facts, value)
}

/// The text workload an encoder's execution context is prepared for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextSize {
    pub tokens: usize,
    pub rows: usize,
}

/// A vision patch encoder (`PatchEncoder`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionEncoder {
    pub patch_size: u32,
    /// Input patches per output feature on each spatial axis.
    pub downsample: u32,
    pub output_size: u32,
    pub output_dtype: DType,
    /// Patch rows of one packed slot; `None` for an unpacked encoder.
    pub max_patches: Option<u32>,
    /// Patch rows of the largest image the image policy admits.
    pub image_patches: u32,
    /// Values per packed patch row.
    pub row_width: u32,
    /// The image policy's pixel dtype.
    pub pixel_dtype: DType,
}

impl VisionEncoder {
    /// Feature rows of one packed slot.
    pub fn slot_tokens(&self) -> Option<u32> {
        self.max_patches
            .map(|patches| patches / (self.downsample * self.downsample))
    }
}

/// A latent encoder (`PatchAutoencoder`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentEncoder {
    /// Pixels per latent patch on each spatial axis.
    pub downsample: u32,
    /// Values per latent patch row.
    pub width: u32,
    pub output_dtype: DType,
}

/// A text encoder (`TextEncoder`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextEncoder {
    pub hidden_size: u32,
    pub output_dtype: DType,
    /// The encoder places vision tokens with rotary coordinates.
    pub positions: bool,
}

/// What an encoder computes, as the `EncoderRunner` fact methods report it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderInfo {
    Vision(VisionEncoder),
    Latent(LatentEncoder),
    Text(TextEncoder),
    /// A video model's condition latent encoders (`VideoEncoder` and
    /// `AudioEncoder`), whose Python runners serve one input shape each.
    Condition,
}

impl EncoderInfo {
    /// Describe the encoder at module `path` of the loaded model, before
    /// its component is bound. `dtype` is the deployment's model dtype.
    pub fn describe(
        interface: &Interface,
        loaded: &Handle,
        path: &str,
        dtype: DType,
    ) -> Result<Self> {
        let path = Any::from(tvm_ffi::String::from(path));
        let method = |name: &str| interface.import(&format!("{ENCODER}.{name}"));
        let extent = |name: &str| -> Result<u32> {
            let function = method(name)?;
            let value: i64 = function.call_as(&[loaded.view(), (&path).into()])?;
            u32::try_from(value)
                .map_err(|_| Error::Interface(format!("{} is out of range", function.name())))
        };
        let dtype_named = |function: &Function, name: tvm_ffi::String| -> Result<DType> {
            DType::from_name(name.as_str()).ok_or_else(|| {
                Error::Interface(format!(
                    "{} reports the unknown dtype {:?}",
                    function.name(),
                    name.as_str()
                ))
            })
        };
        let output_dtype = || -> Result<DType> {
            let function = method("output_dtype")?;
            let name = function.call_as(&[loaded.view(), (&path).into()])?;
            dtype_named(&function, name)
        };
        let kind: tvm_ffi::String = method("kind")?.call_as(&[loaded.view(), (&path).into()])?;
        Ok(match kind.as_str() {
            "vision" => {
                let patch_size = extent("patch_size")?;
                let max_patches = extent("max_patches")?;
                let downsample = extent("downsample")?;
                if downsample == 0 || max_patches % (downsample * downsample) != 0 {
                    return Err(Error::Interface(
                        "packed vision slots must hold whole output features".into(),
                    ));
                }
                let pixels = method("pixel_dtype")?;
                let pixel_dtype = dtype_named(&pixels, pixels.call_as(&[loaded.view()])?)?;
                Self::Vision(VisionEncoder {
                    patch_size,
                    downsample,
                    output_size: extent("output_width")?,
                    output_dtype: output_dtype()?,
                    max_patches: (max_patches > 0).then_some(max_patches),
                    image_patches: extent("image_patches")?,
                    // A patch row holds the RGB values of one patch.
                    row_width: 3 * patch_size * patch_size,
                    pixel_dtype,
                })
            }
            "latent" => Self::Latent(LatentEncoder {
                downsample: extent("downsample")?,
                width: extent("output_width")?,
                output_dtype: output_dtype()?,
            }),
            "text" => Self::Text(TextEncoder {
                hidden_size: extent("output_width")?,
                // A text encoder's hidden states carry the model dtype.
                output_dtype: dtype,
                positions: method("places_vision")?
                    .call_as::<bool>(&[loaded.view(), (&path).into()])?,
            }),
            other => {
                return Err(Error::Interface(format!(
                    "{ENCODER}.kind reports the unknown encoder kind {other:?}"
                )));
            }
        })
    }

    /// The text size the encoder's execution context prepares its
    /// operators for: the most rows one numerical call passes through its
    /// projections. A latent encoder's convolutions need none.
    pub fn context_size(&self, config: &EncoderConfig) -> Option<TextSize> {
        match self {
            Self::Vision(vision) => Some(match vision.max_patches {
                Some(max_patches) => {
                    // Every packed slot is two segments: its image and its
                    // padding.
                    let slots = packed_capacity(config.max_images);
                    TextSize {
                        tokens: slots * max_patches as usize,
                        rows: 2 * slots,
                    }
                }
                None => TextSize {
                    tokens: unpacked_rows(vision, config),
                    rows: config.max_images.max(1),
                },
            }),
            Self::Text(_) => Some(TextSize {
                tokens: config
                    .text_capacities
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(config.max_text_tokens)
                    .max(1),
                rows: 1,
            }),
            Self::Latent(_) | Self::Condition => None,
        }
    }
}

/// Lay out packed images in `slots` fixed slots of `max_patches` rows.
///
/// Each slot is described by two `[rows, columns]` grids: its image's patch
/// grid (`[0, 0]` for an empty slot), then a `downsample` rows high grid
/// covering the slot's remaining rows (`[0, 0]` when the image fills it).
/// Images take the leading slots in order. This is
/// `PatchEncoder.packed_grids`, which the captured encode reads from device
/// memory.
pub fn packed_grids(
    shapes: &[[u32; 2]],
    slots: usize,
    max_patches: u32,
    downsample: u32,
) -> Result<Vec<[i64; 2]>> {
    if shapes.len() > slots {
        return Err(Error::Invalid("packed images exceed their slots".into()));
    }
    let mut grids = Vec::with_capacity(2 * slots);
    for index in 0..slots {
        let shape = shapes.get(index).copied().unwrap_or([0, 0]);
        let rows = u64::from(shape[0]) * u64::from(shape[1]);
        if index < shapes.len()
            && (shape.contains(&0)
                || shape.iter().any(|side| side % downsample != 0)
                || rows > u64::from(max_patches))
        {
            return Err(Error::Invalid(
                "packed image grids must align with spatial downsampling within the slot capacity"
                    .into(),
            ));
        }
        let padding = u64::from(max_patches) - rows;
        grids.push([i64::from(shape[0]), i64::from(shape[1])]);
        grids.push(if padding == 0 {
            [0, 0]
        } else {
            [
                i64::from(downsample),
                (padding / u64::from(downsample)) as i64,
            ]
        });
    }
    Ok(grids)
}

/// Packed vision slots of one numerical call.
fn packed_capacity(max_images: usize) -> usize {
    max_images.clamp(1, MAX_PACKED_IMAGES)
}

/// Patch rows one unpacked vision call passes: the largest admitted image,
/// or the configured row budget when larger.
fn unpacked_rows(vision: &VisionEncoder, config: &EncoderConfig) -> usize {
    (vision.image_patches as usize)
        .max(config.row_budget)
        .max(1)
}

/// Patch rows an image passes through a vision encoder: its packed rows,
/// or its pixels' patches.
fn image_rows(image: &PreparedImage, patch_size: u32) -> usize {
    match image.shape[..] {
        [rows, _] => rows as usize,
        [_, height, width] => {
            (height as usize / patch_size as usize) * (width as usize / patch_size as usize)
        }
        _ => 0,
    }
}

/// The text capacity a prompt of `tokens` is encoded at: the smallest
/// configured capacity holding it, or the prompt's own length without
/// configured capacities.
pub fn text_capacity(capacities: &[usize], tokens: usize) -> Result<usize> {
    if tokens == 0 {
        return Err(Error::Invalid("text encoding requires a token".into()));
    }
    if capacities.is_empty() {
        return Ok(tokens);
    }
    crate::cuda_graph::bucket(capacities, tokens).ok_or_else(|| {
        Error::Invalid(format!(
            "prompt of {tokens} tokens exceeds the configured text capacities"
        ))
    })
}

/// The vision tokens of a prompt, for a text encoder that places them.
#[derive(Clone, Copy, Debug)]
pub struct PromptVision<'a> {
    /// `(time, height, width)` patch grids of the prompt's image blocks, in
    /// prompt order.
    pub image_grids: &'a [[u32; 3]],
    /// The same for its video blocks.
    pub video_grids: &'a [[u32; 3]],
    /// The vision rows of its placeholders, in placeholder order.
    pub rows: &'a StorageView,
}

/// Startup bounds of an encoder's buffers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Images one batch may carry (`max_batch_calls`).
    pub max_images: usize,
    /// Patch rows one unpacked vision call may pass beside the largest
    /// image (`max_batch_tokens`); larger groups of one pixel shape encode
    /// in consecutive calls.
    pub row_budget: usize,
    /// Configured text capacities, ascending; empty for exact lengths.
    pub text_capacities: Vec<usize>,
    /// The longest admitted prompt.
    pub max_text_tokens: usize,
}

/// A graph an encoder replays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EncoderGraph {
    /// A packed vision encode of this many slots.
    Packed { slots: usize },
    /// A text-only encode at this capacity.
    Text { capacity: usize },
}

/// The results of one encoder invocation set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Encoded {
    /// Each input's result shape, in input order; empty on a pipeline stage
    /// that computes no result.
    pub shapes: Vec<Vec<i64>>,
    /// Numerical invocations by how they ran.
    pub replays: u64,
    pub eager: u64,
}

/// A recorded encode: what is known of its result when it is recorded,
/// and the calls whose shapes the job's completion returns.
pub(crate) struct Recorded {
    encoded: Encoded,
    /// The inputs of each returning call, in call order.
    returning: Vec<Vec<usize>>,
    /// The method the returning calls call, for its errors.
    method: String,
}

impl Recorded {
    /// The encode's result, from the completion of the job that performed
    /// it: each returning call's shapes fill its inputs' places.
    /// Whether a call of the encode returns shapes, so that its result
    /// needs the completion of the job that performed it.
    pub(crate) fn returns(&self) -> bool {
        !self.returning.is_empty()
    }

    pub(crate) fn finish(self, completion: &mut Completion) -> Result<Encoded> {
        let Self {
            mut encoded,
            returning,
            method,
        } = self;
        for group in returning {
            // One shape per input: each as its length, then its extents.
            let returned = completion.returned()?;
            let mut values = returned.iter().copied();
            for &index in &group {
                let rank = values.next().and_then(|rank| usize::try_from(rank).ok());
                let shape: Option<Vec<i64>> = rank.map(|rank| values.by_ref().take(rank).collect());
                match (rank, shape) {
                    (Some(rank), Some(shape)) if shape.len() == rank => {
                        encoded.shapes[index] = shape;
                    }
                    _ => {
                        return Err(Error::Interface(format!(
                            "{method} returns one shape per input"
                        )));
                    }
                }
            }
            if values.next().is_some() {
                return Err(Error::Interface(format!(
                    "{method} returns one shape per input"
                )));
            }
        }
        Ok(encoded)
    }
}

impl Encoded {
    fn record(&mut self, mode: GraphMode) {
        match mode {
            GraphMode::Replay => self.replays += 1,
            GraphMode::Eager | GraphMode::Capture => self.eager += 1,
        }
    }
}

/// Packed vision buffers: `pixels [capacity * max_patches, row_width]`,
/// `grids [2 * capacity, 2]` int64 and `features [capacity * slot_tokens,
/// output_size]`, shared by every captured slot count (each uses the
/// leading rows).
struct PackedVision {
    capacity: usize,
    pixels: StorageView,
    grids: StorageView,
    features: StorageView,
}

/// The Python half (`uniserve_models.runner.encoder.EncoderRunner`) and its
/// methods, each called with the runner first.
struct EncoderMethods {
    runner: Handle,
    encode: Function,
    position_ids: Function,
    prepare_resident: Function,
    image_facts: Function,
    warm: Function,
    pack_images: Function,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
}

impl EncoderMethods {
    fn new(interface: &Interface, runner: Handle) -> Result<Self> {
        let method = |name: &str| interface.import(&format!("{ENCODER}.{name}"));
        Ok(Self {
            runner,
            encode: method("encode")?,
            position_ids: method("position_ids")?,
            prepare_resident: method("prepare_resident")?,
            image_facts: method("image_facts")?,
            warm: method("warm")?,
            pack_images: method("pack_images")?,
            capture: GraphFunctions::from_interface(interface)?,
        })
    }
}

/// The methods of a Python half that owns its execution context.
struct OwnedContext {
    kernels: Function,
    close: Function,
}

/// The buffers of a text encoder's forward.
struct TextBuffers {
    /// The `[max capacity]` int64 prompt, zero-padded.
    tokens: StorageView,
    /// The `[max capacity, hidden]` features of the padded prompt.
    features: StorageView,
    /// The `[3, max capacity]` int64 rotary coordinates of a prompt with
    /// vision tokens, for an encoder that places them.
    positions: Option<StorageView>,
    /// Whether this rank's pipeline stage emits the features.
    final_stage: bool,
}

/// The runner of one bound encoder call.
pub struct EncoderRunner {
    owner: String,
    info: EncoderInfo,
    device: Device,
    /// The runner's stream, shared with its lane; `None` on the host.
    stream: Option<Stream>,
    methods: EncoderMethods,
    packed: Option<PackedVision>,
    text: Option<TextBuffers>,
    text_capacities: Vec<usize>,
    config: EncoderConfig,
    graphs: Option<GraphTable<EncoderGraph>>,
    /// Capture is enabled: graph pools were supplied at startup.
    capturing: bool,
    /// The component binding a video model's text encoder owns, with its
    /// workspaces; its vision tower computes with the same binding.
    /// The methods releasing the execution context this runner's Python
    /// half constructed (an image encoder, or a text encoder whose vision
    /// tower computes with the same context).
    context: Option<OwnedContext>,
    /// The Python runners of condition latent encoders.
    conditions: Option<Box<Conditions>>,
}

fn allocate(device: Device, dtype: DType, shape: &[i64]) -> Result<StorageView> {
    let bytes = (shape.iter().product::<i64>().max(1) as usize) * dtype.bytes();
    let memory: Arc<dyn Backing> = match device {
        Device::Cuda(index) => Memory::device(index, bytes)?,
        Device::Cpu => Memory::host(bytes),
    };
    StorageView::contiguous(memory, 0, dtype, shape)
}

impl EncoderRunner {
    /// The runner of an image model's vision or latent encoder: allocate
    /// the buffers of the encoder `info` describes and construct its Python
    /// half, whose execution context is prepared for
    /// [`EncoderInfo::context_size`].
    pub fn image(
        setup: &ImageSetup<'_>,
        info: EncoderInfo,
        config: &EncoderConfig,
    ) -> Result<Self> {
        let interface = setup.interface;
        let owner = setup.owner();
        let (device, stream) = (setup.device, setup.stream.clone());
        if device.is_cuda() != stream.is_some() {
            return Err(Error::Invariant(
                "an encoder runs on a stream exactly on a CUDA device".into(),
            ));
        }

        let mut packed = None;
        match info {
            EncoderInfo::Vision(vision) => {
                if let (Some(max_patches), Some(slot_tokens)) =
                    (vision.max_patches, vision.slot_tokens())
                {
                    let capacity = packed_capacity(config.max_images) as i64;
                    let pixels = allocate(
                        device,
                        vision.pixel_dtype,
                        &[
                            capacity * i64::from(max_patches),
                            i64::from(vision.row_width),
                        ],
                    )?;
                    // Rows past an image keep earlier values, which only
                    // padding reads; they start finite.
                    pixels.zero(stream.as_ref())?;
                    packed = Some(PackedVision {
                        capacity: capacity as usize,
                        pixels,
                        grids: allocate(device, DType::I64, &[2 * capacity, 2])?,
                        features: allocate(
                            device,
                            vision.output_dtype,
                            &[
                                capacity * i64::from(slot_tokens),
                                i64::from(vision.output_size),
                            ],
                        )?,
                    });
                }
            }
            EncoderInfo::Latent(_) => {}
            EncoderInfo::Text(_) | EncoderInfo::Condition => {
                return Err(Error::Invariant(
                    "only an image model's vision or latent encoder is constructed here".into(),
                ));
            }
        }
        if let Some(stream) = &stream {
            // The zero fill must complete before Python reads the buffers.
            stream.synchronize()?;
        }

        let size = info.context_size(config);
        let mut arguments = setup.arguments();
        arguments.push(Any::from(integer_list(
            size.iter()
                .flat_map(|size| [size.tokens as i64, size.rows as i64]),
        )));
        let method = |name: &str| interface.import(&format!("{ENCODER}.{name}"));
        let runner = Handle::new(method("image")?.call_values(&arguments)?)?;
        let context = OwnedContext {
            kernels: method("kernels")?,
            close: method("close")?,
        };
        let methods = EncoderMethods::new(interface, runner)?;
        Ok(Self {
            graphs: packed
                .as_ref()
                .map(|_| GraphTable::new(owner.clone(), methods.capture.release.clone())),
            owner,
            info,
            device,
            stream,
            methods,
            packed,
            text: None,
            text_capacities: Vec::new(),
            config: config.clone(),
            capturing: false,
            context: Some(context),
            conditions: None,
        })
    }

    /// The runner of a text encoder (`path` `""`) or of its vision tower
    /// (`"vision"`) over its constructed Python half `runner`
    /// (`uniserve_models.runner.encoder.EncoderRunner`). A text encoder's
    /// token, feature and position buffers are allocated here for the
    /// largest prompt capacity.
    #[expect(
        clippy::too_many_arguments,
        reason = "the Python half and what its buffers are built from"
    )]
    fn over(
        interface: &Interface,
        component: &str,
        device: Device,
        runner: Handle,
        path: &str,
        info: EncoderInfo,
        config: &EncoderConfig,
        stream: Stream,
    ) -> Result<Self> {
        let owner = if path.is_empty() {
            component.to_owned()
        } else {
            format!("{component}.{path}")
        };
        let methods = EncoderMethods::new(interface, runner)?;

        let mut text_capacities = config.text_capacities.clone();
        text_capacities.sort_unstable();
        text_capacities.dedup();
        let (text, graphs) = match info {
            EncoderInfo::Text(text) => {
                let capacity = text_capacities
                    .last()
                    .copied()
                    .unwrap_or(config.max_text_tokens)
                    .max(1) as i64;
                let final_stage = interface
                    .import(&format!("{ENCODER}.final_stage"))?
                    .call_as::<bool>(&[methods.runner.view()])?;
                let buffers = TextBuffers {
                    tokens: allocate(device, DType::I64, &[capacity])?,
                    features: allocate(
                        device,
                        text.output_dtype,
                        &[capacity, i64::from(text.hidden_size)],
                    )?,
                    positions: text
                        .positions
                        .then(|| allocate(device, DType::I64, &[3, capacity]))
                        .transpose()?,
                    final_stage,
                };
                (
                    Some(buffers),
                    Some(GraphTable::new(
                        owner.clone(),
                        methods.capture.release.clone(),
                    )),
                )
            }
            EncoderInfo::Vision(_) => (None, None),
            EncoderInfo::Latent(_) | EncoderInfo::Condition => {
                return Err(Error::Invariant(
                    "only a text encoder and its vision tower share a context".into(),
                ));
            }
        };
        Ok(Self {
            graphs,
            owner,
            info,
            device,
            stream: Some(stream),
            methods,
            packed: None,
            text,
            text_capacities,
            config: config.clone(),
            capturing: false,
            context: None,
            conditions: None,
        })
    }

    pub fn info(&self) -> EncoderInfo {
        self.info
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The runner's stream; `None` on the host.
    pub fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    /// Images one packed numerical call encodes; `None` when unpacked.
    pub fn packed_capacity(&self) -> Option<usize> {
        self.packed.as_ref().map(|packed| packed.capacity)
    }

    /// The identity of the runner's graph pool owner: the address of its
    /// packed grids, which no other runner shares. `None` when unpacked.
    pub fn graph_key(&self) -> Option<usize> {
        self.packed
            .as_ref()
            .map(|packed| packed.grids.address() as usize)
    }

    /// Encode one black `side` x `side` image through the model's input
    /// transform and discard it: the startup warmup of an unpacked image
    /// encoder (packed vision encoders warm by capturing). A latent encoder
    /// sampling its posterior draws from the device's default generator.
    pub fn warm(&self, side: u32) -> Result<()> {
        match (&self.info, &self.packed) {
            (EncoderInfo::Vision(_), None) | (EncoderInfo::Latent(_), _) => {
                self.methods.warm.invoke(&[
                    self.methods.runner.view(),
                    (&Any::from(i64::from(side))).into(),
                ])
            }
            _ => Ok(()),
        }
    }

    /// Record the preparation of a resident image product for this encoder
    /// into `job`; `signed_unit` declares pixel values in `[-1, 1]` rather
    /// than `[0, 1]`. The job's completion carries the prepared image
    /// ([`Completion::handle`]), of which [`Self::image_facts`] records what
    /// planning needs.
    pub(crate) fn prepare_resident(
        &self,
        image: &StorageView,
        tower: Tower,
        signed_unit: bool,
        job: &Job<'_>,
    ) -> Result<()> {
        if matches!(self.info, EncoderInfo::Text(_)) {
            return Err(Error::Invalid(format!("{} encodes no images", self.owner)));
        }
        job.call_handle(MethodCall::new(
            &self.methods.prepare_resident,
            vec![
                Argument::Handle(self.methods.runner.clone()),
                Argument::Tensor(image.clone()),
                Argument::Text(tower.name().to_owned()),
                Argument::Bool(signed_unit),
            ],
        ));
        Ok(())
    }

    /// Record the call that reports the facts of a prepared `image` into
    /// `job`; the job's completion carries them ([`Completion::returned`])
    /// and [`Self::prepared`] reads them.
    pub(crate) fn image_facts(&self, image: &Handle, job: &Job<'_>) {
        job.call_returning(MethodCall::new(
            &self.methods.image_facts,
            vec![Argument::Handle(image.clone())],
        ));
    }

    /// The prepared `image` with the `facts` its facts call returned.
    pub(crate) fn prepared(&self, image: Handle, facts: &[i64]) -> Result<PreparedImage> {
        PreparedImage::from_facts(&self.methods.image_facts, image, facts)
    }

    /// Encode images, writing image `i`'s result into `out[i]`.
    ///
    /// Unpacked encoders run one numerical call per pixel shape, in first
    /// appearance order; packed vision runs chunks of the packed capacity.
    /// Returns each image's result shape.
    ///
    /// FIXME: unpacked vision and latent encoding are eager Python calls
    /// over Python-held pixels of request-chosen shapes, and a latent
    /// encoder's posterior draws from the device's default generator
    /// inside the call instead of reading noise the caller supplies.
    pub fn encode(&mut self, images: &[&PreparedImage], out: &[StorageView]) -> Result<Encoded> {
        // At once: the recorded encode, performed here.
        let job = Job::new();
        let recorded = self.record(images, out, &job)?;
        let mut completion = job.perform().check().map_err(|failure| failure.error)?;
        recorded.finish(&mut completion)
    }

    /// Record the encoding of `images` into `job`, image `i`'s result into
    /// `out[i]`: a packed encoder's grid upload, packing call, graph launch
    /// or eager call and feature copies per chunk; an unpacked encoder's
    /// eager call per group of one pixel shape. The job's completion gives
    /// the result ([`Recorded::finish`]): an unpacked call returns the
    /// shapes it wrote.
    pub(crate) fn record(
        &mut self,
        images: &[&PreparedImage],
        out: &[StorageView],
        job: &Job<'_>,
    ) -> Result<Recorded> {
        if images.len() != out.len() {
            return Err(Error::Invariant(
                "encoder images and products must align".into(),
            ));
        }
        let mut result = Recorded {
            encoded: Encoded {
                shapes: vec![Vec::new(); images.len()],
                ..Encoded::default()
            },
            returning: Vec::new(),
            method: self.methods.encode.name().to_owned(),
        };
        if images.is_empty() {
            return Ok(result);
        }
        match (&self.info, &self.packed) {
            (EncoderInfo::Vision(_), Some(_)) => {
                self.encode_packed(images, out, &mut result.encoded, job)?
            }
            (EncoderInfo::Vision(_) | EncoderInfo::Latent(_), _) => {
                let groups = match self.info {
                    EncoderInfo::Vision(vision) => {
                        // Each call stays within the rows its projections
                        // were prepared for.
                        let budget = unpacked_rows(&vision, &self.config);
                        shape_groups(images)
                            .into_iter()
                            .flat_map(|group| {
                                chunk_rows(group, budget, |index| {
                                    image_rows(images[index], vision.patch_size)
                                })
                            })
                            .collect()
                    }
                    _ => shape_groups(images),
                };
                for group in groups {
                    // Each image's product follows the images as a tensor
                    // of its own.
                    let mut arguments = vec![
                        Argument::Handle(self.methods.runner.clone()),
                        Argument::Handles(
                            group
                                .iter()
                                .map(|&index| images[index].value.clone())
                                .collect(),
                        ),
                    ];
                    arguments.extend(
                        group
                            .iter()
                            .map(|&index| Argument::Tensor(out[index].clone())),
                    );
                    job.call_returning(MethodCall::new(&self.methods.encode, arguments));
                    result.returning.push(group);
                    result.encoded.record(GraphMode::Eager);
                }
            }
            (EncoderInfo::Text(_) | EncoderInfo::Condition, _) => {
                return Err(Error::Invalid(format!("{} encodes no images", self.owner)));
            }
        }
        Ok(result)
    }

    /// The arguments of the Python half's `encode` of `slots` packed slots.
    fn packed_arguments(&self, slots: usize) -> Result<Vec<Argument>> {
        let (EncoderInfo::Vision(vision), Some(packed)) = (&self.info, &self.packed) else {
            return Err(Error::Invariant("encoder packs no images".into()));
        };
        let slots = slots as i64;
        let patches = i64::from(vision.max_patches.unwrap_or_default());
        let tokens = i64::from(vision.slot_tokens().unwrap_or_default());
        Ok(vec![
            Argument::Handle(self.methods.runner.clone()),
            Argument::Tensor(packed.pixels.prefix(slots * patches)?),
            Argument::Tensor(packed.grids.prefix(2 * slots)?),
            Argument::Tensor(packed.features.prefix(slots * tokens)?),
        ])
    }

    fn encode_packed(
        &mut self,
        images: &[&PreparedImage],
        out: &[StorageView],
        result: &mut Encoded,
        job: &Job<'_>,
    ) -> Result<()> {
        let (EncoderInfo::Vision(vision), Some(packed)) = (&self.info, &self.packed) else {
            return Err(Error::Invariant("encoder packs no images".into()));
        };
        let vision = *vision;
        let (capacity, pixels, grids, features) = (
            packed.capacity,
            packed.pixels.clone(),
            packed.grids.clone(),
            packed.features.clone(),
        );
        let max_patches = vision.max_patches.unwrap_or_default();
        let slot_tokens = u64::from(vision.slot_tokens().unwrap_or_default());
        let row_bytes = u64::from(vision.output_size) * vision.output_dtype.bytes() as u64;
        let shapes = images
            .iter()
            .map(|image| {
                image
                    .grid
                    .filter(|_| image.shape.len() == 2)
                    .ok_or_else(|| {
                        Error::Invalid(
                            "packed vision images require patch rows covering their grids".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let stream = self.stream.clone();
        let issue = job.on(stream.as_ref());
        let mut offset = 0;
        for chunk in shapes.chunks(capacity) {
            let slots = chunk.len();
            let layout = packed_grids(chunk, slots, max_patches, vision.downsample)?;
            grids
                .prefix(2 * slots as i64)?
                .upload(grid_bytes(&layout).as_slice(), issue)?;

            // FIXME: prepared images are Python-held, so their patch rows
            // reach the slots through one call on the Python thread.
            let mut pack = vec![
                Argument::Handle(self.methods.runner.clone()),
                Argument::Tensor(pixels.clone()),
            ];
            pack.extend(
                images[offset..offset + slots]
                    .iter()
                    .map(|image| Argument::Handle(image.value.clone())),
            );
            job.call(MethodCall::new(&self.methods.pack_images, pack));

            let arguments = self.packed_arguments(slots)?;
            let mode = match (self.graph(EncoderGraph::Packed { slots })?, &stream) {
                (Some(graph), Some(_)) => {
                    issue.launch(graph)?;
                    GraphMode::Replay
                }
                _ => {
                    job.call(MethodCall::new(&self.methods.encode, arguments));
                    GraphMode::Eager
                }
            };
            if let Some(table) = &mut self.graphs {
                table.record(mode);
            }

            // Slot `i` holds image `i`'s features in its leading rows; the
            // features and the product share their dtype, so the rows are
            // copied as they are.
            let mut rows_out = CopyBatch::new(SourceOrder::Stream);
            for (index, &[rows, columns]) in chunk.iter().enumerate() {
                let count =
                    u64::from(rows) * u64::from(columns) / u64::from(vision.downsample).pow(2);
                let target = out[offset + index].address();
                let source = features.address() + index as u64 * slot_tokens * row_bytes;
                rows_out.push(target, source, (count * row_bytes) as usize);
                result.shapes[offset + index] = vec![count as i64, i64::from(vision.output_size)];
            }
            // SAFETY: each image's rows lie in its slot's features and its
            // target is the product the caller reserved, both alive until
            // the batch commits; the copies follow the encode in the
            // issuer's order, on the stream or in host memory.
            unsafe { issue.copy(&rows_out) }?;
            result.record(mode);
            offset += slots;
        }
        Ok(())
    }

    /// Encode vision blocks whose patch rows are already packed, such as a
    /// media reader's `vision_pixels` product.
    ///
    /// `rows` holds every block's patch rows back to back, in block order;
    /// `grids` holds each block's `(time, height, width)` patch grid, whose
    /// product is the block's row count. The blocks' features fill `out`'s
    /// rows in block order. Each block is one sample of one numerical call,
    /// so a block's features do not depend on the other blocks.
    pub(crate) fn encode_patches(
        &mut self,
        rows: &StorageView,
        grids: &[[u32; 3]],
        out: &StorageView,
        job: &Job<'_>,
    ) -> Result<()> {
        if !matches!(self.info, EncoderInfo::Vision(_)) {
            return Err(Error::Invalid(format!(
                "{} encodes no vision blocks",
                self.owner
            )));
        }
        job.call(MethodCall::new(
            &self.methods.encode,
            vec![
                Argument::Handle(self.methods.runner.clone()),
                Argument::Tensor(rows.clone()),
                Argument::Tensor(out.clone()),
                Argument::Integers(
                    grids
                        .iter()
                        .flatten()
                        .map(|&extent| i64::from(extent))
                        .collect(),
                ),
            ],
        ));
        Ok(())
    }

    /// The arguments of the Python half's `encode` of a prompt padded to
    /// `capacity`, without vision tokens.
    fn text_arguments(&self, capacity: usize) -> Result<Vec<Argument>> {
        let Some(text) = &self.text else {
            return Err(Error::Invalid(format!("{} encodes no text", self.owner)));
        };
        Ok(vec![
            Argument::Handle(self.methods.runner.clone()),
            Argument::Tensor(text.tokens.prefix(capacity as i64)?),
            Argument::Tensor(text.features.prefix(capacity as i64)?),
        ])
    }

    /// Record the encoding of one prompt of `tokens` ids into `out`, its
    /// `[tokens, hidden]` feature rows in the product's dtype, as operations
    /// of `job` on the encoder's stream: the token upload, the launch of
    /// the prompt capacity's graph or the eager call, and the copy of the
    /// prompt's rows. Returns the mode the forward runs in.
    ///
    /// `vision` carries the prompt's vision tokens, for an encoder that
    /// places them. A pipeline stage that emits no features writes nothing.
    pub(crate) fn encode_text(
        &mut self,
        tokens: &[i64],
        out: &StorageView,
        vision: Option<PromptVision<'_>>,
        job: &Job<'_>,
    ) -> Result<GraphMode> {
        let (encode, position_ids) = (&self.methods.encode, &self.methods.position_ids);
        let (Some(text), Some(stream)) = (&self.text, &self.stream) else {
            return Err(Error::Invalid(format!("{} encodes no text", self.owner)));
        };
        let (encode, position_ids, stream) = (encode.clone(), position_ids.clone(), stream.clone());
        let issue = job.on(Some(&stream));
        let capacity = text_capacity(&self.text_capacities, tokens.len())?;
        if capacity as i64 > text.tokens.shape()[0] {
            return Err(Error::Invalid(
                "text encoder input exceeds its token capacity".into(),
            ));
        }
        // Causal attention keeps the zero padding out of the prompt's rows.
        let mut padded = tokens.to_vec();
        padded.resize(capacity, 0);
        let bytes: Vec<u8> = padded
            .iter()
            .flat_map(|token| token.to_ne_bytes())
            .collect();
        text.tokens.upload(&bytes, issue)?;
        let (features, final_stage) = (text.features.clone(), text.final_stage);

        let runner = Argument::Handle(self.methods.runner.clone());
        let mut arguments = self.text_arguments(capacity)?;
        let mode = match vision {
            Some(PromptVision {
                image_grids,
                video_grids,
                rows,
            }) => {
                let positions = self
                    .text
                    .as_ref()
                    .and_then(|text| text.positions.as_ref())
                    .ok_or_else(|| {
                        Error::Invalid(format!("{} places no vision tokens", self.owner))
                    })?;
                let grids = |grids: &[[u32; 3]]| {
                    Argument::Integers(
                        grids
                            .iter()
                            .flatten()
                            .map(|&extent| i64::from(extent))
                            .collect(),
                    )
                };
                job.call(MethodCall::new(
                    &position_ids,
                    vec![
                        runner,
                        Argument::Integers(padded.clone()),
                        grids(image_grids),
                        grids(video_grids),
                        Argument::Tensor(positions.clone()),
                    ],
                ));
                arguments.push(Argument::Tensor(positions.clone()));
                arguments.push(Argument::Tensor(rows.clone()));
                job.call(MethodCall::new(&encode, arguments));
                GraphMode::Eager
            }
            // Without configured capacities a prompt is encoded at its
            // exact length, which no graph serves.
            None if self.text_capacities.is_empty() => {
                job.call(MethodCall::new(&encode, arguments));
                GraphMode::Eager
            }
            None => match self.graph(EncoderGraph::Text { capacity })? {
                Some(graph) => {
                    issue.launch(graph)?;
                    GraphMode::Replay
                }
                None => {
                    job.call(MethodCall::new(&encode, arguments));
                    GraphMode::Eager
                }
            },
        };
        if let Some(table) = &mut self.graphs {
            table.record(mode);
        }
        if final_stage {
            // The features and the product share the model dtype, so the
            // prompt's rows are copied as they are.
            let count = out.numel() * out.dtype().bytes() as u64;
            let mut rows = CopyBatch::new(SourceOrder::Stream);
            rows.push(out.address(), features.address(), count as usize);
            // SAFETY: the features hold at least the prompt's rows, the
            // caller bound `out` for them, both outlive the batch, and the
            // copy follows the encode on the stream.
            unsafe { issue.copy(&rows) }?;
        }
        Ok(mode)
    }

    /// The captured graph of `key`, or `None` for eager execution: capture
    /// is disabled or no capacity was configured. After seal a missing
    /// graph is an error.
    fn graph(&mut self, key: EncoderGraph) -> Result<Option<&crate::cuda_graph::CUDAGraph>> {
        let capturing = self.capturing;
        let Some(table) = self.graphs.as_mut().filter(|_| capturing) else {
            return Ok(None);
        };
        Ok(match table.lookup(&key, None)? {
            Lookup::Replay(graph) => Some(graph),
            Lookup::Capture => None,
        })
    }

    /// Warm and, with `pool`, capture every graph this encoder launches:
    /// packed slot counts `capacity..=1` and text capacities, largest
    /// first. `check` runs after each capture (the graph storage budget);
    /// a failed check releases that graph. Without a pool the same
    /// forwards run once on the Python thread, so lazy initialization
    /// precedes serving.
    ///
    /// FIXME: startup issues this on the runner's stream at once and drains
    /// it, from the thread that starts the rank; as a job of the lane it
    /// would be recorded like the serving forward.
    pub fn capture(
        &mut self,
        pool: Option<&Arc<GraphPool<Handle>>>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        if let Some(pool) = pool
            && Device::Cuda(pool.device) != self.device
        {
            return Err(Error::Invariant(
                "an encoder captures into a pool of its device".into(),
            ));
        }
        // The forwards to capture, largest first, each with its arguments.
        let mut forwards = Vec::new();
        match (&self.info, &self.packed, &self.text) {
            (EncoderInfo::Vision(vision), Some(packed), _) => {
                for slots in (1..=packed.capacity).rev() {
                    let layout = packed_grids(
                        &[],
                        slots,
                        vision.max_patches.unwrap_or_default(),
                        vision.downsample,
                    )?;
                    forwards.push((
                        EncoderGraph::Packed { slots },
                        Some(grid_bytes(&layout)),
                        self.packed_arguments(slots)?,
                    ));
                }
            }
            (EncoderInfo::Text(_), _, Some(text)) => {
                text.tokens.zero(self.stream.as_ref())?;
                for &capacity in self.text_capacities.iter().rev() {
                    forwards.push((
                        EncoderGraph::Text { capacity },
                        None,
                        self.text_arguments(capacity)?,
                    ));
                }
            }
            _ => {}
        }
        for (key, grids, arguments) in forwards {
            if let (EncoderGraph::Packed { slots }, Some(grids), Some(packed)) =
                (key, &grids, &self.packed)
            {
                packed
                    .grids
                    .prefix(2 * slots as i64)?
                    .upload(grids, self.stream.as_ref())?;
            }
            // An eager forward first, so kernels compile outside the
            // capture.
            MethodCall::new(&self.methods.encode, arguments.clone()).invoke()?;
            let (Some(pool), Some(stream), Device::Cuda(device)) =
                (pool, &self.stream, self.device)
            else {
                continue;
            };
            let capture = &self.methods.capture;
            let encode = MethodCall::new(&self.methods.encode, arguments);
            let graph = capture.capture(&pool.value, stream.raw(), device, || {
                encode.invoke().map(drop)
            })?;
            if let Err(error) = check() {
                if let Err(cleanup) = capture.release.invoke(&[graph.handle().view()]) {
                    tracing::warn!(target: "uniserve_worker.graphs", "{}: releasing a refused graph failed: {cleanup}", self.owner);
                }
                return Err(error);
            }
            self.graphs
                .as_mut()
                .ok_or_else(|| Error::Invariant("encoder has no graph table".into()))?
                .insert(key, None, graph)?;
        }
        self.capturing = pool.is_some();
        Ok(())
    }

    /// Refuse further captures: serving replays what startup captured.
    pub fn seal(&mut self) {
        if let Some(table) = &mut self.graphs {
            table.seal();
        }
    }

    /// Release the runner's graphs, then what it owns: the Python runners
    /// of condition encoders and the execution context it owns. The
    /// caller drained its stream. An aborted close keeps everything until
    /// the process exits.
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        let mut result = self
            .graphs
            .as_mut()
            .map_or(Ok(()), |table| table.close(aborted));
        if let Some(conditions) = &mut self.conditions {
            result = result.and(conditions.close(aborted));
        }
        if let Some(context) = &self.context {
            let flag = Any::from(aborted);
            result = result.and(
                context
                    .close
                    .invoke(&[self.methods.runner.view(), (&flag).into()]),
            );
        }
        result
    }
}

/// Split a group into consecutive chunks of at most `budget` rows (a
/// larger member takes a chunk of its own).
fn chunk_rows(group: Vec<usize>, budget: usize, rows: impl Fn(usize) -> usize) -> Vec<Vec<usize>> {
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut used = 0;
    for index in group {
        let count = rows(index);
        match chunks.last_mut() {
            Some(chunk) if used + count <= budget => {
                chunk.push(index);
                used += count;
            }
            _ => {
                chunks.push(vec![index]);
                used = count;
            }
        }
    }
    chunks
}

/// Image indices grouped by exact pixel shape, groups in first-appearance
/// order.
fn shape_groups(images: &[&PreparedImage]) -> Vec<Vec<usize>> {
    let mut groups: Vec<(Vec<i64>, Vec<usize>)> = Vec::new();
    for (index, image) in images.iter().enumerate() {
        match groups.iter_mut().find(|(shape, _)| *shape == image.shape) {
            Some((_, members)) => members.push(index),
            None => groups.push((image.shape.clone(), vec![index])),
        }
    }
    groups.into_iter().map(|(_, members)| members).collect()
}

/// Native-endian int64 words of a grid layout, as the device reads them.
fn grid_bytes(layout: &[[i64; 2]]) -> Vec<u8> {
    layout
        .iter()
        .flatten()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// The condition latent encoders of a video model: the Python runners of
/// its video encoder and of its audio encoder, one per input shapes.
struct Conditions {
    video: RunnerClass,
    audio: RunnerClass,
    loaded: Handle,
    /// Frames of each visual unit of a condition of some frame count.
    unit_frames: Function,
    /// The runner of each call's input shapes, created at its first call.
    video_runners: HashMap<Vec<i64>, Methods>,
    audio_runners: HashMap<Vec<i64>, Methods>,
}

impl Conditions {
    fn new(video: RunnerSetup<'_>, audio: RunnerSetup<'_>) -> Result<Self> {
        Ok(Self {
            unit_frames: video.interface.import(&format!("{ENCODER}.unit_frames"))?,
            loaded: video.loaded.clone(),
            video: RunnerClass::new(video.interface, ENCODER, "encode", video.arguments())?,
            audio: RunnerClass::new(audio.interface, ENCODER, "encode", audio.arguments())?,
            video_runners: HashMap::new(),
            audio_runners: HashMap::new(),
        })
    }

    /// Encode visual units, each `[pixel_offset, frames, width, height,
    /// unit]`, into consecutive rows of `out`.
    fn encode_video(
        &mut self,
        pixels: &StorageView,
        units: &[[i64; 5]],
        out: &StorageView,
        job: &Job<'_>,
    ) -> Result<()> {
        // The call's input shapes: each unit's frames at its raster.
        let mut shapes = Vec::with_capacity(units.len() * 4);
        for &[_, frames, width, height, unit] in units {
            let value = self
                .unit_frames
                .call_values(&[self.loaded.to_any(), Any::from(frames)])?;
            let lengths = crate::ffi::integers(value)
                .map_err(|error| Error::numerical(self.unit_frames.name(), &error))?;
            let length = lengths.get(unit as usize).copied().ok_or_else(|| {
                Error::Invalid("a condition unit lies outside its condition".into())
            })?;
            shapes.extend([length, height, width, 3]);
        }
        // FIXME: the runner of shapes seen for the first time is constructed
        // here, where the forward is recorded, not by the job: the recorded
        // call takes the runner the construction returns.
        if !self.video_runners.contains_key(&shapes) {
            let runner = self
                .video
                .construct(vec![Argument::Integers(shapes.clone())])?;
            self.video_runners.insert(shapes.clone(), runner);
        }
        let runner = &self.video_runners[&shapes];
        job.call(MethodCall::new(
            &self.video.forward,
            vec![
                Argument::Handle(runner.runner().clone()),
                Argument::Tensor(pixels.clone()),
                Argument::Tensor(out.clone()),
                Argument::Integers(units.iter().flatten().copied().collect()),
            ],
        ));
        Ok(())
    }

    /// Encode audio tracks of `lengths` samples into consecutive rows of
    /// `out`.
    fn encode_audio(
        &mut self,
        samples: &StorageView,
        lengths: &[u64],
        out: &StorageView,
        job: &Job<'_>,
    ) -> Result<()> {
        let shapes: Vec<i64> = lengths
            .iter()
            .flat_map(|&length| [length as i64, 2])
            .collect();
        // FIXME: as for a video runner, a runner of new shapes is
        // constructed where the forward is recorded.
        if !self.audio_runners.contains_key(&shapes) {
            let runner = self
                .audio
                .construct(vec![Argument::Integers(shapes.clone())])?;
            self.audio_runners.insert(shapes.clone(), runner);
        }
        let runner = &self.audio_runners[&shapes];
        job.call(MethodCall::new(
            &self.audio.forward,
            vec![
                Argument::Handle(runner.runner().clone()),
                Argument::Tensor(samples.clone()),
                Argument::Tensor(out.clone()),
                Argument::Integers(lengths.iter().map(|&length| length as i64).collect()),
            ],
        ));
        Ok(())
    }

    pub(crate) fn kernels(&self) -> Result<Vec<KernelRecord>> {
        let mut records = self.video.kernels(self.video_runners.values())?;
        records.extend(self.audio.kernels(self.audio_runners.values())?);
        Ok(records)
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let video = self.video.close(aborted, self.video_runners.values());
        video.and(self.audio.close(aborted, self.audio_runners.values()))
    }
}

/// What a video model's text encoder component is bound with.
pub(crate) struct TextSetup<'a> {
    pub(crate) interface: &'a crate::model::interface::Interface,
    pub(crate) loaded: &'a Handle,
    pub(crate) component: &'a str,
    /// The component's root module path.
    pub(crate) path: &'a str,
    pub(crate) device: i32,
    pub(crate) stream: Stream,
    /// The stream's shared work areas, a Python `Scratch`.
    pub(crate) scratch: Handle,
    pub(crate) attention: String,
    /// Prompt capacities, the longest admitted prompt and, as the row
    /// budget, the most vision patch rows one request carries.
    pub(crate) config: EncoderConfig,
    /// Whether the component also encodes condition media with its vision
    /// tower.
    pub(crate) vision: bool,
    /// The model dtype the encoders compute in.
    pub(crate) dtype: DType,
}

impl EncoderRunner {
    /// The runners of a video model's text encoder component: the prompt
    /// encoder, which owns the execution context, and, for conditioned
    /// tasks, the vision tower bound with it.
    pub(crate) fn text(setup: TextSetup<'_>) -> Result<(Self, Option<Self>)> {
        let TextSetup {
            interface,
            loaded,
            component,
            path,
            device,
            stream,
            scratch,
            attention,
            config,
            vision,
            dtype,
        } = setup;
        let text_info = EncoderInfo::describe(interface, loaded, path, dtype)?;
        let vision_info = if vision {
            Some(EncoderInfo::describe(
                interface,
                loaded,
                &format!("{path}.vision"),
                dtype,
            )?)
        } else {
            None
        };
        // One vision call encodes every vision block of a request. Their
        // patch rows together are within the row budget, and a block holds
        // at least one feature, so the budget also bounds the blocks.
        let config = match &vision_info {
            Some(EncoderInfo::Vision(tower)) => EncoderConfig {
                max_images: (config.row_budget / (tower.downsample as usize).pow(2)).max(1),
                ..config
            },
            _ => config,
        };
        // The context serves both encoders, so its operators are prepared
        // for the larger of their workloads: a prompt padded to the largest
        // capacity, and a request's vision blocks.
        let size = [
            text_info.context_size(&config),
            vision_info.and_then(|info| info.context_size(&config)),
        ]
        .into_iter()
        .flatten()
        .reduce(|left, right| TextSize {
            tokens: left.tokens.max(right.tokens),
            rows: left.rows.max(right.rows),
        })
        .ok_or_else(|| Error::Invariant("a text encoder declares its size".into()))?;
        let method = |name: &str| interface.import(&format!("{ENCODER}.{name}"));
        let runner = Handle::new(interface.import(ENCODER)?.call_values(&[
            loaded.to_any(),
            Any::from(tvm_ffi::String::from(path)),
            Any::from(stream.raw() as i64),
            Any::from(tvm_ffi::String::from(attention.as_str())),
            scratch.to_any(),
            Any::from(integer_list([size.tokens as i64, size.rows as i64])),
        ])?)?;
        let device = Device::Cuda(device);
        let vision = vision_info
            .map(|info| {
                let tower = Handle::new(method("tower")?.call_values(&[
                    runner.to_any(),
                    Any::from(tvm_ffi::String::from("vision")),
                ])?)?;
                EncoderRunner::over(
                    interface,
                    component,
                    device,
                    tower,
                    "vision",
                    info,
                    &config,
                    stream.clone(),
                )
            })
            .transpose()?;
        let mut text = EncoderRunner::over(
            interface, component, device, runner, "", text_info, &config, stream,
        )?;
        text.context = Some(OwnedContext {
            kernels: method("kernels")?,
            close: method("close")?,
        });
        Ok((text, vision))
    }

    /// The runner of a video model's condition latent encoders. `video`
    /// and `audio` name the encoders' module paths; the other fields of
    /// the setups agree.
    pub(crate) fn conditions(video: RunnerSetup<'_>, audio: RunnerSetup<'_>) -> Result<Self> {
        let interface = video.interface;
        Ok(Self {
            owner: video.path.to_owned(),
            info: EncoderInfo::Condition,
            device: Device::Cuda(video.device),
            stream: Some(video.stream.clone()),
            methods: EncoderMethods::new(interface, Handle::none())?,
            packed: None,
            text: None,
            text_capacities: Vec::new(),
            config: EncoderConfig::default(),
            graphs: None,
            capturing: false,
            context: None,
            conditions: Some(Box::new(Conditions::new(video, audio)?)),
        })
    }

    fn condition_encoders(&mut self) -> Result<&mut Conditions> {
        self.conditions
            .as_deref_mut()
            .ok_or_else(|| Error::Invalid(format!("{} encodes no conditions", self.owner)))
    }

    /// Encode a condition's visual units, each `[pixel_offset, frames,
    /// width, height, unit]`, into consecutive rows of `out`.
    pub(crate) fn encode_condition_units(
        &mut self,
        pixels: &StorageView,
        units: &[[i64; 5]],
        out: &StorageView,
        job: &Job<'_>,
    ) -> Result<()> {
        self.condition_encoders()?
            .encode_video(pixels, units, out, job)
    }

    /// Encode condition audio tracks of `lengths` samples into consecutive
    /// rows of `out`.
    pub(crate) fn encode_condition_tracks(
        &mut self,
        samples: &StorageView,
        lengths: &[u64],
        out: &StorageView,
        job: &Job<'_>,
    ) -> Result<()> {
        self.condition_encoders()?
            .encode_audio(samples, lengths, out, job)
    }

    /// The kernels the operators of the runner's execution context or of the
    /// condition encoders selected.
    pub(crate) fn kernels(&self) -> Result<Vec<KernelRecord>> {
        match (&self.context, &self.conditions) {
            (Some(context), _) => kernel_records(&context.kernels, &self.methods.runner, ENCODER),
            (_, Some(conditions)) => conditions.kernels(),
            _ => Ok(Vec::new()),
        }
    }
}

/// The patch grids of a request's vision blocks, images and videos apart,
/// each in request order as the text encoder places them.
fn vision_grids(conditions: &[VideoCondition]) -> (Vec<[u32; 3]>, Vec<[u32; 3]>) {
    let mut images = Vec::new();
    let mut videos = Vec::new();
    for condition in conditions {
        let Some(vision) = &condition.vision else {
            continue;
        };
        let grid = [vision.grid.t, vision.grid.h, vision.grid.w];
        match condition.media {
            uniserve_core::ConditionMedia::Video { .. } => videos.push(grid),
            _ => images.push(grid),
        }
    }
    (images, videos)
}

/// Encode a request's prompt into its `conditioning` rows, splicing the
/// vision features of its condition media in at their placeholders when the
/// call reads them. Every rank of the text encoder computes the features
/// and publishes its copy.
pub(crate) fn encode_text(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job<'_>,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.outputs.len() != 1 || call.inputs.len() > 1 {
        return Err(Error::Invalid(
            "text encoding reads at most one vision product and writes its conditioning".into(),
        ));
    }
    let (tokens, conditions) = {
        let request = parts.video.request(slot, call.request_key)?;
        (Arc::clone(&request.tokens), Arc::clone(&request.conditions))
    };
    if tokens.is_empty() {
        return Err(Error::Invalid(
            "text conditioning requires admitted prompt tokens".into(),
        ));
    }
    let width = row_width(&call.outputs[0])?;
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let visual = call
        .inputs
        .first()
        .map(|input| {
            batch.inputs.read(
                parts.owners().store,
                input,
                call.call_id,
                batch.device,
                issue,
            )
        })
        .transpose()?;
    let write = bind_rows(
        parts,
        batch,
        index,
        &call.outputs[0],
        vec![tokens.len() as u64, width],
        placement,
    )?;

    let text = &mut parts
        .encoders
        .text
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no text encoder".into()))?
        .runner;
    let (image_grids, video_grids) = vision_grids(&conditions);
    let vision = visual.as_ref().map(|read| PromptVision {
        image_grids: &image_grids,
        video_grids: &video_grids,
        rows: read.value(),
    });
    let mode = text.encode_text(&tokens, &write.view(), vision, job)?;
    batch.record(mode);
    publish(
        parts,
        batch,
        call,
        index,
        &write,
        &[tokens.len() as u64, width],
        issue,
    )
}

/// Encode the vision blocks of a request's condition media into one
/// `vision_features` row per vision placeholder, in request order.
pub(crate) fn encode_vision(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job<'_>,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "vision encoding requires one input and one output".into(),
        ));
    }
    let conditions = Arc::clone(&parts.video.request(slot, call.request_key)?.conditions);
    let blocks: Vec<([u32; 3], u64)> = conditions
        .iter()
        .filter_map(|condition| condition.vision.as_ref())
        .map(|vision| {
            (
                [vision.grid.t, vision.grid.h, vision.grid.w],
                u64::from(vision.tokens),
            )
        })
        .collect();
    if blocks.is_empty() {
        return Err(Error::Invalid(
            "vision encoding requires a vision block".into(),
        ));
    }
    let width = row_width(&call.outputs[0])?;
    let tokens: u64 = blocks.iter().map(|block| block.1).sum();
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let patches = batch.inputs.read(
        parts.owners().store,
        &call.inputs[0],
        call.call_id,
        batch.device,
        issue,
    )?;
    let write = bind_rows(
        parts,
        batch,
        index,
        &call.outputs[0],
        vec![tokens, width],
        placement,
    )?;

    let vision = parts
        .encoders
        .vision
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank's text encoder reads no vision".into()))?;
    // Each block's features are consecutive rows of the product.
    let grids: Vec<[u32; 3]> = blocks.iter().map(|block| block.0).collect();
    vision.encode_patches(patches.value(), &grids, &write.view(), job)?;
    publish(parts, batch, call, index, &write, &[tokens, width], issue)
}

/// Encode a request's conditions into denoiser rows: a round of visual
/// units, this rank's of them, or every audio track.
pub(crate) fn encode_conditions(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job<'_>,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "condition encoding requires one input and one output".into(),
        ));
    }
    let name = product_name(parts.info, call, &call.outputs[0])?.to_owned();
    let conditions = Arc::clone(&parts.video.request(slot, call.request_key)?.conditions);
    if conditions.is_empty() {
        return Err(Error::Invalid(
            "condition encoding requires admitted conditions".into(),
        ));
    }
    let rank = parts.rank;
    let owners = parts.owners();
    let latent = parts
        .encoders
        .latent
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let width = row_width(&call.outputs[0])?;
    match name.as_str() {
        "condition_video_latents" => {
            let range = decode_range(plan, call)?;
            // Every visual unit as (condition, unit, rows), in encoding order.
            let all: Vec<(usize, u32, u64)> = conditions
                .iter()
                .enumerate()
                .flat_map(|(index, condition)| {
                    condition
                        .latent_units
                        .iter()
                        .enumerate()
                        .map(move |(unit, &rows)| (index, unit as u32, u64::from(rows)))
                })
                .collect();
            let (cursor, count) = (range.cursor as usize, range.max_units as usize);
            if count == 0 || cursor + count > all.len() {
                return Err(Error::Invalid(
                    "condition encoding exceeds the admitted units".into(),
                ));
            }
            let round = &all[cursor..cursor + count];
            let run = latent.units(rank, 0, count as u32)?;
            if run.is_empty() {
                return Err(Error::Invalid(
                    "condition encoding assigns this rank no unit".into(),
                ));
            }
            let rows: u64 = round.iter().map(|unit| unit.2).sum();
            let first: u64 = round[..run.start as usize].iter().map(|unit| unit.2).sum();
            let last: u64 = round[..run.end as usize].iter().map(|unit| unit.2).sum();
            let pixels = batch.inputs.read(
                owners.store,
                &call.inputs[0],
                call.call_id,
                batch.device,
                issue,
            )?;
            let mut offsets = Vec::new();
            let mut cursor_pixels = 0u64;
            for condition in conditions.iter() {
                offsets.push(cursor_pixels);
                cursor_pixels += condition.pixel_bytes() / 3;
            }
            let units: Vec<[i64; 5]> = round[run.start as usize..run.end as usize]
                .iter()
                .map(|&(condition, unit, _)| {
                    let (frames, canvas) = conditions[condition].media.pixels().unwrap_or((
                        0,
                        Canvas {
                            width: 0,
                            height: 0,
                        },
                    ));
                    [
                        offsets[condition] as i64,
                        i64::from(frames),
                        i64::from(canvas.width),
                        i64::from(canvas.height),
                        i64::from(unit),
                    ]
                })
                .collect();
            let write = bind_outputs(
                owners.store,
                &[OutputShape {
                    product: call.outputs[0].clone(),
                    shape: vec![rows, width],
                    region: Some(vec![first..last, 0..width]),
                }],
                batch.device,
                placement,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invariant("condition rows were not bound".into()))?;
            batch.calls[index].writes.push(write.clone());
            latent
                .runner
                .encode_condition_units(pixels.value(), &units, &write.view(), job)?;
            publish_bound(parts, batch, call, index, &write, false, issue)?;
        }
        "condition_audio_latents" => {
            // The audio call is one unit, the component's first rank's.
            let run = latent.units(rank, 0, 1)?;
            if run.is_empty() {
                return Ok(());
            }
            let lengths: Vec<u64> = conditions
                .iter()
                .filter_map(|condition| condition.media.audio().map(|track| track.samples))
                .collect();
            let rows: u64 = conditions
                .iter()
                .map(|condition| u64::from(condition.audio_rows))
                .sum();
            let samples = batch.inputs.read(
                owners.store,
                &call.inputs[0],
                call.call_id,
                batch.device,
                issue,
            )?;
            let write = bind_outputs(
                owners.store,
                &[OutputShape {
                    product: call.outputs[0].clone(),
                    shape: vec![rows, width],
                    region: None,
                }],
                batch.device,
                placement,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invariant("condition rows were not bound".into()))?;
            batch.calls[index].writes.push(write.clone());
            latent
                .runner
                .encode_condition_tracks(samples.value(), &lengths, &write.view(), job)?;
            publish_bound(parts, batch, call, index, &write, false, issue)?;
        }
        other => {
            return Err(Error::Invalid(format!(
                "latent encoding does not produce {other}"
            )));
        }
    }
    Ok(())
}

// Image vision and latent encoding batches (`VisionEncoding`,
// `LatentEncoding`): each active call encodes one image into its
// `encoder_output` feature product, an inline request image the host lane
// prepared before the batch launched, or a resident image product (feedback
// of a generated image), which the encoder prepares on its stream. Features
// are written in place into the product's reserved storage and published
// with the image's canvas dimensions (`Metadata::Feature`).

/// The tower an encoding kind runs.
pub(crate) fn tower(call: MediaCall) -> Result<Tower> {
    match call {
        MediaCall::VisionEncoding => Ok(Tower::Vit),
        MediaCall::LatentEncoding => Ok(Tower::Vae),
        other => Err(Error::Invalid(format!(
            "{} is not an image encoding",
            other.as_str()
        ))),
    }
}

/// The encoder serving a batch, with its device and stream.
pub(crate) fn image_placement(
    image: &ImageParts<'_>,
    plan: &BatchPlan,
) -> Result<(Option<i32>, Option<Stream>, usize)> {
    let first = plan
        .calls
        .first()
        .ok_or_else(|| Error::Invariant("an image batch has calls".into()))?;
    let runner = &image.encoders.image[image.encoders.route(&first.component, first.code)?];
    let stream = runner.stream().cloned();
    Ok((runner.device().cuda_index(), stream, 0))
}

impl ImageBatch<'_> {
    /// Encode every active call's image into its feature product.
    /// `prepared` holds the host-prepared inline images by call index.
    pub(crate) fn encode(
        &mut self,
        call: MediaCall,
        prepared: HashMap<usize, PreparedImage>,
    ) -> Result<ForwardStats> {
        // The forward records into the batch's job, taken for its length so
        // that the batch's owners stay borrowable beside it.
        let mut job = std::mem::take(&mut self.job);
        let stats = self.record_encode(&mut job, call, prepared);
        self.job = job;
        stats
    }

    /// Record the encoding. Its jobs: the preparation of the resident
    /// images and then the read of their facts, when the batch has any;
    /// the encode calls; and, after an unpacked encoder's calls returned
    /// the shapes they wrote, the features' fences and exports.
    fn record_encode(
        &mut self,
        job: &mut Job<'static>,
        call: MediaCall,
        mut prepared: HashMap<usize, PreparedImage>,
    ) -> Result<ForwardStats> {
        let tower = tower(call)?;
        let plan = self.plan;
        let active = self.active();
        let Some(&first) = active.first() else {
            return Ok(ForwardStats::default());
        };
        let route = self
            .parts
            .encoders
            .route(&plan.calls[first].component, CallKind::Media(call))?;
        let target = self.target();
        let stream = self.stream.clone();

        // Every image, inline or resident, prepared for the encoder. A
        // resident image is prepared by a call of the job.
        let mut images: Vec<Option<PreparedImage>> = Vec::with_capacity(active.len());
        let mut resident = Vec::new();
        for (position, &index) in active.iter().enumerate() {
            let call = &plan.calls[index];
            if call.encoder_output.is_none() {
                return Err(Error::Invalid(
                    "image encoding requires an encoder output".into(),
                ));
            }
            match (call.input_image.is_some(), &call.image_input) {
                (true, None) => {
                    images.push(Some(prepared.remove(&index).ok_or_else(|| {
                        Error::Invariant("an inline image was not prepared".into())
                    })?))
                }
                (false, Some(reference)) => {
                    let read = self.storage.tensors.consume(
                        reference,
                        call.call_id,
                        Some(target),
                        job.on(stream.as_ref()),
                    )?;
                    let range = match read.metadata() {
                        Some(Metadata::Image {
                            value_range: Some(range),
                            ..
                        }) => range,
                        _ => {
                            self.reads.push(read);
                            return Err(Error::Invalid(
                                "resident image encoding requires an image value range".into(),
                            ));
                        }
                    };
                    let runner = &self.parts.encoders.image[route];
                    let value = read.value().clone();
                    self.reads.push(read);
                    runner.prepare_resident(&value, tower, range == ValueRange::SignedUnit, job)?;
                    resident.push(position);
                    images.push(None);
                }
                _ => {
                    return Err(Error::Invalid(
                        "image encoding requires exactly one inline or resident image".into(),
                    ));
                }
            }
        }
        if !resident.is_empty() {
            // The prepared images come back with the job that prepared
            // them; their facts with the job that reads them.
            let runner = &self.parts.encoders.image[route];
            let mut completion = submit(&self.lanes, self.kind, job)?;
            let handles = resident
                .iter()
                .map(|_| completion.handle())
                .collect::<Result<Vec<_>>>()?;
            for handle in &handles {
                runner.image_facts(handle, job);
            }
            let mut completion = submit(&self.lanes, self.kind, job)?;
            for (&position, handle) in resident.iter().zip(handles) {
                images[position] = Some(runner.prepared(handle, &completion.returned()?)?);
            }
        }
        let images: Vec<PreparedImage> = images
            .into_iter()
            .map(|image| image.ok_or_else(|| Error::Invariant("an image was not prepared".into())))
            .collect::<Result<_>>()?;

        // The feature products, reserved in their scheduler allocations.
        let allocations = allocations(plan);
        let bindings: Vec<_> = active
            .iter()
            .filter_map(|&index| plan.calls[index].encoder_output.clone())
            .map(|reference| (reference, target))
            .collect();
        let writes = self.storage.tensors.reserve_features(
            &bindings,
            &allocations,
            OutputPlacement::default(),
        )?;
        self.writes.extend(writes.iter().cloned());
        let runner = &mut self.parts.encoders.image[route];
        let products: Vec<StorageView> = writes.iter().map(|write| write.view()).collect();
        let borrowed: Vec<&PreparedImage> = images.iter().collect();
        let recorded = runner.record(&borrowed, &products, job)?;

        // An unpacked encoder's calls return the shapes they wrote, which
        // the features are fenced and exported with: the next job's work.
        let encoded = if recorded.returns() {
            recorded.finish(&mut submit(&self.lanes, self.kind, job)?)?
        } else {
            recorded.finish(&mut Completion::default())?
        };

        let issue = job.on(stream.as_ref());
        for (((write, image), shape), &index) in
            writes.iter().zip(&images).zip(&encoded.shapes).zip(&active)
        {
            let shape: Vec<u64> = shape.iter().map(|&extent| extent as u64).collect();
            self.storage.tensors.produced(
                write,
                &shape,
                issue,
                Some(Metadata::Feature {
                    height: image.height,
                    width: image.width,
                }),
            )?;
            // Other ranks and workers import the feature from its export,
            // located when the batch's commit is applied.
            let export = Publications::issue_feature(
                &self.storage.tensors,
                &self.storage.transports,
                &plan.calls[index],
                write,
                self.reports,
                issue,
            )?;
            if let Some(export) = export {
                self.features.push((index, write.clone(), export));
            }
        }

        let mut stats = ForwardStats::default();
        stats.cuda_graph_replays += encoded.replays;
        if encoded.replays > 0 {
            *stats.mode_counts.entry("graph_replay".into()).or_default() += encoded.replays;
        }
        if encoded.eager > 0 {
            *stats.mode_counts.entry("eager".into()).or_default() += encoded.eager;
        }
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_grids_describe_images_and_slot_padding() -> Result<()> {
        // Two images in three slots of 64 patch rows, downsampled by 2:
        // each image's grid, then a 2-row grid over its slot's padding.
        let layout = packed_grids(&[[4, 8], [8, 8]], 3, 64, 2)?;
        assert_eq!(
            layout,
            vec![[4, 8], [2, 16], [8, 8], [0, 0], [0, 0], [2, 32]]
        );
        assert_eq!(packed_grids(&[], 1, 64, 2)?, vec![[0, 0], [2, 32]]);
        Ok(())
    }

    #[test]
    fn packed_grids_reject_unaligned_oversized_and_excess_images() {
        assert!(packed_grids(&[[3, 4]], 1, 64, 2).is_err());
        assert!(packed_grids(&[[10, 8]], 1, 64, 2).is_err());
        assert!(packed_grids(&[[2, 2], [2, 2]], 1, 64, 2).is_err());
        assert!(packed_grids(&[[0, 4]], 1, 64, 2).is_err());
    }

    #[test]
    fn prompts_take_the_smallest_holding_text_capacity() -> Result<()> {
        assert_eq!(text_capacity(&[256, 512, 1024], 300)?, 512);
        assert_eq!(text_capacity(&[256, 512, 1024], 256)?, 256);
        assert!(text_capacity(&[256, 512], 513).is_err());
        assert!(text_capacity(&[256], 0).is_err());
        // Without configured capacities a prompt keeps its own length.
        assert_eq!(text_capacity(&[], 37)?, 37);
        Ok(())
    }

    #[test]
    fn groups_split_into_chunks_within_their_row_budget() {
        let rows = [3, 4, 2, 9, 1];
        let chunks = chunk_rows(vec![0, 1, 2, 3, 4], 6, |index| rows[index]);
        assert_eq!(chunks, vec![vec![0], vec![1, 2], vec![3], vec![4]]);
    }

    #[test]
    fn unpacked_images_group_by_pixel_shape_in_appearance_order() {
        let image = |shape: &[i64]| PreparedImage {
            value: Handle::none(),
            shape: shape.to_vec(),
            grid: None,
            height: 1,
            width: 1,
        };
        let (a, b, c) = (
            image(&[3, 32, 32]),
            image(&[3, 16, 32]),
            image(&[3, 32, 32]),
        );
        assert_eq!(shape_groups(&[&a, &b, &c]), vec![vec![0, 2], vec![1]]);
    }

    #[test]
    fn a_packed_context_covers_every_slot_of_its_capacity() {
        let info = VisionEncoder {
            patch_size: 16,
            downsample: 3,
            output_size: 32,
            output_dtype: DType::BF16,
            max_patches: Some(2520),
            image_patches: 2520,
            row_width: 768,
            pixel_dtype: DType::F32,
        };
        assert_eq!(info.slot_tokens(), Some(280));
        // Two segments per slot.
        let config = EncoderConfig {
            max_images: 40,
            ..EncoderConfig::default()
        };
        assert_eq!(
            EncoderInfo::Vision(info).context_size(&config),
            Some(TextSize {
                tokens: 16 * 2520,
                rows: 32,
            })
        );
    }
}

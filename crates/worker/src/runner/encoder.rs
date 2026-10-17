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
//! `uniserve.runner.encoder.EncoderRunner`, constructed directly
//! with its stream's shared scratch ([`EncoderRunner::text`]): the text
//! encoder's half owns the execution context, its tower borrows it, and
//! `encode` takes the tensors of one forward as arguments.
//!
//! An image model's vision and latent encoders compute through the same
//! Python class, constructed over a bound component by
//! `EncoderRunner.bound` ([`EncoderRunner::image`]):
//!
//! | kind | module | methods |
//! | --- | --- | --- |
//! | [`EncoderInfo::Vision`] | `PatchEncoder` | `encode`, `prepare_resident`, `warm`; packed: `encode` over the slot tensors |
//! | [`EncoderInfo::Latent`] | `PatchAutoencoder` | `encode`, `prepare_resident`, `draw_noise`, `warm` |
//!
//! Such a runner receives its device's product arena once; an unpacked
//! `encode` writes its results into the regions of it the scheduler placed
//! them in ([`Placement`]), in the product's dtype, and returns the results'
//! shapes, which the caller publishes.
//!
//! **Images.** An image's facts (`EncoderRunner.image_facts`: its canvas,
//! patch grid and prepared pixel shape) are known before its pixels. Inline
//! request images are decoded and transformed on a host lane
//! ([`ImagePreparation::prepare`]) into a block of the runner's
//! [`PinnedPool`]; resident image products are transformed on the device by
//! `prepare_resident`, into the packed slot they are encoded from or into
//! the runner's resident pixels. Unpacked vision encoders encode
//! host-prepared images of one pixel shape per numerical call, in
//! first-appearance order, so an image's features do not depend on the
//! other images of its batch, and each resident image alone.
//!
//! **Posterior noise.** A latent encoder that samples its posterior reads
//! its noise: before each image's encode, `draw_noise` fills the runner's
//! noise buffer from the image's request seed and the request's latent
//! encodings before it.
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

mod image;
pub(crate) mod inputs;
mod packed;
pub(crate) mod runners;
mod start;
mod text;
mod video;
pub(crate) mod warmup;

use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_cuda::Stream;

use crate::config::Device;
use crate::cuda_graph::GraphPool;
use crate::cuda_graph::{GraphMode, GraphTable, Lookup};
use crate::ffi::{Backing, DType, Function, Handle, Interpreter, integer_list};
use crate::lane::{Capture, Completion, Job};
use crate::parallel::joins::GraphFunctions;
use crate::profiling::kernels::KernelRecord;
use crate::runner::ImageSetup;
use crate::runner::methods::{Argument, MethodCall, kernels_call, list_kernels};
use crate::storage::PinnedPool;
use crate::storage::memory::{Memory, StorageView};
use crate::{Error, Result};

pub(crate) use image::{EncodeResume, ImageFacts, ImagePixels, image_placement};
pub use image::{ImagePreparation, PreparedImage, Tower};
pub(crate) use packed::packed_grids;
use packed::{PackedVision, grid_bytes, packed_capacity};
pub(crate) use text::PromptVision;
use text::TextBuffers;
use video::Conditions;
pub(crate) use video::{
    ConditionSetup, TextSetup, accept_conditions, encode_conditions, encode_text, encode_vision,
    setup_conditions,
};

const ENCODER: &str = "uniserve.runner.encoder.EncoderRunner";

/// The largest packed vision slot count: it bounds retained activations
/// independently of the admitted batch size. Larger batches replay full
/// chunks and then their exact remaining slot count.
pub(crate) const MAX_PACKED_IMAGES: usize = 16;

/// The text workload an encoder's execution context is prepared for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextSize {
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
    /// Values of the largest image the `vit` transform prepares.
    pub image_elements: u64,
}

impl VisionEncoder {
    /// Feature rows of one packed slot.
    pub(crate) fn slot_tokens(&self) -> Option<u32> {
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
    /// The image policy's pixel dtype.
    pub pixel_dtype: DType,
    /// Values of the largest image the `vae` transform prepares.
    pub image_elements: u64,
    /// Values of the largest posterior noise the encoder draws; zero when
    /// it takes its posterior's mean.
    pub noise_elements: u64,
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
    pub fn describe(loaded: &Handle, path: &str, dtype: DType) -> Result<Self> {
        let path = Any::from(tvm_ffi::String::from(path));
        let method = |name: &str| Interpreter::import(&format!("{ENCODER}.{name}"));
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
        let pixel_dtype = || -> Result<DType> {
            let function = method("pixel_dtype")?;
            let name = function.call_as(&[loaded.view()])?;
            dtype_named(&function, name)
        };
        let count = |function: Function, arguments: &[Any]| -> Result<u64> {
            let arguments: Vec<_> = arguments.iter().map(Into::into).collect();
            let value: i64 = function.call_as(&arguments)?;
            u64::try_from(value)
                .map_err(|_| Error::Interface(format!("{} is out of range", function.name())))
        };
        let image_elements = |tower: Tower| -> Result<u64> {
            count(
                method("image_elements")?,
                &[
                    loaded.to_any(),
                    Any::from(tvm_ffi::String::from(tower.name())),
                ],
            )
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
                Self::Vision(VisionEncoder {
                    patch_size,
                    downsample,
                    output_size: extent("output_width")?,
                    output_dtype: output_dtype()?,
                    max_patches: (max_patches > 0).then_some(max_patches),
                    image_patches: extent("image_patches")?,
                    // A patch row holds the RGB values of one patch.
                    row_width: 3 * patch_size * patch_size,
                    pixel_dtype: pixel_dtype()?,
                    image_elements: image_elements(Tower::Vit)?,
                })
            }
            "latent" => Self::Latent(LatentEncoder {
                downsample: extent("downsample")?,
                width: extent("output_width")?,
                output_dtype: output_dtype()?,
                pixel_dtype: pixel_dtype()?,
                image_elements: image_elements(Tower::Vae)?,
                noise_elements: count(method("noise_elements")?, &[loaded.to_any(), path.clone()])?,
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
    pub(crate) fn context_size(&self, config: &EncoderConfig) -> Option<TextSize> {
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

/// Patch rows one unpacked vision call passes: the largest admitted image,
/// or the configured row budget when larger.
fn unpacked_rows(vision: &VisionEncoder, config: &EncoderConfig) -> usize {
    (vision.image_patches as usize)
        .max(config.row_budget)
        .max(1)
}

/// Patch rows an image passes through a vision encoder: its packed rows,
/// or its pixels' patches.
fn image_rows(facts: &ImageFacts, patch_size: u32) -> usize {
    match facts.shape[..] {
        [rows, _] => rows as usize,
        [_, height, width] => {
            (height as usize / patch_size as usize) * (width as usize / patch_size as usize)
        }
        _ => 0,
    }
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
pub(crate) enum EncoderGraph {
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

/// The Python half (`uniserve.runner.encoder.EncoderRunner`) and its
/// methods, each called with the runner first.
struct EncoderMethods {
    runner: Handle,
    encode: Function,
    position_ids: Function,
    prepare_resident: Function,
    draw_noise: Function,
    warm: Function,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
}

impl EncoderMethods {
    fn new(runner: Handle) -> Result<Self> {
        let method = |name: &str| Interpreter::import(&format!("{ENCODER}.{name}"));
        Ok(Self {
            runner,
            encode: method("encode")?,
            position_ids: method("position_ids")?,
            prepare_resident: method("prepare_resident")?,
            draw_noise: method("draw_noise")?,
            warm: method("warm")?,
            capture: GraphFunctions::import()?,
        })
    }
}

/// The methods of a Python half that owns its execution context.
struct OwnedContext {
    kernels: Function,
    close: Function,
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
    /// The host blocks inline images are prepared into, page-locked for
    /// the runner's device.
    pinned: Arc<PinnedPool>,
    /// An unpacked image encoder's device storage of one resident image's
    /// prepared pixels: the largest image its tower prepares, flat, in the
    /// pixel dtype. A packed encoder prepares them in their slot.
    resident: Option<StorageView>,
    /// A sampling latent encoder's posterior noise: the largest draw, flat,
    /// FP32.
    noise: Option<StorageView>,
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
    /// `EncoderInfo::context_size`.
    pub fn image(
        setup: &ImageSetup<'_>,
        info: EncoderInfo,
        config: &EncoderConfig,
    ) -> Result<Self> {
        let owner = setup.owner();
        let (device, stream) = (setup.device, setup.stream.clone());
        if device.is_cuda() != stream.is_some() {
            return Err(Error::Invariant(
                "an encoder runs on a stream exactly on a CUDA device".into(),
            ));
        }

        let (mut packed, mut resident, mut noise) = (None, None, None);
        match info {
            EncoderInfo::Vision(vision) => {
                if vision.max_patches.is_none() {
                    resident = Some(allocate(
                        device,
                        vision.pixel_dtype,
                        &[vision.image_elements as i64],
                    )?);
                }
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
                    // SAFETY: a new allocation the encoder keeps until it
                    // closes after its stream drained; its forwards run
                    // behind the fill on that stream.
                    unsafe { pixels.zero(stream.as_ref()) }?;
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
            EncoderInfo::Latent(latent) => {
                resident = Some(allocate(
                    device,
                    latent.pixel_dtype,
                    &[latent.image_elements as i64],
                )?);
                if latent.noise_elements > 0 {
                    noise = Some(allocate(
                        device,
                        DType::F32,
                        &[latent.noise_elements as i64],
                    )?);
                }
            }
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
        let method = |name: &str| Interpreter::import(&format!("{ENCODER}.{name}"));
        let runner = Handle::new(method("image")?.call_values(&arguments)?)?;
        let context = OwnedContext {
            kernels: method("kernels")?,
            close: method("close")?,
        };
        let methods = EncoderMethods::new(runner)?;
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
            pinned: PinnedPool::new(device.cuda()),
            resident,
            noise,
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
    /// (`uniserve.runner.encoder.EncoderRunner`). A text encoder's
    /// token, feature and position buffers are allocated here for the
    /// largest prompt capacity.
    fn over(
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
        let methods = EncoderMethods::new(runner)?;

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
                let final_stage = Interpreter::import(&format!("{ENCODER}.final_stage"))?
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
            pinned: PinnedPool::new(device.cuda()),
            resident: None,
            noise: None,
            text,
            text_capacities,
            config: config.clone(),
            capturing: false,
            context: None,
            conditions: None,
        })
    }

    /// The host blocks inline images for this runner are prepared into.
    pub fn pinned(&self) -> &Arc<PinnedPool> {
        &self.pinned
    }

    /// The dtype of the pixels an image encoder reads; `None` for a text
    /// or condition encoder.
    pub fn pixel_dtype(&self) -> Option<DType> {
        match self.info {
            EncoderInfo::Vision(vision) => Some(vision.pixel_dtype),
            EncoderInfo::Latent(latent) => Some(latent.pixel_dtype),
            EncoderInfo::Text(_) | EncoderInfo::Condition => None,
        }
    }

    pub(crate) fn device(&self) -> Device {
        self.device
    }

    /// The runner's stream; `None` on the host.
    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    /// The identity of the runner's graph pool owner: the address of its
    /// packed grids, which no other runner shares. `None` when unpacked.
    pub(crate) fn graph_key(&self) -> Option<usize> {
        self.packed
            .as_ref()
            .map(|packed| packed.grids.address() as usize)
    }

    /// Record into `job` the encoding of one black `side` x `side` image
    /// through the model's input transform, discarded: the startup warmup
    /// of an unpacked image encoder (packed vision encoders warm by
    /// capturing), one eager call of the Python half's `warm`. A latent
    /// encoder that samples its posterior encodes with zero noise.
    pub(crate) fn warm(&self, side: u32, job: &Job) {
        if let (EncoderInfo::Vision(_), None) | (EncoderInfo::Latent(_), _) =
            (&self.info, &self.packed)
        {
            job.call(MethodCall::new(
                &self.methods.warm,
                vec![
                    Argument::Handle(self.methods.runner.clone()),
                    Argument::Integer(i64::from(side)),
                ],
            ));
        }
    }

    /// Encode images, writing image `i`'s result into `out[i]`. A latent
    /// encoder that samples its posterior draws image `i`'s noise at
    /// `draws[i]`: its request's seed and the request's latent encodings
    /// before it.
    ///
    /// An unpacked vision encoder runs one numerical call per group of
    /// host-prepared images of one pixel shape, in first appearance order,
    /// and one per resident image; a latent encoder one per image; packed
    /// vision runs chunks of the packed capacity. Returns each image's
    /// result shape. The caller keeps host-prepared pixels until the
    /// encode's device work completed.
    pub fn encode(
        &mut self,
        images: &[&PreparedImage],
        draws: &[(u64, u64)],
        out: &[StorageView],
    ) -> Result<Encoded> {
        // At once: the recorded encode, performed here.
        let job = Job::new();
        let recorded = self.record(images, draws, out, &job)?;
        let mut completion = job.perform().check().map_err(|failure| failure.error)?;
        recorded.finish(&mut completion)
    }

    /// Record the encoding of `images` into `job`, image `i`'s result into
    /// `out[i]` and a sampling latent encoder's noise of image `i` at
    /// `draws[i]`: a packed encoder's grid upload, pixel copies and resident
    /// preparations, graph launch or eager call and feature copies per
    /// chunk; an unpacked encoder's resident preparations, noise draws and
    /// eager calls. The job's completion gives the result
    /// ([`Recorded::finish`]): an unpacked call returns the shapes it wrote.
    pub(crate) fn record(
        &mut self,
        images: &[&PreparedImage],
        draws: &[(u64, u64)],
        out: &[StorageView],
        job: &Job,
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
        let runner = || Argument::Handle(self.methods.runner.clone());
        match (&self.info, &self.packed) {
            (EncoderInfo::Vision(_), Some(_)) => {
                self.encode_packed(images, out, &mut result.encoded, job)?
            }
            (EncoderInfo::Vision(vision), None) => {
                // Host-prepared images of one pixel shape share a call,
                // within the rows its projections were prepared for; each
                // resident image is prepared into the resident pixels and
                // encoded alone.
                let budget = unpacked_rows(vision, &self.config);
                let host = images.iter().enumerate().filter_map(|(index, image)| {
                    matches!(image.pixels, ImagePixels::Host(_))
                        .then_some((index, image.facts.shape.as_slice()))
                });
                let resident = (0..images.len())
                    .filter(|&index| matches!(images[index].pixels, ImagePixels::Resident { .. }))
                    .map(|index| vec![index]);
                let groups: Vec<Vec<usize>> = shape_groups(host)
                    .into_iter()
                    .flat_map(|group| {
                        chunk_rows(group, budget, |index| {
                            image_rows(&images[index].facts, vision.patch_size)
                        })
                    })
                    .chain(resident)
                    .collect();
                for group in groups {
                    let mut grids = Vec::new();
                    let mut arguments = vec![runner(), Argument::None];
                    for &index in &group {
                        let image = images[index];
                        if let Some([rows, columns]) = image.facts.grid {
                            grids.extend([i64::from(rows), i64::from(columns)]);
                        }
                        arguments.push(Argument::Tensor(self.pixels(image, Tower::Vit, job)?));
                    }
                    arguments[1] = Argument::Integers(grids);
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
            (EncoderInfo::Latent(_), _) => {
                if draws.len() != images.len() {
                    return Err(Error::Invariant(
                        "latent images and their posterior draws must align".into(),
                    ));
                }
                for (index, image) in images.iter().enumerate() {
                    let pixels = self.pixels(image, Tower::Vae, job)?;
                    let mut arguments = vec![
                        runner(),
                        Argument::Tensor(pixels),
                        Argument::Tensor(out[index].clone()),
                    ];
                    if let Some(noise) = &self.noise {
                        let [_, height, width] = image.facts.shape[..] else {
                            return Err(Error::Invariant(
                                "latent pixels are [channels, height, width]".into(),
                            ));
                        };
                        let (seed, draw) = draws[index];
                        job.call(MethodCall::new(
                            &self.methods.draw_noise,
                            vec![
                                runner(),
                                // The seed's bits; the Python half reads them
                                // as unsigned.
                                Argument::Integer(seed as i64),
                                Argument::Integer(draw as i64),
                                Argument::Integer(height),
                                Argument::Integer(width),
                                Argument::Tensor(noise.clone()),
                            ],
                        ));
                        arguments.push(Argument::Tensor(noise.clone()));
                    }
                    job.call_returning(MethodCall::new(&self.methods.encode, arguments));
                    result.returning.push(vec![index]);
                    result.encoded.record(GraphMode::Eager);
                }
            }
            (EncoderInfo::Text(_) | EncoderInfo::Condition, _) => {
                return Err(Error::Invalid(format!("{} encodes no images", self.owner)));
            }
        }
        Ok(result)
    }

    /// The pixels an unpacked call reads for `image`: its host-prepared
    /// pixels, or the runner's resident pixels after recording the resident
    /// image's preparation into them for `tower`. Calls run in the job's
    /// order on one stream, so the next resident image is prepared after
    /// the call that read the last one.
    fn pixels(&self, image: &PreparedImage, tower: Tower, job: &Job) -> Result<StorageView> {
        match &image.pixels {
            ImagePixels::Host(pixels) => Ok(pixels.clone()),
            ImagePixels::Resident {
                image: value,
                signed_unit,
            } => {
                let resident = self.resident.as_ref().ok_or_else(|| {
                    Error::Invariant("an unpacked image encoder holds resident pixels".into())
                })?;
                let elements = image.facts.elements();
                if elements > resident.numel() as i64 {
                    return Err(Error::Invalid(format!(
                        "a resident image of {elements} values exceeds {}'s {} resident values",
                        self.owner,
                        resident.numel()
                    )));
                }
                let pixels = resident
                    .prefix(elements)?
                    .reshape(resident.dtype(), &image.facts.shape)?;
                self.prepare_resident(value, tower, *signed_unit, &pixels, job);
                Ok(pixels)
            }
        }
    }

    /// Record the preparation of a resident image product for `tower` into
    /// `out` (`EncoderRunner.prepare_resident`).
    fn prepare_resident(
        &self,
        image: &StorageView,
        tower: Tower,
        signed_unit: bool,
        out: &StorageView,
        job: &Job,
    ) {
        job.call(MethodCall::new(
            &self.methods.prepare_resident,
            vec![
                Argument::Handle(self.methods.runner.clone()),
                Argument::Tensor(image.clone()),
                Argument::Text(tower.name().to_owned()),
                Argument::Bool(signed_unit),
                Argument::Tensor(out.clone()),
            ],
        ));
    }
}

impl EncoderRunner {
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
        job: &Job,
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
}

impl EncoderRunner {
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

    /// Record in `job` the forward of `key`, whose `encode` takes
    /// `arguments`, on the encoder's stream: the launch of the key's
    /// captured graph, or the eager call. Startup's capture passes the
    /// graph `pool`: the eager call then warms the key, so kernels compile
    /// outside the capture, and the key is captured behind it; the job's
    /// completion carries the graph. A capture is counted when its graph is
    /// stored.
    fn forward(
        &mut self,
        key: EncoderGraph,
        arguments: Vec<Argument>,
        pool: Option<&GraphPool>,
        job: &Job,
    ) -> Result<GraphMode> {
        let stream = self.stream.clone();
        let graph = match pool {
            Some(_) => None,
            None => self.graph(key)?,
        };
        let mode = match (graph, &stream) {
            (Some(graph), Some(_)) => {
                job.on(stream.as_ref()).launch(graph)?;
                GraphMode::Replay
            }
            _ => {
                let capture = match (pool, &stream, self.device) {
                    (Some(pool), Some(stream), Device::Cuda(device)) => Some(Capture {
                        functions: self.methods.capture.clone(),
                        pool: pool.value.clone(),
                        peers: Vec::new(),
                        stream: stream.raw(),
                        device,
                        method: MethodCall::new(&self.methods.encode, arguments.clone()),
                        breaks: None,
                    }),
                    _ => None,
                };
                job.call(MethodCall::new(&self.methods.encode, arguments));
                match capture {
                    Some(capture) => {
                        job.capture(capture);
                        GraphMode::Capture
                    }
                    None => GraphMode::Eager,
                }
            }
        };
        Ok(mode)
    }

    /// Warm and, with `pool`, capture every graph this encoder launches:
    /// packed slot counts `capacity..=1` and text capacities, largest
    /// first. Each forward is one job, as serving records it: its input
    /// writes, its eager call and, with a pool, its capture. `perform`
    /// performs a job and returns its completion: at a rank's startup the
    /// encoder's lane performs it, outside a rank the calling thread
    /// ([`Self::capture`]). `check` runs after each capture (the graph
    /// storage budget); a failed check releases that graph. Without a pool
    /// the same forwards run once, so lazy initialization precedes serving.
    pub(crate) fn capture_with(
        &mut self,
        pool: Option<&Arc<GraphPool>>,
        check: &mut dyn FnMut() -> Result<()>,
        perform: &mut dyn FnMut(Job) -> Result<Completion>,
    ) -> Result<()> {
        if let Some(pool) = pool
            && Device::Cuda(pool.device) != self.device
        {
            return Err(Error::Invariant(
                "an encoder captures into a pool of its device".into(),
            ));
        }
        // The forwards to capture, largest first, each with its arguments.
        // The first forward's job writes what every forward reads.
        let mut job = Job::new();
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
                // Every capacity encodes the zero prompt.
                // SAFETY: the text encoder's token input, kept until it closes
                // after its stream drained; only the warm forwards read it,
                // recorded behind the fill on the same stream.
                unsafe { text.tokens.zero(job.on(self.stream.as_ref())) }?;
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
            let recorded = std::mem::take(&mut job);
            if let (EncoderGraph::Packed { slots }, Some(grids), Some(packed)) =
                (key, &grids, &self.packed)
            {
                // SAFETY: the encoder's grid input, kept until it closes after
                // its stream drained; the warm forwards reading it run on its
                // stream, each job performed before the next writes.
                unsafe {
                    packed
                        .grids
                        .prefix(2 * slots as i64)?
                        .upload(grids, recorded.on(self.stream.as_ref()))
                }?;
            }
            let mode = self.forward(key, arguments, pool.map(Arc::as_ref), &recorded)?;
            let mut completion = perform(recorded)?;
            if mode != GraphMode::Capture {
                continue;
            }
            let graph = completion.captured()?;
            if let Err(error) = check() {
                let release = &self.methods.capture.release;
                if let Err(cleanup) = release.invoke(&[graph.handle().view()]) {
                    tracing::warn!(
                        "{}: releasing a refused graph failed: {cleanup}",
                        self.owner
                    );
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

    /// `Self::capture_with` for a caller outside a rank, which has no
    /// lane: each job is performed on the calling thread, as
    /// [`Self::encode`] performs its job.
    pub fn capture(
        &mut self,
        pool: Option<&Arc<GraphPool>>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        self.capture_with(pool, check, &mut |job| {
            job.perform().check().map_err(|failure| failure.error)
        })
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
fn shape_groups<'a>(shapes: impl IntoIterator<Item = (usize, &'a [i64])>) -> Vec<Vec<usize>> {
    let mut groups: Vec<(&[i64], Vec<usize>)> = Vec::new();
    for (index, shape) in shapes {
        match groups.iter_mut().find(|(group, _)| *group == shape) {
            Some((_, members)) => members.push(index),
            None => groups.push((shape, vec![index])),
        }
    }
    groups.into_iter().map(|(_, members)| members).collect()
}

impl EncoderRunner {
    /// The calls that list the kernels the operators of the runner's
    /// execution context or of the condition encoders selected.
    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        match (&self.context, &self.conditions) {
            (Some(context), _) => vec![kernels_call(&context.kernels, &self.methods.runner)],
            (_, Some(conditions)) => conditions.kernel_calls(),
            _ => Vec::new(),
        }
    }

    /// The kernels the operators of the runner's execution context or of the
    /// condition encoders selected.
    pub(crate) fn kernels(&self) -> Result<Vec<KernelRecord>> {
        list_kernels(self.kernel_calls())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_split_into_chunks_within_their_row_budget() {
        let rows = [3, 4, 2, 9, 1];
        let chunks = chunk_rows(vec![0, 1, 2, 3, 4], 6, |index| rows[index]);
        assert_eq!(chunks, vec![vec![0], vec![1, 2], vec![3], vec![4]]);
    }

    #[test]
    fn unpacked_images_group_by_pixel_shape_in_appearance_order() {
        let (a, b, c) = ([3, 32, 32], [3, 16, 32], [3, 32, 32]);
        assert_eq!(
            shape_groups([(0, &a[..]), (1, &b[..]), (2, &c[..])]),
            vec![vec![0, 2], vec![1]]
        );
        // Members keep their own indices.
        assert_eq!(
            shape_groups([(4, &b[..]), (7, &a[..])]),
            vec![vec![4], vec![7]]
        );
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
            image_elements: 3 * 2520 * 256,
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

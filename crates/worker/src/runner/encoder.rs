//! Encoder runners: vision patch encoders, latent encoders and text encoders.
//!
//! An [`EncoderRunner`] serves one encoder call of a bound component: the
//! module at a path within the binding's root (the root itself, or a nested
//! encoder such as a text encoder's vision tower bound with it). It owns the
//! encoder's Rust buffers, its graph table and its call policy; the module
//! computes through the binding functions of
//! `uniserve_models.interface.encoder`, created once over the buffers:
//!
//! | kind | module | calls |
//! | --- | --- | --- |
//! | [`EncoderInfo::Vision`] | `PatchEncoder` | `encode`, `prepare_resident`; packed: `encode_packed`, `capture_packed`, `release` |
//! | [`EncoderInfo::Latent`] | `PatchAutoencoder` | `encode`, `prepare_resident` |
//! | [`EncoderInfo::Text`] | `TextEncoder` | `text_positions`, `encode_text`, `capture_text`, `release` |
//!
//! Every binding receives its device's product arena once; results are
//! written into product [`Region`]s of it, in the product's dtype, and each
//! call returns the results' shapes, which the caller publishes. Inputs and
//! outputs never import DLPack tensors per call.
//!
//! **Images.** Inline request images are decoded and transformed on a host
//! lane by `image.prepare_input` ([`prepare_input`]), which returns an
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
//! serves every prompt length up to it. Without configured capacities a
//! prompt is encoded at its exact length, eagerly. Prompts with vision
//! tokens (rotary coordinates and vision rows) always run eagerly.
//!
//! Every runner call is ordered on the runner's stream; the caller orders
//! that stream after the producers of its inputs and records the producer
//! fence of its outputs on it.

use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_cuda::Stream;

use super::GraphPool;
use super::graphs::{GraphMode, GraphTable, Lookup};
use crate::config::Device;
use crate::ffi::{
    Backing, DType, Fields, Function, Functions, Handle, Record, Region, integer_list, list,
};
use crate::model::bind::ComponentBinding;
use crate::model::interface::Interface;
use crate::storage::memory::{Memory, StorageView};
use crate::{Error, Result};

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
    fn from_value(owner: &str, value: Any) -> Result<Self> {
        let record = Record::new(owner, value)?;
        let grid = record
            .optional_value("grid")?
            .map(|grid| -> Result<[u32; 2]> {
                let values =
                    crate::ffi::integers(grid).map_err(|error| Error::numerical(owner, &error))?;
                match values[..] {
                    [rows, columns] if rows > 0 && columns > 0 => Ok([rows as u32, columns as u32]),
                    _ => Err(Error::Interface(format!(
                        "{owner} reports a grid that is not two positive extents"
                    ))),
                }
            })
            .transpose()?;
        let dimension = |key: &str| -> Result<u32> {
            u32::try_from(record.count(key)?)
                .ok()
                .filter(|&extent| extent > 0)
                .ok_or_else(|| Error::Interface(format!("{owner} reports an invalid {key}")))
        };
        Ok(Self {
            value: Handle::new(record.value("image")?),
            shape: record.shape("shape")?,
            grid,
            height: dimension("height")?,
            width: dimension("width")?,
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
    PreparedImage::from_value(function.name(), value)
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

/// What an encoder computes, as `encoder.describe` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderInfo {
    Vision(VisionEncoder),
    Latent(LatentEncoder),
    Text(TextEncoder),
}

impl EncoderInfo {
    /// Describe the encoder at `path` within a bound component.
    pub fn describe(
        interface: &Interface,
        binding: &ComponentBinding,
        loaded: &Handle,
        path: &str,
    ) -> Result<Self> {
        let function = interface.function("encoder.describe")?;
        let record = Record::new(
            function.name(),
            function.call_values(&[
                binding.prepared().to_any(),
                loaded.to_any(),
                Any::from(tvm_ffi::String::from(path)),
            ])?,
        )?;
        Self::from_record(&record)
    }

    fn from_record(record: &Record) -> Result<Self> {
        let dtype = |key: &str| -> Result<DType> {
            let name = record.string(key)?;
            DType::from_name(&name).ok_or_else(|| {
                Error::Interface(format!(
                    "{} reports the unknown dtype {name:?}",
                    record.owner()
                ))
            })
        };
        let extent = |key: &str| -> Result<u32> {
            u32::try_from(record.count(key)?)
                .map_err(|_| Error::Interface(format!("{} {key} overflows", record.owner())))
        };
        Ok(match record.string("kind")?.as_str() {
            "vision" => {
                let max_patches = extent("max_patches")?;
                let downsample = extent("downsample")?;
                if downsample == 0 || max_patches % (downsample * downsample) != 0 {
                    return Err(Error::Interface(
                        "packed vision slots must hold whole output features".into(),
                    ));
                }
                Self::Vision(VisionEncoder {
                    patch_size: extent("patch_size")?,
                    downsample,
                    output_size: extent("output_size")?,
                    output_dtype: dtype("output_dtype")?,
                    max_patches: (max_patches > 0).then_some(max_patches),
                    row_width: extent("row_width")?,
                    pixel_dtype: dtype("pixel_dtype")?,
                })
            }
            "latent" => Self::Latent(LatentEncoder {
                downsample: extent("downsample")?,
                width: extent("width")?,
                output_dtype: dtype("output_dtype")?,
            }),
            "text" => Self::Text(TextEncoder {
                hidden_size: extent("hidden_size")?,
                output_dtype: dtype("output_dtype")?,
                positions: record.flag("positions")?,
            }),
            other => {
                return Err(Error::Interface(format!(
                    "{} reports the unknown encoder kind {other:?}",
                    record.owner()
                )));
            }
        })
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
    super::graphs::bucket(capacities, tokens).ok_or_else(|| {
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
    pub rows: &'a Region,
}

/// Startup bounds of an encoder's buffers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Images one batch may carry (`max_batch_calls`).
    pub max_images: usize,
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
    grids: StorageView,
}

/// The binding functions of one encoder kind.
enum Steps {
    Vision {
        encode: Function,
        prepare_resident: Function,
        packed: Option<(Function, Function)>,
    },
    Latent {
        encode: Function,
        prepare_resident: Function,
    },
    Text {
        positions: Function,
        encode: Function,
        capture: Function,
    },
}

/// The runner of one bound encoder call.
pub struct EncoderRunner {
    owner: String,
    info: EncoderInfo,
    device: Device,
    /// The binding's stream (a borrowed view of the lane's); `None` on the
    /// host.
    stream: Option<Stream>,
    steps: Steps,
    /// The device's product arena, as the binding received it.
    products: StorageView,
    packed: Option<PackedVision>,
    /// The text token column `[max capacity]` int64.
    tokens: Option<StorageView>,
    text_capacities: Vec<usize>,
    graphs: Option<GraphTable<EncoderGraph>>,
    /// Capture is enabled: graph pools were supplied at startup.
    capturing: bool,
    /// Buffers bound to the Python binding, retained for its lifetime.
    _buffers: Vec<StorageView>,
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
    /// Describe the encoder at `path` of `binding`, allocate its buffers and
    /// bind them. `products` is the binding device's product arena (a
    /// `uint8` view); `stream` borrows the binding's stream.
    pub fn bind(
        interface: &Interface,
        binding: &ComponentBinding,
        loaded: &Handle,
        path: &str,
        config: &EncoderConfig,
        products: StorageView,
        stream: Option<Stream>,
    ) -> Result<Self> {
        let owner = if path.is_empty() {
            binding.component.clone()
        } else {
            format!("{}.{path}", binding.component)
        };
        let info = EncoderInfo::describe(interface, binding, loaded, path)?;
        let device = binding.device;
        if products.dtype() != DType::U8 || products.shape().len() != 1 {
            return Err(Error::Invariant(
                "a product arena is bound as one uint8 dimension".into(),
            ));
        }
        if device.is_cuda() != stream.is_some() {
            return Err(Error::Invariant(
                "an encoder runs on a stream exactly on a CUDA device".into(),
            ));
        }

        let mut buffers = vec![("products", products.clone())];
        let mut packed = None;
        let mut tokens = None;
        match info {
            EncoderInfo::Vision(vision) => {
                if let (Some(max_patches), Some(slot_tokens)) =
                    (vision.max_patches, vision.slot_tokens())
                {
                    let capacity = config.max_images.clamp(1, MAX_PACKED_IMAGES) as i64;
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
                    let grids = allocate(device, DType::I64, &[2 * capacity, 2])?;
                    let features = allocate(
                        device,
                        vision.output_dtype,
                        &[
                            capacity * i64::from(slot_tokens),
                            i64::from(vision.output_size),
                        ],
                    )?;
                    buffers.extend([
                        ("pixels", pixels),
                        ("grids", grids.clone()),
                        ("features", features),
                    ]);
                    packed = Some(PackedVision {
                        capacity: capacity as usize,
                        grids,
                    });
                }
            }
            EncoderInfo::Latent(_) => {}
            EncoderInfo::Text(text) => {
                let capacity = config
                    .text_capacities
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(config.max_text_tokens)
                    .max(1) as i64;
                let column = allocate(device, DType::I64, &[capacity])?;
                buffers.push(("tokens", column.clone()));
                buffers.push((
                    "features",
                    allocate(
                        device,
                        text.output_dtype,
                        &[capacity, i64::from(text.hidden_size)],
                    )?,
                ));
                if text.positions {
                    buffers.push(("positions", allocate(device, DType::I64, &[3, capacity])?));
                }
                tokens = Some(column);
            }
        }
        if let Some(stream) = &stream {
            // The zero fill must complete before Python reads the buffers.
            stream.synchronize()?;
        }

        let mut tensors = Fields::new();
        for (name, view) in &buffers {
            tensors = tensors.with(name, view.to_tensor()?);
        }
        let function = interface.function("encoder.bind")?;
        let functions = Functions::new(
            function.name(),
            function.call_values(&[
                binding.prepared().to_any(),
                loaded.to_any(),
                Any::from(tvm_ffi::String::from(path)),
                tensors.into_any(),
            ])?,
        )?;
        let (steps, release) = match info {
            EncoderInfo::Vision(_) => {
                let packed_steps = packed
                    .as_ref()
                    .map(|_| -> Result<_> {
                        Ok((
                            functions.get("encode_packed")?,
                            functions.get("capture_packed")?,
                        ))
                    })
                    .transpose()?;
                let release = packed
                    .as_ref()
                    .map(|_| functions.get("release"))
                    .transpose()?;
                (
                    Steps::Vision {
                        encode: functions.get("encode")?,
                        prepare_resident: functions.get("prepare_resident")?,
                        packed: packed_steps,
                    },
                    release,
                )
            }
            EncoderInfo::Latent(_) => (
                Steps::Latent {
                    encode: functions.get("encode")?,
                    prepare_resident: functions.get("prepare_resident")?,
                },
                None,
            ),
            EncoderInfo::Text(_) => (
                Steps::Text {
                    positions: functions.get("text_positions")?,
                    encode: functions.get("encode_text")?,
                    capture: functions.get("capture_text")?,
                },
                Some(functions.get("release")?),
            ),
        };
        let mut text_capacities = config.text_capacities.clone();
        text_capacities.sort_unstable();
        text_capacities.dedup();
        Ok(Self {
            graphs: release.map(|release| GraphTable::new(owner.clone(), release)),
            owner,
            info,
            device,
            stream,
            steps,
            products,
            packed,
            tokens,
            text_capacities,
            capturing: false,
            _buffers: buffers.into_iter().map(|(_, view)| view).collect(),
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

    /// The product arena regions name.
    pub fn products(&self) -> &StorageView {
        &self.products
    }

    /// Images one packed numerical call encodes; `None` when unpacked.
    pub fn packed_capacity(&self) -> Option<usize> {
        self.packed.as_ref().map(|packed| packed.capacity)
    }

    /// Prepare a resident image product for this encoder on its stream.
    /// `signed_unit` declares pixel values in `[-1, 1]` rather than `[0, 1]`.
    pub fn prepare_resident(
        &self,
        image: &Region,
        tower: Tower,
        signed_unit: bool,
    ) -> Result<PreparedImage> {
        let function = match &self.steps {
            Steps::Vision {
                prepare_resident, ..
            }
            | Steps::Latent {
                prepare_resident, ..
            } => prepare_resident,
            Steps::Text { .. } => {
                return Err(Error::Invalid(format!("{} encodes no images", self.owner)));
            }
        };
        let value = function.call_values(&[
            image.to_any(),
            Any::from(tvm_ffi::String::from(tower.name())),
            Any::from(signed_unit),
        ])?;
        PreparedImage::from_value(function.name(), value)
    }

    /// Encode images, writing image `i`'s result into `out[i]`.
    ///
    /// Unpacked encoders run one numerical call per pixel shape, in first
    /// appearance order; packed vision runs chunks of the packed capacity.
    /// Returns each image's result shape.
    pub fn encode(&mut self, images: &[&PreparedImage], out: &[Region]) -> Result<Encoded> {
        if images.len() != out.len() {
            return Err(Error::Invariant(
                "encoder images and products must align".into(),
            ));
        }
        let mut result = Encoded {
            shapes: vec![Vec::new(); images.len()],
            ..Encoded::default()
        };
        if images.is_empty() {
            return Ok(result);
        }
        match &self.steps {
            Steps::Vision {
                packed: Some(_), ..
            } => self.encode_packed(images, out, &mut result)?,
            Steps::Vision { encode, .. } | Steps::Latent { encode, .. } => {
                let encode = encode.clone();
                for group in shape_groups(images) {
                    let shapes = call_shapes(
                        &encode,
                        &[
                            image_list(group.iter().map(|&index| images[index])),
                            region_list(group.iter().map(|&index| &out[index])),
                        ],
                        group.len(),
                    )?;
                    for (index, shape) in group.into_iter().zip(shapes) {
                        result.shapes[index] = shape;
                    }
                    result.record(GraphMode::Eager);
                }
            }
            Steps::Text { .. } => {
                return Err(Error::Invalid(format!("{} encodes no images", self.owner)));
            }
        }
        Ok(result)
    }

    fn encode_packed(
        &mut self,
        images: &[&PreparedImage],
        out: &[Region],
        result: &mut Encoded,
    ) -> Result<()> {
        let (
            EncoderInfo::Vision(vision),
            Some(packed),
            Steps::Vision {
                packed: Some((encode, _)),
                ..
            },
        ) = (&self.info, &self.packed, &self.steps)
        else {
            return Err(Error::Invariant("encoder packs no images".into()));
        };
        let (vision, encode) = (*vision, encode.clone());
        let (capacity, grids) = (packed.capacity, packed.grids.clone());
        let max_patches = vision.max_patches.unwrap_or_default();
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
        let mut offset = 0;
        for chunk in shapes.chunks(capacity) {
            let slots = chunk.len();
            let layout = packed_grids(chunk, slots, max_patches, vision.downsample)?;
            grids
                .prefix(2 * slots as i64)?
                .upload(grid_bytes(&layout).as_slice(), self.stream.as_ref())?;
            let (graph, mode) = self.graph(EncoderGraph::Packed { slots })?;
            let shapes = call_shapes(
                &encode,
                &[
                    image_list(images[offset..offset + slots].iter().copied()),
                    Any::from(slots as i64),
                    graph.map_or_else(Any::default, |graph| graph.to_any()),
                    region_list(out[offset..offset + slots].iter()),
                ],
                slots,
            )?;
            for (index, shape) in shapes.into_iter().enumerate() {
                result.shapes[offset + index] = shape;
            }
            result.record(mode);
            offset += slots;
        }
        Ok(())
    }

    /// Encode one prompt of `tokens` ids into `out`.
    ///
    /// `vision` carries the prompt's vision tokens, for an encoder that
    /// places them. Returns the features' shape,
    /// or `None` on a pipeline stage that computes no features.
    pub fn encode_text(
        &mut self,
        tokens: &[i64],
        out: &Region,
        vision: Option<PromptVision<'_>>,
    ) -> Result<(Option<Vec<i64>>, GraphMode)> {
        let (
            Steps::Text {
                positions, encode, ..
            },
            Some(column),
        ) = (&self.steps, &self.tokens)
        else {
            return Err(Error::Invalid(format!("{} encodes no text", self.owner)));
        };
        let (positions, encode) = (positions.clone(), encode.clone());
        let capacity = text_capacity(&self.text_capacities, tokens.len())?;
        if capacity as i64 > column.shape()[0] {
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
        column.upload(&bytes, self.stream.as_ref())?;

        let (graph, mode, place, visual) = match vision {
            Some(PromptVision {
                image_grids,
                video_grids,
                rows,
            }) => {
                let grids =
                    |grids: &[[u32; 3]]| {
                        list(grids.iter().map(|grid| {
                            Any::from(integer_list(grid.iter().map(|&x| i64::from(x))))
                        }))
                    };
                positions.invoke(&[
                    (&Any::from(capacity as i64)).into(),
                    (&Any::from(integer_list(padded.iter().copied()))).into(),
                    (&Any::from(grids(image_grids))).into(),
                    (&Any::from(grids(video_grids))).into(),
                ])?;
                (None, GraphMode::Eager, true, rows.to_any())
            }
            None if self.text_capacities.is_empty() => {
                (None, GraphMode::Eager, false, Any::default())
            }
            None => {
                let (graph, mode) = self.graph(EncoderGraph::Text { capacity })?;
                (graph, mode, false, Any::default())
            }
        };
        let value = encode.call_values(&[
            Any::from(capacity as i64),
            Any::from(tokens.len() as i64),
            graph.map_or_else(Any::default, |graph| graph.to_any()),
            Any::from(place),
            visual,
            out.to_any(),
        ])?;
        let shape =
            crate::ffi::integers(value).map_err(|error| Error::numerical(encode.name(), &error))?;
        Ok(((!shape.is_empty()).then_some(shape), mode))
    }

    /// The resident graph of `key`, or eager execution when capture is
    /// disabled. After seal a missing graph is an error.
    fn graph(&mut self, key: EncoderGraph) -> Result<(Option<Handle>, GraphMode)> {
        let Some(table) = self.graphs.as_mut().filter(|_| self.capturing) else {
            return Ok((None, GraphMode::Eager));
        };
        let found = match table.lookup(&key, None)? {
            Lookup::Replay(graph) => Some(graph.clone()),
            Lookup::Capture => None,
        };
        let mode = if found.is_some() {
            GraphMode::Replay
        } else {
            GraphMode::Eager
        };
        table.record(mode);
        Ok((found, mode))
    }

    /// Warm and, with `pool`, capture every graph this encoder replays:
    /// packed slot counts `capacity..=1` and text capacities, largest
    /// first. `check` runs after each capture (the graph storage budget);
    /// a failed check releases that graph. Without a pool the same calls
    /// run eagerly once, so lazy initialization precedes serving.
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
        let pool_value = pool.map_or_else(Any::default, |pool| pool.value.to_any());
        let mut captured = Vec::new();
        match (&self.info, &self.packed, &self.steps) {
            (
                EncoderInfo::Vision(vision),
                Some(packed),
                Steps::Vision {
                    packed: Some((_, capture)),
                    ..
                },
            ) => {
                for slots in (1..=packed.capacity).rev() {
                    let layout = packed_grids(
                        &[],
                        slots,
                        vision.max_patches.unwrap_or_default(),
                        vision.downsample,
                    )?;
                    packed
                        .grids
                        .prefix(2 * slots as i64)?
                        .upload(grid_bytes(&layout).as_slice(), self.stream.as_ref())?;
                    let graph =
                        capture.call_values(&[Any::from(slots as i64), pool_value.clone()])?;
                    captured.push((EncoderGraph::Packed { slots }, graph));
                }
            }
            (EncoderInfo::Text(_), _, Steps::Text { capture, .. }) => {
                let capture = capture.clone();
                for &capacity in self.text_capacities.clone().iter().rev() {
                    let zeros = vec![0u8; capacity * 8];
                    if let Some(column) = &self.tokens {
                        column.upload(&zeros, self.stream.as_ref())?;
                    }
                    let graph =
                        capture.call_values(&[Any::from(capacity as i64), pool_value.clone()])?;
                    self.keep(EncoderGraph::Text { capacity }, graph, check)?;
                }
            }
            _ => {}
        }
        for (key, graph) in captured {
            self.keep(key, graph, check)?;
        }
        self.capturing = pool.is_some();
        Ok(())
    }

    /// Record one captured graph after the storage check; a warm call
    /// without a pool returns no graph.
    fn keep(
        &mut self,
        key: EncoderGraph,
        graph: Any,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        if crate::ffi::is_none(&graph) {
            return Ok(());
        }
        let table = self
            .graphs
            .as_mut()
            .ok_or_else(|| Error::Invariant("encoder has no graph table".into()))?;
        let graph = Handle::new(graph);
        if let Err(error) = check() {
            // Not yet in the table: release it here.
            let _ = table.insert(key, None, graph);
            return Err(error);
        }
        table.insert(key, None, graph)
    }

    /// Refuse further captures: serving replays what startup captured.
    pub fn seal(&mut self) {
        if let Some(table) = &mut self.graphs {
            table.seal();
        }
    }

    /// Release the runner's graphs; the caller drained its stream. An
    /// aborted close keeps them until the process exits.
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        self.graphs
            .as_mut()
            .map_or(Ok(()), |table| table.close(aborted))
    }
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

fn image_list<'a>(images: impl Iterator<Item = &'a PreparedImage>) -> Any {
    Any::from(list(images.map(|image| image.value.to_any())))
}

fn region_list<'a>(regions: impl Iterator<Item = &'a Region>) -> Any {
    Any::from(list(regions.map(Region::to_any)))
}

/// Call an encode function and read one result shape per input.
fn call_shapes(function: &Function, args: &[Any], count: usize) -> Result<Vec<Vec<i64>>> {
    let value: tvm_ffi::Array<Any> = tvm_ffi::Array::try_from(function.call_values(args)?)
        .map_err(|error| Error::numerical(function.name(), &error))?;
    if value.len() != count {
        return Err(Error::Interface(format!(
            "{} returns one shape per input",
            function.name()
        )));
    }
    value
        .iter()
        .map(|shape| {
            crate::ffi::integers(shape).map_err(|error| Error::numerical(function.name(), &error))
        })
        .collect()
}

/// Native-endian int64 words of a grid layout, as the device reads them.
fn grid_bytes(layout: &[[i64; 2]]) -> Vec<u8> {
    layout
        .iter()
        .flatten()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
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
    fn unpacked_images_group_by_pixel_shape_in_appearance_order() {
        let image = |shape: &[i64]| PreparedImage {
            value: Handle::new(Any::default()),
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
    fn encoder_descriptions_are_validated() -> Result<()> {
        let record = |fields: Fields| Record::new("encoder.describe", fields.into_any());
        let vision = Fields::new()
            .text("kind", "vision")
            .with("patch_size", 16_i64)
            .with("downsample", 3_i64)
            .with("output_size", 32_i64)
            .text("output_dtype", "bfloat16")
            .with("max_patches", 2520_i64)
            .with("row_width", 768_i64)
            .text("pixel_dtype", "float32");
        let EncoderInfo::Vision(info) = EncoderInfo::from_record(&record(vision)?)? else {
            panic!("vision encoder");
        };
        assert_eq!(info.slot_tokens(), Some(280));
        assert_eq!(info.pixel_dtype, DType::F32);

        let unaligned = Fields::new()
            .text("kind", "vision")
            .with("patch_size", 16_i64)
            .with("downsample", 3_i64)
            .with("output_size", 32_i64)
            .text("output_dtype", "bfloat16")
            .with("max_patches", 100_i64)
            .with("row_width", 768_i64)
            .text("pixel_dtype", "float32");
        assert!(EncoderInfo::from_record(&record(unaligned)?).is_err());
        let unknown = Fields::new().text("kind", "audio");
        assert!(EncoderInfo::from_record(&record(unknown)?).is_err());
        Ok(())
    }
}

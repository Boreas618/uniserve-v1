//! Weights: the layout the reading operator declares, the key retained storage files them under,
//! and the loader that converts a checkpoint's tensors once (specs/worker.md 4.3, 8).
//!
//! ## The layout
//!
//! [`WeightLayout`] is the reading operator's arrangement of the bytes, named by the operator and
//! carried with the weight. The checkpoint's own tensors are laid out per model framework (an
//! expert's gate and up projections are separate matrices, gate first); an operator that reads one
//! fused `[experts, 2·intermediate, hidden]` weight with the up projection first declares
//! [`WeightLayout::UP_GATE`] and the loader produces that arrangement from the checkpoint's parts,
//! once, before the bytes are published.
//!
//! ## The key
//!
//! Retained weights are keyed by exactly the four inputs the design names (specs/worker.md 8): the
//! identity of every checkpoint file whose bytes produced the weight, the element type the weight
//! has after conversion, the layout name, and the weight's region in its logical tensor. No program
//! name or revision enters the key: a program-layer update keeps every retained weight (specs/worker.md
//! 7.3). A change of dtype, quantization, layout or checkpoint therefore changes the key, and a
//! change of an operator's program does not.
//!
//! ## Publishing
//!
//! Conversion happens here and nowhere else, and its output goes straight to retained storage
//! through [`RetainedWeights`], which the executor implements over the Storage Server and its
//! device mappings. `nn` owns no device memory: a [`Weight`] is a [`Backing`] plus the view the
//! operator reads.

use std::fmt;

use ops::fp4::ScaleLayout;
use ops::{DType, Shape, TensorView};

use crate::checkpoint::{Checkpoint, FileIdentity};
use crate::error::{Error, Result};
use crate::tensor::Backing;

/// The arrangement of a weight's bytes that its reading operator declares.
///
/// The names enter the retained key, so each one is a stable statement about layout, not about the
/// operator's program or tuning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WeightLayout(&'static str);

impl WeightLayout {
    /// The checkpoint's own contiguous row-major arrangement, unchanged.
    pub const ROW_MAJOR: Self = Self("row_major");

    /// A routed expert's first projection: `[experts, 2·intermediate, hidden]`, up before gate.
    pub const UP_GATE: Self = Self("up_gate");

    /// A routed expert's second projection: `[experts, hidden, intermediate]`.
    pub const DOWN: Self = Self("down");

    /// A calibrated NVFP4 matrix's packed E2M1 values: `[rows, columns / 2]` U8, one byte per pair.
    pub const NVFP4_VALUES: Self = Self("nvfp4_values");

    /// A calibrated NVFP4 matrix's E4M3 block scales in the linear layout: `[rows, columns / 16]`.
    pub const NVFP4_SCALES_LINEAR: Self = Self("nvfp4_scales_linear");

    /// The same scales in the swizzled 128x4 layout the cuDNN product reads.
    pub const NVFP4_SCALES_SWIZZLED: Self = Self("nvfp4_scales_swizzled_128x4");

    /// The layout named `name`.
    pub const fn named(name: &'static str) -> Self {
        Self(name)
    }

    /// The layout's name.
    pub const fn name(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for WeightLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// An explicitly requested quantization method.
///
/// The names are SGLang's `quantization` values (specs/worker.md 1, 9): an explicit value converts
/// the checkpoint's weights once, at load; with none, the checkpoint's own representation is kept.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Quantization {
    /// FP8 E4M3 with per-tensor scales.
    Fp8,
    /// FP8 E4M3 with block scales.
    Mxfp8,
    /// NVFP4 E2M1 with block scales.
    NvFp4,
}

impl fmt::Display for Quantization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Fp8 => "fp8",
            Self::Mxfp8 => "mxfp8",
            Self::NvFp4 => "nvfp4",
        })
    }
}

/// Where a weight's bytes come from: one region of one checkpoint tensor, or whole tensors fused in
/// order along their first axis.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WeightSource {
    /// `shape` elements starting at element `start` of `tensor`, row-major.
    Region {
        /// The checkpoint tensor's name.
        tensor: String,
        /// The first element of the region, in the tensor's row-major order.
        start: u64,
        /// The region's extents.
        shape: Shape,
    },
    /// `shape` elements of `tensor` whose last axis begins at element `start` and repeats every
    /// `stride` elements.
    ///
    /// A tensor-parallel rank reads one column block of a row-parallel projection: every row of the
    /// checkpoint's tensor contributes `shape.last()` elements, and consecutive rows start `stride`
    /// elements apart. The loader gathers those rows and converts them like any other weight, so
    /// the retained storage of a shard is contiguous like the rest.
    Strided {
        /// The checkpoint tensor's name.
        tensor: String,
        /// The first element of the first row, in the tensor's row-major order.
        start: u64,
        /// The gathered region's extents.
        shape: Shape,
        /// Elements between the first elements of consecutive rows.
        row_stride: u64,
    },
    /// The row-major concatenation of `parts` along their first axis, viewed as `shape`.
    ///
    /// Every part must have the same extents after its first axis. This is how a routed expert's
    /// gate and up projections become one fused weight, and how per-expert matrices become one
    /// stacked weight: the parts are named in the order the reading operator expects.
    Parts {
        /// The checkpoint tensors, in fused order.
        parts: Vec<String>,
        /// The fused tensor's extents.
        shape: Shape,
    },
}

/// One weight the model asks the loader for.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WeightRequest {
    /// The logical tensor's name; part of the retained key and of every diagnostic.
    pub tensor: String,
    /// Where the bytes come from.
    pub source: WeightSource,
    /// The arrangement the reading operator declares.
    pub layout: WeightLayout,
    /// The logical extents after conversion.
    pub shape: Shape,
}

impl WeightRequest {
    /// A weight read unchanged from `shape` elements starting at element `start` of `tensor`.
    pub fn region(tensor: impl Into<String>, start: u64, shape: Shape) -> Self {
        let tensor = tensor.into();
        Self {
            source: WeightSource::Region {
                tensor: tensor.clone(),
                start,
                shape,
            },
            tensor,
            layout: WeightLayout::ROW_MAJOR,
            shape,
        }
    }

    /// A weight gathered from `tensor`'s rows, each row starting `stride` elements after the last.
    pub fn strided(tensor: impl Into<String>, start: u64, shape: Shape, row_stride: u64) -> Self {
        let tensor = tensor.into();
        Self {
            source: WeightSource::Strided {
                tensor: tensor.clone(),
                start,
                shape,
                row_stride,
            },
            tensor,
            layout: WeightLayout::ROW_MAJOR,
            shape,
        }
    }

    /// A weight fused from `parts`, viewed as `shape`, in `layout`.
    pub fn parts(
        tensor: impl Into<String>,
        parts: Vec<String>,
        shape: Shape,
        layout: WeightLayout,
    ) -> Self {
        Self {
            tensor: tensor.into(),
            source: WeightSource::Parts { parts, shape },
            layout,
            shape,
        }
    }

    /// The same request in another layout.
    pub fn with_layout(mut self, layout: WeightLayout) -> Self {
        self.layout = layout;
        self
    }
}

/// The region of a logical tensor a retained weight occupies.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct WeightRegion {
    /// The checkpoint tensors' names, in fused order (one name for a plain region).
    pub tensors: Vec<String>,
    /// The first element of the region in the logical tensor.
    pub start: u64,
    /// The logical tensor's extents.
    pub shape: Shape,
    /// The source rows' stride in elements, when the region is a strided sub-matrix. Two gatherings
    /// of one tensor differ in it, so they cannot collide in retained storage.
    pub stride: Option<u64>,
}

impl fmt::Display for WeightRegion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.tensors.join("+"))?;
        write!(
            formatter,
            "[{}..{}]",
            self.start,
            self.start + self.shape.numel()
        )?;
        write!(formatter, "{:?}", self.shape.dims())?;
        if let Some(stride) = self.stride {
            write!(formatter, "|stride_{stride}")?;
        }
        Ok(())
    }
}

/// What retained storage files a weight under (specs/worker.md 8).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct WeightKey {
    /// The identity of every checkpoint file the weight was read from, in part order.
    pub files: Vec<FileIdentity>,
    /// The element type of the retained bytes.
    pub dtype: DType,
    /// The layout of the retained bytes.
    pub layout: WeightLayout,
    /// The region of the logical tensor.
    pub region: WeightRegion,
}

impl fmt::Display for WeightKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let files = self
            .files
            .iter()
            .map(FileIdentity::as_str)
            .collect::<Vec<_>>()
            .join("+");
        write!(
            formatter,
            "{files}|{}|{}|{}",
            self.dtype, self.layout, self.region
        )
    }
}

/// Retained storage, as the loader publishes converted weights to it.
///
/// The executor implements this over the Storage Server and its read-only device mappings: a
/// weight already retained under the key is handed back without reading the checkpoint again, and a
/// converted weight is published once (specs/worker.md 8, "只转换一次").
pub trait RetainedWeights {
    /// The weight retained under `key`, when the store already holds it.
    fn retained(&mut self, key: &WeightKey) -> Result<Option<Backing>>;

    /// Publish `bytes` as the storage of `key`, with `shape` elements of `dtype`.
    ///
    /// The store keeps the bytes and answers the read-only region a layer's view is carved from.
    /// Publishing the same key twice is the caller's error; the store may reject it or answer the
    /// existing region.
    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> Result<Backing>;
}

/// A calibrated NVFP4 matrix, as the `nvfp4_gemm` operator reads it.
///
/// The two weights are the encoded fields of the checkpoint, retained like any other weight; the
/// global scale is a plain value because it enters a kernel as a launch scalar.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedWeight {
    /// `[rows, columns / 2]` U8: two E2M1 codes a byte, packed along the contraction.
    pub packed: Weight,
    /// `ScaleLayout::shape(rows, columns)` U8: one E4M3 block scale per sixteen elements.
    pub scales: Weight,
    /// The reciprocal of the checkpoint's FP32 tensor scale.
    pub global_scale: f32,
    /// The layout the block scales are stored in.
    pub layout: ScaleLayout,
}

/// A weight a layer reads: its view and the region that keeps the bytes alive.
#[derive(Clone, Copy, Debug)]
pub struct Weight {
    /// The read-only region the view addresses.
    pub backing: Backing,
    /// The view the operator reads.
    pub view: TensorView,
}

impl Weight {
    /// The operator-facing view.
    pub fn view(&self) -> TensorView {
        self.view
    }

    /// The weight's element type.
    pub fn dtype(&self) -> DType {
        self.view.dtype
    }

    /// The weight's extents.
    pub fn shape(&self) -> Shape {
        self.view.shape
    }
}

/// The boundary between a checkpoint and retained storage: converts a weight once and publishes it.
pub struct WeightLoader<'a> {
    checkpoint: &'a Checkpoint,
    store: &'a mut dyn RetainedWeights,
    dtype: Option<DType>,
    quantization: Option<Quantization>,
}

impl<'a> WeightLoader<'a> {
    /// A loader over `checkpoint` publishing to `store`.
    ///
    /// `dtype` and `quantization` are the launch's explicit precision choices; with neither, a
    /// weight keeps the checkpoint's representation (specs/worker.md 1).
    pub fn new(
        checkpoint: &'a Checkpoint,
        store: &'a mut dyn RetainedWeights,
        dtype: Option<DType>,
        quantization: Option<Quantization>,
    ) -> Self {
        Self {
            checkpoint,
            store,
            dtype,
            quantization,
        }
    }

    /// Load `request`, converting and publishing it the first time and reusing it afterwards.
    pub fn load(&mut self, request: &WeightRequest) -> Result<Weight> {
        if self.quantization.is_some() {
            return Err(Error::Unsupported(format!(
                "quantization `{}` is converted at load only for the FP8 and NVFP4 step; the BF16 \
                 slice keeps each checkpoint's own representation",
                self.quantization
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            )));
        }

        let (names, start, source_shape, stride) = match &request.source {
            WeightSource::Region {
                tensor,
                start,
                shape,
            } => (vec![tensor.clone()], *start, *shape, None),
            WeightSource::Strided {
                tensor,
                start,
                shape,
                row_stride,
            } => (vec![tensor.clone()], *start, *shape, Some(*row_stride)),
            WeightSource::Parts { parts, shape } => (parts.clone(), 0, *shape, None),
        };
        if let WeightSource::Parts { shape, .. } = &request.source
            && !request.shape.matches(shape)
        {
            return Err(Error::weight(
                &request.tensor,
                format!(
                    "the request asks for {:?} but its parts are fused as {shape:?}",
                    request.shape.dims()
                ),
            ));
        }

        let files = self.identities(&names)?;
        let source_dtype = self.checkpoint_dtype(&names)?;
        let dtype = self.target_dtype(&names)?;
        let region = WeightRegion {
            tensors: names.clone(),
            start,
            shape: source_shape,
            stride,
        };
        let key = WeightKey {
            files,
            dtype,
            layout: request.layout,
            region,
        };

        if let Some(backing) = self.store.retained(&key)? {
            return Ok(Weight {
                view: backing.view(dtype, request.shape)?,
                backing,
            });
        }

        let bytes = self.source_bytes(&request.source, &key)?;
        // The key carries the retained element type; the bytes still hold the checkpoint's, so the
        // conversion is decided by the source, not by the key.
        let bytes = if source_dtype != dtype {
            convert(source_dtype, dtype, &bytes, &request.tensor)?
        } else {
            bytes
        };
        let expected = request
            .shape
            .numel()
            .checked_mul(dtype.bytes().ok_or_else(|| {
                Error::weight(
                    &request.tensor,
                    format!("{dtype} has no whole-byte element size"),
                )
            })?)
            .ok_or_else(|| Error::weight(&request.tensor, "the weight's bytes overflow"))?;
        if bytes.len() as u64 != expected {
            return Err(Error::weight(
                &request.tensor,
                format!(
                    "{} bytes were read, {expected} expected for {:?} of {dtype}",
                    bytes.len(),
                    request.shape.dims()
                ),
            ));
        }

        check_layout(&request.tensor, &request.source, request.layout)?;
        let backing = self.store.publish(&key, dtype, request.shape, &bytes)?;
        Ok(Weight {
            view: backing.view(dtype, request.shape)?,
            backing,
        })
    }

    /// Publish the weight-normalized form of `v` and `g` as one weight.
    ///
    /// A checkpoint stores a weight-normalized convolution as two tensors: `v` with the weight's
    /// extents and `g` with one magnitude per output channel (PyTorch keeps it as `[channels, 1,
    /// 1]`). The weight the convolution reads is `v · g / ‖v‖`, where the norm reduces every axis
    /// but the first per channel; spec 9 computes it once at loading, so a loaded model never
    /// carries the pair. The computation is FP32 and the result keeps the checkpoint's element type
    /// unless the loader was given another.
    ///
    /// `name` is the logical tensor the key records; the key also names both sources, so it cannot
    /// be confused with a plain region of either.
    pub fn weight_norm(&mut self, name: &str, v: &str, g: &str) -> Result<Weight> {
        let value = self.checkpoint.read(v)?;
        let gain = self.checkpoint.read(g)?;
        if value.shape.rank() == 0 {
            return Err(Error::weight(
                name,
                "a weight-normalized tensor has an axis".to_owned(),
            ));
        }
        let channels = value.shape.dim(0);
        if gain.shape.numel() != channels {
            return Err(Error::weight(
                name,
                format!(
                    "a gain of {} elements for {channels} output channels",
                    gain.shape.numel()
                ),
            ));
        }
        let dtype = self.dtype.unwrap_or(value.dtype);
        if !dtype.is_float() || !value.dtype.is_float() {
            return Err(Error::weight(
                name,
                format!(
                    "weight normalization of {} is not floating point",
                    value.dtype
                ),
            ));
        }
        let names = vec![v.to_owned(), g.to_owned()];
        let key = WeightKey {
            files: self.identities(&names)?,
            dtype,
            layout: WeightLayout::ROW_MAJOR,
            region: WeightRegion {
                // The logical name first: the double underscore marks a computed weight, so the key
                // is a region neither source names on its own.
                tensors: vec![format!("{name}__weight_norm"), v.to_owned(), g.to_owned()],
                start: 0,
                shape: value.shape,
                stride: None,
            },
        };
        if let Some(backing) = self.store.retained(&key)? {
            return Ok(Weight {
                view: backing.view(dtype, value.shape)?,
                backing,
            });
        }

        let values = to_f32(value.dtype, &value.bytes, v)?;
        let gains = to_f32(gain.dtype, &gain.bytes, g)?;
        let per_channel = usize::try_from(value.shape.numel() / channels.max(1))
            .map_err(|_| Error::weight(name, "more elements than a usize".to_owned()))?;
        let mut normalized = Vec::with_capacity(values.len());
        for (channel, gain) in gains.iter().enumerate() {
            let rows = &values[channel * per_channel..][..per_channel];
            let sum = rows.iter().map(|value| value * value).sum::<f32>();
            let scale = gain / sum.sqrt();
            normalized.extend(rows.iter().map(|value| value * scale));
        }
        let bytes = from_f32(dtype, &normalized, name)?;
        let backing = self.store.publish(&key, dtype, value.shape, &bytes)?;
        Ok(Weight {
            view: backing.view(dtype, value.shape)?,
            backing,
        })
    }

    /// Borrow this loader for another checkpoint: the same store, the same conversion, a
    /// different shard set.
    ///
    /// A pipeline's components each keep their own index in their own directory (`transformer/`,
    /// `vae/`, `text_encoder/`), so a model crate opens one [`Checkpoint`] per component and reads
    /// each through one loader. Retained storage stays the loader's, so a weight converted once for
    /// one component is reused for another and never converted twice (specs/worker.md 8).
    pub fn for_checkpoint<'b>(&'b mut self, checkpoint: &'b Checkpoint) -> WeightLoader<'b> {
        WeightLoader {
            checkpoint,
            store: self.store,
            dtype: self.dtype,
            quantization: self.quantization,
        }
    }

    /// Load a calibrated NVFP4 matrix: its packed E2M1 values, its block scales and its tensor
    /// scale.
    ///
    /// A calibrated checkpoint already holds the encoded fields (`values`, `block_scale`,
    /// `tensor_scale`, `uniserve/quantization/quantizer.py`), so nothing is quantized here: the
    /// three are published through retained storage under keys naming the scale layout, and a
    /// deployment that asks to quantize a BF16 checkpoint at load does it with the
    /// `nvfp4_quantize` operator on the device, which is where the reference encodes too.
    ///
    /// `values` is `[rows, columns / 2]` U8 (two E2M1 codes a byte), `scales` holds the block
    /// scales in the layout `ops::fp4::ScaleLayout::shape(rows, columns)` names — U8, as every
    /// program reads them — and `global_scale` is the reciprocal of the checkpoint's FP32 tensor
    /// scale, the scalar `nvfp4_quantize` takes. The product op's `alpha` is the two tensor
    /// scales' product, which the runner copies into one device scalar once at load.
    pub fn nvfp4(
        &mut self,
        values: &WeightRequest,
        scales: &WeightRequest,
        layout: ScaleLayout,
        global_scale: f32,
    ) -> Result<QuantizedWeight> {
        if !(global_scale.is_finite() && global_scale > 0.0) {
            return Err(Error::weight(
                &values.tensor,
                format!("a global scale of {global_scale} is not a positive number"),
            ));
        }
        if values.shape.last() == 0 {
            return Err(Error::weight(
                &values.tensor,
                "a packed NVFP4 matrix has a contraction axis".to_owned(),
            ));
        }
        let rows = values.shape.numel() / values.shape.last();
        let columns = values.shape.last() * 2;
        let expected = layout.shape(rows, columns);
        if scales.shape != expected {
            return Err(Error::weight(
                &scales.tensor,
                format!(
                    "the {layout} block scales of a [{rows}, {columns}] matrix are {:?}, not {:?}",
                    expected.dims(),
                    scales.shape.dims()
                ),
            ));
        }
        let scale_layout = match layout {
            ScaleLayout::Linear => WeightLayout::NVFP4_SCALES_LINEAR,
            ScaleLayout::Swizzled128X4 => WeightLayout::NVFP4_SCALES_SWIZZLED,
        };
        let mut values_request = values.clone().with_layout(WeightLayout::NVFP4_VALUES);
        let mut scales_request = scales.clone().with_layout(scale_layout);
        values_request.shape = Shape::new(&[rows, columns / 2]);
        scales_request.shape = expected;
        let packed = self.load(&values_request)?;
        if packed.dtype() != DType::U8 {
            return Err(Error::weight(
                &values_request.tensor,
                format!("packed E2M1 values are U8 bytes, not {}", packed.dtype()),
            ));
        }
        let scales = self.load(&scales_request)?;
        if scales.dtype() != DType::U8 {
            return Err(Error::weight(
                &scales_request.tensor,
                format!("block scales are U8 bytes, not {}", scales.dtype()),
            ));
        }
        Ok(QuantizedWeight {
            packed,
            scales,
            global_scale,
            layout,
        })
    }

    /// Publish host-computed bytes as a weight, without a checkpoint tensor behind them.
    ///
    /// The rotary frequency table is the first such weight: `inv_freq` is computed once at loading
    /// from `rope_theta` and `rope_scaling` (specs/worker.md 9), and the `rotary_factors` operator
    /// reads it as a device weight, so it takes the same retained-storage path as a checkpoint
    /// weight. The key carries no file identity — there is no file — and `name` identifies the
    /// constant, so two calls with the same name, element type, extents and bytes reuse one
    /// retained region and a different constant needs a different name.
    pub fn constant(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> Result<Weight> {
        let element = dtype.bytes().ok_or_else(|| {
            Error::weight(name, format!("{dtype} has no whole-byte element size"))
        })?;
        let expected = shape
            .numel()
            .checked_mul(element)
            .ok_or_else(|| Error::weight(name, "the constant's bytes overflow"))?;
        if bytes.len() as u64 != expected {
            return Err(Error::weight(
                name,
                format!(
                    "{} bytes for {:?} of {dtype}, {expected} expected",
                    bytes.len(),
                    shape.dims()
                ),
            ));
        }
        let key = WeightKey {
            files: Vec::new(),
            dtype,
            layout: WeightLayout::ROW_MAJOR,
            region: WeightRegion {
                tensors: vec![name.to_owned()],
                start: 0,
                shape,
                stride: None,
            },
        };
        let backing = match self.store.retained(&key)? {
            Some(backing) => backing,
            None => self.store.publish(&key, dtype, shape, bytes)?,
        };
        Ok(Weight {
            view: backing.view(dtype, shape)?,
            backing,
        })
    }

    /// The identity of each of `names`' files, in order.
    fn identities(&self, names: &[String]) -> Result<Vec<FileIdentity>> {
        names
            .iter()
            .map(|name| {
                self.checkpoint
                    .identity(name)
                    .cloned()
                    .ok_or_else(|| Error::MissingTensor(name.clone()))
            })
            .collect()
    }

    /// The element type every named tensor has in the checkpoint.
    fn checkpoint_dtype(&self, names: &[String]) -> Result<DType> {
        let mut dtype = None;
        for name in names {
            let location = self
                .checkpoint
                .tensor(name)
                .ok_or_else(|| Error::MissingTensor(name.clone()))?;
            match dtype {
                None => dtype = Some(location.dtype),
                Some(seen) if seen != location.dtype => {
                    return Err(Error::weight(
                        name,
                        format!(
                            "the fused parts mix {seen} and {source} elements",
                            source = location.dtype
                        ),
                    ));
                }
                Some(_) => {}
            }
        }
        dtype.ok_or_else(|| Error::weight("", "a weight has no source tensor"))
    }

    /// The element type the weight is retained as.
    fn target_dtype(&self, names: &[String]) -> Result<DType> {
        let checkpoint = self.checkpoint_dtype(names)?;
        let dtype = self.dtype.unwrap_or(checkpoint);
        // A representation that is not converted (BF16 kept as BF16, a packed NVFP4 field of
        // bytes kept as bytes) is read as it is; only a stated change goes through `convert`.
        if dtype == checkpoint {
            return Ok(dtype);
        }
        if !dtype.is_float() || !checkpoint.is_float() {
            return Err(Error::weight(
                names.first().map(String::as_str).unwrap_or(""),
                format!("converting {checkpoint} to {dtype} is not a floating-point conversion"),
            ));
        }
        Ok(dtype)
    }

    /// The bytes a source names, in the source's logical order.
    fn source_bytes(&self, source: &WeightSource, key: &WeightKey) -> Result<Vec<u8>> {
        match source {
            WeightSource::Region {
                tensor,
                start,
                shape,
            } => {
                let location = self
                    .checkpoint
                    .tensor(tensor)
                    .ok_or_else(|| Error::MissingTensor(tensor.clone()))?;
                let bytes = self.checkpoint.bytes(location)?.to_vec();
                let element = location.dtype.bytes().ok_or_else(|| {
                    Error::weight(tensor, format!("{} has no whole-byte size", location.dtype))
                })?;
                let begin = usize::try_from(start * element).map_err(|_| {
                    Error::weight(tensor, "the region's first element is past usize::MAX")
                })?;
                let end = usize::try_from((start + shape.numel()) * element).map_err(|_| {
                    Error::weight(tensor, "the region's last element is past usize::MAX")
                })?;
                if end > bytes.len() {
                    return Err(Error::weight(
                        tensor,
                        format!(
                            "elements {start}..{} of {:?} reach past the tensor's {}",
                            start + shape.numel(),
                            location.shape.dims(),
                            bytes.len()
                        ),
                    ));
                }
                Ok(bytes[begin..end].to_vec())
            }
            WeightSource::Strided {
                tensor,
                start,
                shape,
                row_stride,
            } => {
                let location = self
                    .checkpoint
                    .tensor(tensor)
                    .ok_or_else(|| Error::MissingTensor(tensor.clone()))?;
                let element = location.dtype.bytes().ok_or_else(|| {
                    Error::weight(tensor, format!("{} has no whole-byte size", location.dtype))
                })?;
                if shape.rank() == 0 {
                    return Err(Error::weight(
                        tensor,
                        "a gathered weight has an axis".to_owned(),
                    ));
                }
                let row = shape.last();
                if *row_stride < row {
                    return Err(Error::weight(
                        tensor,
                        format!("a stride of {row_stride} overlaps rows of {row} elements"),
                    ));
                }
                let rows = shape.numel() / row.max(1);
                let bytes = self.checkpoint.bytes(location)?;
                let mut gathered =
                    Vec::with_capacity(usize::try_from(shape.numel() * element).unwrap_or(0));
                for index in 0..rows {
                    let begin = start + index * row_stride;
                    let from = usize::try_from(begin * element).map_err(|_| {
                        Error::weight(tensor, "a gathered row starts past usize::MAX")
                    })?;
                    let to = usize::try_from((begin + row) * element).map_err(|_| {
                        Error::weight(tensor, "a gathered row ends past usize::MAX")
                    })?;
                    if to > bytes.len() {
                        return Err(Error::weight(
                            tensor,
                            format!(
                                "row {index} (elements {begin}..{}) reaches past the tensor's {} bytes",
                                begin + row,
                                bytes.len()
                            ),
                        ));
                    }
                    gathered.extend_from_slice(&bytes[from..to]);
                }
                Ok(gathered)
            }
            WeightSource::Parts { parts, .. } => {
                let mut fused = Vec::new();
                let mut part_shape: Option<Shape> = None;
                for name in parts {
                    let location = self
                        .checkpoint
                        .tensor(name)
                        .ok_or_else(|| Error::MissingTensor(name.clone()))?;
                    if location.shape.rank() == 0 {
                        return Err(Error::weight(name, "a scalar cannot be fused"));
                    }
                    match part_shape {
                        None => part_shape = Some(location.shape),
                        Some(first) => {
                            let same_tail = first.dims().iter().skip(1).eq(location
                                .shape
                                .dims()
                                .iter()
                                .skip(1));
                            if !same_tail {
                                return Err(Error::weight(
                                    name,
                                    format!(
                                        "fusing {:?} with {:?} joins different tails",
                                        first.dims(),
                                        location.shape.dims()
                                    ),
                                ));
                            }
                        }
                    }
                    fused.extend_from_slice(self.checkpoint.bytes(location)?);
                }
                if fused.is_empty() {
                    return Err(Error::weight(
                        &key.region.tensors.join("+"),
                        "a fused weight has no part",
                    ));
                }
                Ok(fused)
            }
        }
    }
}

/// Verify the loader can produce `layout` from `source`.
fn check_layout(tensor: &str, source: &WeightSource, layout: WeightLayout) -> Result<()> {
    match layout {
        WeightLayout::ROW_MAJOR => Ok(()),
        WeightLayout::UP_GATE | WeightLayout::DOWN => match source {
            WeightSource::Parts { .. } => Ok(()),
            WeightSource::Region { .. } | WeightSource::Strided { .. } => Err(Error::weight(
                tensor,
                format!(
                    "layout {layout} fuses several projections; a single region cannot produce it"
                ),
            )),
        },
        // The NVFP4 fields are byte buffers: a packed matrix, a fused stack of packed experts or
        // the strided rows of either are all the same bytes to the product.
        WeightLayout::NVFP4_VALUES
        | WeightLayout::NVFP4_SCALES_LINEAR
        | WeightLayout::NVFP4_SCALES_SWIZZLED => Ok(()),
        other => Err(Error::weight(
            tensor,
            format!("layout `{other}` is not one this loader produces"),
        )),
    }
}

/// Convert the floating-point elements of `bytes` from `from` to `to`.
///
/// The checkpoint is stored in one of the three floating-point formats every model here uses, and
/// a conversion goes through `f32`: the design keeps a checkpoint's own representation unless the
/// launch states another (specs/worker.md 1), and an explicit statement is one conversion.
fn convert(from: DType, to: DType, bytes: &[u8], tensor: &str) -> Result<Vec<u8>> {
    match (from, to) {
        (DType::BF16, DType::F32)
        | (DType::BF16, DType::F16)
        | (DType::F16, DType::F32)
        | (DType::F16, DType::BF16)
        | (DType::F32, DType::BF16)
        | (DType::F32, DType::F16) => {}
        (same, target) if same == target => return Ok(bytes.to_vec()),
        (source, target) => {
            return Err(Error::weight(
                tensor,
                format!("converting {source} to {target} is not implemented"),
            ));
        }
    };
    let source_bytes = usize::try_from(from.bytes().unwrap_or(4)).unwrap_or(4);
    let target_bytes = usize::try_from(to.bytes().unwrap_or(4)).unwrap_or(4);
    if !bytes.len().is_multiple_of(source_bytes) {
        return Err(Error::weight(
            tensor,
            format!("{} bytes are not whole {from} elements", bytes.len()),
        ));
    }
    let count = bytes.len() / source_bytes;
    let mut converted = Vec::with_capacity(count * target_bytes);
    for index in 0..count {
        let chunk = &bytes[index * source_bytes..][..source_bytes];
        let value = match from {
            DType::BF16 => {
                f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16)
            }
            DType::F16 => half_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            DType::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            other => {
                return Err(Error::weight(
                    tensor,
                    format!("converting {other} is not a floating-point conversion"),
                ));
            }
        };
        match to {
            DType::BF16 => {
                converted.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes())
            }
            DType::F16 => converted.extend_from_slice(&f32_to_half(value).to_le_bytes()),
            DType::F32 => converted.extend_from_slice(&value.to_le_bytes()),
            other => {
                return Err(Error::weight(
                    tensor,
                    format!("converting to {other} is not a floating-point conversion"),
                ));
            }
        }
    }
    Ok(converted)
}

/// The `f32` values of `bytes`, which hold `dtype` elements.
fn to_f32(dtype: DType, bytes: &[u8], tensor: &str) -> Result<Vec<f32>> {
    let converted = convert(dtype, DType::F32, bytes, tensor)?;
    Ok(converted
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// `values` as `dtype` elements.
fn from_f32(dtype: DType, values: &[f32], tensor: &str) -> Result<Vec<u8>> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    convert(DType::F32, dtype, &bytes, tensor)
}

/// `value` divided by `2^shift`, rounded to nearest with ties to even.
fn round_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    let quotient = value >> shift;
    let remainder = value & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if remainder > half || (remainder == half && quotient & 1 == 1) {
        quotient + 1
    } else {
        quotient
    }
}

/// The `f32` value of an IEEE half-precision bit pattern.
fn half_to_f32(half: u16) -> f32 {
    let sign = u32::from(half & 0x8000) << 16;
    let exponent = (half >> 10) & 0x1f;
    let mantissa = u32::from(half & 0x03ff);
    let bits = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                // A subnormal half is `mantissa · 2^-24`. Normalizing it left until the leading bit
                // reaches the half's implicit position gives the float's exponent and fraction.
                let leading = 31 - mantissa.leading_zeros();
                let shift = 10 - leading;
                let fraction = (mantissa << shift) & 0x03ff;
                let exponent = 127 - 15 + 1 - shift as i32;
                sign | ((exponent as u32) << 23) | (fraction << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// The IEEE half-precision bit pattern nearest to `value`.
fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0 {
        // A zero or subnormal `f32` underflows to a signed zero in the half's exponent range; the
        // largest half subnormal is still an order of magnitude above the largest `f32` subnormal.
        return sign;
    }
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    let unbiased = exponent as i32 - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        // A normal half: drop thirteen mantissa bits, round to nearest with ties to even, and let
        // the rounding carry into the exponent — a carry out of the largest finite half is the
        // infinity that nearest-rounding gives.
        let half_exponent = ((unbiased + 15) as u32) << 10;
        let total = half_exponent + round_even(mantissa, 13);
        return if total >= 0x7c00 {
            sign | 0x7c00
        } else {
            sign | total as u16
        };
    }
    // A subnormal half is a whole number of `2^-24` units: the significand at the float's own scale
    // (`1.m · 2^23`) divided by `2^(-unbiased - 1)`, rounded to nearest with ties to even.
    let significand = mantissa | 0x0080_0000;
    let shift = (-unbiased - 1) as u32;
    if shift > 24 {
        return sign;
    }
    let rounded = round_even(significand, shift);
    if rounded >= 0x0400 {
        // Rounded up out of the subnormals into the smallest normal.
        return sign | 0x0400;
    }
    sign | rounded as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_trips_the_values_a_checkpoint_holds() {
        // A subnormal half has an absolute precision of 2^-24, not a relative one, so the bound is
        // half a unit there and relative only from the smallest normal upward.
        const SMALLEST_NORMAL: f32 = 6.103_515_6e-5;
        const HALF_UNIT: f32 = 2.980_232_2e-8;
        for value in [0.0_f32, 1.0, -1.0, 0.5, 0.1, 65_504.0, 1e-5, 1.192_092_9e-7] {
            let back = half_to_f32(f32_to_half(value));
            let bound = if value.abs() < SMALLEST_NORMAL {
                HALF_UNIT
            } else {
                value.abs() * 1e-3
            };
            assert!((back - value).abs() <= bound, "{value:e} became {back:e}");
        }
    }

    #[test]
    fn a_half_conversion_holds_the_subnormal_range_and_the_specials() {
        // A quarter of the half's magnitude range is subnormal, and a checkpoint holds RMS weights
        // and epsilon-like constants there; the encoding is a whole number of 2^-24 units.
        for (value, units) in [
            (1e-5_f32, 168_u16),
            (5.960_464_5e-8, 1),
            (1.192_092_9e-7, 2),
            (6.097_555_6e-5, 1023),
        ] {
            assert_eq!(f32_to_half(value), units, "{value:e}");
            assert!(
                (half_to_f32(units) - value).abs() <= 5.960_464_5e-8,
                "{value:e}"
            );
        }
        // Below half a unit of the smallest subnormal the value is zero; past half a unit it is
        // one, and exactly half a unit ties between 0 and 1, where even wins.
        assert_eq!(f32_to_half(2.0e-8), 0);
        assert_eq!(f32_to_half(-2.0e-8), 0x8000);
        assert_eq!(f32_to_half(2.980_232_2e-8), 0);
        assert_eq!(f32_to_half(4.5e-8), 1);
        // A value that rounds up out of the subnormals becomes the smallest normal.
        assert_eq!(f32_to_half(6.101e-5), 0x0400);
        // Ties round to even, and the largest finite magnitude's neighbourhood rounds to infinity.
        assert_eq!(half_to_f32(f32_to_half(65504.0)), 65504.0);
        assert_eq!(f32_to_half(65_520.0), 0x7c00);
        assert_eq!(f32_to_half(70_000.0), 0x7c00);
        assert!(f32_to_half(f32::NAN) & 0x7c00 == 0x7c00);
        assert!(f32_to_half(f32::NAN) & 0x03ff != 0);
        assert_eq!(f32_to_half(f32::INFINITY), 0x7c00);
        assert_eq!(half_to_f32(0x8000), -0.0);
    }

    #[test]
    fn a_key_names_the_checkpoint_the_type_the_layout_and_the_region() {
        let key = WeightKey {
            files: vec![
                FileIdentity::of(std::path::Path::new("/dev/null")).unwrap_or_else(|_| {
                    FileIdentity::of(std::path::Path::new("/proc/self/exe")).unwrap()
                }),
            ],
            dtype: DType::BF16,
            layout: WeightLayout::UP_GATE,
            region: WeightRegion {
                tensors: vec!["up".to_owned(), "gate".to_owned()],
                start: 0,
                shape: Shape::new(&[128, 1536, 2048]),
                stride: None,
            },
        };
        let text = key.to_string();
        assert!(text.contains("bf16"));
        assert!(text.contains("up_gate"));
        assert!(text.contains("up+gate"));
    }
}

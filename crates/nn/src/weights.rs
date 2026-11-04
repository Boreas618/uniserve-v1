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
        write!(formatter, "{:?}", self.shape.dims())
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

        let (names, start, source_shape) = match &request.source {
            WeightSource::Region {
                tensor,
                start,
                shape,
            } => (vec![tensor.clone()], *start, *shape),
            WeightSource::Parts { parts, shape } => (parts.clone(), 0, *shape),
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
        let dtype = self.target_dtype(&names)?;
        let region = WeightRegion {
            tensors: names.clone(),
            start,
            shape: source_shape,
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
        let bytes = match self.dtype {
            Some(target) if target != key.dtype => {
                convert(key.dtype, target, &bytes, &request.tensor)?
            }
            _ => bytes,
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
            WeightSource::Region { .. } => Err(Error::weight(
                tensor,
                format!(
                    "layout {layout} fuses several projections; a single region cannot produce it"
                ),
            )),
        },
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
            DType::BF16 => converted.extend_from_slice(&(value.to_bits() >> 16).to_le_bytes()),
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
                // A subnormal half is a normal float scaled by 2^-24.
                let mut exponent = 127 - 15 - 10;
                let mut mantissa = mantissa;
                while mantissa & 0x0400 == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                sign | ((exponent as u32) << 23) | ((mantissa & 0x03ff) << 13)
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
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    // Round to nearest, ties to even, in the half's precision.
    let unbiased = exponent as i32 - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        let subnormal = (mantissa | 0x0080_0000) >> ((-14 - unbiased + 1) as u32);
        let rounded = (subnormal + 1) >> 1;
        return sign | rounded as u16;
    }
    let half_exponent = ((unbiased + 15) as u16) << 10;
    let rounded = mantissa + 0x1000;
    sign | half_exponent | ((rounded >> 13) as u16 & 0x03ff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_trips_the_values_a_checkpoint_holds() {
        for value in [0.0_f32, 1.0, -1.0, 0.5, 0.1, 65_504.0, 1e-5] {
            let half = f32_to_half(value);
            let back = half_to_f32(half);
            assert!(
                (back - value).abs() <= value.abs() * 1e-3,
                "{value} became {back}"
            );
        }
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
            },
        };
        let text = key.to_string();
        assert!(text.contains("bf16"));
        assert!(text.contains("up_gate"));
        assert!(text.contains("up+gate"));
    }
}

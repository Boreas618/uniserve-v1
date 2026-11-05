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

use std::collections::BTreeMap;
use std::fmt;

use ops::fp4::ScaleLayout;
use ops::{DType, Shape, TensorView};
use sha2::{Digest, Sha256};

use crate::checkpoint::{Checkpoint, FileIdentity};
use crate::error::{Error, Result};
use crate::tensor::{Backing, DevicePtr};

/// The suffix one export spells each field of a calibrated NVFP4 projection with (specs/worker.md
/// 8, 9).
///
/// A calibrated checkpoint states a projection's fields as four tensor names that differ only in
/// their suffix, and the suffix set is a property of the export, not of the format: two calibration
/// runs of the same model written by different toolchains name the same three numbers differently.
/// The loader is told which vocabulary its checkpoint speaks, because reading it as the other one
/// would either fail or — worse — accept a half-converted checkpoint whose plain `weight` is a
/// BF16 matrix beside a `weight_scale` nothing normalizes.
///
/// The two constructors are the export shapes this tree meets. **`weight_scale` is the same name in
/// both and means the same thing** — the K16 block scales — so it is not a mistake that the two
/// disagree about the packed values and the per-tensor scale only; renaming it in either
/// constructor would be. The per-tensor scale is `weight_scale_2` in the vLLM spelling and
/// `weight_tensor_scale` in the diffusers one, and an export that carries no activation scale says
/// so by having no tensor under [`Nvfp4Names::input_scale`], which the loader reports as this
/// vocabulary rather than as a missing name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Nvfp4Names {
    /// What this vocabulary is called, in refusals: the export that writes it.
    pub export: &'static str,
    /// The packed E2M1 values, two codes a byte.
    pub values: &'static str,
    /// The K16 block scales, one byte each in `F8_E4M3` or `U8`.
    pub block_scale: &'static str,
    /// The per-tensor scale the block scales are normalized against.
    pub tensor_scale: &'static str,
    /// The calibration's activation scale, when the export states one.
    pub input_scale: &'static str,
}

impl Nvfp4Names {
    /// The vLLM/ModelOpt spelling: `weight`, `weight_scale`, `weight_scale_2`, `input_scale`.
    ///
    /// This is what a Qwen3-MoE calibrated checkpoint states, and what
    /// `uniserve/quantization/quantizer.py` writes.
    pub const fn vllm() -> Self {
        Self {
            export: "vLLM ModelOpt",
            values: "weight",
            block_scale: "weight_scale",
            tensor_scale: "weight_scale_2",
            input_scale: "input_scale",
        }
    }

    /// The diffusers/ModelOpt spelling: `weight_packed`, `weight_scale`, `weight_tensor_scale`,
    /// `input_scale`.
    ///
    /// This is what the FastH3 export states, in every component that is quantized (its
    /// `transformer` and `vae` state 100 and 216 packed projections respectively), and its VAE adds
    /// `scale1`/`scale2` beside them for the layers ModelOpt calibrated in a second scheme.
    pub const fn diffusers() -> Self {
        Self {
            export: "diffusers ModelOpt",
            values: "weight_packed",
            block_scale: "weight_scale",
            tensor_scale: "weight_tensor_scale",
            input_scale: "input_scale",
        }
    }
}

impl fmt::Display for Nvfp4Names {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} (`{}`, `{}`, `{}`)",
            self.export, self.values, self.block_scale, self.tensor_scale
        )
    }
}

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

    /// The same values after `shuffleMatrixA`: rows permuted inside blocks of sixteen or
    /// thirty-two, which is the arrangement the trtllm-gen products read (`useShuffledMatrix`).
    pub const NVFP4_VALUES_SHUFFLED: Self = Self("nvfp4_values_shuffled");

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

impl Quantization {
    /// The element type this policy's encoded weights are stored in.
    pub const fn encoding(self) -> DType {
        match self {
            Self::Fp8 | Self::Mxfp8 => DType::F8E4M3,
            Self::NvFp4 => DType::U8,
        }
    }
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
    /// The element type this tensor is read as whatever the deployment states, when it has one.
    ///
    /// A representation the model pins per tensor - the reference keeps a timestep embedding in
    /// FP32 and a modulation projection in BF16 at every preset (`weights.py:186-224`) - cannot be
    /// expressed by the loader's policy, which is one value for every weight it reads. A pin wins
    /// over the policy, and it is part of the retained key: two loaders of one checkpoint that read
    /// one tensor at two types retain two different weights.
    pub pinned: Option<DType>,
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
            pinned: None,
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
            pinned: None,
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
            pinned: None,
        }
    }

    /// The same request read as `dtype`, whatever the loader's policy states.
    pub fn pinned(mut self, dtype: DType) -> Self {
        self.pinned = Some(dtype);
        self
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

/// What produced a weight that no checkpoint file names.
///
/// A checkpoint weight is named by the files it was read from and its region in them. A computed
/// one — a rotary frequency table, fused output heads, a modulation table — has no file, so the only
/// thing that can name it is the computation: the class that ran it, and the numbers it produced.
/// A label is not an identity: two models that name their own fused heads
/// `denoiser.video.student_heads.weight` are two producers, and a key holding only the name lets the
/// second one map the first one's bytes, silently, under an identical extent.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Provenance {
    /// The class that computed the weight, as the crate that placed it names itself.
    pub producer: &'static str,
    /// A digest of the bytes it produced: the same numbers are one weight under any name, and
    /// different numbers are two.
    pub content: [u8; 32],
}

impl Provenance {
    /// The provenance of a weight a class computed from `bytes`, the numbers it produced.
    pub fn computed(producer: &'static str, bytes: &[u8]) -> Self {
        let mut digest = Sha256::new();
        digest.update(bytes);
        Self {
            producer,
            content: digest.finalize().into(),
        }
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "computed:{}:", self.producer)?;
        for byte in self.content {
            write!(formatter, "{byte:02x}")?;
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
    /// What produced the bytes, for a weight no checkpoint file names.
    ///
    /// `None` for a weight read from a checkpoint, whose files and region are its provenance; a
    /// computed weight states the class that computed it and a digest of the numbers it produced,
    /// so two producers that compute different tables under one name cannot share retained bytes.
    pub provenance: Option<Provenance>,
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
        )?;
        if let Some(provenance) = &self.provenance {
            write!(formatter, "|{provenance}")?;
        }
        Ok(())
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

/// A retained store in host memory: a model loaded, and asked for what it would launch, with no
/// device.
///
/// A loading audit — which programs a model's request space needs, which keys its layers build, what
/// a checkpoint's mapping produces — is a question about shapes, dtypes and layouts. None of it needs
/// a device: the checkpoint is host-mapped, `WeightLoader` converts on the host, and a store that
/// keeps the bytes in host memory is enough to build every layer and ask each for its descriptor.
///
/// The addresses it answers are **host** addresses, which is why the type is named for the host and
/// not for storage in general: a view carved from one is a shape and an element type, and a call that
/// launches it would hand a host pointer to a device. A model built over this store is for
/// enumeration, never for execution.
pub struct HostStore {
    /// One buffer per published region, eight-byte aligned so any element type can be read from it.
    buffers: Vec<Vec<u64>>,
    /// The regions, in publication order, with the key each was published under.
    published: Vec<(WeightKey, Backing)>,
}

impl HostStore {
    /// An empty store.
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            published: Vec::new(),
        }
    }

    /// How many regions the store holds.
    pub fn len(&self) -> usize {
        self.published.len()
    }

    /// Whether the store holds nothing.
    pub fn is_empty(&self) -> bool {
        self.published.is_empty()
    }

    /// The keys of every published region, in publication order.
    pub fn keys(&self) -> impl Iterator<Item = &WeightKey> {
        self.published.iter().map(|(key, _)| key)
    }
}

impl Default for HostStore {
    fn default() -> Self {
        Self::new()
    }
}

impl RetainedWeights for HostStore {
    fn retained(&mut self, key: &WeightKey) -> Result<Option<Backing>> {
        Ok(self
            .published
            .iter()
            .find(|(published, _)| published == key)
            .map(|(_, backing)| *backing))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        _dtype: DType,
        _shape: Shape,
        bytes: &[u8],
    ) -> Result<Backing> {
        // Publishing the same key twice answers the first region: a store that replaced it would
        // hand two layers two buffers for one weight, and the second load would silently win.
        if let Some((_, backing)) = self
            .published
            .iter()
            .find(|(published, _)| published == key)
        {
            return Ok(*backing);
        }
        let words = bytes.len().div_ceil(8);
        let mut buffer = vec![0_u64; words];
        // SAFETY: the buffer has been sized to hold `bytes.len()` bytes rounded up to whole words,
        // and both pointers are valid for that many bytes with no overlap: the source is the
        // caller's slice and the destination is a buffer this function owns.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        let address = DevicePtr::new(buffer.as_ptr() as u64);
        let backing = Backing::new(address, bytes.len() as u64);
        self.buffers.push(buffer);
        self.published.push((key.clone(), backing));
        Ok(backing)
    }
}

/// The arrangement of a calibrated matrix's packed E2M1 values.
///
/// The values are two codes a byte along the contraction whichever arrangement they are in; what
/// varies is the order of the **rows**, because the trtllm-gen products read their A operand with
/// `useShuffledMatrix`: rows permuted inside blocks of sixteen or thirty-two so that the tile a
/// thread block loads is contiguous. It is a permutation of the shard's own bytes, applied once at
/// load — never a step, and never a kernel — and a matrix whose row count is not a whole number of
/// blocks has no such arrangement, which the loader refuses by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueLayout {
    /// The shard's own row order.
    Plain,
    /// `shuffleMatrixA`'s row order, for a product whose epilogue tile is `epilogue_tile_m` rows.
    Shuffled {
        /// Rows one MMA tile of the product covers; the shuffle block is thirty-two rows when this
        /// is a whole number of 128s and sixteen otherwise, as the reference's
        /// `get_shuffle_block_size` states.
        epilogue_tile_m: u32,
    },
}

impl ValueLayout {
    /// The layout name a weight published in this arrangement carries.
    pub fn layout(self) -> WeightLayout {
        match self {
            Self::Plain => WeightLayout::NVFP4_VALUES,
            Self::Shuffled { .. } => WeightLayout::NVFP4_VALUES_SHUFFLED,
        }
    }

    /// The rows one shuffle block holds, or `None` when the arrangement is plain.
    pub fn block(self) -> Option<u64> {
        match self {
            Self::Plain => None,
            Self::Shuffled { epilogue_tile_m } => {
                Some(if epilogue_tile_m % 128 == 0 { 32 } else { 16 })
            }
        }
    }

    /// `bytes` with every expert's rows moved into this arrangement.
    ///
    /// `bytes` holds `experts` packed matrices of `rows` rows by `row_bytes` bytes, one after
    /// another; a plain arrangement is the bytes unchanged.
    fn arrange(self, bytes: &[u8], experts: u64, rows: u64, row_bytes: u64) -> Vec<u8> {
        let Some(block) = self.block() else {
            return bytes.to_vec();
        };
        let mut arranged = bytes.to_vec();
        let stride = row_bytes as usize;
        let lanes = block / 8;
        for expert in 0..experts {
            let base = expert * rows * row_bytes;
            for row in 0..rows {
                // The reference's maps in closed form: within its block, a source row `r` moves to
                // `(r % lanes) * 8 + r / lanes`, the interleave of `lanes` eight-row groups that the
                // artifact's `srcToDstBlk16RowMap` and `srcToDstBlk32RowMap` list. The block is
                // taken first: a row's offset inside its own block decides where it goes, and its
                // block decides where that block starts.
                let within = row % block;
                let destination = (row / block) * block + (within % lanes) * 8 + within / lanes;
                let source = (base + row * row_bytes) as usize;
                let target = (base + destination * row_bytes) as usize;
                arranged[target..target + stride].copy_from_slice(&bytes[source..source + stride]);
            }
        }
        arranged
    }
}

/// One expert's linear block scales in the 128x4 layout the products read.
///
/// The launcher's own `computeSFIndex` (TensorRT-LLM's `fp8Op.h`) written out: rows padded to 128,
/// scale columns padded to 4, and each `[128, 4]` tile laid out so that a thread block's scales are
/// contiguous — column groups of four stride 512, thirty-two-row groups stride 16, groups of 128 rows
/// stride `128 · padded_columns`. The padding is zero, which is a scale of zero on a block that does
/// not exist.
fn swizzled_scales(linear: &[u8], rows: u64, blocks: u64) -> Vec<u8> {
    let padded_rows = rows.div_ceil(128) * 128;
    let padded_columns = blocks.div_ceil(4) * 4;
    let mut swizzled = vec![0_u8; (padded_rows * padded_columns) as usize];
    for row in 0..padded_rows {
        for column in 0..padded_columns {
            if row >= rows || column >= blocks {
                continue;
            }
            let index = (column % 4)
                + (column / 4) * 512
                + (row % 32) * 16
                + ((row % 128) / 32) * 4
                + (row / 128) * (128 * padded_columns);
            swizzled[index as usize] = linear[(row * blocks + column) as usize];
        }
    }
    swizzled
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
    ///
    /// It enters a kernel as a launch scalar, which is why it is a plain value.
    pub global_scale: f32,
    /// The layout the block scales are stored in.
    pub layout: ScaleLayout,
    /// The reciprocal of the checkpoint's activation tensor scale, `[1]` F32, when the caller asked
    /// for it.
    ///
    /// A calibrated export states a static per-module `input_scale` for its activations, and an
    /// encoder needs it as a **device operand**: the product that consumes the encoding reads it as
    /// each thread block starts, so a host scalar would serialize the pair. It is calibration data
    /// of one tensor, invariant across requests, so it lives with the weights rather than in a
    /// descriptor or a request. `None` says the consumer is weight-only and asked for none, which is
    /// a different fact from a calibrated checkpoint that omitted it — that one is refused by
    /// tensor name.
    pub input_scale: Option<Weight>,
}

/// One packed table per projection: the axis order the batched-GEMM entry reads and the shape a
/// checkpoint carries a tensor for.
struct Projections {
    /// Projections per expert: two for a fused up/gate stack, one for a down stack.
    per: usize,
    /// Experts in the stack.
    experts: usize,
}

impl Projections {
    /// The order over `per` projections of `experts` experts, both as published extents.
    fn new(per: u64, experts: u64) -> Self {
        Self {
            per: usize::try_from(per).unwrap_or(0),
            experts: usize::try_from(experts).unwrap_or(0),
        }
    }

    /// How many scales the table holds.
    fn len(&self) -> usize {
        self.per * self.experts
    }

    /// `scales`, collected expert-major, as the table's little-endian f32 bytes.
    fn bytes(&self, scales: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.len() * 4);
        for projection in 0..self.per {
            for expert in 0..self.experts {
                bytes.extend_from_slice(&scales[expert * self.per + projection].to_le_bytes());
            }
        }
        bytes
    }
}

/// A calibrated NVFP4 **stack of experts**, as `gather_gemm_nvfp4` and `route_gemm_nvfp4` read it.
///
/// A model's expert block multiplies whole stacks: its weights are `[experts, rows, columns/2]` U8,
/// its block scales carry one entry per expert, and the tensor scale varies per expert **and per
/// projection** — so the scalar [`QuantizedWeight::global_scale`] cannot serve it, and a model may
/// not build the table itself (constructing a weight is the loader's, not a model's). The loader
/// publishes all three from the checkpoint's per-expert, per-projection tensors: the encoded values
/// and scales concatenated in the operator's reading order, and the reciprocals as a
/// `[projections, experts]` f32 weight — one packed table per projection, which is the shape the
/// batched-GEMM entry reads and the shape the checkpoint itself carries a tensor per.
///
/// The stack fused from an expert's `up_proj` and `gate_proj` is why the second axis exists: each of
/// those has its own `weight_scale_2` in the checkpoint, so one scalar per expert would dequantize
/// the gate half with the up half's scale and be wrong by the ratio of the two, silently.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedStack {
    /// `[experts, rows, columns/2]` U8: two E2M1 codes a byte, packed along the contraction.
    pub values: Weight,
    /// `[experts, ...]` in the checkpoint's scale type — `F8_E4M3`, or `U8` where the export labels
    /// the same bytes as raw — the block scales in [`Self::layout`], one per sixteen elements.
    /// Every expert of a stack agrees on it, and it is published unchanged: a model puts
    /// [`Self::scales`]`.dtype()` in the product's descriptor and nothing converts.
    pub scales: Weight,
    /// `[projections, experts]` f32, row-major, in the stack's own projection order: the reciprocal
    /// of each expert's **per-projection** checkpoint tensor scale.
    ///
    /// One packed table per projection rather than one row per expert, because that is what the
    /// batched-GEMM entry takes: it reads each projection's table as `experts` consecutive floats. A
    /// `[experts, projections]` table handed over as two rows would walk column 0 for even experts and
    /// column 1 for odd ones, which is a wrong dequantization with nothing to show for it.
    pub global_scale: Weight,
    /// The layout the block scales are stored in.
    pub layout: ScaleLayout,
    /// `[1]` f32: the activation encoder's global scale for this stack's activations — **1.0**.
    ///
    /// One, and published rather than defaulted at the call site, because of what the encoder's two
    /// levels do. With `s = max(|x / g|) / 6` and a code `round(x / (g · s))`, the effective step is
    /// `g · s = max(|x|) / 6`: **the code does not depend on `g`, and neither does the reconstruction
    /// `code · s' · token`, in which the encoder's own factor cancels.** So the calibrated activation
    /// scale changes no value in a dynamically quantized activation, and folding it into
    /// [`Self::global_scale`] would fold a factor into a table that has already absorbed it. The
    /// checkpoint's per-expert activation scales stay published beside the weights as the
    /// checkpoint's own data ([`Self::input_scale`]) and no operand multiplies a value by them.
    pub encode_scale: Weight,
    /// `[projections, experts]` f32, row-major, in the same order: the reciprocal of each expert's
    /// **per-projection** activation tensor scale, when the caller asked for it.
    ///
    /// A ModelOpt MoE export calibrates each expert's projections separately, so one activation
    /// scale per stack would encode every expert but one at the wrong magnitude. `None` says the
    /// consumer is weight-only; a calibrated checkpoint asked for this and missing it is refused by
    /// tensor name.
    pub input_scale: Option<Weight>,
}

/// Whether `dtype` is one of the two element types a block-scale field may have.
///
/// The block scales are eight bits per sixteen elements, and a checkpoint stores each as an E4M3
/// float — or, where an export labels the same bytes as raw, as `U8`. Both are one byte, both are
/// published unchanged, and nothing converts: a reader that multiplied a scale by a constant would
/// be wrong, and a loader that relabeled it would hide which one the checkpoint meant.
fn block_scale_dtype(tensor: &str, dtype: DType) -> Result<DType> {
    match dtype {
        DType::U8 | DType::F8E4M3 => Ok(dtype),
        other => Err(Error::weight(
            tensor,
            format!(
                "block scales are one byte each — F8_E4M3, or U8 where the export says so — not                  {other}"
            ),
        )),
    }
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
    /// The checkpoint the weights come from, or `None` for a loader that states shapes only.
    checkpoint: Option<&'a Checkpoint>,
    store: Option<&'a mut dyn RetainedWeights>,
    dtype: Option<DType>,
    quantization: Option<Quantization>,
    /// The checkpoint tensors' extents, when a shapes-only loader states them from a table instead of
    /// from a checkpoint's headers. A computed weight — the folded form of a weight-normalized
    /// convolution — has no request to take its extents from, so it is answered from here.
    extents: Option<&'a BTreeMap<String, (DType, Shape)>>,
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
            checkpoint: Some(checkpoint),
            store: Some(store),
            dtype,
            quantization,
            extents: None,
        }
    }

    /// A loader that states every weight's extents and reads nothing.
    ///
    /// `load` answers each request with a view of the requested element type and shape at address
    /// zero, `constant` with the shape it was given, and every path that would need bytes — a
    /// checkpoint read, a retained-store round trip, a block scale — refuses by name. The caller is a
    /// model's `load` driven by a **recording** forward (`Forward::recording`): the shapes a load
    /// derives are what decide the programs a model launches, and the values never reach a kernel. So
    /// a walk of a model's programs needs no shards and no device.
    /// The checkpoint this loader reads, when it reads one.
    ///
    /// A shapes-only loader has none: it states extents and reads nothing, so a caller that needs a
    /// fact only a shard holds must say what it does without it.
    pub fn checkpoint(&self) -> Option<&Checkpoint> {
        self.checkpoint
    }

    pub fn shapes(dtype: DType) -> Self {
        Self::stated(dtype, None)
    }

    /// A shapes-only loader that also states the checkpoint tensors' extents from `extents`.
    ///
    /// A weight a module merely reads takes its extents from the request, so a shapes-only loader needs
    /// no table for it. A weight **computed at load** does: the folded form of a weight-normalized
    /// convolution reduces `v · g / ‖v‖`, and `v`'s extents are a fact of the checkpoint's header. The
    /// table is that header's content — produced where the weights are, used where they are not — so a
    /// walk needs no shard while still stating the shapes the checkpoint really holds.
    pub fn stated(dtype: DType, extents: Option<&'a BTreeMap<String, (DType, Shape)>>) -> Self {
        Self {
            checkpoint: None,
            store: None,
            dtype: Some(dtype),
            quantization: None,
            extents,
        }
    }

    /// Whether this loader reads a checkpoint, or only states shapes.
    pub const fn is_shapes(&self) -> bool {
        self.checkpoint.is_none()
    }

    /// The checkpoint this loader reads.
    fn shards(&self) -> Result<&'a Checkpoint> {
        self.checkpoint.ok_or_else(|| {
            Error::Unsupported(format!(
                "this loader states weights' shapes and reads no checkpoint: {}",
                "a weight's bytes were asked for"
            ))
        })
    }

    /// The retained store this loader publishes into.
    fn retention(&mut self) -> Result<&mut (dyn RetainedWeights + 'a)> {
        match self.store.as_deref_mut() {
            Some(store) => Ok(store),
            None => Err(Error::Unsupported(
                "this loader states weights' shapes and retains nothing".to_owned(),
            )),
        }
    }

    /// Load `request`, converting and publishing it the first time and reusing it afterwards.
    pub fn load(&mut self, request: &WeightRequest) -> Result<Weight> {
        // A shapes-only loader states the extents a launch reads and reads no bytes at all.
        if self.is_shapes() {
            let dtype = request.pinned.or(self.dtype).ok_or_else(|| {
                Error::weight(
                    &request.tensor,
                    "a shapes-only loader has no checkpoint to read the element type from"
                        .to_owned(),
                )
            })?;
            let bytes = request
                .shape
                .numel()
                .checked_mul(dtype.bytes().unwrap_or(0))
                .ok_or_else(|| Error::weight(&request.tensor, "the weight's bytes overflow"))?;
            let backing = Backing::new(DevicePtr::new(0), bytes);
            return Ok(Weight {
                view: backing.view(dtype, request.shape)?,
                backing,
            });
        }
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
        let dtype = self.target_dtype(&names, request.pinned)?;
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

            provenance: None,
        };

        if let Some(backing) = self.retention()?.retained(&key)? {
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
        let backing = self
            .retention()?
            .publish(&key, dtype, request.shape, &bytes)?;
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
        if self.is_shapes() {
            // The folded weight keeps `v`'s extents and one gain per output channel; both are facts of
            // the checkpoint's header, which the table states.
            let extents = self.extents.ok_or_else(|| {
                Error::Unsupported(format!(
                    "{name}: this loader states shapes but holds no table of the checkpoint's \
                     extents, and a weight-normalized weight's extents are not in any request"
                ))
            })?;
            let (value_dtype, shape) = extents.get(v).copied().ok_or_else(|| {
                Error::Unsupported(format!(
                    "{name}: the extents table does not name `{v}`, which the folded weight is read from"
                ))
            })?;
            let (_, gain) = extents.get(g).copied().ok_or_else(|| {
                Error::Unsupported(format!(
                    "{name}: the extents table does not name `{g}`, which carries the gain"
                ))
            })?;
            if gain.numel() != shape.dim(0) {
                return Err(Error::weight(
                    name,
                    format!(
                        "a gain of {} elements for {} output channels",
                        gain.numel(),
                        shape.dim(0)
                    ),
                ));
            }
            let dtype = self.dtype.unwrap_or(value_dtype);
            let bytes = shape
                .numel()
                .checked_mul(dtype.bytes().unwrap_or(0))
                .ok_or_else(|| Error::weight(name, "the weight's bytes overflow".to_owned()))?;
            let backing = Backing::new(DevicePtr::new(0), bytes);
            return Ok(Weight {
                view: backing.view(dtype, shape)?,
                backing,
            });
        }
        let value = self.shards()?.read(v)?;
        let gain = self.shards()?.read(g)?;
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

            provenance: None,
        };
        if let Some(backing) = self.retention()?.retained(&key)? {
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
        let backing = self
            .retention()?
            .publish(&key, dtype, value.shape, &bytes)?;
        Ok(Weight {
            view: backing.view(dtype, value.shape)?,
            backing,
        })
    }

    /// Borrow this loader for another checkpoint: the same store, the same element-type and
    /// quantization policy, a different shard set.
    ///
    /// A pipeline root holds one component per subdirectory (`transformer/`, `vae/`,
    /// `text_encoder/`, …), each with its own index and its own file identities, so a model crate
    /// opens the root once and reads every component through one store: a weight converted once for
    /// one component is reused for another and never converted twice (specs/worker.md 8).
    ///
    /// The policy is the loader's, not the component's, so a component that its policy does not
    /// describe is refused here: an FP8-policy loader over a BF16 component, or a float policy over
    /// a component whose weights are an FP8 or NVFP4 encoding, would read weights as something they
    /// are not, which is a wrong answer rather than a convenience. A model that needs two policies
    /// builds two loaders over the same store.
    pub fn reborrow<'b>(&'b mut self, checkpoint: &'b Checkpoint) -> Result<WeightLoader<'b>> {
        self.check_policy(checkpoint)?;
        let dtype = self.dtype;
        let quantization = self.quantization;
        Ok(WeightLoader {
            checkpoint: Some(checkpoint),
            store: Some(self.retention()?),
            dtype,
            quantization,
            extents: None,
        })
    }

    /// Whether this loader's policy describes `checkpoint`'s weights.
    fn check_policy(&self, checkpoint: &Checkpoint) -> Result<()> {
        let held = checkpoint.dtypes();
        let listed = held
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        if let Some(quantization) = self.quantization {
            let encoding = quantization.encoding();
            if !held.contains(&encoding) {
                return Err(Error::policy(format!(
                    "a {quantization} policy encodes weights as {encoding}, and this checkpoint \
                     holds {listed}"
                )));
            }
        }
        if let Some(dtype) = self.dtype {
            for encoded in [DType::F8E4M3, DType::F4E2M1] {
                if held.contains(&encoded) {
                    return Err(Error::policy(format!(
                        "this checkpoint holds {encoded} encoded weights, which a {dtype} element \
                         type does not describe; the quantization is the loader's policy, not its \
                         element type"
                    )));
                }
            }
        }
        Ok(())
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
    /// `file` is the arrangement the checkpoint stores its block scales in and `product` the one the
    /// deployment's kernels read; when the two differ the table is permuted here, and
    /// [`QuantizedWeight::layout`] states the product's. `values` is `[rows, columns / 2]` U8 (two
    /// E2M1 codes a byte), `scales` holds the block scales in `file`'s arrangement, in whatever
    /// one-byte type the checkpoint stores them (`F8_E4M3`, or `U8` where the export says so;
    /// [`QuantizedWeight::scales`]`.dtype()` is that type, unchanged), and `global_scale` is the
    /// reciprocal of the checkpoint's FP32 tensor
    /// scale, the scalar `nvfp4_quantize` takes. The product op's `alpha` is the two tensor
    /// scales' product, which the runner copies into one device scalar once at load.
    // The two arrangements are separate arguments on purpose: one is the file's statement, the other
    // the deployment's, and naming both is what keeps a caller from passing one for the other.
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4(
        &mut self,
        values: &WeightRequest,
        scales: &WeightRequest,
        input_scale: Option<&WeightRequest>,
        names: Nvfp4Names,
        file: ScaleLayout,
        product: ScaleLayout,
        global_scale: f32,
    ) -> Result<QuantizedWeight> {
        // The requests name the fields, so their absence is where a wrong vocabulary shows up: a
        // checkpoint opened as the other export's spelling has a `weight_packed` where this asked
        // for a `weight`, and the refusal says which spelling was expected and why.
        for (request, field, suffix) in [
            (values, "packed E2M1 values", names.values),
            (scales, "K16 block scales", names.block_scale),
        ] {
            self.check_nvfp4_field(&request.tensor, names, field, suffix)?;
        }
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
        let expected = file.shape(rows, columns);
        if scales.shape != expected {
            return Err(Error::weight(
                &scales.tensor,
                format!(
                    "the {file} block scales of a [{rows}, {columns}] matrix are {:?}, not {:?}",
                    expected.dims(),
                    scales.shape.dims()
                ),
            ));
        }
        if file == ScaleLayout::Swizzled128X4 && product == ScaleLayout::Linear {
            return Err(Error::weight(
                &scales.tensor,
                "a swizzled file rewritten into the linear table is a permutation this loader does \
                 not have: a calibrated export stores its block scales linear"
                    .to_owned(),
            ));
        }
        let product_layout = match product {
            ScaleLayout::Linear => WeightLayout::NVFP4_SCALES_LINEAR,
            ScaleLayout::Swizzled128X4 => WeightLayout::NVFP4_SCALES_SWIZZLED,
        };
        let mut values_request = values.clone().with_layout(WeightLayout::NVFP4_VALUES);
        values_request.shape = Shape::new(&[rows, columns / 2]);
        let packed = self.load(&values_request)?;
        if packed.dtype() != DType::U8 {
            return Err(Error::weight(
                &values_request.tensor,
                format!("packed E2M1 values are U8 bytes, not {}", packed.dtype()),
            ));
        }
        // A calibrated export never packs an all-zero matrix: every E2M1 code would be zero, the
        // product would answer zeros for every activation, and a cache filled from it would read as
        // "this projection is broken" long after the loading boundary could have said so. One host
        // read of a few megabytes per projection, at load only, buys that sentence.
        if !self.is_shapes() {
            let bytes = self.shards()?.read(&values_request.tensor)?;
            if bytes.bytes.iter().all(|byte| *byte == 0) {
                return Err(Error::weight(
                    &values_request.tensor,
                    "every packed E2M1 byte is zero: the product would answer zeros for every \
                     activation, so this tensor's calibration did not run"
                        .to_owned(),
                ));
            }
            // The block scales are the second way a product answers zeros for every activation:
            // an E4M3 scale of zero multiplies its whole block to zero whatever the codes are. One
            // zero scale is legitimate (a block whose values are all zero); every scale zero is a
            // calibration that did not run, and it is invisible in the values themselves.
            let file_scales = self.shards()?.read(&scales.tensor)?;
            if file_scales.bytes.iter().all(|byte| *byte == 0) {
                return Err(Error::weight(
                    &scales.tensor,
                    "every block scale is zero: the product would answer zeros for every \
                     activation whatever the codes hold, so this tensor's calibration did not run"
                        .to_owned(),
                ));
            }
        }
        // The file's arrangement and the product's are two statements, and the permutation between
        // them happens here once: a kernel must never read a table under a claim it does not meet.
        let scales = if file == product {
            let mut request = scales.clone().with_layout(product_layout);
            request.shape = expected;
            let loaded = self.load(&request)?;
            block_scale_dtype(&request.tensor, loaded.dtype())?;
            loaded
        } else {
            let bytes = self.shards()?.read(&scales.tensor)?;
            let dtype = block_scale_dtype(&scales.tensor, bytes.dtype)?;
            let blocks = file.columns(columns);
            // The request may name a column block of a row-parallel matrix, so the rows are gathered
            // at its stride before the permutation: permuting the file's first `rows · blocks`
            // elements instead would publish another rank's scales under this rank's key.
            let (start, stride) = match &scales.source {
                WeightSource::Region { start, .. } => (*start, blocks),
                WeightSource::Strided {
                    start, row_stride, ..
                } => (*start, *row_stride),
                WeightSource::Parts { .. } => {
                    return Err(Error::weight(
                        &scales.tensor,
                        "block scales gathered from fused parts have no single row stride"
                            .to_owned(),
                    ));
                }
            };
            let mut linear = Vec::with_capacity((rows * blocks) as usize);
            for row in 0..rows {
                let first = start + row * stride;
                let end = first + blocks;
                let end = usize::try_from(end).unwrap_or(usize::MAX);
                let first = usize::try_from(first).unwrap_or(usize::MAX);
                let row_bytes = bytes.bytes.get(first..end).ok_or_else(|| {
                    Error::weight(
                        &scales.tensor,
                        format!("elements {first}..{end} of a row are past the tensor's {blocks}-block rows"),
                    )
                })?;
                linear.extend_from_slice(row_bytes);
            }
            let arranged = swizzled_scales(&linear, rows, blocks);
            let shape = product.shape(rows, columns);
            let files = self.identities(std::slice::from_ref(&scales.tensor))?;
            self.publish_computed(
                WeightKey {
                    files,
                    dtype,
                    layout: product_layout,
                    region: WeightRegion {
                        tensors: std::iter::once(format!("{}__nvfp4_scales", scales.tensor))
                            .collect(),
                        start: 0,
                        shape,
                        stride: None,
                    },

                    provenance: None,
                },
                shape,
                &arranged,
            )?
        };
        let input_scale = match input_scale {
            None => None,
            Some(request) => {
                self.check_nvfp4_field(
                    &request.tensor,
                    names,
                    "calibration's activation scale",
                    names.input_scale,
                )?;
                let reciprocal = self.activation_reciprocal(request)?;
                let shape = Shape::new(&[1]);
                let bytes = reciprocal.to_le_bytes();
                // The activation's reciprocal is computed from one file: the suffix names the
                // published piece, not a tensor the checkpoint holds, so it is not an identity.
                let files = self.identities(std::slice::from_ref(&request.tensor))?;
                Some(
                    self.publish_computed(
                        WeightKey {
                            files,
                            dtype: DType::F32,
                            layout: WeightLayout::ROW_MAJOR,
                            region: WeightRegion {
                                tensors: std::iter::once(format!(
                                    "{}__nvfp4_input_scale",
                                    request.tensor
                                ))
                                .collect(),
                                start: 0,
                                shape,
                                stride: None,
                            },

                            provenance: None,
                        },
                        shape,
                        &bytes,
                    )?,
                )
            }
        };
        Ok(QuantizedWeight {
            packed,
            scales,
            global_scale,
            layout: product,
            input_scale,
        })
    }

    /// Check that a checkpoint holds one of a projection's NVFP4 fields, naming the export's
    /// spelling when it does not.
    ///
    /// The spelling is the whole point: the same number is `weight` in one export and
    /// `weight_packed` in another, so "not found" against a name the loader expected to be right
    /// hides which export the checkpoint came from. A checkpoint opened with one vocabulary and
    /// written in the other is a loading-boundary error, not a corrupt checkpoint.
    fn check_nvfp4_field(
        &self,
        tensor: &str,
        names: Nvfp4Names,
        field: &str,
        suffix: &str,
    ) -> Result<()> {
        if self.shards()?.tensor(tensor).is_some() {
            return Ok(());
        }
        Err(Error::weight(
            tensor,
            format!(
                "this checkpoint was opened as the {names} spelling of a calibrated NVFP4 \
                 projection, whose {field} is `{suffix}`, and it holds no tensor of this name; the \
                 other export's spelling names the same field differently"
            ),
        ))
    }

    /// Read one tensor's single floating number and answer its reciprocal.
    ///
    /// Both tensor scales a calibrated NVFP4 export states — the weight's `weight_scale_2` and the
    /// activation's `input_scale` — are one positive float each, and every kernel that consumes one
    /// takes the reciprocal. A zero, a negative or a non-finite scale is refused here rather than
    /// encoded at the wrong magnitude.
    fn activation_reciprocal(&self, request: &WeightRequest) -> Result<f32> {
        Ok(1.0 / self.tensor_scale(request)?)
    }

    /// One tensor's single positive scale, as the checkpoint states it.
    ///
    /// Both tensor scales a calibrated NVFP4 export writes — the weight's `weight_scale_2` and the
    /// activation's `input_scale` — are one positive float each, and a loader that synthesized one
    /// would encode at an invented magnitude. A projection reads its weight's scale to build the
    /// product's `alpha` and its activation's to build the encoder's operand, so the reading is
    /// public and the reciprocals are its callers'.
    pub fn tensor_scale(&self, request: &WeightRequest) -> Result<f32> {
        let tensor = self.shards()?.read(&request.tensor)?;
        if tensor.shape.numel() != 1 || !tensor.dtype.is_float() {
            return Err(Error::weight(
                &request.tensor,
                format!(
                    "a tensor scale is one floating number, not {} {:?}",
                    tensor.dtype,
                    tensor.shape.dims()
                ),
            ));
        }
        let scale = to_f32(tensor.dtype, &tensor.bytes, &request.tensor)?[0];
        if !(scale.is_finite() && scale > 0.0) {
            return Err(Error::weight(
                &request.tensor,
                format!("a tensor scale of {scale} is not a positive number"),
            ));
        }
        Ok(scale)
    }

    /// Load the encoded fields of a whole expert stack: `experts` experts, each contributing
    /// `projections.len()` projections in the order the operator reads them.
    ///
    /// The checkpoint stores each expert's projection as its own tensor, so a stack is a
    /// concatenation: expert-major, and within an expert the projections in `projections`' order
    /// (the gated product reads up first, gate second). The three fields answer the three the
    /// operators take — packed values, block scales, and one reciprocal per **expert and
    /// projection** — and all three are published through retained storage under one key per field,
    /// so a second load of the same checkpoint costs nothing.
    ///
    /// The tensor scales are two per expert for a fused `up`/`gate` stack: the checkpoint gives each
    /// projection its own `weight_scale_2`, and dequantizing the gate half with the up half's scale
    /// would be wrong by the ratio of the two with nothing to show for it.
    ///
    /// `name` names the stack in keys and errors and must not collide with a tensor's: a computed
    /// weight needs its own name. A block-scale layout whose per-expert entry is padded, or a
    /// projection count that would break the swizzle's 128-row tiles, is refused here rather than
    /// read as a table no program expects.
    #[allow(
        clippy::too_many_arguments,
        reason = "the design's stack carries its own name, the checkpoint's prefix, the projections, \
                  the expert count, the export's spelling and the scale layout"
    )]
    pub fn nvfp4_stack(
        &mut self,
        name: &str,
        prefix: &str,
        projections: &[&str],
        experts: u32,
        names: Nvfp4Names,
        values_layout: ValueLayout,
        layout: ScaleLayout,
        input_scale: bool,
    ) -> Result<QuantizedStack> {
        if experts == 0 || projections.is_empty() {
            return Err(Error::weight(
                name,
                "a stack has experts and at least one projection".to_owned(),
            ));
        }
        let count = u64::from(experts);
        let per = projections.len() as u64;

        // Every field of every expert, by the export's own suffix set.
        let spelling = names;
        let mut fields = Vec::new();
        let mut rows = None;
        let mut columns = None;
        let mut values_bytes = Vec::new();
        let mut scales_bytes = Vec::new();
        let mut reciprocals = Vec::with_capacity(experts as usize);
        let mut activation_reciprocals = Vec::with_capacity(experts as usize * projections.len());
        let mut scale_shape = None;
        let mut scale_dtype = None;
        for expert in 0..count {
            for projection in projections {
                let stem = format!("{prefix}.{expert}.{projection}_proj");
                let (values_name, scales_name) = (
                    format!("{stem}.{}", spelling.values),
                    format!("{stem}.{}", spelling.block_scale),
                );
                let tensor_name = format!("{stem}.{}", spelling.tensor_scale);
                fields.push(values_name.clone());
                fields.push(scales_name.clone());
                fields.push(tensor_name.clone());
                let activation_name = format!("{stem}.{}", spelling.input_scale);
                if input_scale {
                    fields.push(activation_name.clone());
                }

                self.check_nvfp4_field(
                    &values_name,
                    spelling,
                    "packed E2M1 values",
                    spelling.values,
                )?;
                let values = self.shards()?.read(&values_name)?;
                if values.dtype != DType::U8 {
                    return Err(Error::weight(
                        &values_name,
                        format!("packed E2M1 values are U8 bytes, not {}", values.dtype),
                    ));
                }
                let (rows_here, columns_here) = match values.shape.dims() {
                    [rows, packed] => (*rows, packed * 2),
                    other => {
                        return Err(Error::weight(
                            &values_name,
                            format!("a packed matrix is two axes, not {other:?}"),
                        ));
                    }
                };
                // The stack's geometry is the first expert's; every other projection must agree,
                // because one set of views serves them all.
                let agreed = |seen: &mut Option<u64>, here: u64, what: &str| -> Result<()> {
                    match *seen {
                        None => *seen = Some(here),
                        Some(agreed) if agreed != here => {
                            return Err(Error::weight(
                                &values_name,
                                format!("{here} {what} against the stack's {agreed}"),
                            ));
                        }
                        Some(_) => {}
                    }
                    Ok(())
                };
                agreed(&mut rows, rows_here, "rows")?;
                agreed(&mut columns, columns_here, "columns")?;

                let scales = self.shards()?.read(&scales_name)?;
                let here = block_scale_dtype(&scales_name, scales.dtype)?;
                match scale_dtype {
                    None => scale_dtype = Some(here),
                    Some(agreed) if agreed != here => {
                        return Err(Error::weight(
                            &scales_name,
                            format!(
                                "block scales are {here} here against the stack's {agreed}: one \
                                 stack is one scale type"
                            ),
                        ));
                    }
                    Some(_) => {}
                }
                // The shard's table is the checkpoint's own arrangement — one block scale per
                // sixteen elements, row-major — and the published one is the deployment's. A
                // converted export that already holds the padded, interleaved table is refused here
                // rather than published after a second pass over it.
                let expected = ScaleLayout::Linear.shape(rows_here, columns_here);
                if scales.shape != expected {
                    return Err(Error::weight(
                        &scales_name,
                        format!(
                            "a checkpoint's block scales of a [{rows_here}, {columns_here}] matrix \
                             are linear {:?}, not {:?}: the published {layout} table is this \
                             loader's to build, not the export's to state",
                            expected.dims(),
                            scales.shape.dims()
                        ),
                    ));
                }

                let tensor = self.shards()?.read(&tensor_name)?;
                if tensor.shape.numel() != 1 || !tensor.dtype.is_float() {
                    return Err(Error::weight(
                        &tensor_name,
                        format!(
                            "a tensor scale is one floating number, not {} {:?}",
                            tensor.dtype,
                            tensor.shape.dims()
                        ),
                    ));
                }
                let scale = to_f32(tensor.dtype, &tensor.bytes, &tensor_name)?[0];
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(Error::weight(
                        &tensor_name,
                        format!("a tensor scale of {scale} is not a positive number"),
                    ));
                }
                reciprocals.push(1.0 / scale);
                if input_scale {
                    let request = WeightRequest::region(&activation_name, 0, Shape::new(&[1]));
                    activation_reciprocals.push(self.activation_reciprocal(&request)?);
                }

                values_bytes.extend_from_slice(&values.bytes);
                scales_bytes.extend_from_slice(&scales.bytes);
                scale_shape = Some(expected);
            }
        }

        let rows = rows.unwrap_or(0) * per;
        let columns = columns.unwrap_or(0);
        let shard_scales = ScaleLayout::Linear.shape(rows, columns);
        let entry = scale_shape.map(|shape: Shape| shape.numel()).unwrap_or(0);
        if shard_scales.numel() != entry * per {
            return Err(Error::weight(
                name,
                format!(
                    "a linear entry for {rows} rows of {columns} elements is {} block scales, and \
                     {per} concatenated entries hold {entry} each",
                    shard_scales.numel() / per.max(1)
                ),
            ));
        }

        // The values' row arrangement: a plain shard needs nothing, a shuffled one is permuted here
        // once, at load. A product whose A operand is read with `useShuffledMatrix` cannot read a
        // shard's own row order, so a row count that is not a whole number of shuffle blocks has no
        // arrangement at all and is refused by name rather than published in an order nothing reads.
        if let Some(block) = values_layout.block()
            && !rows.is_multiple_of(block)
        {
            return Err(Error::weight(
                name,
                format!(
                    "a {values_layout:?} arrangement permutes {block}-row blocks, and this stack's                      matrices have {rows} rows"
                ),
            ));
        }
        let values_bytes = values_layout.arrange(&values_bytes, count, rows, (columns / 2).max(1));

        let values_shape = Shape::new(&[count, rows, columns / 2]);
        let scales_shape = Shape::new(&[
            count,
            layout.shape(rows, columns).dim(0),
            layout.shape(rows, columns).dim(1),
        ]);
        // `reciprocals` was collected expert-major, in the checkpoint's own reading order; the
        // published table is projection-major, one packed row per projection.
        let projections_shape = Shape::new(&[per, count]);
        let strides = Projections::new(per, count);
        if reciprocals.len() != strides.len() {
            return Err(Error::weight(
                name,
                format!(
                    "{per} projections of {count} experts are {} scales, and the checkpoint gave {}",
                    strides.len(),
                    reciprocals.len()
                ),
            ));
        }
        let scale_bytes: Vec<u8> = strides.bytes(&reciprocals);
        let files = self.identities(&fields)?;

        let values = self.publish_computed(
            WeightKey {
                files: files.clone(),
                dtype: DType::U8,
                layout: values_layout.layout(),
                region: WeightRegion {
                    tensors: std::iter::once(format!("{name}__nvfp4_values"))
                        .chain(fields.iter().cloned())
                        .collect(),
                    start: 0,
                    shape: values_shape,
                    stride: None,
                },

                provenance: None,
            },
            values_shape,
            &values_bytes,
        )?;
        // The stack's scale type is the checkpoint's own: an E4M3 field stays E4M3, and the model
        // puts it in the descriptor unchanged.
        let scale_dtype = scale_dtype.unwrap_or(DType::U8);
        // One expert's linear table becomes the published arrangement: rows permuted the way the
        // values' rows were when the values are shuffled (the scales are read beside the row), then
        // padded and interleaved when the deployment's kernels read the swizzled 128x4 table.
        let mut arranged_scales = Vec::with_capacity(scales_shape.numel() as usize);
        let blocks = ScaleLayout::Linear.columns(columns);
        for expert in 0..usize::try_from(count).unwrap_or(0) {
            let entry = &scales_bytes
                [expert * (rows * blocks) as usize..(expert + 1) * (rows * blocks) as usize];
            let entry = values_layout.arrange(entry, 1, rows, blocks);
            match layout {
                ScaleLayout::Linear => arranged_scales.extend_from_slice(&entry),
                ScaleLayout::Swizzled128X4 => {
                    arranged_scales.extend_from_slice(&swizzled_scales(&entry, rows, blocks));
                }
            }
        }
        let scales_bytes = arranged_scales;
        let scales = self.publish_computed(
            WeightKey {
                files: files.clone(),
                dtype: scale_dtype,
                layout: match layout {
                    ScaleLayout::Linear => WeightLayout::NVFP4_SCALES_LINEAR,
                    ScaleLayout::Swizzled128X4 => WeightLayout::NVFP4_SCALES_SWIZZLED,
                },
                region: WeightRegion {
                    tensors: std::iter::once(format!("{name}__nvfp4_scales"))
                        .chain(fields.iter().cloned())
                        .collect(),
                    start: 0,
                    shape: scales_shape,
                    stride: None,
                },

                provenance: None,
            },
            scales_shape,
            &scales_bytes,
        )?;
        let global_scale = self.publish_computed(
            WeightKey {
                files,
                dtype: DType::F32,
                layout: WeightLayout::ROW_MAJOR,
                region: WeightRegion {
                    tensors: std::iter::once(format!("{name}__nvfp4_tensor_scales"))
                        .chain(fields.iter().cloned())
                        .collect(),
                    start: 0,
                    shape: projections_shape,
                    stride: None,
                },

                provenance: None,
            },
            projections_shape,
            &scale_bytes,
        )?;
        let input_scale = if input_scale {
            let bytes = strides.bytes(&activation_reciprocals);
            let shape = projections_shape;
            let files = self.identities(
                &fields
                    .iter()
                    .filter(|field| field.ends_with(&format!(".{}", spelling.input_scale)))
                    .cloned()
                    .collect::<Vec<_>>(),
            )?;
            Some(self.publish_computed(
                WeightKey {
                    files,
                    dtype: DType::F32,
                    layout: WeightLayout::ROW_MAJOR,
                    region: WeightRegion {
                        tensors: std::iter::once(format!("{name}__nvfp4_input_scales")).collect(),
                        start: 0,
                        shape,
                        stride: None,
                    },

                    provenance: None,
                },
                shape,
                &bytes,
            )?)
        } else {
            None
        };
        // The encoder's global scale is one: the calibrated activation scale cancels out of both the
        // code and the reconstruction (see `QuantizedStack::encode_scale`), and a host value here
        // would be a number someone later "corrects" back to the checkpoint's.
        let encode_scale = self.constant(
            &format!("{name}__nvfp4_encode_scale"),
            DType::F32,
            Shape::new(&[1]),
            &1.0_f32.to_ne_bytes(),
            "nn",
        )?;
        Ok(QuantizedStack {
            values,
            scales,
            global_scale,
            layout,
            encode_scale,
            input_scale,
        })
    }

    /// Publish bytes this loader computed, under a key it built, and answer the weight.
    fn publish_computed(&mut self, key: WeightKey, shape: Shape, bytes: &[u8]) -> Result<Weight> {
        if let Some(backing) = self.retention()?.retained(&key)? {
            return Ok(Weight {
                view: backing.view(key.dtype, shape)?,
                backing,
            });
        }
        let backing = self.retention()?.publish(&key, key.dtype, shape, bytes)?;
        Ok(Weight {
            view: backing.view(key.dtype, shape)?,
            backing,
        })
    }

    /// Publish host-computed bytes as a weight, without a checkpoint tensor behind them.
    ///
    /// The rotary frequency table is the first such weight: `inv_freq` is computed once at loading
    /// from `rope_theta` and `rope_scaling` (specs/worker.md 9), and the `rotary_factors` operator
    /// reads it as a device weight, so it takes the same retained-storage path as a checkpoint
    /// weight. There is no file to name it, so the key states the **provenance**: the class that
    /// computed it, `producer`, and a digest of the bytes it produced. Two computations of the same
    /// numbers share one retained region; two that compute different numbers under one name never
    /// do. A name alone is not an identity — it is a label the caller chose, and two models that
    /// choose the same label for their own constants are exactly the collision this prevents.
    pub fn constant(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
        producer: &'static str,
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
        // A shapes-only loader states the constant's extents and keeps the bytes nowhere.
        if self.is_shapes() {
            let backing = Backing::new(DevicePtr::new(0), expected);
            return Ok(Weight {
                view: backing.view(dtype, shape)?,
                backing,
            });
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
            provenance: Some(Provenance::computed(producer, bytes)),
        };
        let backing = match self.retention()?.retained(&key)? {
            Some(backing) => backing,
            None => self.retention()?.publish(&key, dtype, shape, bytes)?,
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
                self.shards()?
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
                .shards()?
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
    fn target_dtype(&self, names: &[String], pinned: Option<DType>) -> Result<DType> {
        let checkpoint = self.checkpoint_dtype(names)?;
        // A pin is the model's statement about this tensor; the policy is the deployment's about
        // every tensor it reads. The pin wins, because a representation the reference fixes at
        // every preset is not a property of the deployment.
        let dtype = pinned.or(self.dtype).unwrap_or(checkpoint);
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
                    .shards()?
                    .tensor(tensor)
                    .ok_or_else(|| Error::MissingTensor(tensor.clone()))?;
                let bytes = self.shards()?.bytes(location)?.to_vec();
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
                    .shards()?
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
                let bytes = self.shards()?.bytes(location)?;
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
                        .shards()?
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
                    fused.extend_from_slice(self.shards()?.bytes(location)?);
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
///
/// It is public because a boundary between two computations can need exactly one conversion and no
/// loader: a model that hands a decoder a tensor its own layers produced states the pair it needs,
/// and `tensor` is what the refusal names when the pair is not one of the six. A caller that
/// converts on a path a launch did not state is widening the rule above, so the conversion belongs
/// where the boundary itself is stated.
pub fn convert(from: DType, to: DType, bytes: &[u8], tensor: &str) -> Result<Vec<u8>> {
    match (from, to) {
        (DType::BF16, DType::F32)
        | (DType::BF16, DType::F16)
        | (DType::F16, DType::F32)
        | (DType::F16, DType::BF16)
        | (DType::F32, DType::BF16)
        | (DType::F32, DType::F16)
        | (DType::F32, DType::I16)
        | (DType::I16, DType::F32) => {}
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
            // A PCM sample is an integer that F32 holds exactly, so the read is lossless.
            DType::I16 => f32::from(i16::from_le_bytes([chunk[0], chunk[1]])),
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
            DType::I16 => converted.extend_from_slice(&f32_to_i16(value).to_le_bytes()),
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

/// The `i16` a sample becomes: nearest, ties to even, saturating at both ends.
///
/// A PCM track is a clip, not a wrap: a sample beyond the range stores at the end of it rather
/// than folding to the opposite sign, and a NaN stores as zero (the saturating cast's answer, and a
/// NaN in a checkpoint is already corrupt).
fn f32_to_i16(value: f32) -> i16 {
    let rounded = value.round_ties_even();
    if rounded >= f32::from(i16::MAX) {
        i16::MAX
    } else if rounded <= f32::from(i16::MIN) {
        i16::MIN
    } else {
        rounded as i16
    }
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

/// The bfloat16 bit pattern nearest to `value`.
///
/// A bf16 is an `f32` whose low sixteen bits are dropped, with round-to-nearest-even: the rounding
/// adds one less than half a unit in the last place, and the surviving bit decides a tie. That is
/// what the device's own conversion does, so a producer that narrows on the host and a layer that
/// narrows on the card agree bit for bit.
///
/// A `NaN` is returned as a `NaN` with its sign and payload's top bits: the rounding add would
/// otherwise carry a large payload into an infinity, and a boundary that turns a `NaN` into an
/// infinity is a boundary that hides a divergence.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Quiet, with the payload's top six bits: an `f32` NaN is any exponent of all ones with a
        // non-zero mantissa, and the truncation keeps both.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// The `f32` value of an IEEE half-precision bit pattern.
///
/// Public because a container that must hold `f32` (the diagnosis' `.npy`, which numpy reads) is
/// written from a payload the device produced at the checkpoint's own width: the widening is one
/// exact conversion and belongs beside its inverse rather than in every producer that needs it.
pub fn half_to_f32(half: u16) -> f32 {
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
///
/// Public with [`half_to_f32`]: a producer that narrows a tensor to the deployment's type and a
/// producer that widens one for a file are the same two functions read in opposite directions.
pub fn f32_to_half(value: f32) -> u16 {
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

    /// A shapes-only loader answers a weight request with the extents the caller asked for, at
    /// address zero, and reads nothing: no checkpoint, no shard, no device.
    ///
    /// It exists for the walk — a model's own `load` driven by a recording forward — so what it owes is
    /// the shapes a `load` derives, and what it must not do is pretend to have bytes: a path that needs
    /// them refuses by name rather than answering a zero.
    #[test]
    fn a_shapes_only_loader_states_extents_and_reads_no_bytes() {
        let mut loader = WeightLoader::shapes(DType::BF16);
        assert!(loader.is_shapes());

        let request = WeightRequest::region("block.proj.weight", 0, Shape::new(&[64, 128]));
        let weight = loader.load(&request).expect("the stated weight");
        assert_eq!(weight.shape().dims(), &[64, 128]);
        assert_eq!(weight.dtype(), DType::BF16);
        assert_eq!(weight.view().ptr.address(), 0);

        // A computed weight states its extents too: the H3 loader publishes tables this way.
        let bytes = [0u8, 0, 0, 0, 1, 0, 0, 0];
        let constant = loader
            .constant(
                "video_vae.voxel_rows",
                DType::I32,
                Shape::new(&[2]),
                &bytes,
                "test",
            )
            .expect("the stated constant");
        assert_eq!(constant.shape().dims(), &[2]);
        assert_eq!(constant.view().ptr.address(), 0);
    }

    /// The block-scale arrangement is the launcher's `computeSFIndex`: a permutation of the
    /// checkpoint's row-major table into the 128x4 layout, padded with zeros.
    ///
    /// Three properties, and the third is the one that catches a skipped arrangement: the values
    /// survive exactly once each; the result is not the padded row-major table, so an arrangement that
    /// never ran fails here rather than passing as "the same numbers"; and one index follows the
    /// reference's own arithmetic
    /// (`(column % 4) + (column / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 +
    /// (row / 128) * (128 * padded_columns)`), so a bijection that moved the right values to the wrong
    /// places is not mistaken for the layout the products read.
    ///
    /// The shapes keep `rows * blocks` inside a byte, because the table is bytes: `(56, 4)` pads the
    /// rows only and `(8, 8)` pads both axes.
    ///
    /// **The padding is a zero, so the data starts at one.** "Every scale survives exactly once" is
    /// counted over non-zero entries here, and a table carrying a genuine zero would be one short —
    /// which is the caller's obligation, not the assertion's: a zero scale is a block the file says is
    /// empty, indistinguishable in the arranged table from a hole, so a test of this bijection uses
    /// values a checkpoint would never write as padding.
    #[test]
    fn the_block_scales_are_permuted_into_the_laid_out_table() {
        for (rows, blocks) in [(56_u64, 4_u64), (8, 8)] {
            let linear: Vec<u8> = (1..=rows * blocks).map(|value| value as u8).collect();
            let swizzled = swizzled_scales(&linear, rows, blocks);
            let padded_rows = rows.div_ceil(128) * 128;
            let padded_columns = blocks.div_ceil(4) * 4;
            assert_eq!(
                swizzled.len() as u64,
                padded_rows * padded_columns,
                "the table is padded to 128 rows and 4 columns"
            );

            let mut present: Vec<u8> = swizzled
                .iter()
                .copied()
                .filter(|value| *value != 0)
                .collect();
            let mut expected = linear.clone();
            present.sort_unstable();
            expected.sort_unstable();
            assert_eq!(present, expected, "every scale survives exactly once");

            let mut row_major = vec![0_u8; (padded_rows * padded_columns) as usize];
            for row in 0..rows {
                for column in 0..blocks {
                    row_major[(row * padded_columns + column) as usize] =
                        linear[(row * blocks + column) as usize];
                }
            }
            assert_ne!(
                swizzled, row_major,
                "an arrangement that never ran is the row-major table"
            );
        }

        // One index from the formula, at a row inside the first thirty-two and a column past the
        // first group of four: the values are `row * blocks + column`, so this one is 5 * 8 + 6.
        let (rows, blocks) = (128_u64, 8_u64);
        let linear: Vec<u8> = (1..=rows * blocks).map(|value| value as u8).collect();
        let swizzled = swizzled_scales(&linear, rows, blocks);
        let index = 594_usize;
        // The data starts at one so that a genuine zero cannot be mistaken for the table's padding;
        // the pinned value is therefore `row * blocks + column + 1`, not the position itself.
        assert_eq!(swizzled[index], (5 * blocks + 6 + 1) as u8);
    }

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
    fn a_bf16_conversion_rounds_like_the_hardware() {
        // A value whose low half is zero keeps exactly its top sixteen bits.
        for value in [0.0_f32, 1.0, -1.0, 2.0, 0.5, 1.0e-30, 3.0e38] {
            assert_eq!(
                f32_to_bf16(value),
                (value.to_bits() >> 16) as u16,
                "{value:e}"
            );
        }
        // Half a unit in the last place ties between two neighbours and the even one wins; a value
        // past the tie rounds up.
        assert_eq!(f32_to_bf16(1.0 + 2.0_f32.powi(-8)), 0x3f80);
        assert_eq!(f32_to_bf16(1.0 + 3.0 * 2.0_f32.powi(-9)), 0x3f81);
        // The specials keep their meaning: an infinity is the top exponent with a zero mantissa, and
        // a NaN stays a NaN instead of carrying into that infinity.
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xff80);
        assert!(f32::from_bits(u32::from(f32_to_bf16(f32::NAN)) << 16).is_nan());
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

            provenance: None,
        };
        let text = key.to_string();
        assert!(text.contains("bf16"));
        assert!(text.contains("up_gate"));
        assert!(text.contains("up+gate"));
    }

    /// A PCM sample converts to its nearest integer, ties to even, and clamps at both ends: a clip
    /// does not wrap, and the two ends are the values an audio codec actually reaches.
    #[test]
    fn a_sample_converts_to_the_nearest_sixteen_bit_integer() {
        let bytes: Vec<u8> = [
            0.0f32, 0.5, 1.5, 2.5, -0.5, -1.5, 32767.4, 32768.0, -32768.4, -40000.0, 3.0,
        ]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
        let converted = convert(DType::F32, DType::I16, &bytes, "pcm").unwrap();
        let samples: Vec<i16> = converted
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_le_bytes(*pair))
            .collect();
        assert_eq!(
            samples,
            [0, 0, 2, 2, 0, -2, 32767, 32767, -32768, -32768, 3],
            "nearest with ties to even, saturated at both ends"
        );
        // And back: every one of those integers is a float exactly.
        let back = convert(DType::I16, DType::F32, &converted, "pcm").unwrap();
        let floats: Vec<f32> = back
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        assert_eq!(
            floats,
            [
                0.0, 0.0, 2.0, 2.0, 0.0, -2.0, 32767.0, 32767.0, -32768.0, -32768.0, 3.0
            ]
        );
    }

    /// Every other pair that names I16 is refused, and the refusal names both types.
    #[test]
    fn a_sample_converts_only_to_and_from_f32() {
        for (from, to) in [
            (DType::BF16, DType::I16),
            (DType::I16, DType::BF16),
            (DType::F16, DType::I16),
            (DType::I16, DType::F16),
            (DType::U8, DType::I16),
            (DType::I16, DType::U8),
        ] {
            let bytes = vec![0u8; 2];
            let error = convert(from, to, &bytes, "pcm").unwrap_err().to_string();
            assert!(
                error.contains(&from.to_string()) && error.contains(&to.to_string()),
                "{from} to {to} was refused without naming both: {error}"
            );
        }
    }
}

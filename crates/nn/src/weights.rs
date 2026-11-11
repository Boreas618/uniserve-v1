//! Weights: the layout the reading operator declares, the key retained storage files them under,
//! and the loader that converts a checkpoint's tensors once.
//!
//! A loaded weight is a [`Tensor`]: over retained storage on the device a model serves from, over
//! host memory when a model is loaded on the CPU device, and a meta tensor when it is loaded for
//! a profile run. Modules reach the loader through a [`VarBuilder`](crate::VarBuilder), which
//! names their tensors by their path in the model.
//!
//! ## The layout
//!
//! [`WeightLayout`] is the reading operator's arrangement of the bytes, named by the operator and
//! filed with the weight. The checkpoint's own tensors are laid out per model framework (an
//! expert's gate and up projections are separate matrices, gate first); an operator that reads
//! one fused `[experts, 2·intermediate, hidden]` weight with the up projection first declares
//! [`WeightLayout::UP_GATE`] and the loader produces that arrangement from the checkpoint's
//! parts, once, before the bytes are published.
//!
//! ## The key
//!
//! Retained weights are keyed by exactly four inputs: the identity of every checkpoint file whose
//! bytes produced the weight, the element type the weight has after conversion, the layout name,
//! and the weight's region in its logical tensor. No kernel name or revision enters the key: a
//! kernel update keeps every retained weight. A change of element type, quantization, layout or
//! checkpoint therefore changes the key, and a change of an operator's kernel does not.
//!
//! ## Publishing
//!
//! Conversion happens here and nowhere else, and its output goes straight to retained storage
//! through [`RetainedWeights`], which execution implements over the Storage Server and its
//! device mappings and [`CpuWeights`] implements over host memory. `nn` owns no device memory: a
//! weight is the tensor the store returned for its key.
//!
//! ## Meta loading
//!
//! [`WeightLoader::meta`] runs the same loading code over the same shard headers and stops where
//! bytes would be read: every weight it returns is a meta tensor of the element type and extents
//! the device load gives, and it publishes nothing, so a profile run of a model's load needs no
//! payload and no device.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use backend::allocator::ALIGNMENT;
use ops::values::{bf16_to_f32, f16_to_f32, f32_to_f16};
use sha2::{Digest, Sha256};
use uniserve_tensor::{CpuStorage, DType, Device, Shape, Storage, Tensor};

use crate::checkpoint::{Checkpoint, FileIdentity, TensorLocation};
use crate::error::{Error, Result};
use crate::quantization::{Declaration, Scheme};
use crate::var_builder::VarBuilder;

/// The arrangement of a weight's bytes that its reading operator declares.
///
/// The names enter the retained key, so each one is a stable statement about layout, not about
/// the operator's kernel or tuning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WeightLayout(&'static str);

impl WeightLayout {
    /// The checkpoint's own contiguous row-major arrangement, unchanged.
    pub const ROW_MAJOR: Self = Self("row_major");

    /// A routed expert's first projection: `[experts, 2·intermediate, hidden]`, up before gate.
    pub const UP_GATE: Self = Self("up_gate");

    /// A routed expert's second projection: `[experts, hidden, intermediate]`.
    pub const DOWN: Self = Self("down");

    /// A calibrated NVFP4 matrix's packed E2M1 values: `[rows, columns / 2]` U8, one byte per
    /// pair.
    pub const NVFP4_VALUES: Self = Self("nvfp4_values");

    /// The same values after `shuffleMatrixA`: rows permuted inside blocks of sixteen or
    /// thirty-two, which is the arrangement the trtllm-gen products read (`useShuffledMatrix`).
    pub const NVFP4_VALUES_SHUFFLED: Self = Self("nvfp4_values_shuffled");

    /// Packed values whose two row halves interleave in sixteen-row blocks, the second half's
    /// first: the row order a gated product reads when each gate block precedes its up block.
    pub const NVFP4_VALUES_GATE_FIRST: Self = Self("nvfp4_values_gate_first");

    /// Packed values whose two row halves interleave row by row before the shuffle: the row
    /// order a gated product reads its linear and gate rows side by side in.
    pub const NVFP4_VALUES_INTERLEAVED_SHUFFLED: Self = Self("nvfp4_values_interleaved_shuffled");

    /// A calibrated NVFP4 matrix's E4M3 block scales in the linear layout:
    /// `[rows, columns / 16]`.
    pub const NVFP4_SCALES_LINEAR: Self = Self("nvfp4_scales_linear");

    /// The same scales in the swizzled 128x4 layout the cuDNN product reads.
    pub const NVFP4_SCALES_SWIZZLED: Self = Self("nvfp4_scales_swizzled_128x4");

    /// The layout named `name`.
    pub const fn named(name: &'static str) -> Self {
        Self(name)
    }
}

impl fmt::Display for WeightLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

/// An explicitly requested quantization method.
///
/// The names are SGLang's `quantization` values: an explicit value converts the checkpoint's
/// weights once, at load; with none, the checkpoint's own representation is kept.
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

/// Where a weight's bytes come from: one region of one checkpoint tensor, or whole tensors fused
/// in order along their first axis.
///
/// Starts, strides and block extents count elements of the checkpoint tensor in its row-major
/// order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WeightSource {
    /// `shape` elements starting at element `start` of `tensor`, row-major.
    Region {
        /// The checkpoint tensor's name.
        tensor: String,
        /// The first element of the region.
        start: u64,
        /// The region's extents.
        shape: Shape,
    },
    /// `shape` elements of `tensor` whose last axis begins at element `start` and repeats every
    /// `row_stride` elements.
    ///
    /// A tensor-parallel rank reads one column block of a row-parallel projection: every row of
    /// the checkpoint's tensor contributes the last extent of `shape`, and consecutive rows
    /// start `row_stride` elements apart. The loader gathers those rows and converts them like
    /// any other weight, so the retained storage of a shard is contiguous like the rest.
    Strided {
        /// The checkpoint tensor's name.
        tensor: String,
        /// The first element of the first row.
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
        /// The block each part contributes, or the whole part when absent.
        ///
        /// A tensor-parallel rank holds its block of every routed expert: the same rows of each
        /// gate and up projection, and the same columns of each down projection. Every part
        /// gives the same block, so the fused weight is the rank's shard of every part in order.
        block: Option<PartBlock>,
    },
    /// The row-major concatenation of one block of each part, every part naming its own block,
    /// viewed as `shape`.
    ///
    /// This is how a stacked checkpoint tensor (every expert's projections in one
    /// `[experts, rows, columns]` tensor) becomes the stack its reading operator expects in
    /// another order: each block is one expert's half, in the order the operator reads them.
    Blocks {
        /// The checkpoint tensors and the block each contributes, in fused order.
        parts: Vec<(String, PartBlock)>,
        /// The fused tensor's extents.
        shape: Shape,
    },
}

/// One block of a part: `rows` rows of `columns` elements, the first at element `start` of the
/// part in its row-major order and each next one `row_stride` elements after it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PartBlock {
    /// The block's first element in the part.
    pub start: u64,
    /// Rows in the block.
    pub rows: u64,
    /// Elements in one row of the block.
    pub columns: u64,
    /// Elements between the first elements of consecutive rows in the part.
    pub row_stride: u64,
}

/// What produced a weight that no checkpoint file names.
///
/// A checkpoint weight is named by the files it was read from and its region in them. A computed
/// one (a rotary frequency table, fused output heads, a modulation table) has no file, so the
/// only thing that can name it is the computation: the class that ran it, and the numbers it
/// produced. A label is not an identity: two models that name their own fused heads
/// `denoiser.video.student_heads.weight` are two producers, and a key holding only the name lets
/// the second one map the first one's bytes, silently, under an identical extent.
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

/// What retained storage files a weight under.
///
/// Its text ([`fmt::Display`]) is the name the store keeps the bytes by, so a field's spelling
/// there is fixed for as long as stored weights are to be found again.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WeightKey {
    /// The identity of every checkpoint file the weight was read from, in part order.
    pub files: Vec<FileIdentity>,
    /// The element type of the retained bytes.
    pub dtype: DType,
    /// The layout of the retained bytes.
    pub layout: WeightLayout,
    /// The checkpoint tensors' names, in fused order (one name for a plain region).
    pub tensors: Vec<String>,
    /// The first element of the weight's region in the logical tensor.
    pub start: u64,
    /// The extents of the region in the logical tensor.
    pub shape: Shape,
    /// The source rows' stride in elements, when the region is a strided sub-matrix. Two
    /// gatherings of one tensor differ in it, so they cannot collide in retained storage.
    pub stride: Option<u64>,
    /// What produced the bytes, for a weight no checkpoint file names.
    ///
    /// `None` for a weight read from a checkpoint, whose files and region are its provenance; a
    /// computed weight states the class that computed it and a digest of the numbers it
    /// produced, so two producers that compute different tables under one name cannot share
    /// retained bytes.
    pub provenance: Option<Provenance>,
}

impl std::hash::Hash for WeightKey {
    /// Hashes every field equality compares; the tensor library's `Shape` is hashed by its
    /// extents, having no hash of its own.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.files.hash(state);
        self.dtype.hash(state);
        self.layout.hash(state);
        self.tensors.hash(state);
        self.start.hash(state);
        self.shape.dims().hash(state);
        self.stride.hash(state);
        self.provenance.hash(state);
    }
}

impl WeightKey {
    /// The region's text: the tensors, the element range and the extents, as diagnostics name a
    /// weight.
    fn region(&self) -> String {
        let mut region = format!(
            "{}[{}..{}]{:?}",
            self.tensors.join("+"),
            self.start,
            self.start + numel(&self.shape),
            self.shape.dims()
        );
        if let Some(stride) = self.stride {
            region.push_str(&format!("|stride_{stride}"));
        }
        region
    }
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
            ops::abi::name(self.dtype),
            self.layout,
            self.region()
        )?;

        if let Some(provenance) = &self.provenance {
            write!(formatter, "|{provenance}")?;
        }
        Ok(())
    }
}

/// Retained storage, as the loader publishes converted weights to it.
///
/// Execution implements this over the Storage Server and its read-only device mappings, giving
/// each weight as a tensor over retained storage on the device it serves from
/// (`backend::retained`): a weight already retained under the key is returned without reading
/// the checkpoint again, and a converted weight is published once. [`CpuWeights`] is the store of
/// a model loaded on the CPU device.
///
/// A loader and the builders made from it share one store, so the methods take `&self`; a store
/// that changes state keeps it behind its own lock or cell.
pub trait RetainedWeights {
    /// The weight retained under `key` as a tensor of `shape` elements of `key.dtype`, when the
    /// store already holds it.
    ///
    /// # Errors
    ///
    /// When the store cannot be queried, or holds another byte count under the key than `shape`
    /// elements take.
    fn retained(&self, key: &WeightKey, shape: &Shape) -> Result<Option<Tensor>>;

    /// Publishes `bytes` as the storage of `key`, `shape` elements of `key.dtype`, and returns
    /// the read-only tensor over them.
    ///
    /// Publishing the same key twice is the caller's error; the store may reject it or return
    /// the tensor it already holds.
    ///
    /// # Errors
    ///
    /// When the store cannot keep the bytes.
    fn publish(&self, key: &WeightKey, shape: &Shape, bytes: &[u8]) -> Result<Tensor>;

    /// A tensor the model writes after the load, holding `bytes` until it does.
    ///
    /// Published storage is read-only on the device, so a table a model derives from its weights
    /// cannot live in it: the first launch that writes it faults. A store whose storage a kernel
    /// may write returns that, which is what this default does.
    ///
    /// # Errors
    ///
    /// As [`RetainedWeights::publish`].
    fn writable(&self, key: &WeightKey, shape: &Shape, bytes: &[u8]) -> Result<Tensor> {
        self.publish(key, shape, bytes)
    }
}

/// Retained storage on the CPU device: every published weight is a CPU tensor holding its bytes.
///
/// A model loaded through this store computes on the CPU device, where each fused operator runs
/// its formula in tensor operations: the load of a host check, and of the reference a device
/// check compares with.
#[derive(Default)]
pub struct CpuWeights {
    /// The weights published so far, by key.
    held: RefCell<HashMap<WeightKey, Tensor>>,
}

impl CpuWeights {
    /// A store that holds nothing yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// The keys of the weights published so far, in no particular order.
    pub fn keys(&self) -> Vec<WeightKey> {
        self.held.borrow().keys().cloned().collect()
    }
}

impl RetainedWeights for CpuWeights {
    fn retained(&self, key: &WeightKey, shape: &Shape) -> Result<Option<Tensor>> {
        match self.held.borrow().get(key) {
            Some(tensor) => Ok(Some(tensor.reshape(shape.clone())?)),
            None => Ok(None),
        }
    }

    fn publish(&self, key: &WeightKey, shape: &Shape, bytes: &[u8]) -> Result<Tensor> {
        let tensor = Tensor::from_raw_buffer(bytes, key.dtype, shape.dims(), &Device::Cpu)?;

        self.held.borrow_mut().insert(key.clone(), tensor.clone());
        Ok(tensor)
    }

    fn writable(&self, key: &WeightKey, shape: &Shape, bytes: &[u8]) -> Result<Tensor> {
        // A writable table is one load's own: it is never returned for a later key.
        Ok(Tensor::from_raw_buffer(
            bytes,
            key.dtype,
            shape.dims(),
            &Device::Cpu,
        )?)
    }
}

/// The first address a meta loader gives a weight.
///
/// Far above any region an execution context allocates in, so a weight and an activation of one
/// profile run never share an address, and far from zero, so no weight reads as a null pointer
/// to an argument check.
const META_BASE: u64 = 1 << 48;

/// The next address a meta loader gives a weight.
///
/// Every meta loader of the process draws from this one cursor, so two loaders of one profile
/// run (one per pipeline stage, say) never give one address to two different weights.
static META_CURSOR: AtomicU64 = AtomicU64::new(META_BASE);

/// The tensor scale a meta loader returns for every calibrated scale it is asked for.
///
/// A meta loader reads no payload, so it has no scale to give; one is a positive, finite number,
/// so every check a caller makes on a scale passes, and a value derived from it is a valid one.
/// A scale only ever reaches a kernel as data (a launch scalar or a published table), so a
/// profile run over this placeholder states the same kernels as a device load over the real
/// scales.
const META_TENSOR_SCALE: f32 = 1.0;

/// Where a loader's weights go.
enum Retention<'a> {
    /// Retained storage, which converted bytes are published to.
    Store(&'a dyn RetainedWeights),
    /// Meta tensors, by the key a device load would publish them under: a key given once is
    /// given the same tensor again, as a retained store returns a key it already holds, so two
    /// modules that share a weight on the device share it here too. Shared by the loaders
    /// reborrowed from one meta loader.
    Meta(Rc<RefCell<HashMap<WeightKey, Tensor>>>),
}

/// One contiguous byte range of a checkpoint tensor that a weight's bytes are gathered from.
///
/// A load computes its spans from the shard headers alone, so every bound a device load checks
/// is checked by a meta load too, without a payload byte read; a device load then copies the
/// spans in order.
pub(crate) struct Span<'c> {
    pub(crate) location: &'c TensorLocation,
    /// The first byte, within the tensor's own bytes.
    pub(crate) from: usize,
    /// One past the last byte, within the tensor's own bytes.
    pub(crate) to: usize,
}

/// How many bytes the shard's header states `location` holds.
///
/// `Checkpoint::open` has checked the range against the tensor's extents and element width and
/// against the shard's length, so this is the length of the slice [`Checkpoint::bytes`] returns.
pub(crate) fn stored_bytes(location: &TensorLocation) -> u64 {
    location.bytes.1 - location.bytes.0
}

/// The element count of `shape`, in the width a load's byte arithmetic runs in.
pub(crate) fn numel(shape: &Shape) -> u64 {
    shape.elem_count() as u64
}

/// The extent of `axis` of `shape`, or zero for an axis the shape does not have.
pub(crate) fn dim(shape: &Shape, axis: usize) -> u64 {
    shape.dims().get(axis).map_or(0, |extent| *extent as u64)
}

/// The extent of the last axis of `shape`, or zero for a shape of no axes.
pub(crate) fn last(shape: &Shape) -> u64 {
    shape.dims().last().map_or(0, |extent| *extent as u64)
}

/// The shape of `dims`, which are element counts a file's header or a module's widths gave.
pub(crate) fn extents(dims: &[u64]) -> Shape {
    Shape::from(
        dims.iter()
            .map(|extent| *extent as usize)
            .collect::<Vec<_>>(),
    )
}

/// The boundary between a checkpoint and retained storage: converts a weight once and publishes
/// it.
///
/// A loader is one of two kinds, and both run the same decisions over the same shard headers:
///
/// - a loader over a store ([`WeightLoader::new`]) reads each tensor's payload, converts it and
///   publishes it to a [`RetainedWeights`] store;
/// - a **meta** loader ([`WeightLoader::meta`]) reads no payload and publishes nothing: every
///   weight it returns is a meta tensor of the element type and extents the other kind returns.
///
/// Every refusal a load decides from a tensor's name, element type or extents is a meta load's
/// refusal too. Only a refusal that depends on a tensor's values (an all-zero matrix asked to
/// hold a value, a zero or non-finite tensor scale) needs the payload.
pub struct WeightLoader<'a> {
    /// The checkpoint the weights come from.
    checkpoint: &'a Checkpoint,
    /// Where the weights go: retained storage, or meta tensors.
    retention: Retention<'a>,
    dtype: Option<DType>,
    quantization: Option<Quantization>,
    /// What the checkpoint declares about its packed weights, resolved against its shards; dense
    /// until a model [`declares`](WeightLoader::declare) one.
    declaration: Declaration,
}

impl<'a> WeightLoader<'a> {
    /// A loader over `checkpoint` publishing to `store`.
    ///
    /// `dtype` and `quantization` are the launch's explicit precision choices; with neither, a
    /// weight keeps the checkpoint's representation.
    pub fn new(
        checkpoint: &'a Checkpoint,
        store: &'a dyn RetainedWeights,
        dtype: Option<DType>,
        quantization: Option<Quantization>,
    ) -> Self {
        Self {
            checkpoint,
            retention: Retention::Store(store),
            dtype,
            quantization,
            declaration: Declaration::dense(),
        }
    }

    /// A loader over `checkpoint` that reads no tensor payload and publishes nothing.
    ///
    /// Each weight's element type and extents come from the shard headers [`Checkpoint::open`]
    /// already parsed, and `dtype` and `quantization` are the same precision choices
    /// [`WeightLoader::new`] takes. Every method returns what the device load of the same
    /// arguments returns (the same element type, the same extents, the same representation, the
    /// block-scale type the checkpoint stores, the published layout) and the same refusals,
    /// because it runs the device load's own code up to the point where bytes would be read. The
    /// tensors it returns are meta tensors at distinct addresses that no memory backs; a
    /// weight's key gives the same tensor twice, as a retained store returns a key it holds.
    ///
    /// What a meta load cannot give is a tensor's **value**: [`Self::tensor_scale`] returns a
    /// placeholder of one, a value-dependent refusal does not run, and the bytes of a computed
    /// constant derived from a placeholder are not the device's. The caller is a model's load
    /// for a profile run, which launches nothing and reads no weight: the shapes and element
    /// types decide the kernels a model launches, and the values never enter a kernel key.
    pub fn meta(
        checkpoint: &'a Checkpoint,
        dtype: Option<DType>,
        quantization: Option<Quantization>,
    ) -> Self {
        Self {
            checkpoint,
            retention: Retention::Meta(Rc::default()),
            dtype,
            quantization,
            declaration: Declaration::dense(),
        }
    }

    /// The checkpoint this loader reads.
    pub fn checkpoint(&self) -> &'a Checkpoint {
        self.checkpoint
    }

    /// Reads every later weight by `declaration`, resolved against this loader's checkpoint.
    ///
    /// The resolution reads the shard headers once: a weight is packed when its shard stores a
    /// block scale beside it, whatever the declaration's exclusion list says. A meta loader's
    /// checkpoint holds the same headers, so it resolves the same way.
    pub fn declare(&mut self, declaration: &Declaration) {
        self.declaration = declaration.resolve(self.checkpoint);
    }

    /// How the checkpoint stores the weight named `weight`, by the resolved declaration.
    pub fn scheme(&self, weight: &str) -> Scheme {
        self.declaration.scheme(weight)
    }

    /// The builder of the checkpoint's root: its tensors are named as the checkpoint names them,
    /// and a module's are reached by pushing its path ([`VarBuilder::pp`]).
    pub fn root(&self) -> VarBuilder<'_, 'a> {
        VarBuilder::new(self)
    }

    /// Whether this loader returns meta tensors instead of reading and publishing.
    pub(crate) fn is_meta(&self) -> bool {
        matches!(self.retention, Retention::Meta(_))
    }

    /// The weight already held under `key`, as `shape` elements of its type.
    fn retained(&self, key: &WeightKey, shape: &Shape) -> Result<Option<Tensor>> {
        match &self.retention {
            Retention::Store(store) => match store.retained(key, shape)? {
                Some(tensor) => Ok(Some(held(key, shape, tensor)?)),
                None => Ok(None),
            },
            Retention::Meta(tensors) => match tensors.borrow().get(key) {
                Some(tensor) => Ok(Some(tensor.reshape(shape.clone())?)),
                None => Ok(None),
            },
        }
    }

    /// Publishes the weight of `key`, `shape` elements of `key.dtype`, and returns its tensor.
    ///
    /// `bytes` produces the converted bytes from the checkpoint's payload; only a loader over a
    /// store calls it, so everything a meta load must decide happens before this call. A
    /// `writable` table goes to the store's writable storage and is never returned for a later
    /// key.
    pub(crate) fn publish_with<'c>(
        &self,
        key: &WeightKey,
        shape: &Shape,
        writable: bool,
        bytes: impl FnOnce(&Self) -> Result<Cow<'c, [u8]>>,
    ) -> Result<Tensor> {
        match &self.retention {
            Retention::Meta(tensors) => {
                let element = ops::abi::bytes(key.dtype).ok_or_else(|| {
                    Error::weight(
                        &key.region(),
                        format!(
                            "{} has no whole-byte element size",
                            ops::abi::name(key.dtype)
                        ),
                    )
                })?;
                let length = numel(shape)
                    .checked_mul(element)
                    .ok_or_else(|| Error::weight(&key.region(), "the weight's bytes overflow"))?;

                let tensor = meta_tensor(key, shape, length)?;
                if !writable {
                    tensors.borrow_mut().insert(key.clone(), tensor.clone());
                }
                Ok(tensor)
            }

            Retention::Store(store) => {
                let produced = bytes(self)?;
                let tensor = if writable {
                    store.writable(key, shape, &produced)?
                } else {
                    store.publish(key, shape, &produced)?
                };
                held(key, shape, tensor)
            }
        }
    }

    /// Loads the weight `name` whose tensors the checkpoint stores as weight-only NVFP4
    /// ([`Scheme::Nvfp4Weights`]), as the BF16 matrix their codes and scales state.
    ///
    /// `source` names the logical matrix, `[rows, columns]` of every tensor it reads, exactly as
    /// a dense read names it: a tensor-parallel block, a strided column block or a fused stack
    /// of experts. Each tensor is decoded once for the read, `code · block_scale · tensor_scale`
    /// in FP32 rounded once to BF16 (round to nearest even), and the source's region is gathered
    /// from the decoded matrices. The product that reads the weight is then the 16-bit product
    /// the declaration states.
    ///
    /// # Errors
    ///
    /// A tensor whose fields are missing or of another extent, or the source's own refusals.
    pub(crate) fn load_decoded(
        &self,
        name: &str,
        source: &WeightSource,
        layout: WeightLayout,
        shape: &Shape,
    ) -> Result<Tensor> {
        let names = source_names(source);
        let mut logical = Vec::with_capacity(names.len());

        for tensor in &names {
            let packed = self.checkpoint.locate(tensor)?;
            if packed.dtype != DType::U8 || packed.shape.rank() != 2 {
                return Err(Error::weight(
                    tensor,
                    format!(
                        "a weight-only NVFP4 weight is [rows, columns / 2] U8, not {} {:?}",
                        ops::abi::name(packed.dtype),
                        packed.shape.dims()
                    ),
                ));
            }

            let (rows, columns) = (dim(&packed.shape, 0), dim(&packed.shape, 1) * 2);
            logical.push(TensorLocation {
                file: packed.file,
                tensor: tensor.clone(),
                dtype: DType::BF16,
                shape: extents(&[rows, columns]),
                bytes: (0, rows * columns * 2),
            });
        }

        let files = self.identities(&names)?;
        let (start, source_shape, stride) = source_region(source);
        let key = WeightKey {
            files,
            dtype: DType::BF16,
            layout,
            tensors: names.clone(),
            start,
            shape: source_shape,
            stride,
            provenance: None,
        };
        if let Some(weight) = self.retained(&key, shape)? {
            return Ok(weight);
        }

        let spans = spans_in(source, &key, &|tensor| {
            logical.iter().find(|location| location.tensor == tensor)
        })?;
        let read: u64 = spans.iter().map(|span| (span.to - span.from) as u64).sum();
        if read != numel(shape) * 2 {
            return Err(Error::weight(
                name,
                format!(
                    "{read} bytes were read, {} expected for {:?} of bf16",
                    numel(shape) * 2,
                    shape.dims()
                ),
            ));
        }
        check_layout(name, source, layout)?;

        self.publish_with(&key, shape, false, |loader| {
            let mut decoded: Vec<(String, Vec<u8>)> = Vec::new();
            let mut bytes = Vec::with_capacity(read as usize);

            for span in &spans {
                let tensor = &span.location.tensor;
                if !decoded.iter().any(|(decoded, _)| decoded == tensor) {
                    let matrix = loader.decode_nvfp4(tensor, &span.location.shape)?;
                    decoded.push((tensor.clone(), matrix));
                }

                let (_, matrix) = decoded
                    .iter()
                    .find(|(decoded, _)| decoded == tensor)
                    .ok_or_else(|| Error::weight(tensor, "a decoded matrix went missing"))?;
                bytes.extend_from_slice(&matrix[span.from..span.to]);
            }
            Ok(Cow::Owned(bytes))
        })
    }

    /// The BF16 bytes of the `[rows, columns]` matrix the NVFP4 fields of `name` state.
    fn decode_nvfp4(&self, name: &str, shape: &Shape) -> Result<Vec<u8>> {
        let stem = name.strip_suffix("weight").ok_or_else(|| {
            Error::weight(name, "an NVFP4 weight is a module's `weight`".to_owned())
        })?;
        let (rows, columns) = (dim(shape, 0), dim(shape, 1));

        let packed = self.checkpoint.read(name)?;
        let scales_name = format!("{stem}weight_scale");
        let scales = self.checkpoint.locate(&scales_name)?;
        if scales.dtype != DType::F8E4M3 || numel(&scales.shape) * 16 != rows * columns {
            return Err(Error::weight(
                name,
                format!(
                    "block scales of {} {:?} for a {rows} × {columns} matrix; one F8_E4M3 per \
                     sixteen elements expected",
                    ops::abi::name(scales.dtype),
                    scales.shape.dims()
                ),
            ));
        }

        let scales = self.checkpoint.read(&scales_name)?;
        let tensor_scale = self.tensor_scale(&format!("{stem}weight_scale_2"))?;
        Ok(crate::quantization::decode_nvfp4(
            &packed,
            &scales,
            tensor_scale,
            columns as usize,
        ))
    }

    /// Loads the weight `name` from `source`, converting and publishing it the first time and
    /// reusing it afterwards.
    ///
    /// `shape` is the logical extents the weight is returned in, `layout` the arrangement its
    /// reading operator declares. `pinned` is the element type this tensor is read as whatever
    /// the deployment states: a representation the model pins per tensor (the reference keeps a
    /// timestep embedding in FP32 and a modulation projection in BF16 at every preset) wins over
    /// the loader's policy, and it is part of the retained key. `nonzero` asks that the tensor
    /// hold a value: a matrix that multiplies the hidden state is never all zero in a trained or
    /// a calibrated checkpoint, so an all-zero read of the bytes this load holds is refused by
    /// name, where a norm's scale or a bias may legitimately be zero and is not asked.
    ///
    /// # Errors
    ///
    /// When the checkpoint lacks a tensor the source names, the source reaches past one, the
    /// element types cannot be converted, the layout cannot be produced from the source, or the
    /// store refuses.
    pub fn load(
        &self,
        name: &str,
        source: &WeightSource,
        layout: WeightLayout,
        shape: &Shape,
        pinned: Option<DType>,
        nonzero: bool,
    ) -> Result<Tensor> {
        if let Some(quantization) = self.quantization {
            return Err(Error::Unsupported(format!(
                "quantization `{quantization}` is converted at load only for the FP8 and NVFP4 \
                 step; the BF16 slice keeps each checkpoint's own representation"
            )));
        }

        let names = source_names(source);
        let (start, source_shape, stride) = source_region(source);
        if let WeightSource::Parts { shape: fused, .. } | WeightSource::Blocks { shape: fused, .. } =
            source
            && shape != fused
        {
            return Err(Error::weight(
                name,
                format!(
                    "the weight is asked as {:?} but its parts are fused as {:?}",
                    shape.dims(),
                    fused.dims()
                ),
            ));
        }

        let files = self.identities(&names)?;
        let source_dtype = self.checkpoint_dtype(&names)?;
        let dtype = self.target_dtype(&names, pinned)?;
        let key = WeightKey {
            files,
            dtype,
            layout,
            tensors: names.clone(),
            start,
            shape: source_shape,
            stride,
            provenance: None,
        };

        if let Some(weight) = self.retained(&key, shape)? {
            return Ok(weight);
        }

        // The bytes the source names, computed from the headers: every bound below is checked
        // on a meta load as on a device load, and only the copy waits for the publish.
        let spans = self.spans(source, &key)?;
        if source_dtype != dtype {
            conversion(source_dtype, dtype, name)?;
        }

        let source_element = ops::abi::bytes(source_dtype).ok_or_else(|| {
            Error::weight(
                name,
                format!("{} has no whole-byte size", ops::abi::name(source_dtype)),
            )
        })?;
        let element = ops::abi::bytes(dtype).ok_or_else(|| {
            Error::weight(
                name,
                format!("{} has no whole-byte element size", ops::abi::name(dtype)),
            )
        })?;
        let expected = numel(shape)
            .checked_mul(element)
            .ok_or_else(|| Error::weight(name, "the weight's bytes overflow"))?;
        let read = spans
            .iter()
            .map(|span| (span.to - span.from) as u64)
            .sum::<u64>();

        // A conversion keeps the element count and changes its width.
        let converted = read / source_element * element;
        if converted != expected {
            return Err(Error::weight(
                name,
                format!(
                    "{converted} bytes were read, {expected} expected for {:?} of {}",
                    shape.dims(),
                    ops::abi::name(dtype)
                ),
            ));
        }
        check_layout(name, source, layout)?;

        self.publish_with(&key, shape, false, |loader| {
            let bytes = loader.gather(&spans)?;

            // The region's own bytes, before any conversion: one read, which this load was
            // doing anyway, so the check costs no second pass over the shard.
            if nonzero && bytes.iter().all(|byte| *byte == 0) {
                return Err(Error::weight(
                    name,
                    "every stored byte of the region is zero: the product would give zeros for \
                     every activation, so this tensor is missing its export"
                        .to_owned(),
                ));
            }

            // The key carries the retained element type; the bytes still hold the checkpoint's,
            // so the conversion is decided by the source, not by the key.
            if source_dtype != dtype {
                Ok(Cow::Owned(convert(source_dtype, dtype, &bytes, name)?))
            } else {
                Ok(Cow::Owned(bytes))
            }
        })
    }

    /// Publishes the weight-normalized form of `v` and `g` as one weight.
    ///
    /// A checkpoint stores a weight-normalized convolution as two tensors: `v` with the weight's
    /// extents and `g` with one magnitude per output channel (PyTorch keeps it as
    /// `[channels, 1, 1]`). The weight the convolution reads is `v · g / ‖v‖`, where the norm
    /// reduces every axis but the first per channel; it is computed once at loading, so a loaded
    /// model never holds the pair. The computation is FP32 and the result keeps the checkpoint's
    /// element type unless `pinned` or the loader's policy states another. Its extents are
    /// `v`'s, as the header states them.
    ///
    /// `name` is the logical tensor the key records; the key also names both sources, so it
    /// cannot be confused with a plain region of either.
    ///
    /// # Errors
    ///
    /// When a tensor is missing, the gain is not one magnitude per output channel, or an element
    /// type is not floating point.
    pub fn weight_norm(
        &self,
        name: &str,
        v: &str,
        g: &str,
        pinned: Option<DType>,
    ) -> Result<Tensor> {
        let value = self.checkpoint.locate(v)?;
        let gain = self.checkpoint.locate(g)?;
        if value.shape.rank() == 0 {
            return Err(Error::weight(
                name,
                "a weight-normalized tensor has an axis".to_owned(),
            ));
        }

        let channels = dim(&value.shape, 0);
        if numel(&gain.shape) != channels {
            return Err(Error::weight(
                name,
                format!(
                    "a gain of {} elements for {channels} output channels",
                    numel(&gain.shape)
                ),
            ));
        }

        let dtype = pinned.or(self.dtype).unwrap_or(value.dtype);
        if !dtype.is_float() || !value.dtype.is_float() {
            return Err(Error::weight(
                name,
                format!(
                    "weight normalization of {} is not floating point",
                    ops::abi::name(value.dtype)
                ),
            ));
        }

        // The computation reads both tensors as FP32 and writes the result back: each of the
        // three is one conversion this loader must have.
        conversion(value.dtype, DType::F32, v)?;
        conversion(gain.dtype, DType::F32, g)?;
        conversion(DType::F32, dtype, name)?;

        let names = vec![v.to_owned(), g.to_owned()];
        let key = WeightKey {
            files: self.identities(&names)?,
            dtype,
            layout: WeightLayout::ROW_MAJOR,
            // The logical name first: the double underscore marks a computed weight, so the key
            // is a region neither source names on its own.
            tensors: vec![format!("{name}__weight_norm"), v.to_owned(), g.to_owned()],
            start: 0,
            shape: value.shape.clone(),
            stride: None,
            provenance: None,
        };
        if let Some(weight) = self.retained(&key, &value.shape)? {
            return Ok(weight);
        }

        let (value_dtype, gain_dtype) = (value.dtype, gain.dtype);
        let elements = numel(&value.shape);
        self.publish_with(&key, &value.shape, false, |loader| {
            let values = to_f32(value_dtype, &loader.checkpoint.read(v)?, v)?;
            let gains = to_f32(gain_dtype, &loader.checkpoint.read(g)?, g)?;
            let per_channel = usize::try_from(elements / channels.max(1))
                .map_err(|_| Error::weight(name, "more elements than a usize".to_owned()))?;

            let mut normalized = Vec::with_capacity(values.len());
            for (channel, gain) in gains.iter().enumerate() {
                let rows = &values[channel * per_channel..][..per_channel];
                let sum = rows.iter().map(|value| value * value).sum::<f32>();
                let scale = gain / sum.sqrt();
                normalized.extend(rows.iter().map(|value| value * scale));
            }
            Ok(Cow::Owned(from_f32(dtype, &normalized, name)?))
        })
    }

    /// This loader over another checkpoint: the same store, the same element-type and
    /// quantization policy, a different shard set.
    ///
    /// A pipeline root holds one stage per subdirectory (`transformer/`, `vae/`,
    /// `text_encoder/`, …), each with its own index and its own file identities, so a model
    /// crate opens the root once and reads every stage through one store: a weight converted
    /// once for one stage is reused for another and never converted twice. A meta loader's
    /// reborrow is a meta loader over the same meta tensors.
    ///
    /// The policy is the loader's, not the stage's, so a stage that its policy does not describe
    /// is refused here: an FP8-policy loader over a BF16 stage, or a float policy over a stage
    /// whose weights are an FP8 or NVFP4 encoding, would read weights as something they are not.
    /// A model that needs two policies builds two loaders over the same store.
    ///
    /// # Errors
    ///
    /// [`Error::Policy`] when the policy does not describe `checkpoint`.
    pub fn reborrow<'b>(&'b self, checkpoint: &'b Checkpoint) -> Result<WeightLoader<'b>> {
        self.check_policy(checkpoint)?;

        let retention = match &self.retention {
            Retention::Store(store) => Retention::Store(*store),
            Retention::Meta(tensors) => Retention::Meta(Rc::clone(tensors)),
        };
        Ok(WeightLoader {
            checkpoint,
            retention,
            dtype: self.dtype,
            quantization: self.quantization,
            declaration: self.declaration.clone(),
        })
    }

    /// Whether this loader's policy describes `checkpoint`'s weights.
    fn check_policy(&self, checkpoint: &Checkpoint) -> Result<()> {
        let held = checkpoint.dtypes();
        let listed = held
            .iter()
            .map(|dtype| ops::abi::name(*dtype))
            .collect::<Vec<_>>()
            .join(", ");

        if let Some(quantization) = self.quantization {
            let encoding = quantization.encoding();
            if !held.contains(&encoding) {
                return Err(Error::policy(format!(
                    "a {quantization} policy encodes weights as {}, and this checkpoint holds \
                     {listed}",
                    ops::abi::name(encoding)
                )));
            }
        }

        if let Some(dtype) = self.dtype {
            for encoded in [DType::F8E4M3, DType::F4] {
                if held.contains(&encoded) {
                    return Err(Error::policy(format!(
                        "this checkpoint holds {} encoded weights, which a {} element type does \
                         not describe; the quantization is the loader's policy, not its element \
                         type",
                        ops::abi::name(encoded),
                        ops::abi::name(dtype)
                    )));
                }
            }
        }
        Ok(())
    }

    /// Runs `check` over the checkpoint's payload, on a loader that reads one.
    ///
    /// A refusal that depends on a tensor's values and not on its element type or extents can
    /// only be made with the payload in hand; a meta load skips it.
    pub(crate) fn check_values(&self, check: impl FnOnce(&Checkpoint) -> Result<()>) -> Result<()> {
        if self.is_meta() {
            return Ok(());
        }
        check(self.checkpoint)
    }

    /// The single positive scale the tensor named `tensor` holds, as the checkpoint states it.
    ///
    /// Both tensor scales a calibrated NVFP4 export writes (the weight's `weight_scale_2` and
    /// the activation's `input_scale`) are one positive float each, as are the two of a
    /// per-tensor FP8 projection, and a loader that synthesized one would encode at an invented
    /// magnitude. A projection reads its weight's scale to build the product's `alpha` and its
    /// activation's to build the encoder's operand; the reciprocals are its callers'.
    ///
    /// The tensor must be one floating number by its header on either kind of loader. A meta
    /// loader reads no value and returns a placeholder of one.
    ///
    /// # Errors
    ///
    /// When the tensor is missing or not one floating number, or its value is zero, negative or
    /// not finite.
    pub fn tensor_scale(&self, tensor: &str) -> Result<f32> {
        let location = self.checkpoint.locate(tensor)?;
        if numel(&location.shape) != 1 || !location.dtype.is_float() {
            return Err(Error::weight(
                tensor,
                format!(
                    "a tensor scale is one floating number, not {} {:?}",
                    ops::abi::name(location.dtype),
                    location.shape.dims()
                ),
            ));
        }
        conversion(location.dtype, DType::F32, tensor)?;
        if self.is_meta() {
            return Ok(META_TENSOR_SCALE);
        }

        let bytes = self.checkpoint.read(tensor)?;
        let scale = to_f32(location.dtype, &bytes, tensor)?[0];
        if !(scale.is_finite() && scale > 0.0) {
            return Err(Error::weight(
                tensor,
                format!("a tensor scale of {scale} is not a positive number"),
            ));
        }
        Ok(scale)
    }

    /// Publishes bytes this loader computes, under a key it built, and returns the weight.
    ///
    /// A key already held is returned without calling `bytes`, and a meta loader never calls it.
    pub(crate) fn publish_computed<'c>(
        &self,
        key: &WeightKey,
        shape: &Shape,
        bytes: impl FnOnce(&Self) -> Result<Cow<'c, [u8]>>,
    ) -> Result<Tensor> {
        if let Some(weight) = self.retained(key, shape)? {
            return Ok(weight);
        }
        self.publish_with(key, shape, false, bytes)
    }

    /// Publishes host-computed bytes as a weight, without a checkpoint tensor behind them.
    ///
    /// The rotary frequency table is such a weight: `inv_freq` is computed once at loading from
    /// `rope_theta` and `rope_scaling`, and the operator that reads it reads a device weight, so
    /// it takes the same retained-storage path as a checkpoint weight. There is no file to name
    /// it, so the key states the **provenance**: the class that computed it, `producer`, and a
    /// digest of the bytes it produced. Two computations of the same numbers share one retained
    /// weight; two that compute different numbers under one name never do.
    ///
    /// # Errors
    ///
    /// When `bytes` is not `shape` elements of `dtype`, or `dtype` has no whole-byte element.
    pub fn constant(
        &self,
        name: &str,
        dtype: DType,
        shape: &Shape,
        bytes: &[u8],
        producer: &'static str,
    ) -> Result<Tensor> {
        self.computed(name, dtype, shape, bytes, producer, false)
    }

    /// A table the model writes once after the load, holding `bytes` until then.
    ///
    /// It is a [`Self::constant`] that stays writable on the device, and it is this process's
    /// own: no later load reads it back, because the launch that fills it runs in every load.
    ///
    /// # Errors
    ///
    /// As [`Self::constant`].
    pub fn writable_constant(
        &self,
        name: &str,
        dtype: DType,
        shape: &Shape,
        bytes: &[u8],
        producer: &'static str,
    ) -> Result<Tensor> {
        self.computed(name, dtype, shape, bytes, producer, true)
    }

    fn computed(
        &self,
        name: &str,
        dtype: DType,
        shape: &Shape,
        bytes: &[u8],
        producer: &'static str,
        writable: bool,
    ) -> Result<Tensor> {
        let element = ops::abi::bytes(dtype).ok_or_else(|| {
            Error::weight(
                name,
                format!("{} has no whole-byte element size", ops::abi::name(dtype)),
            )
        })?;
        let expected = numel(shape)
            .checked_mul(element)
            .ok_or_else(|| Error::weight(name, "the constant's bytes overflow"))?;
        if bytes.len() as u64 != expected {
            return Err(Error::weight(
                name,
                format!(
                    "{} bytes for {:?} of {}, {expected} expected",
                    bytes.len(),
                    shape.dims(),
                    ops::abi::name(dtype)
                ),
            ));
        }

        let key = WeightKey {
            files: Vec::new(),
            dtype,
            layout: WeightLayout::ROW_MAJOR,
            tensors: vec![name.to_owned()],
            start: 0,
            shape: shape.clone(),
            stride: None,
            provenance: Some(Provenance::computed(producer, bytes)),
        };
        if writable {
            self.publish_with(&key, shape, true, |_| Ok(Cow::Borrowed(bytes)))
        } else {
            self.publish_computed(&key, shape, |_| Ok(Cow::Borrowed(bytes)))
        }
    }

    /// The identity of each of `names`' files, in order.
    pub(crate) fn identities(&self, names: &[String]) -> Result<Vec<FileIdentity>> {
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
                            "the fused parts mix {} and {} elements",
                            ops::abi::name(seen),
                            ops::abi::name(location.dtype)
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
                format!(
                    "converting {} to {} is not a floating-point conversion",
                    ops::abi::name(checkpoint),
                    ops::abi::name(dtype)
                ),
            ));
        }
        Ok(dtype)
    }

    /// The byte ranges a source names, in the source's logical order, computed from the headers.
    ///
    /// Every bound is checked against the length the header states for each tensor, which is the
    /// length of the payload slice a device load copies from.
    fn spans(&self, source: &WeightSource, key: &WeightKey) -> Result<Vec<Span<'a>>> {
        let checkpoint = self.checkpoint;
        spans_in(source, key, &|name| checkpoint.tensor(name))
    }

    /// The payload bytes of `spans`, concatenated in order.
    ///
    /// Consecutive spans of one tensor are read in one [`Checkpoint::payload`] call, so a strided
    /// shard of an adapter-fused tensor, one span a row, reads the adapter's factors once.
    fn gather(&self, spans: &[Span<'_>]) -> Result<Vec<u8>> {
        let total = spans.iter().map(|span| span.to - span.from).sum();
        let mut gathered = Vec::with_capacity(total);
        let mut at = 0;

        while at < spans.len() {
            let location = spans[at].location;
            let run = spans[at..]
                .iter()
                .take_while(|span| std::ptr::eq(span.location, location))
                .count();
            let ranges = spans[at..at + run]
                .iter()
                .map(|span| (span.from, span.to))
                .collect::<Vec<_>>();

            gathered.extend(self.checkpoint.payload(location, &ranges)?);
            at += run;
        }
        Ok(gathered)
    }
}

/// `tensor`, which a store returned for `key`, checked to be `shape` elements of the key's type.
///
/// A store that gave another extent or element type would have a module compute on bytes it was
/// not loaded for, so the disagreement is a loading failure here and not a refused launch later.
fn held(key: &WeightKey, shape: &Shape, tensor: Tensor) -> Result<Tensor> {
    if tensor.dims() != shape.dims() || tensor.dtype() != key.dtype {
        return Err(Error::weight(
            &key.region(),
            format!(
                "retained storage returned {} {:?} for a weight of {} {:?}",
                ops::abi::name(tensor.dtype()),
                tensor.dims(),
                ops::abi::name(key.dtype),
                shape.dims()
            ),
        ));
    }
    Ok(tensor)
}

/// A meta tensor of `shape` elements of the key's type, `bytes` bytes long, at an address of its
/// own.
fn meta_tensor(key: &WeightKey, shape: &Shape, bytes: u64) -> Result<Tensor> {
    // Every weight takes at least one block, so even an empty weight has an address no other
    // weight shares.
    let span = bytes
        .max(1)
        .checked_next_multiple_of(ALIGNMENT)
        .ok_or_else(|| Error::weight(&key.region(), "the weight's bytes overflow"))?;
    let address = META_CURSOR.fetch_add(span, Ordering::Relaxed);

    if address.checked_add(span).is_none() {
        return Err(Error::weight(
            &key.region(),
            "the meta loader's address range is exhausted",
        ));
    }
    Ok(backend::retained(
        address,
        shape.clone(),
        key.dtype,
        &backend::meta(),
    )?)
}

/// The checkpoint tensors `source` reads, each once, in the order it first names them.
pub(crate) fn source_names(source: &WeightSource) -> Vec<String> {
    match source {
        WeightSource::Region { tensor, .. } | WeightSource::Strided { tensor, .. } => {
            vec![tensor.clone()]
        }
        WeightSource::Parts { parts, .. } => parts.clone(),
        WeightSource::Blocks { parts, .. } => {
            let mut names: Vec<String> = Vec::new();
            for (name, _) in parts {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
            names
        }
    }
}

/// The first element, the extents and the row stride that key `source`'s region.
///
/// A block's start and stride key the region: two ranks' blocks of one expert stack have one
/// fused shape and must not share retained storage, and two ranks' blocks of one stacked tensor
/// begin at different rows.
fn source_region(source: &WeightSource) -> (u64, Shape, Option<u64>) {
    match source {
        WeightSource::Region { start, shape, .. } => (*start, shape.clone(), None),
        WeightSource::Strided {
            start,
            shape,
            row_stride,
            ..
        } => (*start, shape.clone(), Some(*row_stride)),
        WeightSource::Parts { shape, block, .. } => (
            block.map_or(0, |block| block.start),
            shape.clone(),
            block.map(|block| block.row_stride),
        ),
        WeightSource::Blocks { parts, shape } => {
            let first = parts.first().map(|(_, block)| *block);
            (
                first.map_or(0, |block| block.start),
                shape.clone(),
                first.map(|block| block.row_stride),
            )
        }
    }
}

/// The byte ranges `source` names, in its logical order, over the tensors `locate` finds: the
/// checkpoint's own, or the decoded matrices a weight-only NVFP4 read gathers from.
///
/// Every bound is checked against the length each location states.
fn spans_in<'l>(
    source: &WeightSource,
    key: &WeightKey,
    locate: &dyn Fn(&str) -> Option<&'l TensorLocation>,
) -> Result<Vec<Span<'l>>> {
    let element_of = |name: &str, location: &TensorLocation| {
        ops::abi::bytes(location.dtype).ok_or_else(|| {
            Error::weight(
                name,
                format!("{} has no whole-byte size", ops::abi::name(location.dtype)),
            )
        })
    };

    match source {
        WeightSource::Region {
            tensor,
            start,
            shape,
        } => {
            let location = locate(tensor).ok_or_else(|| Error::MissingTensor(tensor.clone()))?;
            let length = stored_bytes(location);
            let element = element_of(tensor, location)?;

            let begin = usize::try_from(start * element).map_err(|_| {
                Error::weight(tensor, "the region's first element is past usize::MAX")
            })?;
            let end = usize::try_from((start + numel(shape)) * element).map_err(|_| {
                Error::weight(tensor, "the region's last element is past usize::MAX")
            })?;
            if end as u64 > length {
                return Err(Error::weight(
                    tensor,
                    format!(
                        "elements {start}..{} of {:?} reach past the tensor's {length}",
                        start + numel(shape),
                        location.shape.dims(),
                    ),
                ));
            }

            Ok(vec![Span {
                location,
                from: begin,
                to: end,
            }])
        }

        WeightSource::Strided {
            tensor,
            start,
            shape,
            row_stride,
        } => {
            let location = locate(tensor).ok_or_else(|| Error::MissingTensor(tensor.clone()))?;
            let element = element_of(tensor, location)?;
            if shape.rank() == 0 {
                return Err(Error::weight(
                    tensor,
                    "a gathered weight has an axis".to_owned(),
                ));
            }

            let row = last(shape);
            if *row_stride < row {
                return Err(Error::weight(
                    tensor,
                    format!("a stride of {row_stride} overlaps rows of {row} elements"),
                ));
            }

            let rows = numel(shape) / row.max(1);
            let length = stored_bytes(location);
            let mut spans = Vec::with_capacity(usize::try_from(rows).unwrap_or(0));
            for index in 0..rows {
                let begin = start + index * row_stride;
                let from = usize::try_from(begin * element)
                    .map_err(|_| Error::weight(tensor, "a gathered row starts past usize::MAX"))?;
                let to = usize::try_from((begin + row) * element)
                    .map_err(|_| Error::weight(tensor, "a gathered row ends past usize::MAX"))?;

                if to as u64 > length {
                    return Err(Error::weight(
                        tensor,
                        format!(
                            "row {index} (elements {begin}..{}) reaches past the tensor's \
                             {length} bytes",
                            begin + row,
                        ),
                    ));
                }
                spans.push(Span { location, from, to });
            }
            Ok(spans)
        }

        WeightSource::Blocks { parts, .. } => {
            let mut spans = Vec::new();
            for (name, block) in parts {
                spans.extend(block_spans(locate, name, block)?);
            }
            fused(spans, key)
        }

        WeightSource::Parts {
            parts,
            block: Some(block),
            ..
        } => {
            let mut spans = Vec::new();
            for name in parts {
                spans.extend(block_spans(locate, name, block)?);
            }
            fused(spans, key)
        }

        WeightSource::Parts { parts, .. } => {
            let mut spans = Vec::with_capacity(parts.len());
            let mut part_shape: Option<&Shape> = None;

            for name in parts {
                let location = locate(name).ok_or_else(|| Error::MissingTensor(name.clone()))?;
                if location.shape.rank() == 0 {
                    return Err(Error::weight(name, "a scalar cannot be fused"));
                }

                match part_shape {
                    None => part_shape = Some(&location.shape),
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

                let to = usize::try_from(stored_bytes(location))
                    .map_err(|_| Error::weight(name, "a part is past usize::MAX"))?;
                spans.push(Span {
                    location,
                    from: 0,
                    to,
                });
            }
            fused(spans, key)
        }
    }
}

/// `spans`, the parts of a fused weight, refused when no part holds a byte.
fn fused<'l>(spans: Vec<Span<'l>>, key: &WeightKey) -> Result<Vec<Span<'l>>> {
    if spans.iter().all(|span| span.from == span.to) {
        return Err(Error::weight(
            &key.tensors.join("+"),
            "a fused weight has no part",
        ));
    }
    Ok(spans)
}

/// The byte ranges of `block`'s rows in the checkpoint tensor `name`, checked against its length.
fn block_spans<'a>(
    locate: &dyn Fn(&str) -> Option<&'a TensorLocation>,
    name: &str,
    block: &PartBlock,
) -> Result<Vec<Span<'a>>> {
    let location = locate(name).ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
    let element = ops::abi::bytes(location.dtype).ok_or_else(|| {
        Error::weight(
            name,
            format!("{} has no whole-byte size", ops::abi::name(location.dtype)),
        )
    })?;
    if block.row_stride < block.columns {
        return Err(Error::weight(
            name,
            format!(
                "a stride of {} overlaps rows of {} elements",
                block.row_stride, block.columns
            ),
        ));
    }

    let length = stored_bytes(location);
    let mut spans = Vec::with_capacity(usize::try_from(block.rows).unwrap_or(0));
    for row in 0..block.rows {
        let first = (block.start + row * block.row_stride) * element;
        let last = first + block.columns * element;
        let (Ok(from), Ok(to)) = (usize::try_from(first), usize::try_from(last)) else {
            return Err(Error::weight(name, "a block row is past usize::MAX"));
        };

        if to as u64 > length {
            return Err(Error::weight(
                name,
                format!(
                    "row {row} of the block (bytes {from}..{to}) reaches past the part's {length} \
                     bytes"
                ),
            ));
        }
        spans.push(Span { location, from, to });
    }
    Ok(spans)
}

/// Verifies the loader can produce `layout` from `source`.
fn check_layout(tensor: &str, source: &WeightSource, layout: WeightLayout) -> Result<()> {
    match layout {
        WeightLayout::ROW_MAJOR => Ok(()),
        WeightLayout::UP_GATE | WeightLayout::DOWN => match source {
            WeightSource::Parts { .. } | WeightSource::Blocks { .. } => Ok(()),
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

/// The elements of `tensor`, a contiguous or strided CPU tensor of integers or floats, as the
/// bytes a device holds them in: row-major, little-endian.
///
/// # Errors
///
/// When `tensor` is not on the CPU device, or its element type is a packed one, which no module
/// computes a table in.
pub(crate) fn host_bytes(name: &str, tensor: &Tensor) -> Result<Vec<u8>> {
    let tensor = tensor.contiguous()?;
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cpu(storage) = &*storage else {
        return Err(Error::weight(
            name,
            "a table a module computes is a CPU tensor until it is published",
        ));
    };
    let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
        Error::weight(
            name,
            "a contiguous tensor's elements are one range of its storage",
        )
    })?;

    /// The little-endian bytes of `values`, each through `bytes`.
    fn collect<T: Copy, const N: usize>(values: &[T], bytes: impl Fn(T) -> [u8; N]) -> Vec<u8> {
        values.iter().flat_map(|value| bytes(*value)).collect()
    }

    Ok(match storage {
        CpuStorage::U8(values) => values[start..end].to_vec(),
        CpuStorage::U32(values) => collect(&values[start..end], u32::to_le_bytes),
        CpuStorage::I16(values) => collect(&values[start..end], i16::to_le_bytes),
        CpuStorage::I32(values) => collect(&values[start..end], i32::to_le_bytes),
        CpuStorage::I64(values) => collect(&values[start..end], i64::to_le_bytes),
        CpuStorage::BF16(values) => {
            collect(&values[start..end], |value| value.to_bits().to_le_bytes())
        }
        CpuStorage::F16(values) => {
            collect(&values[start..end], |value| value.to_bits().to_le_bytes())
        }
        CpuStorage::F32(values) => collect(&values[start..end], f32::to_le_bytes),
        CpuStorage::F64(values) => collect(&values[start..end], f64::to_le_bytes),
        _ => {
            return Err(Error::weight(
                name,
                format!(
                    "a table of {} elements has no host values to publish",
                    ops::abi::name(tensor.dtype())
                ),
            ));
        }
    })
}

/// Converts the floating-point elements of `bytes` from `from` to `to`.
///
/// The checkpoint is stored in one of the three floating-point formats every model here uses,
/// and a conversion goes through `f32`: a checkpoint's own representation is kept unless the
/// launch states another, and an explicit statement is one conversion. `tensor` is what the
/// refusal names when the pair is not one this loader converts.
///
/// A weight narrowed to BF16 keeps its top sixteen bits: the loader truncates where the tensor
/// library's `to_dtype` rounds to nearest. Served weights were narrowed this way, so a module
/// that narrows a weight at load asks the loader for the element type and does not cast.
///
/// # Errors
///
/// When the pair is not converted, or `bytes` is not whole elements of `from`.
pub fn convert(from: DType, to: DType, bytes: &[u8], tensor: &str) -> Result<Vec<u8>> {
    conversion(from, to, tensor)?;
    if from == to {
        return Ok(bytes.to_vec());
    }

    let source_bytes = usize::try_from(ops::abi::bytes(from).unwrap_or(4)).unwrap_or(4);
    let target_bytes = usize::try_from(ops::abi::bytes(to).unwrap_or(4)).unwrap_or(4);
    if !bytes.len().is_multiple_of(source_bytes) {
        return Err(Error::weight(
            tensor,
            format!(
                "{} bytes are not whole {} elements",
                bytes.len(),
                ops::abi::name(from)
            ),
        ));
    }

    let count = bytes.len() / source_bytes;
    let mut converted = Vec::with_capacity(count * target_bytes);
    for index in 0..count {
        let chunk = &bytes[index * source_bytes..][..source_bytes];
        let value = match from {
            DType::BF16 => bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            DType::F16 => f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            DType::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            // A PCM sample is an integer that F32 holds exactly, so the read is lossless.
            DType::I16 => f32::from(i16::from_le_bytes([chunk[0], chunk[1]])),
            other => {
                return Err(Error::weight(
                    tensor,
                    format!(
                        "converting {} is not a floating-point conversion",
                        ops::abi::name(other)
                    ),
                ));
            }
        };

        match to {
            // The top sixteen bits: truncation, as the function's documentation states.
            DType::BF16 => {
                converted.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            DType::F16 => converted.extend_from_slice(&f32_to_f16(value).to_le_bytes()),
            DType::F32 => converted.extend_from_slice(&value.to_le_bytes()),
            DType::I16 => converted.extend_from_slice(&f32_to_i16(value).to_le_bytes()),
            other => {
                return Err(Error::weight(
                    tensor,
                    format!(
                        "converting to {} is not a floating-point conversion",
                        ops::abi::name(other)
                    ),
                ));
            }
        }
    }
    Ok(converted)
}

/// Whether [`convert`] converts `from` to `to`: one of its pairs, or the same type.
///
/// A load decides this from the element types alone, before any byte is read, so a meta load
/// refuses the pairs a device load refuses.
pub(crate) fn conversion(from: DType, to: DType, tensor: &str) -> Result<()> {
    match (from, to) {
        (DType::BF16, DType::F32)
        | (DType::BF16, DType::F16)
        | (DType::F16, DType::F32)
        | (DType::F16, DType::BF16)
        | (DType::F32, DType::BF16)
        | (DType::F32, DType::F16)
        | (DType::F32, DType::I16)
        | (DType::I16, DType::F32) => Ok(()),
        (same, target) if same == target => Ok(()),
        (source, target) => Err(Error::weight(
            tensor,
            format!(
                "converting {} to {} is not implemented",
                ops::abi::name(source),
                ops::abi::name(target)
            ),
        )),
    }
}

/// The `i16` a sample becomes: nearest, ties to even, saturating at both ends.
///
/// A PCM track is a clip, not a wrap: a sample beyond the range stores at the end of it instead
/// of folding to the opposite sign, and a NaN stores as zero (the saturating cast's value, and a
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
pub(crate) fn to_f32(dtype: DType, bytes: &[u8], tensor: &str) -> Result<Vec<f32>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_names_the_checkpoint_the_type_the_layout_and_the_region() {
        let file = FileIdentity::of(std::path::Path::new("/proc/self/exe")).unwrap();
        let key = WeightKey {
            files: vec![file.clone()],
            dtype: DType::BF16,
            layout: WeightLayout::UP_GATE,
            tensors: vec!["up".to_owned(), "gate".to_owned()],
            start: 0,
            shape: Shape::from((128, 1536, 2048)),
            stride: None,
            provenance: None,
        };

        assert_eq!(
            key.to_string(),
            format!(
                "{file}|bf16|up_gate|up+gate[0..{}][128, 1536, 2048]",
                128 * 1536 * 2048
            )
        );

        // A strided region and a computed weight add their own fields, in this order.
        let strided = WeightKey {
            stride: Some(64),
            provenance: Some(Provenance::computed("nn::Linear", &[1, 2, 3])),
            ..key
        };
        let text = strided.to_string();
        let tail = text
            .split("|stride_64|computed:nn::Linear:")
            .nth(1)
            .unwrap();
        assert_eq!(tail.len(), 64, "a SHA-256 digest in hexadecimal");
    }

    /// A PCM sample converts to its nearest integer, ties to even, and clamps at both ends: a
    /// clip does not wrap, and the two ends are the values an audio codec actually reaches.
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
        let floats = to_f32(DType::I16, &converted, "pcm").unwrap();
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
            let error = convert(from, to, &[0u8; 2], "pcm").unwrap_err().to_string();
            assert!(
                error.contains(ops::abi::name(from)) && error.contains(ops::abi::name(to)),
                "{} to {} was refused without naming both: {error}",
                ops::abi::name(from),
                ops::abi::name(to)
            );
        }
    }

    /// A weight narrowed to BF16 keeps its top sixteen bits, where rounding to nearest would
    /// carry into them.
    #[test]
    fn a_weight_narrowed_to_bf16_is_truncated() {
        // 1 + 2⁻⁸ + 2⁻⁹ lies three quarters of a BF16 step above one: nearest is the next value
        // up, the truncation is one.
        let value = 1.0_f32 + 2f32.powi(-8) + 2f32.powi(-9);
        let narrowed = convert(DType::F32, DType::BF16, &value.to_le_bytes(), "w").unwrap();
        assert_eq!(u16::from_le_bytes([narrowed[0], narrowed[1]]), 0x3f80);
    }

    #[test]
    fn a_computed_table_s_bytes_are_its_elements_in_row_major_order() {
        let table = Tensor::arange(0_u32, 6, &Device::Cpu)
            .unwrap()
            .reshape((2, 3))
            .unwrap();
        // The transpose is a strided view: its bytes are the elements it names, in its order.
        let bytes = host_bytes("table", &table.t().unwrap()).unwrap();
        let elements: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(elements, [0, 3, 1, 4, 2, 5]);
    }
}

//! A linear projection `out = input · weightᵀ + bias`, in whichever representation the checkpoint
//! stores its weight, and the parallel forms built from it.
//!
//! ## Loading
//!
//! [`Linear::load`] reads one module's `weight` (and `bias`) through its [`Scope`] and chooses the
//! representation from the checkpoint's quantization declaration: a dense weight is one `gemm`; a
//! calibrated NVFP4 weight is an activation encoder (`nvfp4_quantize`) and a packed product
//! (`nvfp4_gemm`) whose `alpha` is the weight's tensor scale times the activation's. The caller
//! names the module, its widths and its split, and never the representation.
//!
//! ## Parallelism
//!
//! [`Parallel::Column`] holds this rank's block of the output channels: its rows of the
//! checkpoint's `[out, in]` matrix and its slice of the bias. [`Parallel::Row`] holds this rank's
//! block of the contraction: its columns of every row, gathered at load into one contiguous weight.
//! A row-parallel product is a partial sum, so its forward reduces the output over the split's
//! group through [`Forward::all_reduce`] and refuses a forward that carries no group. The bias of a
//! row-parallel projection is added on rank 0 alone, so the reduction counts it once.
//!
//! [`QkvLinear`] composes the query, key and value projections of an attention (separate checkpoint
//! tensors or windows of one fused tensor, the key/value heads replicated when the split is wider
//! than they are); [`MergedColumnLinear`] reads several checkpoint matrices as one product.

use ops::fp4::{
    BLOCK_ELEMENTS, GemmArgs, GemmDescriptor, Nvfp4Gemm, Nvfp4Quantize, QuantizeArgs,
    QuantizeDescriptor, ScaleLayout,
};
use ops::gemm::{self, Descriptor, Epilogue, Gemm, WeightOrder};
use ops::{DType, Shape, TensorView};

use crate::error::{Error, Result};
use crate::family::Split;
use crate::forward::Forward;
use crate::layers::ACCUMULATE;
use crate::quantization::Scheme;
use crate::weights::{Nvfp4Names, QuantizedWeight, Scope, Weight, WeightLayout, WeightRequest};

/// Elements of a row one activation-encoding program covers.
///
/// A whole number of sixteen-element scale blocks, as the layout requires, and a key field of
/// `nvfp4_quantize`: every packed projection and expert block states this one value, so one launch
/// site compiles one program.
pub const ENCODE_CHUNK: u32 = 1024;

/// The encoder's chunk is a whole number of scale blocks; a chunk that were not would have a partial
/// block with no scale to name it.
const _: () = assert!(ENCODE_CHUNK.is_multiple_of(BLOCK_ELEMENTS));

/// The arrangement a calibrated ModelOpt export stores its block scales in: linear, one E4M3 scale
/// per sixteen elements, row-major. The published table is built from it at load.
const FILE_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Linear;

/// The arrangement the packed product reads the weight's block scales in.
const PRODUCT_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

/// Whether a projection adds a bias.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bias {
    /// The module states no bias.
    Absent,
    /// The module states a `bias` beside its `weight`.
    Present,
}

/// How a projection is split across a tensor-parallel group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Parallel {
    /// One rank holds the whole projection.
    None,
    /// The output channels are split: this rank computes its block of them.
    Column(Split),
    /// The contraction is split: this rank computes a partial sum, which its forward reduces.
    Row(Split),
}

/// The tile one encoding of a `columns`-wide row runs in.
///
/// The package refuses a tile that does not divide the row (`nvfp4_quantize`'s `build.py` raises on
/// `columns % block != 0`), so the requested tile is halved until it divides. A row narrower than
/// the largest tile — the mixture-of-experts' 768-element second encode — gets the largest tile that
/// fits it rather than a key no package can build.
pub fn encode_block(columns: u32, requested: u32) -> u32 {
    let mut block = requested.max(1);
    while block > 1 && !columns.is_multiple_of(block) {
        block /= 2;
    }
    block
}

/// A packed NVFP4 weight and the operands its encoder and product read.
#[derive(Clone, Copy)]
struct Packed {
    /// `[n, k/2]` U8: two E2M1 codes a byte along the contraction.
    values: Weight,
    /// The block scales in [`PRODUCT_SCALE_LAYOUT`], in the checkpoint's one-byte scale type.
    scales: Weight,
    /// `[1]` F32: the encoder's global scale, the reciprocal of the activation's tensor scale.
    input_scale: Weight,
    /// `[1]` F32: the product's alpha, the weight's tensor scale times the activation's.
    alpha: Weight,
}

/// The representation a projection's weight is held in.
#[derive(Clone)]
enum Form {
    /// The checkpoint's own matrix, in `order`.
    Dense { weight: Weight, order: WeightOrder },
    /// A calibrated NVFP4 matrix.
    Packed(Box<Packed>),
}

/// A linear projection in the representation its checkpoint stores.
pub struct Linear {
    form: Form,
    bias: Option<Weight>,
    k: u32,
    n: u32,
    /// Ranks this projection's partial sums are reduced over: above one for a row-parallel split.
    reduce: u32,
}

impl Linear {
    /// A dense projection over an already bound `weight` stored in `order`.
    ///
    /// The constructor for weights no checkpoint module states as one — a table a model computes at
    /// load, or the head's view of a vocabulary shard. `weight` is `[k, n]` in
    /// [`WeightOrder::KxN`] or `[n, k]` in [`WeightOrder::NxK`]; the bias, when present, has `n`
    /// elements.
    ///
    /// # Errors
    ///
    /// When the weight is not a matrix or the bias another width.
    pub fn new(weight: Weight, bias: Option<Weight>, order: WeightOrder) -> Result<Self> {
        if weight.shape().rank() != 2 {
            return Err(Error::shape(
                "Linear",
                format!("a weight is a matrix, not {:?}", weight.shape().dims()),
            ));
        }
        let (k, n) = match order {
            WeightOrder::KxN => (weight.shape().dim(0), weight.shape().dim(1)),
            WeightOrder::NxK => (weight.shape().dim(1), weight.shape().dim(0)),
        };
        Self::build(Form::Dense { weight, order }, bias, k, n, 1)
    }

    /// The projection the module at `scope` states: `inputs` to `outputs` channels, both the
    /// whole module's, split as `parallel` says.
    ///
    /// The checkpoint stores the module as `weight` `[outputs, inputs]` and, with [`Bias::Present`],
    /// `bias` `[outputs]`; a calibrated NVFP4 export stores `weight` packed beside `weight_scale`,
    /// `weight_scale_2` and `input_scale`, and the declaration the model's loader was given decides
    /// which this module is.
    ///
    /// # Errors
    ///
    /// When the split does not divide the split axis, a tensor is missing or of another extent, a
    /// dense matrix is all zero, or a packed module states a bias.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        inputs: u32,
        outputs: u32,
        bias: Bias,
        parallel: Parallel,
    ) -> Result<Self> {
        let (inputs, outputs) = (u64::from(inputs), u64::from(outputs));
        let (local_in, local_out) = match parallel {
            Parallel::None => (inputs, outputs),
            Parallel::Column(split) => (inputs, share(scope, outputs, split, "output channels")?),
            Parallel::Row(split) => (share(scope, inputs, split, "contraction")?, outputs),
        };
        let name = scope.name("weight");
        match scope.loader().scheme(&name) {
            Scheme::Dense => {
                let weight =
                    scope.load(weight_request(&name, inputs, local_out, local_in, parallel))?;
                let bias = match (bias, parallel) {
                    (Bias::Absent, _) => None,
                    // A partial sum's bias is added once, on the first rank, so the reduction counts
                    // it once.
                    (Bias::Present, Parallel::Row(split)) if split.rank != 0 => None,
                    (Bias::Present, Parallel::Row(_) | Parallel::None) => {
                        Some(scope.tensor("bias", &[outputs])?)
                    }
                    (Bias::Present, Parallel::Column(split)) => {
                        Some(scope.load(WeightRequest::region(
                            scope.name("bias"),
                            u64::from(split.rank) * local_out,
                            Shape::new(&[local_out]),
                        ))?)
                    }
                };
                let reduce = match parallel {
                    Parallel::Row(split) => split.size,
                    _ => 1,
                };
                Self::build(
                    Form::Dense {
                        weight,
                        order: WeightOrder::NxK,
                    },
                    bias,
                    local_in,
                    local_out,
                    reduce,
                )
            }
            Scheme::Nvfp4 => {
                if bias == Bias::Present {
                    return Err(Error::weight(
                        &name,
                        "a packed projection states a bias, and no calibrated NVFP4 export writes \
                         one",
                    ));
                }
                Self::load_packed(scope, inputs, local_in, local_out, parallel)
            }
        }
    }

    /// The packed fields of the module at `scope`, at this rank's block.
    ///
    /// The checkpoint stores two E2M1 codes a byte along the contraction, so a row is `in / 2`
    /// bytes and its block scales are `in / 16` of them; a rank's block is the same rows or columns
    /// of both. The weight's `weight_scale_2` and the activation's `input_scale` are one positive
    /// float each: the encoder reads the activation's reciprocal, and the product's alpha is the two
    /// scales' product, published once as a `[1]` F32 constant beside the weights.
    fn load_packed(
        scope: &mut Scope<'_, '_>,
        inputs: u64,
        local_in: u64,
        local_out: u64,
        parallel: Parallel,
    ) -> Result<Self> {
        let spelling = Nvfp4Names::vllm();
        let values = weight_request(
            &scope.name(spelling.values),
            inputs / 2,
            local_out,
            local_in / 2,
            parallel,
        );
        let scales = weight_request(
            &scope.name(spelling.block_scale),
            inputs / 16,
            local_out,
            local_in / 16,
            parallel,
        );
        let tensor_scale =
            WeightRequest::region(scope.name(spelling.tensor_scale), 0, Shape::new(&[1]));
        let input_scale =
            WeightRequest::region(scope.name(spelling.input_scale), 0, Shape::new(&[1]));
        let alpha_name = scope.name("alpha");
        let loader = scope.loader();
        let global_scale = loader.tensor_scale(&tensor_scale)?;
        // A dense product dequantizes `code * block_scale` on both sides, so its alpha is the two
        // tensor scales' product: the weight's and the activation's.
        let product_alpha = global_scale * loader.tensor_scale(&input_scale)?;
        let quantized = loader.nvfp4(
            &values,
            &scales,
            Some(&input_scale),
            spelling,
            FILE_SCALE_LAYOUT,
            PRODUCT_SCALE_LAYOUT,
            global_scale,
        )?;
        let alpha = loader.constant(
            &alpha_name,
            DType::F32,
            Shape::new(&[1]),
            &product_alpha.to_le_bytes(),
            "nn::Linear",
        )?;
        let mut linear = Self::packed(&quantized, alpha)?;
        if let Parallel::Row(split) = parallel {
            linear.reduce = split.size;
        }
        Ok(linear)
    }

    /// A projection over a calibrated NVFP4 matrix and its product's `alpha`.
    ///
    /// `weight.packed` is `[n, k/2]` bytes and `weight.scales` its block scales in the product's
    /// layout; the contraction must be whole sixteen-element blocks. A weight-only export has no
    /// activation scale and **cannot serve this projection**, which is refused here by name rather
    /// than encoded at an invented magnitude.
    pub(crate) fn packed(weight: &QuantizedWeight, alpha: Weight) -> Result<Self> {
        let refuse = |problem: String| Error::shape("Linear", problem);
        if weight.packed.shape().rank() != 2 || weight.packed.dtype() != DType::U8 {
            return Err(refuse(format!(
                "a packed weight is a U8 matrix, not {} {:?}",
                weight.packed.dtype(),
                weight.packed.shape().dims()
            )));
        }
        let n = weight.packed.shape().dim(0);
        let k = weight.packed.shape().dim(1) * 2;
        if k == 0 || !k.is_multiple_of(u64::from(BLOCK_ELEMENTS)) {
            return Err(refuse(format!(
                "a contraction of {k} elements is not whole sixteen-element blocks"
            )));
        }
        // One layout governs a product: it states both how the activation's block scales are
        // arranged and, by naming the product, how the weight's are read.
        if weight.layout != PRODUCT_SCALE_LAYOUT {
            return Err(refuse(format!(
                "the weight's block scales are {} and the product reads {PRODUCT_SCALE_LAYOUT}",
                weight.layout
            )));
        }
        let expected = weight.layout.shape(n, k);
        if weight.scales.shape() != expected {
            return Err(refuse(format!(
                "the block scales are {:?}; {n} rows of {k} elements take {:?}",
                weight.scales.shape().dims(),
                expected.dims()
            )));
        }
        let Some(input_scale) = weight.input_scale else {
            return Err(refuse(
                "this projection has no calibrated activation input scale: a weight-only export \
                 cannot be encoded by a program that reads one"
                    .to_owned(),
            ));
        };
        for (what, scalar) in [("input scale", input_scale), ("alpha", alpha)] {
            if scalar.dtype() != DType::F32 || scalar.shape().numel() != 1 {
                return Err(refuse(format!(
                    "the {what} is {} {:?}, and the product reads one FP32 number",
                    scalar.dtype(),
                    scalar.shape().dims()
                )));
            }
        }
        Self::build(
            Form::Packed(Box::new(Packed {
                values: weight.packed,
                scales: weight.scales,
                input_scale,
                alpha,
            })),
            None,
            k,
            n,
            1,
        )
    }

    /// A projection of `form`, its geometry checked.
    fn build(form: Form, bias: Option<Weight>, k: u64, n: u64, reduce: u32) -> Result<Self> {
        if let Some(bias) = bias
            && bias.shape().numel() != n
        {
            return Err(Error::shape(
                "Linear",
                format!(
                    "a bias of {} elements cannot be added to {n} output channels",
                    bias.shape().numel()
                ),
            ));
        }
        Ok(Self {
            form,
            bias,
            k: u32::try_from(k)
                .map_err(|_| Error::shape("Linear", format!("a contraction of {k} elements")))?,
            n: u32::try_from(n)
                .map_err(|_| Error::shape("Linear", format!("{n} output channels")))?,
            reduce,
        })
    }

    /// The contraction length.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// The number of output channels this rank computes.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// The weight view: the dense matrix, or the packed values of a packed projection.
    pub fn weight(&self) -> TensorView {
        match &self.form {
            Form::Dense { weight, .. } => weight.view(),
            Form::Packed(packed) => packed.values.view(),
        }
    }

    /// Whether the weight is packed, so the product writes whole row classes ([`ops::fp4::packed_rows`]).
    pub const fn is_packed(&self) -> bool {
        matches!(self.form, Form::Packed(_))
    }

    /// `out = input · weightᵀ + bias`, reduced over the split's group when the projection is
    /// row-parallel.
    ///
    /// `input` is `[m, k]` and `out` is `[m, n]`. A packed product is launched at the row class of
    /// `m` ([`ops::fp4::packed_rows`]) and writes whole tiles, so `out` must address a region that holds the
    /// class's rows; its first `m` rows are the answer.
    ///
    /// # Errors
    ///
    /// When a view does not fit the projection, an operator refuses, or a row-parallel projection
    /// runs on a forward that carries no group.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        let k = u64::from(self.k);
        let n = u64::from(self.n);
        input
            .require_row("Linear", input.dtype, k)
            .map_err(Error::from)?;
        out.require_row("Linear", out.dtype, n)
            .map_err(Error::from)?;
        if out.rows() != input.rows() {
            return Err(Error::shape(
                "Linear",
                format!(
                    "{} input rows against {} output rows",
                    input.rows(),
                    out.rows()
                ),
            ));
        }
        match &self.form {
            Form::Dense { weight, order } => {
                let descriptor = Descriptor {
                    input: input.dtype,
                    weight: weight.dtype(),
                    accumulate: ACCUMULATE,
                    output: out.dtype,
                    order: *order,
                    epilogue: self.epilogue(),
                    k: self.k,
                };
                let args = gemm::Args {
                    input,
                    weight: weight.view(),
                    bias: self.bias.as_ref().map(Weight::view),
                    out,
                    alpha: 1.0,
                };
                f.operate::<Gemm>(&descriptor, args)?;
            }
            Form::Packed(packed) => self.forward_packed(f, packed, input, out)?,
        }
        if self.reduce > 1 {
            // A partial sum over this rank's block of the contraction: the group sums the ranks'.
            if f.split().is_none() {
                return Err(Error::shape(
                    "Linear",
                    format!(
                        "a row-parallel projection over {} ranks runs on a forward that carries \
                         no group to reduce over",
                        self.reduce
                    ),
                ));
            }
            f.all_reduce(out)?;
        }
        Ok(())
    }

    /// [`Self::forward`] into an output of `dtype` reserved from the forward's arena.
    ///
    /// A packed product writes whole row classes ([`ops::fp4::packed_rows`]), so the reservation holds the
    /// class's rows and the answer is the view of its first `m` rows; a dense product's
    /// reservation is exactly `[m, n]`.
    ///
    /// # Errors
    ///
    /// As [`Self::forward`], a row count above the largest packed class, or an arena that cannot
    /// hold the output.
    pub fn apply(&self, f: &Forward, input: TensorView, dtype: DType) -> Result<TensorView> {
        let rows = input.rows();
        let held = if self.is_packed() {
            let counted = u32::try_from(rows)
                .map_err(|_| Error::shape("Linear", format!("{rows} activation rows")))?;
            u64::from(ops::fp4::packed_rows(counted)?)
        } else {
            rows
        };
        let n = u64::from(self.n);
        let region = f.arena().tensor(dtype, Shape::new(&[held, n]))?;
        let out = TensorView::new(region.ptr, dtype, Shape::new(&[rows, n]));
        self.forward(f, input, out)?;
        Ok(out)
    }

    /// The epilogue the bias asks for.
    fn epilogue(&self) -> Epilogue {
        if self.bias.is_some() {
            Epilogue::Bias
        } else {
            Epilogue::None
        }
    }

    /// Encode `input` and contract it with the packed weight, at the row class of its rows.
    fn forward_packed(
        &self,
        f: &Forward,
        packed: &Packed,
        input: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let k = u64::from(self.k);
        let rows = input.rows();
        if rows == 0 {
            return Err(Error::shape(
                "Linear",
                "a projection of no rows has no program".to_owned(),
            ));
        }
        // The encoder reads the activation's own type, which NVFP4 encodes from BF16 or FP16.
        if !matches!(input.dtype, DType::BF16 | DType::F16) {
            return Err(Error::shape(
                "Linear",
                format!("{} activations; NVFP4 encodes BF16 or FP16", input.dtype),
            ));
        }
        let counted = u32::try_from(rows)
            .map_err(|_| Error::shape("Linear", format!("{rows} activation rows")))?;
        let class = u64::from(ops::fp4::packed_rows(counted)?);

        // 1. Encode the activation into the layout the product reads, into this forward's scratch.
        //
        // The encode is two-level: the block's E4M3 scale is relative to the row's own factor
        // (`amax(row) · global_scale`), and the product dequantizes a code as
        // `code · block_scale · row factor`. The regions are class-height and the encode writes the
        // request's rows into their prefix; the product's rows past the request's are computed
        // from bytes nothing reads and discarded with the rows they land in.
        let arena = f.arena();
        let values_region = arena.tensor(DType::U8, Shape::new(&[class, k / 2]))?;
        let scale_shape = PRODUCT_SCALE_LAYOUT.shape(class, k);
        let scales = arena.tensor(DType::U8, scale_shape)?;
        let token_scale_region = arena.tensor(DType::F32, Shape::new(&[class]))?;
        let values = prefix(values_region, &[rows, k / 2]);
        // One scalar per row, so the view is rank one: the encode checks `[rows]`.
        let token_scale = prefix(token_scale_region, &[rows]);
        let encoded_scale_shape = PRODUCT_SCALE_LAYOUT.shape(rows, k);
        let encoded_scales = if encoded_scale_shape == scale_shape {
            scales
        } else {
            prefix(
                scales,
                &[encoded_scale_shape.rows(), encoded_scale_shape.last()],
            )
        };
        let quantize = QuantizeDescriptor {
            input: input.dtype,
            columns: self.k,
            layout: PRODUCT_SCALE_LAYOUT,
            block: encode_block(self.k, ENCODE_CHUNK),
            pdl: false,
        };
        let encoded = QuantizeArgs {
            input,
            values,
            scales: encoded_scales,
            global_scale: packed.input_scale.view(),
            token_scale,
        };
        f.operate::<Nvfp4Quantize>(&quantize, encoded)?;

        // 2. Contract the encoded activation with the weight, over the class's rows: the product
        // writes whole tiles, so it is handed the class-height region `out` addresses.
        let descriptor = GemmDescriptor {
            packed: DType::U8,
            scale: packed.scales.dtype(),
            accumulate: ACCUMULATE,
            output: out.dtype,
            layout: PRODUCT_SCALE_LAYOUT,
            epilogue: self.epilogue(),
            m: u32::try_from(class).unwrap_or(u32::MAX),
            n: self.n,
            k: self.k,
        };
        // The scale's type is the checkpoint's own statement, so a plane declared as anything but
        // E4M3 makes a key no store built - refused here, where the layer can say which plane.
        ops::fp4::check_gemm(&descriptor)?;
        let args = GemmArgs {
            input: values_region,
            weight: packed.values.view(),
            input_scales: scales,
            weight_scales: packed.scales.view(),
            alpha: packed.alpha.view(),
            bias: self.bias.as_ref().map(Weight::view),
            out: prefix(out, &[class, u64::from(self.n)]),
        };
        f.operate::<Nvfp4Gemm>(&descriptor, args)
    }
}

/// `extent` split over `split`'s ranks: this rank's share, refused when the split does not divide.
fn share(scope: &Scope<'_, '_>, extent: u64, split: Split, what: &str) -> Result<u64> {
    let size = u64::from(split.size);
    if size == 0 || split.rank >= split.size || !extent.is_multiple_of(size) {
        return Err(Error::weight(
            &scope.name("weight"),
            format!(
                "rank {} of a split of {} does not hold a share of {extent} {what}",
                split.rank, split.size
            ),
        ));
    }
    Ok(extent / size)
}

/// The request for this rank's block of the `[outputs, columns]` matrix `tensor`, whose rows are
/// `columns` elements wide; the block is `rows` rows of `width` elements.
///
/// A column-parallel block is contiguous rows; a row-parallel one is every row's own columns, a
/// gather of rows `columns` elements apart.
fn weight_request(
    tensor: &str,
    columns: u64,
    rows: u64,
    width: u64,
    parallel: Parallel,
) -> WeightRequest {
    let shape = Shape::new(&[rows, width]);
    let request = match parallel {
        Parallel::None => WeightRequest::region(tensor, 0, shape),
        Parallel::Column(split) => {
            WeightRequest::region(tensor, u64::from(split.rank) * rows * columns, shape)
        }
        Parallel::Row(split) => {
            WeightRequest::strided(tensor, u64::from(split.rank) * width, shape, columns)
        }
    };
    // A matrix that multiplies the hidden state is never all zero in a trained or calibrated
    // checkpoint: the product would answer zeros for every activation, so the region this load
    // already reads is asked to hold a value.
    request.nonzero()
}

/// The first rows of `buffer` as a row-major view of `dims`, at the buffer's address.
fn prefix(buffer: TensorView, dims: &[u64]) -> TensorView {
    let shape = Shape::new(dims);
    TensorView::with_strides(
        buffer.ptr,
        buffer.dtype,
        shape,
        ops::Strides::row_major(shape),
    )
}

/// The heads an attention's query, key and value projections hold.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QkvSpec {
    /// The hidden width the projections contract.
    pub hidden: u32,
    /// Query heads of the whole module.
    pub heads: u32,
    /// Key/value heads of the whole module.
    pub kv_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
}

/// Where a checkpoint stores an attention's three projections.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QkvNames<'n> {
    /// Three modules, query, key and value, each with its own `weight` (and `bias`).
    Separate([&'n str; 3]),
    /// One module whose `weight` stacks the three `[query; key; value]` along its rows.
    Fused(&'n str),
}

/// Where a checkpoint stores an attention's three biases.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QkvBias<'n> {
    /// No projection adds a bias.
    Absent,
    /// Each projection's module states its own `bias` (for [`QkvNames::Fused`], the fused
    /// module's `bias`, stacked as the weight is).
    Present,
    /// The scope states one bias tensor per projection under these names, in query, key, value
    /// order.
    Windows([&'n str; 3]),
}

/// An attention's query, key and value projections, each its own product.
///
/// The three are loaded from separate checkpoint modules or as row windows of one fused module and
/// split over a tensor-parallel group by heads. When the group is wider than the key/value heads,
/// each key/value head is replicated on `size / kv_heads` consecutive ranks, the rule the reference
/// projects use, so every rank holds the head its query heads read.
pub struct QkvLinear {
    query: Linear,
    key: Linear,
    value: Linear,
}

impl QkvLinear {
    /// The three projections of the attention at `scope`, at `split`'s rank.
    ///
    /// # Errors
    ///
    /// When the split divides neither the query heads nor (or into) the key/value heads, or a
    /// projection cannot be loaded.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        spec: QkvSpec,
        names: QkvNames<'_>,
        bias: QkvBias<'_>,
        split: Split,
    ) -> Result<Self> {
        let path = scope.path().to_owned();
        let refuse = |problem: String| Error::weight(&path, problem);
        if split.size == 0 || split.rank >= split.size || !spec.heads.is_multiple_of(split.size) {
            return Err(refuse(format!(
                "a split of {} does not divide {} query heads",
                split.size, spec.heads
            )));
        }
        // The key/value heads' own split: the group's, or - when the group is wider than the heads
        // - one head per `size / kv_heads` ranks.
        let kv_split = if split.size <= spec.kv_heads {
            if !spec.kv_heads.is_multiple_of(split.size) {
                return Err(refuse(format!(
                    "a split of {} does not divide {} key/value heads",
                    split.size, spec.kv_heads
                )));
            }
            split
        } else {
            if !split.size.is_multiple_of(spec.kv_heads) {
                return Err(refuse(format!(
                    "a split of {} does not replicate {} key/value heads evenly",
                    split.size, spec.kv_heads
                )));
            }
            Split {
                size: spec.kv_heads,
                rank: split.rank / (split.size / spec.kv_heads),
            }
        };
        let query_width = spec.heads * spec.head_dim;
        let kv_width = spec.kv_heads * spec.head_dim;
        let parallel = |split: Split| {
            if split.size == 1 {
                Parallel::None
            } else {
                Parallel::Column(split)
            }
        };
        match names {
            QkvNames::Separate(modules) => {
                let module_bias = match bias {
                    QkvBias::Absent => Bias::Absent,
                    QkvBias::Present => Bias::Present,
                    QkvBias::Windows(_) => {
                        return Err(refuse(
                            "separate projections state their own biases".to_owned(),
                        ));
                    }
                };
                let mut load = |module: &str, width: u32, split: Split| {
                    Linear::load(
                        &mut scope.scope(module),
                        spec.hidden,
                        width,
                        module_bias,
                        parallel(split),
                    )
                };
                Ok(Self {
                    query: load(modules[0], query_width, split)?,
                    key: load(modules[1], kv_width, kv_split)?,
                    value: load(modules[2], kv_width, kv_split)?,
                })
            }
            QkvNames::Fused(module) => {
                let hidden = u64::from(spec.hidden);
                // The query rows, then the key rows, then the value rows; each split's block is its
                // own rows of its window.
                let local_query = u64::from(query_width / split.size);
                let local_kv = u64::from(kv_width / kv_split.size);
                let windows = [
                    (0, local_query, split),
                    (u64::from(query_width), local_kv, kv_split),
                    (u64::from(query_width + kv_width), local_kv, kv_split),
                ];
                let mut projections = Vec::with_capacity(3);
                for (index, (base, rows, split)) in windows.into_iter().enumerate() {
                    let first = base + u64::from(split.rank) * rows;
                    let weight = scope.scope(module).rows("weight", first, &[rows, hidden])?;
                    let bias = match bias {
                        QkvBias::Absent => None,
                        QkvBias::Present => {
                            Some(scope.scope(module).rows("bias", first, &[rows])?)
                        }
                        QkvBias::Windows(names) => {
                            Some(scope.rows(names[index], u64::from(split.rank) * rows, &[rows])?)
                        }
                    };
                    projections.push(Linear::new(weight, bias, WeightOrder::NxK)?);
                }
                let [query, key, value]: [Linear; 3] = projections
                    .try_into()
                    .map_err(|_| refuse("three windows of one fused module".to_owned()))?;
                Ok(Self { query, key, value })
            }
        }
    }

    /// The query projection.
    pub fn query(&self) -> &Linear {
        &self.query
    }

    /// The key projection.
    pub fn key(&self) -> &Linear {
        &self.key
    }

    /// The value projection.
    pub fn value(&self) -> &Linear {
        &self.value
    }

    /// Project `input` into the three outputs, query first.
    ///
    /// # Errors
    ///
    /// As [`Linear::forward`].
    pub fn forward(
        &self,
        f: &Forward,
        input: TensorView,
        query: TensorView,
        key: TensorView,
        value: TensorView,
    ) -> Result<()> {
        self.query.forward(f, input, query)?;
        self.key.forward(f, input, key)?;
        self.value.forward(f, input, value)
    }
}

/// Several checkpoint modules' projections read as one product: their weights stacked along the
/// output channels in the order named, their biases likewise.
///
/// One launch computes what the modules compute one after another, and the output's columns are
/// each module's in order.
pub struct MergedColumnLinear {
    linear: Linear,
}

impl MergedColumnLinear {
    /// The modules `modules` under `scope`, each `[outputs[i], inputs]`, as one projection.
    ///
    /// # Errors
    ///
    /// When `modules` and `outputs` disagree in count, or a module cannot be read.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        modules: &[&str],
        inputs: u32,
        outputs: &[u32],
        bias: Bias,
    ) -> Result<Self> {
        if modules.len() != outputs.len() || modules.is_empty() {
            return Err(Error::weight(
                scope.path(),
                format!(
                    "{} modules with {} output widths",
                    modules.len(),
                    outputs.len()
                ),
            ));
        }
        let total: u64 = outputs.iter().map(|width| u64::from(*width)).sum();
        let stems: Vec<String> = modules.iter().map(|module| scope.name(module)).collect();
        let parts = |leaf: &str| -> Vec<String> {
            stems.iter().map(|stem| format!("{stem}.{leaf}")).collect()
        };
        let merged = scope.name(&modules.join("+"));
        let weight = scope.load(
            WeightRequest::parts(
                format!("{merged}.weight"),
                parts("weight"),
                Shape::new(&[total, u64::from(inputs)]),
                WeightLayout::ROW_MAJOR,
            )
            .nonzero(),
        )?;
        let bias = match bias {
            Bias::Absent => None,
            Bias::Present => Some(scope.load(WeightRequest::parts(
                format!("{merged}.bias"),
                parts("bias"),
                Shape::new(&[total]),
                WeightLayout::ROW_MAJOR,
            ))?),
        };
        Ok(Self {
            linear: Linear::new(weight, bias, WeightOrder::NxK)?,
        })
    }

    /// The merged output width.
    pub const fn n(&self) -> u32 {
        self.linear.n()
    }

    /// `out = input · [weight₀; weight₁; …]ᵀ + [bias₀; bias₁; …]`.
    ///
    /// # Errors
    ///
    /// As [`Linear::forward`].
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        self.linear.forward(f, input, out)
    }
}

#[cfg(test)]
mod tests {
    use ops::DevicePtr;

    use super::*;
    use crate::Backing;

    fn weight(dtype: DType, shape: &[u64]) -> Weight {
        Weight {
            backing: Backing::new(DevicePtr::new(0x1_0000), 1 << 20),
            view: TensorView::new(DevicePtr::new(0x1_0000), dtype, Shape::new(shape)),
        }
    }

    /// A calibrated projection: `[n, k]` logical, so the packed weight is `[n, k/2]` bytes, the
    /// block scales carry the shape the product's layout gives them, and the activation encoder's
    /// scale is published beside them.
    fn quantized(n: u64, k: u64) -> QuantizedWeight {
        QuantizedWeight {
            packed: weight(DType::U8, &[n, k / 2]),
            scales: weight(DType::F8E4M3, PRODUCT_SCALE_LAYOUT.shape(n, k).dims()),
            global_scale: 0.125,
            layout: PRODUCT_SCALE_LAYOUT,
            input_scale: Some(weight(DType::F32, &[1])),
        }
    }

    fn packed(n: u64, k: u64) -> Linear {
        Linear::packed(&quantized(n, k), weight(DType::F32, &[1])).unwrap()
    }

    #[test]
    fn a_packed_projection_states_its_contraction_and_its_channels() {
        let projection = packed(8, 64);
        assert_eq!(projection.k(), 64);
        assert_eq!(projection.n(), 8);
        assert!(projection.is_packed());
        assert_eq!(projection.weight().dtype, DType::U8);
        assert_eq!(projection.weight().shape.dims(), &[8, 32]);
    }

    /// One forward of `projection` over `rows` activation rows of `input` on a recording forward:
    /// its answer, the programs it launched, and the arena it reserved beyond its operands.
    fn recorded_forward(
        projection: &Linear,
        rows: u64,
        input: DType,
    ) -> (Result<()>, Vec<(String, String)>, u64) {
        let arena = crate::Arena::counting();
        let f = Forward::recording(&arena);
        let class = u64::from(ops::fp4::packed_rows(rows as u32).expect("a row class"));
        let activations = arena
            .tensor(input, Shape::new(&[rows, u64::from(projection.k())]))
            .expect("a counting arena");
        let out = arena
            .tensor(DType::BF16, Shape::new(&[class, u64::from(projection.n())]))
            .expect("a counting arena");
        let out = prefix(out, &[rows, u64::from(projection.n())]);
        let operands = arena.used();
        let (answer, launches) =
            crate::forward::recorded(|| projection.forward(&f, activations, out));
        (answer, launches, arena.peak() - operands)
    }

    /// The keys a recorded forward launched for `operator`.
    fn keys_of<'a>(launches: &'a [(String, String)], operator: &str) -> Vec<&'a str> {
        launches
            .iter()
            .filter(|(name, _)| name == operator)
            .map(|(_, key)| key.as_str())
            .collect()
    }

    /// A scales plane declared as bytes is refused where the key is built, not at the launch.
    #[test]
    fn a_scales_plane_declared_as_bytes_is_refused_where_the_key_is_built() {
        let mut bytes = quantized(8, 64);
        bytes.scales = weight(DType::U8, PRODUCT_SCALE_LAYOUT.shape(8, 64).dims());
        let projection = Linear::packed(&bytes, weight(DType::F32, &[1])).unwrap();
        let (answer, ..) = recorded_forward(&projection, 16, DType::BF16);
        let text = answer
            .expect_err("a byte scales plane is not an NVFP4 operand")
            .to_string();
        assert!(
            text.contains("f8e4m3 block scale"),
            "the refusal names the pair: {text}"
        );
    }

    #[test]
    fn a_weight_only_export_cannot_be_encoded() {
        // A checkpoint that states no activation scale at all is not one an encoder may invent a
        // magnitude for.
        let mut weight_only = quantized(8, 64);
        weight_only.input_scale = None;
        let error = Linear::packed(&weight_only, weight(DType::F32, &[1]))
            .err()
            .expect("a weight-only export has no encoder scale");
        assert!(
            error
                .to_string()
                .contains("no calibrated activation input scale"),
            "{error}"
        );
        // A published scale that is not one FP32 number is not the encoder's operand.
        let mut wrong = quantized(8, 64);
        wrong.input_scale = Some(weight(DType::U8, &[1]));
        assert!(Linear::packed(&wrong, weight(DType::F32, &[1])).is_err());
    }

    #[test]
    fn a_packed_product_is_keyed_per_row_class_and_a_dense_one_is_not() {
        let packed = packed(8, 64);
        let launched = |rows: u64| {
            let (answer, launches, _) = recorded_forward(&packed, rows, DType::BF16);
            answer.expect("the recorded forward");
            launches
        };
        let (one, eight) = (launched(1), launched(8));
        // One and eight rows are the 128-row class: the same program, not two.
        assert_eq!(keys_of(&one, "nvfp4_gemm"), keys_of(&eight, "nvfp4_gemm"));
        assert_ne!(
            keys_of(&launched(128), "nvfp4_gemm"),
            keys_of(&launched(129), "nvfp4_gemm")
        );
        // The activation encoding is one module whatever the row count.
        assert_eq!(
            keys_of(&one, "nvfp4_quantize"),
            keys_of(&launched(129), "nvfp4_quantize")
        );

        // A dense projection's key takes no row count at all.
        let dense = Linear::new(weight(DType::BF16, &[8, 64]), None, WeightOrder::NxK).unwrap();
        let dense_keys = |rows: u64| {
            let arena = crate::Arena::counting();
            let f = Forward::recording(&arena);
            let input = arena
                .tensor(DType::BF16, Shape::new(&[rows, 64]))
                .expect("a counting arena");
            let out = arena
                .tensor(DType::BF16, Shape::new(&[rows, 8]))
                .expect("a counting arena");
            let (answer, launches) = crate::forward::recorded(|| dense.forward(&f, input, out));
            answer.expect("the recorded forward");
            launches
        };
        assert_eq!(dense_keys(1), dense_keys(129));
    }

    /// The encoder is keyed by the activation's own type: an FP16 stack encodes FP16, and a type
    /// NVFP4 does not encode is refused at the call rather than relabelled.
    #[test]
    fn a_packed_projection_encodes_the_activation_it_is_given() {
        let packed = packed(8, 64);
        let (answer, launches, _) = recorded_forward(&packed, 4, DType::F16);
        answer.expect("an FP16 activation encodes");
        assert!(
            keys_of(&launches, "nvfp4_quantize")
                .iter()
                .all(|key| key.contains("f16") && !key.contains("bf16")),
            "{launches:?}"
        );
        let (refused, ..) = recorded_forward(&packed, 4, DType::F32);
        assert!(refused.unwrap_err().to_string().contains("f32"));
    }

    #[test]
    fn the_projection_reserves_the_arena_its_class_writes() {
        let packed = packed(8, 64);
        let reserved = |rows: u64| {
            let (answer, _, reserved) = recorded_forward(&packed, rows, DType::BF16);
            answer.expect("the recorded forward");
            reserved
        };
        assert_eq!(reserved(15), reserved(128));
        assert_eq!(reserved(129), reserved(256));
        assert!(reserved(15) > 15 * 32 + 15 * 4);
    }

    #[test]
    fn a_packed_projection_refuses_what_it_cannot_contract() {
        let alpha = weight(DType::F32, &[1]);
        // A contraction that is not whole sixteen-element blocks has no encoder.
        assert!(Linear::packed(&quantized(8, 62), alpha).is_err());
        // A weight that is not packed bytes is a BF16 matrix, which this product does not read.
        let mut unpacked = quantized(8, 64);
        unpacked.packed = weight(DType::BF16, &[8, 32]);
        assert!(Linear::packed(&unpacked, alpha).is_err());
        // The weight's scales answer the product's layout.
        let mut linear = quantized(8, 64);
        linear.layout = ScaleLayout::Linear;
        linear.scales = weight(DType::F8E4M3, ScaleLayout::Linear.shape(8, 64).dims());
        assert!(Linear::packed(&linear, alpha).is_err());
        // The scales answer the layout's own shape, so a matrix of them is not one.
        let mut unscaled = quantized(8, 64);
        unscaled.scales = weight(DType::U8, &[8, 4]);
        assert!(Linear::packed(&unscaled, alpha).is_err());
        // The alpha is one FP32 number.
        assert!(Linear::packed(&quantized(8, 64), weight(DType::F32, &[2])).is_err());
    }

    /// The encoder's tile divides the row it encodes, and it is the largest power of two that does.
    #[test]
    fn the_encode_block_divides_the_row_it_encodes() {
        for (columns, requested, expected) in [
            (4096u32, 1024u32, 1024u32),
            (2048, 1024, 1024),
            (768, 1024, 256),
            (768, 512, 256),
            (768, 256, 256),
            (96, 1024, 32),
            (64, 1024, 64),
        ] {
            let block = encode_block(columns, requested);
            assert_eq!(block, expected, "columns {columns}, requested {requested}");
            assert!(columns.is_multiple_of(block));
        }
    }

    /// A row-parallel projection's partial sums are reduced over the group, and a forward with no
    /// group refuses rather than answering one rank's share.
    #[test]
    fn a_row_parallel_projection_reduces_or_refuses() {
        let mut row = Linear::new(weight(DType::BF16, &[8, 32]), None, WeightOrder::NxK).unwrap();
        row.reduce = 2;
        let arena = crate::Arena::counting();
        let input = arena.tensor(DType::BF16, Shape::new(&[4, 32])).unwrap();
        let out = arena.tensor(DType::BF16, Shape::new(&[4, 8])).unwrap();
        let alone = Forward::recording(&arena);
        let refused = row.forward(&alone, input, out).unwrap_err().to_string();
        assert!(refused.contains("no group"), "{refused}");
        let grouped = Forward::recording(&arena).with_split(1, 2).unwrap();
        row.forward(&grouped, input, out)
            .expect("a split forward reduces");
    }
}

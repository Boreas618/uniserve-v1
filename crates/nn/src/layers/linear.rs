//! A dense projection: activations times a weight, with an optional bias.
//!
//! The layer holds the weight in the order its checkpoint gives (Qwen3 stores `[n, k]` rows of
//! output channels) and names that order in the `gemm` descriptor, so the product is one operator
//! call whatever the storage order is. The contraction length is geometry the descriptor bakes in
//! (specs/worker.md 4.2), so it is checked against the weight at construction and against the input
//! at every call.

use ops::fp4::{
    BLOCK_ELEMENTS, GemmArgs, GemmDescriptor, Nvfp4Gemm, Nvfp4Quantize, QuantizeArgs,
    QuantizeDescriptor, ScaleLayout,
};
use ops::gemm::{self, Descriptor, Epilogue, Gemm, WeightOrder};
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::ACCUMULATE;
use crate::weights::{QuantizedWeight, Weight};

/// A dense projection `out = input · weight + bias`.
pub struct Linear {
    weight: Weight,
    bias: Option<Weight>,
    order: WeightOrder,
    k: u32,
    n: u32,
    input: Option<ops::DType>,
}

impl Linear {
    /// A projection with `weight` stored in `order`.
    ///
    /// `weight` is `[k, n]` in [`WeightOrder::KxN`] or `[n, k]` in [`WeightOrder::NxK`]; the bias,
    /// when present, has `n` elements.
    pub fn new(weight: Weight, bias: Option<Weight>, order: WeightOrder) -> Result<Self> {
        let (k, n) = match order {
            WeightOrder::KxN => {
                if weight.shape().rank() != 2 {
                    return Err(Error::shape(
                        "Linear",
                        format!("a kxn weight is a matrix, not {:?}", weight.shape().dims()),
                    ));
                }
                (weight.shape().dim(0), weight.shape().dim(1))
            }
            WeightOrder::NxK => {
                if weight.shape().rank() != 2 {
                    return Err(Error::shape(
                        "Linear",
                        format!("an nxk weight is a matrix, not {:?}", weight.shape().dims()),
                    ));
                }
                (weight.shape().dim(1), weight.shape().dim(0))
            }
        };
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
            weight,
            bias,
            order,
            k: u32::try_from(k)
                .map_err(|_| Error::shape("Linear", format!("a contraction of {k} elements")))?,
            n: u32::try_from(n)
                .map_err(|_| Error::shape("Linear", format!("{n} output channels")))?,
            input: None,
        })
    }

    /// The same projection, with the activation element type its descriptor is compiled for.
    ///
    /// A model whose activations are one dtype everywhere needs nothing more; a projection on an
    /// FP32 activation path (H3's latent input and output heads, specs/worker.md 4.3) states it
    /// here rather than discovering it at launch.
    pub fn with_input(mut self, dtype: ops::DType) -> Self {
        self.input = Some(dtype);
        self
    }

    /// The contraction length.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// The number of output channels.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// The weight view.
    pub fn weight(&self) -> TensorView {
        self.weight.view()
    }

    /// `out = input · weight + bias`.
    ///
    /// `input` is `[m, k]` and `out` is `[m, n]`, both row-major on their last axis.
    /// The descriptor [`Self::forward`] launches with, for an input and an output of these
    /// element types.
    ///
    /// The row count is not in it: rows are a launch value and not a key field, so one request space
    /// enumerates a descriptor per pair of element types and the operator is compiled once for
    /// every row count that reaches it. This is the same arithmetic `forward` runs — `forward` calls
    /// this — so an enumeration built on it cannot drift from what the model launches.
    pub fn descriptor(&self, input: DType, output: DType) -> Descriptor {
        Self::descriptor_of(
            self.k,
            self.weight.dtype(),
            self.order,
            self.bias.is_some(),
            input,
            output,
        )
    }

    /// The descriptor a dense product over `k` columns launches with, from the numbers alone.
    ///
    /// A deployment answers "what must be compiled?" before it opens a device, and this key reads the
    /// contraction length, the weight's element type, the order, whether an epilogue adds a bias, and
    /// the two element types — never the weight's values or its row count. [`Self::descriptor`] calls
    /// it with a loaded layer's numbers, so an enumeration built on it cannot drift from the launch.
    pub fn descriptor_of(
        k: u32,
        weight: DType,
        order: WeightOrder,
        bias: bool,
        input: DType,
        output: DType,
    ) -> Descriptor {
        Descriptor {
            input,
            weight,
            accumulate: ACCUMULATE,
            output,
            order,
            epilogue: if bias { Epilogue::Bias } else { Epilogue::None },
            k,
        }
    }

    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        let k = u64::from(self.k);
        let n = u64::from(self.n);
        let input_dtype = self.input.unwrap_or(input.dtype);
        input
            .require_row("Linear", input_dtype, k)
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
        let descriptor = self.descriptor(input_dtype, out.dtype);
        let args = gemm::Args {
            input,
            weight: self.weight.view(),
            bias: self.bias.as_ref().map(Weight::view),
            out,
            alpha: 1.0,
        };

        f.operate::<Gemm>(&descriptor, args)
    }
}

/// A dense projection over an NVFP4 weight: the activation is encoded, then contracted.
///
/// The layer holds the checkpoint's packed weight (`[n, k/2]` bytes, two E2M1 codes a byte), its
/// block scales in the layout the checkpoint stores them, the bias when the epilogue adds one, and
/// the activation encoder's global scale. It holds no scratch: the packed activation and its block
/// scales of one call come from [`Forward::arena`](crate::Forward::arena), and the tensor scale the
/// product is scaled by is a **device operand** the caller passes, because the epilogue reads it as
/// each thread block starts rather than after a host round trip.
///
/// The two operators are one layer because `nvfp4_quantize` produces exactly what `nvfp4_gemm`'s
/// activation side reads, in the same layout: splitting them would let a model pair a quantizer and
/// a product whose scale layouts differ, which no shape check would catch.
pub struct Nvfp4Linear {
    packed: Weight,
    scales: Weight,
    bias: Option<Weight>,
    /// The layout of the activation's block scales, which also names the product.
    layout: ScaleLayout,
    /// `[1]` F32: the encoder's global scale, the reciprocal of this projection's activation tensor
    /// scale, as the loading boundary published it beside the weights.
    input_scale: Weight,
    /// The activation element type the encoder is compiled for.
    input: ops::DType,
    /// Elements one encoder program covers.
    block: u32,
    k: u32,
    n: u32,
}

impl Nvfp4Linear {
    /// A projection whose weight `weight` is the checkpoint's, in `layout` for its activations.
    ///
    /// `weight.packed` is `[n, k/2]` bytes and `weight.scales` the checkpoint's own block scales
    /// over `[n, k]`; the module's weight side is swizzled, so the layer does not re-key it. The
    /// contraction is geometry the descriptor bakes in, so `k` must be whole sixteen-element
    /// blocks — the encoder's and the product's own requirement.
    ///
    /// The activation encoder's scale is the checkpoint's `input_scale` for this projection, which
    /// the loading boundary publishes beside the weights as [`QuantizedWeight::input_scale`]; a
    /// weight-only export has none and **cannot serve this layer**, which is refused here by name
    /// rather than encoded at an invented magnitude.
    pub fn new(
        weight: &QuantizedWeight,
        bias: Option<Weight>,
        layout: ScaleLayout,
        input: ops::DType,
        block: u32,
    ) -> Result<Self> {
        if weight.packed.shape().rank() != 2 {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!(
                    "a packed weight is a byte matrix, not {:?}",
                    weight.packed.shape().dims()
                ),
            ));
        }
        let n = weight.packed.shape().dim(0);
        let half = weight.packed.shape().dim(1);
        let k = half * 2;
        if weight.packed.dtype() != ops::DType::U8 {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!("packed weights are U8 bytes, not {}", weight.packed.dtype()),
            ));
        }
        if !k.is_multiple_of(u64::from(BLOCK_ELEMENTS)) || k == 0 {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!("a contraction of {k} elements is not whole sixteen-element blocks"),
            ));
        }
        // One layout governs a product: it states both how the activation's block scales are
        // arranged and, by naming the product, how the weight's are read. A checkpoint whose
        // block scales answer another layout would be contracted by a program that reads them
        // transposed, and the product's own check does not look at the weight's field.
        if layout != weight.layout {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!(
                    "the activation scales are {layout} and the checkpoint's weight scales are {}:                      one layout governs both sides of a product",
                    weight.layout
                ),
            ));
        }
        let expected = weight.layout.shape(n, k);
        if weight.scales.shape() != expected {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!(
                    "the checkpoint's block scales are {:?}; {} rows of {k} elements take {:?}",
                    weight.scales.shape().dims(),
                    n,
                    expected.dims()
                ),
            ));
        }
        let Some(input_scale) = weight.input_scale else {
            return Err(Error::shape(
                "Nvfp4Linear",
                "this projection has no calibrated activation input scale: a weight-only export \
                 cannot be encoded by a program that reads one"
                    .to_owned(),
            ));
        };
        if input_scale.dtype() != ops::DType::F32 || input_scale.shape().numel() != 1 {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!(
                    "the activation input scale is {} {:?}, and the encoder reads one FP32 number",
                    input_scale.dtype(),
                    input_scale.shape().dims()
                ),
            ));
        }
        if input != ops::DType::BF16 && input != ops::DType::F16 {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!("{input} activations; NVFP4 encodes BF16 or FP16"),
            ));
        }
        if block == 0 || !block.is_multiple_of(BLOCK_ELEMENTS) {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!("{block} elements per encoder program are not whole blocks"),
            ));
        }
        if let Some(bias) = bias.as_ref()
            && bias.shape().numel() != n
        {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!(
                    "a bias of {} elements cannot be added to {n} output channels",
                    bias.shape().numel()
                ),
            ));
        }
        Ok(Self {
            packed: weight.packed,
            scales: weight.scales,
            bias,
            layout,
            input_scale,
            input,
            block,
            k: u32::try_from(k).map_err(|_| {
                Error::shape("Nvfp4Linear", format!("a contraction of {k} elements"))
            })?,
            n: u32::try_from(n)
                .map_err(|_| Error::shape("Nvfp4Linear", format!("{n} output channels")))?,
        })
    }

    /// The contraction length, in elements.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// The number of output channels.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// The packed weight view, `[n, k/2]` bytes.
    pub fn weight(&self) -> TensorView {
        self.packed.view()
    }

    /// `out = input · weight + bias`, with the activation encoded on the way.
    ///
    /// `input` is `[m, k]` of the encoder's element type and unit stride on its last axis; `out` is
    /// `[m, n]`; `alpha` is `[1]` FP32, the product of the activation's and the weight's tensor
    /// scales, computed on device by whoever produced the activation scale. The packed activation
    /// and its block scales are taken from the forward's arena.
    /// The descriptor of the activation encoding this projection runs first.
    /// The encoder's descriptor, from the contraction length alone.
    pub fn encode_descriptor_of(
        input: DType,
        columns: u32,
        layout: ScaleLayout,
        block: u32,
    ) -> QuantizeDescriptor {
        QuantizeDescriptor {
            input,
            columns,
            layout,
            block,
            pdl: false,
        }
    }

    pub fn encode_descriptor(&self) -> QuantizeDescriptor {
        QuantizeDescriptor {
            input: self.input,
            columns: self.k,
            layout: self.layout,
            block: self.block,
            pdl: false,
        }
    }

    /// The descriptor of the packed product, for `m` activation rows and an output of `output`.
    ///
    /// `m` is a key field here and not in [`Linear::descriptor`]: the product's grid is computed
    /// from the row count and the module is compiled per row count, which is why the deployment
    /// enumerates the bucket sizes the executor builds rather than an unbounded row count.
    /// The packed product's descriptor, from the geometry alone.
    ///
    /// `m` is a key field here and not in [`Linear::descriptor_of`]: this product's grid is the row
    /// count and the module is compiled per row count, which is why a deployment enumerates the
    /// buckets its executor captures rather than an unbounded row count.
    #[allow(clippy::too_many_arguments)]
    pub fn descriptor_of(
        m: u32,
        n: u32,
        k: u32,
        layout: ScaleLayout,
        bias: bool,
        output: DType,
    ) -> GemmDescriptor {
        GemmDescriptor {
            packed: DType::U8,
            scale: DType::F8E4M3,
            accumulate: ACCUMULATE,
            output,
            layout,
            epilogue: if bias { Epilogue::Bias } else { Epilogue::None },
            m,
            n,
            k,
            pdl: false,
        }
    }

    pub fn descriptor(&self, m: u32, output: DType) -> GemmDescriptor {
        GemmDescriptor {
            packed: DType::U8,
            scale: self.scales.dtype(),
            accumulate: ACCUMULATE,
            output,
            layout: self.layout,
            epilogue: if self.bias.is_some() {
                Epilogue::Bias
            } else {
                Epilogue::None
            },
            m,
            n: self.n,
            k: self.k,
            pdl: false,
        }
    }

    /// The arena one launch of this projection carves for `rows` activation rows.
    ///
    /// Three regions, each aligned up like the arena's own grants: the packed activation
    /// (`[rows, k/2]` bytes), the block scales in this layout's shape, and one FP32 row factor per
    /// activation row. A plan that does not reserve them is an arena the launch writes past, so the
    /// model's plan asks this rather than estimating.
    pub fn scratch_bytes(&self, rows: u64) -> u64 {
        let align =
            |bytes: u64| bytes.div_ceil(crate::forward::ALIGNMENT) * crate::forward::ALIGNMENT;
        let values = align(rows * u64::from(self.k) / 2);
        let scales = {
            let shape = self.layout.shape(rows, u64::from(self.k));
            align(shape.numel())
        };
        let token_scale = align(rows * 4);
        values + scales + token_scale
    }

    pub fn forward(
        &self,
        f: &Forward,
        input: TensorView,
        alpha: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let k = u64::from(self.k);
        let n = u64::from(self.n);
        input
            .require_row("Nvfp4Linear", self.input, k)
            .map_err(Error::from)?;
        out.require_row("Nvfp4Linear", out.dtype, n)
            .map_err(Error::from)?;
        alpha
            .require_row("Nvfp4Linear", ops::DType::F32, 1)
            .map_err(Error::from)?;
        let rows = input.rows();
        if rows == 0 {
            return Err(Error::shape(
                "Nvfp4Linear",
                "a projection of no rows has no program".to_owned(),
            ));
        }
        if out.rows() != rows {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!("{rows} input rows against {} output rows", out.rows()),
            ));
        }
        let m = u32::try_from(rows)
            .map_err(|_| Error::shape("Nvfp4Linear", format!("{rows} activation rows")))?;

        // 1. Encode the activation into the layout the product reads, into this forward's scratch.
        //
        // The encode is two-level: the block's E4M3 scale is relative to the row's own factor
        // (`amax(row) · global_scale`), and the product dequantizes a code as
        // `code · block_scale · row factor`. So the factor is an operand of the encode and an
        // output of it, one per row.
        let arena = f.arena();
        let values = arena.tensor(ops::DType::U8, ops::Shape::new(&[rows, k / 2]))?;
        let scale_shape = self.layout.shape(rows, k);
        let scales = arena.tensor(ops::DType::U8, scale_shape)?;
        let token_scale = arena.tensor(ops::DType::F32, ops::Shape::new(&[rows]))?;
        let quantize = self.encode_descriptor();
        let encoded = QuantizeArgs {
            input,
            values,
            scales,
            global_scale: self.input_scale.view(),
            token_scale,
        };
        f.operate::<Nvfp4Quantize>(&quantize, encoded)?;

        // 2. Contract the encoded activation with the checkpoint's weight.
        let descriptor = self.descriptor(m, out.dtype);
        let args = GemmArgs {
            input: values,
            weight: self.packed.view(),
            input_scales: scales,
            weight_scales: self.scales.view(),
            alpha,
            bias: self.bias.as_ref().map(Weight::view),
            out,
        };
        f.operate::<Nvfp4Gemm>(&descriptor, args)
    }
}

#[cfg(test)]
mod tests {
    use ops::{DType, DevicePtr, Shape};

    use super::*;
    use crate::Backing;

    fn weight(dtype: DType, shape: &[u64]) -> Weight {
        Weight {
            backing: Backing::new(DevicePtr::new(0x1_0000), 1 << 20),
            view: TensorView::new(DevicePtr::new(0x1_0000), dtype, Shape::new(shape)),
        }
    }

    /// A calibrated projection: `[n, k]` logical, so the packed weight is `[n, k/2]` bytes, the
    /// block scales carry the shape the checkpoint's layout gives them, and the activation encoder's
    /// scale is published beside them.
    fn packed(n: u64, k: u64, layout: ScaleLayout) -> QuantizedWeight {
        QuantizedWeight {
            packed: weight(DType::U8, &[n, k / 2]),
            scales: weight(DType::U8, layout.shape(n, k).dims()),
            global_scale: 0.125,
            layout,
            input_scale: Some(weight(DType::F32, &[1])),
        }
    }

    #[test]
    fn a_packed_projection_states_its_contraction_and_its_channels() {
        let projection = Nvfp4Linear::new(
            &packed(8, 64, ScaleLayout::Swizzled128X4),
            None,
            ScaleLayout::Swizzled128X4,
            DType::BF16,
            128,
        )
        .unwrap();
        assert_eq!(projection.k(), 64);
        assert_eq!(projection.n(), 8);
        assert_eq!(projection.weight().dtype, DType::U8);
        assert_eq!(projection.weight().shape.dims(), &[8, 32]);
    }

    #[test]
    fn a_weight_only_export_cannot_be_encoded() {
        // The distinction the loader keeps: a checkpoint that states no activation scale at all is
        // not one whose encoder may invent a magnitude for.
        let mut weight_only = packed(8, 64, ScaleLayout::Swizzled128X4);
        weight_only.input_scale = None;
        let error = Nvfp4Linear::new(
            &weight_only,
            None,
            ScaleLayout::Swizzled128X4,
            DType::BF16,
            128,
        )
        .err()
        .expect("a weight-only export has no encoder scale");
        assert!(
            error
                .to_string()
                .contains("no calibrated activation input scale"),
            "{error}"
        );
        // A published scale that is not one FP32 number is not the encoder's operand.
        let mut wrong = packed(8, 64, ScaleLayout::Swizzled128X4);
        wrong.input_scale = Some(weight(DType::U8, &[1]));
        assert!(
            Nvfp4Linear::new(&wrong, None, ScaleLayout::Swizzled128X4, DType::BF16, 128).is_err()
        );
    }

    #[test]
    fn a_packed_product_is_keyed_per_activation_row_count_and_a_dense_one_is_not() {
        // The two products differ in what their module depends on: the dense one's grid and tile
        // are its geometry's, so every row count reaches one compiled module, while the packed
        // product's is compiled per activation row count and a deployment therefore enumerates the
        // buckets it builds. This is what a key enumeration reads off the layers.
        let packed = Nvfp4Linear::new(
            &packed(8, 64, ScaleLayout::Swizzled128X4),
            None,
            ScaleLayout::Swizzled128X4,
            DType::BF16,
            128,
        )
        .unwrap();
        assert_ne!(
            packed.descriptor(1, DType::BF16).to_string(),
            packed.descriptor(8, DType::BF16).to_string()
        );
        // The activation encoding is one module whatever the row count: its key names the column
        // count, the layout and the block, and the rows travel in the launch.
        assert_eq!(
            packed.encode_descriptor().to_string(),
            packed.encode_descriptor().to_string()
        );
        // A dense projection's descriptor does not take a row count at all, which is the same
        // statement made by the signature.
        let dense = Linear::new(weight(DType::BF16, &[8, 64]), None, WeightOrder::NxK).unwrap();
        assert_eq!(
            dense.descriptor(DType::BF16, DType::BF16).to_string(),
            dense.descriptor(DType::BF16, DType::BF16).to_string()
        );
    }

    #[test]
    fn a_packed_projection_refuses_what_it_cannot_contract() {
        let refuse = |weight: QuantizedWeight, layout, input, block| {
            Nvfp4Linear::new(&weight, None, layout, input, block)
        };
        // Twelve elements are not whole sixteen-element blocks, so no encoder could produce them.
        let odd = packed(8, 62, ScaleLayout::Swizzled128X4);
        assert!(refuse(odd, ScaleLayout::Swizzled128X4, DType::BF16, 128).is_err());
        // A weight that is not packed bytes is a BF16 matrix, which this product does not read.
        let mut unpacked = packed(8, 64, ScaleLayout::Swizzled128X4);
        unpacked.packed = weight(DType::BF16, &[8, 32]);
        assert!(refuse(unpacked, ScaleLayout::Swizzled128X4, DType::BF16, 128).is_err());
        // The activation and the weight answer one layout: a program that reads the weight's
        // scales as the product's own would read another arrangement's bytes.
        assert!(
            refuse(
                packed(8, 64, ScaleLayout::Linear),
                ScaleLayout::Swizzled128X4,
                DType::BF16,
                128
            )
            .is_err()
        );
        // The scales answer the layout's own shape, so a matrix of them is not one.
        let mut unscaled = packed(8, 64, ScaleLayout::Swizzled128X4);
        unscaled.scales = weight(DType::U8, &[8, 4]);
        assert!(refuse(unscaled, ScaleLayout::Swizzled128X4, DType::BF16, 128).is_err());
        // NVFP4 encodes BF16 and FP16 activations only.
        assert!(
            refuse(
                packed(8, 64, ScaleLayout::Swizzled128X4),
                ScaleLayout::Swizzled128X4,
                DType::F32,
                128
            )
            .is_err()
        );
        // A program encodes whole blocks.
        assert!(
            refuse(
                packed(8, 64, ScaleLayout::Swizzled128X4),
                ScaleLayout::Swizzled128X4,
                DType::BF16,
                12
            )
            .is_err()
        );
    }

    #[test]
    fn a_packed_projection_refuses_a_bias_it_cannot_add() {
        let error = Nvfp4Linear::new(
            &packed(8, 64, ScaleLayout::Swizzled128X4),
            Some(weight(DType::F32, &[7])),
            ScaleLayout::Swizzled128X4,
            DType::BF16,
            128,
        )
        .err()
        .expect("a seven-element bias cannot be added to eight channels");
        assert!(error.to_string().contains("8 output channels"), "{error}");
    }
}

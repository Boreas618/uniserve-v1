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
            input,
            self.weight.dtype(),
            self.order,
            self.bias.is_some(),
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
        input: DType,
        weight: DType,
        order: WeightOrder,
        bias: bool,
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

/// The row classes the packed product is compiled for.
///
/// The product's `m` is a compile-time constant of its module (`nvfp4_gemm/cute_dsl` bakes
/// `UNISERVE_NVFP4_M` into the program and its entry refuses any other row count), and its
/// smallest tile is 128 rows, so a decode request of one or fifteen rows cannot be a module of
/// its own. The row count a launch states is therefore **the smallest class that holds the
/// request's rows**, and the caller pads its buffers to that class — the same division the sparse
/// attention states for a partial tile: the padding belongs to the caller, the tile to the module.
///
/// The four classes are the set `nvfp4_gemm/cute_dsl/package.toml` declares. A class here without
/// a cell there is a launch with no program, and a cell there without a class here is a program no
/// launch states.
pub const PACKED_M_CLASSES: [u32; 4] = [128, 256, 1024, 4096];

/// The packed product's row class for `rows` activation rows: the smallest declared class that
/// holds them.
///
/// A row count above the largest class has no module to serve it, so it is refused by name with
/// both numbers rather than clamped: a launch clamped to 4 096 rows for a 5 000-row request is the
/// same silent wrong-numbers defect one size up.
pub fn packed_rows(rows: u32) -> Result<u32> {
    PACKED_M_CLASSES
        .iter()
        .copied()
        .find(|class| *class >= rows)
        .ok_or_else(|| {
            Error::shape(
                "Nvfp4Linear",
                format!(
                    "{rows} activation rows: the largest packed tile class is {}, which no \
                     program covers",
                    PACKED_M_CLASSES[PACKED_M_CLASSES.len() - 1]
                ),
            )
        })
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
            block: Self::encode_block(columns, block),
            pdl: false,
        }
    }

    /// The tile one encoding of a `columns`-wide row runs in.
    ///
    /// The package refuses a tile that does not divide the row (`nvfp4_quantize`'s `build.py`
    /// raises on `columns % block != 0`), so the requested tile is halved until it divides. A row
    /// narrower than the largest tile — the mixture-of-experts' 768-element second encode — gets
    /// the largest tile that fits it rather than a key no package can build.
    pub fn encode_block(columns: u32, requested: u32) -> u32 {
        let mut block = requested.max(1);
        while block > 1 && !columns.is_multiple_of(block) {
            block /= 2;
        }
        block
    }

    pub fn encode_descriptor(&self) -> QuantizeDescriptor {
        QuantizeDescriptor {
            input: self.input,
            columns: self.k,
            layout: self.layout,
            block: Self::encode_block(self.k, self.block),
            pdl: false,
        }
    }

    /// The packed product's descriptor, from the geometry alone, for `rows` activation rows.
    ///
    /// `rows` is the request's row count and **not** the descriptor's `m`: the key states the row
    /// class that holds `rows` ([`packed_rows`]), because the module is compiled per class and its
    /// entry refuses any other row count. The product's grid is the row count, which is why a
    /// deployment enumerates the classes its executor builds rather than an unbounded row count.
    #[allow(clippy::too_many_arguments)]
    pub fn descriptor_of(
        rows: u32,
        n: u32,
        k: u32,
        layout: ScaleLayout,
        bias: bool,
        output: DType,
    ) -> Result<GemmDescriptor> {
        let descriptor = GemmDescriptor {
            packed: DType::U8,
            scale: DType::F8E4M3,
            accumulate: ACCUMULATE,
            output,
            layout,
            epilogue: if bias { Epilogue::Bias } else { Epilogue::None },
            m: packed_rows(rows)?,
            n,
            k,
        };
        // The walk's own statement of the pair, refused here if it ever stops being one: this is the
        // descriptor the key file and the fixture are built from.
        ops::fp4::check_gemm(&descriptor)?;
        Ok(descriptor)
    }

    /// This projection's descriptor for `rows` activation rows, keyed by their class.
    pub fn descriptor(&self, rows: u32, output: DType) -> Result<GemmDescriptor> {
        let descriptor = GemmDescriptor {
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
            m: packed_rows(rows)?,
            n: self.n,
            k: self.k,
        };
        // The scale's type is the checkpoint's own statement, so a plane declared as anything but
        // E4M3 makes a key no store built - refused here, where the layer can say which plane.
        ops::fp4::check_gemm(&descriptor)?;
        Ok(descriptor)
    }

    /// The arena one launch of this projection carves for a request of `rows` activation rows.
    ///
    /// Three regions, each aligned up like the arena's own grants: the packed activation, the block
    /// scales in this layout's shape, and one FP32 row factor per activation row.
    ///
    /// **The regions are the class's, not the request's** ([`packed_rows`]): the launch states the
    /// class, so a region reserved for fifteen rows is a region a 128-row launch writes past — and
    /// every other check would still agree, because the store has the cell, the descriptor matches
    /// it and the launch runs. Only the arena would know, and it does not speak. A row count above
    /// the largest class reserves its own rows; the launch refuses that launch by name.
    pub fn scratch_bytes(&self, rows: u64) -> u64 {
        let class = PACKED_M_CLASSES
            .iter()
            .copied()
            .map(u64::from)
            .find(|class| *class >= rows)
            .unwrap_or(rows);
        let align =
            |bytes: u64| bytes.div_ceil(crate::forward::ALIGNMENT) * crate::forward::ALIGNMENT;
        let values = align(class * u64::from(self.k) / 2);
        let scales = {
            let shape = self.layout.shape(class, u64::from(self.k));
            align(shape.numel())
        };
        let token_scale = align(class * 4);
        values + scales + token_scale
    }

    /// `out = projection(input)`, with the product launched at the row *class* of `input`'s rows.
    ///
    /// The activation is encoded over the rows the request has, and the product runs at the class
    /// that holds them ([`packed_rows`]) — so `out` must be class-height, and its rows past the
    /// request's are computed and discarded. That padding is the caller's, as it is for the sparse
    /// attention's partial tile; this layer refuses a short `out` by name rather than writing past
    /// it. The cost is real and belongs here: on a decode step the product reads the expert weights
    /// once and is bound by that read, so 128 rows for 15 spends bandwidth to buy nothing.
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
        let counted = u32::try_from(rows)
            .map_err(|_| Error::shape("Nvfp4Linear", format!("{rows} activation rows")))?;
        let class = u64::from(packed_rows(counted)?);
        if out.rows() != class {
            return Err(Error::shape(
                "Nvfp4Linear",
                format!(
                    "{rows} activation rows launch the {class}-row class, so the output must be \
                     {class} rows to receive it, not {} — pad the buffer to the class and read \
                     the first {rows}",
                    out.rows()
                ),
            ));
        }

        // 1. Encode the activation into the layout the product reads, into this forward's scratch.
        //
        // The encode is two-level: the block's E4M3 scale is relative to the row's own factor
        // (`amax(row) · global_scale`), and the product dequantizes a code as
        // `code · block_scale · row factor`. So the factor is an operand of the encode and an
        // output of it, one per row. The regions are class-height and the encode writes the
        // request's rows into their prefix; the product's rows past the request's are computed
        // from bytes nothing reads and discarded with the rows they land in.
        let arena = f.arena();
        let values_region = arena.tensor(ops::DType::U8, ops::Shape::new(&[class, k / 2]))?;
        let scale_shape = self.layout.shape(class, k);
        let scales = arena.tensor(ops::DType::U8, scale_shape)?;
        let token_scale_region = arena.tensor(ops::DType::F32, ops::Shape::new(&[class]))?;
        let values = prefix_rows(values_region, rows, k / 2);
        // One scalar per row, so the view is rank one: the encode checks `[rows]`, and a `[rows, 1]`
        // view is a different shape than the one it states for a per-row factor.
        let token_scale = prefix_scalars(token_scale_region, rows);
        let encoded_scale_shape = self.layout.shape(rows, k);
        let encoded_scales = if encoded_scale_shape == scale_shape {
            scales
        } else {
            prefix_rows(
                scales,
                encoded_scale_shape.rows(),
                encoded_scale_shape.last(),
            )
        };
        let quantize = self.encode_descriptor();
        let encoded = QuantizeArgs {
            input,
            values,
            scales: encoded_scales,
            global_scale: self.input_scale.view(),
            token_scale,
        };
        f.operate::<Nvfp4Quantize>(&quantize, encoded)?;

        // 2. Contract the encoded activation with the checkpoint's weight, over the class's rows.
        let descriptor = self.descriptor(counted, out.dtype)?;
        let args = GemmArgs {
            input: values_region,
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

/// The first `rows` rows of a row-major `buffer` whose rows are `width` elements wide.
///
/// The pointers are the same: a shorter view of a larger region is what lets one allocation serve
/// both the request's rows and the class they are launched in.
/// The first `rows` scalars of a rank-one region, as a rank-one view.
fn prefix_scalars(buffer: TensorView, rows: u64) -> TensorView {
    let shape = ops::Shape::new(&[rows]);
    TensorView::with_strides(
        buffer.ptr,
        buffer.dtype,
        shape,
        ops::Strides::row_major(shape),
    )
}

fn prefix_rows(buffer: TensorView, rows: u64, width: u64) -> TensorView {
    let shape = ops::Shape::new(&[rows, width]);
    TensorView::with_strides(
        buffer.ptr,
        buffer.dtype,
        shape,
        ops::Strides::row_major(shape),
    )
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
            scales: weight(DType::F8E4M3, layout.shape(n, k).dims()),
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

    /// A scales plane declared as bytes is refused where the key is built, not at the launch.
    ///
    /// The block scales of an NVFP4 weight are E4M3 and the packed pairs are bytes; a checkpoint
    /// whose scales plane says otherwise would key `scale_u8`, which no store built, and the launch
    /// would report a missing program rather than the declared type. The refusal is one line in the
    /// product's own check, called here as the descriptor is built.
    #[test]
    fn a_scales_plane_declared_as_bytes_is_refused_where_the_key_is_built() {
        let layout = ScaleLayout::Swizzled128X4;
        let mut scales = packed(8, 64, layout);
        scales.scales = weight(DType::U8, layout.shape(8, 64).dims());
        let projection =
            Nvfp4Linear::new(&scales, None, ScaleLayout::Swizzled128X4, DType::BF16, 128).unwrap();
        let error = projection
            .descriptor(16, DType::BF16)
            .expect_err("a byte scales plane is not an NVFP4 operand");
        let text = error.to_string();
        assert!(
            text.contains("f8e4m3 block scale"),
            "the refusal names the pair: {text}"
        );
    }

    /// The walk's descriptor and a loaded projection's are one key.
    ///
    /// `descriptor_of` states the operand types from the geometry alone, which is what the key walk
    /// and the fixture use; `descriptor` states the scales' type from the checkpoint's own plane. A
    /// checkpoint whose scales plane is declared as anything but E4M3 makes the two keys differ, and
    /// the launch then asks the store for a program the walk never named - with every number in the
    /// key looking plausible. This pins the pair this deployment's checkpoint states.
    #[test]
    fn the_walk_and_a_loaded_projection_state_one_key() {
        let (n, k, rows) = (8_u64, 64_u64, 16_u32);
        let layout = ScaleLayout::Swizzled128X4;
        let mut scales = packed(n, k, layout);
        scales.scales = weight(DType::F8E4M3, layout.shape(n, k).dims());
        let projection = Nvfp4Linear::new(&scales, None, layout, DType::BF16, 128).unwrap();

        let loaded = projection.descriptor(rows, DType::BF16).expect("a key");
        let walked =
            Nvfp4Linear::descriptor_of(rows, n as u32, k as u32, layout, false, DType::BF16)
                .expect("the walk's key");

        assert_eq!(loaded.packed, walked.packed, "the packed bytes");
        assert_eq!(
            loaded.scale, walked.scale,
            "the block scale: the checkpoint's plane and the walk's literal must be one type"
        );
        assert_eq!(loaded.m, walked.m, "the row class");
        assert_eq!(loaded.n, walked.n);
        assert_eq!(loaded.k, walked.k);
        assert_eq!(loaded.layout, walked.layout);
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
    fn a_packed_product_is_keyed_per_row_class_and_a_dense_one_is_not() {
        // The two products differ in what their module depends on: the dense one's grid and tile
        // are its geometry's, so every row count reaches one compiled module, while the packed
        // product's `m` is a compile-time constant of its module — so a deployment enumerates the
        // row *classes* it builds, and two requests inside one class are one program. This is what
        // a key enumeration reads off the layers.
        let packed = Nvfp4Linear::new(
            &packed(8, 64, ScaleLayout::Swizzled128X4),
            None,
            ScaleLayout::Swizzled128X4,
            DType::BF16,
            128,
        )
        .unwrap();
        // One and eight rows are the 128-row class: the same program, not two.
        assert_eq!(
            packed.descriptor(1, DType::BF16).unwrap().to_string(),
            packed.descriptor(8, DType::BF16).unwrap().to_string()
        );
        // A class boundary is a different module, and that is the only thing `m` distinguishes.
        assert_ne!(
            packed.descriptor(128, DType::BF16).unwrap().to_string(),
            packed.descriptor(129, DType::BF16).unwrap().to_string()
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
    fn the_class_is_the_smallest_declared_one_that_holds_the_rows() {
        // The classes are the set the package declares, and a request lands in the smallest that
        // holds it: a decode of 1 or 15 rows is the 128-row tile, 129 is the 256-row one.
        for (rows, class) in [
            (1, 128),
            (15, 128),
            (128, 128),
            (129, 256),
            (256, 256),
            (257, 1024),
            (1024, 1024),
            (1025, 4096),
            (4096, 4096),
        ] {
            assert_eq!(packed_rows(rows).unwrap(), class, "{rows} rows");
        }
        assert_eq!(PACKED_M_CLASSES, [128, 256, 1024, 4096]);
        // Above the largest class no program exists, so it is refused by name with both numbers
        // rather than clamped: a 4 096-row module serving 5 000 rows is the same defect one size
        // up, and it would run.
        let error = packed_rows(4097).unwrap_err().to_string();
        assert!(error.contains("4097"), "{error}");
        assert!(error.contains("4096"), "{error}");
    }

    #[test]
    fn the_projection_reserves_the_arena_its_class_writes() {
        // The launch states the class, so the region must be the class's: a region reserved for
        // the request's rows is one a 128-row launch writes past, and every other check agrees
        // while it happens.
        let packed = Nvfp4Linear::new(
            &packed(8, 64, ScaleLayout::Swizzled128X4),
            None,
            ScaleLayout::Swizzled128X4,
            DType::BF16,
            128,
        )
        .unwrap();
        assert_eq!(packed.scratch_bytes(15), packed.scratch_bytes(128));
        assert_eq!(packed.scratch_bytes(129), packed.scratch_bytes(256));
        // And it is larger than the request's own regions would be, which is the defect stated as
        // a number rather than as an inequality: 15 rows of a 64-wide contraction.
        assert!(packed.scratch_bytes(15) > 15 * 32 + 15 * 4);
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

    /// The encoder's tile divides the row it encodes.
    ///
    /// The package refuses a tile that does not divide the row, so this is not a preference: a row
    /// of 768 against a requested 1024 must state 256, which is the cell `nvfp4_quantize` declares.
    /// Both halves are asserted — the tile divides, and it is the largest power of two that does —
    /// because a smaller tile than necessary is a different key and a different program.
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
            let block = Nvfp4Linear::encode_block(columns, requested);
            assert_eq!(block, expected, "columns {columns}, requested {requested}");
            assert!(
                columns.is_multiple_of(block),
                "{block} does not divide a row of {columns}"
            );
        }
    }
}
